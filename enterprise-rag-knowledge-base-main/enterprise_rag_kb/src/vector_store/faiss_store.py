"""
Enterprise RAG - FAISS Vector Store
======================================
Stores document embeddings in a FAISS index for fast similarity search.
Supports: IndexFlatL2, IndexIVFFlat (approximate, faster for large datasets),
and IndexHNSWFlat (graph-based, best speed/accuracy tradeoff).
"""

import os
import pickle
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import config
from src.logger import get_logger
from src.ingestion.text_splitter import Chunk

logger = get_logger(__name__)


class FAISSVectorStore:
    """
    FAISS-backed vector store with metadata management.

    Architecture:
    - FAISS index: stores float32 embeddings → fast ANN search
    - Metadata dict: stores Chunk objects indexed by integer position
    - Persistence: both are saved to disk and reloaded on startup
    """

    def __init__(
        self,
        embedding_dim: int = None,
        index_type: str = None,
        index_path: str = None,
        metadata_path: str = None,
    ):
        self.embedding_dim = embedding_dim or config.embedding.embedding_dim
        self.index_type = index_type or config.vector_store.index_type
        self.index_path = index_path or config.vector_store.index_path
        self.metadata_path = metadata_path or config.vector_store.metadata_path

        self._index = None
        self._chunks: List[Chunk] = []
        self._id_map: Dict[int, str] = {}  # faiss_idx → chunk_id

    # ── Index Management ────────────────────────────────────────────────────

    def _build_index(self, n_vectors: int = 0):
        """Build a new FAISS index based on config."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

        dim = self.embedding_dim

        if self.index_type == "IndexFlatL2":
            # Exact search — best for small datasets (< 50k vectors)
            index = faiss.IndexFlatL2(dim)
            logger.info(f"Built IndexFlatL2 (exact search, dim={dim})")

        elif self.index_type == "IndexFlatIP":
            # Exact cosine similarity search (for normalized vectors)
            index = faiss.IndexFlatIP(dim)
            logger.info(f"Built IndexFlatIP (exact cosine, dim={dim})")

        elif self.index_type == "IndexIVFFlat":
            # Approximate — faster for 50k-1M vectors
            nlist = min(100, max(1, n_vectors // 39))
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.nprobe = config.vector_store.nprobe
            logger.info(f"Built IndexIVFFlat (nlist={nlist}, nprobe={index.nprobe})")

        elif self.index_type == "IndexHNSWFlat":
            # Graph-based — fastest for large datasets, good recall
            M = 32  # Number of neighbors during construction
            index = faiss.IndexHNSWFlat(dim, M)
            index.hnsw.efSearch = config.vector_store.ef_search
            logger.info(f"Built IndexHNSWFlat (M={M}, efSearch={index.hnsw.efSearch})")

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Wrap with IDMap so we can use custom IDs
        return index

    # ── Core Operations ─────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        """
        Add chunks and their embeddings to the vector store.
        Can be called multiple times to incrementally add documents.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match")

        embeddings = np.array(embeddings, dtype=np.float32)

        if self._index is None:
            self._index = self._build_index(n_vectors=len(chunks))

            # IVF indexes need training
            if hasattr(self._index, 'is_trained') and not self._index.is_trained:
                logger.info("Training IVF index...")
                self._index.train(embeddings)

        start_idx = len(self._chunks)
        self._index.add(embeddings)

        for i, chunk in enumerate(chunks):
            faiss_idx = start_idx + i
            self._id_map[faiss_idx] = chunk.chunk_id
            self._chunks.append(chunk)

        logger.info(f"Added {len(chunks)} chunks -> total: {len(self._chunks)}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = None,
        filter_metadata: Dict[str, Any] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for most similar chunks.

        Args:
            query_embedding: Query vector (1D array)
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0–1 for normalized)
            filter_metadata: Optional metadata filter dict

        Returns:
            List of (Chunk, score) tuples, sorted by relevance
        """
        if self._index is None or len(self._chunks) == 0:
            logger.warning("Vector store is empty. Index documents first.")
            return []

        query = np.array([query_embedding], dtype=np.float32)

        # Search with extra buffer if filtering
        search_k = top_k * 3 if filter_metadata else top_k
        search_k = min(search_k, len(self._chunks))

        distances, indices = self._index.search(query, search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue

            chunk = self._chunks[idx]

            # Convert L2 distance to similarity score [0, 1]
            if self.index_type in ("IndexFlatL2", "IndexIVFFlat"):
                score = 1.0 / (1.0 + float(dist))
            else:
                score = float(dist)  # IP index returns similarity directly

            # Apply score threshold
            if score_threshold and score < score_threshold:
                continue

            # Apply metadata filter
            if filter_metadata:
                match = all(
                    chunk.metadata.get(k) == v
                    for k, v in filter_metadata.items()
                )
                if not match:
                    continue

            results.append((chunk, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a chunk by its ID."""
        for chunk in self._chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self):
        """Save the index and metadata to disk."""
        if self._index is None:
            logger.warning("Nothing to save - index is empty")
            return

        import faiss
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, self.index_path)

        with open(self.metadata_path, "wb") as f:
            pickle.dump({
                "chunks": self._chunks,
                "id_map": self._id_map,
                "embedding_dim": self.embedding_dim,
                "index_type": self.index_type,
            }, f)

        logger.info(f"Saved vector store: {len(self._chunks)} chunks -> {self.index_path}")

    def load(self):
        """Load the index and metadata from disk."""
        if not Path(self.index_path).exists():
            raise FileNotFoundError(f"No saved index at: {self.index_path}")

        import faiss
        self._index = faiss.read_index(self.index_path)

        with open(self.metadata_path, "rb") as f:
            data = pickle.load(f)

        self._chunks = data["chunks"]
        self._id_map = data["id_map"]
        self.embedding_dim = data["embedding_dim"]

        logger.info(f"Loaded vector store: {len(self._chunks)} chunks from {self.index_path}")

    def is_loaded(self) -> bool:
        return self._index is not None and len(self._chunks) > 0

    # ── Statistics ───────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": len(self._chunks),
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "unique_documents": len(set(c.doc_id for c in self._chunks)),
            "unique_sources": len(set(c.metadata.get("source", "") for c in self._chunks)),
            "index_path": self.index_path,
        }

    def __len__(self):
        return len(self._chunks)

    def __repr__(self):
        return f"FAISSVectorStore(chunks={len(self._chunks)}, dim={self.embedding_dim}, type={self.index_type})"
