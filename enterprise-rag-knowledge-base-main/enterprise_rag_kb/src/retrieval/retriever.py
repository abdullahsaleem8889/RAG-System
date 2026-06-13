"""
Enterprise RAG - Hybrid Retriever
====================================
Combines dense (semantic) and sparse (keyword) retrieval
using Reciprocal Rank Fusion (RRF) for optimal results.

Why Hybrid?
- Dense retrieval: great for semantic/conceptual queries
- Sparse (BM25): great for exact keyword/entity matches
- Hybrid: best of both worlds, especially for enterprise docs
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import config
from src.logger import get_logger
from src.ingestion.text_splitter import Chunk
from src.embeddings.embedding_engine import EmbeddingEngine
from src.vector_store.faiss_store import FAISSVectorStore

logger = get_logger(__name__)


class BM25Retriever:
    """
    BM25 sparse retriever for keyword-based matching.
    Uses rank_bm25 library (pip install rank-bm25).
    """

    def __init__(self, chunks: List[Chunk] = None):
        self._bm25 = None
        self._chunks = chunks or []
        self._tokenized_corpus = []

        if chunks:
            self._build_index(chunks)

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization."""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if len(t) > 1]

    def _build_index(self, chunks: List[Chunk]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank-bm25 not installed. BM25 retrieval disabled. Run: pip install rank-bm25")
            return

        self._chunks = chunks
        self._tokenized_corpus = [self._tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 index built with {len(chunks)} documents")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        if self._bm25 is None:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._chunks[idx], float(scores[idx])))

        return results


class HybridRetriever:
    """
    Hybrid retriever that fuses dense + sparse results using
    Reciprocal Rank Fusion (RRF).

    RRF Formula: score(d) = Σ 1/(k + rank(d))
    where k=60 is a smoothing constant.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_engine: EmbeddingEngine,
        chunks: List[Chunk] = None,
        mode: str = None,
        dense_weight: float = None,
        sparse_weight: float = None,
    ):
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.mode = mode or config.retrieval.retrieval_mode
        self.dense_weight = dense_weight or config.retrieval.dense_weight
        self.sparse_weight = sparse_weight or config.retrieval.sparse_weight

        self.bm25 = BM25Retriever(chunks or vector_store._chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Dict[str, Any] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top-k most relevant chunks for a query.

        Args:
            query: User's query string
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of (Chunk, relevance_score) sorted by relevance
        """
        top_k = top_k or config.retrieval.top_k
        retrieve_k = top_k * 3  # Get more, then re-rank

        if self.mode == "dense":
            return self._dense_retrieve(query, top_k, filter_metadata)

        elif self.mode == "sparse":
            results = self.bm25.search(query, top_k)
            return results

        elif self.mode == "hybrid":
            return self._hybrid_retrieve(query, top_k, retrieve_k, filter_metadata)

        else:
            raise ValueError(f"Unknown retrieval mode: {self.mode}")

    def _dense_retrieve(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Pure dense retrieval using FAISS."""
        query_embedding = self.embedding_engine.embed_query(query)
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )
        return results

    def _hybrid_retrieve(
        self,
        query: str,
        top_k: int,
        retrieve_k: int,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Hybrid retrieval using Reciprocal Rank Fusion.
        Combines dense and sparse rankings.
        """
        # Dense results
        query_embedding = self.embedding_engine.embed_query(query)
        dense_results = self.vector_store.search(
            query_embedding,
            top_k=retrieve_k,
            filter_metadata=filter_metadata,
        )

        # Sparse results
        sparse_results = self.bm25.search(query, retrieve_k)

        # Reciprocal Rank Fusion
        rrf_scores = defaultdict(float)
        chunk_map = {}

        # Dense ranking contribution
        k_rrf = 60  # standard constant
        for rank, (chunk, score) in enumerate(dense_results):
            rrf_scores[chunk.chunk_id] += self.dense_weight * (1.0 / (k_rrf + rank + 1))
            chunk_map[chunk.chunk_id] = chunk

        # Sparse ranking contribution
        for rank, (chunk, score) in enumerate(sparse_results):
            rrf_scores[chunk.chunk_id] += self.sparse_weight * (1.0 / (k_rrf + rank + 1))
            chunk_map[chunk.chunk_id] = chunk

        # Sort by fused score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = [
            (chunk_map[cid], rrf_scores[cid])
            for cid in sorted_ids[:top_k]
            if cid in chunk_map
        ]

        logger.debug(
            f"Hybrid retrieval: dense={len(dense_results)}, "
            f"sparse={len(sparse_results)}, fused={len(results)}"
        )
        return results

    def update_index(self, new_chunks: List[Chunk]):
        """Update BM25 index with new chunks (after adding to vector store)."""
        all_chunks = self.vector_store._chunks
        self.bm25 = BM25Retriever(all_chunks)
        logger.info(f"BM25 index updated: {len(all_chunks)} total chunks")
