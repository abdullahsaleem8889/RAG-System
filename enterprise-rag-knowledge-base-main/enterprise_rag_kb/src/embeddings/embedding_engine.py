"""
Enterprise RAG - Embedding Engine
====================================
Generates dense vector embeddings using sentence-transformers.
Supports batching, caching, and multiple model choices.
"""

import os
import pickle
import hashlib
import numpy as np
from typing import List, Union, Optional, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import config
from src.logger import get_logger
from src.ingestion.text_splitter import Chunk

logger = get_logger(__name__)


class EmbeddingEngine:
    """
    Generates text embeddings using sentence-transformers.

    Features:
    - Batch processing for efficiency
    - In-memory LRU-style cache to avoid re-embedding
    - Normalized embeddings for cosine similarity
    - Progress tracking for large batches
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = None,
        use_cache: bool = True,
        cache_path: str = None,
    ):
        self.model_name = model_name or config.embedding.model_name
        self.device = device or config.embedding.device
        self.batch_size = batch_size or config.embedding.batch_size
        self.normalize = config.embedding.normalize_embeddings
        self.use_cache = use_cache
        self.cache_path = cache_path or str(
            Path(config.vector_store.index_path).parent / "embed_cache.pkl"
        )

        self._model = None
        self._cache: Dict[str, np.ndarray] = {}

        if use_cache and Path(self.cache_path).exists():
            self._load_cache()

    @property
    def model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                
                # Suppress future warning for getting embedding dimension
                dim = getattr(self._model, "get_embedding_dimension", getattr(self._model, "get_sentence_embedding_dimension", None))
                if dim:
                    dim_val = dim()
                else:
                    dim_val = 384
                
                logger.info(f"[OK] Model loaded | Embedding dim: {dim_val}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension without forcing model load if possible."""
        # Use config value first to avoid loading model just for dim
        if self._model is not None:
            dim_func = getattr(self._model, "get_embedding_dimension", getattr(self._model, "get_sentence_embedding_dimension", None))
            return dim_func() if dim_func else config.embedding.embedding_dim
        return config.embedding.embedding_dim  # fast path — no model load needed

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.
        Returns np.ndarray of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])

        # Check cache for each text
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if self.use_cache and key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Embed uncached texts in batches
        if uncached_texts:
            logger.info(f"Embedding {len(uncached_texts)} texts (batch_size={self.batch_size})...")
            all_embeddings = []

            # Single encode call — sentence-transformers handles batching internally
            batch_embeddings = self.model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,  # suppress tqdm to reduce overhead
                convert_to_numpy=True,
            )
            all_embeddings = batch_embeddings if len(uncached_texts) > 1 else batch_embeddings.reshape(1, -1)

            # Store in cache and results
            for idx, emb in zip(uncached_indices, all_embeddings):
                key = self._cache_key(texts[idx])
                self._cache[key] = emb
                results[idx] = emb

        embeddings = np.array(results, dtype=np.float32)
        return embeddings

    def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """Embed a list of Chunk objects."""
        texts = [c.content for c in chunks]
        return self.embed_texts(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        # Queries can be prefixed for asymmetric models
        query_prefix = "query: " if "e5" in self.model_name.lower() else ""
        return self.embed_texts([query_prefix + query])[0]

    def compute_similarity(self, query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between a query and multiple documents.
        Returns similarity scores in range [-1, 1].
        """
        if self.normalize:
            # Dot product of normalized vectors = cosine similarity
            scores = doc_embs @ query_emb
        else:
            # Manual cosine similarity
            query_norm = np.linalg.norm(query_emb)
            doc_norms = np.linalg.norm(doc_embs, axis=1)
            scores = (doc_embs @ query_emb) / (doc_norms * query_norm + 1e-8)
        return scores

    def save_cache(self):
        """Persist embedding cache to disk."""
        if self.use_cache and self._cache:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._cache, f)
            logger.info(f"Saved {len(self._cache)} embeddings to cache")

    def _load_cache(self):
        try:
            with open(self.cache_path, "rb") as f:
                self._cache = pickle.load(f)
            logger.info(f"Loaded {len(self._cache)} embeddings from cache")
        except Exception as e:
            logger.warning(f"Could not load embedding cache: {e}")
            self._cache = {}

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(
            f"{self.model_name}:{text[:500]}".encode()
        ).hexdigest()
