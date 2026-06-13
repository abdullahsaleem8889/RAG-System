"""
Enterprise RAG - Cross-Encoder Reranker
=========================================
Reranks retrieved candidates using a cross-encoder model.

Why reranking?
- Bi-encoders (FAISS) are fast but approximate
- Cross-encoders look at query+document together → much more accurate
- We retrieve 5x more candidates, then rerank to top-k
"""

import numpy as np
from typing import List, Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import config
from src.logger import get_logger
from src.ingestion.text_splitter import Chunk

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.
    The model jointly encodes (query, document) for precise relevance scoring.
    """

    def __init__(
        self,
        model_name: str = None,
        top_k: int = None,
        device: str = "cpu",
    ):
        self.model_name = model_name or config.retrieval.reranker_model
        self.top_k = top_k or config.retrieval.reranker_top_k
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading reranker: {self.model_name}")
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device=self.device)
                logger.info("[OK] Reranker loaded")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Chunk, float]],
        top_k: int = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Rerank candidate chunks using cross-encoder scoring.

        Args:
            query: Original user query
            candidates: List of (Chunk, initial_score) from retriever
            top_k: Number to return after reranking

        Returns:
            Reranked list of (Chunk, rerank_score)
        """
        top_k = top_k or self.top_k

        if not candidates:
            return []

        if len(candidates) == 1:
            return candidates

        try:
            # Prepare (query, passage) pairs for cross-encoder
            pairs = [(query, chunk.content) for chunk, _ in candidates]

            # Score all pairs
            scores = self.model.predict(pairs, show_progress_bar=False)

            # Combine chunks with new scores
            reranked = [(chunk, float(score))
                        for (chunk, _), score in zip(candidates, scores)]

            # Sort by reranker score (higher = more relevant)
            reranked.sort(key=lambda x: x[1], reverse=True)

            logger.debug(f"Reranked {len(candidates)} -> top {top_k}")
            return reranked[:top_k]

        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning original order.")
            return candidates[:top_k]


class SimpleReranker:
    """
    Lightweight reranker based on keyword overlap.
    Fallback when cross-encoder model is unavailable.
    """

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Chunk, float]],
        top_k: int = 3,
    ) -> List[Tuple[Chunk, float]]:
        """Rerank using query term presence score."""
        query_terms = set(query.lower().split())

        scored = []
        for chunk, base_score in candidates:
            content_lower = chunk.content.lower()
            term_hits = sum(1 for t in query_terms if t in content_lower)
            term_ratio = term_hits / max(len(query_terms), 1)
            combined = 0.7 * base_score + 0.3 * term_ratio
            scored.append((chunk, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
