"""
Enterprise RAG - Main Pipeline
=================================
Orchestrates: Ingestion → Embedding → Storage → Retrieval → Reranking → Generation

This is the central class that coordinates all components.
Usage:
    pipeline = RAGPipeline()
    pipeline.ingest_directory("./data/raw")
    result = pipeline.query("What is the company's return policy?")
"""

import time
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import config, RAGConfig
from src.logger import get_logger
from src.ingestion.document_loader import DocumentLoader, Document
from src.ingestion.preprocessor import TextPreprocessor
from src.ingestion.text_splitter import RecursiveTextSplitter, Chunk
from src.embeddings.embedding_engine import EmbeddingEngine
from src.vector_store.faiss_store import FAISSVectorStore
from src.retrieval.retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker, SimpleReranker
from src.generation.llm_engine import LLMEngine, GenerationResult

logger = get_logger(__name__)


class RAGPipeline:
    """
    Enterprise-Grade RAG Pipeline

    Capabilities:
    ✓ Multi-format document ingestion (PDF, TXT, MD, CSV, JSON, DOCX)
    ✓ Advanced text chunking with overlap
    ✓ Sentence-transformer embeddings
    ✓ FAISS vector store with persistence
    ✓ Hybrid retrieval (dense + BM25)
    ✓ Cross-encoder reranking
    ✓ LLM answer generation with source attribution
    ✓ Evaluation metrics
    ✓ Query history & analytics
    """

    def __init__(self, cfg: RAGConfig = None, auto_load: bool = True):
        self.cfg = cfg or config
        self._query_history = []

        logger.info("=" * 60)
        logger.info("Initializing Enterprise RAG Pipeline")
        logger.info("=" * 60)

        # Initialize components
        self.loader = DocumentLoader()
        self.preprocessor = TextPreprocessor()
        self.splitter = RecursiveTextSplitter()
        self.embedder = EmbeddingEngine()
        self.vector_store = FAISSVectorStore(
            embedding_dim=self.embedder.embedding_dim
        )
        self.llm = LLMEngine()

        # Try to load existing index
        if auto_load:
            self._try_load_index()

        # Retriever and reranker set up after index is available
        self.retriever = None
        self.reranker = None
        self._setup_retriever()

    def _try_load_index(self):
        """Load existing vector store index if available."""
        try:
            self.vector_store.load()
            logger.info(f"[OK] Loaded existing index: {len(self.vector_store)} chunks")
        except FileNotFoundError:
            logger.info("No existing index found. Ingest documents to create one.")

    def _setup_retriever(self):
        """Initialize retriever and reranker."""
        if self.vector_store.is_loaded():
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedding_engine=self.embedder,
            )

        # Try cross-encoder reranker, fall back to simple
        if self.cfg.retrieval.use_reranker:
            try:
                self.reranker = CrossEncoderReranker()
                # NOTE: Model loads lazily on first query — do NOT force-load here
                logger.info("[OK] Cross-encoder reranker configured (lazy load)")
            except Exception as e:
                logger.warning(f"Cross-encoder unavailable ({e}). Using simple reranker.")
                self.reranker = SimpleReranker()
        else:
            self.reranker = SimpleReranker()

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """Ingest a single file into the knowledge base."""
        return self._ingest_documents(self.loader.load_file(file_path))

    def ingest_directory(self, dir_path: str = None, recursive: bool = True) -> Dict[str, Any]:
        """Ingest all documents from a directory."""
        dir_path = dir_path or str(config.DATA_DIR / "raw")
        documents = self.loader.load_directory(dir_path, recursive)
        return self._ingest_documents(documents)

    def ingest_text(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Directly ingest a text string into the knowledge base."""
        doc = Document(
            content=text,
            metadata=metadata or {"source": "direct_input", "file_type": "text"}
        )
        return self._ingest_documents([doc])

    def _ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Core ingestion pipeline: preprocess → chunk → embed → store."""
        if not documents:
            return {"status": "no_documents", "chunks_added": 0}

        start = time.time()
        logger.info(f"Ingesting {len(documents)} documents...")

        # Step 1: Preprocess
        documents = self.preprocessor.preprocess_batch(documents)

        # Step 2: Chunk
        chunks = self.splitter.split_documents(documents)
        chunk_stats = self.splitter.get_stats(chunks)
        logger.info(f"Chunking stats: {chunk_stats}")

        # Step 3: Embed
        embeddings = self.embedder.embed_chunks(chunks)
        self.embedder.save_cache()

        # Step 4: Store in FAISS
        self.vector_store.add_chunks(chunks, embeddings)
        self.vector_store.save()

        # Step 5: Update retriever
        self._setup_retriever()

        elapsed = time.time() - start
        result = {
            "status": "success",
            "documents_loaded": len(documents),
            "chunks_added": len(chunks),
            "total_chunks": len(self.vector_store),
            "time_seconds": round(elapsed, 2),
            **chunk_stats,
        }
        logger.info(f"Ingestion complete in {elapsed:.2f}s: {result}")
        return result

    # ── Querying ─────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = None,
        mode: str = None,
        return_sources: bool = True,
        fast_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Main query interface.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            mode: Override retrieval mode ("dense", "sparse", "hybrid")
            return_sources: Include source documents in response
            fast_mode: Skip reranking + LLM for fast evaluation (extractive only)

        Returns:
            Dict with answer, sources, confidence, and metadata
        """
        if not self.vector_store.is_loaded():
            return {
                "answer": "Knowledge base is empty. Please ingest documents first.",
                "sources": [],
                "confidence": 0.0,
                "error": "empty_index",
            }

        # Ensure retriever is initialized
        if self.retriever is None:
            self._setup_retriever()
            if self.retriever is None:
                return {
                    "answer": "Retriever could not be initialized. Please re-ingest documents.",
                    "sources": [],
                    "confidence": 0.0,
                    "error": "retriever_init_failed",
                }

        start = time.time()
        top_k = top_k or self.cfg.retrieval.top_k

        logger.info(f"Query: '{question[:80]}...' " if len(question) > 80 else f"Query: '{question}'")

        # Step 1: Retrieve
        if mode:
            self.retriever.mode = mode

        candidates = self.retriever.retrieve(question, top_k=top_k * 2)

        # Step 2: Rerank (skip in fast_mode)
        if fast_mode:
            final_chunks = candidates[:top_k]
        elif self.cfg.retrieval.use_reranker and len(candidates) > 1:
            final_chunks = self.reranker.rerank(question, candidates, top_k=top_k)
        else:
            final_chunks = candidates[:top_k]

        # Step 3: Generate (skip LLM in fast_mode, use extractive fallback only)
        if fast_mode:
            # Use extractive fallback directly — no network call
            from src.generation.llm_engine import GenerationResult
            context, sources = self.llm._format_context(final_chunks, max_chars=3000)
            prompt = self.llm._build_prompt(question, context)
            answer = self.llm._extractive_answer(prompt)
            result = GenerationResult(
                answer=answer,
                sources=sources,
                confidence=self.llm._estimate_confidence(final_chunks, answer),
                context_used=context,
                model_used="extractive-fast",
            )
        else:
            result: GenerationResult = self.llm.generate(question, final_chunks)

        elapsed = time.time() - start

        response = {
            "answer": result.answer,
            "confidence": result.confidence,
            "model": result.model_used,
            "retrieval_time_ms": round(elapsed * 1000),
            "query": question,
        }

        if return_sources:
            response["sources"] = [
                {
                    "filename": chunk.metadata.get("filename", "Unknown"),
                    "source": chunk.metadata.get("source", ""),
                    "page": chunk.metadata.get("page", ""),
                    "relevance_score": round(score, 4),
                    "preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    "chunk_id": chunk.chunk_id,
                }
                for chunk, score in final_chunks
            ]
            response["context"] = result.context_used

        # Record in history
        self._query_history.append({
            "question": question,
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": result.sources,
            "timestamp": time.time(),
        })

        logger.info(f"Query answered in {elapsed:.2f}s | confidence={result.confidence:.2f}")
        return response

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Run multiple queries in sequence."""
        logger.info(f"Running batch of {len(questions)} queries...")
        return [self.query(q) for q in questions]

    # ── Management ───────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        vs_stats = self.vector_store.get_stats()
        return {
            "pipeline_status": "ready" if self.vector_store.is_loaded() else "empty",
            "vector_store": vs_stats,
            "embedding_model": self.embedder.model_name,
            "llm_model": self.llm.groq_model,
            "retrieval_mode": self.cfg.retrieval.retrieval_mode,
            "reranker": type(self.reranker).__name__ if self.reranker else None,
            "queries_served": len(self._query_history),
            "config": {
                "chunk_size": self.cfg.chunking.chunk_size,
                "chunk_overlap": self.cfg.chunking.chunk_overlap,
                "top_k": self.cfg.retrieval.top_k,
            }
        }

    def get_query_history(self) -> List[Dict]:
        """Return query history."""
        return self._query_history

    def reset_index(self):
        """Clear the vector store and reset the pipeline."""
        import shutil
        vs_dir = Path(self.cfg.vector_store.index_path).parent
        if vs_dir.exists():
            shutil.rmtree(vs_dir)
            vs_dir.mkdir(parents=True, exist_ok=True)

        self.vector_store = FAISSVectorStore(embedding_dim=self.embedder.embedding_dim)
        self.retriever = None
        self._query_history = []
        logger.info("Index reset complete")

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all ingested documents."""
        seen = {}
        for chunk in self.vector_store._chunks:
            source = chunk.metadata.get("source", "unknown")
            if source not in seen:
                seen[source] = {
                    "source": source,
                    "filename": chunk.metadata.get("filename", ""),
                    "file_type": chunk.metadata.get("file_type", ""),
                    "chunk_count": 0,
                }
            seen[source]["chunk_count"] += 1
        return list(seen.values())
