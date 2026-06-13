"""
Enterprise RAG - REST API
===========================
FastAPI-based REST API for the RAG pipeline.
Run with: uvicorn api.main:app --reload --port 8000

Endpoints:
  POST /ingest/text      - Ingest raw text
  POST /ingest/file      - Ingest uploaded file
  POST /query            - Query the knowledge base
  GET  /stats            - Pipeline statistics
  GET  /documents        - List all documents
  DELETE /reset          - Reset the index
  GET  /health           - Health check
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import tempfile
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=2)

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Enterprise RAG Knowledge Base API",
    description="""
    A production-ready Retrieval-Augmented Generation API.

    ## Features
    *  Multi-format document ingestion (PDF, TXT, MD, CSV, JSON)
    *  Hybrid retrieval (dense + BM25)
    *  Cross-encoder reranking
    *  LLM-powered answer generation
    *  Built-in evaluation metrics
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy Pipeline ─────────────────────────────────────────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.pipeline.rag_pipeline import RAGPipeline
        _pipeline = RAGPipeline(auto_load=True)
    return _pipeline


@app.on_event("startup")
async def startup_preload():
    """Pre-load pipeline and all lazy models at startup to avoid first-query crashes."""
    import threading
    def _preload():
        p = get_pipeline()
        # Force-load embedding model
        p.embedder.embed_texts(["warmup"])
        # Force-load reranker model
        if p.reranker and hasattr(p.reranker, 'model'):
            pass  # CrossEncoderReranker loads on first rerank call
        print("[API] Pipeline and models pre-loaded successfully")
    threading.Thread(target=_preload, daemon=True).start()


# ── Pydantic Models ───────────────────────────────────────────────────────────

class TextIngestRequest(BaseModel):
    text: str = Field(..., description="Raw text to ingest into knowledge base", min_length=10)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")
    source_name: Optional[str] = Field(default="api_input", description="Source identifier")

class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask the knowledge base", min_length=3)
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    retrieval_mode: Optional[str] = Field(
        default="hybrid",
        description="Retrieval mode: hybrid | dense | sparse"
    )
    return_sources: Optional[bool] = Field(default=True, description="Include source documents")
    return_context: Optional[bool] = Field(default=False, description="Include full context")

class BatchQueryRequest(BaseModel):
    questions: List[str] = Field(..., description="List of questions")
    top_k: Optional[int] = Field(default=5, ge=1, le=20)

class QueryResponse(BaseModel):
    model_config = {"extra": "ignore"}
    answer: str
    confidence: float
    model: str
    retrieval_time_ms: int
    sources: Optional[List[Dict]] = None
    context: Optional[str] = None
    query: str

class IngestResponse(BaseModel):
    model_config = {"extra": "ignore"}
    status: str
    documents_loaded: int
    chunks_added: int
    total_chunks: int
    time_seconds: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Check if the API is running and the pipeline is ready."""
    pipeline = get_pipeline()
    stats = pipeline.get_stats()
    return {
        "status": "healthy",
        "pipeline_status": stats["pipeline_status"],
        "total_chunks": stats["vector_store"].get("total_chunks", 0),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


@app.get("/stats", tags=["System"])
async def get_stats():
    """Get comprehensive pipeline statistics."""
    pipeline = get_pipeline()
    return pipeline.get_stats()


@app.post("/ingest/text", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_text(request: TextIngestRequest):
    """Ingest raw text into the knowledge base."""
    pipeline = get_pipeline()
    metadata = request.metadata or {}
    metadata["source"] = request.source_name

    result = pipeline.ingest_text(request.text, metadata=metadata)

    if result.get("status") != "success":
        raise HTTPException(status_code=500, detail="Ingestion failed")

    return IngestResponse(**result)


@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a document file (PDF, TXT, MD, CSV, JSON)."""
    pipeline = get_pipeline()

    supported = {".pdf", ".txt", ".md", ".csv", ".json", ".docx"}
    ext = Path(file.filename).suffix.lower()
    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {supported}"
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipeline.ingest_file(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

        if result.get("status") != "success":
            raise HTTPException(status_code=500, detail="Ingestion failed")

        return IngestResponse(**result)

    except Exception as e:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """Query the knowledge base and get an AI-generated answer."""
    pipeline = get_pipeline()

    if not pipeline.vector_store.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is empty. Please ingest documents first."
        )

    if request.retrieval_mode not in ("hybrid", "dense", "sparse"):
        raise HTTPException(
            status_code=400,
            detail="retrieval_mode must be: hybrid | dense | sparse"
        )

    # Run heavy pipeline in thread executor to avoid blocking async loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: pipeline.query(
            question=request.question,
            top_k=request.top_k,
            mode=request.retrieval_mode,
            return_sources=request.return_sources,
        )
    )

    if not request.return_context and "context" in result:
        del result["context"]

    return QueryResponse(**result)


@app.post("/query/batch", tags=["Query"])
async def batch_query(request: BatchQueryRequest):
    """Run multiple queries in batch."""
    pipeline = get_pipeline()

    if not pipeline.vector_store.is_loaded():
        raise HTTPException(status_code=503, detail="Knowledge base is empty.")

    loop = asyncio.get_event_loop()

    def _run_batch():
        results = []
        for question in request.questions:
            r = pipeline.query(question, top_k=request.top_k, return_sources=False)
            results.append({
                "question": question,
                "answer": r.get("answer"),
                "confidence": r.get("confidence"),
            })
        return results

    results = await loop.run_in_executor(_executor, _run_batch)
    return {"results": results, "count": len(results)}


@app.get("/documents", tags=["Knowledge Base"])
async def list_documents():
    """List all ingested documents in the knowledge base."""
    pipeline = get_pipeline()
    return {"documents": pipeline.list_documents()}


@app.delete("/reset", tags=["System"])
async def reset_index():
    """ Reset (clear) the entire knowledge base index."""
    pipeline = get_pipeline()
    pipeline.reset_index()
    return {"status": "success", "message": "Knowledge base reset successfully"}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
