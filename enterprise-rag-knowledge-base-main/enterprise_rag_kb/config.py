"""
Enterprise RAG Knowledge Base - Central Configuration
=====================================================
All settings, paths, and model configurations in one place.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# Load .env file automatically
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv not installed, rely on system env vars

# ─── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
LOGS_DIR = BASE_DIR / "logs"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_documents"

# Create directories if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_STORE_DIR, LOGS_DIR, SAMPLE_DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Embedding Configuration ───────────────────────────────────────────────────
@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Alternatives (better quality but slower):
    # "sentence-transformers/all-mpnet-base-v2"
    # "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    batch_size: int = 32
    device: str = "cpu"  # "cuda" if GPU available
    normalize_embeddings: bool = True


# ─── Chunking Configuration ────────────────────────────────────────────────────
@dataclass
class ChunkingConfig:
    chunk_size: int = 512           # tokens per chunk
    chunk_overlap: int = 64         # overlap between chunks
    min_chunk_size: int = 100       # discard chunks smaller than this
    separators: List[str] = field(default_factory=lambda: [
        "\n\n", "\n", ". ", "! ", "? ", " ", ""
    ])
    split_by_sentence: bool = True


# ─── Vector Store Configuration ────────────────────────────────────────────────
@dataclass
class VectorStoreConfig:
    index_type: str = "IndexFlatL2"     # Options: IndexFlatL2, IndexIVFFlat, IndexHNSWFlat
    index_path: str = str(VECTOR_STORE_DIR / "faiss_index")
    metadata_path: str = str(VECTOR_STORE_DIR / "metadata.pkl")
    nprobe: int = 10                    # For IVF index (search accuracy vs speed)
    ef_search: int = 50                 # For HNSW index


# ─── Retrieval Configuration ───────────────────────────────────────────────────
@dataclass
class RetrievalConfig:
    top_k: int = 5                      # Number of docs to retrieve
    retrieval_mode: str = "hybrid"      # Options: "dense", "sparse", "hybrid"
    dense_weight: float = 0.7           # Weight for dense retrieval in hybrid
    sparse_weight: float = 0.3          # Weight for BM25 in hybrid
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 3             # Final docs after reranking
    score_threshold: float = 0.0        # Minimum similarity score


# ─── LLM / Generation Configuration ───────────────────────────────────────────
@dataclass
class GenerationConfig:
    # Groq API (primary, free tier - very fast)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = "llama-3.3-70b-versatile"

    # Google Gemini (secondary fallback)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = "gemini-2.0-flash"

    # HuggingFace Inference API (tertiary fallback)
    hf_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    hf_api_token: str = os.getenv("HF_API_TOKEN", "")

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # System prompt template
    system_prompt: str = (
        "You are an expert enterprise knowledge assistant. "
        "Answer questions accurately and concisely using ONLY the provided context. "
        "If the answer is not in the context, clearly state that you don't have "
        "enough information. Always cite which document your answer comes from."
    )

    # RAG prompt template
    rag_prompt_template: str = """### Context Information:
{context}

### Question:
{question}

### Instructions:
Based ONLY on the context above, provide a comprehensive and accurate answer.
Mention which source documents you used. If the context doesn't contain enough
information, say "Based on available documents, I cannot fully answer this question."

### Answer:"""


# ─── Evaluation Configuration ──────────────────────────────────────────────────
@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ])
    eval_dataset_path: str = str(DATA_DIR / "eval_dataset.json")
    results_path: str = str(DATA_DIR / "eval_results.json")


# ─── Logging Configuration ─────────────────────────────────────────────────────
@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str = str(LOGS_DIR / "rag_pipeline.log")
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    max_bytes: int = 10 * 1024 * 1024   # 10 MB
    backup_count: int = 5


# ─── Master Config ─────────────────────────────────────────────────────────────
@dataclass
class RAGConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Supported file formats
    supported_formats: List[str] = field(default_factory=lambda: [
        ".pdf", ".txt", ".md", ".csv", ".json", ".docx"
    ])


# Global config instance
config = RAGConfig()
