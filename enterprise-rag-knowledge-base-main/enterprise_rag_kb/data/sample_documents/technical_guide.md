
# Enterprise RAG System - Technical Architecture Guide

## Overview
This document describes the technical architecture of our Enterprise
Retrieval-Augmented Generation (RAG) Knowledge Base system.

## System Architecture

### Component 1: Document Ingestion Pipeline
The ingestion pipeline processes multiple document formats and converts them
into a unified format suitable for embedding and storage.

**Supported Formats:**
- PDF documents (using PyPDF)
- Plain text files (.txt)
- Markdown documents (.md)
- CSV data files
- JSON structured data
- DOCX Word documents

**Processing Steps:**
1. File format detection
2. Text extraction
3. Unicode normalization (NFKC)
4. Noise removal (headers, footers, page numbers)
5. Sentence-aware chunking with 64-token overlap
6. Metadata tagging

### Component 2: Embedding Engine
We use `sentence-transformers/all-MiniLM-L6-v2` as the default embedding model.
This model produces 384-dimensional vectors and offers an excellent balance
between speed and quality.

**Model Comparisons:**
| Model | Dimensions | Speed | Quality |
|-------|-----------|-------|---------|
| all-MiniLM-L6-v2 | 384 | Fast | Good |
| all-mpnet-base-v2 | 768 | Medium | Better |
| BAAI/bge-large-en | 1024 | Slow | Best |

### Component 3: FAISS Vector Store
FAISS (Facebook AI Similarity Search) is used for fast approximate nearest
neighbor search.

**Index Types:**
- **IndexFlatL2**: Exact search. Best for < 50,000 vectors. O(n) search time.
- **IndexIVFFlat**: Approximate search with inverted file index. Good for 50k-1M.
- **IndexHNSWFlat**: Hierarchical NSW graph. Best speed/recall tradeoff.

### Component 4: Hybrid Retrieval
The hybrid retriever combines dense and sparse retrieval:

**Dense Retrieval (FAISS)**
- Uses semantic embeddings to find conceptually similar documents
- Best for: paraphrased queries, concept-level questions

**Sparse Retrieval (BM25)**
- Classic TF-IDF based keyword matching
- Best for: exact entity names, codes, specific terms

**Fusion Method: Reciprocal Rank Fusion (RRF)**
```
rrf_score(d) = Σ 1/(k + rank(d))   where k = 60
```

### Component 5: Cross-Encoder Reranker
Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Jointly encodes query + document for precise relevance
- Applied to top-15 candidates, returns top-5
- Reduces false positives significantly

### Component 6: LLM Generation
Default: `mistralai/Mistral-7B-Instruct-v0.2` via HuggingFace Inference API
- Generates answers grounded in retrieved context
- System prompt enforces source attribution and factual grounding
- Fallback chain: HF API → Local pipeline → Extractive

## Performance Benchmarks

### Retrieval Latency (median)
- Embedding query: 8ms
- FAISS search: 2ms
- BM25 search: 5ms
- Reranking: 45ms
- LLM generation: 800ms
- **Total P50**: ~860ms

### Quality Metrics (on enterprise QA benchmark)
- Context Precision: 0.84
- Context Recall: 0.79
- Answer Faithfulness: 0.88
- Answer Relevancy: 0.82

## Deployment Guide

### Prerequisites
```bash
Python >= 3.9
RAM >= 4GB (8GB recommended)
Storage >= 2GB for models and index
```

### Installation
```bash
git clone <repo>
cd enterprise_rag_kb
pip install -r requirements.txt
```

### Quick Start
```python
from src.pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.ingest_directory("./data/raw")
result = pipeline.query("What is the data privacy policy?")
print(result["answer"])
```
