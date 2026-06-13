
# Frequently Asked Questions (FAQ)

## Product Questions

### Q: What is the maximum file size for document upload?
A: The system supports files up to 100MB per upload. For batch ingestion via
the command line, there is no practical size limit. Very large files (>500MB)
should be split before upload for optimal processing performance.

### Q: Which languages are supported?
A: The system primarily supports English with high accuracy. The embedding
model (all-MiniLM-L6-v2) has limited multilingual support. For multilingual
deployments, we recommend using 'paraphrase-multilingual-MiniLM-L12-v2' which
supports 50+ languages.

### Q: How often should the knowledge base be updated?
A: We recommend refreshing the knowledge base whenever source documents are
updated. For dynamic content, consider implementing automated ingestion
pipelines triggered by document change events. Most enterprise customers
refresh their knowledge base weekly or upon major document updates.

### Q: What is the difference between dense and hybrid retrieval?
A: Dense retrieval uses only semantic embeddings (FAISS) to find similar
documents. Hybrid retrieval combines semantic search with BM25 keyword matching
using Reciprocal Rank Fusion. Hybrid mode is recommended for most enterprise
use cases as it handles both conceptual queries and exact keyword matches.

## Technical Questions

### Q: Can I run this fully offline without API keys?
A: Yes! The system has a full offline mode:
- Embeddings: sentence-transformers runs locally (auto-downloaded from HuggingFace)
- Vector Store: FAISS runs locally
- Generation: Use the local transformers pipeline with flan-t5-base
Note: Local generation quality is lower than cloud LLM APIs.

### Q: How many documents can the system handle?
A: The system scales well:
- IndexFlatL2: Optimal for up to 50,000 chunks (~2,000-5,000 pages)
- IndexIVFFlat: Handles 50,000-1,000,000 chunks efficiently
- IndexHNSWFlat: Handles millions of chunks with sub-10ms latency

### Q: What hardware is required?
A: Minimum: 4GB RAM, any modern CPU. Recommended: 8GB RAM, 4 cores.
GPU (CUDA) significantly speeds up embedding generation but is not required.

## Billing & Licensing

### Q: Is HuggingFace API token required?
A: A HuggingFace token is optional. Without it, the system falls back to
local generation using smaller models. For production deployments with high
quality requirements, a HuggingFace Pro subscription ($9/month) is recommended.

### Q: What is the pricing model?
A: The core RAG framework is open source (MIT License). Enterprise support,
custom deployment, and SLA agreements are available on a subscription basis.
Contact enterprise@example.com for pricing.
