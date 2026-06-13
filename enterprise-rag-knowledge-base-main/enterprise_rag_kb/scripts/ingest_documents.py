"""
Enterprise RAG - Sample Document Creator & Ingestor
=====================================================
Creates rich sample documents and ingests them into the knowledge base.
Run: python scripts/ingest_documents.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_sample_documents():
    """Create diverse sample documents for demonstration."""
    from config import SAMPLE_DOCS_DIR
    samples_dir = SAMPLE_DOCS_DIR
    samples_dir.mkdir(parents=True, exist_ok=True)

    # ── Document 1: Company Policy ────────────────────────────────────────────
    (samples_dir / "company_policy.txt").write_text("""
ENTERPRISE KNOWLEDGE BASE - COMPANY POLICY DOCUMENT
====================================================
Version: 2.4 | Last Updated: 2024 | Department: HR & Legal

## 1. DATA PRIVACY POLICY

1.1 Overview
Our organization is committed to protecting the privacy and security of all data
assets, both internal and customer-facing. This policy applies to all employees,
contractors, and third-party vendors with access to company systems.

1.2 Data Classification
Data is classified into four tiers:
- Tier 1 (Public): Marketing materials, press releases, public documentation
- Tier 2 (Internal): Internal communications, project plans, meeting notes
- Tier 3 (Confidential): Financial data, HR records, customer PII
- Tier 4 (Restricted): Trade secrets, security credentials, M&A information

1.3 Data Handling Requirements
All employees must:
a) Store Tier 3/4 data only on encrypted, company-approved storage systems
b) Never transmit sensitive data via personal email or unsecured channels
c) Report any suspected data breach within 2 hours of discovery
d) Complete annual data privacy training (minimum 4 hours)

1.4 Breach Response Protocol
In the event of a data breach:
1. Immediately isolate affected systems
2. Notify the CISO within 2 hours
3. Document the scope and nature of the breach
4. Notify affected parties within 72 hours per GDPR requirements
5. Conduct post-incident analysis within 30 days

## 2. RETURN AND REFUND POLICY

2.1 Standard Returns
Customers may return products within 30 days of purchase for a full refund
provided the item is in original condition with all packaging intact.

2.2 Extended Returns for Premium Members
Premium tier members enjoy a 90-day return window with no questions asked.
Refunds are processed within 3-5 business days to the original payment method.

2.3 Non-Returnable Items
The following cannot be returned:
- Digital downloads and software licenses (once activated)
- Perishable goods
- Customized or personalized products
- Items marked "Final Sale"

2.4 Defective Products
Defective products may be returned at any time within the warranty period
(12 months standard, 24 months for enterprise customers). The company will
cover all shipping costs for defective item returns.

## 3. EMPLOYEE CODE OF CONDUCT

3.1 Professional Standards
All employees are expected to:
- Maintain a high standard of professional conduct at all times
- Treat colleagues, customers, and partners with respect and dignity
- Act in the best interests of the company while maintaining ethical standards
- Avoid conflicts of interest and disclose any potential conflicts promptly

3.2 Social Media Policy
Employees must not:
- Share confidential information on social media platforms
- Make disparaging remarks about the company, clients, or colleagues
- Represent personal opinions as official company positions
Employees are encouraged to share positive company news and achievements
after obtaining approval from the Communications department.

3.3 Anti-Harassment Policy
The company maintains a zero-tolerance policy for harassment of any kind,
including but not limited to: sexual harassment, racial discrimination, age
discrimination, disability discrimination, and bullying. All complaints will
be investigated within 5 business days and appropriate action taken.

## 4. INFORMATION TECHNOLOGY POLICY

4.1 Acceptable Use
Company IT resources including computers, network, email, and software are
provided for business purposes. Limited personal use is permitted provided it:
- Does not interfere with work productivity
- Does not violate any laws or company policies
- Does not consume excessive bandwidth or storage

4.2 Password Requirements
- Minimum 12 characters, must include uppercase, lowercase, numbers, and symbols
- Passwords must be changed every 90 days
- Multi-factor authentication is mandatory for all accounts
- Password sharing is strictly prohibited

4.3 Remote Work Security
Remote workers must:
- Use company-provided VPN for all work-related activities
- Ensure home network is password-protected with WPA3 encryption
- Never use public Wi-Fi without VPN active
- Lock screen when stepping away from workstation
""", encoding="utf-8")

    # ── Document 2: Technical Documentation ──────────────────────────────────
    (samples_dir / "technical_guide.md").write_text("""
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
""", encoding="utf-8")

    # ── Document 3: Financial Report ──────────────────────────────────────────
    (samples_dir / "financial_report.txt").write_text("""
ANNUAL FINANCIAL REPORT - FISCAL YEAR 2024
==========================================
Enterprise Solutions Corporation | Confidential

EXECUTIVE SUMMARY
-----------------
Fiscal Year 2024 marked a transformational year for Enterprise Solutions
Corporation, with total revenue reaching $847.3 million, representing a 23%
year-over-year growth. Net income increased to $127.2 million (15% net margin),
compared to $98.4 million in FY2023.

KEY FINANCIAL HIGHLIGHTS
------------------------
Revenue Breakdown by Segment:
- Software Licenses:    $312.4M  (36.9% of total, +18% YoY)
- SaaS Subscriptions:   $298.7M  (35.3% of total, +41% YoY)
- Professional Services: $156.8M (18.5% of total, +12% YoY)
- Maintenance/Support:   $79.4M  (9.4% of total, +8% YoY)

Operating Expenses:
- Research & Development: $169.5M (20% of revenue)
- Sales & Marketing:      $211.8M (25% of revenue)
- General & Administrative: $84.7M (10% of revenue)

EBITDA: $198.3M (23.4% margin)
Free Cash Flow: $143.7M
Cash and equivalents: $421.6M

GROWTH DRIVERS
--------------
1. AI-Powered Products: Our AI product suite generated $156M in new ARR,
   representing 186% growth from the prior year. Key products include:
   - DataSense AI (predictive analytics): $68M ARR
   - DocuMind (intelligent document processing): $52M ARR
   - KnowledgeBase Pro (RAG platform): $36M ARR

2. Enterprise Customer Expansion: Average contract value increased from
   $124K to $187K (51% increase). Net Revenue Retention Rate: 128%.

3. International Markets: International revenue grew 34% to $254M,
   representing 30% of total revenue. EMEA grew 45%, APAC grew 29%.

RISK FACTORS
------------
1. Competition Risk: Increasing competition from hyperscalers (AWS, Azure, GCP)
   could pressure pricing and market share in cloud-based offerings.

2. Talent Retention: In a competitive AI talent market, retaining key engineers
   and researchers remains challenging. Average engineering salary increased 18%.

3. Regulatory Risk: GDPR, CCPA, and emerging AI regulations may require
   significant compliance investments in FY2025.

OUTLOOK FOR FY2025
------------------
Revenue Guidance: $1.02B - $1.05B (20-24% growth)
Operating Margin Target: 18-20%
R&D Investment: Planned increase to 22% of revenue
Key Initiatives:
- Launch of RAG Enterprise v3.0 with multimodal capabilities
- Expansion to 5 new international markets
- Acquisition of 2 AI startups for talent and technology

The Board of Directors has approved a share buyback program of $100M
over 18 months, reflecting confidence in the company's financial position.
""", encoding="utf-8")

    # ── Document 4: FAQ ───────────────────────────────────────────────────────
    (samples_dir / "faq.md").write_text("""
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
""", encoding="utf-8")

    print(f" Created 4 sample documents in {samples_dir}")
    return samples_dir


def main():
    """Main ingestion script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enterprise RAG - Document Ingestion Script"
    )
    parser.add_argument(
        "--dir", type=str, default=None,
        help="Directory to ingest (default: data/raw)"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Single file to ingest"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Create and ingest sample documents"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Reset the index before ingesting"
    )
    args = parser.parse_args()

    from src.pipeline.rag_pipeline import RAGPipeline

    print(" Enterprise RAG - Document Ingestion")
    print("=" * 50)

    pipeline = RAGPipeline(auto_load=not args.reset)

    if args.reset:
        print("  Resetting index...")
        pipeline.reset_index()

    if args.sample:
        print(" Creating sample documents...")
        create_sample_documents()
        from config import SAMPLE_DOCS_DIR
        result = pipeline.ingest_directory(str(SAMPLE_DOCS_DIR))
    elif args.file:
        result = pipeline.ingest_file(args.file)
    elif args.dir:
        result = pipeline.ingest_directory(args.dir)
    else:
        # Default: try raw dir
        from config import RAW_DATA_DIR, SAMPLE_DOCS_DIR
        raw_dir = RAW_DATA_DIR
        if not any(raw_dir.iterdir()) if raw_dir.exists() else True:
            print(" No documents in data/raw. Creating sample documents...")
            create_sample_documents()
            result = pipeline.ingest_directory(str(SAMPLE_DOCS_DIR))
        else:
            result = pipeline.ingest_directory(str(raw_dir))

    print("\n Ingestion Summary:")
    for k, v in result.items():
        print(f"  {k:25s}: {v}")

    # Quick test query
    print("\n Running test query...")
    test_result = pipeline.query("What is the return policy?")
    print(f"\nQ: What is the return policy?")
    print(f"A: {test_result['answer'][:300]}...")
    print(f"Confidence: {test_result['confidence']:.2f}")


if __name__ == "__main__":
    main()
