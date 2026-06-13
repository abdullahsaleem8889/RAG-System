"""
Enterprise RAG - Evaluation Script
=====================================
Runs comprehensive evaluation of the RAG pipeline.
Run: python scripts/evaluate_pipeline.py
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# Sample evaluation dataset
EVAL_DATASET = [
    {
        "question": "What is the return policy for standard customers?",
        "answer": "Standard customers may return products within 30 days of purchase for a full refund, provided the item is in original condition with all packaging intact."
    },
    {
        "question": "How long does a Premium member have to return items?",
        "answer": "Premium tier members enjoy a 90-day return window with no questions asked."
    },
    {
        "question": "What data classification tiers exist?",
        "answer": "There are four tiers: Tier 1 (Public), Tier 2 (Internal), Tier 3 (Confidential), and Tier 4 (Restricted)."
    },
    {
        "question": "What is the password minimum length requirement?",
        "answer": "Passwords must be a minimum of 12 characters and include uppercase, lowercase, numbers, and symbols."
    },
    {
        "question": "What was the total revenue in fiscal year 2024?",
        "answer": "Total revenue reached $847.3 million, representing a 23% year-over-year growth."
    },
    {
        "question": "Which retrieval mode is recommended for enterprise use?",
        "answer": "Hybrid mode is recommended for most enterprise use cases as it handles both conceptual queries and exact keyword matches."
    },
    {
        "question": "What embedding model is used by default?",
        "answer": "The default embedding model is sentence-transformers/all-MiniLM-L6-v2, which produces 384-dimensional vectors."
    },
    {
        "question": "How quickly must a data breach be reported?",
        "answer": "Suspected data breaches must be reported to the CISO within 2 hours of discovery, and affected parties notified within 72 hours per GDPR requirements."
    },
]


def run_evaluation():
    """Run full pipeline evaluation."""
    from src.pipeline.rag_pipeline import RAGPipeline
    from src.evaluation.evaluator import RAGEvaluator, EvalSample

    print("=" * 60)
    print("Enterprise RAG - Pipeline Evaluation")
    print("=" * 60)

    # Load pipeline
    print("\n1. Loading pipeline...")
    pipeline = RAGPipeline(auto_load=True)

    if not pipeline.vector_store.is_loaded():
        print("  Knowledge base is empty. Ingesting sample documents first...")
        from scripts.ingest_documents import create_sample_documents
        create_sample_documents()
        pipeline.ingest_directory("data/sample_documents")

    stats = pipeline.get_stats()
    print(f"✓ Pipeline ready: {stats['vector_store']['total_chunks']} chunks")

    # Run queries and collect results
    print("\n2. Running evaluation queries...")
    evaluator = RAGEvaluator(use_semantic_similarity=False)
    samples = []

    for i, item in enumerate(EVAL_DATASET):
        print(f"\n  [{i+1}/{len(EVAL_DATASET)}] Q: {item['question'][:60]}...")
        result = pipeline.query(item["question"], top_k=5, return_sources=True)

        contexts = [s["preview"] for s in result.get("sources", [])]

        sample = EvalSample(
            question=item["question"],
            ground_truth_answer=item["answer"],
            generated_answer=result.get("answer", ""),
            retrieved_contexts=contexts,
        )
        samples.append(sample)

        print(f"     Answer (first 80 chars): {result.get('answer', '')[:80]}...")
        print(f"     Confidence: {result.get('confidence', 0):.2f}")

    # Evaluate
    print("\n3. Computing metrics...")
    eval_results = evaluator.evaluate_dataset(samples)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n Aggregate Metrics ({eval_results['num_samples']} samples):")
    print(f"  ┌─────────────────────────────────┬────────┬─────┐")
    print(f"  │ Metric                          │ Score  │ Grade│")
    print(f"  ├─────────────────────────────────┼────────┼─────┤")

    metrics = [
        ("Context Precision", eval_results["context_precision"]),
        ("Context Recall", eval_results["context_recall"]),
        ("Answer Faithfulness", eval_results["answer_faithfulness"]),
        ("Answer Relevancy", eval_results["answer_relevancy"]),
        ("Overall Score", eval_results["overall_score"]),
    ]

    for name, score in metrics:
        grade = "A" if score > 0.8 else ("B" if score > 0.6 else ("C" if score > 0.4 else "D"))
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"  │ {name:31s} │ {score:.4f} │  {grade}  │")

    print(f"  └─────────────────────────────────┴────────┴─────┘")

    # Per-sample results
    print("\n Per-Sample Results:")
    for r in eval_results["per_sample"]:
        overall = r["overall_score"]
        print(f"  • {r['question'][:50]:50s} → {overall:.3f}")

    # Save results
    output_path = "data/eval_results.json"
    evaluator.save_results(eval_results, output_path)
    print(f"\n Results saved to {output_path}")

    return eval_results


if __name__ == "__main__":
    run_evaluation()
