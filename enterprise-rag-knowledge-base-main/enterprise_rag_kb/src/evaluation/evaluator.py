"""
Enterprise RAG - Evaluation Module
=====================================
Implements RAG-specific evaluation metrics:
1. Context Recall       - Are all answer facts in the retrieved context?
2. Context Precision    - Are retrieved chunks actually relevant?
3. Answer Faithfulness  - Is the answer grounded in the context?
4. Answer Relevancy     - Does the answer address the question?

These are inspired by the RAGAS framework.
"""

import re
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import config
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalSample:
    """A single evaluation sample."""
    question: str
    ground_truth_answer: str
    generated_answer: str = ""
    retrieved_contexts: List[str] = field(default_factory=list)
    context_scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Evaluation result for a single sample."""
    question: str
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    overall_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "answer_faithfulness": round(self.answer_faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "overall_score": round(self.overall_score, 4),
        }


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using multiple metrics.

    Metrics:
    - Context Precision: What fraction of retrieved chunks are actually relevant?
    - Context Recall: How much of the ground truth is covered by retrieved context?
    - Answer Faithfulness: Is the answer supported by retrieved context?
    - Answer Relevancy: Does the answer directly address the question?
    """

    def __init__(self, use_semantic_similarity: bool = True):
        self.use_semantic = use_semantic_similarity
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            from src.embeddings.embedding_engine import EmbeddingEngine
            self._embedder = EmbeddingEngine()
        return self._embedder

    # ── Core Metrics ─────────────────────────────────────────────────────────

    def compute_context_precision(
        self,
        question: str,
        contexts: List[str],
        ground_truth: str,
    ) -> float:
        """
        Context Precision = # relevant chunks / # total retrieved chunks.
        A chunk is "relevant" if it contains information useful for the answer.
        """
        if not contexts:
            return 0.0

        relevant_count = 0
        gt_words = set(ground_truth.lower().split())

        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            overlap = len(gt_words & ctx_words) / max(len(gt_words), 1)
            if overlap > 0.1 or self._text_contains_answer(ctx, ground_truth):
                relevant_count += 1

        return relevant_count / len(contexts)

    def compute_context_recall(
        self,
        ground_truth: str,
        contexts: List[str],
    ) -> float:
        """
        Context Recall = fraction of ground truth covered by retrieved context.
        Measures how much of the correct answer is in the retrieved chunks.
        """
        if not contexts or not ground_truth:
            return 0.0

        combined_context = " ".join(contexts).lower()
        gt_sentences = self._split_sentences(ground_truth)

        if not gt_sentences:
            return 0.0

        covered = sum(
            1 for sent in gt_sentences
            if self._sentence_covered_in_context(sent, combined_context)
        )

        return covered / len(gt_sentences)

    def compute_answer_faithfulness(
        self,
        answer: str,
        contexts: List[str],
    ) -> float:
        """
        Answer Faithfulness = fraction of answer claims supported by context.
        Penalizes hallucination (claims not in the retrieved context).
        """
        if not answer or not contexts:
            return 0.0

        combined_context = " ".join(contexts).lower()
        answer_sentences = self._split_sentences(answer)

        if not answer_sentences:
            return 0.0

        faithful_count = sum(
            1 for sent in answer_sentences
            if self._sentence_covered_in_context(sent, combined_context)
        )

        return faithful_count / len(answer_sentences)

    def compute_answer_relevancy(
        self,
        question: str,
        answer: str,
    ) -> float:
        """
        Answer Relevancy = how well the answer addresses the question.
        Uses semantic similarity between question and answer.
        """
        if not answer or not question:
            return 0.0

        # Check for non-answers
        non_answer_phrases = [
            "i don't know", "cannot answer", "not enough information",
            "no information", "unable to", "not found in"
        ]
        if any(p in answer.lower() for p in non_answer_phrases):
            return 0.2

        try:
            # Semantic similarity between question and answer
            q_emb = self.embedder.embed_query(question)
            a_emb = self.embedder.embed_query(answer[:500])  # truncate long answers

            similarity = float(np.dot(q_emb, a_emb) / (
                np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-8
            ))
            return max(0.0, min(1.0, (similarity + 1) / 2))

        except Exception:
            # Fallback: keyword overlap between question and answer
            q_words = set(question.lower().split()) - {'what', 'how', 'why', 'is', 'are', 'the', 'a'}
            a_words = set(answer.lower().split())
            overlap = len(q_words & a_words) / max(len(q_words), 1)
            return min(1.0, overlap * 1.5)

    def evaluate_sample(self, sample: EvalSample) -> EvalResult:
        """Run all metrics on a single sample."""
        precision = self.compute_context_precision(
            sample.question, sample.retrieved_contexts, sample.ground_truth_answer
        )
        recall = self.compute_context_recall(
            sample.ground_truth_answer, sample.retrieved_contexts
        )
        faithfulness = self.compute_answer_faithfulness(
            sample.generated_answer, sample.retrieved_contexts
        )
        relevancy = self.compute_answer_relevancy(
            sample.question, sample.generated_answer
        )

        overall = np.mean([precision, recall, faithfulness, relevancy])

        return EvalResult(
            question=sample.question,
            context_precision=precision,
            context_recall=recall,
            answer_faithfulness=faithfulness,
            answer_relevancy=relevancy,
            overall_score=float(overall),
        )

    def evaluate_dataset(self, samples: List[EvalSample]) -> Dict[str, Any]:
        """
        Evaluate a full dataset and return aggregate metrics.
        """
        logger.info(f"Evaluating {len(samples)} samples...")
        results = []

        for i, sample in enumerate(samples):
            result = self.evaluate_sample(sample)
            results.append(result)
            if (i + 1) % 10 == 0:
                logger.info(f"  Evaluated {i+1}/{len(samples)}")

        # Aggregate
        agg = {
            "num_samples": len(results),
            "context_precision": float(np.mean([r.context_precision for r in results])),
            "context_recall": float(np.mean([r.context_recall for r in results])),
            "answer_faithfulness": float(np.mean([r.answer_faithfulness for r in results])),
            "answer_relevancy": float(np.mean([r.answer_relevancy for r in results])),
            "overall_score": float(np.mean([r.overall_score for r in results])),
            "per_sample": [r.to_dict() for r in results],
        }

        logger.info(
            f"Evaluation complete:\n"
            f"  Context Precision:    {agg['context_precision']:.4f}\n"
            f"  Context Recall:       {agg['context_recall']:.4f}\n"
            f"  Answer Faithfulness:  {agg['answer_faithfulness']:.4f}\n"
            f"  Answer Relevancy:     {agg['answer_relevancy']:.4f}\n"
            f"  Overall Score:        {agg['overall_score']:.4f}"
        )

        return agg

    def save_results(self, results: Dict, output_path: str = None):
        """Save evaluation results to JSON."""
        output_path = output_path or config.evaluation.results_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    def load_eval_dataset(self, path: str) -> List[EvalSample]:
        """Load evaluation dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)

        samples = []
        for item in data:
            samples.append(EvalSample(
                question=item["question"],
                ground_truth_answer=item.get("answer", item.get("ground_truth", "")),
                metadata=item.get("metadata", {}),
            ))
        return samples

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if len(s.split()) >= 3]

    def _text_contains_answer(self, context: str, answer: str) -> bool:
        answer_words = set(answer.lower().split()) - {'the', 'a', 'an', 'is', 'are', 'was'}
        context_lower = context.lower()
        matches = sum(1 for w in answer_words if w in context_lower)
        return matches / max(len(answer_words), 1) > 0.3

    def _sentence_covered_in_context(self, sentence: str, context: str) -> bool:
        words = [w for w in sentence.lower().split() if len(w) > 3]
        if not words:
            return True
        matches = sum(1 for w in words if w in context)
        return matches / len(words) > 0.5
