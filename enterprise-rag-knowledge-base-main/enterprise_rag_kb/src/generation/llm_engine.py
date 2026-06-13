"""
Enterprise RAG - LLM Generation Engine
=========================================
Generates answers from retrieved context using LLMs.

Supports:
1. Groq API (free, ultra-fast, Llama 3.3 70B) — default/primary
2. Google Gemini API (fallback)
3. Extractive fallback (always works, no model needed)
"""

import os
import re
import json
import requests
import numpy as np
from typing import List, Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import config
from src.logger import get_logger
from src.ingestion.text_splitter import Chunk

logger = get_logger(__name__)


class GenerationResult:
    """Structured result from LLM generation."""

    def __init__(
        self,
        answer: str,
        sources: List[str],
        confidence: float,
        context_used: str,
        model_used: str,
        tokens_used: int = 0,
    ):
        self.answer = answer
        self.sources = sources
        self.confidence = confidence
        self.context_used = context_used
        self.model_used = model_used
        self.tokens_used = tokens_used

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
        }

    def __repr__(self):
        return f"GenerationResult(model={self.model_used}, confidence={self.confidence:.2f}, answer_len={len(self.answer)})"


class LLMEngine:
    """
    Generates grounded answers using retrieved context.

    Features:
    - Multiple backend support (Groq, Gemini, extractive)
    - Context formatting with source attribution
    - Confidence estimation
    - Fallback chain (Groq → Gemini → extractive)
    """

    def __init__(self):
        cfg = config.generation

        # Groq (primary)
        self.groq_api_key = cfg.groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.groq_model = cfg.groq_model

        # Gemini (fallback)
        self.gemini_api_key = cfg.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = cfg.gemini_model

        # Generation params
        self.max_new_tokens = cfg.max_new_tokens
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.system_prompt = cfg.system_prompt
        self.prompt_template = cfg.rag_prompt_template

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Chunk, float]],
        max_context_chars: int = 3000,
    ) -> GenerationResult:
        """
        Generate an answer grounded in retrieved context.

        Args:
            query: User's question
            retrieved_chunks: List of (Chunk, relevance_score)
            max_context_chars: Max chars of context to include

        Returns:
            GenerationResult with answer and metadata
        """
        if not retrieved_chunks:
            return GenerationResult(
                answer="I don't have enough information in the knowledge base to answer this question.",
                sources=[],
                confidence=0.0,
                context_used="",
                model_used="none",
            )

        # Format context from chunks
        context, sources = self._format_context(retrieved_chunks, max_context_chars)

        # Build the prompt
        prompt = self._build_prompt(query, context)

        # Try generation backends in order
        answer, model_used = self._generate_with_fallback(prompt)

        # Estimate confidence based on retrieval scores and answer quality
        confidence = self._estimate_confidence(retrieved_chunks, answer)

        return GenerationResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            context_used=context,
            model_used=model_used,
        )

    def _format_context(
        self,
        chunks: List[Tuple[Chunk, float]],
        max_chars: int,
    ) -> Tuple[str, List[str]]:
        """Format retrieved chunks into a context string with source labels."""
        context_parts = []
        sources = []
        total_chars = 0

        for i, (chunk, score) in enumerate(chunks):
            source = chunk.metadata.get("filename", chunk.metadata.get("source", f"Source {i+1}"))
            page = chunk.metadata.get("page", "")
            page_str = f", page {page}" if page else ""

            section_header = f"[Source {i+1}: {source}{page_str} | relevance: {score:.2f}]"
            section_text = chunk.content.strip()

            section = f"{section_header}\n{section_text}"

            if total_chars + len(section) > max_chars and context_parts:
                break

            context_parts.append(section)
            total_chars += len(section)

            if source not in sources:
                sources.append(f"{source}{page_str}")

        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the full RAG prompt."""
        return self.prompt_template.format(
            context=context,
            question=query,
        )

    def _generate_with_fallback(self, prompt: str) -> Tuple[str, str]:
        """Try generation backends in order: Groq → Gemini → Extractive."""

        # 1. Try Groq API (primary — ultra fast)
        if self.groq_api_key:
            try:
                answer = self._groq_generate(prompt)
                if answer:
                    return answer, f"groq/{self.groq_model}"
            except Exception as e:
                logger.warning(f"Groq API failed: {e}. Trying Gemini fallback...")

        # 2. Try Google Gemini API (fallback)
        if self.gemini_api_key:
            try:
                answer = self._gemini_generate(prompt)
                if answer:
                    return answer, f"gemini/{self.gemini_model}"
            except Exception as e:
                logger.warning(f"Gemini API failed: {e}. Using extractive fallback.")

        # 3. Extractive fallback (always works, no model needed)
        answer = self._extractive_answer(prompt)
        return answer, "extractive-fallback"

    def _groq_generate(self, prompt: str) -> str:
        """Call Groq API (OpenAI-compatible endpoint)."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    def _gemini_generate(self, prompt: str) -> str:
        """Call Google Gemini REST API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_api_key}"

        payload = {
            "contents": [{"parts": [{"text": f"{self.system_prompt}\n\n{prompt}"}]}],
            "generationConfig": {
                "maxOutputTokens": self.max_new_tokens,
                "temperature": self.temperature,
                "topP": self.top_p,
            },
        }

        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _extractive_answer(self, prompt: str) -> str:
        """
        Fallback: extract the most relevant sentence from context.
        No model required — pure text processing.
        """
        # Extract context section from prompt
        context_match = re.search(
            r'### Context Information:\n(.*?)\n### Question:',
            prompt,
            re.DOTALL
        )
        question_match = re.search(r'### Question:\n(.+)', prompt)

        if not context_match or not question_match:
            return "Unable to generate an answer. Please check your knowledge base."

        context = context_match.group(1)
        question = question_match.group(1).lower()

        # Find most relevant sentences using keyword overlap
        sentences = re.split(r'(?<=[.!?])\s+', context)
        query_words = set(question.split()) - {'what', 'how', 'why', 'when', 'where', 'is', 'are', 'the', 'a', 'an'}

        best_sentence = ""
        best_score = 0

        for sentence in sentences:
            if len(sentence) < 20:
                continue
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words & sentence_words)
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence

        if best_sentence:
            return (
                f"Based on the available documents:\n\n{best_sentence.strip()}\n\n"
                " This answer was generated using extractive methods. "
                "For better results, configure an API key in your .env file."
            )

        return "The knowledge base contains relevant documents but I couldn't generate a precise answer. Please review the source documents directly."

    def _estimate_confidence(
        self,
        chunks: List[Tuple[Chunk, float]],
        answer: str,
    ) -> float:
        """
        Estimate answer confidence based on:
        1. Normalized retrieval score (handles both L2-converted and cross-encoder logits)
        2. Answer length / quality
        3. Presence of uncertainty phrases
        """
        if not chunks or not answer:
            return 0.0

        # Normalize retrieval scores to [0, 1]
        raw_scores = [s for _, s in chunks]
        normalized = []
        for s in raw_scores:
            if 0.0 <= s <= 1.0:
                normalized.append(s)
            else:
                sig = 1.0 / (1.0 + np.exp(-float(s)))
                normalized.append(float(sig))

        mean_score = float(np.mean(normalized)) if normalized else 0.0

        # Answer quality component
        uncertainty_phrases = [
            "i don't know", "cannot answer", "not enough information",
            "unable to", "no information", "not found"
        ]
        has_uncertainty = any(p in answer.lower() for p in uncertainty_phrases)
        quality_score = 0.3 if has_uncertainty else min(1.0, len(answer) / 300)

        confidence = 0.6 * mean_score + 0.4 * quality_score
        return round(max(0.0, min(1.0, confidence)), 3)
