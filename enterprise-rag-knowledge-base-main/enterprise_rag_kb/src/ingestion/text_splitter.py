"""
Enterprise RAG - Advanced Text Splitter
=========================================
Splits documents into chunks using multiple strategies:
1. Recursive character splitting (default)
2. Sentence-aware splitting
3. Semantic boundary detection (header-based)
4. Fixed-size token chunking
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import config
from src.logger import get_logger
from src.ingestion.document_loader import Document

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A text chunk with full provenance metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    doc_id: str = ""
    chunk_index: int = 0

    def __post_init__(self):
        if not self.chunk_id:
            import hashlib
            self.chunk_id = hashlib.md5(
                f"{self.doc_id}_{self.chunk_index}_{self.content[:50]}".encode()
            ).hexdigest()[:12]

    def __repr__(self):
        preview = self.content[:60].replace("\n", " ")
        return f"Chunk(id={self.chunk_id}, doc={self.doc_id}, idx={self.chunk_index}, '{preview}...')"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "metadata": self.metadata,
            "char_count": len(self.content),
            "word_count": len(self.content.split()),
        }


class RecursiveTextSplitter:
    """
    Splits text recursively using a hierarchy of separators.
    Tries each separator in order; if resulting chunks are still too large,
    recursively splits with the next separator.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None,
        min_chunk_size: int = None,
    ):
        cfg = config.chunking
        self.chunk_size = chunk_size or cfg.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.chunk_overlap
        self.min_chunk_size = min_chunk_size or cfg.min_chunk_size
        self.separators = separators or cfg.separators

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """Split a list of documents into chunks."""
        all_chunks = []
        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)

        logger.info(
            f"Split {len(documents)} documents -> {len(all_chunks)} chunks "
            f"(avg {len(all_chunks)//max(1,len(documents))} chunks/doc)"
        )
        return all_chunks

    def split_document(self, document: Document) -> List[Chunk]:
        """Split a single document into chunks."""
        text = document.content
        if not text.strip():
            return []

        # Use semantic splitting for markdown/structured docs
        if document.metadata.get("file_type") in ("md", "markdown"):
            texts = self._split_markdown(text)
        else:
            texts = self._split_text(text, self.separators)

        chunks = []
        for i, (chunk_text, extra_meta) in enumerate(texts):
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue

            meta = {**document.metadata, **extra_meta}
            meta["chunk_index"] = i
            meta["total_chunks_in_doc"] = len(texts)
            meta["doc_id"] = document.doc_id

            chunks.append(Chunk(
                content=chunk_text.strip(),
                metadata=meta,
                doc_id=document.doc_id,
                chunk_index=i,
            ))

        return chunks

    def _split_text(
        self, text: str, separators: List[str]
    ) -> List[Tuple[str, dict]]:
        """Recursively split text using separator hierarchy."""
        if not separators:
            return self._fixed_size_split(text)

        separator = separators[0]
        remaining_seps = separators[1:]

        if separator == "":
            splits = list(text)
        elif separator == ". ":
            splits = re.split(r'(?<=[.!?])\s+', text)
        else:
            splits = text.split(separator)

        # Merge small splits and recurse on large ones
        result_texts = []
        current = ""

        for split in splits:
            if not split.strip():
                continue

            candidate = (current + separator + split).strip() if current else split.strip()

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    result_texts.extend(self._finalize_chunk(current))
                    # Overlap: carry last part of current into next
                    overlap_text = self._get_overlap(current)
                    current = (overlap_text + " " + split).strip() if overlap_text else split.strip()
                else:
                    # Split is itself too large — recurse
                    sub = self._split_text(split, remaining_seps)
                    result_texts.extend(sub)
                    current = ""

        if current.strip():
            result_texts.extend(self._finalize_chunk(current))

        return result_texts

    def _split_markdown(self, text: str) -> List[Tuple[str, dict]]:
        """
        Smart markdown splitter that respects headers.
        Sections under headers become natural chunks.
        """
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        splits = header_pattern.split(text)

        results = []
        current_section = ""
        current_header = ""
        current_level = 0

        lines = text.split("\n")
        section_lines = []
        header_stack = []

        for line in lines:
            m = re.match(r'^(#{1,6})\s+(.+)$', line)
            if m:
                if section_lines:
                    section_text = "\n".join(section_lines).strip()
                    if len(section_text) >= self.min_chunk_size:
                        ctx = {"header": current_header, "header_level": current_level}
                        if len(section_text) > self.chunk_size:
                            for sub, meta in self._split_text(section_text, self.separators[2:]):
                                results.append((sub, {**ctx, **meta}))
                        else:
                            results.append((section_text, ctx))
                section_lines = [line]
                current_header = m.group(2)
                current_level = len(m.group(1))
            else:
                section_lines.append(line)

        # Final section
        if section_lines:
            section_text = "\n".join(section_lines).strip()
            if len(section_text) >= self.min_chunk_size:
                ctx = {"header": current_header, "header_level": current_level}
                results.append((section_text, ctx))

        # Fallback if no headers found
        if not results:
            return self._split_text(text, self.separators)

        return results

    def _finalize_chunk(self, text: str) -> List[Tuple[str, dict]]:
        return [(text.strip(), {})]

    def _get_overlap(self, text: str) -> str:
        """Return the last `chunk_overlap` characters of text for context continuity."""
        if self.chunk_overlap <= 0 or len(text) <= self.chunk_overlap:
            return ""
        # Try to start overlap at a word boundary
        overlap = text[-self.chunk_overlap:]
        space_idx = overlap.find(" ")
        if space_idx > 0:
            overlap = overlap[space_idx:].strip()
        return overlap

    def _fixed_size_split(self, text: str) -> List[Tuple[str, dict]]:
        """Fallback: split by fixed character count."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append((chunk.strip(), {}))
        return chunks

    def get_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Return statistics about the chunk collection."""
        if not chunks:
            return {}

        lengths = [len(c.content) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_length": round(sum(lengths) / len(lengths)),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_chars": sum(lengths),
            "unique_docs": len(set(c.doc_id for c in chunks)),
        }
