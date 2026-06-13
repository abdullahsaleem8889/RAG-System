"""
Enterprise RAG - Text Preprocessor
====================================
Cleans and normalizes text before chunking.
"""

import re
import unicodedata
from typing import List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.logger import get_logger
from src.ingestion.document_loader import Document

logger = get_logger(__name__)


class TextPreprocessor:
    """
    Cleans raw text documents for better RAG performance.
    Handles: unicode normalization, whitespace, special chars,
    boilerplate removal, and language detection.
    """

    def __init__(
        self,
        remove_headers_footers: bool = True,
        normalize_whitespace: bool = True,
        remove_special_chars: bool = False,
        min_line_length: int = 10,
    ):
        self.remove_headers_footers = remove_headers_footers
        self.normalize_whitespace = normalize_whitespace
        self.remove_special_chars = remove_special_chars
        self.min_line_length = min_line_length

        # Patterns to remove
        self._page_number_pattern = re.compile(
            r'^\s*(page\s*\d+|\d+\s*/\s*\d+|\d+)\s*$', re.IGNORECASE | re.MULTILINE
        )
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self._email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self._excessive_newlines = re.compile(r'\n{3,}')
        self._excessive_spaces = re.compile(r' {2,}')
        self._repeated_chars = re.compile(r'(.)\1{4,}')  # e.g., "-----"

    def preprocess(self, document: Document) -> Document:
        """Process a single document."""
        text = document.content

        # 1. Unicode normalization (NFKC = compatibility decomposition + canonical composition)
        text = unicodedata.normalize("NFKC", text)

        # 2. Remove page numbers
        if self.remove_headers_footers:
            text = self._page_number_pattern.sub("", text)

        # 3. Remove repeated separator characters (e.g., "----------")
        text = self._repeated_chars.sub(r'\1\1\1', text)

        # 4. Normalize whitespace
        if self.normalize_whitespace:
            text = self._excessive_spaces.sub(" ", text)
            text = self._excessive_newlines.sub("\n\n", text)

        # 5. Remove very short lines (usually noise)
        if self.min_line_length > 0:
            lines = text.split("\n")
            lines = [
                line for line in lines
                if len(line.strip()) >= self.min_line_length or line.strip() == ""
            ]
            text = "\n".join(lines)

        # 6. Final strip
        text = text.strip()

        # Update document
        document.content = text
        document.metadata["preprocessed"] = True
        document.metadata["char_count"] = len(text)
        document.metadata["word_count"] = len(text.split())

        return document

    def preprocess_batch(self, documents: List[Document]) -> List[Document]:
        """Preprocess a list of documents."""
        processed = []
        for doc in documents:
            try:
                p = self.preprocess(doc)
                if len(p.content) >= 50:  # discard near-empty docs
                    processed.append(p)
            except Exception as e:
                logger.warning(f"Preprocessing failed for doc {doc.doc_id}: {e}")

        logger.info(f"Preprocessed {len(processed)}/{len(documents)} documents")
        return processed

    def extract_metadata_from_text(self, text: str) -> dict:
        """
        Extract additional metadata signals from text content.
        Useful for structured documents with clear sections.
        """
        metadata = {}

        # Detect if document has numbered sections
        if re.search(r'^\d+\.\s+\w+', text, re.MULTILINE):
            metadata["has_numbered_sections"] = True

        # Detect markdown headers
        if re.search(r'^#{1,6}\s+\w+', text, re.MULTILINE):
            metadata["has_markdown_headers"] = True

        # Detect tables
        if re.search(r'\|.+\|.+\|', text):
            metadata["has_tables"] = True

        # Detect code blocks
        if "```" in text or "    def " in text or "    import " in text:
            metadata["has_code"] = True

        # Rough reading time estimate (200 words/minute)
        word_count = len(text.split())
        metadata["estimated_reading_time_min"] = round(word_count / 200, 1)

        return metadata
