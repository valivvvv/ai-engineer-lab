"""Text splitting helpers for the ingestion (write) path.

`split_text` cuts a document into the small, overlapping pieces we embed and
store as `DocumentChunk` rows. `should_chunk_for_extraction` only decides whether
the extraction LLM is handed the full document or just its first chunks — it does
NOT gate the stored chunks; every document still yields at least one.
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks or [text]


def should_chunk_for_extraction(text: str, max_chars: int = 4000) -> bool:
    return len(text) > max_chars