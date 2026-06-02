"""Ingestion (write) path: a document file in, structured fields + chunks out.

Phase 1 stops at returning `(structured_object, chunks)` in memory; embedding and
persistence are added in Phase 3.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from agent.llm_factory import LLMFactory
from .chunking import should_chunk_for_extraction, split_text
from .loaders import load_document
from .schemas import Contract, Invoice

EXTRACTION_PROVIDER = "gemini"
EXTRACTION_MODEL = "gemini-2.5-flash"
EXTRACTION_CHUNK_LIMIT = 3

_SCHEMA_BY_KIND: dict[str, type[BaseModel]] = {
    "invoice": Invoice,
    "contract": Contract,
}


class IngestionPipeline:
    def __init__(
        self,
        provider: str = EXTRACTION_PROVIDER,
        model: str = EXTRACTION_MODEL,
    ) -> None:
        self._llm = LLMFactory.create(provider=provider, model=model, temperature=0.0)

    def process(
        self, path: str | Path, document_kind: str
    ) -> tuple[BaseModel, list[str]]:
        schema = _SCHEMA_BY_KIND.get(document_kind)
        if schema is None:
            supported = ", ".join(sorted(_SCHEMA_BY_KIND))
            raise ValueError(
                f"Unknown document_kind '{document_kind}'. Supported: {supported}."
            )
        text = load_document(path)
        chunks = split_text(text)
        structured_object = self._extract(text, chunks, schema)
        return structured_object, chunks

    def _extract(
        self, text: str, chunks: list[str], schema: type[BaseModel]
    ) -> BaseModel:
        extractor = self._llm.with_structured_output(schema)
        if should_chunk_for_extraction(text):
            extraction_input = "\n\n".join(chunks[:EXTRACTION_CHUNK_LIMIT])
        else:
            extraction_input = text
        return extractor.invoke(extraction_input)
