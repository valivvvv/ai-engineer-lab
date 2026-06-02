"""Ingestion (write) path: a document file in, a persisted document out.

Loads the file, splits it into chunks, extracts structured fields with the
extraction LLM, embeds every chunk, then stores the `Document` and its
`DocumentChunk`s in one transaction and returns the new document's id.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

import config
from db.database import transaction
from db.repositories import ChunkRepository, DocumentRepository
from rag.embeddings import EmbeddingService
from .chunking import should_chunk_for_extraction, split_text
from .loaders import load_document
from .schemas import Contract, Invoice

EXTRACTION_CHUNK_LIMIT = 3

_SCHEMA_BY_KIND: dict[str, type[BaseModel]] = {
    "invoice": Invoice,
    "contract": Contract,
}


class IngestionPipeline:
    def __init__(self, model_choice: config.ModelChoice = config.EXTRACTION) -> None:
        self._llm = config.build_model(model_choice)
        self._embedding_service = EmbeddingService()

    def process(self, path: str | Path, document_kind: str) -> int:
        schema = _SCHEMA_BY_KIND.get(document_kind)
        if schema is None:
            supported = ", ".join(sorted(_SCHEMA_BY_KIND))
            raise ValueError(
                f"Unknown document_kind '{document_kind}'. Supported: {supported}."
            )
        text = load_document(path)
        chunks = split_text(text)
        structured_object = self._extract(text, chunks, schema)
        embeddings = self._embedding_service.embed_batch(chunks)

        with transaction() as session:
            document = DocumentRepository(session).create(
                filename=Path(path).name,
                content=text,
                doc_metadata=structured_object.model_dump(),
            )
            ChunkRepository(session).create_batch(document.id, chunks, embeddings)
            return document.id

    def _extract(
        self, text: str, chunks: list[str], schema: type[BaseModel]
    ) -> BaseModel:
        extractor = self._llm.with_structured_output(schema)
        if should_chunk_for_extraction(text):
            extraction_input = "\n\n".join(chunks[:EXTRACTION_CHUNK_LIMIT])
        else:
            extraction_input = text
        return extractor.invoke(extraction_input)
