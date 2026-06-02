"""RAG read path: embed a question, retrieve the closest chunks, format them.

Read-only by contract — the write path (embedding + persistence) lives in
`ingestion.pipeline.IngestionPipeline`. `DocumentRetriever` only queries.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from db.models import DocumentChunk
from db.repositories import ChunkRepository
from rag.embeddings import EmbeddingService

DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.3


class DocumentRetriever:
    def __init__(self, db: Session) -> None:
        self._chunk_repository = ChunkRepository(db)
        self._embedding_service = EmbeddingService()

    def search(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> list[tuple[DocumentChunk, float]]:
        query_embedding = self._embedding_service.embed(query)
        return self._chunk_repository.similarity_search(query_embedding, top_k=top_k)

    def get_context(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> str:
        relevant = [
            (chunk, score)
            for chunk, score in self.search(query, top_k=top_k)
            if score >= threshold
        ]
        if not relevant:
            return ""
        return "\n\n".join(
            f"[source: {chunk.document.filename} | score: {score:.2f}]\n{chunk.content}"
            for chunk, score in relevant
        )