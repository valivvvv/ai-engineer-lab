"""Text → vector translator, shared by the write path (ingestion) and the read
path (RAG search).

The sentence-transformer model is loaded once per process, lazily on first use:
loading reads ~120MB of weights and takes seconds, so we neither pay that at
import time nor repeat it per call. Both `embed` and `embed_batch` return plain
Python lists so the values drop straight into pgvector's `Vector(384)` column.
"""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSIONS = 384


class EmbeddingService:
    _model: SentenceTransformer | None = None

    @classmethod
    def _get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            cls._model = SentenceTransformer(EMBEDDING_MODEL)
        return cls._model

    def embed(self, text: str) -> list[float]:
        return self._get_model().encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._get_model().encode(texts)
        return [embedding.tolist() for embedding in embeddings]