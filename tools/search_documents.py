"""search_documents tool — RAG retrieval exposed to the ReAct agent.

Read-only: opens its own short-lived session (the ReAct loop calls tools
without handing them one), builds a DocumentRetriever, and returns the
formatted context string. Errors are left to propagate — ToolWrapper.call
converts them to strings, so no dict-wrapped error handling here.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from db.database import SessionLocal
from rag.retriever import DocumentRetriever
from .registry import register_tool


class SearchDocumentsParams(BaseModel):
    query: str = Field(
        description=(
            "Natural-language search query describing what to find in the "
            "company documents, e.g. 'clauze de reziliere' or 'total factura'."
        ),
        min_length=2,
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="How many of the most relevant excerpts to return.",
    )


@register_tool
def search_documents(params: SearchDocumentsParams) -> str:
    """Searches the indexed company documents (invoices, contracts) and
    returns the most relevant excerpts with their source filename. Use this
    whenever the user asks about contract clauses, invoice details, etc."""
    session = SessionLocal()
    try:
        retriever = DocumentRetriever(session)
        context = retriever.get_context(params.query, top_k=params.top_k)
    finally:
        session.close()
    if not context:
        return "No relevant excerpts found in the indexed documents."
    return context