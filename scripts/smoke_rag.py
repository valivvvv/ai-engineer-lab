"""RAG smoke: index the sample documents through the write path, then run a
RAG query through the read path and check the ranking.

Re-runnable: it wipes the `documents` table first (chunks cascade away via the
FK), so repeated runs start from a clean, deterministic state.

First run is slow — it downloads the embedding model (~120MB) and makes one
Gemini extraction call per document.

Run: .venv/bin/python scripts/smoke_rag.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import delete

from db.database import transaction
from db.models import Document
from ingestion.pipeline import IngestionPipeline
from rag.retriever import DocumentRetriever

DATA_DIR = REPO_ROOT / "data samples"

DOCUMENTS = [
    ("contract_servicii.txt", "contract"),
    ("contract_consultanta.txt", "contract"),
    ("factura_001.txt", "invoice"),
    ("factura_002.txt", "invoice"),
]

QUERY = "clauze de reziliere"
SCORE_THRESHOLD = 0.3


def reset_storage() -> None:
    with transaction() as session:
        session.execute(delete(Document))
    print("cleared documents + chunks")


def index_documents() -> None:
    pipeline = IngestionPipeline()
    for filename, document_kind in DOCUMENTS:
        document_id = pipeline.process(DATA_DIR / filename, document_kind)
        print(f"indexed {filename} ({document_kind}) -> document_id={document_id}")


def query_documents() -> None:
    with transaction() as session:
        retriever = DocumentRetriever(session)
        results = retriever.search(QUERY, top_k=5)

        print(f"\nquery: {QUERY!r}")
        for chunk, score in results:
            print(f"  {score:.3f}  {chunk.document.filename}#{chunk.chunk_index}")

        assert results, "no chunks retrieved"
        top_chunk, top_score = results[0]
        assert "contract" in top_chunk.document.filename, (
            f"expected a contract chunk on top, got {top_chunk.document.filename}"
        )
        assert top_score >= SCORE_THRESHOLD, (
            f"top score {top_score:.3f} below threshold {SCORE_THRESHOLD}"
        )
    print("\nPASS")


if __name__ == "__main__":
    reset_storage()
    index_documents()
    query_documents()
