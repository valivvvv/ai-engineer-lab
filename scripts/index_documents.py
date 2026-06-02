"""On-demand indexer: run the write path over the data samples.

Indexes each document through IngestionPipeline (load → chunk → extract →
embed → store).

- Dedupe: a file whose filename is already in `documents` is skipped, so
  re-runs don't pile up duplicates. (To re-index a changed file, delete its
  row first.)
- Continue-on-error: a single failing file is reported and skipped so the
  rest of the batch still gets indexed.

First run is slow: it downloads the embedding model (~120MB) and makes one
Gemini extraction call per document.

Run: .venv/bin/python scripts/index_documents.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from db.database import SessionLocal
from db.repositories import DocumentRepository
from ingestion.pipeline import IngestionPipeline

DATA_DIR = REPO_ROOT / "data samples"

DOCUMENTS = [
    ("contract_servicii.txt", "contract"),
    ("contract_consultanta.txt", "contract"),
    ("factura_001.txt", "invoice"),
    ("factura_002.txt", "invoice"),
]


def already_indexed(filename: str) -> bool:
    session = SessionLocal()
    try:
        return DocumentRepository(session).get_by_filename(filename) is not None
    finally:
        session.close()


def main() -> None:
    pipeline = IngestionPipeline()
    indexed = 0
    skipped = 0
    failed = 0
    for filename, document_kind in DOCUMENTS:
        if already_indexed(filename):
            skipped += 1
            print(f"skipped {filename} (already indexed)")
            continue
        try:
            document_id = pipeline.process(DATA_DIR / filename, document_kind)
        except Exception as error:
            failed += 1
            print(f"FAILED {filename} ({document_kind}): {error}")
            continue
        indexed += 1
        print(f"indexed {filename} ({document_kind}) -> document_id={document_id}")

    print(f"\ndone: {indexed} indexed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()