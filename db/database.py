"""Database engine, session factory, and transaction boundary.

Self-contained: loads `.env` on import so standalone entry points (Alembic's
`env.py`, smoke scripts) get `DATABASE_URL` without going through `main.py`.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL is not set. Add it to .env "
        "(postgresql://skillab:skillab_dev@localhost:5432/skillab)."
    )

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


@contextmanager
def transaction() -> Iterator[Session]:
    """Yield a session, commit on success, roll back on error, always close."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
