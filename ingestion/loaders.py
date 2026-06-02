"""Maps a file extension to the LangChain loader that can read it.

`load_document` is the single entry point: give it a path, get back the file's
plain text, regardless of the underlying format.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.document_loaders import BaseLoader

_LOADERS_BY_SUFFIX: dict[str, Callable[[str], BaseLoader]] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    ".csv": CSVLoader,
}


def load_document(path: str | Path) -> str:
    resolved_path = Path(path)
    loader_factory = _LOADERS_BY_SUFFIX.get(resolved_path.suffix.lower())
    if loader_factory is None:
        supported = ", ".join(sorted(_LOADERS_BY_SUFFIX))
        raise ValueError(
            f"Unsupported file type '{resolved_path.suffix}' for {resolved_path.name}. "
            f"Supported extensions: {supported}."
        )
    documents = loader_factory(str(resolved_path)).load()
    return "\n\n".join(document.page_content for document in documents)
