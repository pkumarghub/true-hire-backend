from __future__ import annotations

from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(path: str, doc_type: str) -> List[Document]:
    loader = PyPDFLoader(path)
    pages = loader.load()
    for d in pages:
        d.metadata["doc_type"] = doc_type
        d.metadata.setdefault("source", path)
    return pages


