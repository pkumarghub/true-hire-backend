from __future__ import annotations

from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader


def load_docx(path: str, doc_type: str) -> List[Document]:
    loader = Docx2txtLoader(path)
    docs = loader.load()
    for d in docs:
        d.metadata["doc_type"] = doc_type
        d.metadata.setdefault("source", path)
    return docs


