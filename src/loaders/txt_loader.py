from __future__ import annotations

from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader


def load_txt(path: str, doc_type: str) -> List[Document]:
    loader = TextLoader(path, autodetect_encoding=True)
    docs = loader.load()
    for d in docs:
        d.metadata["doc_type"] = doc_type
        d.metadata.setdefault("source", path)
    return docs


