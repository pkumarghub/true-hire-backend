from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document


class Retriever(Protocol):
    def get_relevant_documents(self, query: str) -> list[Document]:
        ...


class SimpleChromaRetriever:
    def __init__(self, chroma_client, collection_name: str, k: int = 5) -> None:
        self.chroma_client = chroma_client
        self.collection_name = collection_name
        self.k = k

    def get_relevant_documents(self, query: str) -> list[Document]:
        results = self.chroma_client.similarity_search(self.collection_name, query, self.k)
        return [doc for doc, _ in results]


def build_retriever(chroma_client, collection_name: str, k: int = 5) -> Retriever:
    return SimpleChromaRetriever(chroma_client, collection_name, k)


