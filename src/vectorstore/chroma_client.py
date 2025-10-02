from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple, Union

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document
from utils.embeddings.embeddings_factory import embedding_factory


class ChromaClient:
    def __init__(self, persist_dir: Optional[str] = None) -> None:
        self.persist_dir = persist_dir or os.getenv("CHROMA_DB_PERSIST_DIR", "./chroma_db")
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)

    def create_collection(self, collection_name: str) -> Collection:
        return self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, collection_name: str, docs: Iterable[Document]) -> List[str]:
        """Add documents using ChromaDB's default embedding function"""
        collection = self.create_collection(collection_name)
        ids: List[str] = []
        metadatas = []
        texts = []
        for d in docs:
            doc_id = d.metadata.get("id") or d.metadata.get("source") or str(len(ids))
            ids.append(str(doc_id))
            metadatas.append(d.metadata)
            texts.append(d.page_content)
        if ids:
            collection.add(ids=ids, metadatas=metadatas, documents=texts)
        return ids

    async def add_documents_with_embeddings(
        self, 
        collection_name: str, 
        docs: Iterable[Document], 
    ) -> List[str]:
        """
        Add documents with custom embeddings using the specified embedding provider.
        
        Args:
            collection_name: Name of the ChromaDB collection
            docs: Iterable of Document objects to store
            
        Returns:
            List of document IDs that were stored
        """
        collection = self.create_collection(collection_name)
        ids: List[str] = []
        metadatas = []
        texts = []
        embedding_vectors = []
        
        for d in docs:
            doc_id = d.metadata.get("id") or d.metadata.get("source") or str(len(ids))
            ids.append(str(doc_id))
            metadatas.append(d.metadata)
            texts.append(d.page_content)
            
            # Generate embedding for this document
            embedding_vector = await embedding_factory.embed(str(d.page_content))
            embedding_vectors.append(embedding_vector)
        
        if ids:
            collection.add(
                ids=ids, 
                metadatas=metadatas, 
                documents=texts,
                embeddings=embedding_vectors
            )
            print(f"Stored {len(ids)} documents with custom embeddings in collection '{collection_name}'")
        
        return ids

    def similarity_search(self, collection_name: str, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        collection = self.create_collection(collection_name)
        results = collection.query(query_texts=[query], n_results=k, include=["metadatas", "documents", "distances"])
        docs: List[Tuple[Document, float]] = []
        for i in range(len(results["ids"][0])):
            text = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            distance = results.get("distances", [[None]])[0][i]
            score = 1.0 - float(distance) if distance is not None else 0.0
            docs.append((Document(page_content=text, metadata=meta), score))
        return docs

    def get_document_by_id(self, collection_name: str, doc_id: str) -> Optional[Document]:
        collection = self.create_collection(collection_name)
        results = collection.get(ids=[doc_id], include=["metadatas", "documents"])
        if results and results.get("documents") and results["documents"][0]:
            return Document(page_content=results["documents"][0], metadata=results["metadatas"][0])
        return None


