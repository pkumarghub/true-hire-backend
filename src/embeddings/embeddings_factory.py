from __future__ import annotations

import os
from typing import Literal

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

try:  # optional
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:  # pragma: no cover
    GoogleGenerativeAIEmbeddings = None  # type: ignore


EmbeddingsProvider = Literal["openai", "gemini", "local"]


def get_embeddings(provider: EmbeddingsProvider | None = None):
    provider = provider or os.getenv("EMBEDDINGS_PROVIDER", "local").lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")
        return OpenAIEmbeddings(api_key=api_key)
    if provider == "gemini":
        if GoogleGenerativeAIEmbeddings is None:
            raise RuntimeError("google-generativeai not installed/configured for Gemini embeddings")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini embeddings")
        return GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=api_key)
    # local
    model_name = os.getenv("LOCAL_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)


