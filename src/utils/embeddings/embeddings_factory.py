from __future__ import annotations

import os
import httpx
from typing import Literal, List, Dict, Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

try:  # optional
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:  # pragma: no cover
    GoogleGenerativeAIEmbeddings = None  # type: ignore


EmbeddingsProvider = Literal["openai", "gemini", "ollama" , "local"]


def get_embeddings(provider: EmbeddingsProvider | None = "ollama"):
    provider = provider or os.getenv("EMBEDDINGS_PROVIDER", "local").lower()
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")
        # OpenAI text-embedding-3-large has 3072 dimensions
        return OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    
    if provider == "gemini":
        if GoogleGenerativeAIEmbeddings is None:
            raise RuntimeError(
                "google-generativeai not installed/configured for Gemini embeddings")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini embeddings")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # Default to local embeddings
    # Using a model with 384 dimensions
    model_name = os.getenv("LOCAL_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)

"""
    BOC - Begin of Ollama Embedding Factory
    Factory for creating embeddings using Ollama API
"""
class OllamaEmbeddingError(Exception):
    pass

class EmbeddingFactory:
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or str(os.getenv("OLLAMA_URL"))
        self.model = model or os.getenv("EMBED_MODEL")
        print(f"Initialized EmbeddingFactory with URL: {self.base_url} and model: {self.model}")

    async def _call_ollama_api(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a call to the Ollama API"""
        url = f"{self.base_url.rstrip('/')}{endpoint}"
        print(f"Calling Ollama API: {url}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload, follow_redirects=True)
                if resp.status_code != 200:
                    print(f"Ollama API call failed: {resp.text}")
                    raise OllamaEmbeddingError(f"Ollama API call failed with status {resp.status_code}")
                
                return resp.json()
        except httpx.ConnectError:
            print(f"Failed to connect to Ollama server at {self.base_url}")
            raise OllamaEmbeddingError("Ollama server connection failed")
        except Exception as e:
            print(f"Error calling Ollama API: {str(e)}")
            raise OllamaEmbeddingError(f"Error calling Ollama API: {str(e)}")

    async def embed(self, text: str) -> List[float]:
        """Get embeddings for a single text string"""
        try:
            # Use only the /api/embeddings endpoint which we confirmed is working
            payload = {"model": self.model, "prompt": text}
            data = await self._call_ollama_api("/api/embeddings", payload)
            
            if "embedding" in data:
                return data["embedding"]
            else:
                print(f"Unexpected response format: {data}")
                raise OllamaEmbeddingError("Unexpected embedding response format")
        except OllamaEmbeddingError:
            raise
        except Exception as e:
            print(f"Error getting embeddings: {str(e)}")
            raise OllamaEmbeddingError(f"Error getting embeddings: {str(e)}")
            
# create a singleton-like instance
embedding_factory = EmbeddingFactory()
