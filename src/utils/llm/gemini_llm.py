from __future__ import annotations

import os
from typing import Optional

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore


def get_gemini_llm(model: str = "gemini-1.5-pro", temperature: float = 0.2):
    model = os.getenv("GOOGLE_GENERATIVE_AI_MODEL", model)
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError("google-generativeai not installed. Install and set GEMINI_API_KEY.")
    api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY. Set it to use Gemini.")
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)


