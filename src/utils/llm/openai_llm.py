from __future__ import annotations

import os
from typing import Optional

from langchain_openai import ChatOpenAI


def get_openai_llm(model: str = "gpt-4o-mini", temperature: float = 0.2) -> ChatOpenAI:
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Set it to use OpenAI.")
    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)


