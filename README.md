# Resume ↔ JD Semantic Search (LangChain + ChromaDB + Streamlit)

Production-ready scaffold for a GenAI tool that indexes resumes and job descriptions, performs semantic search via ChromaDB, and generates candidate summaries with a RetrievalQA chain.

## Quickstart

```bash
pip install uv
uv venv .venv
. .venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate     # macOS/Linux
uv sync
uv pip install -e .            # ensure src/truehire is importable
copy .env.sample .env          # then edit .env to add keys
uv run streamlit run streamlit_app/app.py
uv run pytest -q

# 7. Check for unused dependencies
uv run deptry
```

## Environment (.env)

```
OPENAI_API_KEY=
GEMINI_API_KEY=
CHROMA_DB_PERSIST_DIR=./chroma_db
EMBEDDINGS_PROVIDER=openai
STREAMLIT_SERVER_PORT=8501
```


## Notes
- Embeddings via `EMBEDDINGS_PROVIDER` (openai|gemini|local). Local uses sentence-transformers.
- ChromaDB persists under `CHROMA_DB_PERSIST_DIR`.
- Extend `embeddings/embeddings_factory.py`, `vectorstore/chroma_client.py`, `qa/retrieval_qa.py` as needed.
