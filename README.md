# TrueHire: Resume ↔ JD Semantic Search

TrueHire is an AI-powered CV shortlisting system that matches resumes with job descriptions using semantic search. Built with LangChain, ChromaDB, and FastAPI, it helps recruiters quickly identify the most suitable candidates for a position.

## Features

- **CV Processing**: Upload and process CVs in multiple formats (PDF, DOCX, TXT)
- **Job Description Matching**: Match CVs against job descriptions using semantic search
- **AI-Powered Shortlisting**: Automatically shortlist the best candidates based on job requirements
- **Multiple LLM Support**: Works with OpenAI, Google Gemini, or local embedding models
- **RESTful API**: Easy integration with existing recruitment systems

## Project Structure

```
truehire/
├── api/                # FastAPI application
├── src/                # Core application code
│   ├── controllers/    # Business logic controllers
│   ├── loaders/        # Document loaders (PDF, DOCX, TXT)
│   ├── models/         # Data models
│   ├── routers/        # API routes
│   ├── services/       # Business services
│   ├── utils/          # Utility functions
│   │   ├── embeddings/ # Embedding models
│   │   ├── llm/        # LLM integrations
│   │   ├── qa/         # Question answering
│   │   └── retriever/  # Vector retrieval
│   └── vectorstore/    # ChromaDB integration
└── start_api.py        # API entry point
```

## Installation

### Prerequisites

- Python 3.9+
- [UV](https://github.com/astral-sh/uv) package manager (recommended)
- API keys for OpenAI or Google Gemini (optional for local embeddings)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/truehire.git
cd truehire
```

2. **Create and activate a virtual environment**

```bash
# Using UV (recommended)
pip install uv
uv venv .venv

# Activate the environment
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
```

3. **Install dependencies**

```bash
# Using UV
uv sync
uv pip install -e .  # Install in development mode

# Using pip
# pip install -e .
```

4. **Configure environment variables**

```bash
# Copy the sample environment file
copy .env.sample .env  # Windows
# cp .env.sample .env  # macOS/Linux

# Edit .env to add your API keys and configuration
```

5. **Run tests to verify installation**

```bash
uv run pytest -q
```

## Configuration

Edit the `.env` file to configure the application:

```
# Required for OpenAI embeddings
OPENAI_API_KEY=your_openai_api_key

# Required for Google Gemini embeddings
GEMINI_API_KEY=your_gemini_api_key

# ChromaDB persistence directory
CHROMA_DB_PERSIST_DIR=./chroma_db

# Embedding provider: 'openai', 'gemini', or 'local'
EMBEDDINGS_PROVIDER=openai

# API server port
PORT=9000
```

## Usage

### Starting the API Server

```bash
python start_api.py
```

The API will be available at `http://localhost:9000` with documentation at `http://localhost:9000/docs`.

### API Endpoints

- **POST /api/v1/shortlist-cvs**: Shortlist CVs based on a job description
  - Upload CV files and job description
  - Specify number of CVs to shortlist
  - Choose LLM provider

- **GET /api/v1/db-check**: Check ChromaDB status and collections
- **GET /api/v1/db-purge**: Purge all collections from ChromaDB

## Development

### Adding New Features

1. Follow the existing project structure
2. Add new routes in `src/routers/`
3. Implement business logic in `src/controllers/` and `src/services/`
4. Add new models in `src/models/`

### Extending Embedding Support

Modify `src/utils/embeddings/embeddings_factory.py` to add support for new embedding models.

### Troubleshooting

- **Embedding Dimension Mismatch**: Ensure consistent embedding dimensions when switching providers
- **File Loading Issues**: Check supported file formats in `src/loaders/`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the LLM framework
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [FastAPI](https://github.com/tiangolo/fastapi) for the API framework
