# TrueHire

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0%2B-green)

> AI-powered CV shortlisting system that matches resumes with job descriptions using semantic search

TrueHire helps recruiters quickly identify the most suitable candidates for a position by leveraging advanced NLP techniques. Built with LangChain, ChromaDB, and FastAPI, it provides a robust platform for resume-to-job description matching.

## ✨ Features

- **CV Processing**: Upload and process CVs in multiple formats (PDF, DOCX, TXT)
- **Job Description Matching**: Match CVs against job descriptions using semantic search
- **AI-Powered Shortlisting**: Automatically shortlist the best candidates based on job requirements
- **Multiple LLM Support**: Works with OpenAI, Google Gemini, or local embedding models
- **RESTful API**: Easy integration with existing recruitment systems

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/truehire.git
cd truehire

# Set up environment and install dependencies
pip install uv
uv venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
uv sync

# Configure environment
copy .env.sample .env
# Edit .env with your API keys

# Start the API server
python start_api.py
```

Visit `http://localhost:9000/docs` to explore the API documentation.

## 📋 Prerequisites

- Python 3.9+
- [UV](https://github.com/astral-sh/uv) package manager (recommended)
- API keys for OpenAI or Google Gemini (optional for local embeddings)
- Ollama (for local embeddings)

## 📦 Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/truehire.git
cd truehire
```

### Step 2: Create and activate a virtual environment

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

### Step 3: Install dependencies

```bash
# Using UV
uv sync
uv pip install -e .  # Install in development mode

# Using pip (alternative)
# pip install -e .
```

### Step 4: Configure environment variables

```bash
# Copy the sample environment file
copy .env.sample .env  # Windows
# cp .env.sample .env  # macOS/Linux

# Edit .env to add your API keys and configuration
```

### Step 5: Verify installation

```bash
uv run pytest -q
```

## ⚙️ Configuration

Edit the `.env` file to configure the application:

```ini
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

## 🔍 Usage

### Starting the API Server

```bash
python start_api.py
```

The API will be available at `http://localhost:9000` with documentation at `http://localhost:9000/docs`.

### Example: Shortlisting CVs

```python
import requests

url = 'http://localhost:9000/api/v1/shortlist-cvs'
files = {'cv_files': open('resume.pdf', 'rb')}
data = {
    'num_shortlisted': 3,
    'llm_provider': 'gemini',
    'jd_text': 'Software Engineer with Python experience'
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/shortlist-cvs` | POST | Shortlist CVs based on a job description |
| `/api/v1/db-check` | GET | Check ChromaDB status and collections |
| `/api/v1/db-purge` | GET | Purge all collections from ChromaDB |

## 📁 Project Structure

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
│   │   ├── metadata_extractor.py # Extract metadata from documents
│   │   ├── preprocessing.py # Text preprocessing utilities
│   │   ├── qa/         # Question answering
│   │   └── retriever/  # Vector retrieval
│   └── vectorstore/    # ChromaDB integration
├── start_api.py        # API entry point
└── .env.sample         # Sample environment variables
```

## 🧑‍💻 Development

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the LLM framework
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [FastAPI](https://github.com/tiangolo/fastapi) for the API framework
