#!/usr/bin/env python3
"""
Startup script for the CV Shortlisting API
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

PORT = os.getenv("PORT", 9000)

# Set up environment variables
os.environ.setdefault("CHROMA_DB_PERSIST_DIR", "./chroma_db")
os.environ.setdefault("EMBEDDINGS_PROVIDER", "local")
os.environ.setdefault("LOCAL_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if __name__ == "__main__":
    import uvicorn
    from api.main import app
    
    print("🚀 Starting CV Shortlisting API Server")
    print("=" * 50)
    print("📖 API Documentation: http://localhost:", PORT, "/docs")
    print("🔍 Health Check: http://localhost:", PORT, "/health")
    print("=" * 50)
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info"
    )
