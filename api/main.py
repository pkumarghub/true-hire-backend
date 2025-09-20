from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.loaders.pdf_loader import load_pdf
from src.loaders.docx_loader import load_docx
from src.loaders.txt_loader import load_txt
from src.embeddings.embeddings_factory import get_embeddings
from src.vectorstore.chroma_client import ChromaClient
from src.retriever.retriever_factory import build_retriever
from src.qa.retrieval_qa import build_retrieval_qa_chain
from src.llm.openai_llm import get_openai_llm
from src.llm.gemini_llm import get_gemini_llm
# from src.utils.preprocessing import preprocess_text  # Not needed for now

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

PORT = os.getenv("PORT", 9000)

app = FastAPI(
    title="CV Shortlisting API",
    description="AI-powered CV shortlisting based on Job Description matching",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class CVShortlistRequest(BaseModel):
    """Request model for CV shortlisting"""
    num_shortlisted: int = Field(ge=1, le=50, description="Number of CVs to shortlist")
    llm_provider: str = Field(description="LLM provider: 'openai' or 'gemini'")
    embedding_provider: str = Field(description="Embedding provider: 'openai', 'gemini', or 'local'")
    jd_text: Optional[str] = Field(None, description="Job description as text")
    
    class Config:
        json_schema_extra = {
            "example": {
                "num_shortlisted": 3,
                "llm_provider": "openai",
                "embedding_provider": "local",
                "jd_text": "We are looking for a Python developer with experience in FastAPI..."
            }
        }

class Candidate(BaseModel):
    """Individual candidate information"""
    candidate_id: str
    score: float = Field(description="Similarity score (0-1)")
    metadata: dict = Field(description="Candidate metadata")
    content_preview: str = Field(description="First 800 characters of CV content")

class CVShortlistResponse(BaseModel):
    """Response model for CV shortlisting"""
    success: bool
    message: str
    shortlisted_candidates: List[Candidate]
    total_candidates_processed: int
    jd_summary: Optional[str] = None

# Global variables
persist_dir = os.getenv("CHROMA_DB_PERSIST_DIR", "./chroma_db")
client = ChromaClient(persist_dir=persist_dir)
collection_resumes = "resumes"
collection_jds = "job_descriptions"

def load_file_content(file_path: str, doc_type: str) -> List[str]:
    """Load file content and return list of page contents"""
    if file_path.lower().endswith(".pdf"):
        docs = load_pdf(file_path, doc_type)
    elif file_path.lower().endswith(".docx"):
        docs = load_docx(file_path, doc_type)
    else:
        docs = load_txt(file_path, doc_type)
    
    return [doc.page_content for doc in docs]

def save_uploaded_file(upload_file: UploadFile, temp_dir: Path) -> str:
    """Save uploaded file to temporary directory"""
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = temp_dir / upload_file.filename
    
    with open(file_path, "wb") as f:
        content = upload_file.file.read()
        f.write(content)
    
    return str(file_path)

@app.post("/shortlist-cvs", response_model=CVShortlistResponse)
async def shortlist_cvs(
    num_shortlisted: int = Form(ge=1, le=50, description="Number of CVs to shortlist"),
    llm_provider: str = Form(description="LLM provider: 'openai' or 'gemini'"),
    embedding_provider: str = Form(description="Embedding provider: 'openai', 'gemini', or 'local'"),
    jd_file: Optional[UploadFile] = File(None, description="Job description file (.pdf/.docx/.txt)"),
    jd_text: Optional[str] = Form(None, description="Job description as text"),
    cv_files: List[UploadFile] = File(..., description="CV files (.pdf/.docx/.txt)")
):
    """
    Shortlist CVs based on Job Description matching
    
    This endpoint processes uploaded CVs and shortlists the best matches
    based on semantic similarity with the provided job description.
    """
    try:
        # Validate inputs
        if not jd_text and not jd_file:
            raise HTTPException(status_code=400, detail="Either JD text or JD file must be provided")
        
        if not cv_files:
            raise HTTPException(status_code=400, detail="At least one CV file must be provided")
        
        if llm_provider not in ["openai", "gemini"]:
            raise HTTPException(status_code=400, detail="LLM provider must be 'openai' or 'gemini'")
        
        if embedding_provider not in ["openai", "gemini", "local"]:
            raise HTTPException(status_code=400, detail="Embedding provider must be 'openai', 'gemini', or 'local'")
        
        # Process JD
        final_jd_text = jd_text.strip() if jd_text else ""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Handle JD file if provided
            if jd_file:
                jd_file_path = save_uploaded_file(jd_file, temp_path / "jd")
                jd_contents = load_file_content(jd_file_path, "jd")
                final_jd_text = "\n\n".join(jd_contents).strip()
            
            if not final_jd_text:
                raise HTTPException(status_code=400, detail="No valid JD content found")
            
            # Process CVs
            cv_docs = []
            for cv_file in cv_files:
                cv_file_path = save_uploaded_file(cv_file, temp_path / "cvs")
                cv_contents = load_file_content(cv_file_path, "resume")
                
                for i, content in enumerate(cv_contents):
                    from langchain_core.documents import Document
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": cv_file.filename,
                            "doc_type": "resume",
                            "page": i + 1
                        }
                    )
                    cv_docs.append(doc)
            
            if not cv_docs:
                raise HTTPException(status_code=400, detail="No valid CV content found")
            
            # Index CVs
            client.add_documents(collection_resumes, cv_docs)
            
            # Perform similarity search
            results = client.similarity_search(collection_resumes, final_jd_text, k=num_shortlisted)
            
            if not results:
                return CVShortlistResponse(
                    success=False,
                    message="No matching CVs found",
                    shortlisted_candidates=[],
                    total_candidates_processed=len(cv_docs)
                )
            
            # Format results
            candidates = []
            for idx, (doc, score) in enumerate(results):
                candidate = Candidate(
                    candidate_id=f"candidate_{idx + 1}",
                    score=round(score, 3),
                    metadata=doc.metadata,
                    content_preview=doc.page_content[:800] + ("..." if len(doc.page_content) > 800 else "")
                )
                candidates.append(candidate)
            
            # Generate summary using LLM (optional)
            jd_summary = None
            try:
                retriever = build_retriever(client, collection_resumes, k=3)
                llm = get_openai_llm() if llm_provider == "openai" else get_gemini_llm()
                chain = build_retrieval_qa_chain(retriever, llm)
                response = chain.invoke(final_jd_text)
                jd_summary = response.content if hasattr(response, "content") else str(response)
            except Exception as e:
                # Log error but don't fail the request
                print(f"Warning: Could not generate summary: {e}")
            
            return CVShortlistResponse(
                success=True,
                message=f"Successfully shortlisted {len(candidates)} candidates",
                shortlisted_candidates=candidates,
                total_candidates_processed=len(cv_docs),
                jd_summary=jd_summary
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info"
    )
