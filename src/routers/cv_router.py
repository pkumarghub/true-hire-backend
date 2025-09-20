from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile

from src.controllers.cv_controller import CVController
from src.models.cv_shortlist import CVShortlistResponse
from src.services.cv_service import CVService
from src.vectorstore.chroma_client import ChromaClient

# Initialize router
cv_router = APIRouter(prefix="/api/v1", tags=["CV Shortlisting"])

# Initialize services and controllers
persist_dir = "./chroma_db"
chroma_client = ChromaClient(persist_dir=persist_dir)
cv_service = CVService(chroma_client)
cv_controller = CVController(cv_service)


@cv_router.post("/shortlist-cvs", response_model=CVShortlistResponse)
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
    return await cv_controller.shortlist_cvs(
        num_shortlisted=num_shortlisted,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        jd_file=jd_file,
        jd_text=jd_text,
        cv_files=cv_files
    )
