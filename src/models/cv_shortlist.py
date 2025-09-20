from typing import List, Optional
from pydantic import BaseModel, Field


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
