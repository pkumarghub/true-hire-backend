from typing import List, Optional

from fastapi import File, Form, HTTPException, UploadFile

from src.models.cv_shortlist import CVShortlistResponse
from src.services.cv_service import CVService


class CVController:
    """Controller for CV shortlisting operations"""
    
    def __init__(self, cv_service: CVService):
        self.cv_service = cv_service
    
    async def shortlist_cvs(
        self,
        num_shortlisted: int,
        llm_provider: str,
        embedding_provider: str,
        jd_file: Optional[UploadFile],
        jd_text: Optional[str],
        cv_files: List[UploadFile]
    ) -> CVShortlistResponse:
        """Handle CV shortlisting request"""
        
        # Validate inputs
        if not jd_text and not jd_file:
            raise HTTPException(status_code=400, detail="Either JD text or JD file must be provided")
        
        if not cv_files:
            raise HTTPException(status_code=400, detail="At least one CV file must be provided")
        
        if llm_provider not in ["openai", "gemini"]:
            raise HTTPException(status_code=400, detail="LLM provider must be 'openai' or 'gemini'")
        
        if embedding_provider not in ["openai", "gemini", "local"]:
            raise HTTPException(status_code=400, detail="Embedding provider must be 'openai', 'gemini', or 'local'")
        
        try:
            return await self.cv_service.shortlist_cvs(
                num_shortlisted=num_shortlisted,
                llm_provider=llm_provider,
                embedding_provider=embedding_provider,
                jd_file=jd_file,
                jd_text=jd_text,
                cv_files=cv_files
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
