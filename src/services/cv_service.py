import os
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import UploadFile
from langchain_core.documents import Document

from src.models.cv_shortlist import CVShortlistResponse, Candidate
from src.services.file_service import FileService
from src.vectorstore.chroma_client import ChromaClient
from src.utils.retriever.retriever_factory import build_retriever
from src.utils.qa.retrieval_qa import build_retrieval_qa_chain
from src.utils.llm.openai_llm import get_openai_llm
from src.utils.llm.gemini_llm import get_gemini_llm


class CVService:
    """Service for CV shortlisting operations"""
    
    def __init__(self, chroma_client: ChromaClient):
        self.chroma_client = chroma_client
        self.collection_resumes = "resumes"
        self.file_service = FileService()
    
    async def shortlist_cvs(
        self,
        num_shortlisted: int,
        llm_provider: str,
        embedding_provider: str,
        jd_file: Optional[UploadFile],
        jd_text: Optional[str],
        cv_files: List[UploadFile]
    ) -> CVShortlistResponse:
        """Main business logic for CV shortlisting"""
        
        # Process JD
        final_jd_text = jd_text.strip() if jd_text else ""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Handle JD file if provided
            if jd_file:
                jd_file_path = self.file_service.save_uploaded_file(jd_file, temp_path / "jd")
                jd_contents = self.file_service.load_file_content(jd_file_path, "jd")
                final_jd_text = "\n\n".join(jd_contents).strip()
            
            if not final_jd_text:
                raise ValueError("No valid JD content found")
            
            # Process CVs
            cv_docs = []
            for cv_file in cv_files:
                cv_file_path = self.file_service.save_uploaded_file(cv_file, temp_path / "cvs")
                cv_contents = self.file_service.load_file_content(cv_file_path, "resume")
                
                for i, content in enumerate(cv_contents):
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
                raise ValueError("No valid CV content found")
            
            # Index CVs
            self.chroma_client.add_documents(self.collection_resumes, cv_docs)
            
            # Perform similarity search
            results = self.chroma_client.similarity_search(
                self.collection_resumes, final_jd_text, k=num_shortlisted
            )
            
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
                retriever = build_retriever(self.chroma_client, self.collection_resumes, k=3)
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
