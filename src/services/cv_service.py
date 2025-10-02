import tempfile
import uuid
from datetime import datetime
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
from src.utils.metadata_extractor import extract_parameters_with_llm, candidate_key_params


class CVService:
    """Service for CV shortlisting operations"""

    def __init__(self, chroma_client: ChromaClient):
        self.chroma_client = chroma_client
        self.collection_resumes_base = "resumes"
        self.collection_job_descriptions_base = "job_descriptions"
        self.file_service = FileService()

    async def _store_job_description(self, jd_text: str) -> str:
        """
        Store job description in ChromaDB.

        Args:
            jd_text: The job description text to store

        Returns:
            str: The document ID of the stored job description

        Raises:
            ValueError: If JD text is empty
            RuntimeError: If there's an error storing the document
        """
        if not jd_text or not jd_text.strip():
            raise ValueError("Job description text cannot be empty")

        try:
            # Generate unique ID for this job description
            jd_id = f"jd_{uuid.uuid4().hex[:12]}_{int(datetime.now().timestamp())}"

            llm_metadata_response = extract_parameters_with_llm(jd_text.strip(), candidate_key_params)

            jd_doc = []
            # Create document with metadata
            jd_document = Document(
                page_content=jd_text.strip(),
                metadata={
                    "id": jd_id,
                    "doc_type": "job_description",
                    "created_at": datetime.now().isoformat(),
                    "source": "api_upload",
                    **llm_metadata_response
                }
            )

            jd_doc.append(jd_document)

            # Store in ChromaDB with explicit embeddings using provider-specific collection
            collection_name = self.collection_job_descriptions_base

            await self.chroma_client.add_documents_with_embeddings(
                collection_name, jd_doc
            )

            print(
                f"Successfully stored job description with ID: {jd_id} using embeddings")
            return jd_id

        except Exception as e:
            error_msg = f"Failed to store job description: {str(e)}"
            print(f"Error: {error_msg}")
            raise RuntimeError(error_msg) from e

    async def shortlist_cvs(
        self,
        num_shortlisted: int,
        llm_provider: str,
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
                jd_file_path = self.file_service.save_uploaded_file(
                    jd_file, temp_path / "jd")
                jd_contents = self.file_service.load_file_content(
                    jd_file_path, "jd")
                final_jd_text = "\n\n".join(jd_contents).strip()

            if not final_jd_text:
                raise ValueError("No valid JD content found")

            # Store job description in ChromaDB with embeddings
            stored_jd_id = None
            try:
                # Validate embedding provider availability
                try:
                    stored_jd_id = await self._store_job_description(
                        final_jd_text)
                    print(
                        f"Job description stored with ID: {stored_jd_id}")
                except RuntimeError as e:
                    if "API_KEY" in str(e):
                        print(
                            f"Warning: API key not configured. Raising error")
                        raise e
                    else:
                        raise e
            except Exception as e:
                print(f"Warning: Could not store job description: {e}")

            # Process CVs
            cv_docs = []
            for cv_file in cv_files:
                cv_file_path = self.file_service.save_uploaded_file(
                    cv_file, temp_path / "cvs")
                cv_contents = self.file_service.load_file_content(
                    cv_file_path, "resume")

                for i, content in enumerate(cv_contents):
                    # Generate unique ID for each CV document
                    cv_id = f"cv_{uuid.uuid4().hex[:12]}_{int(datetime.now().timestamp())}_{i}"
                    llm_metadata_response = extract_parameters_with_llm(content.strip(), candidate_key_params)

                    doc = Document(
                        page_content=content,
                        metadata={
                            "id": cv_id,
                            "source": cv_file.filename,
                            "doc_type": "resume",
                            "page": i + 1,
                            "created_at": datetime.now().isoformat(),
                            **llm_metadata_response
                        }
                    )
                    cv_docs.append(doc)

            if not cv_docs:
                raise ValueError("No valid CV content found")

            # Index CVs with custom embeddings using provider-specific collection
            try:
                collection_name = self.collection_resumes_base
                await self.chroma_client.add_documents_with_embeddings(
                    collection_name, cv_docs
                )
                print(
                    f"Successfully stored {len(cv_docs)} CV documents with embeddings in collection '{collection_name}'")
            except RuntimeError as e:
                if "API_KEY" in str(e):
                    print(f"Warning: API key not configured. Raising error")
                    raise e
                else:
                    raise e
            except Exception as e:
                print(f"Error storing CVs with custom embeddings: {e}")
                raise e

            # Perform similarity search using the same collection as CVs
            try:
                collection_name = self.collection_resumes_base
                results = self.chroma_client.similarity_search(
                    collection_name, final_jd_text, k=num_shortlisted
                )
            except Exception as e:
                print(
                    f"Error with similarity search using collection: {e}")
                raise e

            if not results:
                return CVShortlistResponse(
                    success=False,
                    message="No matching CVs found",
                    shortlisted_candidates=[],
                    total_candidates_processed=len(cv_docs),
                    job_description_id=stored_jd_id
                )

            # Format results
            candidates = []
            for idx, (doc, score) in enumerate(results):
                candidate = Candidate(
                    candidate_id=f"candidate_{idx + 1}",
                    score=round(score, 3),
                    metadata=doc.metadata,
                    content_preview=doc.page_content[:800] +
                    ("..." if len(doc.page_content) > 800 else "")
                )
                candidates.append(candidate)

            # Generate summary using LLM (optional)
            jd_summary = None
            try:
                collection_name = self.collection_resumes_base
                retriever = build_retriever(
                    self.chroma_client, collection_name, k=3)
                llm = get_openai_llm() if llm_provider == "openai" else get_gemini_llm()
                chain = build_retrieval_qa_chain(retriever, llm)
                response = chain.invoke(final_jd_text)
                jd_summary = response.content if hasattr(
                    response, "content") else str(response)
            except Exception as e:
                # Log error but don't fail the request
                print(f"Warning: Could not generate summary: {e}")

            return CVShortlistResponse(
                success=True,
                message=f"Successfully shortlisted {len(candidates)} candidates",
                shortlisted_candidates=candidates,
                total_candidates_processed=len(cv_docs),
                jd_summary=jd_summary,
                job_description_id=stored_jd_id
            )
