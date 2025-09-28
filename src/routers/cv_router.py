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
        jd_file=jd_file,
        jd_text=jd_text,
        cv_files=cv_files
    )

#################  BOC - Begin of ChromaDB check #################
################# Temprory router for ChromaDB check #################
@cv_router.get("/db-check")
async def db_check():
    """Check if the ChromaDB is accessible and list collections with full data"""
    try:
        # Get all collections from ChromaDB
        collections = chroma_client.client.list_collections()
        collection_names = [collection.name for collection in collections]
        
        # Get complete data for each collection
        collection_info = []
        for collection in collections:
            # Get all documents from the collection (excluding embeddings to avoid numpy array issues)
            all_documents = collection.get(include=["metadatas", "documents"])
            
            # Format documents for better readability
            documents_data = []
            if all_documents.get("ids"):
                for i in range(len(all_documents["ids"])):
                    doc_data = {
                        "id": all_documents["ids"][i],
                        "document": all_documents["documents"][i] if all_documents.get("documents") and len(all_documents["documents"]) > i else None,
                        "metadata": all_documents["metadatas"][i] if all_documents.get("metadatas") and len(all_documents["metadatas"]) > i else None,
                        "embedding_available": collection.count() > 0  # Just indicate if embeddings exist
                    }
                    documents_data.append(doc_data)
            
            collection_info.append({
                "name": collection.name,
                "id": collection.id,
                "metadata": collection.metadata,
                "count": collection.count(),
                "documents": documents_data
            })
        
        return {
            "status": "success", 
            "collections": collection_names,
            "collection_details": collection_info,
            "total_collections": len(collections)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@cv_router.get("/db-purge")
async def db_purge():
    """Purge all the collection from ChromaDB"""
    try:

        # Get all collections from ChromaDB
        collections = chroma_client.client.list_collections()
        collection_names = [collection.name for collection in collections]

        for collection in collection_names:
            chroma_client.client.delete_collection(collection)

        return {"status": "success", "message": f"Collection  purged successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
