from pathlib import Path
from typing import List
from fastapi import UploadFile

from src.loaders.pdf_loader import load_pdf
from src.loaders.docx_loader import load_docx
from src.loaders.txt_loader import load_txt


class FileService:
    """Service for handling file operations"""
    
    @staticmethod
    def load_file_content(file_path: str, doc_type: str) -> List[str]:
        """Load file content and return list of page contents"""
        if file_path.lower().endswith(".pdf"):
            docs = load_pdf(file_path, doc_type)
        elif file_path.lower().endswith(".docx"):
            docs = load_docx(file_path, doc_type)
        else:
            docs = load_txt(file_path, doc_type)
        
        return [doc.page_content for doc in docs]

    @staticmethod
    def save_uploaded_file(upload_file: UploadFile, temp_dir: Path) -> str:
        """Save uploaded file to temporary directory"""
        temp_dir.mkdir(parents=True, exist_ok=True)
        file_path = temp_dir / upload_file.filename
        
        with open(file_path, "wb") as f:
            content = upload_file.file.read()
            f.write(content)
        
        return str(file_path)
