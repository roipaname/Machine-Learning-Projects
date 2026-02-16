from src.utils.file_readers import read_file_to_text
from src.ai_advisor.rag.document_store import RAGDocumentStore
from loguru import logger
import os
from pathlib import Path
from config.settings import RAG_DIR,BUSINESS_STRATEGY_DOCS_DIR
import uuid

def insert_documents_to_vec_db(document_dir:Path=BUSINESS_STRATEGY_DOCS_DIR):
    """
    Read business strategy documents and store in vectore db"""
    rag_db=RAGDocumentStore()
    for file in document_dir.glob("*.*"):
        content=read_file_to_text(file)
        metadata={"filename":file.name}
        doc_id=str(uuid.uuid4())
        rag_db.add_document(doc_id=doc_id,content=content,metadata=metadata)
    logger.success("All documents inserted to RAG Document Store")


if __name__=="__main__":
    insert_documents_to_vec_db()
