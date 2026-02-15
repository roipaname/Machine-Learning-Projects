from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.settings import Settings
from chromadb.utils import embedding_functions
from loguru import logger
import os
from pathlib import Path
from typing import List,Dict 

class RAGDocumentStore:
    """
    Vector database for business strategy documents
    """
    def __init__(self,persist_directory='data/rag_db'):
        self.embedder=SentenceTransformer('all-MiniLM-L6-v2')
        self.client=chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            )
        )
        try:
            self.collection=self.client.get_collection(name='retention_strategies')
        except:
            self.collection=self.client.create_collection("retention_strategies")
        logger.success("RAG Document Store initialized")