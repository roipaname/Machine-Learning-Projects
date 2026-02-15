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
    def add_document(self,doc_id:str,content:str,metadata:Dict):
        """
        Add business strategy document to knowledge base
        """
        embedding=self.embedder.encode(content).tolist()

        self.collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata],
            embeddings=[embedding]
        )
        logger.success(f"Document {doc_id} added to RAG Document Store")
    def add_documents_bulk(self, documents: List[Dict]):
        """
        Bulk upload documents
        Each document: {'id': str, 'content': str, 'metadata': dict}
        """
        
        ids = [doc['id'] for doc in documents]
        contents = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        embeddings = self.embedder.encode(contents).tolist()
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
        
        logger.success(f"Added {len(documents)} documents")
    def retrieve(self,query:str,customer_context: Dict = None,top_k:int=5)->List[Dict]:
        """
        Retrieve relevant strategy documents
        """

        if customer_context:
            enhanced_query = f"""
            Customer segment: {customer_context.get('account_info', {}).get('tier')}
            Risk level: {customer_context.get('risk_tier')}
            Query: {query}
            """
        else:
            enhanced_query = query
        query_embedding=self.embedder.encode(enhanced_query).tolist()
        results=self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        retrieved_docs = []
        for i in range(len(results['ids'][0])):
            retrieved_docs.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': results['distances'][0][i] if 'distances' in results else None
            })
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents for query")
        return retrieved_docs


