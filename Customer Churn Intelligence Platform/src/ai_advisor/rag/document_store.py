from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
from loguru import logger
from config.settings import RAG_DIR
from typing import List,Dict,Any

class RAGDocumentStore:
    def __init__(self,persist_dir=str(RAG_DIR),embedding_model_name:str="all-MiniLM-L6-v2"):
        self.persist_dir=persist_dir
        self.embedding_model=embedding_model_name
        self.embedder=SentenceTransformer(embedding_model_name)
        self.client=chromadb.PersistentClient(path=self.persist_dir)

        try:
            self.collection=self.client.get_collection(name="retention_strategies")
        except:
            self.collection=self.client.create_collection(name="retention_strategies",embedding_function=self.embedder.encode)
        logger.success("RAG Document Store initialized")

    def add_document(self,doc_id:str,content:str,metedata:Dict[str,Any]):
        """
        Add business strategy document to knowledge base
        """
        embedding=self.embedder.encode(content).tolist()
        self.collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[metedata],
            embeddings=[embedding]
        )
        logger.success(f"Document {doc_id} added to RAG Document Store")
    def add_documents(self,docs:List[Dict[str,Any]]):
        """
        Add multiple business strategy documents to knowledge base
        """
        doc_ids=[doc['doc_id'] for doc in docs]
        contents=[doc['content'] for doc in docs]
        metadatas=[doc['metadata'] for doc in docs]
        embeddings=self.embedder.encode(contents).tolist()
        self.collection.add(
            ids=doc_ids,
            documents=contents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        logger.success(f" {len(doc_ids)} documents added to RAG Document Store")

    def retrieve(self,query:str,top_k:int=5,customer_context:Dict=None):
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
            enhanced_query=query
        embedding=self.embedder.encode(enhanced_query).tolist()
        results=self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=['documents','metadatas','embeddings','distances']
        )
        retrieved_docs=[]
        for i in range(len(results['ids'][0])):
            retrieved_docs.append({
                "doc_id":results['ids'][0][i],
                "content":results['documents'][0][i],
                "metadata":results['metadatas'][0][i],
                "embedding":results['embeddings'][0][i],
                "score": 1 / (1 + results['distances'][0][i])  if 'distances' in results else None
            })
        logger.info(f"Retrieved {len(retrieved_docs)} documents for query")
        return retrieved_docs

if __name__ == "__main__":
    store = RAGDocumentStore()
    
    sample_docs = [
        {
            'id': 'strategy_001',
            'content': """
            Q4 2024 Retention Analysis - Enterprise Tech Segment
            
            For enterprise customers (MRR >$5k) showing engagement decline >40%:
            - Executive Business Review (EBR) within 5 days: 73% save rate
            - Combine with product training: increases to 81%
            - Strategic discount (15-20%): 68% save rate standalone
            - Best approach: EBR + training + conditional discount
            
            Average intervention cost: $2,400
            Average revenue saved: $60,000 annual
            """,
            'metadata': {
                'type': 'analyst_report',
                'segment': 'Enterprise',
                'date': '2024-Q4',
                'success_rate': 0.73
            }
        },
        {
            'id': 'playbook_002',
            'content': """
            Customer Success Playbook v3.2 - High-Touch Intervention
            
            For at-risk customers with high support volume:
            1. Acknowledge pain points in personalized outreach
            2. Assign dedicated CSM for 90 days
            3. Weekly check-ins during stabilization period
            4. Product optimization session within 2 weeks
            5. Executive sponsor call if no improvement in 30 days
            
            Success rate: 65% for customers with >10 tickets/month
            Timeline: 90-day intervention period
            """,
            'metadata': {
                'type': 'playbook',
                'focus': 'support_intensive',
                'success_rate': 0.65
            }
        }
    ]
    
    store.add_documents(sample_docs)
    
    # Test retrieval
    results = store.retrieve("enterprise customer with low engagement")
    for doc in results:
        print(f"\n{doc['id']}: {doc['content'][:200]}...")