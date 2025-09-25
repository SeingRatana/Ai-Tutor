import chromadb
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class ChromaDB:
    """
    Wrapper class for ChromaDB interactions, handling collection creation,
    document addition, and similarity search queries.
    """
    def __init__(self, persist_directory: str, embedding_function):
        self.persist_directory = persist_directory
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name="itc_ams_documents",
                embedding_function=embedding_function
            )
            logger.info(f"ChromaDB initialized. Documents: {self.collection.count()}")
        except Exception as e:
            logger.critical(f"Failed to initialize ChromaDB: {e}", exc_info=True)
            raise

    def add_documents(self, documents: List[Dict]):
        if not documents:
            return
        
        texts = [doc['text'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        ids = [f"doc_{idx}" for idx in range(len(texts))]
        
        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")

    def query_documents(self, query_texts: List[str], n_results: int = 5) -> List[Dict]:
        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            return [{
                "text": doc_text,
                "metadata": doc_metadata,
                "distance": doc_distance
            } for doc_text, doc_metadata, doc_distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )]
        except Exception as e:
            logger.error(f"Query error: {e}")
            return []

    def count_documents(self) -> int:
        return self.collection.count()