import logging
from typing import List
from sentence_transformers import SentenceTransformer
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)

class ChromaCompatibleEmbedder(EmbeddingFunction):
    def __init__(self, model):
        self.model = model
        
    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Embedding model '{model_name}' loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load SentenceTransformer model: {e}", exc_info=True)
            raise

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def get_embedding_function(self) -> ChromaCompatibleEmbedder:
        return ChromaCompatibleEmbedder(self.model)