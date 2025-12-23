import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

class EmbeddingModel:
    _instance = None

    def __init__(self, model_name: str = 'multi-qa-mpnet-base-dot-v1'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"✅ Embedding Model Loaded: {model_name} on {self.device}")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # --- LANGCHAIN INTERFACE METHODS ---

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query string for retrieval."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of strings for indexing."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    # --- YOUR ORIGINAL METHOD (Keep for ingest.py) ---
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)