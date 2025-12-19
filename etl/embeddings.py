from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    _instance = None
    _model = None

    @classmethod
    def get_instance(cls):
        """
        Singleton accessor. Returns the existing instance or creates a new one.
        """
        if cls._instance is None:
            print("🧠 Loading Embedding Model (all-MiniLM-L6-v2)...")
            cls._instance = cls()
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Model Loaded.")
        return cls._instance

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], convert_to_numpy=True)[0].tolist()