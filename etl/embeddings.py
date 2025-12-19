from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    _instance = None
    _model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("🧠 Loading Embedding Model (BAAI/bge-base-en-v1.5)...")
            cls._instance = cls()
            # This model is ~400MB and much smarter than MiniLM
            cls._model = SentenceTransformer('BAAI/bge-base-en-v1.5')
            print("✅ Model Loaded.")
        return cls._instance

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds documents (knowledge chunks).
        BGE does NOT need an instruction prefix for documents, only for queries.
        """
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Embeds the user query.
        CRITICAL: BGE models require this specific instruction for retrieval queries.
        """
        instruction = "Represent this sentence for searching relevant passages: "
        return self._model.encode([instruction + text], convert_to_numpy=True)[0].tolist()