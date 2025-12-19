import os
import sys
import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --- 1. The Embedding Class (Merged Here) ---
class EmbeddingModel:
    _instance = None
    _model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("🧠 Loading Embedding Model (all-MiniLM-L6-v2)...")
            cls._instance = cls()
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Model Loaded.")
        return cls._instance

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

# --- 2. Configuration ---
# We calculate paths based on the script location
current_dir = os.path.dirname(os.path.abspath(__file__))

# If you put this file in root, use current_dir. 
# If in etl/, use parent of current_dir.
if current_dir.endswith("etl"):
    PROJECT_ROOT = os.path.dirname(current_dir)
else:
    PROJECT_ROOT = current_dir

INPUT_DIR = os.path.join(PROJECT_ROOT, "storage", "mined_knowledge")
DB_DIR = os.path.join(PROJECT_ROOT, "storage", "vector_db")
COLLECTION_NAME = "rak_knowledge"

# --- 3. Ingestion Logic ---
def flatten_metadata(metadata: dict) -> dict:
    clean_meta = {}
    for k, v in metadata.items():
        if isinstance(v, list):
            clean_meta[k] = ", ".join(v)
        elif v is None:
            clean_meta[k] = ""
        else:
            clean_meta[k] = str(v)
    return clean_meta

def ingest_data():
    print(f"📂 Opening Vector DB at {DB_DIR}...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    print("🧠 Initializing Embedding Model...")
    embedder = EmbeddingModel.get_instance()

    files = list(Path(INPUT_DIR).glob("*.jsonl"))
    
    if not files:
        print(f"❌ No .jsonl files found in {INPUT_DIR}!")
        return

    print(f"🚀 Ingesting {len(files)} knowledge files...")
    total_chunks = 0

    for file_path in files:
        ids = []
        documents = []
        metadatas = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text_to_embed = f"{data['title']} ({data['category']}):\n{data['content']}"
                    
                    ids.append(data['id'])
                    documents.append(text_to_embed)
                    
                    meta = {
                        "product_id": data.get('product_id', 'unknown'),
                        "product_family": data.get('product_family', 'unknown'),
                        "category": data['category'],
                        "title": data['title'],
                        "source": data.get('source_file', 'unknown')
                    }
                    metadatas.append(flatten_metadata(meta))
                    
                except json.JSONDecodeError:
                    continue

        if not documents:
            continue

        embeddings = embedder.embed_documents(documents)
        
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        total_chunks += len(documents)
        print(f"✅ Indexed {len(documents)} chunks from {file_path.name}")

    print(f"\n🎉 Ingestion Complete! Total Knowledge Chunks: {total_chunks}")

if __name__ == "__main__":
    ingest_data()