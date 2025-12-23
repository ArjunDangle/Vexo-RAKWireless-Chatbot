import os
import sys
import json
import chromadb
from pathlib import Path

# --- PATH SETUP ---
current_script_path = os.path.abspath(__file__)
etl_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(etl_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from etl.embeddings import EmbeddingModel

# Configuration
INPUT_DIR = os.path.join(project_root, "storage", "mined_knowledge")
DB_DIR = os.path.join(project_root, "storage", "vector_db")
COLLECTION_NAME = "rak_knowledge"

def flatten_metadata(metadata: dict) -> dict:
    clean_meta = {}
    for k, v in metadata.items():
        if isinstance(v, list): clean_meta[k] = ", ".join(v)
        elif v is None: clean_meta[k] = ""
        else: clean_meta[k] = str(v)
    return clean_meta

def ingest_data():
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    embedder = EmbeddingModel.get_instance()

    # RGLOB: Recursively find all JSONL files in product folders
    files = list(Path(INPUT_DIR).rglob("*.jsonl"))
    
    if not files:
        print("❌ No mined files found!")
        return

    print(f"🚀 Ingesting {len(files)} product files...")

    for file_path in files:
        ids, documents, metadatas = [], [], []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text_to_embed = f"{data['title']} ({data['category']}):\n{data['content']}"
                    
                    ids.append(data['id']) # MD5 Hash from Miner
                    documents.append(text_to_embed)
                    metadatas.append(flatten_metadata({
                        "product_id": data.get('product_id', 'unknown'),
                        "product_family": data.get('product_family', 'unknown'),
                        "category": data['category'],
                        "title": data['title'],
                        "source": data.get('source_file', 'unknown'),
                        "parent_content": data.get('parent_content', '') # LINK TO BIG CHUNK
                    }))
                except: continue

        if documents:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            print(f"✅ Indexed {len(documents)} snippets from {file_path.parent.name}/{file_path.name}")

    print(f"\n🎉 Ingestion Complete!")

if __name__ == "__main__":
    ingest_data()