import os
import sys
import json
import chromadb
from pathlib import Path

# --- PATH SETUP ---
# Ensures we can find the 'etl' package regardless of where the script is called
current_script_path = os.path.abspath(__file__)
etl_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(etl_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from etl.embeddings import EmbeddingModel

# Configuration using absolute paths to prevent "0 collections" errors
INPUT_DIR = os.path.join(project_root, "storage", "mined_knowledge")
DB_DIR = os.path.join(project_root, "storage", "vector_db")
COLLECTION_NAME = "rak_knowledge"

def flatten_metadata(metadata: dict) -> dict:
    """Ensures all metadata values are ChromaDB compatible strings."""
    clean_meta = {}
    for k, v in metadata.items():
        if isinstance(v, list): 
            clean_meta[k] = ", ".join(map(str, v))
        elif v is None: 
            clean_meta[k] = ""
        else: 
            clean_meta[k] = str(v)
    return clean_meta

def ingest_data():
    print(f"📡 Connecting to Vector DB at: {DB_DIR}")
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # 1. Clear old data to reset dimensions
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"🗑️ Deleted old collection to reset dimensions.")
    except:
        pass 
    
    # 2. Initialize our custom embedder
    embedder = EmbeddingModel.get_instance()
    
    # 3. Create collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # RGLOB: Recursive search for product folders
    files = list(Path(INPUT_DIR).rglob("*.jsonl"))
    
    if not files:
        print(f"❌ No mined files found in {INPUT_DIR}!")
        return

    print(f"🚀 Ingesting {len(files)} product folders...")
    total_chunks = 0

    for file_path in files:
        ids, documents, metadatas = [], [], []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    doc_id = data.get('id')
                    if not doc_id:
                        continue

                    # The text block for vector search
                    text_to_embed = f"{data['title']} ({data['category']}):\n{data['content']}"
                    
                    ids.append(str(doc_id))
                    documents.append(text_to_embed)
                    
                    # Metadata for Context Swapping and Filtering
                    meta = {
                        "product_id": data.get('product_id', 'unknown'),
                        "product_family": data.get('product_family', 'unknown'),
                        "category": data['category'],
                        "title": data['title'],
                        "source": data.get('source_file', 'unknown'),
                        "parent_content": data.get('parent_content') or data.get('content', '')
                    }
                    metadatas.append(flatten_metadata(meta))
                except Exception as e:
                    print(f"⚠️ Error parsing line in {file_path.name}: {e}")
                    continue

        if documents:
            # --- THE FIX ---
            # Explicitly generate embeddings using your multi-qa-mpnet-base-dot-v1 model (768-dim)
            # If we don't pass 'embeddings', Chroma defaults to a 384-dim model internally.
            print(f"🧠 Generating embeddings for {len(documents)} chunks...")
            embeddings = embedder.embed_documents(documents)
            
            # Upsert both the raw text AND the pre-computed 768-dim vectors
            collection.upsert(
                ids=ids, 
                embeddings=embeddings, 
                documents=documents, 
                metadatas=metadatas
            )
            # ----------------
            
            total_chunks += len(documents)
            print(f"✅ Indexed {len(documents)} snippets from {file_path.parent.name}/{file_path.name}")

    print(f"\n🎉 Ingestion Complete! Total chunks in session: {total_chunks}")

if __name__ == "__main__":
    ingest_data()