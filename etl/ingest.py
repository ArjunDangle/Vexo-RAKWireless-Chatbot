import os
import sys
import json
import chromadb
from pathlib import Path

# --- DIAGNOSTIC PATH SETUP ---
# 1. Get the absolute path of the 'etl' folder
current_script_path = os.path.abspath(__file__)
etl_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(etl_dir)

# 2. Add PROJECT_ROOT to Python Path (allows 'from etl import ...')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 3. Add ETL_DIR to Python Path (allows 'import embeddings')
if etl_dir not in sys.path:
    sys.path.insert(0, etl_dir)

print(f"🔍 Diagnostic: Looking for modules in: {project_root}")

# 4. Try Import with Verbose Error Handling
try:
    # Check if file exists first
    if not os.path.exists(os.path.join(etl_dir, "embeddings.py")):
        raise FileNotFoundError("The file 'embeddings.py' does not exist in the 'etl' folder!")

    from etl.embeddings import EmbeddingModel
    print("✅ Import Successful: 'from etl.embeddings import EmbeddingModel'")

except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}")
    print("---------------------------------------------------")
    if "sentence_transformers" in str(e):
        print("💡 SOLUTION: You are missing the AI library.")
        print("   Run this command: pip install sentence-transformers")
    elif "No module named 'etl'" in str(e):
        print("💡 SOLUTION: Python cannot find the 'etl' folder as a package.")
        try:
            # Fallback attempt
            from embeddings import EmbeddingModel
            print("✅ Fallback Successful: 'from embeddings import EmbeddingModel'")
        except ImportError as e2:
            print(f"   Fallback failed: {e2}")
    else:
        print(f"💡 DETAIL: {e}")
    print("---------------------------------------------------\n")
    sys.exit(1) # Stop script if import fails
# -----------------------------

# Configuration
INPUT_DIR = os.path.join(project_root, "storage", "mined_knowledge")
DB_DIR = os.path.join(project_root, "storage", "vector_db")
COLLECTION_NAME = "rak_knowledge"

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

    total_chunks = 0
    files = list(Path(INPUT_DIR).glob("*.jsonl"))
    
    if not files:
        print(f"❌ No .jsonl files found in {INPUT_DIR}!")
        return

    print(f"🚀 Ingesting {len(files)} knowledge files...")

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