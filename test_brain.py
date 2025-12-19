import chromadb
from sentence_transformers import SentenceTransformer

# 1. Setup
db_path = "./storage/vector_db"
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="rak_knowledge")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. The Test Question
query = "How do I update the firmware on RAK7240?"
print(f"❓ Question: {query}\n")

# 3. Embed & Search
query_vector = model.encode([query], convert_to_numpy=True)[0].tolist()

results = collection.query(
    query_embeddings=[query_vector],
    n_results=3  # Get top 3 matches
)

# 4. Print Results
for i, doc in enumerate(results['documents'][0]):
    meta = results['metadatas'][0][i]
    print(f"--- Result {i+1} (Source: {meta['source']}) ---")
    print(doc[:300] + "...") # Print first 300 chars
    print("\n")