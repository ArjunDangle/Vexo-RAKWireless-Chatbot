import time
import chromadb
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.core.config import settings

# Import our Singleton Embedder
# NOTE: Ensure you run uvicorn from the project root for this import to work
from etl.embeddings import EmbeddingModel

class ChatEngine:
    def __init__(self):
        # 1. Connect to Vector DB
        self.client = chromadb.PersistentClient(path=settings.DB_PATH)
        self.collection = self.client.get_collection(name=settings.COLLECTION_NAME)
        
        # 2. Load Embedding Model (Singleton)
        self.embedder = EmbeddingModel.get_instance()
        
        # 3. Setup LLM (Router & Generator)
        self.llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            temperature=0.3
        )

    def _router(self, query: str) -> dict:
        """
        Analyzes the query to extract filters (Product ID, Family).
        """
        prompt = ChatPromptTemplate.from_template(
            """
            Analyze the user's technical query about RAKwireless products.
            Extract the specific Product ID (e.g., "rak7240", "wisblock") if mentioned.
            If no specific product is mentioned, return null.
            
            Return JSON ONLY:
            {{
                "product_id": "string or null",
                "category": "string or null (e.g., 'Troubleshooting', 'How-To')"
            }}
            
            Query: {query}
            """
        )
        chain = prompt | self.llm | JsonOutputParser()
        try:
            filters = chain.invoke({"query": query})
            # Sanitize: Ensure keys exist and are lowercase
            if filters.get("product_id"):
                filters["product_id"] = filters["product_id"].lower()
            return filters
        except:
            return {}

    def get_response(self, query: str) -> dict:
        start_time = time.time()
        
        # 1. ROUTING (The Fix for your issue)
        filters = self._router(query)
        print(f"🧠 Router Logic: {filters}")
        
        # Construct DB Filters
        where_clause = {}
        if filters.get("product_id"):
            where_clause["product_id"] = filters["product_id"]
        
        # 2. RETRIEVAL
        query_vec = self.embedder.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=5,
            where=where_clause if where_clause else None # Apply filter if found
        )
        
        context_text = ""
        sources = []
        
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                context_text += f"---\nSource: {meta['title']}\nContent: {doc}\n"
                
                # Deduplicate sources for the UI
                if not any(s['title'] == meta['title'] for s in sources):
                    sources.append({
                        "title": meta['title'],
                        "url": meta['source'], # This maps to the file path for now
                        "confidence": 0.0 # Placeholder
                    })

        # 3. GENERATION
        if not context_text:
            return {
                "answer": "I couldn't find any specific documents matching your query in the database.",
                "sources": [],
                "latency": time.time() - start_time
            }

        system_prompt = """
        You are an expert RAKwireless technical support AI. 
        Answer the user's question strictly based on the context provided below.
        If the context doesn't contain the answer, say "I don't have that information."
        
        Context:
        {context}
        """
        
        gen_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        
        chain = gen_prompt | self.llm
        response_msg = chain.invoke({"context": context_text, "query": query})
        
        return {
            "answer": response_msg.content,
            "sources": sources,
            "latency": time.time() - start_time
        }