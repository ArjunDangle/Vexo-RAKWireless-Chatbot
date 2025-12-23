import os
import chromadb
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from etl.embeddings import EmbeddingModel
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
DB_DIR = "./storage/vector_db"
COLLECTION_NAME = "rak_knowledge"

class RAKChatEngine:
    def __init__(self):
        # Using a high-reasoning model for technical accuracy
        self.llm = ChatOpenAI(
            model="mistralai/mistral-large-2411", 
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1
        )
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        self.embedder = EmbeddingModel.get_instance()

    def generate_hyde_query(self, query: str) -> str:
        """PHASE 3: Generates a hypothetical technical answer to sharpen vector search."""
        hyde_prompt = ChatPromptTemplate.from_template(
            "Write a concise, highly technical paragraph answering this RAKwireless question. "
            "Include potential AT commands, pin names, or technical specs. \nQuestion: {query}"
        )
        chain = hyde_prompt | self.llm
        return chain.invoke({"query": query}).content

    def retrieve_context(self, query: str, top_k: int = 3):
        """PHASE 4: Retrieves child snippets but swaps them for full Parent context."""
        # Step 1: Query expansion
        hyde_text = self.generate_hyde_query(query)
        query_embedding = self.embedder.embed_query(hyde_text)

        # Step 2: Vector search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        expanded_contexts = []
        sources = set()

        for i in range(len(results['documents'][0])):
            meta = results['metadatas'][0][i]
            
            # SWAP: Use the full Parent section if it exists, otherwise the snippet
            content = meta.get('parent_content') or results['documents'][0][i]
            
            # PHASE 5: Format context with metadata breadcrumbs
            context_block = (
                f"### DOCUMENT: {meta.get('source', 'RAK Doc')}\n"
                f"### SECTION: {meta.get('title', 'General')}\n"
                f"{content}\n"
            )
            expanded_contexts.append(context_block)
            sources.add(meta.get('source', 'Reference Manual'))

        return "\n---\n".join(expanded_contexts), list(sources)

    def ask(self, query: str):
        """PHASE 5: Generates the final expert answer with strict formatting rules."""
        context, sources = self.retrieve_context(query)

        system_prompt = """
        You are a Senior RAKwireless Technical Support Engineer. 
        Your goal is to help developers integrate RAK hardware and software.

        STRICT RULES:
        1. CONTEXT ONLY: Only answer based on the provided technical documentation. 
        2. TABLES: Always use Markdown Tables for pin definitions, specs, or parameters.
        3. CODE: Use Markdown Code Blocks for AT commands, C++, or Python examples.
        4. ACCURACY: If the context mentions a specific product (e.g., WisGateOS 2), do not assume it applies to others.
        5. SOURCES: You must list the source files used at the very end.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"TECHNICAL CONTEXT:\n{context}\n\nUSER QUESTION: {query}")
        ])

        chain = prompt | self.llm
        response = chain.invoke({}) # Variables already formatted in the string
        
        # Append Citations
        citation_text = "\n\n**Sources:**\n- " + "\n- ".join(sources)
        return response.content + citation_text

if __name__ == "__main__":
    bot = RAKChatEngine()
    print("🤖 RAK Support Bot is ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['exit', 'quit']: break
        print(f"\nAssistant: {bot.ask(user_input)}")