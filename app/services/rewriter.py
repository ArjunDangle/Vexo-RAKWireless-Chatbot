from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

class QueryRewriter:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        # This prompt instructs the LLM to de-contextualize the question
        self.prompt = ChatPromptTemplate.from_template(
            """
            Given a chat history and the latest user question which might reference context in the chat history, 
            formulate a standalone question which can be understood without the chat history. 
            
            Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
            
            Chat History:
            {history}
            
            Latest Question: {question}
            
            Standalone Question:
            """
        )
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    def rewrite(self, query: str, history: list[dict]) -> str:
        """
        If history exists, rewrites the query to be standalone.
        If history is empty, returns the original query to save latency.
        """
        if not history:
            return query
            
        # Format history for the prompt (assuming standard role/content dicts)
        history_str = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in history[-4:]]) # Limit to last 4 turns
        
        try:
            print(f"🔄 Rewriting query with history...")
            refined_query = self.chain.invoke({
                "history": history_str,
                "question": query
            })
            print(f"✅ Rewritten: '{query}' -> '{refined_query}'")
            return refined_query.strip()
        except Exception as e:
            print(f"⚠️ Rewriter failed, using original query. Error: {e}")
            return query