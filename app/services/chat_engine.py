import time
import asyncio
import chromadb
import numpy as np
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.core.config import settings
from etl.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

class ChatEngine:
    def __init__(self):
        # 1. Setup DB and Models
        self.client = chromadb.PersistentClient(path=settings.DB_PATH)
        self.embedder = EmbeddingModel.get_instance()
        self.collection = self.client.get_collection(name=settings.COLLECTION_NAME)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            temperature=0.1,
            streaming=True 
        )

        self._init_bm25()

    def _init_bm25(self):
        try:
            all_docs = self.collection.get()
            self.doc_ids, self.documents, self.metadatas = all_docs['ids'], all_docs['documents'], all_docs['metadatas']
            tokenized_corpus = [doc.lower().split(" ") for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
        except Exception as e:
            logger.error(f"BM25 Init Failed: {e}")
            self.bm25 = None

    async def _analyze_query(self, query: str, history: list) -> dict:
        """Consolidates Rewriting, Routing, and HyDE into 1 LLM call to save ~60s."""
        analysis_prompt = ChatPromptTemplate.from_template("""
        You are a RAKwireless Technical Analyst. Analyze the query given the history.
        Return ONLY a JSON object with these keys:
        - standalone_query: self-contained version of the question.
        - product_id: specific RAK product ID (e.g. 'rak7240') or null.
        - hyde_text: 1-sentence technical hypothetical answer.

        History: {history}
        Query: {query}
        """)
        hist_str = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in history[-4:]])
        chain = analysis_prompt | self.llm | JsonOutputParser()
        return await chain.ainvoke({"query": query, "history": hist_str})

    def _hybrid_search(self, standalone_query: str, hyde_text: str, product_filter: str = None) -> list:
        query_vec = self.embedder.embed_query(hyde_text)
        where_clause = {"product_id": product_filter} if product_filter else None

        # Vector Search
        vector_results = self.collection.query(query_embeddings=[query_vec], n_results=15, where=where_clause)
        vec_hits = {id_: {"doc": vector_results['documents'][0][i], "meta": vector_results['metadatas'][0][i], "rank": i + 1} 
                    for i, id_ in enumerate(vector_results['ids'][0])} if vector_results['ids'] else {}

        # BM25 Search
        bm25_hits = {}
        if self.bm25:
            tokenized_query = standalone_query.lower().split(" ")
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:15]
            for rank, idx in enumerate(top_indices, 1):
                if not product_filter or self.metadatas[idx].get("product_id") == product_filter:
                    bm25_hits[self.doc_ids[idx]] = {"doc": self.documents[idx], "meta": self.metadatas[idx], "rank": rank}

        # RRF Fusion
        combined = {}
        for did in set(vec_hits.keys()) | set(bm25_hits.keys()):
            score = (1/(60 + vec_hits[did]['rank']) if did in vec_hits else 0) + (1/(60 + bm25_hits[did]['rank']) if did in bm25_hits else 0)
            combined[did] = score
        
        sorted_ids = sorted(combined, key=combined.get, reverse=True)[:15]
        return [vec_hits.get(did) or bm25_hits.get(did) for did in sorted_ids]

    def _rerank(self, query: str, candidates: list) -> list:
        if not candidates: return []
        scores = self.reranker.predict([[query, c['doc']] for c in candidates])
        for i, s in enumerate(scores): candidates[i]['rerank_score'] = s
        return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:5]

    async def get_response_stream(self, query: str, history: list = []):
        """Streaming response with sources and latency handled in the final yield."""
        start_time = time.time()
        analysis = await self._analyze_query(query, history)
        candidates = self._hybrid_search(analysis['standalone_query'], analysis['hyde_text'], analysis['product_id'])
        top_results = self._rerank(analysis['standalone_query'], candidates)

        context = "\n---\n".join([f"SOURCE: {h['meta']['title']}\n{h['meta'].get('parent_content', h['doc'])}" for h in top_results])
        sources = [{"title": h['meta']['title'], "url": h['meta']['source'], "confidence": float(h['rerank_score'])} for h in top_results]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Senior RAKwireless Engineer. Use context only. Answer technical details precisely."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "CONTEXT:\n{context}\n\nUSER QUESTION: {query}")
        ])

        formatted_history = [("human" if m['role']=="user" else "ai", m['content']) for m in history[-6:]]
        chain = prompt | self.llm

        full_answer = ""
        async for chunk in chain.astream({"query": analysis['standalone_query'], "context": context, "chat_history": formatted_history}):
            if chunk.content:
                full_answer += chunk.content
                yield json.dumps({"type": "token", "content": chunk.content})

        yield json.dumps({
            "type": "final",
            "sources": sources,
            "latency": time.time() - start_time
        })

    async def get_response(self, query: str, history: list = []) -> dict:
        """Standard static response used for terminal testing."""
        start_time = time.time()
        analysis = await self._analyze_query(query, history)
        candidates = self._hybrid_search(analysis['standalone_query'], analysis['hyde_text'], analysis['product_id'])
        top_results = self._rerank(analysis['standalone_query'], candidates)

        context = "\n---\n".join([f"SOURCE: {h['meta']['title']}\n{h['meta'].get('parent_content', h['doc'])}" for h in top_results])
        sources = [{"title": h['meta']['title'], "url": h['meta']['source'], "confidence": float(h['rerank_score'])} for h in top_results]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a RAKwireless Support Engineer. Use context only."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "CONTEXT:\n{context}\n\nQUESTION: {query}")
        ])

        formatted_history = [("human" if m.get('role')=="user" else "ai", m.get('content')) for m in history[-6:]]
        answer = await (prompt | self.llm).ainvoke({"query": analysis['standalone_query'], "context": context, "chat_history": formatted_history})
        
        return {"answer": answer.content, "sources": sources, "latency": time.time() - start_time}