import time
import chromadb
import numpy as np
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.services.rewriter import QueryRewriter
from etl.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

class ChatEngine:
    def __init__(self):
        logger.info("🚀 Initializing Unified Chat Engine...")
        
        # 1. Connect to Vector DB and Embedder
        self.client = chromadb.PersistentClient(path=settings.DB_PATH)
        self.collection = self.client.get_collection(name=settings.COLLECTION_NAME)
        self.embedder = EmbeddingModel.get_instance()
        
        # 2. Load Models
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            temperature=0.1
        )

        # 3. Helpers
        self.rewriter = QueryRewriter(self.llm)
        self._init_bm25()

    def _init_bm25(self):
        """Builds a keyword index for hybrid search fallback."""
        try:
            all_docs = self.collection.get()
            self.doc_ids = all_docs['ids']
            self.documents = all_docs['documents']
            self.metadatas = all_docs['metadatas']
            
            tokenized_corpus = [doc.lower().split(" ") for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"✅ BM25 Ready: {len(self.documents)} chunks")
        except Exception as e:
            logger.error(f"⚠️ BM25 Init Failed: {e}")
            self.bm25 = None

    def _generate_hyde_text(self, query: str) -> str:
        """Generates a hypothetical technical answer to improve vector retrieval."""
        hyde_prompt = ChatPromptTemplate.from_template(
            "Write a concise, highly technical paragraph answering this RAKwireless question. "
            "Include potential AT commands, pin names, or technical specs.\nQuestion: {query}"
        )
        chain = hyde_prompt | self.llm
        return chain.invoke({"query": query}).content

    def _router(self, query: str) -> str:
        """Identifies specific Product IDs to apply metadata filters."""
        prompt = ChatPromptTemplate.from_template(
            "Analyze the query about RAKwireless. Extract the Product ID (e.g., 'rak7240'). "
            "Return JSON ONLY: {{\"product_id\": \"string or null\"}}\nQuery: {query}"
        )
        try:
            res = (prompt | self.llm | JsonOutputParser()).invoke({"query": query})
            return res.get("product_id").lower() if res.get("product_id") else None
        except: 
            return None

    def _hybrid_search(self, query: str, n_candidates: int, product_filter: str = None) -> list:
        # 1. Expand query via HyDE
        hyde_query = self._generate_hyde_text(query)
        query_vec = self.embedder.embed_query(hyde_query)
        
        where_clause = {"product_id": product_filter} if product_filter else None

        # 2. Semantic Search
        vector_results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_candidates,
            where=where_clause
        )
        
        vec_hits = {}
        if vector_results['ids']:
            for i, id_ in enumerate(vector_results['ids'][0]):
                vec_hits[id_] = {
                    "doc": vector_results['documents'][0][i],
                    "meta": vector_results['metadatas'][0][i],
                    "rank": i + 1
                }

        # 3. Keyword Search
        bm25_hits = {}
        if self.bm25:
            tokenized_query = query.lower().split(" ")
            scores = self.bm25.get_scores(tokenized_query)
            top_n_indices = np.argsort(scores)[::-1][:n_candidates]
            
            for rank, idx in enumerate(top_n_indices, 1):
                doc_id = self.doc_ids[idx]
                if product_filter and self.metadatas[idx].get("product_id") != product_filter:
                    continue
                bm25_hits[doc_id] = {
                    "doc": self.documents[idx],
                    "meta": self.metadatas[idx],
                    "rank": rank
                }

        # 4. RRF Fusion
        combined_scores = {}
        k = 60
        all_ids = set(vec_hits.keys()) | set(bm25_hits.keys())
        for doc_id in all_ids:
            score = 0
            if doc_id in vec_hits: score += 1 / (k + vec_hits[doc_id]['rank'])
            if doc_id in bm25_hits: score += 1 / (k + bm25_hits[doc_id]['rank'])
            combined_scores[doc_id] = score

        sorted_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:n_candidates]
        return [vec_hits.get(did) or bm25_hits.get(did) for did in sorted_ids]

    def _rerank(self, query: str, candidates: list, top_k: int = 6) -> list:
        if not candidates: return []
        pairs = [[query, item['doc']] for item in candidates]
        scores = self.reranker.predict(pairs)
        for i, score in enumerate(scores):
            candidates[i]['rerank_score'] = score
        
        return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:top_k]

    def get_response(self, query: str, history: list = []) -> dict:
        start_time = time.time()
        
        # Step 1: Query Processing
        refined_query = self.rewriter.rewrite(query, history)
        product_id = self._router(refined_query)
        
        # Step 2: Retrieval & Scoring
        candidates = self._hybrid_search(refined_query, n_candidates=20, product_filter=product_id)
        top_results = self._rerank(refined_query, candidates)

        # Step 3: Context Enrichment (Parent Swapping)
        context_blocks = []
        sources = []
        for hit in top_results:
            meta = hit['meta']
            content = meta.get('parent_content') or hit['doc']
            context_blocks.append(f"SOURCE: {meta['title']}\nCONTENT: {content}")
            
            if not any(s['title'] == meta['title'] for s in sources):
                sources.append({
                    "title": meta['title'], 
                    "url": meta['source'], 
                    "confidence": float(hit['rerank_score'])
                })

        # Step 4: Generation
        system_prompt = """You are a Senior RAKwireless Technical Support Engineer.
        STRICT RULES:
        1. CONTEXT ONLY: Only answer based on the provided technical documentation.
        2. TABLES: Use Markdown Tables for specifications/pin definitions.
        3. CODE: Use Markdown Code Blocks for AT commands, C++, or Python.
        4. If documentation is missing, state that you cannot provide a confirmed answer."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "TECHNICAL CONTEXT:\n{context}\n\nUSER QUESTION: {query}")
        ])
        
        answer = (prompt | self.llm).invoke({
            "query": refined_query, 
            "context": "\n---\n".join(context_blocks)
        })
        
        return {
            "answer": answer.content,
            "sources": sources,
            "latency": time.time() - start_time
        }