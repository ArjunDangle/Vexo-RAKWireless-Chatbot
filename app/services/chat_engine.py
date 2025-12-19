import time
import chromadb
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.services.rewriter import QueryRewriter
from etl.embeddings import EmbeddingModel

class ChatEngine:
    def __init__(self):
        print("🚀 Initializing Chat Engine...")
        
        # 1. Connect to Vector DB
        self.client = chromadb.PersistentClient(path=settings.DB_PATH)
        self.collection = self.client.get_collection(name=settings.COLLECTION_NAME)
        
        # 2. Load Embedding Model
        self.embedder = EmbeddingModel.get_instance()
        
        # 3. Load Reranker
        print("⚖️ Loading Reranker (ms-marco-MiniLM-L-6-v2)...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # 4. Setup LLM
        self.llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            temperature=0.3
        )

        # 5. Initialize Rewriter
        self.rewriter = QueryRewriter(self.llm)

        # 6. Initialize BM25
        self._init_bm25()

    def _init_bm25(self):
        print("🧠 Building BM25 Keyword Index...")
        try:
            all_docs = self.collection.get() 
            self.doc_ids = all_docs['ids']
            self.documents = all_docs['documents']
            self.metadatas = all_docs['metadatas']
            
            tokenized_corpus = [doc.lower().split(" ") for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"✅ BM25 Index Ready ({len(self.documents)} chunks)")
        except Exception as e:
            print(f"⚠️ Failed to init BM25: {e}")
            self.bm25 = None

    def _hybrid_search(self, query: str, n_candidates: int, where_filter: dict = None) -> list:
        # 1. Vector Search
        query_vec = self.embedder.embed_query(query)
        vector_results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_candidates,
            where=where_filter
        )
        
        vec_hits = {}
        if vector_results['ids']:
            for i, id_ in enumerate(vector_results['ids'][0]):
                vec_hits[id_] = {
                    "doc": vector_results['documents'][0][i],
                    "meta": vector_results['metadatas'][0][i],
                    "rank": i + 1
                }

        # 2. BM25 Search
        bm25_hits = {}
        if self.bm25:
            tokenized_query = query.lower().split(" ")
            scores = self.bm25.get_scores(tokenized_query)
            top_n_indices = np.argsort(scores)[::-1][:n_candidates]
            
            rank = 1
            for idx in top_n_indices:
                doc_id = self.doc_ids[idx]
                if where_filter:
                    meta = self.metadatas[idx]
                    if where_filter.get("product_id") and meta.get("product_id") != where_filter["product_id"]:
                        continue
                bm25_hits[doc_id] = {
                    "doc": self.documents[idx],
                    "meta": self.metadatas[idx],
                    "rank": rank
                }
                rank += 1

        # 3. RRF Fusion
        combined_scores = {}
        k = 60
        all_ids = set(vec_hits.keys()) | set(bm25_hits.keys())
        for doc_id in all_ids:
            score = 0
            if doc_id in vec_hits: score += 1 / (k + vec_hits[doc_id]['rank'])
            if doc_id in bm25_hits: score += 1 / (k + bm25_hits[doc_id]['rank'])
            combined_scores[doc_id] = score

        # Sort and return pool
        sorted_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:n_candidates]
        final_pool = []
        for doc_id in sorted_ids:
            data = vec_hits.get(doc_id) or bm25_hits.get(doc_id)
            final_pool.append(data)
            
        return final_pool

    def _rerank(self, query: str, candidates: list, top_k: int = 8) -> list:
        if not candidates: return []
        pairs = [[query, item['doc']] for item in candidates]
        scores = self.reranker.predict(pairs)
        for i, score in enumerate(scores):
            candidates[i]['rerank_score'] = score
        
        # Sort by rerank score descending
        sorted_candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        return sorted_candidates[:top_k]

    def _log_results(self, title: str, results: list, score_key: str = None):
        """Helper to print clean logs to terminal"""
        print(f"\n{title}")
        print("="*80)
        print(f"{'#':<4} | {'Score':<8} | {'Title':<30} | {'Snippet (First 70 chars)'}")
        print("-" * 80)
        
        for i, res in enumerate(results):
            score = res.get(score_key, 0.0) if score_key else 0.0
            score_str = f"{score:.4f}" if score_key else "N/A"
            title = res['meta']['title'][:28] + ".." if len(res['meta']['title']) > 28 else res['meta']['title']
            snippet = res['doc'].replace("\n", " ")[:70] + "..."
            
            print(f"{i+1:<4} | {score_str:<8} | {title:<30} | {snippet}")
        print("="*80 + "\n")

    def _router(self, query: str) -> dict:
        prompt = ChatPromptTemplate.from_template(
            """
            Analyze the user's technical query about RAKwireless products.
            Extract the specific Product ID (e.g., "rak7240", "wisblock") if mentioned.
            Return JSON ONLY: {{"product_id": "string or null"}}
            Query: {query}
            """
        )
        chain = prompt | self.llm | JsonOutputParser()
        try:
            filters = chain.invoke({"query": query})
            if filters.get("product_id"):
                filters["product_id"] = filters["product_id"].lower()
            return filters
        except:
            return {}

    def get_response(self, query: str, history: list[dict] = []) -> dict:
        start_time = time.time()
        
        # 1. REWRITE
        refined_query = self.rewriter.rewrite(query, history)
        
        # 2. ROUTE
        filters = self._router(refined_query)
        where_clause = {"product_id": filters["product_id"]} if filters.get("product_id") else None
        
        print(f"🔍 Searching: '{refined_query}' | Filter: {where_clause}")

        # 3. HYBRID RETRIEVAL (Pool 25 candidates)
        candidates = self._hybrid_search(refined_query, n_candidates=25, where_filter=where_clause)
        
        if not candidates and where_clause:
            print("⚠️ No results with filter. Retrying broad search...")
            candidates = self._hybrid_search(refined_query, n_candidates=25, where_filter=None)

        # LOG STEP 1: All Candidates
        self._log_results("🌊 Step 1: Hybrid Retrieval Pool (Top 25)", candidates)

        # 4. RERANKING
        top_results = self._rerank(refined_query, candidates, top_k=8)

        # LOG STEP 2: Reranked Winners
        self._log_results("🏆 Step 2: Reranked Winners (Top 8 sent to LLM)", top_results, score_key='rerank_score')

        # 5. GENERATION
        context_text = ""
        sources = []
        
        for hit in top_results:
            meta = hit['meta']
            context_text += f"---\nSource: {meta['title']}\nContent: {hit['doc']}\n"
            
            if not any(s['title'] == meta['title'] for s in sources):
                sources.append({
                    "title": meta['title'],
                    "url": meta['source'],
                    "confidence": float(hit['rerank_score'])
                })

        if not context_text:
            return {
                "answer": "I checked the documentation but couldn't find an answer.",
                "sources": [],
                "latency": time.time() - start_time
            }

        system_prompt = """
        You are an expert RAKwireless technical support AI.
        Answer strictly based on the context.
        Context:
        {context}
        """
        
        chain = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{query}")]) | self.llm
        response_msg = chain.invoke({"context": context_text, "query": refined_query})
        
        return {
            "answer": response_msg.content,
            "sources": sources,
            "latency": time.time() - start_time
        }