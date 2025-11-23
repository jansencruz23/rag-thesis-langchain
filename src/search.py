import os
import time
from typing import Any, Dict
from dotenv import load_dotenv
from src.vector_store import FaissVectorStore
from langchain_groq import ChatGroq
from src.data_loader import load_all_documents

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str="faiss_store", embedding_model: str="all-MiniLM-L6-v2", llm_model: str="llama-3.1-8b-instant"):
        self.vector_store = FaissVectorStore(persist_dir, embedding_model=embedding_model)
        self.history = []

        # Load or build vector store
        faiss_path = os.path.join(persist_dir, "faiss_index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            docs = load_all_documents("data")
            self.vector_store.build_from_documents(docs)
        else:
            self.vector_store.load()
        self.llm = ChatGroq(model=llm_model, temperature=0.1, max_tokens=5000)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vector_store.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content

    def answer(self, query: str, top_k: int = 5, score_threshold: float=0.2, stream: bool=False, summarize: bool=False) -> Dict[str, Any]:
        results = self.vector_store.query(query, top_k=top_k, score_threshold=score_threshold)

        if not results:
            answer = "No relevant documents found."
            sources = []
            context = ""
        else:
            context = '\n\n'.join([r["metadata"].get("text", "") for r in results if r["metadata"]])
            sources = [{
                'source': r["metadata"].get('source_file', r["metadata"].get('source', 'unknown')),
                'page': r["metadata"].get('page', 'unknown'),
                'score': r.get('score', 0), 
                'preview': r["metadata"].get("text", "")[:120] + '...'
            } for r in results]

            # Streaming answer simulation
            prompt = f"""Use the following context to answer the question concisely.
                Context:
                {context}
        
                Question: {query}
        
                Answer:"""
            
            if stream:
                #print('Streaming answer: ')
                for i in range(0, len(prompt), 80):
                    #print(prompt[i:i+80], end='', flush=True)
                    time.sleep(0.05)
                #print()
            response = self.llm.invoke([prompt])
            answer = response.content

        # Add citations to answer
        citations = [f'[{i+1}] {src["source"]} (page {src["page"]})' for i, src in enumerate(sources)]
        answer_with_citations = answer + '\n\nCitations:\n' + '\n'.join(citations) if citations else answer

        # Optionally summarize answer
        summary = None
        if summarize and answer:
            summary_prompt = f'Summarize the following answer in 2 sentences:\n{answer}'
            summary_resp = self.llm.invoke([summary_prompt])
            summary = summary_resp.content

        # Store query history
        self.history.append({
            'query': query,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary
        })

        return {
            'query': query,
            'answer': answer,
            #'sources': sources,
            #'summary': summary,
            #'history': self.history
        }