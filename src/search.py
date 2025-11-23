import os
import time
from typing import Any, Dict, Generator, List
from dotenv import load_dotenv
from src.vector_store import FaissVectorStore
from langchain_ollama import ChatOllama
from src.data_loader import load_all_documents

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str="faiss_store", embedding_model: str="nomic-embed-text", 
                 llm_model: str="gemma3:4b", chunk_size: int=1000, chunk_overlap: int=200):
        self.vector_store = FaissVectorStore(
            persist_dir=persist_dir, 
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.history = []
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Load or build vector store
        faiss_path = os.path.join(persist_dir, "faiss_index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            docs = load_all_documents("data")
            self.vector_store.build_from_documents(docs)
        else:
            self.vector_store.load()

        self.llm = ChatOllama(
            model=llm_model,
            temperature=0.1, 
            max_tokens=5000, 
            num_gpu=1, 
            num_thread=4
        )

    def search_and_summarize(self, query: str, top_k: int=5) -> str:
        results = self.vector_store.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content

    def answer(self, query: str, top_k: int=5, score_threshold: float=0.2, 
               stream: bool=False, summarize: bool=False) -> Dict[str, Any]:
        results = self.vector_store.query(query, top_k=top_k, score_threshold=score_threshold)
        if not results:
            answer = "No relevant documents found."
            sources = []
            context = ""
            chunks = []
        else:
            chunks = []
            for r in results:
                if r["metadata"]:
                    chunks.append({
                        "text": r["metadata"].get("text", ""),
                        "source": r["metadata"].get("source", "unknown"),
                        "page": r["metadata"].get("page", "unknown"),
                        "similarity_score": r["similarity_score"]
                    })

            context = '\n\n'.join([chunk["text"] for chunk in chunks])

            # Create source information
            sources = [{
                'source': chunk["source"],
                'page': chunk["page"],
                'score': chunk["similarity_score"],
                'preview': chunk["text"][:120] + '...'
            } for chunk in chunks]

            # Generate answer
            prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
            
            if stream:
                return self._stream_answer(prompt, query, sources, summarize, chunks)
            else:
                # Non-streaming response
                response = self.llm.invoke([('user', prompt)])
                answer = response.content

                # Add citations to answer
                citations = [f'[{i+1}] {src["source"]} (page {src["page"]})' for i, src in enumerate(sources)]
                answer_with_citations = answer + '\n\nCitations:\n' + '\n'.join(citations) if citations else answer

            # Optionally summarize answer
            summary = None
            if summarize and answer:
                summary_prompt = f'Summarize the following answer in 2 sentences:\n{answer}'
                summary_resp = self.llm.invoke([('user', summary_prompt)])
                summary = summary_resp.content

            # Store query history
            self.history.append({
                'query': query,
                'answer': answer_with_citations,
                'sources': sources,
                'summary': summary
            })

            return {
                'type': 'final',
                'query': query,
                'answer': answer,
                'answer_with_citations': answer_with_citations,
                'summary': summary,
                'chunks': chunks,
                'sources': sources,
                'history': self.history[-5:]
            }
        
    def _stream_answer(self, prompt: str, query: str, sources: List[Dict], 
                       summarize: bool, chunks: List[Dict]) -> Generator[Dict[str, Any], None, None]:
        """Internal method to handle streaming responses"""
        # Stream the answer from the LLM
        stream = self.llm.stream([('user', prompt)])
        answer_chunks = []

        # Send the first chunk with sources information
        yield {
            'type': 'sources',
            'sources': sources,
            'chunks': chunks
        }

        # Stream the answer content
        for chunk in stream:
            if chunk.content:
                answer_chunks.append(chunk.content)
                yield {
                    'type': 'content',
                    'content': chunk.content
                }

        # Combine all chunks to form the complete answer
        answer = "".join(answer_chunks)

        # Add citations to answer
        citations = [f'[{i+1}] {src["source"]} (page {src["page"]})' for i, src in enumerate(sources)]
        answer_with_citations = answer + '\n\nCitations:\n' + '\n'.join(citations) if citations else answer

        # Optionally summarize answer
        summary = None
        if summarize and answer:
            summary_prompt = f'Summarize the following answer in 2 sentences:\n{answer}'
            summary_resp = self.llm.invoke([('user', summary_prompt)])
            summary = summary_resp.content

            # Stream the summary
            yield {
                'type': 'summary',
                'summary': summary
            }

         # Store query history
        self.history.append({
            'query': query,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary
        })

        # Send final chunk with complete response
        yield {
            'type': 'final',
            'query': query,
            'answer': answer,
            'answer_with_citations': answer_with_citations,
            'chunks': chunks,
            'sources': sources,
            'summary': summary,
            'history': self.history[-5:]
        }

    def rebuild_vector_store(self, chunk_size: int = None, chunk_overlap: int = None):
        """Rebuild the vector store with new parameters"""
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap

         # Reinitialize vector store with new parameters
        self.vector_store = FaissVectorStore(
            self.vector_store.persist_dir, 
            embedding_model=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # Rebuild from documents
        docs = load_all_documents("data")
        self.vector_store.build_from_documents(docs)

        return {
            "message": "Vector store rebuilt successfully",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }