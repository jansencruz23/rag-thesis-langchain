import os
from pickletools import dis
import faiss
import numpy as np
import pickle
from typing import List, Any
from src.embedding import EmbeddingPipeline
from langchain_ollama import OllamaEmbeddings

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str="nomic-embed-text", 
                 chunk_size: int=1000, chunk_overlap: int=200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = OllamaEmbeddings(model=embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def build_from_documents(self, documents: List[Any]):
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, 
                                     chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks using Ollama.")
        embeddings = self.model.embed_documents(texts)
        print(f"[INFO] Generated embeddings with shape: {np.array(embeddings).shape}")
        
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", "unknown"),
                "chunk_id": i
            }
            metadatas.append(metadata)

        self.add_embeddings(np.array(embeddings).astype("float32"), metadatas)
        self.save()

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
            
        # Normalize embeddings for better cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query_embedding: np.ndarray, top_k: int=5, score_threshold: float=0.2):
        """
        Searches the FAISS index for the most similar vectors.
        Args:
            query_embedding: The embedding of the query.
            top_k: The number of results to retrieve before filtering.
            score_threshold: The minimum similarity score a result must have to be included.
                             Assumes the score is from Inner Product/Cosine Similarity (higher is better).
        Returns:
            A list of dictionaries, each containing 'index', 'distance', and 'metadata'.
        """
        # Normalize query embedding for consistent similarity calculation
        faiss.normalize_L2(query_embedding)
        D, I = self.index.search(query_embedding, top_k)

        filtered_results = []
        for idx, dist in zip(I[0], D[0]):
            if idx == -1:
                continue 

            if dist >= score_threshold:
                meta = self.metadata[idx] if idx < len(self.metadata) else None
                filtered_results.append({
                    'index': idx, 
                    'similarity_score': float(dist), 
                    'metadata': meta
                })

        return filtered_results
    
    def query(self, query_text: str, top_k: int=5, score_threshold: float=0.2):
        """
        Encodes a text query and searches the vector store.
        Args:
            query_text: The raw text query string.
            top_k: The number of results to retrieve.
            score_threshold: The minimum similarity score for filtering results.
        Returns:
            The list of filtered search results.
        """
        query_emb = np.array([self.model.embed_query(query_text)]).astype("float32")
        return self.search(query_emb, top_k=top_k, score_threshold=score_threshold)