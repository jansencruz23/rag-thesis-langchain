import os
from pickletools import dis
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str="all-MiniLM-L6-v2", 
                 chunk_size: int=1000, chunk_overlap: int=200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def build_from_documents(self, documents: List[Any]):
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, 
        chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype("float32"), metadatas)
        self.save()
        #print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        #print(f"[INFO] Added {embeddings.shape[0]} embeddings to the index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        #print(f"[INFO] Saved FAISS index to {faiss_path} and metadata to {meta_path}.")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        #print(f"[INFO] Loaded FAISS index from {faiss_path} and metadata from {meta_path}.")

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
        D, I = self.index.search(query_embedding, top_k)

        filtered_results = []
        for idx, dist in zip(I[0], D[0]):
            if idx == -1:
                continue 

            if dist >= score_threshold:
                meta = self.metadata[idx] if idx < len(self.metadata) else None
                filtered_results.append({'index': idx, 'distance': dist, 'metadata': meta})

        return filtered_results
    
    def query(self, query_text: str, top_k: int = 5, score_threshold: float=0.2):
        """
        Encodes a text query and searches the vector store.
        Args:
            query_text: The raw text query string.
            top_k: The number of results to retrieve.
            score_threshold: The minimum similarity score for filtering results.
        Returns:
            The list of filtered search results.
        """
        #print(f"[INFO] Querying vector store for: '{query_text}' with score_threshold={score_threshold}")
        query_emb = self.model.encode([query_text]).astype("float32")
        return self.search(query_emb, top_k=top_k, score_threshold=score_threshold)