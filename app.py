from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline

if __name__ == "__main__":
    embedding = EmbeddingPipeline()

    docs = load_all_documents("data")
    chunks = embedding.chunk_documents(docs)
    vectors = embedding.embed_chunks(chunks)

    print(vectors)