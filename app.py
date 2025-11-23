from src.data_loader import load_all_documents
from src.vector_store import FaissVectorStore

if __name__ == "__main__":
    #docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    #store.build_from_documents(docs)
    store.load()
    print(store.query("What is GAN-based Deblurring?"))