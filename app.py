from src.search import RAGSearch

rag_search = RAGSearch()

if __name__ == "__main__":
    query = input("Query: ")
    summary = rag_search.answer(query, top_k=3, summarize=True, stream=True)
    print("Answer:", summary)