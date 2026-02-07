# Thesis RAG Backend

A FastAPI-based backend for querying thesis documents using Retrieval-Augmented Generation (RAG). This system enables semantic search across PDF and text documents, providing AI-generated answers with source citations.

## Features

- **Semantic Search**: Finds relevant document chunks using FAISS vector similarity search
- **Streaming Responses**: Real-time answer streaming via Server-Sent Events
- **Source Citations**: Automatically links answers to source documents with page references
- **Configurable Chunking**: Adjust chunk size and overlap for optimal retrieval
- **Query History**: Tracks recent queries for session continuity
- **Local LLM**: Runs entirely offline using Ollama

## Tech Stack

- **Framework**: FastAPI, Uvicorn
- **RAG Pipeline**: LangChain, LangChain-Ollama
- **Vector Store**: FAISS (with ChromaDB support)
- **Embeddings**: Nomic Embed Text via Ollama
- **LLM**: Gemma3:4b via Ollama
- **Document Processing**: PyPDF, PyMuPDF

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- Required Ollama models:
  ```bash
  ollama pull nomic-embed-text
  ollama pull gemma3:4b
  ```

## Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv rag-venv
   rag-venv\Scripts\activate  # Windows
   # or
   source rag-venv/bin/activate  # Linux/Mac
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file (optional):
   ```
   OLLAMA_HOST=http://localhost:11434
   ```

## Usage

1. Place your documents in the `data/` folder:
   - PDFs go in `data/pdf_files/`
   - Text files go in `data/text_files/`

2. Start the server:
   ```bash
   python app.py
   ```
   The API will be available at `http://localhost:8000`

3. Access the API docs at `http://localhost:8000/docs`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/query` | POST | Query documents with RAG |
| `/rebuild` | POST | Rebuild vector store with new parameters |
| `/config` | GET | Get current configuration |
| `/history` | GET | Get recent query history |

### Query Parameters

- `query` (required): The question to ask
- `top_k`: Number of chunks to retrieve (default: 5)
- `score_threshold`: Minimum similarity score (default: 0.2)
- `stream`: Enable streaming response (default: true)
- `summarize`: Include a summary of the answer (default: false)

## Project Structure

```
backend/
├── app.py              # FastAPI application entry point
├── src/
│   ├── search.py       # RAG search logic
│   ├── vector_store.py # FAISS vector store wrapper
│   ├── embedding.py    # Embedding generation
│   └── data_loader.py  # Document loading utilities
├── data/               # Source documents
├── faiss_store/        # Persisted vector index
└── requirements.txt    # Python dependencies
```

## Configuration

The RAG system can be configured when initializing:

- `embedding_model`: Ollama embedding model (default: "nomic-embed-text")
- `llm_model`: Ollama LLM model (default: "gemma3:4b")
- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)

## License

This project is part of a thesis research work.
