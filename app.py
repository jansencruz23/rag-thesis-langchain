from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import json
from src.search import RAGSearch

# Initialize FastAPI app
app = FastAPI(
    title="Thesis RAG API",
    description="API for querying thesis documents using RAG (Retrieval-Augmented Generation)",
    version="1.0.0"
)

# Initialize RAGSearch instance
rag_search = RAGSearch()

# Define request/response models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The query to search for")
    top_k: int = Field(5, description="Number of top results to retrieve", ge=1, le=20)
    score_threshold: float = Field(0.2, description="Minimum similarity score for results", ge=0.0, le=1.0)
    stream: bool = Field(True, description="Whether to stream the response")
    summarize: bool = Field(False, description="Whether to include a summary of the answer")

class RebuildRequest(BaseModel):
    chunk_size: int = Field(1000, description="Size of text chunks", ge=100, le=5000)
    chunk_overlap: int = Field(200, description="Overlap between chunks", ge=0, le=1000)

class ConfigResponse(BaseModel):
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int

# Define API endpoints
@app.get("/")
async def root():
    return {"message": "Thesis RAG API is running"}

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query the thesis documents"""
    try:
        if request.stream:
            result = rag_search.answer(
                query=request.query,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                stream=request.stream,
                summarize=request.summarize
            )

            if hasattr(result, '__iter__') and not isinstance(result, dict):
                async def generate():
                    for chunk in result:
                        yield f"data: {json.dumps(chunk)}\n\n"

                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            else:
                return result
        else:
            # Non-streaming response
            result = rag_search.answer(
                query=request.query,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                stream=request.stream,
                summarize=request.summarize
            )
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
@app.post("/rebuild")
async def rebuild_vector_store(request: RebuildRequest):
    """Rebuild the vector store with new chunking parameters"""
    try:
        result = rag_search.rebuild_vector_store(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rebuilding vector store: {str(e)}")
    
@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration"""
    return {
        "embedding_model": rag_search.embedding_model,
        "llm_model": rag_search.llm_model,
        "chunk_size": rag_search.chunk_size,
        "chunk_overlap": rag_search.chunk_overlap
    }

@app.get("/history")
async def get_history(limit: int = Query(10, description="Number of recent queries to return", ge=1, le=50)):
    """Get query history"""
    return {"history": rag_search.history[-limit:]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)