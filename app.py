from fastapi import FastAPI, HTTPException, Query, Body, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAGSearch instance
rag_search = RAGSearch(
    embedding_model="nomic-embed-text",
    llm_model="gemma3:4b",
)

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
async def query_documents(
    query: str = Form(..., description="The query to search for"),
    top_k: int = Form(5, ge=1, le=20),
    score_threshold: float = Form(0.2, ge=0.0, le=1.0),
    stream: bool = Form(True),
    summarize: bool = Form(False),
    file: Optional[UploadFile] = File(None, description="Optional image upload")
):
    """Query the thesis documents with optional image"""
    try:
        # 1. Handle Image Logic (Placeholder for your vision model)
        if file:
            print(f"Received file: {file.filename} ({file.content_type})")
            # TODO: Pass 'file' to rag_search.answer() if your RAG supports vision
            # content = await file.read() 
        
        # 2. Call RAG Search (Pass parameters individually)
        if stream:
            result = rag_search.answer(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                stream=stream,
                summarize=summarize
                # image=file (Add this if your RAG class supports it)
            )

            if not isinstance(result, dict):
                def generate():
                    for chunk in result:
                        # print(f"{json.dumps(chunk)}", sep="") # Optional logging
                        yield f"data: {json.dumps(chunk)}\n\n"

                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache, no-transform", 
                        "Connection": "keep-alive", 
                        "X-Accel-Buffering": "no",
                        "Content-Encoding": "identity"}
                )
            else:
                return result
        else:
            # Non-streaming response
            result = rag_search.answer(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                stream=stream,
                summarize=summarize
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