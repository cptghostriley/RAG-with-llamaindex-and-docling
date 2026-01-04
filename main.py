from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import shutil
import os
from create_vectorstore import create_vector_store
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

app = FastAPI(title="RAG API", description="Local RAG System with Docling + Ollama")

UPLOAD_DIR = "D:\\Projects\\advanced_rag_project\\data"
VECTOR_INDEX_DIR = "D:\\Projects\\advanced_rag_project\\vector_index"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Reranker for better retrieval
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=3
)

# Global query engine
_query_engine = None

class Answer(BaseModel):
    response: str
    
    class Config:
        from_attributes = True
    
@app.get("/")
def home():
    return {"message": "You've landed on the RAG API home page."}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF, DOCX, TXT, PPTX) to be ingested into the RAG system."""
    global _query_engine
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create vector store and query engine
    index = create_vector_store(UPLOAD_DIR)
    if index is None:
        raise HTTPException(status_code=500, detail="Error creating vector store")
    
    _query_engine = index.as_query_engine(
        similarity_top_k=6,
        streaming=False,  # Disable streaming for API
        node_postprocessors=[reranker]
    )
    
    return {"message": f"Document '{file.filename}' ingested successfully!"}

@app.post("/ask", response_model=Answer)
def ask_question(query: str):
    """Ask a question about the uploaded documents."""
    global _query_engine
    
    if _query_engine is None:
        raise HTTPException(status_code=400, detail="Please upload a document first")
    
    response = _query_engine.query(query)
    return Answer(response=str(response))