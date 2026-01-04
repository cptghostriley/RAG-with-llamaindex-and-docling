import gradio as gr
import os
import shutil
from create_vectorstore import create_vector_store
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

UPLOAD_DIR = "D:\\Projects\\advanced_rag_project\\data"
VECTOR_INDEX_DIR = "D:\\Projects\\advanced_rag_project\\vector_index"

# Cross-encoder reranker - improves retrieval quality significantly
# Options: "BAAI/bge-reranker-large" (best)
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=3  # Return top 3 after reranking
)

# Clear vector index on app start
if os.path.exists(VECTOR_INDEX_DIR):
    shutil.rmtree(VECTOR_INDEX_DIR)
    print("Cleared existing vector index.")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global variable to hold the query engine (loaded after first document ingestion)
_query_engine = None

def upload_file(file):
    global _query_engine
    if file is None:
        return "No file uploaded"

    file_path = os.path.join(UPLOAD_DIR, os.path.basename(file.name))
    shutil.copy(file.name, file_path)

    index = create_vector_store(UPLOAD_DIR)
    if index is None:
        return "Error creating vector store"
    
    # Create query engine from the newly created index with reranker
    _query_engine = index.as_query_engine(
        similarity_top_k=6,  # Retrieve more initially, reranker will filter to top 3
        streaming=True,
        node_postprocessors=[reranker]
    )
    return f"Document ingested successfully!"

def answer_question(question):
    global _query_engine
    if _query_engine is None:
        return "Please upload and ingest a document first."
    if not question.strip():
        return "Please enter a question"
    response = _query_engine.query(question)
    return str(response)

with gr.Blocks() as demo:
    gr.Markdown("# 📄 Ask the doc (Docling + Ollama)")

    with gr.Tab("Upload Document"):
        file_input = gr.File(label="Upload document")
        upload_btn = gr.Button("Ingest")
        upload_output = gr.Textbox(label="Upload status")
        upload_btn.click(upload_file, file_input, upload_output)

    with gr.Tab("Ask Questions"):
        question = gr.Textbox(label="Your question", lines=3)
        answer = gr.Textbox(label="Answer", lines=10)
        ask_btn = gr.Button("Ask")
        ask_btn.click(answer_question, question, answer)

demo.launch()
