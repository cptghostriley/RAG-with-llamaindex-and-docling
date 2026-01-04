# RAG with LlamaIndex and Docling

> Ask questions about your documents locally - no cloud APIs needed. Upload PDFs, Word docs, or PowerPoints and get accurate answers powered by Ollama LLMs. Features semantic chunking via Docling, vector search with reranking, and both a web UI (Gradio) and REST API (FastAPI).

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange.svg)

## ✨ Features

- **100% Local & Private** - Your data never leaves your machine
- **Multiple Document Formats** - PDF, DOCX, TXT, PPTX support
- **Smart Chunking** - Hybrid chunking with Docling for better context preservation
- **Cross-Encoder Reranking** - Improved retrieval accuracy using sentence-transformers
- **Dual Interface** - Gradio web UI + FastAPI REST endpoints
- **Customizable Models** - Easy to swap embedding models and LLMs via Ollama

## 🏗️ Architecture

```
Document → Docling Parser → Hybrid Chunker → Embeddings (Ollama)
                                                    ↓
Query → Embedding → Vector Search → Reranker → LLM (Ollama) → Answer
```

## 📋 Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- Required Ollama models:
  ```bash
  ollama pull nomic-embed-text
  ollama pull qwen2.5:3b-instruct
  ```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/cptghostriley/RAG-with-llamaindex-and-docling.git
   cd RAG-with-llamaindex-and-docling
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   **Gradio UI:**
   ```bash
   python app.py
   ```

   **FastAPI:**
   ```bash
   uvicorn main:app --reload
   ```

## 📖 Usage

### Gradio Web Interface
1. Open http://localhost:7860
2. Go to "Upload Document" tab and upload your file
3. Click "Ingest" to process the document
4. Switch to "Ask Questions" tab and ask away!

### FastAPI Endpoints
- `GET /` - Health check
- `POST /upload` - Upload and ingest a document
- `POST /ask?query=your question` - Ask a question

Swagger docs available at http://localhost:8000/docs

## 🛠️ Configuration

### Models
Edit `create_vectorstore.py` to change models:
```python
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="qwen2.5:3b-instruct")
```

### Reranker Options
Edit `app.py` to use different rerankers:
```python
# Fast (current)
model="cross-encoder/ms-marco-MiniLM-L-6-v2"

# Better accuracy
model="BAAI/bge-reranker-base"

# Best accuracy (slower)
model="BAAI/bge-reranker-large"
```

## 📁 Project Structure

```
├── app.py                 # Gradio web interface
├── main.py                # FastAPI REST API
├── create_vectorstore.py  # Document processing & indexing
├── chat_ollama.py         # Query engine (standalone)
├── data/                  # Uploaded documents
└── vector_index/          # Persisted vector store
```

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit PRs.

