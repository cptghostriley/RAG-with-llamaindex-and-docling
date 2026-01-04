import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
# Import torch first to avoid DLL loading issues on Windows
import torch

import os
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
# from llama_parse import LlamaParse
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import nest_asyncio
nest_asyncio.apply()

from llama_index.core import Settings

# os.environ["LLAMA_CLOUD_API_KEY"] = "llx-Nf1vhdgHl1E0Zuw4aXRbsSWrw752nSO4PS3YQ7aGSjssbDBP"  

# Configure LlamaIndex to use Ollama (free, local) instead of OpenAI
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="qwen2.5:3b-instruct")

def create_vector_store(data_dir, save_path="D:\\Projects\\advanced_rag_project\\vector_index"):

    if not os.path.isdir(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return None
    
    converter = DocumentConverter()
    # parser = LlamaParse(result_type='text')
    all_docs = []
    
    print(f"Loading all documents from: {data_dir}")
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
             if f.endswith(('.pdf', '.docx', '.txt', '.pptx'))]
    
    if not files:
        print("Error: No supported documents found in directory.")
        return None
    
    results = converter.convert_all(files)
    
    chunker = HybridChunker(tokenizer="nomic-ai/nomic-embed-text-v1",
                            max_tokens=1000, 
                            merge_peers=True)
    
    all_chunks = []
    for result in results:
        doc = result.document  # This is the DoclingDocument
        source_name = os.path.basename(result.input.file.name) if result.input.file else "unknown"
        print(f"  Chunking: {source_name}")
        chunks = list(chunker.chunk(doc))
        for chunk in chunks:
            all_chunks.append((chunk, source_name))

    documents = [
        Document(
            text=chunk.text,
            metadata={"source": source}
        ) 
        for chunk, source in all_chunks
    ]
    
    print(f"\nTotal chunks created: {len(documents)}")
    print("Creating vector index...")
    
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=save_path)
    
    print(f"Vector store saved to: {save_path}")
    return index

if __name__ == "__main__":
    data_folder = "D:\\Projects\\advanced_rag_project\\data"
    create_vector_store(data_folder)


