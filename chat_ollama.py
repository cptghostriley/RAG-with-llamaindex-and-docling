import warnings
warnings.filterwarnings("ignore")

from llama_index.core import StorageContext, load_index_from_storage, Settings, PromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Configure LlamaIndex to use Ollama (once at startup)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="qwen2.5:3b-instruct")

# Custom prompt template
# qa_prompt_template = PromptTemplate(
#     """You are a careful and intelligent question-answering assistant.

# TASK:
# Answer the question using ONLY the information present in the provided context.

# IMPORTANT RULES:
# - The question may use synonyms or paraphrases (e.g., "who wrote this" = "author")
# - You are allowed to rephrase or interpret the question semantically
# - Do NOT use outside knowledge
# - Do NOT guess or hallucinate
# - If the required information is truly missing, reply exactly:
#   "I don't have enough information to answer this question."

# CONTEXT:
# {context_str}

# QUESTION:
# {query_str}

# REASONING (internal, do not reveal):
# - Identify what the question is asking
# - Map it to relevant facts in the context

# FINAL ANSWER:
# """
# )


# Load vector store ONCE at module import
print("Loading vector store...")
storage_context = StorageContext.from_defaults(persist_dir="D:\\Projects\\advanced_rag_project\\vector_index")
index = load_index_from_storage(storage_context)
_query_engine = index.as_query_engine(
    similarity_top_k=4, 
    streaming=True,
    # text_qa_template=qa_prompt_template
)
print("Vector store loaded successfully!")

# while True:
#     query = input("Enter your question (or 'exit' to quit): ")
#     if query.lower() == 'exit':
#         break
#     response = _query_engine.query(query)
    
#     # Stream the response token by token
#     print("Response: ", end="", flush=True)
#     for token in response.response_gen:
#         print(token, end="", flush=True)
#     print() 

def query_engine(query: str):
    response = _query_engine.query(query)
    return response