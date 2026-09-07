# Configuration for Company Knowledge Base Assistant

# Document directory - update this to point to your company documentation
DOCUMENTS_DIR = "./docs"

# Chunking configuration
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# FAISS index paths (relative to src directory)
FAISS_INDEX_PATH = "index.faiss"
CHUNKS_PATH = "chunks.pkl"

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:1.7b"

# RAG retrieval configuration
TOP_K = 5  # final number of chunks passed to the LLM (after fusion)

# --- Hybrid search (vector + BM25 full-text) ---
ENABLE_HYBRID = True     # False -> plain vector search, identical to the old behaviour
VECTOR_TOP_K = 10        # depth of the vector result list before fusion
FTS_TOP_K = 10           # depth of the BM25 result list before fusion
RRF_K = 60               # constant k in the Reciprocal Rank Fusion formula

# --- Query expansion (LLM keyword generation) ---
QUERY_EXPANSION_ENABLED = True
EXPANSION_TEMPERATURE = 0.0   # keep small local models deterministic
EXPANSION_MAX_KEYWORDS = 6    # more than this -> treat the model output as rambling
