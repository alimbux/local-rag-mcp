import faiss
import pickle
import requests
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    OLLAMA_URL,
    OLLAMA_MODEL,
    TOP_K,
    VECTOR_TOP_K,
)

model = SentenceTransformer(EMBEDDING_MODEL)

# Global variables for index and chunks
index = None
chunks = []


def _ensure_index_exists():
    """Ensure FAISS index exists, build it if it doesn't."""
    global index, chunks
    
    # Resolve paths relative to src directory
    src_dir = Path(__file__).parent.parent
    index_path = src_dir / FAISS_INDEX_PATH
    chunks_path = src_dir / CHUNKS_PATH
    
    # Check if index exists
    if index_path.exists() and chunks_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            return True
        except Exception as e:
            print(f"⚠️  Warning: Error loading existing index: {e}")
            print("Rebuilding index...")
    
    # Index doesn't exist or failed to load, build it
    print("📦 Index not found. Building index from documents...")
    try:
        from rag.build_index import build_index
        build_index()
        
        # Load the newly created index
        if index_path.exists() and chunks_path.exists():
            index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            print("✅ Index built and loaded successfully")
            return True
        else:
            print("❌ Failed to build index. No documents found or error occurred.")
            from config import DOCUMENTS_DIR
            docs_path = src_dir / DOCUMENTS_DIR
            print(f"   Check that documents exist in: {docs_path}")
            return False
    except Exception as e:
        print(f"❌ Error building index: {e}")
        import traceback
        traceback.print_exc()
        return False


# Initialize index on module load
_ensure_index_exists()


def vector_search(query: str, top_k: int = VECTOR_TOP_K):
    """Vector similarity search. Returns chunk dicts ranked best-first.

    Out-of-range ids (FAISS returns -1 when the index has fewer than `top_k`
    vectors) are skipped instead of wrapping around to `chunks[-1]`.
    """
    if index is None or len(chunks) == 0:
        if not _ensure_index_exists():
            return []

    if index is None or len(chunks) == 0:
        return []

    q_emb = model.encode([query])
    faiss.normalize_L2(q_emb)

    scores, ids = index.search(q_emb, top_k)
    return [chunks[i] for i in ids[0] if 0 <= i < len(chunks)]


def retrieve(query: str):
    """Backward-compatible vector retrieval (top ``TOP_K`` chunks)."""
    return vector_search(query, TOP_K)


def build_prompt(query, contexts):
    """Build prompt with retrieved context."""
    if not contexts:
        return f"""
<role>You are a helpful assistant that answers questions about company information.</role>
<instructions>Answer the question based on your general knowledge. If you don't know, say so.</instructions>

<query>
{query}
</query>

<assistant>
"""

    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in contexts
    )

    return f"""
<role>You are a helpful assistant that answers questions about company information.</role>
<instructions>Answer the question ONLY based on the context provided below. If the answer is not in the context, say "I don't have that information in the knowledge base."</instructions>

<context>
{context_text}
</context>

<query>
{query}
</query>

<assistant>
"""


def ask_llm(prompt):
    """Query Ollama LLM."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False
        }
    )
    data = response.json()
    if "response" not in data:
        # Ollama returns {"error": "..."} for e.g. a model that isn't pulled.
        raise RuntimeError(
            f"Ollama did not return a response for model '{OLLAMA_MODEL}': "
            f"{data.get('error', data)}"
        )
    return data["response"]


def ask(query: str):
    """Answer a question using RAG."""
    contexts = retrieve(query)
    prompt = build_prompt(query, contexts)
    return ask_llm(prompt), contexts


if __name__ == "__main__":
    while True:
        q = input("\n❓ Question: ")
        if q.lower() in {"exit", "quit"}:
            break
        print("\n🤖 Answer:\n")
        answer, sources = ask(q)
        print(answer)
        if sources:
            print("\n📚 Sources:")
            for src in sources:
                print(f"  - {src['source']}")
