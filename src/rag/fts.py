"""Full-text (BM25) search over the same chunks used by the vector index.

Runs entirely in memory from ``chunks.pkl`` - no extra persistence and no schema
migration. The BM25 index is built lazily on first use and cached for the life
of the process (which is short-lived: one CLI run).
"""
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNKS_PATH, FTS_TOP_K

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

# (bm25, chunks) once built.
_STATE = None

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str):
    r"""Lowercase word tokens. ``\w+`` with UNICODE keeps Cyrillic, Latin and digits."""
    return _TOKEN_RE.findall(text.lower())


def _ensure_index():
    """Build the BM25 index from chunks.pkl once. Returns True on success."""
    global _STATE
    if _STATE is not None:
        return True

    if BM25Okapi is None:
        print("⚠️  rank-bm25 not installed; FTS branch disabled")
        return False

    chunks_path = Path(__file__).parent.parent / CHUNKS_PATH
    if not chunks_path.exists():
        print(f"⚠️  {chunks_path.name} not found; run 'python main.py build-index' first")
        return False

    try:
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
    except Exception as e:
        print(f"⚠️  Could not load chunks for FTS: {e}")
        return False

    if not chunks:
        return False

    corpus = [_tokenize(c["text"]) for c in chunks]
    _STATE = (BM25Okapi(corpus), chunks)
    return True


def fts_search(keywords, top_k: int = FTS_TOP_K):
    """BM25 search for a list of keyword strings.

    Returns chunk dicts ranked best-first. Returns ``[]`` if the index is
    unavailable or the tokenised query is empty.
    """
    if not _ensure_index():
        return []

    bm25, chunks = _STATE

    query_tokens = _tokenize(" ".join(keywords))
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    hits = []
    for i in ranked[:top_k]:
        if scores[i] <= 0:
            break
        hits.append(chunks[i])
    return hits


def reset_cache():
    """Drop the cached index (e.g. after rebuilding chunks.pkl in the same process)."""
    global _STATE
    _STATE = None


if __name__ == "__main__":
    kws = sys.argv[1:] or ["FAISS", "index", "build"]
    print(f"Keywords: {kws}")
    for hit in fts_search(kws):
        print(f"  • {hit['source']} #{hit['chunk_id']}: {hit['text'][:80]}...")
