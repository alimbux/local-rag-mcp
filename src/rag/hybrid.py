"""Hybrid retrieval pipeline.

    user query
        -> LLM query expansion (keywords)
        -> vector search  ||  BM25 full-text search   (run in parallel)
        -> Reciprocal Rank Fusion
        -> top-K chunks

Every stage degrades to a safe empty result instead of raising, so a missing
model, a missing index or a bad LLM response can never crash the caller.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ENABLE_HYBRID,
    QUERY_EXPANSION_ENABLED,
    VECTOR_TOP_K,
    FTS_TOP_K,
    RRF_K,
    TOP_K,
)
from rag.query import vector_search
from rag.fts import fts_search
from rag.fusion import rrf_fuse
from rag.expand import expand_query


def _safe(fn, *args, label=""):
    """Run ``fn(*args)``; on failure log and return ``[]`` so one failing
    branch cannot sink the whole retrieval."""
    try:
        return fn(*args)
    except Exception as e:
        print(f"⚠️  {label} search failed: {e}")
        return []


def hybrid_retrieve(user_query: str, verbose: bool = False):
    """Retrieve chunks for ``user_query`` using query expansion + hybrid search.

    Falls back to plain vector search when ``ENABLE_HYBRID`` is False. Always
    returns a list of chunk dicts (possibly empty); never raises.
    """
    keywords = expand_query(user_query) if QUERY_EXPANSION_ENABLED else []
    if verbose:
        print(f"🔑 Keywords: {keywords if keywords else '(none - using raw query)'}")

    if not ENABLE_HYBRID:
        return vector_search(user_query, TOP_K)

    # Vector branch: semantics of the question, nudged by the keywords.
    vec_query = user_query if not keywords else f"{user_query} {' '.join(keywords)}"
    # FTS branch: exact terms - keywords plus the original wording.
    fts_terms = ([user_query] + keywords) if keywords else [user_query]

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_vec = pool.submit(_safe, vector_search, vec_query, VECTOR_TOP_K, label="vector")
        f_fts = pool.submit(_safe, fts_search, fts_terms, FTS_TOP_K, label="FTS")
        vec_hits = f_vec.result()
        fts_hits = f_fts.result()

    if verbose:
        print(f"📊 vector hits: {len(vec_hits)} | FTS hits: {len(fts_hits)}")

    fused = rrf_fuse([vec_hits, fts_hits], k=RRF_K, top_k=TOP_K)

    if verbose:
        print(f"🔀 fused -> {len(fused)} chunks")

    return fused


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How do I build the FAISS index?"
    for chunk in hybrid_retrieve(q, verbose=True):
        print(f"  • {chunk['source']} #{chunk['chunk_id']}")
