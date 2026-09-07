"""Fusion of ranked result lists from different retrievers.

``rrf_fuse`` (Reciprocal Rank Fusion) is the default; ``weighted_fuse`` is kept
for experimentation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RRF_K, TOP_K


def _key(chunk):
    """Stable identity for a chunk across result lists."""
    return (chunk.get("source"), chunk.get("chunk_id"))


def rrf_fuse(result_lists, k: int = RRF_K, top_k: int = TOP_K):
    """Merge ranked lists of chunk dicts with Reciprocal Rank Fusion:

        RRF(d) = sum_m  1 / (k + rank_m(d))        (rank starts at 1)

    Returns the ``top_k`` chunk dicts ordered by descending fused score.
    """
    scores = {}
    meta = {}
    for hits in result_lists:
        for rank, chunk in enumerate(hits, start=1):
            key = _key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            meta.setdefault(key, chunk)

    ordered = sorted(scores, key=scores.get, reverse=True)
    return [meta[key] for key in ordered[:top_k]]


def weighted_fuse(result_lists, weights, top_k: int = TOP_K):
    """Weighted sum of per-list rank scores, each list normalised to (0, 1].

    An alternative to RRF when you want to bias one retriever over the other.
    """
    scores = {}
    meta = {}
    for hits, weight in zip(result_lists, weights):
        n = len(hits)
        for rank, chunk in enumerate(hits):
            key = _key(chunk)
            norm = (n - rank) / n if n else 0.0
            scores[key] = scores.get(key, 0.0) + weight * norm
            meta.setdefault(key, chunk)

    ordered = sorted(scores, key=scores.get, reverse=True)
    return [meta[key] for key in ordered[:top_k]]


if __name__ == "__main__":
    # Manual RRF check: doc "b" is rank 2 then rank 1 -> 1/62 + 1/61 (highest).
    a = [{"source": "a", "chunk_id": 0}, {"source": "b", "chunk_id": 0}]
    b = [{"source": "b", "chunk_id": 0}, {"source": "c", "chunk_id": 0}]
    for chunk in rrf_fuse([a, b], k=60, top_k=3):
        print(chunk)
