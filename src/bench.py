#!/usr/bin/env python3
"""Retrieval benchmark: vector-only vs hybrid (query expansion + BM25 + RRF).

Usage:
    python bench.py [path/to/bench_questions.json]

The questions file is a JSON list; `expected` is matched as a case-insensitive
substring of a retrieved chunk's `source` path:

    [
      {"question": "What is the vacation policy?", "expected": "vacation-policy.md"},
      {"question": "Who is on the Loan Rangers team?", "expected": "Loan Rangers"}
    ]

Defaults to ./bench_questions.json next to this file.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import TOP_K
from rag.query import vector_search
from rag.hybrid import hybrid_retrieve


def _matches(chunks, expected):
    exp = expected.lower()
    return [c for c in chunks if exp in str(c.get("source", "")).lower()]


def evaluate(runner, questions, k):
    recall = precision = rr = latency = 0.0
    for item in questions:
        q, expected = item["question"], item["expected"]

        t0 = time.perf_counter()
        chunks = runner(q)[:k]
        latency += time.perf_counter() - t0

        hits = _matches(chunks, expected)
        recall += 1.0 if hits else 0.0
        precision += len(hits) / k if k else 0.0

        for rank, c in enumerate(chunks, start=1):
            if expected.lower() in str(c.get("source", "")).lower():
                rr += 1.0 / rank
                break

    n = len(questions)
    return {
        "recall@k": recall / n,
        "precision@k": precision / n,
        "mrr": rr / n,
        "latency_s": latency / n,
    }


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "bench_questions.json"
    if not path.exists():
        print(f"❌ {path} not found.")
        print('Create it, e.g.:  [{"question": "...", "expected": "some-doc.md"}]')
        return

    questions = json.loads(path.read_text(encoding="utf-8"))
    if not questions:
        print("❌ No questions in the bench file.")
        return

    print(f"Loaded {len(questions)} questions | Top-K = {TOP_K}\n")

    modes = {
        "vector-only": lambda q: vector_search(q, TOP_K),
        "hybrid": lambda q: hybrid_retrieve(q),
    }
    rows = {name: evaluate(runner, questions, TOP_K) for name, runner in modes.items()}

    cols = ["recall@k", "precision@k", "mrr", "latency_s"]
    header = f"{'mode':<14}" + "".join(f"{c:>14}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, metrics in rows.items():
        print(f"{name:<14}" + "".join(f"{metrics[c]:>14.4f}" for c in cols))

    print("\nQuestions the hybrid pipeline gets right and vector-only misses:")
    any_diff = False
    for item in questions:
        q, expected = item["question"], item["expected"]
        v = bool(_matches(vector_search(q, TOP_K), expected))
        h = bool(_matches(hybrid_retrieve(q), expected))
        if h and not v:
            any_diff = True
            print(f"  • {q}  (expected: {expected})")
    if not any_diff:
        print("  (none in this question set)")


if __name__ == "__main__":
    main()
