"""LLM-based query expansion / keyword generation.

Produces a short list of keywords/phrases that feed the full-text (BM25) branch
of hybrid retrieval. Built for small local models (Qwen 0.6B-3B): a tiny prompt,
temperature 0, and a fallback that never raises - if the model is unavailable or
returns garbage, the caller keeps working with the raw query.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OLLAMA_MODEL,
    EXPANSION_TEMPERATURE,
    EXPANSION_MAX_KEYWORDS,
)

_PROMPT = """Extract 3-6 short search keywords from the question below.
Keep exact names, abbreviations, file names and commands unchanged.
Answer with the keywords separated by commas. No other text.

Question: {query}
Keywords:"""

_TOKEN_RE = re.compile(r"\w", flags=re.UNICODE)


def _llm_raw(prompt: str) -> str:
    """Single Ollama call, isolated so failures are easy to contain."""
    import requests
    from config import OLLAMA_URL

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": EXPANSION_TEMPERATURE},
        },
        timeout=30,
    )
    return response.json().get("response", "")


def _parse_keywords(text: str):
    """Parse a keyword list out of a small model's free-form answer."""
    text = text.strip()
    if not text:
        return []

    # Drop a markdown code fence if the model wrapped its answer in one.
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

    # Pick the keyword line: prefer the first line that looks like a list
    # (has a comma or is a JSON array), else the first non-empty line. Small
    # models sometimes prepend a sentence of commentary despite instructions.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        text = next(
            (ln for ln in lines if "," in ln or ln[:1] in ("[", "{")),
            lines[0],
        )

    # Strip an echoed "Keywords:" / "Keyword -" prefix.
    text = re.sub(r"^\s*keywords?\s*[:\-]\s*", "", text, flags=re.IGNORECASE)

    raw_items = []

    # Try simple JSON first (list of strings, or dict values).
    if text[:1] in ("[", "{"):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                raw_items = [str(x) for x in data]
            elif isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list):
                        raw_items.extend(str(x) for x in value)
                    else:
                        raw_items.append(str(value))
        except (ValueError, TypeError):
            raw_items = []

    # Fall back to comma-separated parsing.
    if not raw_items:
        raw_items = text.split(",")

    keywords = []
    seen = set()
    for item in raw_items:
        # Trim surrounding quotes, brackets, list bullets and punctuation.
        kw = item.strip().strip("\"'`[](){}.,;:!?*•–—- \t").strip()
        if len(kw) < 2 or not _TOKEN_RE.search(kw):
            continue
        low = kw.lower()
        if low in seen:
            continue
        seen.add(low)
        keywords.append(kw)

    return keywords


def expand_query(user_query: str):
    """Return keywords/phrases for ``user_query`` (excluding the query itself).

    Returns ``[]`` on any failure or unusable model output - never raises.
    """
    try:
        raw = _llm_raw(_PROMPT.format(query=user_query))
    except Exception as e:  # network, timeout, malformed Ollama response, ...
        print(f"⚠️  Query expansion failed ({e}); using raw query")
        return []

    keywords = _parse_keywords(raw)

    if not keywords:
        print("⚠️  Query expansion returned no usable keywords; using raw query")
        return []

    if len(keywords) > EXPANSION_MAX_KEYWORDS:
        # Model rambled - keep the first N rather than discarding everything.
        keywords = keywords[:EXPANSION_MAX_KEYWORDS]

    return keywords


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How do I build the FAISS index?"
    print(f"Query:    {q}")
    print(f"Keywords: {expand_query(q)}")
