# Local RAG/MCP Knowledge Base Assistant

# 📋 The Problem

- **Growing Documentation**: Knowledge scattered across files
- **Information Retrieval**: Hard to find answers without keywords
- **Privacy Concerns**: Cloud solutions may not comply with policies

```
Users → Search → Answer = 😫
```

# ✨ The Solution

A **local, intelligent Q&A system** using:

- **Hybrid RAG**: Vector (FAISS) **+** full-text (BM25) search, merged with RRF
- **Query Expansion**: LLM extracts keywords before the search runs
- **MCP**: Dynamic document access
- **Local LLM**: Privacy-preserving answers (Ollama)

# ✨ Key Benefits

- ✅ Privacy-first (runs locally)
- ✅ No API costs
- ✅ Hybrid search — semantic recall **and** exact-term precision
- ✅ Robust to rare terms, abbreviations, file/command names
- ✅ Intelligent document access
- ✅ Complete data control

# 🏗️ Architecture - Top Level

```
┌──────────────────────┐
│   User Interface     │ (CLI)
└──────────┬───────────┘
           │
           ▼
   [Query Expansion]  (LLM → keywords)
           │
     ┌─────┴─────┐
     ▼           ▼
 [Hybrid RAG]  [MCP]
 Vector+BM25   Tools
     │           │
     └─────┬─────┘
           ▼
    [Ollama LLM]
```

# 🏗️ Architecture - Storage

```
┌────────────────┐   ┌────────────────┐
│  FAISS Index   │   │   BM25 Index   │
│ (vectors,disk) │   │ (in-memory)    │
└────────┬───────┘   └────────┬───────┘
         │                    │
         └─────────┬──────────┘
              ┌────▼─────┐
              │   docs/  │  (recursive: .md .txt .pdf .docx)
              └──────────┘
       + MCP tools read docs/ directly
```

# 🔍 Hybrid RAG Pipeline

1. Document Loading → Read .md, .txt, .pdf, .docx (recursively)
2. Chunking → Split into 700-token chunks (overlap 100)
3. Embedding → SentenceTransformers → FAISS vector index
4. Indexing → BM25 index built in-memory from the same chunks
5. **Query Expansion** → LLM extracts 3–6 keywords (temperature 0)
6. **Parallel search** → vector (FAISS) ∥ full-text (BM25), 10 hits each
7. **Fusion** → Reciprocal Rank Fusion → Top-5 chunks
8. Prompt Building → Create context-aware prompt
9. LLM Generation → Get answer from model

# 🔍 Why Hybrid (Vector + BM25)?

- Vectors capture **meaning**; BM25 captures **exact tokens**
- Small local models + embeddings miss rare terms and acronyms
- BM25 nails file names, commands, error codes, abbreviations
- RRF needs no score calibration between the two retrievers
- Parallel execution → latency ≈ max(vector, BM25), not the sum

# 🔧 MCP - Model Context Protocol

MCP provides **standardized interface** for LLM tool access:

```python
read_document(file_path)
list_documents()
search_documents(query)
```

# 🔧 MCP Benefits

- Tool Use by LLM
- Real-time document access
- Standardized interface
- Easy to extend
- Local tool execution

# 💻 Tech Stack

```
Language:      Python 3.10+
Vector DB:     FAISS
Full-text:     rank-bm25 (BM25Okapi)
Embeddings:    SentenceTransformers
LLM:           Ollama (local)
MCP:           FastMCP
```

# 📁 Project Structure

```
src/
├── config.py           Configuration
├── main.py             CLI entry point
├── assistant.py        Main orchestrator
├── bench.py            Vector-only vs hybrid benchmark
├── rag/
│   ├── ingest.py      Load documents (recursive)
│   ├── chunk.py       Split text
│   ├── embed.py       Generate embeddings
│   ├── build_index.py Build FAISS index
│   ├── query.py       Vector search + generate
│   ├── expand.py      LLM query expansion (+ fallback)
│   ├── fts.py         BM25 full-text search
│   ├── fusion.py      Reciprocal Rank Fusion
│   └── hybrid.py      Orchestrates expand → parallel search → RRF
├── mcp/
│   ├── server.py      MCP tool definitions
│   └── client.py      MCP client wrapper
└── docs/              Documentation
```

# 🚀 Index Building (Setup)

```
$ python main.py build-index

1. Load documents
  ↓
2. Split into chunks
  ↓
3. Generate embeddings
  ↓
4. Build FAISS index
  ↓
5. Save files
```

# 🚀 Query Processing (Runtime)

```
User Question
  ↓
LLM Query Expansion → keywords (or [] on failure)
  ↓
┌───────────────┬───────────────┐
▼               ▼               │  run in parallel
Vector search   BM25 search     │  (ThreadPoolExecutor)
(FAISS, 10)     (rank-bm25, 10) │
└───────┬───────┴───────────────┘
        ▼
Reciprocal Rank Fusion → Top 5 chunks
  ↓
LLM decides: Use MCP tools?
  ↓
Build prompt + context (+ MCP result)
  ↓
Call Ollama
  ↓
Return answer + sources
```

# 🔀 Hybrid Fusion — RRF

```python
# Reciprocal Rank Fusion over the two ranked lists
RRF_Score(d) = Σ  1 / (k + rank_m(d))     # k = 60, rank starts at 1
```

- Each retriever contributes `1/(k+rank)` per document
- Documents found by **both** retrievers rise to the top
- No need to normalise FAISS cosine vs BM25 scores

# ✨ Core Features

- **Hybrid Search**: Vector meaning + BM25 exact tokens
- **Query Expansion**: LLM keywords sharpen the search
- **Parallel Retrieval**: FTS adds accuracy, not latency
- **Graceful Fallback**: Bad LLM output / missing index → raw query, no crash
- **Multi-format**: .md, .txt, .pdf, .docx files (recursive)
- **Source Attribution**: Shows document sources
- **MCP Tools**: LLM can read full documents
- **No External APIs**: Runs locally only

# ⚙️ Configuration Options

```python
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen3:1.7b"   # any local Qwen 0.6B–3B

TOP_K = 5                     # final chunks after fusion

# Hybrid search
ENABLE_HYBRID = True          # False → plain vector search
VECTOR_TOP_K = 10             # vector hits before fusion
FTS_TOP_K = 10               # BM25 hits before fusion
RRF_K = 60                    # RRF constant k

# Query expansion
QUERY_EXPANSION_ENABLED = True
EXPANSION_TEMPERATURE = 0.0   # deterministic keywords
EXPANSION_MAX_KEYWORDS = 6
```

# 🎬 Live Demo - Starting

```bash
$ python main.py
```

Output:
```
🤖 Company Knowledge Base
Ask questions about documentation
Type 'exit' to stop
```

# 🎬 Demo - Query 1

```
❓ What are company values?

🤖 Innovation, integrity, collaboration

📚 Sources:
  • Loan Rangers Team.md
  • Info Security.md
```

# 🎬 Demo - Query 2

```
❓ What documents do we have?

🤖 [Uses MCP list_documents]
  • Loan Rangers Team.md
  • Information Security.md
  • Services.md
```

# 🎬 Demo - Query 3

```
❓ Full security policy?

🤖 [Uses MCP read_document]
[Full document content...]
```

# 🔐 Security - Local vs Cloud

**Cloud**: Data → Internet → Server
- ⚠️ Network transmission
- ⚠️ External storage
- ⚠️ Subscription costs

**Local**: Data → Local System
- ✅ No transmission
- ✅ Local storage only
- ✅ No costs

# 🔐 Implementation Safeguards

- **MCP Sandbox**: Prevents path traversal
- **Local Storage**: Documents stay on device
- **No Telemetry**: No tracking
- **Offline Ready**: Works without internet

# ⚡ Performance Benchmarks

```
Index Building:    ~30s (one-time)
Query Expansion:   0.3-1.5s (LLM keyword call)
Vector ∥ BM25:     ~50ms  (run in parallel)
RRF Fusion:        <1ms
LLM Generation:    2-5s
Total Cycle:       3-8s
```

# 📊 Before / After

`Results.png` — same question, `ENABLE_HYBRID` off vs on:

| | Vector-only | Hybrid (expansion + BM25 + RRF) |
|---|---|---|
| Top source | wrong doc (`QA Lead.md`) | correct doc (`Documentation Lead (Docs).md`) |
| Answer | generic, off-topic | grounded in the right role doc |

Run your own: `python bench.py` → recall@k / precision@k / MRR.

# ⚡ Tuning for Speed

```python
# Skip the extra LLM call:
QUERY_EXPANSION_ENABLED = False

# Drop back to plain vector search:
ENABLE_HYBRID = False

# Faster retrieval:
TOP_K = 3
VECTOR_TOP_K = 5
FTS_TOP_K = 5
```

# 🚢 Deployment - Single Machine

```
1. Install Ollama & Python deps
2. Copy docs/ to server
3. Build index
4. Run with nohup

$ nohup python main.py > log &
```

# 🚢 Scaling - Option 1: FastAPI

```
[HTTP Clients]			[HTTP Clients + Webllm]
       ↓        						 ↓
   [FastAPI]     				 [FastAPI]
       ↓         					 ↓
[Ollama + FAISS]      			  [FAISS]
```

# 🚢 Scaling - Option 2: Distributed

```
[Clients] → [Load Balancer]
             ↓
      [Multiple Retrievers]
```

# 🚢 Storage Scaling

```
Docs     Index      Build
10 MB    ~2 MB      ~5s
100 MB   ~20 MB     ~30s
1 GB     ~200 MB    ~5min
```

# 🔮 Phase 2: Enhanced Features

- ☐ Web UI (Streamlit)
- ☐ API endpoints
- ☐ Multi-language support
- ☐ Document versioning
- ☐ Fine-tuned embeddings

# 🔮 Phase 3: Advanced

- ☐ Conversation memory
- ☐ Multi-hop reasoning
- ☐ Metadata filtering
- ☐ Feedback loop
- ☐ Analytics dashboard

# 🔮 Phase 4: Enterprise

- ☐ User authentication
- ☐ Audit logging
- ☐ Role-based access
- ☐ LLM fine-tuning
- ☐ Cost analysis

# 📊 Why This Works

| Aspect | Traditional | Our Hybrid RAG |
|--------|---|---|
| **Understanding** | Keywords | Semantic **+** lexical |
| **Rare terms** | Hit or miss | BM25 catches them |
| **Answers** | Documents | Direct |
| **Privacy** | Cloud | Local |
| **Cost** | Subscription | One-time |

# ✅ What You Have Now

- Local privacy-first knowledge base
- Hybrid retrieval: FAISS **+** BM25, fused with RRF
- LLM query expansion with a safe fallback
- Parallel search — accuracy without added latency
- Intelligent tool use (MCP)
- A benchmark to measure retrieval quality

# 🙋 Quick Reference

```bash
# Install deps (adds rank-bm25)
pip install -r requirements.txt

# Build index
python main.py build-index

# Run interactively
python main.py

# Benchmark vector-only vs hybrid
python bench.py            # needs bench_questions.json
```

# 📚 Resources

- **FAISS**: facebook/faiss
- **rank-bm25**: dorianbrown/rank_bm25
- **RRF paper**: Cormack et al., 2009
- **Ollama**: ollama.ai
- **FastMCP**: github.com/jlowin/fastmcp
- **Transformers**: huggingface.co

**Thank You!**
