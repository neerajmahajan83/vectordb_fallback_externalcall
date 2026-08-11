# RAG System with External LLM Fallback

A production-ready Retrieval-Augmented Generation (RAG) chatbot that combines a local vector database with an external LLM fallback. Domain-specific questions are answered offline using a local FAISS index and `flan-t5-small`. When no relevant document is found in the vector store, the system seamlessly routes the query to **Groq's `llama-3.3-70b-versatile`** for a general-purpose answer.

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [System Components](#system-components)
- [Knowledge Base](#knowledge-base)
- [Retrieval & Routing Logic](#retrieval--routing-logic)
- [Function Reference](#function-reference)
- [Example Session](#example-session)
- [Extending the System](#extending-the-system)
- [Security](#security)

---

## Architecture

The system follows a two-stage pipeline — retrieve then generate — with an intelligent routing layer that decides whether to answer locally or escalate to an external LLM.

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  Embedding Model            │  all-MiniLM-L6-v2
│  (query → 384-dim vector)   │  runs locally
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  FAISS Vector Search        │  IndexFlatIP (cosine similarity)
│  top-1 document retrieval   │  in-memory index
└────────────┬────────────────┘
             │
     ┌───────┴───────┐
     │               │
score >= 0.3     score < 0.3
     │               │
     ▼               ▼
┌─────────┐    ┌──────────────┐
│ flan-t5 │    │  Groq LLM    │
│ (local) │    │  llama-3.3   │
│ offline │    │  70b (API)   │
└─────────┘    └──────────────┘
     │               │
     └───────┬───────┘
             ▼
      HelpBot Response
      [source labeled]
```

**Routing decision:**
- `score >= 0.3` — a relevant document exists in the vector DB → answered locally by `flan-t5-small` using the retrieved document as context
- `score < 0.3` — no relevant document found → query is sent to Groq `llama-3.3-70b-versatile` for a general answer

---

## Project Structure

```
ragapp/
├── rag_system_with_external_llm.py   # main application
├── .env                               # API key store (never commit)
├── .gitignore                         # excludes .env and cache files
└── rag_system_with_external_llm.md   # this documentation
```

---

## Requirements

Python 3.9 or higher is recommended.

| Package | Version | Purpose |
|---|---|---|
| `sentence-transformers` | latest | Document and query embedding |
| `faiss-cpu` | latest | In-memory vector similarity search |
| `transformers` | latest | Local `flan-t5-small` for text generation |
| `torch` | latest | Backend runtime for transformers |
| `groq` | latest | Groq API client for external LLM fallback |
| `python-dotenv` | latest | Loads environment variables from `.env` |
| `numpy` | latest | Embedding array operations |

---

## Installation

Install all dependencies in one command:

```bash
pip install sentence-transformers faiss-cpu transformers torch groq python-dotenv numpy
```

On first run, the following models are automatically downloaded from Hugging Face and cached locally:

- `all-MiniLM-L6-v2` (~90 MB) — embedding model
- `google/flan-t5-small` (~80 MB) — local generation model

Subsequent runs use the local cache with no internet required (except for Groq API calls).

---

## Configuration

### 1. Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign in or create an account
3. Navigate to **API Keys** → **Create API Key**
4. Copy the generated key (starts with `gsk_`)

### 2. Create the `.env` file

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

> The application validates the key at startup and raises a `ValueError` immediately if it is missing or still set to the placeholder value. This prevents silent failures during inference.

---

## Running the Application

```bash
py rag_system_with_external_llm.py
```

On startup the application:
1. Loads the `.env` file and validates the Groq API key
2. Downloads or loads cached embedding model and flan-t5-small
3. Encodes all documents and builds the FAISS index
4. Starts the interactive CLI chat loop

Type `exit` at any prompt to quit.

---

## System Components

### Embedding Model — `all-MiniLM-L6-v2`

A lightweight sentence transformer that maps text to 384-dimensional dense vectors. Both documents (at startup) and user queries (at runtime) pass through this model. L2 normalization is applied to all vectors so that inner product search is equivalent to cosine similarity scoring.

### Vector Store — FAISS `IndexFlatIP`

Facebook AI Similarity Search (FAISS) stores the normalized document embeddings and performs exact nearest-neighbor search using inner product. The index is rebuilt in memory on every startup from the `documents` list — no persistence layer is required for a small knowledge base.

### Local Generator — `google/flan-t5-small`

An instruction-tuned encoder-decoder model (~80 MB). When a relevant document is retrieved, the model receives the following prompt:

```
Context: <retrieved document>
Question: <user query>
Answer:
```

Generation parameters:

| Parameter | Value | Reason |
|---|---|---|
| `min_new_tokens` | 5 | Prevents empty/EOS-only output |
| `max_new_tokens` | 100 | Enough for a complete sentence |
| `num_beams` | 4 | Beam search for coherent output |
| `early_stopping` | True | Stops cleanly at EOS token |

If the decoded output is still empty after generation, the system falls back to returning the retrieved document text directly.

### External Fallback — Groq `llama-3.3-70b-versatile`

When the FAISS similarity score falls below the threshold, the query is forwarded to Groq's hosted `llama-3.3-70b-versatile` model via REST API. This handles general knowledge questions outside the domain of the local knowledge base.

Groq call parameters:

| Parameter | Value |
|---|---|
| `max_tokens` | 200 |
| `temperature` | 0.3 |
| System prompt | Concise, factual one-to-two sentence answers |

All Groq API errors are caught and returned as readable messages rather than crashing the bot.

---

## Knowledge Base

The current documents cover **Amazon customer support** topics:

| # | Topic |
|---|---|
| 1 | Order tracking |
| 2 | Return policy (30-day window) |
| 3 | Return process |
| 4 | Customer service contact |
| 5 | Amazon Prime benefits |
| 6 | Delayed package handling |
| 7 | Order cancellation |
| 8 | Gift card purchase |
| 9 | Payment method update |
| 10 | Account login |

---

## Retrieval & Routing Logic

The `rag_answer` function is the core of the pipeline:

```
1. Encode the user query → 384-dim float32 vector
2. L2-normalize the query vector
3. Search FAISS index for top-1 nearest document
4. Read the cosine similarity score

   if score >= 0.3:
       → pass (query + retrieved_doc) to flan-t5-small
       → return generated answer  [source: "vector DB + local T5"]

   if score < 0.3:
       → forward raw query to Groq llama-3.3-70b
       → return API response      [source: "Groq LLM (external)"]
```

The threshold `0.3` was chosen as a conservative cutoff for cosine similarity on normalized vectors. A higher value (e.g. `0.5`) would make the system stricter about what counts as a match; a lower value would increase recall at the cost of relevance.

---

## Function Reference

### `answer_with_local_t5(query, retrieved_doc) → str`

Constructs a context-grounded prompt and runs flan-t5-small generation. Returns the decoded answer string, or the raw retrieved document if the model produces empty output.

### `answer_with_groq(query) → str`

Sends the user query to Groq's chat completions endpoint using `llama-3.3-70b-versatile`. Wraps the call in a try/except block and returns a readable error string on failure.

### `rag_answer(query, top_k=1, threshold=0.3) → tuple[str | None, str, str]`

Main RAG pipeline entry point. Returns a 3-tuple:
- `retrieved_doc` — the matched document, or `None` if no match
- `response` — the generated answer string
- `source` — label indicating which system produced the answer

### `run_qa_bot() → None`

Interactive CLI loop. Reads user input, calls `rag_answer`, and prints the labeled response. Handles empty input gracefully and exits cleanly on `exit`.

---

## Example Session

```
Welcome to the RAG Q&A Bot!
  - Questions about Amazon → answered from local vector DB
  - Other questions        → answered by Groq LLM (external)
Type 'exit' to quit.

User: how do I track my Amazon order
HelpBot [vector DB + local T5]: Log into your account, go to Your Orders, and click Track Package.

User: what benefits does Amazon Prime offer
HelpBot [vector DB + local T5]: Amazon Prime members receive free two-day shipping, exclusive deals, and access to Prime Video and Music.

User: what is a neural network
[INFO] No relevant document found in vector DB — calling Groq LLM...

HelpBot [Groq LLM (external)]: A neural network is a machine learning model inspired by the human brain, consisting of layers of interconnected nodes that learn patterns from data.

User: exit
Goodbye!
```

---

## Extending the System

### Add more documents to the knowledge base

Append strings to the `documents` list in the source file. The FAISS index is rebuilt automatically on the next startup — no migration step needed.

```python
documents.append("Your new support document text here.")
```

### Adjust the similarity threshold

Change the `threshold` parameter in `rag_answer` to control routing sensitivity:

```python
# stricter — only very close matches go to local model
_, answer, source = rag_answer(query, threshold=0.5)

# more lenient — more queries handled locally
_, answer, source = rag_answer(query, threshold=0.2)
```

### Swap the external LLM

Replace `GROQ_MODEL` with any other model available on Groq:

```python
GROQ_MODEL = "mixtral-8x7b-32768"   # larger context window
GROQ_MODEL = "llama3-8b-8192"       # faster, smaller
```

### Persist the FAISS index

For large document sets, save and load the index instead of rebuilding each time:

```python
# save
faiss.write_index(index, "docs.index")

# load
index = faiss.read_index("docs.index")
```

---

## Security

- `.env` is listed in `.gitignore` — the API key is never committed to version control
- The application validates the key at startup and exits with a clear error before any model loading if the key is absent
- All Groq API errors are caught at the call site and returned as safe strings — no raw exceptions are surfaced to the user
