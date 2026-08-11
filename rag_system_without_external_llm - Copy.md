# RAG System — Local Vector DB Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions purely from a local vector database using a local language model. No external API calls are made during inference.

---

## How It Works

```
User Query
    │
    ▼
FAISS Vector Search (cosine similarity)
    │
    ├── score >= 0.3  ──►  flan-t5-small (local)
    │                       answers using retrieved document as context
    │
    └── score < 0.3   ──►  "Sorry, I couldn't find a relevant answer."
```

Everything runs offline on your machine after the initial model download.

---

## Project Structure

```
ragapp/
├── rag_system_with_external_llm.py   # main application
├── .env                               # environment config (never commit this)
├── .gitignore                         # excludes .env from git
└── rag_system_with_external_llm.md   # this file
```

---

## Requirements

Install all dependencies:

```bash
pip install sentence-transformers faiss-cpu transformers torch python-dotenv
```

| Package | Purpose |
|---|---|
| `sentence-transformers` | Embeds queries and documents (`all-MiniLM-L6-v2`) |
| `faiss-cpu` | Vector similarity search index |
| `transformers` | Local `flan-t5-small` model for generation |
| `torch` | Required backend for transformers |
| `python-dotenv` | Loads config from `.env` file |

---

## Setup

### 1. Run

```bash
py rag_system_with_external_llm.py
```

Models are downloaded automatically from Hugging Face on first run and cached locally for all subsequent runs.

---

## Components

### Embedding Model — `all-MiniLM-L6-v2`

- Converts documents and queries into 384-dimension vectors
- Downloaded automatically on first run, then runs fully **offline**
- Used for both indexing documents and embedding user queries

### Vector Store — FAISS `IndexFlatIP`

- Stores normalized document embeddings
- Uses **inner product (cosine similarity)** for scoring
- In-memory — rebuilt on every startup from the `documents` list

### Local Generator — `google/flan-t5-small`

- Encoder-decoder seq2seq model (~80MB)
- Runs fully **offline**, no API key needed
- Takes the retrieved document as context and generates a concise answer
- Generation settings: `num_beams=4`, `max_new_tokens=100`, `min_new_tokens=5`

---

## Knowledge Base

The current documents cover Amazon customer support topics:

- Order tracking
- Return policy and return process
- Customer service contact
- Amazon Prime benefits
- Delayed packages
- Order cancellation
- Gift cards
- Payment method updates
- Account login

To add more documents, append strings to the `documents` list in the source file. The FAISS index rebuilds automatically on the next startup.

---

## Example Queries

| Query | Response Source |
|---|---|
| `how do I track my order` | vector DB + local T5 |
| `what does Amazon Prime include` | vector DB + local T5 |
| `how do I cancel an order` | vector DB + local T5 |
| `what is my return window` | vector DB + local T5 |
| `how do I contact support` | vector DB + local T5 |

Queries outside the knowledge base return:
```
HelpBot: Sorry, I couldn't find a relevant answer to your question.
```

---

## Key Functions

### `answer_with_local_t5(query, retrieved_doc)`
Builds a prompt with the retrieved document as context and generates an answer using the local flan-t5-small model.

### `rag_answer(query, top_k=1, threshold=0.3)`
Main RAG pipeline. Embeds the query, searches FAISS, and either generates an answer from the retrieved document or returns a not-found message based on the similarity score.

### `run_qa_bot()`
Interactive CLI loop. Accepts user input, calls `rag_answer`, and prints the response.

---

## Security

- `.env` is listed in `.gitignore` — never committed to version control
- No API keys or external network calls are required to run the application


