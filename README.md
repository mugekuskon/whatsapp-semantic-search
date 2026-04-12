# WhatsApp Semantic Search

A fully local, offline RAG pipeline for searching and querying WhatsApp chat exports using natural language.

## Architecture

```
.txt exports
    │
    ▼
chat_parser.py       — parse Android / iOS export formats.
    │
    ▼
data_processor.py    — clean, chunk (sliding window), normalize.
    │
    ▼
database_manager.py  — embed + store in ChromaDB.
    │
    ▼
search_engine.py     — hybrid search: BM25 + semantic + re-ranker.
    │
    ▼
rag.py               — augment prompt + generate answer via Ollama.
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Ollama from https://ollama.com, then pull the LLM:
ollama pull aya-expanse:8b
```

Place your exported WhatsApp `.txt` files in `data/`.

## Usage

**Step 1 — Ingest your chats into the vector database:**
```bash
python ingest.py
```

**Step 2 — Search:**
```bash
python search_engine.py      
```

**Step 3 — Ask questions (RAG):**
```bash
python rag.py                 
```

## How Search Works

Queries go through three stages:

1. **BM25** — keyword search over all chunks (finds exact word matches, names, brands).
2. **Semantic** — vector similarity search via ChromaDB (finds conceptually related chunks).
3. **Re-ranker** — a cross-encoder (`mmarco-mMiniLMv2-L12-H384-v1`) re-scores the merged candidates by true query-document relevance.

BM25 is weighted 3× over semantic in the RRF merge. Chunks containing all query tokens are pinned to the top before re-ranking.

## Models

| Role | Model |
|---|---|
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` — multilingual, supports Turkish. |
| Re-ranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` — multilingual cross-encoder. |
| LLM | `aya-expanse:8b` via Ollama — multilingual, supports Turkish.|

To change any model, edit `config.py` (embedding model) or `rag.py` (LLM).

## Data Cleaning

`data_processor.py` applies the following before chunking:

- Drops WhatsApp system messages (join/leave/media omitted).
- Drops emoji-only messages.
- Replaces keyboard-smash laughter (`DKDKDKDKDK`, `HDJDJDJDJ`) with `[kahkaha]`.
- Labels URLs semantically (`[youtube videosu]`, `[konum paylaşıldı]`, etc.).
- Labels document attachments.

Patterns are configurable in `config.py`.

## Chunking Strategy

- Sliding window: 8 messages per chunk, step 6.
- Hard cap: 500 chars per chunk, split at line boundaries.
- Long single-line messages (e.g. voice-to-text notes) are force-split at word boundaries.

## Testing Search Quality

Add queries to `queries.json`, then run:

```bash
python test.py                    
python test.py --type direct      
python test.py --type semantic    
python test.py --n-results 5      
```

Direct queries check automatically whether `expected_keyword` appears in results (PASS/FAIL). Semantic queries require manual inspection.

## Project Structure

```
ingest.py              — orchestrates the full parse → embed → store pipeline.
chat_parser.py         — WhatsApp .txt parser (Android + iOS formats).
data_processor.py      — cleaning, chunking, laughter normalization.
database_manager.py    — ChromaDB setup and batch ingestion.
embeddings.py          — standalone embedding utility (not used by pipeline).
search_engine.py       — hybrid BM25 + semantic + re-ranker search.
rag.py                 — RAG pipeline: retrieve → augment → generate.
test.py                — search quality test runner.
```
