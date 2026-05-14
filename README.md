# WhatsApp Semantic Search

A fully local Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions about your WhatsApp chat history and receive grounded answers in Turkish. Built with hybrid search (BM25 + semantic vectors + cross-encoder reranking), a local LLM via Ollama and a Gradio web interface.


## Architecture

```
WhatsApp .txt exports
        │
        ▼
  chat_parser.py        parse Android / iOS export formats, join multi-line messages
        │
        ▼
  data_processor.py     clean noise, normalize Turkish text, chunk into sliding windows
        │
        ▼
  database_manager.py   embed with sentence-transformers, store in ChromaDB
        │
        ▼
  search_engine.py      BM25 + semantic → RRF fusion → exact-match pin → reranker
        │
        ▼
  rag.py                build grounded prompt, generate answer via Ollama
        │
        ▼
  app.py                Gradio web UI — Ask tab and Search tab
```


### How Search Works

Every query goes through six stages:

1. **BM25 keyword search** over the full corpus, finds exact token matches such as names, brands, and specific terms.
2. **Semantic vector search** via ChromaDB, finds conceptually related chunks even when no words overlap.
3. **Reciprocal Rank Fusion (RRF)** merges both ranked lists into one; BM25 is weighted 3× because Turkish agglutination makes semantic models less reliable for inflected word forms.
4. **Exact-match pinning** chunks containing all query tokens are pinned to the top regardless of RRF score.
5. **Deduplication** overlapping windows produce near-identical chunks, duplicates are removed before reranking.
6. **Cross-encoder reranking** sees query and document together, outputs a precise relevance score that cosine similarity alone cannot match.

## Features

* Fully local.
* Hybrid retrieval: BM25 + cosine vector search + Reciprocal Rank Fusion.
* Cross-encoder reranking for high-precision results.
* Turkish-aware cleaning: system message filtering, laughter normalization, URL semantic labeling, locale-specific date parsing.
* Multi-format WhatsApp parser: Android (US), iOS (US), iOS (EU/TR).
* Multi-chat ingestion: multiple exports ingested at once, results tagged by source.
* Gradio web UI with Ask (RAG) and Search (raw retrieval) tabs.
* Grounded generation: the model cites sources and refuses to answer when context is insufficient.


## Tech Stack

| Library | Role |
|---|---|
| `sentence-transformers` | Multilingual text embeddings |
| `chromadb` | Local vector database |
| `rank-bm25` | BM25Okapi keyword search |
| `ollama` | Local LLM inference |
| `gradio` | Web UI |


## Models

| Role | Model | Notes |
|---|---|---|
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | 384-dim, 50+ languages including Turkish |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Multilingual cross-encoder, CPU-friendly |
| LLM | `aya-expanse:8b` via Ollama | Multilingual, strong Turkish, ~8GB RAM |


## Prerequisites

* Python 3.11+
* [Ollama](https://ollama.com) installed and running
* ~10GB free disk space (models + vector store)


## Installation

```bash
git clone <your-repo-url>
cd whatsapp-search

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

ollama pull aya-expanse:8b
```


## Usage

### 1. Export your WhatsApp chats

Place your chat `.txt` files in the `data/` folder. The filename (without extension) becomes the `source` label in results, rename them to something readable like `friend_group.txt`.

### 2. Ingest

```bash
python ingest.py
```

```
==================================================
  Ingestion complete
==================================================
  Sources   : friend_group, group_2, group_3
  Chunks    : 2821
  DB total  : 2821
  DB path   : ./chroma_db
==================================================
```

Additional options:

```bash
python ingest.py --data-dir path/to/chats   # custom data folder
python ingest.py --db-path  path/to/db      # custom ChromaDB path
python ingest.py --no-reset                 # append without wiping existing data
```

Re-ingestion is only needed when you add new chat files or change `EMBEDDING_MODEL`.

### 3. Launch the UI

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860).


**Ask tab**: type a question in Turkish. The pipeline retrieves the most relevant chunks, builds a grounded prompt, and streams a Turkish answer. If the information is not in the chat history, the model responds: *"Bu bilgi sohbet geçmişinde bulunamadı."*

**Search tab**: see the raw retrieved chunks with rerank scores, timestamps and participant names. Use this to verify retrieval quality independently of the LLM.


## Data Cleaning

`data_processor.py` applies the following before chunking:

| Step | What it does |
|---|---|
| System message filtering | Drops WhatsApp-generated notifications in Turkish (join/leave/media omitted) |
| Emoji-only removal | Drops messages with no textual content |
| Laughter normalization | Replaces keyboard-smash laughter (`DKDKDK`, `SJSJSJ`) with `[kahkaha]` |
| URL labeling | Replaces raw URLs with semantic labels (`[youtube videosu]`, `[konum paylaşıldı]`) |
| Document labeling | Replaces file attachment lines with `[belge paylaşıldı: filename]` |

All patterns and domain mappings are configured in `config.py`.


## Project Structure

```
whatsapp-search/
├── app.py                  Gradio web interface
├── ingest.py               CLI: parse → clean → chunk → embed → store
├── chat_parser.py          WhatsApp .txt parser (all 3 export formats)
├── data_processor.py       Cleaning, laughter normalization, URL labeling, chunking
├── database_manager.py     ChromaDB setup and batched ingestion
├── embeddings.py           Standalone embedding utility (not used by main pipeline)
├── search_engine.py        Hybrid BM25 + semantic + reranker search
├── rag.py                  RAG pipeline: retrieve → prompt → generate
├── utils.py                Shared text normalization used across the pipeline
├── config.py               Central config: model, patterns, URL mappings
├── requirements.txt        Pinned dependencies
├── data/                   WhatsApp .txt exports (gitignored)
├── chroma_db/              ChromaDB vector store (auto-created, gitignored)
└── tests/
    ├── test.py             Search quality test runner
    ├── queries.json        Test queries (direct + semantic)
```


## Configuration

All tunable settings live in `config.py`:

```python
# Embedding model: change here, then delete chroma_db/ and re-run ingest.py.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Turkish WhatsApp system message patterns to filter out.
WHATSAPP_SYSTEM_PATTERNS = [...]

URL_DOMAIN_LABELS = [...]
```

Chunking parameters in `data_processor.py`:

```python
window_size     # messages per chunk
overlap       # messages shared between adjacent chunks
```

Retrieval parameters in `search_engine.py`:

```python
RRF_K         # RRF constant
RERANKER_MODEL
```

Generation parameters in `rag.py`:

```python
OLLAMA_MODEL 
N_RESULTS   
```


## Testing

```bash
cd tests

python test.py                   # run all queries
python test.py --type direct     # keyword tests with automatic PASS/FAIL
python test.py --type semantic   # conceptual queries for manual inspection
python test.py --n-results 5     # control results shown per query
```

**Direct tests** check whether an `expected_keyword` appears in the top results, useful for catching regressions after parameter changes. Run these after every change to chunking or config settings.

**Semantic tests** have no automatic verdict. Inspect results manually to judge topical relevance.







