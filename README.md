# WhatsApp Semantic Search

A local, offline RAG pipeline for semantic search over WhatsApp chat exports. Runs entirely on your machine.


## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Place your exported WhatsApp `.txt` files in a `data/` directory.

## Status

- [x] Chat parsing (multi-format)
- [x] Data cleaning and chunking
- [x] Text normalization and embedding
- [x] Vector storage (ChromaDB)
- [ ] Semantic search interface
