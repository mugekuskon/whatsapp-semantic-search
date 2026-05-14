"""
ChromaDB setup and data ingestion for the WhatsApp semantic search pipeline.
Uses the same multilingual model as embeddings.py so that query vectors and
stored vectors are always produced by the same weights.
"""
import uuid
import logging
import json
from pathlib import Path
from datetime import datetime, timezone
import chromadb
from chromadb.api.types import Documents, Embeddings
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL
from utils import normalize_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = EMBEDDING_MODEL
COLLECTION_NAME = "whatsapp_chats"
DB_PATH = "./chroma_db"


class NormalizedEmbeddingFunction:
    """
    ChromaDB-compatible embedding function that applies text normalization
    before encoding, ensuring ingest and query vectors are consistent.
    """
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)

    def name(self) -> str:
        return "NormalizedEmbeddingFunction"

    def _encode(self, texts: Documents) -> Embeddings:
        normalized = [normalize_text(t) for t in texts]
        return self.model.encode(normalized).tolist()

    def __call__(self, input: Documents) -> Embeddings:
        return self._encode(input)

    def embed_documents(self, input: Documents) -> Embeddings:
        return self._encode(input)

    def embed_query(self, input: Documents) -> Embeddings:
        return self._encode(input)


def init_db(db_path: str = "./chroma_db") -> chromadb.Collection:
    """
    Initialise a persistent ChromaDB client and return the collection.
    """
    client = chromadb.PersistentClient(path=db_path)
    log.info("ChromaDB client initialised at '%s'", db_path)

    embedding_fn = NormalizedEmbeddingFunction(model_name=MODEL_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    log.info("Collection '%s' ready (count=%d)", COLLECTION_NAME, collection.count())
    return collection


def _safe_str(value) -> str:
    """
    Convert a value to string; return empty string for None.
    """
    return "" if value is None else str(value)


def _to_timestamp(dt) -> int:
    """Convert a datetime (or ISO string) to a UTC Unix timestamp integer."""
    if dt is None:
        return 0
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def ingest_data(
    chunks_list: list[dict],
    collection: chromadb.Collection,
    batch_size: int = 100,
) -> None:
    """
    Upsert a list of chunk dicts into the ChromaDB collection in batches.

    ChromaDB metadata values must be str, int, or float — this function
    handles the two tricky fields:
      - participants  : list[str]  → joined as a comma-separated string
      - start/end_datetime: datetime | None → converted to ISO string or empty string

    batch_size controls how many chunks are sent per upsert call, preventing
    memory issues on large chat exports.
    """
    if not chunks_list:
        log.warning("ingest_data called with an empty list — nothing to do.")
        return

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for chunk in chunks_list:
        documents.append(chunk.get("chunk_text", ""))

        participants_raw = chunk.get("participants", [])
        participants_str = ", ".join(participants_raw) if participants_raw else ""

        start_dt = chunk.get("start_datetime")
        end_dt   = chunk.get("end_datetime")

        metadatas.append({
            "source": _safe_str(chunk.get("source")),
            "participants": participants_str,
            "start_datetime": _safe_str(start_dt),
            "end_datetime": _safe_str(end_dt),
            "start_timestamp": _to_timestamp(start_dt),
            "end_timestamp": _to_timestamp(end_dt),
        })

        ids.append(str(uuid.uuid4()))

    total = len(documents)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.upsert(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        log.info("Upserted batch %d–%d / %d", start + 1, end, total)

    log.info("Ingestion complete. Collection total: %d", collection.count())


# Quick Test: For testing purposes, we can run this module directly to ingest some dummy data and verify the setup.

if __name__ == "__main__":

    dummy_path = Path(__file__).parent / "tests" / "dummy_chunks.json"
    dummy_chunks = json.loads(dummy_path.read_text(encoding="utf-8"))

    collection = init_db()
    ingest_data(dummy_chunks, collection)
    print(f"\nTotal documents in collection: {collection.count()}")

    results = collection.query(query_texts=["kafe buluşma"], n_results=1)
    print(f"Top result for 'kafe buluşma': {results['documents'][0][0][:80]}...")
