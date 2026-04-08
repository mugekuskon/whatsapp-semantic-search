"""
Full ingestion pipeline: parse → clean → chunk → embed → store in ChromaDB.

Usage:
    python ingest.py                  # ingest data/ into ./chroma_db (resets first)
    python ingest.py --no-reset       # append without wiping existing data
    python ingest.py --data-dir path  # use a different data folder
"""

import argparse
import logging
import chromadb
from data_processor import process_all_chats
from database_manager import init_db, ingest_data, COLLECTION_NAME

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def reset_collection(db_path: str) -> None:
    """Delete and recreate the collection so we start from a clean slate."""
    client = chromadb.PersistentClient(path=db_path)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        log.info("Deleted existing collection '%s'.", COLLECTION_NAME)
    else:
        log.info("No existing collection to delete.")


def run_pipeline(data_dir: str = "data", db_path: str = "./chroma_db", reset: bool = True) -> None:
    if reset:
        reset_collection(db_path)

    log.info("Starting data processing from '%s'...", data_dir)
    chunks = process_all_chats(data_dir=data_dir)

    if not chunks:
        log.error("No chunks produced — check that '%s' contains .txt files.", data_dir)
        return

    log.info("Total chunks to ingest: %d", len(chunks))

    collection = init_db(db_path=db_path)
    ingest_data(chunks, collection)

    sources = {c["source"] for c in chunks}
    print("\n" + "=" * 50)
    print("  Ingestion complete")
    print("=" * 50)
    print(f"  Sources   : {', '.join(sorted(sources))}")
    print(f"  Chunks    : {len(chunks)}")
    print(f"  DB total  : {collection.count()}")
    print(f"  DB path   : {db_path}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest WhatsApp chats into ChromaDB.")
    parser.add_argument("--data-dir", default="data", help="Folder containing .txt chat exports")
    parser.add_argument("--db-path", default="./chroma_db", help="ChromaDB storage path")
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Append to existing collection instead of wiping it first",
    )
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        db_path=args.db_path,
        reset=not args.no_reset,
    )
