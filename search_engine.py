"""
Semantic search over the WhatsApp ChromaDB collection.
"""

import logging
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from database_manager import MODEL_NAME, COLLECTION_NAME

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DB_PATH = "./chroma_db"


def get_db_collection() -> chromadb.Collection:
    """
    Connect to the persistent ChromaDB and return the whatsapp_chats collection.
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    log.info("Connected to collection '%s' (count=%d)", COLLECTION_NAME, collection.count())
    return collection


def semantic_search(query_text: str, n_results: int = 5) -> dict:
    """
    Return the top n_results chunks most semantically similar to query_text.
    """
    collection = get_db_collection()
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results


def print_search_results(results: dict) -> None:
    """
    Pretty-print ChromaDB query results to the terminal.
    """
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        print("No results found.")
        return

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        print("--------------------------------------------------")
        print(f"Result #{i}  |  distance: {dist:.4f}")
        print(f"Time      : {meta.get('start_datetime') or 'N/A'} → {meta.get('end_datetime') or 'N/A'}")
        print(f"From      : {meta.get('participants') or 'N/A'}")
        print(f"Source    : {meta.get('source') or 'N/A'}")
        print()
        print(doc)
    print("--------------------------------------------------")


if __name__ == "__main__":
    query = ""
    print(f"Query: {query}\n")
    results = semantic_search(query, n_results=3)
    print_search_results(results)
