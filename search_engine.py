"""
Hybrid semantic + keyword search over the WhatsApp ChromaDB collection.

Combines ChromaDB vector search with BM25 keyword search, merged via
Reciprocal Rank Fusion (RRF) for best results.
"""

import logging
import re
import chromadb
from rank_bm25 import BM25Okapi
from database_manager import MODEL_NAME, COLLECTION_NAME, NormalizedEmbeddingFunction

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DB_PATH = "./chroma_db"

# RRF constant — higher = smaller penalty for lower-ranked results (60 is standard)
RRF_K = 60


def get_db_collection() -> chromadb.Collection:
    """Connect to the persistent ChromaDB and return the whatsapp_chats collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = NormalizedEmbeddingFunction(model_name=MODEL_NAME)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    log.info("Connected to collection '%s' (count=%d)", COLLECTION_NAME, collection.count())
    return collection


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters for BM25."""
    return re.findall(r"\w+", text.lower())


def _reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """
    Merge multiple ranked lists of IDs into a single ranking using RRF.
    Returns list of (id, score) sorted by descending score.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _load_all_documents(collection: chromadb.Collection) -> tuple[list[str], list[str], list[dict]]:
    """Fetch every document from the collection for BM25 indexing."""
    result = collection.get(include=["documents", "metadatas"])
    return result["ids"], result["documents"], result["metadatas"]


def hybrid_search(query_text: str, n_results: int = 5) -> dict:
    """
    Run hybrid search: semantic (ChromaDB) + keyword (BM25 over ALL docs), merged with RRF.

    BM25 runs over the full collection so exact keyword matches like names or
    brands are never missed due to poor semantic similarity.

    Returns a dict with the same structure as ChromaDB's query() output so
    print_search_results() works unchanged.
    """
    collection = get_db_collection()

    # 1. Fetch all documents for BM25 
    all_ids, all_docs, all_metas = _load_all_documents(collection)
    id_to_doc  = dict(zip(all_ids, all_docs))
    id_to_meta = dict(zip(all_ids, all_metas))

    # 2. BM25 keyword search over entire corpus 
    tokenized_corpus = [_tokenize(doc) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(_tokenize(query_text))
    bm25_ranked_ids = [
        all_ids[i]
        for i in sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    ]

    # 3. Semantic search 
    pool_size = min(collection.count(), max(n_results * 10, 100))
    semantic_results = collection.query(
        query_texts=[query_text],
        n_results=pool_size,
        include=["documents", "metadatas", "distances"],
    )
    semantic_ids = semantic_results["ids"][0]
    id_to_dist = dict(zip(semantic_ids, semantic_results["distances"][0]))

    # 4. Reciprocal Rank Fusion
    # Give BM25 3x weight by repeating its ranked list — this ensures exact
    # keyword matches (names, brands, specific words) always surface to the top.
    merged = _reciprocal_rank_fusion([
        semantic_ids[:pool_size],
        bm25_ranked_ids[:pool_size],
        bm25_ranked_ids[:pool_size],
        bm25_ranked_ids[:pool_size],
    ])[:n_results]

    final_ids, final_docs, final_metas, final_dists = [], [], [], []
    for doc_id, _ in merged:
        final_ids.append(doc_id)
        final_docs.append(id_to_doc[doc_id])
        final_metas.append(id_to_meta[doc_id])
        final_dists.append(id_to_dist.get(doc_id, 1.0))  # 1.0 = max distance if not in semantic pool

    return {
        "ids":       [final_ids],
        "documents": [final_docs],
        "metadatas": [final_metas],
        "distances": [final_dists],
    }


def print_search_results(results: dict) -> None:
    """Pretty-print search results to the terminal."""
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        print("No results found.")
        return

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        if dist >= 1.0:
            match_label = "keyword match"
        else:
            match_label = f"similarity: {(1 - dist) * 100:.1f}%"
        print("--------------------------------------------------")
        print(f"Result #{i}  |  {match_label}")
        print(f"Time      : {meta.get('start_datetime') or 'N/A'} → {meta.get('end_datetime') or 'N/A'}")
        print(f"From      : {meta.get('participants') or 'N/A'}")
        print(f"Source    : {meta.get('source') or 'N/A'}")
        print()
        print(doc)

    print("--------------------------------------------------")


if __name__ == "__main__":
    query = "heinz ketçap"
    print(f"Query: {query}\n")
    results = hybrid_search(query, n_results=5)
    print_search_results(results)
