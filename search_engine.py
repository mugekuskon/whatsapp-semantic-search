"""
Hybrid semantic + keyword search over the WhatsApp ChromaDB collection.
Combines ChromaDB vector search with BM25 keyword search, merged via
Reciprocal Rank Fusion (RRF) for best results.
"""

import logging
import re
from datetime import datetime, timezone, timedelta
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from database_manager import MODEL_NAME, COLLECTION_NAME, NormalizedEmbeddingFunction

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DB_PATH = "./chroma_db"

# RRF constant — higher = smaller penalty for lower-ranked results (60 is standard)
RRF_K = 60

# Multilingual cross-encoder re-ranker — scores query-document relevance directly.
# Much more accurate than cosine similarity or BM25 alone.
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
_reranker: CrossEncoder | None = None

def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        log.info("Loading re-ranker model '%s'...", RERANKER_MODEL)
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, candidates: list[tuple[str, str, dict, float]], top_n: int) -> list[tuple[str, str, dict, float, float]]:
    """
    Re-rank candidates using a cross-encoder.

    Args:
        candidates: list of (id, doc, meta, dist) from hybrid search
        top_n:      how many to return after re-ranking

    Returns:
        list of (id, doc, meta, dist, rerank_score) sorted by rerank_score desc
    """
    reranker = _get_reranker()
    pairs = [(query, doc) for _, doc, _, _ in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(cid, doc, meta, dist, float(score)) for (cid, doc, meta, dist), score in ranked[:top_n]]


def get_db_collection() -> chromadb.Collection:
    """Connect to the persistent ChromaDB and return the whatsapp_chats collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = NormalizedEmbeddingFunction(model_name=MODEL_NAME)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    log.info("Connected to collection '%s' (count=%d)", COLLECTION_NAME, collection.count())
    return collection


def _tokenize(text: str) -> list[str]:
    """
    Lowercase, remove apostrophes, then split on
    non-alphanumeric characters for BM25. This prevents possessives/suffixes
    from being split into meaningless single-letter tokens.
    """
    # Strip all apostrophe variants so Kutay'a / Kutay'a → kutaya (one token)
    text = text.lower()
    text = text.replace("'", "").replace("\u2018", "").replace("\u2019", "")
    return re.findall(r"\w+", text)


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


def _exact_match_ids(query_text: str, all_ids: list[str], all_docs: list[str]) -> list[str]:
    """
    Return IDs of documents that contain ALL query tokens literally.
    These will be pinned to the top of results regardless of RRF score.
    """
    tokens = _tokenize(query_text)
    exact = []
    for doc_id, doc in zip(all_ids, all_docs):
        doc_lower = doc.lower()
        if all(tok in doc_lower for tok in tokens):
            exact.append(doc_id)
    return exact


def _load_all_documents(
    collection: chromadb.Collection,
    where: dict | None = None,
) -> tuple[list[str], list[str], list[dict]]:
    """Fetch documents from the collection, optionally filtered by a ChromaDB where clause."""
    kwargs = {"include": ["documents", "metadatas"]}
    if where:
        kwargs["where"] = where
    result = collection.get(**kwargs)
    return result["ids"], result["documents"], result["metadatas"]


def _months_ago_filter(months: int) -> dict:
    """Return a ChromaDB where filter for chunks newer than `months` ago."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=30 * months)
    return {"start_timestamp": {"$gte": int(cutoff.timestamp())}}


def hybrid_search(query_text: str, n_results: int = 5, months_ago: int | None = None, use_reranker: bool = True) -> dict:
    """
    Run hybrid search: semantic (ChromaDB) + keyword (BM25 over ALL docs), merged with RRF.

    BM25 runs over the full collection so exact keyword matches like names or
    brands are never missed due to poor semantic similarity.

    Returns a dict with the same structure as ChromaDB's query() output so
    print_search_results() works unchanged.
    """
    collection = get_db_collection()
    where = _months_ago_filter(months_ago) if months_ago else None

    # 1. Fetch all documents for BM25
    all_ids, all_docs, all_metas = _load_all_documents(collection, where=where)
    id_to_doc  = dict(zip(all_ids, all_docs))
    id_to_meta = dict(zip(all_ids, all_metas))

    if not all_ids:
        log.warning("No documents found for the given filter.")
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    # 2. BM25 keyword search over entire corpus
    tokenized_corpus = [_tokenize(doc) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(_tokenize(query_text))
    bm25_ranked_ids = [
        all_ids[i]
        for i in sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    ]

    # 3. Semantic search
    pool_size = min(len(all_ids), max(n_results * 10, 100))
    semantic_kwargs = dict(
        query_texts=[query_text],
        n_results=pool_size,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        semantic_kwargs["where"] = where
    semantic_results = collection.query(**semantic_kwargs)
    semantic_ids = semantic_results["ids"][0]
    id_to_dist = dict(zip(semantic_ids, semantic_results["distances"][0]))

    # 4. Reciprocal Rank Fusion
    rrf_ranked = _reciprocal_rank_fusion([
        semantic_ids[:pool_size],
        bm25_ranked_ids[:pool_size],
        bm25_ranked_ids[:pool_size],
        bm25_ranked_ids[:pool_size],
    ])

    # 5. Pin exact matches (docs containing ALL query tokens) to the top
    exact_ids = _exact_match_ids(query_text, all_ids, all_docs)
    exact_set = set(exact_ids)
    pinned   = [(doc_id, score) for doc_id, score in rrf_ranked if doc_id in exact_set]
    the_rest = [(doc_id, score) for doc_id, score in rrf_ranked if doc_id not in exact_set]

    # Fetch a larger candidate pool for re-ranker to work with
    rerank_pool_size = max(n_results * 3, 12)
    merged = (pinned + the_rest)[:rerank_pool_size]

    # Deduplicate by text — overlapping windows can produce near-identical chunks
    seen_texts: set[str] = set()
    candidates = []
    for doc_id, _ in merged:
        text = id_to_doc[doc_id]
        if text not in seen_texts:
            seen_texts.add(text)
            candidates.append((doc_id, text, id_to_meta[doc_id], id_to_dist.get(doc_id, 1.0)))

    # 6. Re-rank with cross-encoder for precise relevance scoring
    if use_reranker and candidates:
        reranked = rerank(query_text, candidates, top_n=n_results)
        final_ids    = [r[0] for r in reranked]
        final_docs   = [r[1] for r in reranked]
        final_metas  = [r[2] for r in reranked]
        final_dists  = [r[3] for r in reranked]
        final_scores = [r[4] for r in reranked]
    else:
        top = candidates[:n_results]
        final_ids    = [r[0] for r in top]
        final_docs   = [r[1] for r in top]
        final_metas  = [r[2] for r in top]
        final_dists  = [r[3] for r in top]
        final_scores = [None] * len(top)

    return {
        "ids":           [final_ids],
        "documents":     [final_docs],
        "metadatas":     [final_metas],
        "distances":     [final_dists],
        "rerank_scores": [final_scores],
    }


def print_search_results(results: dict) -> None:
    """Pretty-print search results to the terminal."""
    documents     = results.get("documents", [[]])[0]
    metadatas     = results.get("metadatas", [[]])[0]
    distances     = results.get("distances", [[]])[0]
    rerank_scores = results.get("rerank_scores", [[]])[0] or [None] * len(documents)

    if not documents:
        print("No results found.")
        return

    for i, (doc, meta, dist, rscore) in enumerate(zip(documents, metadatas, distances, rerank_scores), start=1):
        if rscore is not None:
            match_label = f"rerank score: {rscore:.3f}"
        elif dist >= 1.0:
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
    query = ""
    print(f"Query: {query}\n")
    results = hybrid_search(query, n_results=5)
    print_search_results(results)
