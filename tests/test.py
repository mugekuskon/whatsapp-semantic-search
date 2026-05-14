"""
Search quality test runner.
Reads queries from queries.json, runs each through hybrid_search(), and
prints a structured report showing whether keyword and semantic queries
return relevant results.
"""

import argparse
import json
from pathlib import Path
from search_engine import hybrid_search

QUERIES_FILE = Path(__file__).parent / "queries.json"


def load_queries(query_type=None):
    queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    if query_type:
        queries = [q for q in queries if q["type"] == query_type]
    return queries


def run(query_def, n_results):
    results = hybrid_search(query_def["query"], n_results=n_results)
    docs      = results.get("documents", [[]])[0]
    metas     = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    scores    = results.get("rerank_scores", [[]])[0] or [None] * len(docs)

    kw = query_def.get("expected_keyword")
    kw_pass = any(kw.lower() in d.lower() for d in docs) if kw else None

    print(f"\n[{query_def['type'].upper()}] {query_def['id']} — {query_def['query']}")
    print(f"  {query_def['description']}")
    if kw_pass is not None:
        print(f"  Keyword '{kw}': {'PASS ✓' if kw_pass else 'FAIL ✗'}")

    for i, (doc, meta, dist, rs) in enumerate(zip(docs, metas, distances, scores), 1):
        if rs is not None:
            label = f"rerank {rs:.3f}"
        elif dist >= 1.0:
            label = "keyword match"
        else:
            label = f"{(1 - dist) * 100:.1f}%"

        date = meta.get("start_datetime", "")[:10]
        people = meta.get("participants", "")
        print(f"\n  #{i} [{label}] {date} | {people}")
        for line in doc.splitlines()[:4]:
            print(f"      {line}")
        if len(doc.splitlines()) > 4:
            print("      [...]")

    return kw_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-results", type=int, default=3)
    parser.add_argument("--type", choices=["direct", "semantic"])
    args = parser.parse_args()

    queries = load_queries(args.type)
    passed = total_kw = 0

    for q in queries:
        result = run(q, args.n_results)
        if result is not None:
            total_kw += 1
            if result:
                passed += 1

    print(f"\nKeyword tests: {passed}/{total_kw} passed")


if __name__ == "__main__":
    main()
