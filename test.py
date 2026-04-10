"""
Search quality test runner.
Reads queries from queries.json, runs each through hybrid_search(), and
prints a structured report showing whether keyword and semantic queries
return relevant results.
"""

import argparse
import json
import textwrap
from pathlib import Path
from search_engine import hybrid_search

QUERIES_FILE = Path(__file__).parent / "queries.json"
SEPARATOR = "=" * 60
DIVIDER = "-" * 60


def load_queries(query_type: str | None = None) -> list[dict]:
    with open(QUERIES_FILE, encoding="utf-8") as f:
        queries = json.load(f)
    if query_type:
        queries = [q for q in queries if q["type"] == query_type]
    return queries


def _keyword_found(results: dict, keyword: str) -> bool:
    """Check if the expected keyword appears in any returned document."""
    docs = results.get("documents", [[]])[0]
    return any(keyword.lower() in doc.lower() for doc in docs)


def _format_doc(text: str, width: int = 56, max_lines: int = 6) -> str:
    """Wrap and truncate a document for compact display."""
    lines = []
    for raw_line in text.splitlines():
        lines.extend(textwrap.wrap(raw_line, width) or [""])
    lines = lines[:max_lines]
    if len(text.splitlines()) > max_lines:
        lines.append("  [...]")
    return "\n  ".join(lines)


def run_test(query_def: dict, n_results: int) -> dict:
    """Run a single query and return a result summary dict."""
    query = query_def["query"]
    expected_keyword = query_def.get("expected_keyword")

    results = hybrid_search(query, n_results=n_results)
    docs      = results.get("documents", [[]])[0]
    metas     = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    keyword_check = None
    if expected_keyword:
        keyword_check = _keyword_found(results, expected_keyword)

    return {
        "query_def": query_def,
        "docs": docs,
        "metas": metas,
        "distances": distances,
        "keyword_check": keyword_check,
    }


def print_report(test_result: dict) -> None:
    qd       = test_result["query_def"]
    docs     = test_result["docs"]
    metas    = test_result["metas"]
    dists    = test_result["distances"]
    kw_check = test_result["keyword_check"]

    tag = "[DIRECT]" if qd["type"] == "direct" else "[SEMANTIC]"
    print(SEPARATOR)
    print(f"{tag}  ID: {qd['id']}")
    print(f"Query      : {qd['query']}")
    print(f"Description: {qd['description']}")

    if kw_check is not None:
        status = "PASS ✓" if kw_check else "FAIL ✗"
        print(f"Keyword '{qd['expected_keyword']}' in results: {status}")

    print()

    if not docs:
        print("  No results returned.")
        return

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        if dist >= 1.0:
            score_label = "keyword match"
        else:
            score_label = f"similarity {(1 - dist) * 100:.1f}%"

        print(f"  Result #{i} | {score_label}")
        print(f"  Time : {meta.get('start_datetime') or 'N/A'} → {meta.get('end_datetime') or 'N/A'}")
        print(f"  From : {meta.get('participants') or 'N/A'}")
        print(f"  Text :")
        print(f"  {_format_doc(doc)}")
        print(DIVIDER)


def print_summary(test_results: list[dict]) -> None:
    direct   = [r for r in test_results if r["query_def"]["type"] == "direct"]
    semantic = [r for r in test_results if r["query_def"]["type"] == "semantic"]

    direct_pass  = sum(1 for r in direct   if r["keyword_check"] is True)
    direct_total = len(direct)
    semantic_total = len(semantic)

    print(SEPARATOR)
    print("SUMMARY")
    print(SEPARATOR)
    print(f"  Direct  (keyword) tests : {direct_pass}/{direct_total} keyword checks passed")
    print(f"  Semantic tests run      : {semantic_total}")
    print()
    print("  (Semantic quality requires manual inspection of results above)")
    print(SEPARATOR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run search quality tests from queries.json")
    parser.add_argument("--n-results", type=int, default=3, help="Results per query (default: 3)")
    parser.add_argument("--type", choices=["direct", "semantic"], help="Only run this query type")
    args = parser.parse_args()

    queries = load_queries(query_type=args.type)
    if not queries:
        print(f"No queries found for type='{args.type}'")
        return

    print(SEPARATOR)
    print(f"  WhatsApp Search — Quality Test")
    print(f"  Queries: {len(queries)}  |  Results per query: {args.n_results}")
    print(SEPARATOR)

    test_results = []
    for qd in queries:
        result = run_test(qd, n_results=args.n_results)
        print_report(result)
        test_results.append(result)

    print_summary(test_results)


if __name__ == "__main__":
    main()
