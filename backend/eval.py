"""
eval.py

Quantitative evaluation of semantic search vs keyword search on a labeled
test set, instead of judging quality by eyeballing a handful of demo
queries. This is the kind of harness a real search/retrieval system would
be evaluated with before shipping a change.

Metrics reported, for both semantic and keyword search:
  - Top-1 accuracy   : was the correct FAQ the #1 result?
  - Top-3 accuracy   : was the correct FAQ anywhere in the top 3?
  - MRR              : Mean Reciprocal Rank -- averages 1/rank of the
                        correct result across all queries (1.0 is perfect,
                        0 means it was never found). Rewards ranking the
                        right answer higher, not just "getting it in
                        eventually somewhere in the results.
  - Avg latency (ms) : average time per query, since latency is a real
                        production concern, not just accuracy.

Results are also broken down by category ("paraphrase" vs "literal"),
which is the actual point of this whole project: keyword search should
do fine on literal queries and fail on paraphrased ones, while semantic
search should hold up on both.

Usage:
    python eval.py
    python eval.py --top-k 5 --queries eval_test_set.json --output eval_report.json
"""

import argparse
import json
import os
import time
from collections import defaultdict

from search import get_search_index

DEFAULT_TEST_SET = os.path.join(os.path.dirname(__file__), "eval_test_set.json")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "eval_report.json")


def load_test_set(path: str):
    with open(path, "r") as f:
        return json.load(f)


def rank_of_expected(results, expected_id):
    """
    Given a ranked list of result dicts (each with an 'id') and the id we
    expect to find, return its 1-indexed rank, or None if it isn't
    present in the results at all (e.g. keyword search found nothing).
    """
    for i, r in enumerate(results):
        if r["id"] == expected_id:
            return i + 1
    return None


def evaluate_method(search_index, test_cases, method: str, corpus_size: int):
    """
    Run every test query through one search method ('semantic' or
    'keyword'), and return per-query results plus aggregate metrics.
    We ask for the full corpus back (top_k = corpus_size) so we can see
    the TRUE rank of the correct answer, not just whether it fell inside
    whatever top_k a live search UI happens to use.
    """
    per_query = []
    latencies_ms = []

    for case in test_cases:
        query = case["query"]
        expected_id = case["expected_id"]

        start = time.perf_counter()
        if method == "semantic":
            results = search_index.semantic_search(query, top_k=corpus_size)
        else:
            results = search_index.keyword_search(query, top_k=corpus_size)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

        rank = rank_of_expected(results, expected_id)
        per_query.append({
            "query": query,
            "category": case["category"],
            "expected_id": expected_id,
            "rank": rank,  # None if not found at all
            "top_result_question": results[0]["question"] if results else None,
            "latency_ms": round(elapsed_ms, 3),
        })

    return per_query, latencies_ms


def compute_metrics(per_query, top_k_cutoff: int):
    """Aggregate top-1 / top-k accuracy and MRR from per-query rank data."""
    n = len(per_query)
    if n == 0:
        return {"top_1_accuracy": 0.0, "top_k_accuracy": 0.0, "mrr": 0.0, "n": 0}

    top_1 = sum(1 for q in per_query if q["rank"] == 1)
    top_k = sum(1 for q in per_query if q["rank"] is not None and q["rank"] <= top_k_cutoff)
    mrr = sum((1.0 / q["rank"]) if q["rank"] else 0.0 for q in per_query) / n

    return {
        "top_1_accuracy": top_1 / n,
        "top_k_accuracy": top_k / n,
        "mrr": mrr,
        "n": n,
    }


def metrics_by_category(per_query, top_k_cutoff: int):
    grouped = defaultdict(list)
    for q in per_query:
        grouped[q["category"]].append(q)
    return {cat: compute_metrics(rows, top_k_cutoff) for cat, rows in grouped.items()}


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def print_report(semantic_metrics, keyword_metrics, semantic_by_cat, keyword_by_cat,
                  semantic_latency_avg, keyword_latency_avg, top_k_cutoff, n_total):
    line = "=" * 78
    print(line)
    print("SEMANTIC SEARCH EVALUATION REPORT")
    print(line)
    print(f"Test set: {n_total} queries  |  Top-{top_k_cutoff} accuracy cutoff\n")

    header = f"{'Method':<18}{'Top-1 Acc':>12}{f'Top-{top_k_cutoff} Acc':>12}{'MRR':>10}{'Avg Latency':>15}"
    print(header)
    print("-" * len(header))
    print(f"{'Semantic search':<18}"
          f"{format_pct(semantic_metrics['top_1_accuracy']):>12}"
          f"{format_pct(semantic_metrics['top_k_accuracy']):>12}"
          f"{semantic_metrics['mrr']:>10.3f}"
          f"{semantic_latency_avg:>12.2f} ms")
    print(f"{'Keyword search':<18}"
          f"{format_pct(keyword_metrics['top_1_accuracy']):>12}"
          f"{format_pct(keyword_metrics['top_k_accuracy']):>12}"
          f"{keyword_metrics['mrr']:>10.3f}"
          f"{keyword_latency_avg:>12.2f} ms")

    print("\nBy category:")
    for cat in sorted(set(list(semantic_by_cat.keys()) + list(keyword_by_cat.keys()))):
        sm = semantic_by_cat.get(cat, {"top_1_accuracy": 0.0, "n": 0})
        km = keyword_by_cat.get(cat, {"top_1_accuracy": 0.0, "n": 0})
        label = {
            "paraphrase": "Paraphrase queries -- wording differs from the source text",
            "literal": "Literal queries -- query shares real words with the source text",
        }.get(cat, cat)
        print(f"  {label} (n={sm['n']}):")
        print(f"    Semantic top-1: {format_pct(sm['top_1_accuracy']):<8}"
              f"Keyword top-1: {format_pct(km['top_1_accuracy'])}")

    print(line)


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic vs keyword search.")
    parser.add_argument("--queries", default=DEFAULT_TEST_SET, help="Path to labeled test set JSON.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Where to write the full JSON report.")
    parser.add_argument("--top-k", type=int, default=3, help="Cutoff for 'top-k accuracy' (default 3).")
    args = parser.parse_args()

    test_cases = load_test_set(args.queries)
    search_index = get_search_index()
    corpus_size = len(search_index.documents)

    semantic_per_query, semantic_latencies = evaluate_method(
        search_index, test_cases, "semantic", corpus_size
    )
    keyword_per_query, keyword_latencies = evaluate_method(
        search_index, test_cases, "keyword", corpus_size
    )

    semantic_metrics = compute_metrics(semantic_per_query, args.top_k)
    keyword_metrics = compute_metrics(keyword_per_query, args.top_k)
    semantic_by_cat = metrics_by_category(semantic_per_query, args.top_k)
    keyword_by_cat = metrics_by_category(keyword_per_query, args.top_k)

    semantic_latency_avg = sum(semantic_latencies) / len(semantic_latencies)
    keyword_latency_avg = sum(keyword_latencies) / len(keyword_latencies)

    print_report(
        semantic_metrics, keyword_metrics,
        semantic_by_cat, keyword_by_cat,
        semantic_latency_avg, keyword_latency_avg,
        args.top_k, len(test_cases),
    )

    report = {
        "test_set_size": len(test_cases),
        "top_k_cutoff": args.top_k,
        "summary": {
            "semantic": semantic_metrics,
            "keyword": keyword_metrics,
        },
        "by_category": {
            "semantic": semantic_by_cat,
            "keyword": keyword_by_cat,
        },
        "latency_ms": {
            "semantic_avg": semantic_latency_avg,
            "keyword_avg": keyword_latency_avg,
        },
        "per_query": {
            "semantic": semantic_per_query,
            "keyword": keyword_per_query,
        },
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull per-query report written to {args.output}")


if __name__ == "__main__":
    main()
