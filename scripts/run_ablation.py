"""
Retrieval ablation: compare dense (FAISS-only), BM25-only, and hybrid (RRF)
with and without cross-encoder reranking.

Metrics reported per configuration:
  - recall@5, recall@10  (was an expected source chunk retrieved?)
  - MRR@10               (rank of the first expected chunk)
  - precision@1          (is the top result an expected chunk?)

Match criteria:
  - v2 schema (`expected_source_chunks` list on each QA item) → exact chunk_id match.
  - v1 schema (only `source_section` string) → substring match between
    `source_section` and the retrieved chunk's `doc_name` OR `section_header`.
    This is an approximation; see docs/RETRIEVAL_ABLATION.md.

Prerequisites:
  The FAISS + BM25 indexes must already exist under data/index/. Run
  POST /ingest (or the ingestion entry point) before this script.

Usage:
  python scripts/run_ablation.py                      # run all configurations
  python scripts/run_ablation.py --output out.json    # save detailed report
  python scripts/run_ablation.py --top-k 10           # change retrieval depth
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is importable when script is invoked directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import retriever  # noqa: E402
from evaluation.evaluator import load_golden_qa  # noqa: E402

logger = logging.getLogger(__name__)


# Configurations to benchmark.
CONFIGURATIONS = [
    ("dense", "vector", False),
    ("dense+rerank", "vector", True),
    ("bm25", "bm25", False),
    ("bm25+rerank", "bm25", True),
    ("hybrid", "hybrid", False),
    ("hybrid+rerank", "hybrid", True),
]


def _chunk_matches_expectation(
    chunk_id: str,
    doc_name: str,
    section_header: str,
    expected_chunk_ids: list[str] | None,
    expected_section: str | None,
) -> bool:
    """
    Return True if the retrieved chunk satisfies the expected-source criterion.

    v2 path: exact chunk_id match against the expected list (strict).
    v1 path: approximate match on doc/section substrings (lenient).
    """
    if expected_chunk_ids:
        return chunk_id in set(expected_chunk_ids)

    if not expected_section:
        return False

    # v1 approximation. The v1 `source_section` field packs both the document
    # and the section, e.g. "TRM Guidelines - User Access Management". We
    # split on " - " and check that BOTH parts appear (loose substring) in
    # the retrieved chunk's metadata.
    parts = [p.strip().lower() for p in expected_section.split("-")]
    doc_token = parts[0] if parts else ""
    section_token = parts[1] if len(parts) > 1 else ""

    doc_l = doc_name.lower()
    section_l = section_header.lower()

    doc_ok = (
        not doc_token
        or doc_token in doc_l
        or doc_l in doc_token
        or any(w in doc_l for w in doc_token.split() if len(w) > 3)
    )
    section_ok = (
        not section_token
        or section_token in section_l
        or any(w in section_l for w in section_token.split() if len(w) > 3)
    )
    return doc_ok and section_ok


def _evaluate_question(
    question: str,
    expected_chunk_ids: list[str] | None,
    expected_section: str | None,
    mode: str,
    rerank: bool,
    top_k: int,
) -> dict:
    """Run a single question and compute per-question hit data."""
    results = retriever.search(question, top_k=top_k, mode=mode, rerank=rerank)

    hits = [
        _chunk_matches_expectation(
            r.chunk_id,
            r.doc_name,
            r.section_header,
            expected_chunk_ids,
            expected_section,
        )
        for r in results
    ]

    first_hit_rank = next((i + 1 for i, h in enumerate(hits) if h), None)

    return {
        "question": question,
        "first_hit_rank": first_hit_rank,
        "hit_at_5": any(hits[:5]),
        "hit_at_10": any(hits[:10]),
        "hit_at_1": hits[0] if hits else False,
        "num_results": len(results),
    }


def _aggregate(per_question: list[dict]) -> dict:
    """Compute aggregate metrics across all questions."""
    n = len(per_question)
    if n == 0:
        return {
            "n": 0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr_at_10": 0.0,
            "precision_at_1": 0.0,
        }

    recall_5 = sum(1 for q in per_question if q["hit_at_5"]) / n
    recall_10 = sum(1 for q in per_question if q["hit_at_10"]) / n
    precision_1 = sum(1 for q in per_question if q["hit_at_1"]) / n

    reciprocal_ranks = []
    for q in per_question:
        r = q["first_hit_rank"]
        if r is not None and r <= 10:
            reciprocal_ranks.append(1.0 / r)
        else:
            reciprocal_ranks.append(0.0)
    mrr_10 = sum(reciprocal_ranks) / n

    return {
        "n": n,
        "recall_at_5": round(recall_5, 4),
        "recall_at_10": round(recall_10, 4),
        "mrr_at_10": round(mrr_10, 4),
        "precision_at_1": round(precision_1, 4),
    }


def run_ablation(qa_pairs: list[dict], top_k: int = 10) -> dict:
    """Run every configuration in CONFIGURATIONS and return a structured report."""
    report: dict = {"top_k": top_k, "n_qa": len(qa_pairs), "configurations": {}}

    for label, mode, rerank in CONFIGURATIONS:
        logger.info(
            "Evaluating configuration: %s (mode=%s, rerank=%s)",
            label,
            mode,
            rerank,
        )

        per_question = []
        for qa in qa_pairs:
            q = qa["question"]
            try:
                per_q = _evaluate_question(
                    question=q,
                    expected_chunk_ids=qa.get("expected_source_chunks"),
                    expected_section=qa.get("source_section"),
                    mode=mode,
                    rerank=rerank,
                    top_k=top_k,
                )
            except Exception as e:
                logger.error("Question failed (%s): %s", q[:60], e)
                per_q = {
                    "question": q,
                    "error": str(e),
                    "first_hit_rank": None,
                    "hit_at_5": False,
                    "hit_at_10": False,
                    "hit_at_1": False,
                    "num_results": 0,
                }
            per_question.append(per_q)

        report["configurations"][label] = {
            "mode": mode,
            "rerank": rerank,
            "aggregate": _aggregate(per_question),
            "per_question": per_question,
        }

    return report


def _print_summary(report: dict) -> None:
    """Print a compact summary table to stdout."""
    print("\n" + "=" * 84)
    print(f"Retrieval Ablation — {report['n_qa']} QA pairs, top_k={report['top_k']}")
    print("=" * 84)
    header = f"{'configuration':<16} {'recall@5':>10} {'recall@10':>10} {'MRR@10':>10} {'P@1':>8}"
    print(header)
    print("-" * 84)
    for label, cfg in report["configurations"].items():
        a = cfg["aggregate"]
        print(
            f"{label:<16} {a['recall_at_5']:>10.4f} {a['recall_at_10']:>10.4f} "
            f"{a['mrr_at_10']:>10.4f} {a['precision_at_1']:>8.4f}"
        )
    print("=" * 84)


def main() -> int:
    parser = argparse.ArgumentParser(description="MASquery retrieval ablation")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Retrieval depth per query (default: 10 so both recall@5 and recall@10 are covered)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="If set, write the full JSON report to this path",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Ensure index is loaded; fail cleanly with the required next step.
    if not retriever.is_index_loaded():
        try:
            retriever.load_index()
        except FileNotFoundError as e:
            logger.error(
                "Index not found. Run POST /ingest (or the ingestion entry "
                "point) to build the FAISS + BM25 indexes first. Details: %s",
                e,
            )
            return 1

    qa_pairs = load_golden_qa()
    report = run_ablation(qa_pairs, top_k=args.top_k)

    _print_summary(report)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nFull report saved to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
