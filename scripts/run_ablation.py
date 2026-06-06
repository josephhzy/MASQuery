"""
Retrieval ablation: compare dense (FAISS-only), BM25-only, and hybrid (RRF)
with and without cross-encoder reranking.

Metrics reported per configuration:
  - recall@5, recall@10  (was an expected source chunk retrieved?)
  - MRR@10               (rank of the first expected chunk)
  - precision@1          (is the top result an expected chunk?)

Match criteria:
  - v2 schema (`expected_source_chunks` list on each QA item) → exact chunk_id match.
  - v1 schema (only `source_section` string): splits source_section on " - " into
    a doc token (text before the dash) and a section token (text after); a hit
    requires the doc token to loosely match doc_name AND the section token to loosely
    match section_header.
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
    # split on " - " into a doc token and a section token, then require the
    # retrieved chunk's doc_name to start with the doc token AND its
    # section_header to loosely match the section token.
    parts = [p.strip().lower() for p in expected_section.split(" - ", 1)]
    doc_token = parts[0] if parts else ""
    section_token = parts[1] if len(parts) > 1 else ""

    section_l = section_header.lower()

    doc_ok = not doc_token or doc_name.lower().replace("_", " ").replace("-", " ").startswith(doc_token)
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


def _split_qa(qa_pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split a golden QA list into answerable and adversarial subsets.

    Adversarial pairs carry ``should_refuse: true`` (and have no expected
    corpus chunk to retrieve, by construction). Mixing them into retrieval
    metrics blurs "should retrieve" with "should refuse" into one number;
    the ablation reports the two separately.
    """
    answerable = [qa for qa in qa_pairs if not qa.get("should_refuse")]
    adversarial = [qa for qa in qa_pairs if qa.get("should_refuse")]
    return answerable, adversarial


def _retrieval_signal_for_adversarial(question: str, mode: str, rerank: bool, top_k: int) -> dict:
    """Per-adversarial descriptor: top retrieval score + count.

    For ``should_refuse`` pairs there is no expected chunk to score against,
    so retrieval recall/MRR/P@1 are not meaningful. What IS meaningful for
    the refusal pipeline is the **score the gate would see** — i.e. the top
    relevance score returned by the retriever. We expose it here so the
    refusal-threshold sweep (see docs/REFUSAL_THRESHOLD.md) can read this
    JSON directly without re-running retrieval.
    """
    results = retriever.search(question, top_k=top_k, mode=mode, rerank=rerank)
    top_score = results[0].relevance_score if results else None
    return {
        "question": question,
        "num_results": len(results),
        "top_score": round(top_score, 4) if top_score is not None else None,
    }


def run_ablation(qa_pairs: list[dict], top_k: int = 10) -> dict:
    """Run every configuration in CONFIGURATIONS and return a structured report.

    Retrieval metrics (recall, MRR, P@1) are computed on the **answerable**
    subset only. Adversarial pairs (``should_refuse: true``) are reported in
    a parallel block as descriptive top-score data — they are inputs to the
    refusal-threshold work, not retrieval-quality measurements.
    """
    answerable, adversarial = _split_qa(qa_pairs)
    report: dict = {
        "top_k": top_k,
        "n_qa": len(qa_pairs),
        "n_answerable": len(answerable),
        "n_adversarial": len(adversarial),
        "configurations": {},
    }

    for label, mode, rerank in CONFIGURATIONS:
        logger.info(
            "Evaluating configuration: %s (mode=%s, rerank=%s) — %d answerable, %d adversarial",
            label,
            mode,
            rerank,
            len(answerable),
            len(adversarial),
        )

        per_question = []
        for qa in answerable:
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

        adversarial_signals: list[dict] = []
        for qa in adversarial:
            try:
                sig = _retrieval_signal_for_adversarial(qa["question"], mode=mode, rerank=rerank, top_k=top_k)
                sig["category"] = qa.get("category")
                adversarial_signals.append(sig)
            except Exception as e:
                logger.error("Adversarial question failed (%s): %s", qa["question"][:60], e)
                adversarial_signals.append(
                    {
                        "question": qa["question"],
                        "category": qa.get("category"),
                        "error": str(e),
                        "num_results": 0,
                        "top_score": None,
                    }
                )

        report["configurations"][label] = {
            "mode": mode,
            "rerank": rerank,
            "aggregate": _aggregate(per_question),
            "per_question": per_question,
            "adversarial": adversarial_signals,
        }

    return report


def _print_summary(report: dict) -> None:
    """Print a compact summary table to stdout."""
    n_ans = report.get("n_answerable", report["n_qa"])
    n_adv = report.get("n_adversarial", 0)

    print("\n" + "=" * 84)
    print(
        f"Retrieval Ablation — {report['n_qa']} pairs "
        f"({n_ans} answerable + {n_adv} adversarial), top_k={report['top_k']}"
    )
    print("=" * 84)
    print("Retrieval metrics (answerable subset only):")
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

    if n_adv > 0:
        print("\nAdversarial top-score distribution (refusal-pipeline signal, not retrieval quality):")
        adv_header = f"{'configuration':<16} {'n':>4} {'min':>8} {'mean':>8} {'max':>8}"
        print(adv_header)
        print("-" * 50)
        for label, cfg in report["configurations"].items():
            scores = [s["top_score"] for s in cfg.get("adversarial", []) if s.get("top_score") is not None]
            if not scores:
                print(f"{label:<16} {0:>4} {'—':>8} {'—':>8} {'—':>8}")
                continue
            print(
                f"{label:<16} {len(scores):>4} {min(scores):>8.4f} "
                f"{sum(scores) / len(scores):>8.4f} {max(scores):>8.4f}"
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
