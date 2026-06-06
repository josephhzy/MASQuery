"""
Evaluation runner for MAS RegQ&A.

Loads golden QA pairs from golden_qa.yaml, runs each question through the
retrieval + generation pipeline, and scores with RAGAS-inspired metrics.

Usage:
    python -m evaluation.evaluator                 # evaluate all QA pairs
    python -m evaluation.evaluator --mode vector   # evaluate with vector-only search
    python -m evaluation.evaluator --no-rerank     # disable cross-encoder reranking
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import yaml

from evaluation.metrics import answer_correctness, answer_faithfulness, context_relevance

logger = logging.getLogger(__name__)

GOLDEN_QA_PATH = Path(__file__).parent / "golden_qa.yaml"


def load_golden_qa(path: Optional[Path] = None) -> list[dict]:
    """
    Load golden QA pairs from YAML file.

    Expected format:
    ```yaml
    - question: "..."
      expected_answer: "..."
      source_section: "..."
    ```
    """
    qa_path = path or GOLDEN_QA_PATH
    if not qa_path.exists():
        raise FileNotFoundError(f"Golden QA file not found: {qa_path}")

    with open(qa_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError("Golden QA file must be a YAML list of question/answer pairs")

    return data


def evaluate_single(
    question: str,
    expected_answer: str,
    search_mode: str = "vector",
    rerank: bool = True,
    top_k: int = 5,
) -> dict:
    """
    Evaluate a single QA pair through the full pipeline.

    Returns:
        Dict with question, generated_answer, expected_answer, contexts,
        and metric scores.
    """
    import retriever
    from generator import generate_answer

    # Retrieve
    results = retriever.search(question, top_k=top_k, mode=search_mode, rerank=rerank)
    contexts = [r.text for r in results]

    # Generate (pass mode/rerank so the Layer-1 score gate runs on the right scale)
    gen_result = generate_answer(question, results, search_mode=search_mode, rerank=rerank)

    # Score
    cr = context_relevance(question, contexts)
    af = answer_faithfulness(gen_result.answer, contexts)
    ac = answer_correctness(gen_result.answer, expected_answer)

    return {
        "question": question,
        "expected_answer": expected_answer,
        "generated_answer": gen_result.answer,
        "confidence": gen_result.confidence,
        "is_answerable": gen_result.is_answerable,
        "num_contexts": len(contexts),
        "metrics": {
            "context_relevance": round(cr, 4),
            "answer_faithfulness": round(af, 4),
            "answer_correctness": round(ac, 4),
        },
    }


def evaluate_refusal(
    question: str,
    search_mode: str = "vector",
    rerank: bool = True,
    top_k: int = 5,
) -> dict:
    """Run an adversarial (``should_refuse: true``) question through the pipeline.

    For these pairs, faithfulness/correctness/context-relevance are not the
    right metric — the right question is "did the system decline to answer?"
    Returns ``is_answerable`` and the categories that drove the decision so
    refusal recall can be aggregated separately.
    """
    import retriever
    from generator import generate_answer

    results = retriever.search(question, top_k=top_k, mode=search_mode, rerank=rerank)
    gen_result = generate_answer(question, results, search_mode=search_mode, rerank=rerank)

    return {
        "question": question,
        "is_answerable": gen_result.is_answerable,
        "refused": not gen_result.is_answerable,
        "confidence": gen_result.confidence,
        "num_contexts": len(results),
        "top_score": round(results[0].relevance_score, 4) if results else None,
        "generated_answer": gen_result.answer,
    }


def evaluate_all(
    qa_pairs: Optional[list[dict]] = None,
    search_mode: str = "vector",
    rerank: bool = True,
    top_k: int = 5,
) -> dict:
    """
    Evaluate all golden QA pairs and compute aggregate scores.

    The QA list is split into:
      - **answerable** (no ``should_refuse`` flag): graded on context relevance,
        answer faithfulness, and answer correctness (heuristic proxies using
        embedding similarity and token overlap — not RAGAS).
      - **adversarial** (``should_refuse: true``): graded only on refusal
        recall — did the system decline rather than fabricate? Mixing the two
        into a single average blurs "should answer" with "should refuse" into
        one number that means neither.

    Returns:
        Dict with per-question results and aggregate metrics for each subset.
    """
    if qa_pairs is None:
        qa_pairs = load_golden_qa()

    answerable_pairs = [qa for qa in qa_pairs if not qa.get("should_refuse")]
    adversarial_pairs = [qa for qa in qa_pairs if qa.get("should_refuse")]

    answerable_results: list[dict] = []
    for i, qa in enumerate(answerable_pairs, 1):
        question = qa["question"]
        expected = qa["expected_answer"]
        logger.info(f"Answerable [{i}/{len(answerable_pairs)}]: {question[:60]}...")

        try:
            result = evaluate_single(
                question,
                expected,
                search_mode=search_mode,
                rerank=rerank,
                top_k=top_k,
            )
            answerable_results.append(result)
        except Exception as e:
            logger.error(f"Failed to evaluate answerable question {i}: {e}")
            answerable_results.append(
                {
                    "question": question,
                    "expected_answer": expected,
                    "error": str(e),
                    "metrics": {
                        "context_relevance": 0.0,
                        "answer_faithfulness": 0.0,
                        "answer_correctness": 0.0,
                    },
                }
            )

    adversarial_results: list[dict] = []
    for i, qa in enumerate(adversarial_pairs, 1):
        question = qa["question"]
        category = qa.get("category")
        logger.info(f"Adversarial [{i}/{len(adversarial_pairs)}, {category}]: {question[:60]}...")
        try:
            result = evaluate_refusal(
                question,
                search_mode=search_mode,
                rerank=rerank,
                top_k=top_k,
            )
            result["category"] = category
            adversarial_results.append(result)
        except Exception as e:
            logger.error(f"Failed to evaluate adversarial question {i}: {e}")
            adversarial_results.append(
                {
                    "question": question,
                    "category": category,
                    "error": str(e),
                    "refused": False,
                    "is_answerable": True,
                }
            )

    # Aggregate — answerable subset (RAGAS-style metrics)
    n_ans = len([r for r in answerable_results if "metrics" in r])
    if n_ans:
        ans_agg = {
            "context_relevance": sum(r["metrics"]["context_relevance"] for r in answerable_results if "metrics" in r)
            / n_ans,
            "answer_faithfulness": sum(
                r["metrics"]["answer_faithfulness"] for r in answerable_results if "metrics" in r
            )
            / n_ans,
            "answer_correctness": sum(r["metrics"]["answer_correctness"] for r in answerable_results if "metrics" in r)
            / n_ans,
        }
        ans_agg["overall"] = round(sum(ans_agg.values()) / 3, 4)
        answerable_aggregate = {k: round(v, 4) for k, v in ans_agg.items()}
    else:
        answerable_aggregate = {
            "context_relevance": 0.0,
            "answer_faithfulness": 0.0,
            "answer_correctness": 0.0,
            "overall": 0.0,
        }

    # Aggregate — adversarial subset (refusal recall + per-category breakdown)
    n_adv = len(adversarial_results)
    refused = sum(1 for r in adversarial_results if r.get("refused"))
    refusal_recall = round(refused / n_adv, 4) if n_adv else 0.0

    by_category: dict[str, dict] = {}
    for r in adversarial_results:
        cat = r.get("category") or "uncategorised"
        bucket = by_category.setdefault(cat, {"n": 0, "refused": 0})
        bucket["n"] += 1
        if r.get("refused"):
            bucket["refused"] += 1
    for bucket in by_category.values():
        bucket["refusal_recall"] = round(bucket["refused"] / bucket["n"], 4) if bucket["n"] else 0.0

    adversarial_aggregate = {
        "refusal_recall": refusal_recall,
        "refused": refused,
        "n": n_adv,
        "by_category": by_category,
    }

    return {
        "config": {
            "search_mode": search_mode,
            "rerank": rerank,
            "top_k": top_k,
            "num_qa_pairs": len(qa_pairs),
            "num_answerable": len(answerable_pairs),
            "num_adversarial": len(adversarial_pairs),
        },
        "answerable": {
            "aggregate_metrics": answerable_aggregate,
            "per_question": answerable_results,
        },
        "adversarial": {
            "aggregate_metrics": adversarial_aggregate,
            "per_question": adversarial_results,
        },
    }


def main():
    """CLI entrypoint for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate MAS RegQ&A pipeline")
    parser.add_argument("--mode", default="vector", choices=["hybrid", "vector", "bm25"])
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Ensure index is loaded
    import retriever

    if not retriever.is_index_loaded():
        logger.info("Loading index...")
        retriever.load_index()

    report = evaluate_all(
        search_mode=args.mode,
        rerank=not args.no_rerank,
        top_k=args.top_k,
    )

    cfg = report["config"]
    print("\n" + "=" * 60)
    print("MAS RegQ&A Evaluation Report")
    print("=" * 60)
    print(f"Search mode: {args.mode} | Rerank: {not args.no_rerank} | Top-K: {args.top_k}")
    print(
        f"QA pairs: {cfg['num_qa_pairs']} ({cfg['num_answerable']} answerable + {cfg['num_adversarial']} adversarial)"
    )

    print("-" * 60)
    print("Answerable subset — heuristic metrics (embedding similarity + token overlap; not RAGAS):")
    ans_agg = report["answerable"]["aggregate_metrics"]
    print(f"  Context Relevance:   {ans_agg['context_relevance']:.4f}")
    print(f"  Answer Faithfulness: {ans_agg['answer_faithfulness']:.4f}")
    print(f"  Answer Correctness:  {ans_agg['answer_correctness']:.4f}")
    print(f"  Overall:             {ans_agg['overall']:.4f}")

    print("-" * 60)
    print("Adversarial subset — refusal recall:")
    adv_agg = report["adversarial"]["aggregate_metrics"]
    print(f"  Refused / total:     {adv_agg['refused']} / {adv_agg['n']}")
    print(f"  Refusal Recall:      {adv_agg['refusal_recall']:.4f}")
    if adv_agg.get("by_category"):
        print("  By category:")
        for cat, bucket in adv_agg["by_category"].items():
            print(f"    {cat:<32} {bucket['refused']}/{bucket['n']} (recall={bucket['refusal_recall']:.4f})")
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
