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

    # Generate
    gen_result = generate_answer(question, results)

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


def evaluate_all(
    qa_pairs: Optional[list[dict]] = None,
    search_mode: str = "vector",
    rerank: bool = True,
    top_k: int = 5,
) -> dict:
    """
    Evaluate all golden QA pairs and compute aggregate scores.

    Returns:
        Dict with per-question results and aggregate metrics.
    """
    if qa_pairs is None:
        qa_pairs = load_golden_qa()

    results = []
    for i, qa in enumerate(qa_pairs, 1):
        question = qa["question"]
        expected = qa["expected_answer"]
        logger.info(f"Evaluating [{i}/{len(qa_pairs)}]: {question[:60]}...")

        try:
            result = evaluate_single(
                question, expected,
                search_mode=search_mode, rerank=rerank, top_k=top_k,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to evaluate question {i}: {e}")
            results.append({
                "question": question,
                "expected_answer": expected,
                "error": str(e),
                "metrics": {
                    "context_relevance": 0.0,
                    "answer_faithfulness": 0.0,
                    "answer_correctness": 0.0,
                },
            })

    # Aggregate
    n = len(results)
    agg = {
        "context_relevance": sum(r["metrics"]["context_relevance"] for r in results) / n if n else 0,
        "answer_faithfulness": sum(r["metrics"]["answer_faithfulness"] for r in results) / n if n else 0,
        "answer_correctness": sum(r["metrics"]["answer_correctness"] for r in results) / n if n else 0,
    }
    agg["overall"] = round(sum(agg.values()) / 3, 4)

    return {
        "config": {
            "search_mode": search_mode,
            "rerank": rerank,
            "top_k": top_k,
            "num_qa_pairs": n,
        },
        "aggregate_metrics": {k: round(v, 4) for k, v in agg.items()},
        "per_question": results,
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

    # Print summary
    print("\n" + "=" * 60)
    print("MAS RegQ&A Evaluation Report")
    print("=" * 60)
    print(f"Search mode: {args.mode} | Rerank: {not args.no_rerank} | Top-K: {args.top_k}")
    print(f"QA pairs evaluated: {report['config']['num_qa_pairs']}")
    print("-" * 60)

    agg = report["aggregate_metrics"]
    print(f"  Context Relevance:   {agg['context_relevance']:.4f}")
    print(f"  Answer Faithfulness: {agg['answer_faithfulness']:.4f}")
    print(f"  Answer Correctness:  {agg['answer_correctness']:.4f}")
    print(f"  Overall:             {agg['overall']:.4f}")
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
