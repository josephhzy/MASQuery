# Retrieval Ablation

This document describes the hybrid-retrieval ablation: how dense-only (FAISS), BM25-only, and hybrid-RRF fusion compare on standard retrieval metrics. Real numbers from a run of `scripts/run_ablation.py` against the 13 golden QA pairs are recorded below. Full JSON report: `ablation_report.json` at the project root.

## Why this ablation exists

The original README retrieval headline was *hybrid search via Reciprocal Rank Fusion*; this ablation measured whether that claim held up. A reasonable question to ask is: *why hybrid, not dense-only?* Without an ablation, the answer is a design claim. With an ablation, it's a measurement.

Hybrid retrieval adds BM25 indexing, RRF fusion, and a separate tokenisation path. That overhead is only justified if hybrid outperforms dense-only on recall. This ablation measures whether it does.

## Metrics

All metrics are computed over the golden QA set (`evaluation/golden_qa.yaml`).

- **Recall@k** — fraction of questions for which at least one expected source chunk appears in the top-k retrieved. Requires `expected_source_chunks` to be populated in the golden QA (see `eval/EXPANSION_PLAN.md` — the v1 13-pair set uses `source_section` strings, not chunk IDs, so recall@k is approximated by a two-part word-overlap match: `source_section` is split on ` - `, the part before is checked against `doc_name` (vacuously true on this corpus since every doc_name contains a word from the document token), and the part after is checked against `section_header` — both must match). Report both `recall@5` and `recall@10`.
- **MRR (Mean Reciprocal Rank)** — mean over questions of `1 / (rank of first expected chunk)`, with 0 if no expected chunk appears in top-10.
- **Precision@1** — was the first retrieved chunk an expected one?

## Measured table

Run: `python scripts/run_ablation.py --output ablation_report.json`, `top_k=10`, n=13 golden QA pairs, v1 `source_section` substring match (see caveat at the bottom).

| Retrieval mode | Rerank | Recall@5 | Recall@10 | MRR@10 | Precision@1 | Notes |
|----------------|-------:|---------:|----------:|-------:|------------:|-------|
| Dense (FAISS only) | off | 0.7692 | 0.7692 | 0.5513 | 0.3846 | FAISS cosine similarity, `all-MiniLM-L6-v2` |
| Dense (FAISS only) | on  | 0.6923 | 0.7692 | **0.5726** | **0.4615** | + cross-encoder rerank top-N — **best MRR/P@1 (tied with hybrid+rerank); the current default** |
| BM25 only | off | 0.6923 | 0.6923 | 0.4167 | 0.3077 | Default Okapi tokeniser |
| BM25 only | on  | 0.6923 | 0.6923 | 0.5538 | 0.4615 | + cross-encoder rerank top-N |
| Hybrid (RRF) | off | 0.7692 | 0.7692 | 0.5410 | 0.3846 | RRF k=60 over dense + BM25 |
| Hybrid (RRF) | on  | 0.6923 | 0.7692 | 0.5726 | 0.4615 | Opt-in; identical metrics to dense+rerank on this set |

## What the measurement shows

Against the 13-pair smoke-test set, with v1 substring matching:

- **Recall@10 is tied** across dense-off, dense-on, hybrid-off, and hybrid-on — all at 0.7692 (10/13). BM25-only trails at 0.6923 (9/13). **Recall@5 is highest *without* rerank:** dense-off and hybrid-off hit 0.7692, but turning rerank on drops both to 0.6923 — the cross-encoder pushes the one relevant chunk on a single question below rank 5 (it recovers by rank 10).
- **The best ordering is a tie at the top:** dense+rerank and hybrid+rerank are numerically identical (MRR@10 0.5726, P@1 0.4615), with bm25+rerank also at P@1 0.4615. Dense+rerank is chosen as the default because it matches the best ordering while needing only the dense index — no BM25/RRF plumbing.
- **Rerank improves ordering in every mode** (vs the same mode without rerank). On dense: MRR +0.021 (0.5513 → 0.5726), P@1 +0.077 (0.3846 → 0.4615). On BM25: MRR +0.137, P@1 +0.154. On hybrid: MRR +0.032 (0.5410 → 0.5726), P@1 +0.077. Reranking helps top-1, but here at the cost of one recall@5 hit on dense/hybrid.
- **BM25 underperforms across the board.** No metric where BM25-only wins. The Okapi tokeniser is not capturing anything dense isn't already covering, at least in this sample.
- **Hybrid does not beat dense.** Hybrid+rerank is numerically identical to dense+rerank on every metric here, so the extra BM25 index + RRF plumbing adds nothing on this corpus at this scale. The README's original "hybrid search" design claim does not hold up as a measured win.

## Interpreting the result

- The honest read of this ablation is that **dense + rerank is the configuration to ship** at this corpus size: it ties for the best ordering (MRR@10 / P@1) while needing only the dense index. MASquery's current default is dense + rerank; the hybrid path (BM25 + FAISS + RRF) is retained as an opt-in mode.
- Two caveats that matter:
  - **Sample size = 13.** These numbers are a smoke-test, not a production benchmark. A single question flipping recall@5 is worth 0.077, and dense+rerank vs hybrid+rerank are tied — so "dense wins" here means "dense matches hybrid for less machinery," not "dense dominates."
  - **v1 substring matching is generous.** It counts a hit if the retrieved `section_header` shares a >3-character word with the expected `source_section`. A real recall number requires the v2 schema (`expected_source_chunks` with explicit chunk IDs) per `eval/EXPANSION_PLAN.md`.
- **Recommended next step:** author the v2 expected-chunks fields on the existing 13 pairs, re-run, and confirm whether the dense-wins result survives strict matching. If it does, drop BM25.
- **Do not** read "hybrid = best" into this table. It isn't.

## Decision

- **Default set to `dense+rerank`.** `SEARCH_MODE=vector` is the config default and the evaluator CLI default; hybrid is preserved as an opt-in mode (`SEARCH_MODE=hybrid` or per-request `"search_mode": "hybrid"`).
- **Hybrid code is retained, not deleted.** `retriever._reciprocal_rank_fusion`, the BM25 index build, and the hybrid branch of `retriever.search()` are unchanged. The path is still exercised by `tests/test_retriever.py::test_hybrid_search_returns_results`.
- **The ablation runs on the answerable subset (13 pairs)** of the 23-pair golden set; the 10 adversarials have no expected corpus chunk to match and are routed to the refusal path, not the retrieval-quality one. BM25 adversarial top_scores (ablation_report.json) range from 11 to 30 — roughly 40-100x above a 0.3-style cosine threshold — so the BM25 path cannot share a threshold with the dense path; a separate calibrated threshold would be required before any score gate could be applied to BM25-only retrieval. On the answerable subset, dense+rerank and hybrid+rerank tie for the best ordering (MRR@10 0.5726, P@1 0.4615).
- **Layer-1 refusal pre-gate (historical numbers, before the score-scale fix):** Under the previous gate — which compared `max(score)` against `SIMILARITY_THRESHOLD = 0.3` regardless of score scale — 8/10 should-refuse pairs were caught under dense+rerank (out-of-corpus 4/4, fabricated 2/3, near-miss 2/3), with two false negatives: the fabricated "MAS Guideline G-7890" pair (CE logit +2.26) and the "material outsourcing vs material adverse change" near-miss pair (+2.43). The same gate also refused 2/13 *answerable* questions (TRM incident reporting, TRM data protection) where the cross-encoder rescored all top chunks below 0.3. Those numbers were the asymmetry that motivated the fix: a cosine threshold has no defensible meaning against CE logits.
- **Layer-1 refusal under the current code (post-fix):** The score-based half of Layer-1 only runs when the caller declares the score scale is cosine — `search_mode="vector", rerank=False`. On the default dense+rerank path it is suppressed; Layer-1 reduces to "refuse if zero results", and the adversarial-subset numbers above no longer apply. Refusal on the default path now relies on Layer-2 (`generator._detect_refusal` on the LLM response), which requires an LLM call per pair to re-measure end-to-end. See `docs/REFUSAL_THRESHOLD.md` for the per-path calibrated-threshold work that would restore a Layer-1 score gate on the reranked default.
- **Why defer the hybrid story rather than cut it:** n=13 answerable with v1 substring matching is a smoke test, not a definitive result. The dense-wins outcome holds on this sample, but it has not been tested against the 50+ pair set described in `eval/EXPANSION_PLAN.md` or against v2 strict `expected_source_chunks` matching. Once that set exists, the ablation runs again and the default is re-evaluated against it.
- **What to watch for a flip back to hybrid:** lexical queries (MAS Notice / clause numbers, defined acronyms) surfacing in query logs where dense alone misses them, or v2 recall@5 showing hybrid ahead at 50+ pairs. Neither is true today.

## Original hypothesis (for the record)

Before running, the expectation was:

- BM25 contributes unique recall on clause numbers and defined acronyms → **not observed in this sample.**
- Dense contributes unique recall on paraphrased questions → **consistent with the dense ≥ BM25 pattern.**
- Hybrid-RRF dominates recall → **tied with dense, did not win.**
- Cross-encoder rerank mostly moves the right chunks up → **confirmed — rerank improves MRR and P@1 across every mode.**

The measurement falsified the first and third hypotheses for this corpus at this scale.

## The script

`scripts/run_ablation.py` runs the ablation end-to-end. It:

1. Loads the FAISS index, metadata, and BM25 index (`retriever.load_index()`).
2. Loads the golden QA (`evaluation.evaluator.load_golden_qa`).
3. For each `(mode, rerank)` combination in the table, runs `retriever.search(question, mode=..., rerank=...)` for every golden question.
4. Computes recall@5, recall@10, MRR@10, and precision@1 against `source_section` (v1) or `expected_source_chunks` (v2).
5. Prints a single consolidated table and optionally saves JSON.

**Prerequisites:** ingestion has been run (FAISS + BM25 indexes exist in `data/index/`).

**Caveat:** v1 uses `source_section` string matching, which is an approximation. The real recall numbers will only be meaningful once the golden set is upgraded to v2 with explicit `expected_source_chunks` (see `eval/EXPANSION_PLAN.md`).

## Running it

```bash
conda activate masquery
python scripts/run_ablation.py --output ablation_report.json
```

The JSON report contains per-configuration aggregates and per-question raw results.
