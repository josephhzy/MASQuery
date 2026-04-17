# Retrieval Ablation

This document describes the hybrid-retrieval ablation: how dense-only (FAISS), BM25-only, and hybrid-RRF fusion compare on standard retrieval metrics. Real numbers from a run of `scripts/run_ablation.py` against the 13 golden QA pairs are recorded below. Full JSON report: `ablation_report.json` at the project root.

## Why this ablation exists

The README's retrieval headline is *hybrid search via Reciprocal Rank Fusion*. A reasonable question to ask is: *why hybrid, not dense-only?* Without an ablation, the answer is a design claim. With an ablation, it's a measurement.

Hybrid retrieval earns its complexity only if it beats both baselines on recall. If dense-only wins on a given corpus, there is no justification for the extra BM25 index, the RRF plumbing, and the tokenisation path. Running the ablation is the difference between "we did this because of a principled retrieval-diversity argument" and "we did this because it's what we'd measured to work best on this corpus".

## Metrics

All metrics are computed over the golden QA set (`evaluation/golden_qa.yaml`).

- **Recall@k** — fraction of questions for which at least one expected source chunk appears in the top-k retrieved. Requires `expected_source_chunks` to be populated in the golden QA (see `eval/EXPANSION_PLAN.md` — the v1 13-pair set uses `source_section` strings, not chunk IDs, so recall@k is currently approximated by a substring match between the expected section name and retrieved `section_header`). Report both `recall@5` and `recall@10`.
- **MRR (Mean Reciprocal Rank)** — mean over questions of `1 / (rank of first expected chunk)`, with 0 if no expected chunk appears in top-10.
- **Precision@1** — was the first retrieved chunk an expected one?

## Measured table

Run: `python scripts/run_ablation.py --output ablation_report.json`, `top_k=10`, n=13 golden QA pairs, v1 `source_section` substring match (see caveat at the bottom).

| Retrieval mode | Rerank | Recall@5 | Recall@10 | MRR@10 | Precision@1 | Notes |
|----------------|-------:|---------:|----------:|-------:|------------:|-------|
| Dense (FAISS only) | off | 0.7692 | 0.7692 | 0.5513 | 0.3846 | FAISS cosine similarity, `all-MiniLM-L6-v2` |
| Dense (FAISS only) | on  | 0.7692 | 0.7692 | **0.6179** | **0.5385** | + cross-encoder rerank top-N — **best MRR and P@1** |
| BM25 only | off | 0.6923 | 0.6923 | 0.4167 | 0.3077 | Default Okapi tokeniser |
| BM25 only | on  | 0.6923 | 0.6923 | 0.5538 | 0.4615 | + cross-encoder rerank top-N |
| Hybrid (RRF) | off | 0.7692 | 0.7692 | 0.5410 | 0.3846 | RRF k=60 over dense + BM25 |
| Hybrid (RRF) | on  | 0.6923 | 0.7692 | 0.5726 | 0.4615 | **Current default configuration** |

## What the measurement shows

Against the 13-pair smoke-test set, with v1 substring matching:

- **Recall@5 and Recall@10 are tied** across dense-off, dense-on, hybrid-off, and hybrid-on-at-10 — all at 0.7692 (10/13). BM25-only trails at 0.6923 (9/13). Hybrid+rerank drops a single question out of recall@5 vs hybrid-off (0.6923 at k=5, 0.7692 at k=10), which the cross-encoder reranker pushed below rank 5 on one question.
- **Dense+rerank is the top ordering configuration** on both MRR@10 (0.6179) and Precision@1 (0.5385). Hybrid+rerank lags by ~0.045 on MRR and ~0.077 on P@1.
- **Rerank helps every mode.** On dense: MRR +0.067, P@1 +0.154. On BM25: MRR +0.137, P@1 +0.154. On hybrid: MRR +0.032, P@1 +0.077. So the cross-encoder does its job — it moves the right chunk up.
- **BM25 underperforms across the board.** No metric where BM25-only wins. The Okapi tokeniser is not capturing anything dense isn't already covering, at least in this sample.
- **Hybrid does not beat dense.** With rerank, dense is ahead on every ordering metric and tied on recall. On this corpus, at this scale, the extra BM25 index + RRF plumbing is not earning its complexity — the README's "hybrid search" design claim does not hold up as a measured win.

## Interpreting the result

- The honest read of this ablation is that **dense + rerank is the configuration to ship** at this corpus size. MASquery's current default is dense + rerank; the hybrid path (BM25 + FAISS + RRF) is retained as an opt-in mode.
- Two caveats that matter:
  - **Sample size = 13.** These numbers are a smoke-test, not a production benchmark. A single question flipping recall@5 is worth 0.077. The MRR / P@1 gap is within noise for a set this size.
  - **v1 substring matching is generous.** It counts a hit if the retrieved `section_header` shares a >3-character word with the expected `source_section`. A real recall number requires the v2 schema (`expected_source_chunks` with explicit chunk IDs) per `eval/EXPANSION_PLAN.md`.
- **Recommended next step:** author the v2 expected-chunks fields on the existing 13 pairs, re-run, and confirm whether the dense-wins result survives strict matching. If it does, drop BM25.
- **Do not** read "hybrid = best" into this table. It isn't.

## Decision

- **Default set to `dense+rerank`.** `SEARCH_MODE=vector` is the config default and the evaluator CLI default; hybrid is preserved as an opt-in mode (`SEARCH_MODE=hybrid` or per-request `"search_mode": "hybrid"`).
- **Hybrid code is retained, not deleted.** `retriever._reciprocal_rank_fusion`, the BM25 index build, and the hybrid branch of `retriever.search()` are unchanged. The path is still exercised by `tests/test_retriever.py::test_hybrid_search_returns_results`.
- **Re-run on the expanded 23-pair set (13 answerable + 10 adversarial)** reproduces the dense+rerank lead on the answerable subset exactly (MRR@10 0.6179, P@1 0.5385). On the full 23-pair run, the raw retrieval-hit metrics are depressed because the 10 adversarials have no expected corpus chunk to match — that's the intended shape: they're routed to the refusal path, not the retrieval-quality one.
- **Layer-1 refusal pre-gate measured on the adversarial subset:** 9/10 should-refuse pairs correctly caught by the `max(score) < 0.3` pre-gate under dense+rerank (out-of-corpus 4/4, fabricated 3/3, near-miss 2/3). The single false negative is the "material outsourcing vs material adverse change" conflation pair — the retriever legitimately fires on "material outsourcing" and the cross-encoder scores the top chunk at +1.99, above threshold; that case is expected to fall to the Layer-2 refusal-pattern check in `generator._detect_refusal`, which needs an LLM call to measure end-to-end. The pre-gate also incorrectly refuses 2/13 answerable questions (TRM incident reporting, TRM data protection) where the cross-encoder rescored all top chunks below 0.3 on this corpus. That asymmetry — CE logits used against a cosine-scale threshold — is a known pragma, not a principled scoring choice; see `docs/REFUSAL_THRESHOLD.md` for the PR-sweep methodology that would replace it.
- **Why defer the hybrid story rather than cut it:** n=13 answerable with v1 substring matching is a smoke-test, not a verdict. The dense-wins result survives this sample; it has not been tested against the 50+ pair set described in `eval/EXPANSION_PLAN.md` or against v2 strict `expected_source_chunks` matching. Once that set exists, the ablation runs again and the default is re-evaluated on evidence, not defended on priors.
- **What to watch for a flip back to hybrid:** lexical queries (MAS Notice / clause numbers, defined acronyms) surfacing in production logs where dense alone misses them, or v2 recall@5 showing hybrid ahead at 50+ pairs. Neither is true today.

## Original hypothesis (for the record)

Before running, the expectation was:

- BM25 contributes unique recall on clause numbers and defined acronyms → **not observed in this sample.**
- Dense contributes unique recall on paraphrased questions → **consistent with the dense ≥ BM25 pattern.**
- Hybrid-RRF dominates recall → **tied with dense, did not win.**
- Cross-encoder rerank mostly moves the right chunks up → **confirmed — rerank improves MRR and P@1 across every mode.**

The measurement falsified the first and third hypotheses for this corpus at this scale. Recording this outcome honestly is the point of running the ablation in the first place.

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

The JSON report contains per-configuration aggregates and per-question raw results so you can audit any surprise.
