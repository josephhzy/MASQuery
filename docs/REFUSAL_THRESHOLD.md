# Refusal Threshold

This document explains how MASquery decides to refuse a question, what the current threshold values are, and how they should be justified. The honest answer up front: **the current values are hardcoded heuristics, not the output of a precision-recall sweep.** This doc says exactly what was chosen, where it lives, and what the defensible methodology looks like.

## Where refusal decisions are made

Refusal is two-layered. Both layers are observable in the API response via `is_answerable` and the fallback message.

### Layer 1 — Pre-generation retrieval gate

`generator.generate_answer` (in `generator.py`) short-circuits before the LLM call if retrieval is weak:

```python
if not results or (scores and max(scores) < SIMILARITY_THRESHOLD):
    return GenerationResult(
        answer=<standard refusal message>,
        confidence="low",
        is_answerable=False,
        retrieval_scores=scores,
    )
```

The threshold is `config.SIMILARITY_THRESHOLD = 0.3`. Reading: if the top retrieved chunk's cosine similarity to the query is below 0.3, we refuse without asking the LLM.

**Scope caveat — the threshold is only on the right scale for dense-only, no-rerank mode.** The 0.3 figure is a cosine-similarity number. It only makes sense against raw FAISS cosine scores. In every other path the score passed to this gate lives on a different scale, and `retriever.search()` only applies the `SIMILARITY_THRESHOLD` pre-filter in `search_mode == "vector"` with `do_rerank == False`:

| Search path | Score on top result | Layer-1 pre-filter in `retriever.search`? | Layer-1 gate in `generator` comparable to 0.3? |
|-------------|---------------------|--------------------------------------------|-------------------------------------------------|
| Dense (`vector`), no rerank | Cosine similarity ∈ [-1, 1] | **Yes** — drops chunks below 0.3 inside `search()` | Yes — this is the only path where the 0.3 threshold is on the intended scale |
| Dense (`vector`) + rerank (**default**) | Cross-encoder logit | No — threshold pre-filter is skipped | No — CE logits are not cosine, 0.3 is arbitrary here |
| BM25, no rerank | Unbounded TF-IDF score | No | No |
| BM25 + rerank | Cross-encoder logit | No | No |
| Hybrid (RRF), no rerank | RRF score ≈ 2/(k + rank) | No | No |
| Hybrid + rerank | Cross-encoder logit | No | No |

The production default is **dense + rerank**, which is the second row. On that path:

- `retriever.search()` does **not** apply any 0.3 pre-filter, by design (see the prominent comment block in `retriever.py::search`).
- `generator.generate_answer()` still performs its `max(scores) < SIMILARITY_THRESHOLD` check, but the score there is a CE logit, not a cosine similarity. Comparing CE logits against 0.3 is not a principled operation — it fires non-deterministically relative to its documented purpose.

This is a known asymmetry in v1 (see the refusal table in `README.md`: 2/13 false-positive refusals on answerable questions under the pre-gate are directly attributable to CE-logit-vs-cosine-threshold mismatch). The v1 response is to **rely on Layer-2** (post-generation refusal detection) on the default path and to treat Layer-1's contribution on reranked paths as a best-effort floor, not a calibrated gate.

### Layer 2 — Post-generation refusal detection

After the LLM responds, `generator._detect_refusal` runs a small regex list over the response text:

```python
refusal_patterns = [
    r"cannot answer.*based on",
    r"documents? do not contain",
    r"do not contain (sufficient|relevant|enough) information",
    r"no relevant information",
    r"not covered in.*excerpt",
    r"insufficient information",
    r"unable to (find|answer|provide)",
]
```

If any pattern matches, `is_answerable=False`. This layer catches cases where retrieval found something but the LLM decided the excerpts did not actually answer the question — a good signal, because it indicates the system prompt's "say I don't know" instruction is being respected.

### A note on confidence vs answerability

These are distinct concepts. `compute_confidence` in `generator.py` uses two other thresholds:

```python
CONFIDENCE_HIGH_THRESHOLD = 0.65
CONFIDENCE_MEDIUM_THRESHOLD = 0.45
```

High confidence requires top score ≥ 0.65 AND at least two chunks above `SIMILARITY_THRESHOLD`. Medium requires top score ≥ 0.45. These do not drive refusal — they drive the user-facing `"high" / "medium" / "low"` label. See `docs/CONFIDENCE_SCORING.md`.

## Current values and how they were chosen

| Threshold | Value | Location | How it was picked |
|-----------|------:|----------|-------------------|
| `SIMILARITY_THRESHOLD` | 0.3 | `config.py` | Heuristic. Below ~0.3 cosine, `all-MiniLM-L6-v2` embeddings on MAS text are typically unrelated. Not measured against an adversarial golden set. |
| `CONFIDENCE_HIGH_THRESHOLD` | 0.65 | `config.py` | Heuristic. Scores above this on MiniLM typically reflect strong topical overlap. Not measured. |
| `CONFIDENCE_MEDIUM_THRESHOLD` | 0.45 | `config.py` | Heuristic midpoint between the low and high bands. Not measured. |
| Layer 2 refusal patterns | — | `generator.py` | Authored by inspection of a handful of Claude refusals. No precision/recall measurement. |

**This is not a defensible choice for a production compliance system.** It is a defensible *prototype* choice — the values are in the right ballpark for MiniLM on medium-length English regulatory text — but the right number cannot be known without the sweep described below.

## The methodology that should be used

### Step 1 — Build a refusal-labelled set

You cannot sweep a refusal threshold without labels. Requires a QA set where each item has a `should_refuse` boolean. The plan for this set is in `eval/EXPANSION_PLAN.md` (targets ≥10 adversarial items across out-of-corpus topics, fabricated defined terms, and near-miss retrieval).

### Step 2 — Run the retrieval-only pipeline over the labelled set

For each (question, should_refuse) pair, record:

- `top_score` — cosine similarity of the top retrieved chunk
- `num_above_threshold(t)` — for a sweep of candidate thresholds `t`
- `retrieval_scores` — full list for later bucketed analysis

### Step 3 — Compute the precision-recall curve over thresholds

For each candidate threshold `t` in, say, `[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]`:

- **Predicted refusal:** `top_score < t`
- **True refusal:** `should_refuse == True`
- Compute `refusal_precision = TP / (TP + FP)` and `refusal_recall = TP / (TP + FN)`
- Compute F1 and F2 (F2 weights recall more heavily; arguably correct for a compliance tool that prefers silence over hallucination)

Plot precision-recall across the threshold sweep. Mark the operating point.

### Step 4 — Pick the operating point with an explicit tradeoff

Two reasonable stances:

- **"Default to silence" stance:** pick the lowest threshold on the curve where refusal recall is ≥ 0.9 on adversarial items. Accept that some legitimate questions will be refused (false refusals), because confident hallucination is worse than an apology.
- **"Answer when possible" stance:** pick the threshold that maximises F1 on the refusal-labelled set. Accept that a few adversarial items will get answered, because excessive refusal makes the tool useless.

For a compliance-assistant use case the "default to silence" stance is the right starting position. The operating point should be reported, not merely chosen.

### Step 5 — Separately tune the post-generation refusal detector

The regex list in `_detect_refusal` has its own precision-recall profile:

- **False positive:** LLM gave a real answer, regex matched the word "insufficient" in a non-refusal context.
- **False negative:** LLM refused with a phrasing none of the patterns match.

Against the same adversarial set, measure these rates and expand the pattern list if needed. Consider replacing it with an LLM-judged classification, or with a small fine-tuned classifier if this becomes a real bottleneck.

## What to report when the sweep is run

A single `docs/REFUSAL_THRESHOLD_CURVE.md` with:

- The refusal-labelled set's size and composition
- The full precision-recall table across thresholds
- A plot (PNG) of the curve with the chosen operating point highlighted
- The chosen threshold value
- The residual false-refuse and false-answer cases (each listed — these are the failure modes that slipped through)
- A re-evaluation date (thresholds drift as the corpus grows)

## Until that sweep exists

The current `SIMILARITY_THRESHOLD = 0.3` is a placeholder documented as such. The honest explanation for "how 0.3 was picked":

> It is a heuristic anchor for MiniLM on this corpus. The PR sweep has not been run yet because the adversarial set is the gating dependency for the scale-up plan in `eval/EXPANSION_PLAN.md`. The code path that uses the threshold is isolated so swapping in a measured value is a one-line change once the set exists.

That framing is more accurate than "I don't know".

For the follow-up "the refusal threshold is dead on the default config, isn't it?", the honest answer:

> Yes, for the reranked path. `retriever.search()` skips the 0.3 pre-filter on every path except dense-only-no-rerank because the threshold is a cosine number and the other paths produce BM25, RRF, or CE-logit scores that aren't on that scale. On the default (vector + rerank) pipeline, Layer-1 is a best-effort floor against CE logits; the real refusal guard is Layer-2 — `_detect_refusal` on the LLM output. The honest fix is a separate per-path threshold calibrated on a labelled set, which is the same gating dependency as the 0.3 sweep above. That is documented in `retriever.py` with a block comment at the gate site.
