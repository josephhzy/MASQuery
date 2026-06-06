# Refusal Threshold

This document explains how MASquery decides to refuse a question, what the current threshold values are, and how they should be justified. The current values are hardcoded heuristics, not the output of a precision-recall sweep. This doc says exactly what was chosen, where it lives, and what the defensible methodology looks like.

## Where refusal decisions are made

Refusal is two-layered. Both layers are observable in the API response via `is_answerable` and the fallback message.

### Layer 1 — Pre-generation retrieval gate

`generator.generate_answer` (in `generator.py`) short-circuits before the LLM call if retrieval produced nothing usable. The score-based half of the gate is **only applied when scores are on the cosine scale** — the caller declares the scale by passing `search_mode` and `rerank` keyword arguments:

```python
def generate_answer(query, results, *, search_mode=None, rerank=None):
    mode = (search_mode or SEARCH_MODE).lower()
    do_rerank = rerank if rerank is not None else RERANK_ENABLED
    score_is_cosine = mode == "vector" and not do_rerank
    ...
    score_gate_fires = score_is_cosine and scores and max(scores) < SIMILARITY_THRESHOLD
    if not results or score_gate_fires:
        return <standard refusal message>
```

The threshold is `config.SIMILARITY_THRESHOLD = 0.3`. Reading: when the score is a cosine similarity (dense + no rerank), we refuse if the top chunk is below 0.3 without asking the LLM. On every other path, the score-based half of the gate is suppressed and only the `not results` check remains.

**Why the gate is scale-aware.** The 0.3 figure is a cosine-similarity number; it has no defensible meaning against BM25's unbounded TF-IDF score, against the RRF rank-fusion score, or against a cross-encoder logit. The earlier behavior (a bare `max(scores) < 0.3` check against whatever scale was in use) silently refused real answers when CE logits went negative for valid matches and silently caught some adversarials by accident — neither was a principled outcome. The fix declares the score scale at the call site and only fires the threshold gate on the path where it has cosine semantics.

| Search path | Score on top result | Layer-1 pre-filter in `retriever.search`? | Layer-1 gate in `generator` comparable to 0.3? |
|-------------|---------------------|--------------------------------------------|-------------------------------------------------|
| Dense (`vector`), no rerank | Cosine similarity ∈ [-1, 1] | **Yes** — drops chunks below 0.3 inside `search()` | **Yes** — this is the only path where the 0.3 threshold runs (caller declares `search_mode="vector", rerank=False`) |
| Dense (`vector`) + rerank (**default**) | Cross-encoder logit | No — threshold pre-filter is skipped | **No** — score gate suppressed; only `not results` check applies |
| BM25, no rerank | Unbounded TF-IDF score (top_scores 11–30 on this corpus; see ablation_report.json) | No | **No** — score gate suppressed |
| BM25 + rerank | Cross-encoder logit | No | **No** — score gate suppressed |
| Hybrid (RRF), no rerank | RRF score ≈ 2/(k + rank) | No | **No** — score gate suppressed |
| Hybrid + rerank | Cross-encoder logit | No | **No** — score gate suppressed |

The production default is **dense + rerank**, which is the second row. On that path:

- `retriever.search()` does **not** apply any 0.3 pre-filter, by design (see the prominent comment block in `retriever.py::search`).
- `generator.generate_answer()` does **not** apply the score-based half of the gate either. Layer-1 reduces to "refuse if zero results"; Layer-2 (post-generation refusal detection) is the primary refusal guard.

The function signature takes `search_mode` and `rerank` keyword arguments so the call site is responsible for declaring the score scale:

```python
gen_result = generate_answer(question, results, search_mode=mode, rerank=rerank)
```

`main.py::query_endpoint` passes the active values; `evaluation/evaluator.py` does the same.

This closes the previous CE-logit-vs-cosine-threshold mismatch. It does **not** replace the missing precision-recall sweep — it just stops the gate from misfiring on the wrong scale. A per-path calibrated threshold (one number for cosine, one for RRF, one for CE logit) is still the right next step once an adversarial labelled set exists. See `eval/EXPANSION_PLAN.md`.

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

A threshold sweep requires a labelled set. Requires a QA set where each item has a `should_refuse` boolean. The plan for this set is in `eval/EXPANSION_PLAN.md` (targets ≥10 adversarial items across out-of-corpus topics, fabricated MAS references, and near-miss retrieval).

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

For a compliance-assistant use case, the "default to silence" stance prioritises refusal recall over false-refusal rate. The operating point should be reported with a precision-recall rationale, not merely chosen.

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

> For the follow-up "what does Layer-1 actually do on the default config?", the honest answer: on dense + rerank (the default), Layer-1 only catches the empty-retrieval case — it returns a refusal when zero chunks come back. The score-based half of the gate is suppressed because the score there is a cross-encoder logit, not a cosine similarity, and 0.3 has no defensible meaning against CE logits. Refusal on that path is delegated to Layer-2 (`_detect_refusal` on the LLM's response). On dense-only no-rerank, Layer-1's score gate is on the right scale and does fire. The honest next step is a separate per-path threshold calibrated on a labelled set — one number for cosine, one for RRF, one for CE logit — which is the same gating dependency as the 0.3 sweep above. That separation is documented in `retriever.py` with a block comment at the gate site and enforced in `generator.generate_answer` by the `score_is_cosine` check.
