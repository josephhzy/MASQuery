# Confidence Scoring

This document explains what the `confidence` field in the `/query` response means, how it is computed, and how it maps to the user-facing display. It is a separate concept from `is_answerable` — confidence describes *retrieval strength*, not *answer correctness*.

## TL;DR

- Confidence is a **retrieval signal**, not an answer-correctness signal.
- It is computed from the cosine similarity scores of the retrieved chunks, before the LLM sees anything.
- The three labels — `high`, `medium`, `low` — map to two threshold boundaries defined in `config.py`.
- A `high` confidence answer can still be wrong. A `low` confidence answer can still be right. The label says "how good was retrieval," not "how good is the answer."

## Implementation

Function: `compute_confidence` in `generator.py`. Reproduced conceptually:

```python
def compute_confidence(scores: List[float]) -> str:
    if not scores:
        return "low"

    top_score = scores[0]
    supporting = sum(1 for s in scores if s >= SIMILARITY_THRESHOLD)

    if top_score >= CONFIDENCE_HIGH_THRESHOLD and supporting >= 2:
        return "high"
    elif top_score >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "medium"
    else:
        return "low"
```

Inputs:

| Input | Source | What it measures |
|-------|--------|-------------------|
| `scores[0]` (top score) | `RetrievalResult.relevance_score` of the highest-ranked chunk | How close the most-similar retrieved chunk is to the query |
| `supporting` count | Number of chunks above `SIMILARITY_THRESHOLD` | Whether the top chunk is a lucky singleton or part of a cluster |

Thresholds (from `config.py`):

| Constant | Value | Role |
|----------|------:|------|
| `SIMILARITY_THRESHOLD` | 0.3 | Minimum cosine similarity to be counted as "supporting". Also used as the refusal gate in Layer 1 of answerability. |
| `CONFIDENCE_HIGH_THRESHOLD` | 0.65 | Top score must clear this for `high`. |
| `CONFIDENCE_MEDIUM_THRESHOLD` | 0.45 | Top score must clear this for `medium`. |

## What each label means

### `high`

`top_score ≥ 0.65` **and** at least two chunks above 0.3.

**Reading:** the top chunk is strongly similar to the query AND is not an isolated match — multiple nearby chunks in the corpus also align with the query. This is the retrieval profile of "the question is directly on-topic for content we have."

Still does not guarantee correctness. A strong retrieval match can still produce a wrong answer if:
- The LLM paraphrases away from the retrieved content
- The cited chunk says something close-but-not-equal to what the question asks
- The retrieved content is itself imprecise (regulator hedges, defined terms change across documents)

### `medium`

`top_score ≥ 0.45` (and not high).

**Reading:** the top chunk has meaningful overlap with the query, but either it's not a strong match or there aren't multiple supporting chunks. This is the retrieval profile of "we found something plausible, but the user should be alert."

### `low`

Everything else.

**Reading:** the top chunk is weakly similar or entirely missing. This is the retrieval profile of "the corpus probably doesn't cover this question well."

Note the interaction with refusal: if `top_score < SIMILARITY_THRESHOLD = 0.3`, `generate_answer` short-circuits *before* the LLM call and returns a refusal message with `confidence="low"`, `is_answerable=False`. Confidence is therefore always `low` on a refused answer, but `low` alone does not mean refusal — `0.3 ≤ top_score < 0.45` produces `low` confidence with a real LLM answer.

## Why these thresholds and not others

See `docs/REFUSAL_THRESHOLD.md` — the same critique applies. The 0.65 / 0.45 / 0.3 values are heuristic anchors for `all-MiniLM-L6-v2` on English regulatory text. They have not been calibrated against a labelled "what should confidence be for this question?" dataset.

A principled calibration would:

1. Assemble a set of (question, true answer, retrieved top_score) triples across a range of retrieval difficulty.
2. For each label (`high / medium / low`), fit an empirical probability of answer correctness conditional on the retrieval profile.
3. Pick thresholds such that:
   - `high` implies ≥90% conditional correctness
   - `medium` implies 60–80%
   - `low` implies <60%
4. Report the calibration curve.

Until that curve exists, the labels are triage hints, not probabilities.

## User-facing display

The Streamlit UI (`streamlit_app.py`) renders confidence as a coloured badge:

| Label | Colour | UI message |
|-------|--------|------------|
| `high` | green | Strong retrieval match — multiple supporting excerpts found |
| `medium` | amber | Moderate retrieval match — review sources carefully |
| `low` | red | Weak retrieval match — treat as indicative only |

The API response exposes confidence as a plain string (`"high" | "medium" | "low"`) so non-UI clients (e.g. a pipeline script) can filter by it.

## What confidence is *not*

- Not a faithfulness score — see `docs/VERIFICATION.md` for the citation-level verification signal.
- Not a correctness guarantee — the LLM can still get the answer wrong under `high` confidence.
- Not comparable across search modes. `bm25` scores are TF-IDF; `hybrid` scores are RRF. The thresholds above are calibrated for cosine similarity (vector mode) and BM25/hybrid modes pass through the labelling logic without a regime switch. This is a known mild incorrectness — fine for a prototype, should be parameterised by mode in a v2.

## Where this matters in practice

- **Filter for high-confidence answers in scripted use:** `jq '.[] | select(.confidence=="high")' responses.json`.
- **Route low-confidence queries to human review in production:** the right place to attach an escalation path.
- **Debug retrieval quality:** if many queries come back `low`, the corpus is likely missing content or chunking is too aggressive. Inspect `retrieval_scores` in the response.

## Planned improvements

Aligned with the roadmap in `docs/REFUSAL_THRESHOLD.md`:

1. Calibrate thresholds against an annotated set where each question has a ground-truth "is this well-supported by the corpus?" label.
2. Emit raw `top_score` and `supporting_count` in the API response alongside the label, so downstream systems can apply their own cutoffs.
3. Parameterise thresholds by search mode (or normalise scores onto a common scale) so `hybrid` and `bm25` confidence means the same thing as `vector` confidence.
