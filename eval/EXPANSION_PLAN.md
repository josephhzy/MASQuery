# Golden QA Expansion Plan: 13 → 50+

This document is the plan to grow the golden QA set from a 13-pair smoke test into something that can be called an evaluation benchmark. It is not the plan to grow it *quickly*. The bottleneck is annotation quality, not annotation volume.

## Target state

- **≥ 50 QA pairs** across the five MAS guideline documents, with deliberate coverage of sub-regulations.
- **≥ 10 adversarial pairs**: questions the corpus cannot answer, questions that depend on a single footnote, questions whose answer spans multiple sections, questions that use MAS-defined terms in slightly wrong ways.
- **Two independent annotators** per pair, with disagreements resolved by discussion and logged.
- **Inter-annotator agreement reported** (Cohen's κ on answer equivalence judgements).
- **Versioned.** Set v1 = current 13. Set v2 = expanded set. Each evaluation run records which version was used.

## Topic coverage target

A deliberate distribution that reflects both document length and realistic compliance coverage. Not just "more of what we already have".

| Document | Current | Target | Rationale |
|----------|--------:|-------:|-----------|
| TRM Guidelines | 6 | 14 | Broadest doc; subdivide into access, incident, IT audit, cyber, cloud, data, BCP-in-TRM, third-party risk, logging/monitoring |
| Outsourcing | 3 | 10 | Add sub-outsourcing, cross-border, cloud-as-outsourcing, termination rights, audit rights |
| BCM | 2 | 8 | Add tabletop testing, crisis comms, dependencies mapping, reverse stress test |
| Fair Dealing | 1 | 8 | Severely underrepresented; add product suitability, vulnerable customers, complaints, sales incentives |
| E-Payments | 1 | 10 | Severely underrepresented; add transaction limits, notification to customer, refund timelines, multi-factor for high-risk |
| **Subtotal** | 13 | 50 | |
| Adversarial / refusal-expected | 10 | 20 | 4 out-of-corpus + 3 fabricated + 3 near-miss shipped; double at scale-up |
| **Total** | 23 | 70 | |

## Adversarial subset design

The refusal behaviour is a first-class feature of MASquery. You cannot measure refusal precision without questions that *should* be refused. Proposed adversarial categories:

1. **Out-of-corpus topics (4 pairs).** E.g. "What are the listing rules for SGX main board?" — legitimate financial regulation in Singapore but outside the ingested corpus. Expected outcome: refusal.
2. **Fabricated defined terms (2 pairs).** E.g. "What does MAS Notice 9999 say about…" — sounds plausible, does not exist. Expected outcome: refusal.
3. **Near-miss retrieval (2 pairs).** Questions whose embeddings match a chunk that does *not* answer them. E.g. an Outsourcing question phrased in TRM vocabulary so BM25 fires on the wrong document. Expected outcome: answer from the correct document, or refusal if retrieval fails.
4. **Multi-hop / cross-document (2 pairs).** E.g. "If a material outsourcing causes a BCM incident, what must be reported and to whom?" — requires chunks from two documents. Expected outcome: answer that cites both documents.

## Annotation protocol

Each QA pair goes through a structured process. The protocol exists so that when someone asks "how were these built?" the answer is not vibes.

### Step 1 — Primary author

Primary author reads the relevant section(s) and writes:
- **`question`** — in the voice of a compliance analyst, not a copy-paste of the regulator's heading
- **`expected_answer`** — a concise paraphrase in plain English (3–6 sentences)
- **`source_section`** — free-form label for traceability
- **`expected_source_chunks`** — new field: a list of expected `chunk_id`s that SHOULD be retrieved for this question. Enables recall@k metrics.
- **`category`** — new field: one of `{standard, adversarial_out_of_corpus, adversarial_fabricated, adversarial_near_miss, multi_hop}`
- **`should_refuse`** — new field: boolean. `true` only for adversarial out-of-corpus and fabricated cases.

### Step 2 — Second annotator

Independent second annotator reviews each pair:
- Reads the source section(s) independently
- Writes their own version of the expected answer, blind
- Judges whether the primary author's question is well-posed (unambiguous, answerable from the corpus, not leading)
- Judges whether the primary author's expected answer is correct

### Step 3 — Agreement measurement

For each pair, both annotators produce a boolean judgement: "does the primary author's expected answer correctly reflect the source?" Cohen's κ is computed on that boolean across all pairs.

- **κ ≥ 0.8** — strong agreement; pairs go into v2.
- **0.6 ≤ κ < 0.8** — moderate agreement; per-pair disagreements are discussed and resolved before inclusion.
- **κ < 0.6** — weak agreement; the whole protocol is re-examined before publishing.

### Step 4 — Disagreement resolution

Disagreements are logged per pair. The pair either:
- Gets rewritten until both annotators agree, OR
- Gets dropped if the underlying regulator text is ambiguous

The drop list is itself useful — it's the set of compliance questions where even expert humans disagree, which is a meta-finding worth surfacing.

## Schema extensions

`golden_qa.yaml` v2 fields (additive, backward compatible):

```yaml
- question: "..."
  expected_answer: "..."
  source_section: "..."
  expected_source_chunks: ["TRM_Guidelines_p24_c0", "TRM_Guidelines_p25_c1"]
  category: "standard"            # or adversarial_*, multi_hop
  should_refuse: false
  annotators: ["author", "reviewer"]
  notes: "Optional — why this pair exists, known tricky bits"
```

With these fields populated, the evaluator gains:

- **`retrieval_recall@k`** — fraction of `expected_source_chunks` that appear in top-k retrieval.
- **`retrieval_mrr`** — mean reciprocal rank of the first expected chunk.
- **`refusal_precision`** — of pairs where the system refused, how many had `should_refuse=True`.
- **`refusal_recall`** — of pairs with `should_refuse=True`, how many the system actually refused.

## Timeline and effort estimate

Honest numbers. No heroics.

| Phase | Work | Estimated effort |
|-------|------|------------------|
| 1. Extend schema + loader | Update `golden_qa.yaml` format, update `load_golden_qa`, update evaluator to compute new metrics | ~0.5 day |
| 2. Write primary drafts (47 new pairs) | ~20 min per pair including reading the section | ~2 days |
| 3. Recruit second annotator | Find someone willing who isn't the primary author | — |
| 4. Independent review (47 pairs × 2 annotators) | ~15 min per pair for the reviewer | ~2 days of reviewer time |
| 5. Resolve disagreements | Varies; budget 30% rework rate | ~1 day |
| 6. Compute κ, publish agreement report | Mechanical | ~0.25 day |

**Total elapsed:** ~1–2 weeks, gated by reviewer availability.

## What NOT to do

- **Do not let an LLM write the expected answers.** The whole point is that the benchmark does not share failure modes with the system under test. An LLM-written golden answer against an LLM-produced answer is a closed loop.
- **Do not scale the set without the second annotator.** A 50-pair single-annotator set is not more credible than a 13-pair single-annotator set — it just claims more.
- **Do not fabricate adversarial examples from the corpus.** Out-of-corpus adversarial questions must reference things genuinely outside the corpus, not mangled versions of what's inside. The latter tests retrieval robustness (useful) but is not a real refusal test.
- **Do not drop the 13 v1 pairs.** They may be thin, but they're consistent. Keep them as v1 and report v2 alongside.

## Open questions

- Should we include a "Chinese-language compliance analyst phrasing" subset? Some MAS notices have bilingual audiences.
- Do we need an "interpretive" split where the question isn't about what the regulator says but about edge-case interpretations? Probably out of scope for a RAG benchmark.
- Should the second annotator also write their own expected answer from scratch, then compare to the primary author's via `answer_correctness`? This would give an upper bound on what the metric can score.
