# Citation Verification: What It Does, What It Doesn't, and the Upgrade Path

This document is an honest description of the citation verification layer. It is the single most important hallucination guardrail in MASquery, and it is also currently the weakest. This page exists so that no downstream reader is surprised by what it can and can't catch.

## TL;DR

| Layer | Today | Target |
|-------|-------|--------|
| Citation extraction | Regex (`tracer.CITATION_PATTERN`) | Unchanged — regex is the right tool for format extraction |
| Verification (does this citation correspond to a real retrieved chunk?) | Fuzzy token/substring match on doc name + section header + page overlap, gated by a hand-tuned score threshold | Cross-encoder NLI between the claim and the candidate chunk; entailment ≥ threshold required |
| Hallucinated-citation signal | "Unverified" label on the response | Same label, but driven by contradiction/neutral classification rather than surface-form mismatch |

## What the code actually does today

File: `tracer.py`.

### Step 1 — Extract citations from the LLM response

`extract_citations(response_text)` runs a permissive regex (`tracer.CITATION_PATTERN`) over the LLM's output. The canonical citation shape is:

```
[Source: <document>, Section: <section>, Page: <page>]
```

**Accepted variants (all case-insensitive, whitespace-tolerant around colons and commas):**

| Variant | Example | Notes |
|--------|---------|-------|
| Canonical | `[Source: TRM Guidelines, Section: 5.2 Access Control, Page: 12]` | The format the system prompt asks for. |
| Parenthesised | `(Source: TRM Guidelines, Section: 5.2 Access Control, Page: 12)` | Some LLMs emit `(...)` instead of `[...]`. |
| Page omitted | `[Source: TRM Guidelines, Section: 5.2 Access Control]` | Citation still extracts; page overlap contributes 0 to the match score. |
| Abbreviated section label | `Sec.`, `Part` in place of `Section` | e.g. `[Source: TRM Guidelines, Sec.: 5.2, Page: 12]` |
| Abbreviated page label | `Page`, `Pages`, `Pg.`, `p.` | e.g. `[Source: TRM Guidelines, Section: 5.2, Pg. 12]` |
| Extra whitespace | `[  Source :  TRM Guidelines , Section:  5.2 , Page:  12 ]` | Tolerated around every token boundary. |
| Page ranges and lists | `Page: 8-10, 12` → `[8, 9, 10, 12]` | Parsed by `_parse_pages`. |

**Not accepted (silent drop):** citations with no closing bracket/paren, citations where the document clause is missing, citations that use keywords other than `Source`. Any such citation falls through to the "no citations found" path, which returns all retrieved chunks tagged `verified=False` — the conservative default.

Unit coverage for these forms lives in `tests/test_tracer.py::TestExtractCitations`.

### Step 2 — Match citation to provided chunks

`verify_citations(citations, provided_results)` then tries to find, for each extracted citation, a `RetrievalResult` that matches. Matching logic in `_find_matching_result`:

- **Document name (`doc_score`):**
  - Substring containment either direction → 2 points
  - `_word_overlap > 0.5` → 1 point
  - Otherwise → 0 points and skip (document mismatch is fatal)
- **Section header (contributes to `other_score`):**
  - Substring containment either direction → +2
  - `_word_overlap > 0.3` → +1
- **Page number intersection:**
  - Any page in the citation also in the chunk's pages → +1

A citation is considered verified if the total score is ≥ `CITATION_MATCH_THRESHOLD` (module-level constant, currently `3`). Otherwise it is returned as `verified=False` with `chunk_id="unverified"`. Boundary coverage for the threshold lives in `tests/test_tracer.py::TestCitationMatchThreshold`.

### Step 3 — Fall back to retrieved chunks

If the LLM emitted zero parseable citations, `trace_response` returns all retrieved chunks tagged `verified=False` — so the UI can still show sources, but the user knows the LLM did not explicitly cite them.

## Known failure modes

These are the cases the current implementation gets wrong. They are not hypothetical — they fall directly out of reading the matching code.

### 1. False reject — valid paraphrased citation is flagged unverified

**Scenario:** The LLM cites "TRM Guidelines, Section: User Access Management, Page: 24" but the retrieved chunk's `section_header` stored at ingestion time is "Access Control" (detected by font size from a different heading on the same page). Word overlap between "User Access Management" and "Access Control" is 1/3 ≈ 0.33 on the shorter side — it clears the 0.3 threshold, so we might get other_score = 1. Document name matches ("TRM Guidelines" ⊆ "TRM_Guidelines") for doc_score = 2. If the page number matches, total = 4, verified. If the page is off by one because the section header stretches across a page boundary, total = 3 — still verified, barely. If the paraphrase is aggressive ("authentication and access control"), word overlap drops below 0.3 and we miss it even though the citation is real.

**Consequence:** A correct answer with a correctly cited source is displayed as "unverified" because a regulator's heading vocabulary doesn't match the LLM's paraphrased vocabulary.

### 2. False accept — fabricated citation that reuses real tokens slips through

**Scenario:** The LLM hallucinates a citation like "TRM Guidelines, Section: Access Management, Page: 24" for a statement that is *not actually supported* by the Access Management chunk on page 24. The verifier doesn't read the chunk text against the claim — it only checks that a retrieved chunk with a matching doc + section + page exists. A real chunk does. Total score ≥ 3. Verified.

**Consequence:** A fabricated claim is stamped "verified: true" in the response, which is worse than a bare hallucination because the provenance lends it unearned credibility.

### 3. Generic section headers

**Scenario:** Many regulatory documents have generic section headers ("Introduction", "Scope", "Definitions"). A citation like "Outsourcing Guidelines, Section: Introduction, Page: 3" will match easily on substring containment. The verifier will pass a citation to a section that has nothing to do with the claim, if the LLM picks a low-information chunk that happens to be page-adjacent.

### 4. Tables and footnotes

When PyMuPDF mis-extracts a table (see `docs/EXTRACTION_LIMITATIONS.md`), the chunk text may be garbled but the metadata (doc + section + page) is intact. The verifier will happily "verify" a citation against a chunk whose text is unreadable. The user sees a green "verified" badge attached to unreadable source.

## What the `verified` flag currently guarantees

Given the above, here is what `verified: true` on a source actually means today:

- A retrieved chunk exists whose document name has non-trivial lexical overlap with the citation's document name
- AND whose section header has non-trivial lexical overlap with the citation's section, OR whose page numbers intersect the citation's pages
- AND the sum of those signals is ≥ 3 on a hand-coded scale

It does **not** mean the claim is supported by the chunk text. It means the citation is not a fully made-up pointer.

## The upgrade path: cross-encoder NLI

The principled fix is to check whether the LLM's claim is semantically entailed by the chunk it cites. This is a natural-language-inference (NLI) task.

### Proposed pipeline

1. **Keep regex extraction.** It's the right tool for parsing citation format.
2. **Keep the fuzzy match as a cheap pre-filter.** It rules out citations whose document or section is fundamentally wrong, without a model call.
3. **Add an NLI head per surviving (claim, chunk) pair.**
   - Model: a cross-encoder fine-tuned on MNLI / ANLI / FEVER. Candidates: `cross-encoder/nli-deberta-v3-base`, `cross-encoder/nli-roberta-base`, or a domain-fine-tuned variant. (We already load a cross-encoder for reranking — this is one more inference hop.)
   - Input: `claim = <sentence from LLM answer that sits adjacent to the citation>`, `premise = <full chunk text of the matched RetrievalResult>`.
   - Output: softmax over {entailment, neutral, contradiction}.
4. **Gate `verified` on entailment score.** Proposed threshold: `entailment >= 0.7` AND `contradiction < 0.1`. Otherwise return `verified=False` and tag with `reason ∈ {neutral, contradicted, low_confidence}`.
5. **Report false-accept and false-reject rates** on a held-out adversarial set — see `eval/EXPANSION_PLAN.md` for the design of that set.

### Why not an LLM-as-judge?

- **Cost.** Every answered query would incur at least one extra full-model call.
- **Latency.** Cross-encoder NLI is 10–30 ms per pair on CPU; an LLM-as-judge adds hundreds of ms per pair.
- **Determinism.** NLI cross-encoders give a reproducible score; LLM judges drift across model versions.
- **Calibration.** An NLI head has a clear probabilistic interpretation; an LLM judge's "yes this supports the claim" does not.

LLM-as-judge is a reasonable fallback for *evaluation-time* scoring of the verifier itself, not for *serving-time* verification.

### Where the change lives

`tracer.py` already isolates verification as its own function (`_find_matching_result`). The NLI upgrade can be implemented as a new step after `_find_matching_result` returns a candidate — the rest of the pipeline does not change. The new code would call `retriever._get_cross_encoder()`-style singleton loading so that the NLI model, like the reranker, is lazy-loaded once.

## Cost/benefit summary

| Signal | Cost | Catches false reject | Catches false accept |
|--------|------|----------------------|----------------------|
| Regex citation extraction | Negligible | N/A | N/A |
| Fuzzy doc/section/page match (today) | Negligible | Partially (trips on paraphrase) | No |
| Cross-encoder NLI (proposed) | ~10–30 ms per (claim, chunk) pair | Yes — paraphrase entailed | Yes — unsupported claim = neutral or contradiction |
| LLM-as-judge | Seconds, non-deterministic | Yes | Yes, but unstable |

## Honest posture

The `prototype scope` note in the README is not a humility move — it is a statement of fact. Today's verifier prevents typos and outright fabricated document names. It does not protect against the harder failure mode: a plausible-looking citation that points to a real chunk whose content does not actually support the claim. The architecture is built to accept an NLI upgrade; the upgrade just hasn't shipped.
