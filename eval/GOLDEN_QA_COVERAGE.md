# Golden QA Coverage

The golden QA set lives at `evaluation/golden_qa.yaml`. This document enumerates every entry, describes how the set was constructed, and states the known limitations so the sample size caveat is visible up front.

> **Scale:** 23 pairs total — 13 answerable + 10 adversarial (out-of-corpus / fabricated / near-miss). This is a smoke test, not an evaluation benchmark. Scaling plan lives in `eval/EXPANSION_PLAN.md`. The enumeration below covers the 13 answerable pairs; see `evaluation/golden_qa.yaml` for the adversarial subset.

## Topic distribution

The corpus ingests five MAS guideline documents (`data/raw/`):

- `TRM_Guidelines.pdf` — Technology Risk Management
- `BCM_Guidelines.pdf` — Business Continuity Management
- `Outsourcing_Guidelines.pdf`
- `Fair_Dealing_Guidelines.pdf`
- `E_Payments_Guidelines.pdf`

The 13 QA pairs distribute across those documents as follows:

| Document | Count | Share |
|----------|------:|------:|
| TRM Guidelines | 6 | 46% |
| Outsourcing Guidelines | 3 | 23% |
| BCM Guidelines | 2 | 15% |
| Fair Dealing Guidelines | 1 | 8% |
| E-Payments Guidelines | 1 | 8% |

TRM is overrepresented — it is the longest and most subsection-heavy document, so it naturally yields more distinct Q&A angles, but the distribution is not defensible as a balanced benchmark.

## Full enumeration

Numbers correspond to the order in `evaluation/golden_qa.yaml`.

### TRM (6)

1. **Access control requirements under the TRM Guidelines** — tagged section `User Access Management`. Tests retrieval of MFA, RBAC, access reviews, joiner-mover-leaver controls.
2. **Incident reporting requirements under TRM** — tagged `Incident Management`. Tests that retrieval distinguishes "reportable incidents" from generic incident handling.
3. **Data protection requirements for financial institutions** — tagged `Data Protection`. Tests encryption, access controls, DLP.
4. **Requirements for IT audit under the TRM Guidelines** — tagged `IT Audit`. Tests independence-of-audit-function retrieval.
5. **Cloud computing risk management requirements** — tagged `Cloud Computing`. Tests retrieval of the cloud-specific subsection — this one is adversarial-adjacent because the general TRM controls also match lexically.
6. **Requirements for cyber security risk management** — tagged `Cyber Security`. Tests retrieval of the cyber-specific framework, distinguishing from generic TRM.

### Outsourcing (3)

7. **Outsourcing notification requirements for banks** — tagged `Notification Requirements`. Numeric-free compliance obligation.
8. **Due diligence requirements for material outsourcing** — tagged `Due Diligence`. Tests retrieval where "material" is a defined term.
9. **Requirements for monitoring outsourced arrangements** — tagged `Ongoing Monitoring`. Tests distinction between pre-engagement and ongoing obligations.

### BCM (2)

10. **What must a financial institution include in its business continuity plan** — tagged `Business Continuity Planning`.
11. **Recovery time objectives under the BCM Guidelines** — tagged `Recovery Objectives`. Tests retrieval of RTO-specific content.

### Fair Dealing (1)

12. **Fair dealing outcome requirements for financial institutions** — tagged `Fair Dealing Outcomes`. The only Fair Dealing pair; coverage here is thin.

### E-Payments (1)

13. **Liability limits for unauthorised e-payment transactions** — tagged `Liability Framework`. The only E-Payments pair.

## Construction methodology

These pairs were authored by the project owner using the following process:

1. **Document skim.** Read each of the five MAS guideline PDFs end-to-end.
2. **Pick anchor sections.** Choose sections that a realistic compliance analyst would ask about — not trivia, but concrete obligations.
3. **Draft the question.** Phrase it the way an analyst would, not in copy-paste document language, so retrieval has to generalise.
4. **Write the expected answer from the document.** The expected answer is a concise paraphrase of what the section says, in plain English, without copying wording verbatim.
5. **Record the source section.** Used as a free-form traceability label (the RAG pipeline doesn't depend on it; it's there for a future human reviewer).

## Known limitations

Call these out plainly to any reviewer:

1. **Single annotator.** The pairs were authored by one person. There is no inter-annotator agreement (Cohen's κ) and no review pass. This is the single biggest credibility gap. See `eval/EXPANSION_PLAN.md`.
2. **Adversarial coverage is thin.** 10 adversarial pairs were added (4 out-of-corpus, 3 fabricated MAS references, 3 near-miss) to exercise the refusal path — they drive the Layer-1 refusal numbers in the README. Still small: refusal precision on 10 items is a coin-flip-away-from-a-coin-flip. Target is 20+ at the scale-up.
3. **Topic imbalance.** Fair Dealing and E-Payments are each represented by one answerable pair. A single evaluation data point is a coin flip, not a measurement.
4. **Near-miss tests exist but are shallow.** 3 near-miss adversarial pairs probe retrieval failure modes (cross-document phrasing, defined-term mismatch). Not enough to characterise the full retrieval failure surface — more are planned in `eval/EXPANSION_PLAN.md`.
5. **Expected answers are paraphrases, not regulator quotations.** This is deliberate — it lets `answer_correctness` tolerate reasonable rewordings — but it means the golden answer is itself an interpretation, not a ground truth.
6. **Question phrasing is author-biased.** The author chose phrasings that "feel natural". A different phrasing could produce different retrieval behaviour. A larger set would need a variety of phrasing styles per concept.

## How to use this set honestly

- Treat the aggregate scores from `python -m evaluation.evaluator` as **diagnostics, not metrics**. A score of 0.72 on 13 pairs is not meaningfully different from 0.78.
- Look at per-question results to find cases where retrieval fails or the LLM refuses unexpectedly — those are debugging signals.
- Do not publish these numbers as the system's "accuracy". Until this set is expanded (see `EXPANSION_PLAN.md`) they are indicative only.
