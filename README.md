# MASquery — RAG-based regulatory document Q&A with source tracing

[![CI](https://github.com/josephhzy/MASquery/actions/workflows/ci.yml/badge.svg)](https://github.com/josephhzy/MASquery/actions/workflows/ci.yml)

A Retrieval-Augmented Generation (RAG) pipeline for answering questions about Monetary Authority of Singapore (MAS) regulatory documents. Every answer is grounded in retrieved source chunks, and each citation is checked against the context actually sent to the model. The goal is to reduce hallucination risk on compliance questions.

> **Prototype scope:** Citation verification is regex-based and fuzzy-matched, and answers without explicit citations are flagged as unverified. This lowers hallucination risk; it does not guarantee zero hallucination. `docs/VERIFICATION.md` covers the upgrade path to NLI-backed entailment scoring.

Demo: run `streamlit run streamlit_app.py` locally (start the API first with `uvicorn main:app --reload`), or `docker-compose up --build` for API + UI together. (Requires an LLM API key for query generation — see Quick Start.)

## Design highlights

- **Dense retrieval + cross-encoder rerank (measured default).** Design started with hybrid on the hypothesis that regulatory queries benefit from both lexical and semantic recall. Measured ablation (see `docs/RETRIEVAL_ABLATION.md`) showed dense + cross-encoder reranking ties hybrid+rerank on the current eval set and needs less machinery; default is `dense+rerank`. Hybrid remains available via `SEARCH_MODE=hybrid` (or per-request `"search_mode": "hybrid"`) for corpora where lexical recall matters more.
- **Confidence-gated refusal.** A two-layer answerability check: a Layer-1 score gate (active only on dense + no-rerank; suppressed on the default dense+rerank path) and a Layer-2 post-generation refusal detector (primary guard on the default path). Both layers are calibrated to return no answer when retrieval confidence is too low — see the Refusal section for what is and is not yet measured.
- **Verification with an upgrade path.** Citations today are verified with regex and fuzzy token matching, a deliberately simple first pass. That layer is isolated in `tracer.py`, so it can be swapped for a cross-encoder NLI (entailment/neutral/contradiction) stage without touching retrieval or generation. See `docs/VERIFICATION.md`.

## Context

MASquery follows on from a sibling project, **Financialbench**, which benchmarked financial-domain LLMs and measured hallucination rates and answer stability on numeric questions. That benchmark surfaced two failure modes: confident fabrication, and unstable answers across reruns. MASquery requires every claim to cite a retrieved chunk, and citations are verified against retrieved chunks by surface-form matching (doc name, section, page) — which catches outright hallucinated references but not a semantically unsupported claim that reuses real metadata (see `docs/VERIFICATION.md`). The system returns a no-answer response when retrieval confidence is low (Layer-2 refusal detection handles the default path; see the Refusal section for what is and is not yet measured).

Financialbench measures the hallucination problem; MASquery addresses it in a compliance setting.

## Results

**Status:** Retrieval metrics measured on 13 answerable golden QA pairs against the indexed corpus (5 MAS PDFs, 184 pages, 200 chunks). The golden set also carries 10 adversarial pairs (out-of-corpus / fabricated / near-miss) used for the refusal check below. Generation metrics (faithfulness, correctness) require an LLM API key for the configured provider (`OPENAI_API_KEY` by default, or `ANTHROPIC_API_KEY` when `LLM_PROVIDER=anthropic`) and are reported as _API-key gated_ (evaluation not yet run — requires a configured LLM API key).

### Retrieval (measured — `scripts/run_ablation.py`, top_k=10, n=13 answerable; recall approximated via `source_section` substring match — see `docs/RETRIEVAL_ABLATION.md`)

| Configuration | Recall@5 | Recall@10 | MRR@10 | Precision@1 |
|---------------|---------:|----------:|-------:|------------:|
| Dense (FAISS) | 0.7692 | 0.7692 | 0.5513 | 0.3846 |
| **Dense + rerank (default)** | 0.6923 | 0.7692 | 0.5726 | 0.4615 |
| BM25 | 0.6923 | 0.6923 | 0.4167 | 0.3077 |
| BM25 + rerank | 0.6923 | 0.6923 | 0.5538 | 0.4615 |
| Hybrid (RRF) | 0.7692 | 0.7692 | 0.5410 | 0.3846 |
| Hybrid + rerank (opt-in) | 0.6923 | 0.7692 | 0.5726 | 0.4615 |

Reranking lifts top-of-list ordering: P@1 rises +0.08 to +0.15 across modes (dense 0.3846 → 0.4615) and MRR@10 improves. The trade-off is a small drop in recall@5 on this tiny set, where the cross-encoder pushes the one relevant chunk just outside the top 5 on a single question (dense/hybrid recall@5 0.7692 → 0.6923, recovering by recall@10). On this 13-pair initial evaluation set, hybrid does **not** beat dense: dense+rerank ties hybrid+rerank for the best MRR@10 (0.5726) and P@1 (0.4615) while needing only the dense index, so it is the default (dense+rerank and hybrid+rerank are numerically tied on this 13-pair smoke test — dense is the default because it requires less machinery, not because it dominates at scale). Interpretation caveats and next steps are in `docs/RETRIEVAL_ABLATION.md`.

### Retrieval quality vs. question (measured)

| Metric | Dense + Rerank (default) | Method |
|--------|-------------------------|--------|
| Context Relevance (mean) | **0.5546** | Cosine similarity between question and top-5 retrieved chunks, averaged over 13 questions |
| Context Relevance (min / max) | 0.4767 / 0.6984 | Per-question range |

### Refusal (current behavior under the score-scale fix)

> **Note:** On the current default path (dense+rerank), Layer-1 score gating is suppressed and Layer-2 (LLM-response pattern matching) is the primary refusal guard — Layer-2 precision/recall on adversarial pairs has not yet been measured.

The Layer-1 score gate is now **scale-aware**: `max(score) < SIMILARITY_THRESHOLD` only runs on the dense-only no-rerank path, where scores are cosine similarities. On the default dense + rerank path the score gate is suppressed (a 0.3 cosine threshold has no defensible meaning against cross-encoder logits) and Layer-1 reduces to "refuse if zero results"; refusal on adversarials is delegated to Layer-2 (`generator._detect_refusal` on the LLM response). See `docs/REFUSAL_THRESHOLD.md` for the rationale and the per-path calibrated-threshold work that would restore a Layer-1 score gate on the reranked default.

#### Layer 1 — historical numbers, pre-fix (before the score-scale gate was made scale-aware)

These numbers were measured under the **previous** gate, which compared `max(score) < 0.3` against whatever scale was returned (cosine for dense-only, CE logit for any reranked path). They are kept here for transparency about what changed; under the current code the dense+rerank default no longer fires Layer-1 on score, so these numbers do not describe today's runtime behavior.

| Slice | Outcome / total | Notes |
|-------|----------------:|-------|
| **Out-of-corpus (RBI / Fed / SEC / HKMA)** | **4 / 4** | Caught under the old gate because CE logits for out-of-corpus topics happened to fall below 0.3 — accidental, not principled. |
| **Fabricated MAS references (Notice 999, Guideline G-7890, Circular TRM-2099)** | **2 / 3** | Notice 999 (−3.02) and TRM-2099 (−4.38) fell below 0.3; G-7890 slipped through at +2.26 — the old gate's catches were accidents of the CE-logit distribution, not principled. |
| **Near-miss (capital requirement, material-outsourcing vs MAC, downtime hours)** | **2 / 3** | Old false negative: "material outsourcing vs material adverse change" — the retriever legitimately fires on the real defined term and the cross-encoder scored the top chunk at +2.43. Now (and under the old gate) that case falls to Layer-2 (`_detect_refusal`). |
| **Answerable (false-positive refusals under the pre-fix gate)** | **2 / 13 incorrectly refused** | TRM incident reporting and TRM data protection — the cross-encoder rescored all top chunks below 0.3 even though the questions were answerable. This was the bug that motivated the fix; see `docs/REFUSAL_THRESHOLD.md`. |

#### Layer 1 — current numbers (post-fix)

On the default dense+rerank path: 0/13 false-positive refusals (the score gate is suppressed) and 0/10 adversarials caught at Layer-1 by score (also suppressed). Adversarial refusal on the default path is API-key gated behind Layer-2. To exercise the cosine-scale Layer-1 gate, run with `SEARCH_MODE=vector` and `RERANK_ENABLED=false` — that path's cosine top_scores for all 10 adversarials are already recorded in `ablation_report.json` (range 0.42–0.57, all above the 0.3 gate), so that path also provides 0/10 Layer-1 adversarial catches at the current threshold.

### Generation (LLM-judged metrics not run — API key required)

| Metric | Status | How to populate |
|--------|--------|-----------------|
| Answer Faithfulness | Heuristic scores not published — re-run evaluator locally (no API key required for the metric computation, but `generate_answer` requires a key to produce the answer; no scores recorded yet); LLM-judged version not run | LLM-judged version requires an API key — re-run evaluator with a key configured |
| Answer Correctness | Heuristic scores not published — re-run evaluator locally (no API key required for the metric computation, but `generate_answer` requires a key to produce the answer; no scores recorded yet); LLM-judged version not run | Same as above |
| Layer-2 Refusal Precision | _API-key gated_ | Needs an LLM call per adversarial pair so `_detect_refusal` can inspect the response. With the post-fix Layer-1 score gate suppressed on the dense+rerank default, Layer-2 is the **primary** refusal guard on that path, so re-measuring it is the most useful open eval task. |

**Evaluation basis — golden QA set (13 answerable + 10 adversarial = 23 pairs):** Initial evaluation scale. The 13 answerable pairs cover the five ingested documents:

| Topic | Count | Example question |
|-------|-------|------------------|
| TRM (Technology Risk Management) | 6 | Access control, incident reporting, data protection, IT audit, cloud, cyber security |
| Outsourcing | 3 | Notification, due diligence, ongoing monitoring |
| BCM (Business Continuity Management) | 2 | BCP contents, recovery time objectives |
| Fair Dealing | 1 | Fair dealing outcomes |
| E-Payments | 1 | Liability for unauthorised transactions |

The 10 adversarial pairs exercise the refusal path (out-of-corpus 4, fabricated MAS references 3, near-miss 3) — see `evaluation/golden_qa.yaml` and the refusal table above. See `eval/GOLDEN_QA_COVERAGE.md` (known limitations and full enumeration) and `eval/EXPANSION_PLAN.md` (scale-up plan including two-annotator protocol).

## Architecture

```
MAS PDFs -> PyMuPDF Extraction -> Section-Aware Chunking -> Sentence-Transformer Embeddings
                                                                    |
                                                              +-----+-----+
                                                              |           |
                                                         FAISS Index  BM25 Index
                                                              |           |
User Query -> Embed ----------------> FAISS Dense Search  ----+    # <-- default path
                                      (BM25 path is opt-in via SEARCH_MODE=hybrid or bm25)
                                              |
                                    Cross-Encoder Reranking (default on)
                                              |
                                     Prompt Assembly -> LLM API -> Source Tracer
                                                                        |
                                              JSON Response: answer + sources + confidence

# BM25 index and hybrid RRF fusion remain wired up but opt-in:
# set SEARCH_MODE=hybrid (or pass "search_mode": "hybrid" per-request) to
# route through the BM25 + FAISS + RRF path instead.
```

**Key design decisions:**
- **Dense retrieval + rerank (default), hybrid available as opt-in:** Default is FAISS cosine similarity with a cross-encoder rerank — the best-measured ordering on the current eval set (`docs/RETRIEVAL_ABLATION.md`). The hybrid path (BM25 + FAISS + RRF) is still implemented and opt-in via `SEARCH_MODE=hybrid`, kept because the lexical signal may help on other corpora.
- **Cross-encoder reranking:** Joint query-document scoring after retrieval for higher precision at the cost of latency.
- **Two-layer hallucination resistance:** Layer-1 (score gate) and Layer-2 (post-generation refusal-pattern detector). Layer-1 only fires on the dense + no-rerank path where the score is a cosine similarity; it is suppressed on all other paths (see `docs/REFUSAL_THRESHOLD.md`). Citation verification checks every answer against the chunks actually provided.
- **Section header prepending:** Every chunk carries its section context for self-contained retrieval.
- **Tables extracted and indexed** (`find_tables()` row-major output folded into chunks; garbled column-major duplicate not yet stripped — see `docs/EXTRACTION_LIMITATIONS.md`). Regression test: `tests/test_chunker.py::test_tables_are_consumed_into_chunks`.
- **Source tracing:** Every citation in the LLM's response is verified against the chunks actually provided.

## Quick Start (< 5 min after first install)

> **First run:** `pip install` downloads ~800 MB (torch + faiss); the embedding and rerank models (~500 MB) download on first server start. Subsequent runs are fast.

### Prerequisites
- Python 3.11+ (via conda or system Python)
- An LLM API key — OpenAI by default ([platform.openai.com](https://platform.openai.com/api-keys)), or Anthropic ([console.anthropic.com](https://console.anthropic.com/)) by setting `LLM_PROVIDER=anthropic`

### Setup

```bash
# 1. Create environment and install dependencies
conda create -n masquery python=3.11 -y
conda activate masquery
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (default provider; or set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY)

# 3. (Optional) Re-fetch MAS regulatory documents
# PDFs are included in the repo. Only needed after a zip download or Docker
# setup without a mounted volume (git cloners can skip this step).
python scripts/download_docs.py

# 4. Start the server (auto-ingests PDFs on first run)
uvicorn main:app --reload
```

### First Query

```bash
# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the access control requirements under the TRM Guidelines?"}'
```

### Docker Quick Start

```bash
# Build and run with docker-compose
cp .env.example .env
# Edit .env with your API key

docker-compose up --build

# API at http://localhost:8000
# Streamlit at http://localhost:8501
```

Or build manually:

```bash
# Run from the project root after: python scripts/download_docs.py
docker build -t masquery .
docker run -p 8000:8000 --env-file .env -v ./data:/app/data masquery
```

## Search Modes

MASquery supports three search modes combining lexical and semantic retrieval:

| Mode | Description | Best For |
|------|-------------|----------|
| `vector` (default) | FAISS cosine similarity, reranked by a cross-encoder | General use — best measured ordering on this corpus (see `docs/RETRIEVAL_ABLATION.md`) |
| `hybrid` (opt-in) | BM25 + FAISS vector search fused via Reciprocal Rank Fusion | Corpora where lexical recall (clause numbers, defined acronyms) adds meaningful signal over dense alone |
| `bm25` | BM25 lexical search only | Queries with specific regulatory terms, clause numbers, or acronyms (note: Layer-1 score gating is suppressed on this path; adversarial refusal relies entirely on Layer-2) |

Configure globally via environment variable:
```bash
SEARCH_MODE=vector  # or "hybrid" or "bm25"
```

Or per-request:
```json
{"question": "What is MAS Notice 644?", "search_mode": "bm25"}
```

### Reciprocal Rank Fusion (RRF)

In hybrid mode, results from BM25 and FAISS are combined using RRF with k=60 (the standard constant). Each document's fused score is the sum of `1/(k + rank)` across both result lists. Documents appearing in both lists get a natural boost.

## Cross-Encoder Reranking

After retrieval, results are optionally reranked using a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for improved precision. The cross-encoder scores each (query, chunk) pair jointly, which is more accurate than independent embedding similarity but slower.

- Enabled by default. Disable via `RERANK_ENABLED=false` or per-request `"rerank": false`
- Reranks the top retrieved candidates and returns the best `top_k`

## Evaluation

MASquery includes three locally-computed evaluation metrics with RAGAS-inspired names (embedding similarity + token overlap; no LLM-as-judge):

| Metric | What It Measures |
|--------|-----------------|
| **Context Relevance** | Mean question–context embedding similarity (a proxy, not RAGAS context precision) |
| **Answer Faithfulness** | Is the answer grounded in the context? (sentence-level embedding + token overlap heuristic; no API key required for this metric) |
| **Answer Correctness** | Does the answer match the expected golden answer? (semantic cosine + token F1 heuristic; no API key required for this metric) |

> *All three metrics are lightweight heuristics (embedding similarity + token overlap) — not the LLM-graded per-sentence decomposition used by the RAGAS library.*

**Note:** All three metrics are computed on the generated answer, so running the evaluator requires an LLM API key to produce the answer (see Results section above for key setup). The metric computations themselves use only local embeddings — it is the `generate_answer` step that is API-key gated.

### Running Evaluation

```bash
# Evaluate with default settings (dense vector search + reranking)
python -m evaluation.evaluator

# Evaluate hybrid (BM25 + FAISS + RRF) with reranking
python -m evaluation.evaluator --mode hybrid

# Evaluate vector-only search without reranking
python -m evaluation.evaluator --mode vector --no-rerank

# Save results to JSON
python -m evaluation.evaluator --output eval_report.json
```

Golden QA pairs are in `evaluation/golden_qa.yaml` — **23 total entries: 13 answerable + 10 adversarial**, covering TRM, BCM, Outsourcing, Fair Dealing, and E-Payments guidelines plus out-of-corpus, fabricated, and near-miss adversarial pairs. The retrieval ablation uses the 13 answerable pairs only; the 10 adversarial pairs are used for the refusal audit (see the refusal table above and `docs/REFUSAL_THRESHOLD.md`). This is an initial evaluation scale; see `eval/GOLDEN_QA_COVERAGE.md` and `eval/EXPANSION_PLAN.md` for the topic breakdown and scale-up plan.

## API Reference

### POST /query -- Ask a regulatory question
```json
// Request
{
  "question": "What are the outsourcing notification requirements?",
  "top_k": 5,
  "search_mode": "vector",
  "rerank": true
}

// Response (note: relevance_score on the default rerank path is a cross-encoder logit, unbounded — not a cosine similarity value)
{
  "answer": "Under the Guidelines on Outsourcing...",
  "confidence": "medium",
  "is_answerable": true,
  "sources": [
    {
      "document": "Outsourcing_Guidelines",
      "section": "3.1 Notification Requirements",
      "page_numbers": [8],
      "chunk_id": "Outsourcing_Guidelines_p8_c0",
      "relevance_score": 2.14,
      "verified": true,
      "text_excerpt": "..."
    }
  ],
  "query": "What are the outsourcing notification requirements?",
  "model": "gpt-4o-mini",  // configured default; override with OPENAI_MODEL env var
  "retrieval_k": 5,
  "search_mode": "vector",
  "rerank_enabled": true
}
```

### POST /ingest -- Ingest PDF documents
Processes all PDFs in `data/raw/` and builds both the FAISS vector index and BM25 lexical index.

### GET /documents -- List indexed documents
Returns all ingested documents with chunk counts and page counts.

### GET /health -- System health check
Checks FAISS index, embedding model, search mode, reranking status, and whether the LLM API key is configured.

## Deeper Documentation

- `docs/VERIFICATION.md` — What citation verification does today (regex + fuzzy), its failure modes, and the NLI upgrade path.
- `docs/REFUSAL_THRESHOLD.md` — Where refusal gating lives, current hardcoded values, and how to justify them with a precision-recall sweep.
- `docs/CONFIDENCE_SCORING.md` — What inputs feed the confidence label and how "high / medium / low" map to the UI.
- `docs/RETRIEVAL_ABLATION.md` — Measured dense vs BM25 vs hybrid ablation results (recall@5/10, MRR@10, P@1), decision record (dense+rerank chosen as default), and interpretation. Script: `scripts/run_ablation.py`.
- `docs/EXTRACTION_LIMITATIONS.md` — PyMuPDF table and footnote limitations, with mitigation options (Camelot, pdfplumber).
- `eval/GOLDEN_QA_COVERAGE.md` — Topic-by-topic enumeration of the 13 answerable golden pairs.
- `eval/EXPANSION_PLAN.md` — Plan to grow golden QA to 50+ with inter-annotator agreement.

## Configuration

All tunables are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEARCH_MODE` | `vector` | Search strategy: vector (default, dense + rerank), hybrid, bm25 |
| `RERANK_ENABLED` | `true` | Cross-encoder reranking on/off |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `CHUNK_SIZE` | `600` | Target tokens per chunk |
| `SIMILARITY_THRESHOLD` | `0.3` | Cosine similarity refusal threshold (dense + no-rerank path only; suppressed on default dense+rerank path — see `docs/REFUSAL_THRESHOLD.md`) |
| `RRF_K` | `60` | RRF fusion constant |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `LLM_PROVIDER` | `openai` | Answer-generation provider: `openai` (default) or `anthropic` |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model (used when `LLM_PROVIDER=openai`) |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` | Anthropic model (used when `LLM_PROVIDER=anthropic`) |

> **LLM determinism:** `gpt-4o-mini` supports `temperature=0`. For reproducible answers across reruns, set `LLM_PROVIDER=anthropic` (claude-haiku-4-5), which also honours `temperature=0`. The confidence label and refusal check work correctly on either provider.

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and PR:
1. **Lint** -- ruff check + format verification
2. **Test** -- pytest with all unit tests
3. **Docker** -- build verification (no push)

## Testing

```bash
conda activate masquery
python -m pytest tests/ -v
```

## Project Structure

```
├── config.py              # All configurable parameters
├── ingest.py              # PDF extraction with PyMuPDF
├── chunker.py             # Section-aware chunking
├── embeddings.py          # Sentence-transformer wrapper
├── retriever.py           # FAISS + BM25 + RRF retrieval with cross-encoder reranking (default: dense+rerank; hybrid is opt-in)
├── generator.py           # LLM API + confidence scoring
├── tracer.py              # Citation extraction + verification
├── main.py                # FastAPI endpoints
├── streamlit_app.py       # Streamlit frontend
├── evaluation/
│   ├── metrics.py         # RAGAS-inspired evaluation metrics
│   ├── evaluator.py       # Evaluation runner
│   └── golden_qa.yaml     # Golden QA pairs for evaluation
├── prompts/
│   └── system_prompt.txt
├── scripts/
│   ├── download_docs.py
│   └── run_ablation.py       # Dense vs BM25 vs hybrid retrieval comparison
├── docs/
│   ├── VERIFICATION.md       # Citation verification: today and the NLI upgrade path
│   ├── REFUSAL_THRESHOLD.md  # Refusal gating and the PR-sweep methodology
│   ├── CONFIDENCE_SCORING.md # What the confidence label measures
│   ├── RETRIEVAL_ABLATION.md # Retrieval ablation schema
│   └── EXTRACTION_LIMITATIONS.md  # PyMuPDF tables/footnotes caveats
├── eval/
│   ├── GOLDEN_QA_COVERAGE.md # Topic-by-topic enumeration of the 13 answerable golden pairs
│   └── EXPANSION_PLAN.md     # Plan to grow to 50+ pairs
├── tests/
│   ├── test_retriever.py  # Hybrid search + RRF tests
│   ├── test_evaluation.py # Evaluation metrics tests
│   ├── test_chunker.py
│   ├── test_generator.py
│   ├── test_tracer.py
│   ├── test_api.py
│   └── ...
├── Dockerfile             # Multi-stage Docker build
├── docker-compose.yml     # API + Streamlit services
├── .github/workflows/
│   └── ci.yml             # Lint + test + Docker build
└── data/
    ├── raw/               # MAS PDF documents
    └── index/             # FAISS index + metadata
```
