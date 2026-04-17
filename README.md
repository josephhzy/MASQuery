# MAS RegQ&A -- RAG-based Regulatory Document Q&A with Source Tracing

[![CI](https://github.com/josephhzy/MASquery/actions/workflows/ci.yml/badge.svg)](https://github.com/josephhzy/MASquery/actions/workflows/ci.yml)

A Retrieval-Augmented Generation (RAG) pipeline for answering questions about MAS (Monetary Authority of Singapore) regulatory documents. Every answer is grounded in retrieved source chunks with citations that are verified against the actual context provided to the model -- built to reduce hallucination risk in compliance-critical contexts.

> **Prototype scope:** Citation verification is regex-based and fuzzy-matched. Answers without explicit citations are flagged as unverified. This is a strong foundation for traceable RAG, not a guarantee of zero hallucination. See `docs/VERIFICATION.md` for the upgrade path to NLI-backed entailment scoring.

Demo: run `make streamlit` locally — public deployment pending.

## What's Novel

- **Dense retrieval + cross-encoder rerank (measured default).** Design started with hybrid on the hypothesis that regulatory queries benefit from both lexical and semantic recall. Measured ablation (see `docs/RETRIEVAL_ABLATION.md`) showed dense + cross-encoder reranking outperforms on the current eval set; default is `dense+rerank`. Hybrid remains available as `--mode hybrid` / `SEARCH_MODE=hybrid` for corpora where lexical recall matters more.
- **Confidence-gated refusal.** A two-layer answerability check: a pre-generation retrieval score filter and a post-generation refusal-pattern detector. The system is designed to say "I don't know" rather than hallucinate.
- **NLI-upgrade-path verification.** Citations today are verified with regex + fuzzy token matching (honest prototype). The architecture keeps that layer isolated in `tracer.py` so it can be swapped for a cross-encoder NLI (entailment/neutral/contradiction) stage without touching retrieval or generation. See `docs/VERIFICATION.md`.

## Context

MASquery is the architectural response to findings from a sibling project, **Financialbench**, which benchmarked financial-domain LLMs and measured hallucination rates and answer stability on numeric questions. That benchmark surfaced two failure modes: confident fabrication and unstable answers across reruns. MASquery is the other half of the story — a RAG system that *prevents* those failures by construction: every claim must come from a retrieved chunk, and the system refuses rather than guesses when retrieval is weak.

Read together: Financialbench measures the hallucination problem; MASquery is a reference design for mitigating it in a compliance-critical domain.

## Results

**Status:** Retrieval metrics measured on 13 answerable golden QA pairs against the indexed corpus (5 MAS PDFs, 184 pages, 200 chunks). The golden set also carries 10 adversarial pairs (out-of-corpus / fabricated / near-miss) used for the refusal check below. Generation metrics (faithfulness, correctness) require an Anthropic API key and are reported as _API-key gated_ until a run is captured.

### Retrieval (measured — `scripts/run_ablation.py`, top_k=10, n=13 answerable)

| Configuration | Recall@5 | Recall@10 | MRR@10 | Precision@1 |
|---------------|---------:|----------:|-------:|------------:|
| Dense (FAISS) | 0.7692 | 0.7692 | 0.5513 | 0.3846 |
| **Dense + rerank (default)** | 0.7692 | 0.7692 | 0.6179 | 0.5385 |
| BM25 | 0.6923 | 0.6923 | 0.4167 | 0.3077 |
| BM25 + rerank | 0.6923 | 0.6923 | 0.5538 | 0.4615 |
| Hybrid (RRF) | 0.7692 | 0.7692 | 0.5410 | 0.3846 |
| Hybrid + rerank (opt-in) | 0.6923 | 0.7692 | 0.5726 | 0.4615 |

Reranking clearly improves ordering (P@1 lifts ~+0.08–0.15 across modes). On this 13-pair smoke-test, hybrid does **not** beat dense — the cross-encoder rerank on dense alone produces the top MRR@10 (0.6179) and P@1 (0.5385). Interpretation caveats and next steps are in `docs/RETRIEVAL_ABLATION.md`.

### Retrieval quality vs. question (measured)

| Metric | Hybrid + Rerank | Method |
|--------|-----------------|--------|
| Context Relevance (mean) | **0.5405** | Cosine similarity between question and top-5 retrieved chunks, averaged over 13 questions |
| Context Relevance (min / max) | 0.4616 / 0.6984 | Per-question range |

### Refusal (measured on adversarial subset — Layer 1 pre-gate only, no LLM call)

| Slice | Correctly refused / total | Notes |
|-------|--------------------------:|-------|
| **Out-of-corpus (RBI / Fed / SEC / HKMA)** | **4 / 4** | Pre-gate catches all four cross-regulator queries under dense+rerank |
| **Fabricated MAS references (Notice 999, Guideline G-7890, Circular TRM-2099)** | **3 / 3** | All fabricated-identifier queries refused by Layer 1 |
| **Near-miss (capital requirement, material-outsourcing vs MAC, downtime hours)** | **2 / 3** | One false negative: the "material outsourcing vs material adverse change" conflation legitimately fires the retriever on the real defined term. That case is expected to fall to Layer 2 (`generator._detect_refusal`), which requires an LLM call to verify end-to-end. |
| **Answerable (false-positive refusals under pre-gate)** | **2 / 13** | TRM incident reporting and TRM data protection are rescored below the current `max(score) < 0.3` threshold by the cross-encoder. Documented asymmetry — CE logits vs cosine threshold — see `docs/REFUSAL_THRESHOLD.md`. |

### Generation (not run — requires `ANTHROPIC_API_KEY`)

| Metric | Status | How to populate |
|--------|--------|-----------------|
| Answer Faithfulness | _API-key gated_ | Set `ANTHROPIC_API_KEY`, then run `python -m evaluation.evaluator --output eval_report.json` |
| Answer Correctness | _API-key gated_ | Same as above |
| Layer-2 Refusal Precision | _API-key gated_ | Needs an LLM call per adversarial pair so `_detect_refusal` can inspect the response. Layer-1 numbers above cover the pre-LLM pre-gate only. |

**Evaluation basis — golden QA set (13 answerable + 10 adversarial = 23 pairs):** Smoke-test scale. The 13 answerable pairs cover the five ingested documents:

| Topic | Count | Example question |
|-------|-------|------------------|
| TRM (Technology Risk Management) | 6 | Access control, incident reporting, data protection, IT audit, cloud, cyber security |
| Outsourcing | 3 | Notification, due diligence, ongoing monitoring |
| BCM (Business Continuity Management) | 2 | BCP contents, recovery time objectives |
| Fair Dealing | 1 | Fair dealing outcomes |
| E-Payments | 1 | Liability for unauthorised transactions |

The 10 adversarial pairs exercise the refusal path (out-of-corpus 4, fabricated MAS references 3, near-miss 3) — see `evaluation/golden_qa.yaml` and the refusal table above. See `eval/GOLDEN_QA_COVERAGE.md` for the full enumeration of the answerable set and `eval/EXPANSION_PLAN.md` for the plan to grow to 50+ pairs with inter-annotator agreement.

## Architecture

```
MAS PDFs -> PyMuPDF Extraction -> Section-Aware Chunking -> Sentence-Transformer Embeddings
                                                                    |
                                                              +-----+-----+
                                                              |           |
                                                         FAISS Index  BM25 Index
                                                              |           |
User Query -> Embed ----------------> FAISS Dense Search  ----+   (default)
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
- **Dense retrieval + rerank (default), hybrid available as opt-in:** Default is FAISS cosine similarity with a cross-encoder rerank — the best-measured ordering on the current eval set (`docs/RETRIEVAL_ABLATION.md`). The hybrid path (BM25 + FAISS + RRF) is still implemented and opt-in via `SEARCH_MODE=hybrid`, kept because the lexical signal may earn its keep on other corpora.
- **Cross-encoder reranking:** Joint query-document scoring after retrieval for higher precision at the cost of latency.
- **Two-layer hallucination resistance:** A Layer-1 retrieval-score pre-gate (active on the dense-only, no-rerank path — the threshold is a cosine figure and is not on-scale for the reranked default, see `docs/REFUSAL_THRESHOLD.md`) plus a Layer-2 post-generation refusal-pattern detector on the LLM output, and citation verification of every answer against the chunks actually provided.
- **Section header prepending:** Every chunk carries its section context for self-contained retrieval.
- **Tables preserved into the index:** PyMuPDF's `find_tables()` output is wrapped in explicit `[Table start]` / `[Table end]` markers and folded into the chunk text, so the cleanly-extracted table is what gets retrieved rather than PyMuPDF's column-interleaved body text for the same region. See `docs/EXTRACTION_LIMITATIONS.md` for the pathology this avoids and `tests/test_chunker.py::test_tables_are_consumed_into_chunks` for the regression test.
- **Source tracing:** Every citation in the LLM's response is verified against the chunks actually provided.

## Quick Start (< 5 minutes)

### Prerequisites
- Python 3.11+ (via conda or system Python)
- Anthropic API key ([get one here](https://console.anthropic.com/))

### Setup

```bash
# 1. Create environment and install dependencies
conda create -n masquery python=3.11 -y
conda activate masquery
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 3. Download MAS regulatory documents
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
docker build -t masquery .
docker run -p 8000:8000 --env-file .env -v ./data:/app/data masquery
```

## Search Modes

MASquery supports three search modes combining lexical and semantic retrieval:

| Mode | Description | Best For |
|------|-------------|----------|
| `vector` (default) | FAISS cosine similarity, reranked by a cross-encoder | General use — best measured ordering on this corpus (see `docs/RETRIEVAL_ABLATION.md`) |
| `hybrid` (opt-in) | BM25 + FAISS vector search fused via Reciprocal Rank Fusion | Corpora where lexical recall (clause numbers, defined acronyms) adds meaningful signal over dense alone |
| `bm25` | BM25 lexical search only | Queries with specific regulatory terms, clause numbers, or acronyms |

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

After retrieval, results are optionally reranked using a cross-encoder (`ms-marco-MiniLM-L-6-v2`) for improved precision. The cross-encoder scores each (query, chunk) pair jointly, which is more accurate than independent embedding similarity but slower.

- Enabled by default. Disable via `RERANK_ENABLED=false` or per-request `"rerank": false`
- Reranks the top retrieved candidates and returns the best `top_k`

## Evaluation

MASquery includes a RAGAS-inspired evaluation module with three metrics:

| Metric | What It Measures |
|--------|-----------------|
| **Context Relevance** | Are the retrieved chunks relevant to the question? (embedding similarity) |
| **Answer Faithfulness** | Is the answer grounded in the context? (sentence-level + token overlap) |
| **Answer Correctness** | Does the answer match the expected golden answer? (semantic + F1) |

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

Golden QA pairs are in `evaluation/golden_qa.yaml` -- **23 total entries: 13 answerable + 10 adversarial**, covering TRM, BCM, Outsourcing, Fair Dealing, and E-Payments guidelines plus out-of-corpus, fabricated, and near-miss adversarial pairs. The retrieval ablation uses the 13 answerable pairs only; the 10 adversarial pairs are used for the refusal audit (see the refusal table above and `docs/REFUSAL_THRESHOLD.md`). This is a smoke-test scale; see `eval/GOLDEN_QA_COVERAGE.md` and `eval/EXPANSION_PLAN.md` for the topic breakdown and scale-up plan.

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

// Response
{
  "answer": "Under the Guidelines on Outsourcing...",
  "confidence": "high",
  "is_answerable": true,
  "sources": [
    {
      "document": "Outsourcing_Guidelines",
      "section": "3.1 Notification Requirements",
      "page_numbers": [8],
      "chunk_id": "Outsourcing_Guidelines_p8_c0",
      "relevance_score": 0.85,
      "verified": true,
      "text_excerpt": "..."
    }
  ],
  "query": "What are the outsourcing notification requirements?",
  "model": "claude-haiku-4-5-20251001",
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
- `docs/RETRIEVAL_ABLATION.md` — Schema for dense vs BM25 vs hybrid ablation on recall@5 and MRR. Script: `scripts/run_ablation.py`.
- `docs/EXTRACTION_LIMITATIONS.md` — The PyMuPDF table/footnote limitation honestly documented, with mitigation options (Camelot, pdfplumber).
- `eval/GOLDEN_QA_COVERAGE.md` — Topic-by-topic enumeration of the 13 golden pairs.
- `eval/EXPANSION_PLAN.md` — Plan to grow golden QA to 50+ with inter-annotator agreement.

## Configuration

All tunables are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEARCH_MODE` | `vector` | Search strategy: vector (default, dense + rerank), hybrid, bm25 |
| `RERANK_ENABLED` | `true` | Cross-encoder reranking on/off |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `CHUNK_SIZE` | `600` | Target tokens per chunk |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity to include |
| `RRF_K` | `60` | RRF fusion constant |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `RERANK_MODEL` | `ms-marco-MiniLM-L-6-v2` | Cross-encoder model |

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
├── retriever.py           # Hybrid search: FAISS + BM25 + RRF + reranking
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
│   ├── GOLDEN_QA_COVERAGE.md # Topic-by-topic enumeration of the 13 golden pairs
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
