"""
Central configuration for MAS RegQ&A pipeline.
All tunable parameters live here — no magic numbers in other modules.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# ── Chunking ───────────────────────────────────────────────────────────
CHUNK_SIZE = 600  # target tokens (middle of 500-800 range)
CHUNK_OVERLAP = 100  # token overlap between chunks
CHUNK_MIN_SIZE = 50  # discard chunks smaller than this (tokens)

# Separators tuned for regulatory documents (priority order)
CHUNK_SEPARATORS = [
    "\n\n",  # paragraph breaks (strongest boundary)
    "\n",  # line breaks
    ". ",  # sentence boundaries
    "; ",  # clause boundaries
    ", ",  # last resort
]

# ── Embeddings ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ── Retrieval ──────────────────────────────────────────────────────────
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3  # minimum cosine similarity to include

# ── Hybrid search ─────────────────────────────────────────────────────
# "vector" (FAISS only, current default — dense + rerank won the ablation,
# see docs/RETRIEVAL_ABLATION.md), "hybrid" (BM25 + vector + RRF, opt-in),
# "bm25" (BM25 only).
VALID_SEARCH_MODES = ("hybrid", "vector", "bm25")
SEARCH_MODE = os.getenv("SEARCH_MODE", "vector").lower()
if SEARCH_MODE not in VALID_SEARCH_MODES:
    raise ValueError(f"Invalid SEARCH_MODE={SEARCH_MODE!r}. Must be 'hybrid', 'vector', or 'bm25'.")
RRF_K = 60  # Reciprocal Rank Fusion constant (standard default)
BM25_TOP_K_MULTIPLIER = 3  # fetch more from BM25 before fusing (it's cheap)

# ── Reranking ─────────────────────────────────────────────────────────
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Confidence scoring ─────────────────────────────────────────────────
CONFIDENCE_HIGH_THRESHOLD = 0.65
CONFIDENCE_MEDIUM_THRESHOLD = 0.45

# ── LLM Provider ───────────────────────────────────────────────────────
# Default: openai + gpt-4o-mini. Set LLM_PROVIDER=anthropic in .env to swap.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # "anthropic" | "openai"
if LLM_PROVIDER not in ("anthropic", "openai"):
    raise ValueError(f"Invalid LLM_PROVIDER={LLM_PROVIDER!r}. Must be 'anthropic' or 'openai'.")

# ── Generation (Anthropic Claude) ──────────────────────────────────────
# Key is loaded here but validated lazily in generator._call_claude() so that
# ingestion-only flows and tests that don't touch generation can still run
# without an API key configured.
# Note: Check https://console.anthropic.com for the latest available models.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = "claude-haiku-4-5"

# ── Generation (OpenAI ChatGPT) ─────────────────────────────────────────
# Key is validated lazily in generator._call_openai().
# Note: Check https://platform.openai.com/docs/models for the latest models.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

MAX_TOKENS = 1024
TEMPERATURE = 0.0  # deterministic; applied on Anthropic and non-gpt-5 OpenAI paths (gpt-5 family omits this parameter)

# ── API resilience ─────────────────────────────────────────────────────
API_TIMEOUT = 30  # seconds
API_MAX_RETRIES = 3
API_RETRY_WAIT = 1  # seconds (exponential backoff base)

# ── HTTP surface hardening ─────────────────────────────────────────────
# CORS_ALLOWED_ORIGINS: comma-separated list. Use "*" only for local/demo.
# In production, set this to a concrete origin list, e.g.
#   CORS_ALLOWED_ORIGINS=https://masquery.example.com,https://internal.example
# Default is "*" to preserve current local-demo behavior.
_cors_raw = os.getenv("CORS_ALLOWED_ORIGINS", "*")
CORS_ALLOWED_ORIGINS = [o.strip() for o in _cors_raw.split(",") if o.strip()] or ["*"]

# INGEST_TOKEN: opt-in shared-secret guard for POST /ingest. When set, requests
# must carry the matching value in the X-Ingest-Token header (or the endpoint
# returns 401). When unset, /ingest stays open — fine for local development,
# but DO set this for any deployment that exposes the API to untrusted callers.
INGEST_TOKEN = os.getenv("INGEST_TOKEN") or None

# ── Index file paths ───────────────────────────────────────────────────
FAISS_INDEX_PATH = INDEX_DIR / "mas_regulations.index"
METADATA_PATH = INDEX_DIR / "mas_regulations_metadata.json"

# ── Index versioning ───────────────────────────────────────────────────
# Bump this whenever chunking strategy or embedding model changes so stale
# indexes are detected at load time rather than silently producing bad results.
INDEX_VERSION = "1"
