"""
FastAPI application for MAS RegQ&A.
Four endpoints: POST /query, POST /ingest, GET /documents, GET /health.
"""

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import anthropic
import openai as openai_lib
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import embeddings
import retriever
from config import (
    ANTHROPIC_API_KEY,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    INDEX_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    RERANK_ENABLED,
    SEARCH_MODE,
    TOP_K,
)
from chunker import chunk_all_documents
from generator import generate_answer
from ingest import ingest_directory
from tracer import trace_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ingest_lock = threading.Lock()


# -- Pydantic models --

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=2000, description="The regulatory question to answer")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of chunks to retrieve")
    search_mode: Optional[str] = Field(default=None, description="Search mode: 'hybrid', 'vector', or 'bm25'")
    rerank: Optional[bool] = Field(default=None, description="Enable/disable cross-encoder reranking")

    @field_validator("question")
    @classmethod
    def question_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty or only whitespace.")
        return v.strip()


class SourceResponse(BaseModel):
    document: str
    section: str
    page_numbers: List[int]
    chunk_id: str
    relevance_score: float
    verified: bool
    text_excerpt: str


class QueryResponse(BaseModel):
    answer: str
    confidence: str
    is_answerable: bool
    sources: List[SourceResponse]
    query: str
    model: str
    retrieval_k: int
    search_mode: str
    rerank_enabled: bool


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    total_chunks: int
    index_path: str


class DocumentInfo(BaseModel):
    name: str
    chunk_count: int
    page_count: int


class DocumentsResponse(BaseModel):
    documents: List[DocumentInfo]
    total_documents: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    faiss_index_loaded: bool
    faiss_index_size: int
    llm_provider: str
    api_key_configured: bool
    embedding_model_loaded: bool
    documents_loaded: int
    search_mode: str
    rerank_enabled: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str


# -- Application lifecycle --

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load FAISS index and embedding model at startup. Auto-ingest if no index exists."""
    try:
        embeddings.get_model()
        logger.info("Embedding model loaded at startup")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")

    try:
        retriever.load_index()
        logger.info("FAISS index loaded at startup")
    except FileNotFoundError:
        logger.info("No FAISS index found -- auto-ingesting from data/raw/")
        try:
            all_pages = ingest_directory(RAW_DIR)
            if all_pages:
                pages_by_doc: dict = {}
                for page in all_pages:
                    pages_by_doc.setdefault(page.doc_name, []).append(page)
                all_chunks = chunk_all_documents(pages_by_doc)
                retriever.build_index(all_chunks)
                logger.info(f"Auto-ingest complete: {len(pages_by_doc)} documents indexed")
            else:
                logger.warning(f"No PDFs found in {RAW_DIR} -- index not built")
        except Exception as e:
            logger.error(f"Auto-ingest failed: {e}")

    # Validate LLM API key at startup
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY is not set -- /query will fail until configured")
    else:
        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY is not set -- /query will fail until configured")

    yield


app = FastAPI(
    title="MAS RegQ&A",
    description="RAG-based Q&A for MAS regulatory documents with source tracing",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add X-Process-Time header to every response for observability."""
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - start:.3f}s"
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return all HTTP errors in the consistent ErrorResponse schema."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": f"HTTP {exc.status_code}", "detail": exc.detail},
    )


# -- Endpoints --

@app.post("/query", response_model=QueryResponse, responses={503: {"model": ErrorResponse}})
def query_endpoint(request: QueryRequest):
    """Ask a question about MAS regulations. Returns answer with source tracing."""
    if not retriever.is_index_loaded():
        raise HTTPException(
            status_code=503,
            detail="No document index loaded. Run POST /ingest first.",
        )

    k = request.top_k or TOP_K

    results = retriever.search(
        request.question,
        top_k=k,
        mode=request.search_mode,
        rerank=request.rerank,
    )

    try:
        gen_result = generate_answer(request.question, results)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except (anthropic.AuthenticationError, openai_lib.AuthenticationError):
        provider = LLM_PROVIDER.capitalize()
        env_var = "ANTHROPIC_API_KEY" if LLM_PROVIDER == "anthropic" else "OPENAI_API_KEY"
        raise HTTPException(
            status_code=401,
            detail=(
                f"{provider} API key is invalid or has been revoked. "
                f"Update {env_var} in your .env file."
            ),
        )
    except (anthropic.RateLimitError, openai_lib.RateLimitError):
        raise HTTPException(
            status_code=429,
            detail=f"{LLM_PROVIDER.capitalize()} API rate limit reached. Please retry after a short wait.",
        )
    except (anthropic.APIConnectionError, openai_lib.APIConnectionError):
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach the {LLM_PROVIDER.capitalize()} API. Check your network connection.",
        )

    sources = trace_response(gen_result.answer, results)

    active_mode = request.search_mode or SEARCH_MODE
    active_rerank = request.rerank if request.rerank is not None else RERANK_ENABLED

    return QueryResponse(
        answer=gen_result.answer,
        confidence=gen_result.confidence,
        is_answerable=gen_result.is_answerable,
        sources=[
            SourceResponse(
                document=s.document,
                section=s.section,
                page_numbers=s.page_numbers,
                chunk_id=s.chunk_id,
                relevance_score=s.relevance_score,
                verified=s.verified,
                text_excerpt=s.text_excerpt,
            )
            for s in sources
        ],
        query=request.question,
        model=gen_result.model,
        retrieval_k=k,
        search_mode=active_mode,
        rerank_enabled=active_rerank,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(pdf_directory: Optional[str] = Form(default=None)):
    """
    Ingest PDF documents and build the FAISS index.

    Args:
        pdf_directory: Optional form field. Path to a directory containing PDFs.
                       Defaults to data/raw/ if not provided.
    """
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Another ingest operation is already in progress. Please wait.",
        )
    try:
        return _do_ingest(pdf_directory)
    finally:
        _ingest_lock.release()


def _do_ingest(pdf_directory: Optional[str] = None) -> IngestResponse:
    """Internal ingest logic, called under lock."""
    dir_path = Path(pdf_directory) if pdf_directory else RAW_DIR

    try:
        resolved = dir_path.resolve()
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid directory path.")
    if resolved != PROJECT_ROOT.resolve() and PROJECT_ROOT.resolve() not in resolved.parents:
        raise HTTPException(
            status_code=400,
            detail="Directory must be within the project directory.",
        )

    if not dir_path.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {dir_path}")

    all_pages = ingest_directory(dir_path)
    if not all_pages:
        raise HTTPException(status_code=400, detail=f"No PDF files found in {dir_path}")

    pages_by_doc: dict = {}
    for page in all_pages:
        pages_by_doc.setdefault(page.doc_name, []).append(page)

    all_chunks = chunk_all_documents(pages_by_doc)

    try:
        num_indexed = retriever.build_index(all_chunks)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return IngestResponse(
        status="ingested",
        documents_processed=len(pages_by_doc),
        total_chunks=num_indexed,
        index_path=str(INDEX_DIR),
    )


@app.get("/documents", response_model=DocumentsResponse)
def documents_endpoint():
    """List all ingested documents and their chunk counts."""
    docs = retriever.get_indexed_documents()
    total_chunks = sum(d["chunk_count"] for d in docs)

    return DocumentsResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total_documents=len(docs),
        total_chunks=total_chunks,
    )


@app.get("/health", response_model=HealthResponse)
def health_endpoint():
    """
    Health check: verifies FAISS index, embedding model, and API key.
    Returns component-level status.
    """
    faiss_loaded = retriever.is_index_loaded()
    faiss_size = retriever.get_index_size()
    model_loaded = embeddings._model is not None

    api_key_configured = bool(OPENAI_API_KEY if LLM_PROVIDER == "openai" else ANTHROPIC_API_KEY)

    docs = retriever.get_indexed_documents()
    if faiss_loaded and api_key_configured and model_loaded:
        status = "healthy"
    elif faiss_loaded or model_loaded:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        faiss_index_loaded=faiss_loaded,
        faiss_index_size=faiss_size,
        llm_provider=LLM_PROVIDER,
        api_key_configured=api_key_configured,
        embedding_model_loaded=model_loaded,
        documents_loaded=len(docs),
        search_mode=SEARCH_MODE,
        rerank_enabled=RERANK_ENABLED,
    )
