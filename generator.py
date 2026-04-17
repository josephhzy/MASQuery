"""
Answer generation using Anthropic Claude or OpenAI ChatGPT API.
Assembles prompt from system prompt + retrieved chunks + user query.
Includes confidence scoring and answerability detection.
Provider is selected via LLM_PROVIDER in .env ("anthropic" or "openai").
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List

import anthropic
import openai as openai_lib
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    API_MAX_RETRIES,
    API_RETRY_WAIT,
    API_TIMEOUT,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    MAX_TOKENS,
    PROMPTS_DIR,
    SIMILARITY_THRESHOLD,
    TEMPERATURE,
)
from retriever import RetrievalResult

logger = logging.getLogger(__name__)

# In-memory cache for query results: avoids duplicate LLM API calls.
# Key = hash of (question, chunk_ids, provider). Cleared on process restart.
_query_cache: Dict[str, "GenerationResult"] = {}

# Cache for system prompt
_system_prompt: str | None = None

# Cached API clients (created once, reused across calls)
_anthropic_client: anthropic.Anthropic | None = None
_openai_client: openai_lib.OpenAI | None = None


def _get_anthropic_client() -> anthropic.Anthropic:
    """Return a cached Anthropic client (created on first call)."""
    global _anthropic_client
    if _anthropic_client is None:
        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file based on .env.example."
            )
        _anthropic_client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            timeout=API_TIMEOUT,
        )
    return _anthropic_client


def _get_openai_client() -> openai_lib.OpenAI:
    """Return a cached OpenAI client (created on first call)."""
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file based on .env.example."
            )
        _openai_client = openai_lib.OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=API_TIMEOUT,
        )
    return _openai_client


def load_system_prompt() -> str:
    """Load and cache the system prompt from file."""
    global _system_prompt
    if _system_prompt is None:
        prompt_path = PROMPTS_DIR / "system_prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"System prompt not found at {prompt_path}. "
                "Ensure prompts/system_prompt.txt exists."
            )
        content = prompt_path.read_text().strip()
        if not content:
            raise ValueError(f"System prompt is empty at {prompt_path}.")
        if len(content) < 50:
            raise ValueError(
                f"System prompt appears corrupted (too short) at {prompt_path}."
            )
        _system_prompt = content
    return _system_prompt


def format_context(results: List[RetrievalResult]) -> str:
    """
    Format retrieval results into the context string for the prompt.
    Each chunk is clearly delimited with its metadata.
    """
    if not results:
        return "No relevant document excerpts were found for this query."

    parts = []
    for result in results:
        pages_str = ", ".join(str(p) for p in result.page_numbers)
        part = (
            f"--- Excerpt {result.rank} ---\n"
            f"Document: {result.doc_name}\n"
            f"Section: {result.section_header}\n"
            f"Page(s): {pages_str}\n"
            f"Relevance Score: {result.relevance_score:.2f}\n\n"
            f"{result.text}\n"
        )
        parts.append(part)

    return "\n".join(parts)


def compute_confidence(scores: List[float]) -> str:
    """
    Compute retrieval confidence from similarity scores.

    This reflects how well the retrieved chunks matched the query — NOT
    whether Claude's answer is factually correct. High retrieval confidence
    means strong semantic overlap between query and corpus; it does not
    guarantee answer quality. Use source verification in tracer.py for that.

    high: top score >= 0.65 and multiple supporting chunks
    medium: top score >= 0.45
    low: below medium threshold
    """
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


def _detect_refusal(response_text: str) -> bool:
    """Check if Claude's response indicates it cannot answer."""
    refusal_patterns = [
        r"cannot answer.*based on",
        r"documents? do not contain",
        r"do not contain (sufficient|relevant|enough) information",
        r"no relevant information",
        r"not covered in.*excerpt",
        r"insufficient information",
        r"unable to (find|answer|provide)",
    ]
    lower = response_text.lower()
    return any(re.search(pattern, lower) for pattern in refusal_patterns)


@retry(
    retry=retry_if_exception_type((anthropic.APITimeoutError, anthropic.APIConnectionError)),
    stop=stop_after_attempt(API_MAX_RETRIES),
    wait=wait_exponential(multiplier=API_RETRY_WAIT, min=1, max=10),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"Claude API retry {retry_state.attempt_number}/{API_MAX_RETRIES}"
    ),
)
def _call_claude(system: str, user_message: str) -> str:
    """
    Make a Claude API call with retry logic.

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not configured.
        anthropic.AuthenticationError: If API key is invalid.
        anthropic.APITimeoutError: After all retries exhausted.
    """
    client = _get_anthropic_client()

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


@retry(
    retry=retry_if_exception_type((openai_lib.APITimeoutError, openai_lib.APIConnectionError)),
    stop=stop_after_attempt(API_MAX_RETRIES),
    wait=wait_exponential(multiplier=API_RETRY_WAIT, min=1, max=10),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"OpenAI API retry {retry_state.attempt_number}/{API_MAX_RETRIES}"
    ),
)
def _call_openai(system: str, user_message: str) -> str:
    """
    Make an OpenAI ChatGPT API call with retry logic.

    Raises:
        ValueError: If OPENAI_API_KEY is not configured.
        openai.AuthenticationError: If API key is invalid.
        openai.APITimeoutError: After all retries exhausted.
    """
    client = _get_openai_client()

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_completion_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content


@dataclass
class GenerationResult:
    """Structured result from answer generation."""
    answer: str
    confidence: str          # "high" | "medium" | "low"
    is_answerable: bool
    retrieval_scores: List[float]
    model: str = field(default_factory=lambda: OPENAI_MODEL if LLM_PROVIDER == "openai" else ANTHROPIC_MODEL)


def generate_answer(query: str, results: List[RetrievalResult]) -> GenerationResult:
    """
    Generate an answer using the configured LLM (Anthropic Claude or OpenAI),
    grounded in retrieved chunks.

    Two-layer answerability:
    1. Pre-generation: skip LLM if retrieval scores are too low
    2. Post-generation: detect refusal in LLM's response

    Args:
        query: The user's regulatory question.
        results: Retrieved chunks from FAISS.

    Returns:
        GenerationResult with answer, confidence, and metadata.
    """
    # Check query cache to avoid duplicate API calls
    chunk_ids = tuple(r.chunk_id for r in results)
    cache_key = hashlib.sha256(
        f"{query}|{chunk_ids}|{LLM_PROVIDER}".encode()
    ).hexdigest()

    if cache_key in _query_cache:
        logger.info("Query cache hit — returning cached response")
        return _query_cache[cache_key]

    scores = [r.relevance_score for r in results]
    confidence = compute_confidence(scores)

    # Layer 1: Pre-generation check — no relevant context found
    if not results or (scores and max(scores) < SIMILARITY_THRESHOLD):
        return GenerationResult(
            answer=(
                "I could not find relevant information in the indexed MAS regulatory "
                "documents to answer this question. This may mean the topic is not "
                "covered in the current document corpus, or the question may need to "
                "be rephrased."
            ),
            confidence="low",
            is_answerable=False,
            retrieval_scores=scores,
        )

    # Assemble prompt
    system_prompt = load_system_prompt()
    context = format_context(results)
    user_message = f"CONTEXT EXCERPTS:\n{context}\n\nQUESTION:\n{query}"

    # Dispatch to the configured LLM provider.
    # Surface distinct error types so callers can return the right HTTP status
    # code (401 vs 429 vs 503) instead of a generic 500.
    try:
        if LLM_PROVIDER == "openai":
            answer = _call_openai(system_prompt, user_message)
        else:
            answer = _call_claude(system_prompt, user_message)
    except (anthropic.AuthenticationError, openai_lib.AuthenticationError):
        logger.error("Invalid API key for provider: %s", LLM_PROVIDER)
        raise
    except (anthropic.RateLimitError, openai_lib.RateLimitError):
        logger.error("API rate limit reached for provider: %s", LLM_PROVIDER)
        raise
    except (anthropic.APIConnectionError, openai_lib.APIConnectionError) as e:
        logger.error("API connection error (%s): %s", LLM_PROVIDER, e)
        raise
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"LLM API call failed ({type(e).__name__}): {e}")
        # Graceful degradation: return chunks without generated answer
        return GenerationResult(
            answer=(
                "The AI model is currently unavailable. Below are the most relevant "
                "regulatory document excerpts found for your query. Please review "
                "them directly.\n\n" + context
            ),
            confidence=confidence,
            is_answerable=False,
            retrieval_scores=scores,
        )

    # Layer 2: Post-generation check — did Claude refuse?
    is_answerable = not _detect_refusal(answer)

    result = GenerationResult(
        answer=answer,
        confidence=confidence,
        is_answerable=is_answerable,
        retrieval_scores=scores,
    )

    # Cache the result for future identical queries
    _query_cache[cache_key] = result

    return result
