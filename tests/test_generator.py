"""Tests for the answer generation module."""

from unittest.mock import patch


from generator import (
    _detect_refusal,
    compute_confidence,
    format_context,
    generate_answer,
)
from retriever import RetrievalResult


def _make_retrieval_results(scores=None):
    """Create mock RetrievalResult objects for testing."""
    if scores is None:
        scores = [0.85, 0.72, 0.45]

    results = []
    for i, score in enumerate(scores):
        results.append(
            RetrievalResult(
                chunk_id=f"test_p{i}_c{i}",
                text=f"[Section: Test Section {i}] Content about regulation topic {i}.",
                doc_name=f"TestDoc_{i}",
                page_numbers=[i + 1],
                section_header=f"Test Section {i}",
                relevance_score=score,
                rank=i + 1,
            )
        )
    return results


class TestFormatContext:
    def test_formats_chunks_with_metadata(self):
        """format_context should include document, section, pages, and text."""
        results = _make_retrieval_results([0.85])
        context = format_context(results)

        assert "TestDoc_0" in context
        assert "Test Section 0" in context
        assert "Excerpt 1" in context
        assert "0.85" in context
        assert "Content about regulation topic 0" in context

    def test_multiple_results_all_included(self):
        """All results should appear in the context string."""
        results = _make_retrieval_results([0.85, 0.72, 0.45])
        context = format_context(results)

        assert "Excerpt 1" in context
        assert "Excerpt 2" in context
        assert "Excerpt 3" in context
        assert "TestDoc_0" in context
        assert "TestDoc_1" in context
        assert "TestDoc_2" in context

    def test_empty_results_returns_no_excerpts_message(self):
        """An empty result list should return a descriptive message."""
        context = format_context([])
        assert "No relevant document excerpts" in context

    def test_page_numbers_formatted_correctly(self):
        """Multiple page numbers should be comma-separated."""
        result = RetrievalResult(
            chunk_id="multi_page",
            text="Content spanning multiple pages.",
            doc_name="MultiPageDoc",
            page_numbers=[5, 6, 7],
            section_header="Section A",
            relevance_score=0.90,
            rank=1,
        )
        context = format_context([result])
        assert "5, 6, 7" in context


class TestComputeConfidence:
    def test_high_confidence(self):
        """Top score >= 0.65 with multiple supporting chunks -> high."""
        scores = [0.80, 0.70, 0.50]
        assert compute_confidence(scores) == "high"

    def test_medium_confidence(self):
        """Top score >= 0.45 but < 0.65 -> medium."""
        scores = [0.50, 0.20]
        assert compute_confidence(scores) == "medium"

    def test_low_confidence(self):
        """Top score below medium threshold -> low."""
        scores = [0.20, 0.10]
        assert compute_confidence(scores) == "low"

    def test_empty_scores_returns_low(self):
        """No scores at all should return low."""
        assert compute_confidence([]) == "low"

    def test_high_single_score_is_medium(self):
        """A single high score without supporting chunks should be medium,
        because 'high' requires >= 2 supporting chunks."""
        # Only one score above SIMILARITY_THRESHOLD (0.3)
        scores = [0.70]
        assert compute_confidence(scores) == "medium"


class TestDetectRefusal:
    def test_detects_cannot_answer(self):
        assert _detect_refusal("I cannot answer this based on the provided documents.")

    def test_detects_insufficient_info(self):
        assert _detect_refusal("There is insufficient information in the excerpts.")

    def test_detects_no_relevant_info(self):
        assert _detect_refusal("No relevant information was found in the documents.")

    def test_normal_answer_not_flagged(self):
        assert not _detect_refusal("Financial institutions must implement MFA for privileged accounts.")

    def test_detects_documents_do_not_contain(self):
        assert _detect_refusal("The documents do not contain relevant information about this topic.")


class TestGenerateAnswer:
    def test_low_score_skips_llm(self):
        """When all scores are below SIMILARITY_THRESHOLD, LLM should not be called."""
        results = _make_retrieval_results([0.10, 0.05])

        gen_result = generate_answer("What is the policy?", results)

        assert gen_result.is_answerable is False
        assert gen_result.confidence == "low"
        assert "could not find" in gen_result.answer.lower()

    def test_empty_results_skips_llm(self):
        """No results should produce an unanswerable response without calling the LLM."""
        gen_result = generate_answer("What is the policy?", [])

        assert gen_result.is_answerable is False
        assert gen_result.confidence == "low"

    def test_calls_anthropic_by_default(self):
        """When LLM_PROVIDER is anthropic, _call_claude should be invoked."""
        results = _make_retrieval_results([0.85, 0.70])

        with (
            patch("generator.LLM_PROVIDER", "anthropic"),
            patch("generator.load_system_prompt", return_value="System prompt text here that is long enough."),
            patch("generator._call_claude", return_value="MFA is required.") as mock_claude,
            patch("generator._call_openai") as mock_openai,
        ):
            gen_result = generate_answer("What is MFA?", results)

            mock_claude.assert_called_once()
            mock_openai.assert_not_called()
            assert gen_result.answer == "MFA is required."

    def test_calls_openai_when_configured(self):
        """When LLM_PROVIDER is openai, _call_openai should be invoked."""
        results = _make_retrieval_results([0.85, 0.70])

        with (
            patch("generator.LLM_PROVIDER", "openai"),
            patch("generator.load_system_prompt", return_value="System prompt text here that is long enough."),
            patch("generator._call_openai", return_value="OpenAI says MFA is needed.") as mock_openai,
            patch("generator._call_claude") as mock_claude,
        ):
            gen_result = generate_answer("What is MFA?", results)

            mock_openai.assert_called_once()
            mock_claude.assert_not_called()
            assert gen_result.answer == "OpenAI says MFA is needed."

    def test_detects_refusal_in_response(self):
        """Post-generation refusal detection should set is_answerable=False."""
        results = _make_retrieval_results([0.85, 0.70])

        with (
            patch("generator.LLM_PROVIDER", "anthropic"),
            patch("generator.load_system_prompt", return_value="System prompt text here that is long enough."),
            patch("generator._call_claude", return_value="I cannot answer this based on the provided documents."),
        ):
            gen_result = generate_answer("Unrelated question?", results)

            assert gen_result.is_answerable is False

    def test_api_failure_returns_graceful_degradation(self):
        """When the LLM API fails with a generic error, should return a degraded response."""
        results = _make_retrieval_results([0.85, 0.70])

        with (
            patch("generator.LLM_PROVIDER", "anthropic"),
            patch("generator.load_system_prompt", return_value="System prompt text here that is long enough."),
            patch("generator._call_claude", side_effect=RuntimeError("API down")),
        ):
            gen_result = generate_answer("What is the policy?", results)

            assert gen_result.is_answerable is False
            assert "unavailable" in gen_result.answer.lower()

    def test_result_includes_retrieval_scores(self):
        """GenerationResult should carry the original retrieval scores."""
        results = _make_retrieval_results([0.85, 0.72])

        with (
            patch("generator.LLM_PROVIDER", "anthropic"),
            patch("generator.load_system_prompt", return_value="System prompt text here that is long enough."),
            patch("generator._call_claude", return_value="Answer text."),
        ):
            gen_result = generate_answer("Test?", results)

            assert gen_result.retrieval_scores == [0.85, 0.72]

    def test_cached_response_avoids_duplicate_call(self):
        """The query cache should prevent duplicate LLM calls for the same inputs."""
        import generator

        results = _make_retrieval_results([0.85, 0.70])

        with (
            patch("generator.LLM_PROVIDER", "anthropic"),
            patch("generator.load_system_prompt", return_value="System prompt text here that is long enough."),
            patch("generator._call_claude", return_value="Cached answer.") as mock_claude,
        ):
            # Clear cache before test
            generator._query_cache.clear()

            first = generate_answer("What is MFA requirement?", results)
            second = generate_answer("What is MFA requirement?", results)

            # Claude should only be called once; second call should hit cache
            assert mock_claude.call_count == 1
            assert first.answer == second.answer == "Cached answer."

            # Clean up
            generator._query_cache.clear()
