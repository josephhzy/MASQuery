"""Tests for the evaluation module metrics."""

from unittest.mock import MagicMock, patch

from evaluation.metrics import answer_correctness, answer_faithfulness, context_relevance
from evaluation.evaluator import evaluate_refusal, evaluate_all


class TestContextRelevance:
    def test_relevant_context_scores_high(self):
        """Contexts closely matching the question should score high."""
        question = "What are the access control requirements?"
        contexts = [
            "Access control requirements include multi-factor authentication for critical systems.",
            "Financial institutions must implement role-based access control.",
        ]
        score = context_relevance(question, contexts)
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # should be reasonably relevant

    def test_irrelevant_context_scores_low(self):
        """Completely unrelated contexts should score lower."""
        question = "What are the access control requirements?"
        contexts = [
            "The recipe calls for two cups of flour and one egg.",
            "Jupiter is the fifth planet from the Sun.",
        ]
        score = context_relevance(question, contexts)
        assert 0.0 <= score <= 1.0
        # Unrelated domains produce very low cosine similarity (~0.1-0.25); < 0.4 is a meaningful discriminator
        assert score < 0.4

    def test_empty_contexts_returns_zero(self):
        score = context_relevance("What is MFA?", [])
        assert score == 0.0


class TestAnswerFaithfulness:
    def test_faithful_answer_scores_high(self):
        """An answer using context language should score high."""
        answer = "Multi-factor authentication must be implemented for all critical systems."
        contexts = [
            "Multi-factor authentication should be implemented for all critical systems in the institution.",
        ]
        score = answer_faithfulness(answer, contexts)
        assert 0.0 <= score <= 1.0
        assert score > 0.4

    def test_hallucinated_answer_scores_lower(self):
        """An answer containing info NOT in context should score lower."""
        answer = "Quantum computing encryption is required for all blockchain transactions."
        contexts = [
            "Financial institutions must implement access controls and multi-factor authentication.",
        ]
        score = answer_faithfulness(answer, contexts)
        assert 0.0 <= score <= 1.0
        # A blatant hallucination (no token or semantic overlap with the context)
        # must land well below the faithful answer's > 0.4 floor — otherwise the
        # metric is not actually detecting groundedness.
        assert score < 0.4

    def test_compliance_domain_hallucination_scores_lower(self):
        """A wrong-but-plausible compliance answer sharing domain vocabulary
        must still score below a faithful answer — the metric should not be
        fooled by regulatory token overlap alone."""
        # Context states 1 business day; answer fabricates 14 days.
        contexts = [
            "Financial institutions must notify MAS within 1 business day of "
            "discovering a major IT security incident under the MAS Technology "
            "Risk Management Guidelines."
        ]
        faithful_answer = (
            "Financial institutions must notify MAS within 1 business day of "
            "discovering a major IT security incident."
        )
        hallucinated_answer = (
            "Financial institutions must notify MAS within 14 days of "
            "discovering a major IT security incident under the Guidelines."
        )
        faithful_score = answer_faithfulness(faithful_answer, contexts)
        hallucinated_score = answer_faithfulness(hallucinated_answer, contexts)
        assert faithful_score > 0.4, "faithful answer should score above 0.4"
        assert hallucinated_score < faithful_score, (
            "domain-vocabulary hallucination must score below the faithful answer"
        )

    def test_empty_answer_returns_zero(self):
        score = answer_faithfulness("", ["Some context here."])
        assert score == 0.0

    def test_empty_contexts_returns_zero(self):
        score = answer_faithfulness("Some answer here.", [])
        assert score == 0.0


class TestAnswerCorrectness:
    def test_matching_answers_score_high(self):
        """Semantically similar answers should score high."""
        answer = "Banks must implement MFA for privileged accounts."
        expected = "Financial institutions should use multi-factor authentication for privileged accounts."
        score = answer_correctness(answer, expected)
        assert 0.0 <= score <= 1.0
        assert score > 0.3

    def test_unrelated_answers_score_low(self):
        """Completely different answers should score low."""
        answer = "The weather forecast shows rain tomorrow."
        expected = "Banks must implement encryption for data at rest and in transit."
        score = answer_correctness(answer, expected)
        assert 0.0 <= score <= 1.0
        assert score < 0.3

    def test_empty_answer_returns_zero(self):
        score = answer_correctness("", "Expected answer text.")
        assert score == 0.0

    def test_empty_expected_returns_zero(self):
        score = answer_correctness("Some answer.", "")
        assert score == 0.0


class TestGoldenQALoading:
    def test_golden_qa_loads(self):
        """The golden QA YAML file should load without errors."""
        from evaluation.evaluator import load_golden_qa

        qa_pairs = load_golden_qa()
        assert len(qa_pairs) >= 23  # 13 answerable + 10 adversarial

        for qa in qa_pairs:
            assert "question" in qa
            assert "expected_answer" in qa
            assert len(qa["question"]) > 10
            assert len(qa["expected_answer"]) > 10

        adversarial = [qa for qa in qa_pairs if qa.get("should_refuse")]
        assert len(adversarial) == 10, (
            f"Expected 10 adversarial pairs, got {len(adversarial)}"
        )
        for qa in adversarial:
            assert qa.get("category"), (
                f"Adversarial pair missing non-empty 'category': {qa['question'][:60]}"
            )


class TestEvaluateRefusal:
    def _make_gen_result(self, is_answerable: bool):
        r = MagicMock()
        r.is_answerable = is_answerable
        r.confidence = 0.5
        r.answer = "I cannot answer that." if not is_answerable else "Some answer."
        return r

    def _make_search_results(self, n=2):
        results = []
        for i in range(n):
            m = MagicMock()
            m.relevance_score = 0.6 + i * 0.05
            results.append(m)
        return results

    @patch("generator.generate_answer")
    @patch("retriever.search")
    def test_refused_when_not_answerable(self, mock_search, mock_gen):
        mock_search.return_value = self._make_search_results()
        mock_gen.return_value = self._make_gen_result(is_answerable=False)

        result = evaluate_refusal("What is unrelated to MAS?")

        assert result["refused"] is True
        assert result["is_answerable"] is False

    @patch("generator.generate_answer")
    @patch("retriever.search")
    def test_not_refused_when_answerable(self, mock_search, mock_gen):
        mock_search.return_value = self._make_search_results()
        mock_gen.return_value = self._make_gen_result(is_answerable=True)

        result = evaluate_refusal("What are the capital requirements?")

        assert result["refused"] is False
        assert result["is_answerable"] is True


class TestEvaluateAll:
    def _qa_pairs(self):
        return [
            {"question": "What is the capital requirement?", "expected_answer": "8% of RWA"},
            {"question": "Tell me a joke.", "expected_answer": "", "should_refuse": True, "category": "off-topic"},
        ]

    def _make_single_result(self):
        return {
            "question": "What is the capital requirement?",
            "expected_answer": "8% of RWA",
            "generated_answer": "8% of RWA",
            "confidence": 0.9,
            "is_answerable": True,
            "num_contexts": 3,
            "metrics": {
                "context_relevance": 0.8,
                "answer_faithfulness": 0.7,
                "answer_correctness": 0.75,
            },
        }

    def _make_refusal_result(self):
        return {
            "question": "Tell me a joke.",
            "is_answerable": False,
            "refused": True,
            "confidence": 0.1,
            "num_contexts": 2,
            "top_score": 0.1,
            "generated_answer": "I cannot answer that.",
        }

    @patch("evaluation.evaluator.evaluate_refusal")
    @patch("evaluation.evaluator.evaluate_single")
    def test_aggregate_refusal_recall(self, mock_single, mock_refusal):
        mock_single.return_value = self._make_single_result()
        mock_refusal.return_value = self._make_refusal_result()

        report = evaluate_all(qa_pairs=self._qa_pairs())

        adv_agg = report["adversarial"]["aggregate_metrics"]
        assert adv_agg["refusal_recall"] == 1.0
        assert adv_agg["refused"] == 1
        assert adv_agg["n"] == 1
        assert "off-topic" in adv_agg["by_category"]

    @patch("evaluation.evaluator.evaluate_refusal")
    @patch("evaluation.evaluator.evaluate_single")
    def test_error_fallback_counts_as_not_refused(self, mock_single, mock_refusal):
        """Verify the silent-failure behaviour is observable (documents the risk)."""
        mock_single.return_value = self._make_single_result()
        mock_refusal.side_effect = RuntimeError("pipeline error")

        report = evaluate_all(qa_pairs=self._qa_pairs())

        adv_agg = report["adversarial"]["aggregate_metrics"]
        # Exception path sets refused=False — refusal_recall should be 0.0
        assert adv_agg["refusal_recall"] == 0.0
        assert adv_agg["refused"] == 0
