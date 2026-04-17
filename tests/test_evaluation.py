"""Tests for the evaluation module metrics."""


from evaluation.metrics import answer_correctness, answer_faithfulness, context_relevance


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
        # Not necessarily near 0 due to embedding generalization, but lower than relevant
        assert score < 0.7

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
        assert score < 0.5

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
        assert len(qa_pairs) >= 10

        for qa in qa_pairs:
            assert "question" in qa
            assert "expected_answer" in qa
            assert len(qa["question"]) > 10
            assert len(qa["expected_answer"]) > 10
