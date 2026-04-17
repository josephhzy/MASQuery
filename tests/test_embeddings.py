"""Tests for the embeddings module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from config import EMBEDDING_DIMENSION


class TestGetModel:
    def test_returns_model_instance(self):
        """get_model() should return a SentenceTransformer (or mock)."""
        import embeddings

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = EMBEDDING_DIMENSION

        with patch("embeddings.SentenceTransformer", return_value=mock_model):
            # Reset singleton so it reloads
            embeddings._model = None

            model = embeddings.get_model()
            assert model is not None
            assert model is mock_model

            # Clean up
            embeddings._model = None

    def test_singleton_returns_same_instance(self):
        """Calling get_model() twice should return the exact same object."""
        import embeddings

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = EMBEDDING_DIMENSION

        with patch("embeddings.SentenceTransformer", return_value=mock_model) as mock_cls:
            embeddings._model = None

            first = embeddings.get_model()
            second = embeddings.get_model()

            assert first is second
            # Constructor should only be called once (singleton)
            assert mock_cls.call_count == 1

            embeddings._model = None

    def test_dimension_mismatch_raises(self):
        """get_model() should raise ValueError if model dimension != config."""
        import embeddings

        mock_model = MagicMock()
        # Return a wrong dimension
        mock_model.get_sentence_embedding_dimension.return_value = 999

        with patch("embeddings.SentenceTransformer", return_value=mock_model):
            embeddings._model = None

            with pytest.raises(ValueError, match="dimension mismatch"):
                embeddings.get_model()

            embeddings._model = None


class TestEmbedTexts:
    def test_returns_correct_shape(self):
        """embed_texts should return an ndarray of shape (N, EMBEDDING_DIMENSION)."""
        import embeddings

        texts = ["Hello world", "Test sentence", "Another text"]
        fake_embeddings = np.random.randn(len(texts), EMBEDDING_DIMENSION).astype(np.float32)

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings

        with patch.object(embeddings, "get_model", return_value=mock_model):
            result = embeddings.embed_texts(texts)

            assert isinstance(result, np.ndarray)
            assert result.shape == (3, EMBEDDING_DIMENSION)
            assert result.dtype == np.float32

    def test_calls_model_encode_with_normalize(self):
        """embed_texts should call encode with normalize_embeddings=True."""
        import embeddings

        texts = ["Test"]
        fake_embeddings = np.random.randn(1, EMBEDDING_DIMENSION).astype(np.float32)

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings

        with patch.object(embeddings, "get_model", return_value=mock_model):
            embeddings.embed_texts(texts, batch_size=32)

            mock_model.encode.assert_called_once()
            call_kwargs = mock_model.encode.call_args
            assert call_kwargs[1]["normalize_embeddings"] is True
            assert call_kwargs[1]["batch_size"] == 32

    def test_single_text_returns_2d_array(self):
        """Even a single text should produce a 2D array of shape (1, dim)."""
        import embeddings

        fake_embeddings = np.random.randn(1, EMBEDDING_DIMENSION).astype(np.float32)

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings

        with patch.object(embeddings, "get_model", return_value=mock_model):
            result = embeddings.embed_texts(["single text"])
            assert result.shape == (1, EMBEDDING_DIMENSION)


class TestEmbedQuery:
    def test_returns_2d_array(self):
        """embed_query should return an ndarray of shape (1, EMBEDDING_DIMENSION)."""
        import embeddings

        fake_embedding = np.random.randn(1, EMBEDDING_DIMENSION).astype(np.float32)

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embedding

        with patch.object(embeddings, "get_model", return_value=mock_model):
            result = embeddings.embed_query("What is MFA?")

            assert isinstance(result, np.ndarray)
            assert result.shape == (1, EMBEDDING_DIMENSION)
            assert result.dtype == np.float32

    def test_calls_encode_with_single_item_list(self):
        """embed_query wraps the query in a list before calling encode."""
        import embeddings

        fake_embedding = np.random.randn(1, EMBEDDING_DIMENSION).astype(np.float32)

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embedding

        with patch.object(embeddings, "get_model", return_value=mock_model):
            embeddings.embed_query("test query")

            args = mock_model.encode.call_args[0]
            assert args[0] == ["test query"]
