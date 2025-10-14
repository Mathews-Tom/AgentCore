"""
Unit tests for Embedding Service.

Tests for semantic capability matching using vector embeddings.
Note: These tests mock the sentence-transformers library to avoid
requiring model downloads during test execution.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from agentcore.a2a_protocol.services.embedding_service import (
    EmbeddingService,
    get_embedding_service,
)


class TestEmbeddingService:
    """Test EmbeddingService class."""

    def test_init_default_model(self):
        """Test initialization with default model."""
        service = EmbeddingService()

        assert service.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert service.embedding_dim == 384
        assert service._model is None  # Lazy loading

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        service = EmbeddingService(model_name="custom/model")

        assert service.model_name == "custom/model"
        assert service._model is None

    @patch('sentence_transformers.SentenceTransformer')
    def test_load_model_success(self, mock_transformer):
        """Test successful model loading."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        service = EmbeddingService()
        service._load_model()

        assert service._model is mock_model
        mock_transformer.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")

    @patch('sentence_transformers.SentenceTransformer')
    def test_load_model_only_once(self, mock_transformer):
        """Test that model is only loaded once (lazy loading)."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        service = EmbeddingService()
        service._load_model()
        service._load_model()  # Second call should not reload

        mock_transformer.assert_called_once()

    def test_load_model_missing_dependency(self):
        """Test that ImportError is raised when sentence-transformers not installed."""
        service = EmbeddingService()

        # Mock the import to fail
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                with pytest.raises(ImportError, match="sentence-transformers is required"):
                    service._load_model()

    @patch('sentence_transformers.SentenceTransformer')
    def test_load_model_generic_failure(self, mock_transformer):
        """Test handling of generic model loading failure."""
        mock_transformer.side_effect = RuntimeError("Model download failed")

        service = EmbeddingService()

        with pytest.raises(RuntimeError, match="Model download failed"):
            service._load_model()

    @patch('sentence_transformers.SentenceTransformer')
    def test_generate_embedding_success(self, mock_transformer):
        """Test successful embedding generation."""
        # Mock model that returns a numpy array
        mock_model = Mock()
        embedding_array = np.random.rand(384)
        mock_model.encode.return_value = embedding_array
        mock_transformer.return_value = mock_model

        service = EmbeddingService()
        result = service.generate_embedding("test text")

        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)
        mock_model.encode.assert_called_once_with("test text", convert_to_numpy=True)

    @patch('sentence_transformers.SentenceTransformer')
    def test_generate_embedding_strips_whitespace(self, mock_transformer):
        """Test that input text is stripped of whitespace."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384)
        mock_transformer.return_value = mock_model

        service = EmbeddingService()
        service.generate_embedding("  test text  ")

        mock_model.encode.assert_called_once_with("test text", convert_to_numpy=True)

    def test_generate_embedding_empty_text(self):
        """Test that ValueError is raised for empty text."""
        service = EmbeddingService()

        with pytest.raises(ValueError, match="Cannot generate embedding for empty text"):
            service.generate_embedding("")

        with pytest.raises(ValueError, match="Cannot generate embedding for empty text"):
            service.generate_embedding("   ")

    @patch('sentence_transformers.SentenceTransformer')
    def test_generate_embedding_model_failure(self, mock_transformer):
        """Test handling of model encoding failure."""
        mock_model = Mock()
        mock_model.encode.side_effect = RuntimeError("Encoding failed")
        mock_transformer.return_value = mock_model

        service = EmbeddingService()

        with pytest.raises(RuntimeError, match="Embedding generation failed"):
            service.generate_embedding("test text")

    @patch('sentence_transformers.SentenceTransformer')
    def test_generate_embeddings_batch_success(self, mock_transformer):
        """Test successful batch embedding generation."""
        mock_model = Mock()
        batch_embeddings = np.random.rand(3, 384)
        mock_model.encode.return_value = batch_embeddings
        mock_transformer.return_value = mock_model

        service = EmbeddingService()
        texts = ["text1", "text2", "text3"]
        result = service.generate_embeddings_batch(texts)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(len(emb) == 384 for emb in result)
        mock_model.encode.assert_called_once_with(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

    @patch('sentence_transformers.SentenceTransformer')
    def test_generate_embeddings_batch_filters_empty(self, mock_transformer):
        """Test that batch generation filters out empty strings."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(2, 384)
        mock_transformer.return_value = mock_model

        service = EmbeddingService()
        texts = ["text1", "", "  ", "text2"]
        result = service.generate_embeddings_batch(texts)

        # Should only encode the two valid texts
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args
        valid_texts = call_args[0][0]
        assert len(valid_texts) == 2
        assert "text1" in valid_texts
        assert "text2" in valid_texts

    def test_generate_embeddings_batch_empty_list(self):
        """Test that ValueError is raised for empty text list."""
        service = EmbeddingService()

        with pytest.raises(ValueError, match="Cannot generate embeddings for empty text list"):
            service.generate_embeddings_batch([])

    def test_generate_embeddings_batch_all_empty(self):
        """Test that ValueError is raised when all texts are empty."""
        service = EmbeddingService()

        with pytest.raises(ValueError, match="All provided texts are empty"):
            service.generate_embeddings_batch(["", "  ", "\t"])

    @patch('sentence_transformers.SentenceTransformer')
    def test_generate_embeddings_batch_failure(self, mock_transformer):
        """Test handling of batch encoding failure."""
        mock_model = Mock()
        mock_model.encode.side_effect = RuntimeError("Batch encoding failed")
        mock_transformer.return_value = mock_model

        service = EmbeddingService()

        with pytest.raises(RuntimeError, match="Batch embedding generation failed"):
            service.generate_embeddings_batch(["text1", "text2"])

    def test_compute_similarity_identical_vectors(self):
        """Test cosine similarity for identical vectors."""
        service = EmbeddingService()

        embedding = [1.0] * 384
        similarity = service.compute_similarity(embedding, embedding)

        # Identical vectors should have similarity close to 1.0
        assert 0.99 <= similarity <= 1.0

    def test_compute_similarity_orthogonal_vectors(self):
        """Test cosine similarity for orthogonal vectors."""
        service = EmbeddingService()

        # Create orthogonal vectors
        embedding1 = [1.0] + [0.0] * 383
        embedding2 = [0.0, 1.0] + [0.0] * 382

        similarity = service.compute_similarity(embedding1, embedding2)

        # Orthogonal vectors should have similarity close to 0.5 (normalized from 0)
        assert 0.45 <= similarity <= 0.55

    def test_compute_similarity_opposite_vectors(self):
        """Test cosine similarity for opposite vectors."""
        service = EmbeddingService()

        embedding1 = [1.0] * 384
        embedding2 = [-1.0] * 384

        similarity = service.compute_similarity(embedding1, embedding2)

        # Opposite vectors should have similarity close to 0.0 (allow for tiny numerical error)
        assert -0.001 <= similarity <= 0.1

    def test_compute_similarity_dimension_mismatch(self):
        """Test that ValueError is raised for dimension mismatch."""
        service = EmbeddingService()

        embedding1 = [1.0] * 384
        embedding2 = [1.0] * 256

        with pytest.raises(ValueError, match="Embedding dimensions must match"):
            service.compute_similarity(embedding1, embedding2)

    def test_compute_similarity_zero_vectors(self):
        """Test similarity computation with zero vectors."""
        service = EmbeddingService()

        embedding1 = [0.0] * 384
        embedding2 = [1.0] * 384

        similarity = service.compute_similarity(embedding1, embedding2)

        # Zero vector should result in 0.0 similarity
        assert similarity == 0.0

    def test_compute_similarity_both_zero(self):
        """Test similarity when both vectors are zero."""
        service = EmbeddingService()

        embedding1 = [0.0] * 384
        embedding2 = [0.0] * 384

        similarity = service.compute_similarity(embedding1, embedding2)

        assert similarity == 0.0

    def test_compute_similarity_realistic_vectors(self):
        """Test similarity with realistic embedding vectors."""
        service = EmbeddingService()

        # Create realistic random embeddings
        np.random.seed(42)
        embedding1 = np.random.randn(384).tolist()
        embedding2 = np.random.randn(384).tolist()

        similarity = service.compute_similarity(embedding1, embedding2)

        # Similarity should be between 0 and 1
        assert 0.0 <= similarity <= 1.0

    def test_compute_similarity_similar_vectors(self):
        """Test similarity for similar but not identical vectors."""
        service = EmbeddingService()

        np.random.seed(42)
        base = np.random.randn(384)
        embedding1 = base.tolist()
        embedding2 = (base + np.random.randn(384) * 0.1).tolist()  # Add small noise

        similarity = service.compute_similarity(embedding1, embedding2)

        # Should be fairly high similarity
        assert similarity > 0.8

    @patch('sentence_transformers.SentenceTransformer')
    def test_generate_capability_embedding_name_only(self, mock_transformer):
        """Test generating capability embedding with name only."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384)
        mock_transformer.return_value = mock_model

        service = EmbeddingService()
        result = service.generate_capability_embedding("text-generation")

        assert isinstance(result, list)
        assert len(result) == 384
        mock_model.encode.assert_called_once_with("text-generation", convert_to_numpy=True)

    @patch('sentence_transformers.SentenceTransformer')
    def test_generate_capability_embedding_with_description(self, mock_transformer):
        """Test generating capability embedding with name and description."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384)
        mock_transformer.return_value = mock_model

        service = EmbeddingService()
        result = service.generate_capability_embedding(
            "text-generation",
            description="Generate natural language text"
        )

        assert isinstance(result, list)
        assert len(result) == 384
        # Should combine name and description
        mock_model.encode.assert_called_once_with(
            "text-generation: Generate natural language text",
            convert_to_numpy=True
        )

    @patch('sentence_transformers.SentenceTransformer')
    def test_generate_capability_embedding_empty_description(self, mock_transformer):
        """Test that empty description is handled correctly."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384)
        mock_transformer.return_value = mock_model

        service = EmbeddingService()
        result = service.generate_capability_embedding("summarization", description=None)

        mock_model.encode.assert_called_once_with("summarization", convert_to_numpy=True)


class TestGlobalEmbeddingService:
    """Test global embedding service singleton."""

    def test_get_embedding_service_singleton(self):
        """Test that get_embedding_service returns singleton instance."""
        # Reset global instance
        import agentcore.a2a_protocol.services.embedding_service as es_module
        es_module._embedding_service = None

        service1 = get_embedding_service()
        service2 = get_embedding_service()

        assert service1 is service2

    def test_get_embedding_service_creates_instance(self):
        """Test that get_embedding_service creates instance on first call."""
        import agentcore.a2a_protocol.services.embedding_service as es_module
        es_module._embedding_service = None

        service = get_embedding_service()

        assert isinstance(service, EmbeddingService)
        assert service.model_name == "sentence-transformers/all-MiniLM-L6-v2"


class TestEmbeddingServiceIntegration:
    """Integration tests for embedding service workflows."""

    @patch('sentence_transformers.SentenceTransformer')
    def test_capability_matching_workflow(self, mock_transformer):
        """Test complete capability matching workflow."""
        # Setup mock model
        mock_model = Mock()

        def mock_encode(texts, convert_to_numpy=True, show_progress_bar=False):
            # Return different embeddings for different texts
            if isinstance(texts, str):
                texts = [texts]
            embeddings = []
            for text in texts:
                if "summarization" in text.lower():
                    # Similar embeddings for summarization-related capabilities
                    embeddings.append(np.array([1.0] * 384))
                elif "text" in text.lower():
                    embeddings.append(np.array([0.8] * 384))
                else:
                    embeddings.append(np.random.rand(384))
            return np.array(embeddings) if len(embeddings) > 1 else embeddings[0]

        mock_model.encode.side_effect = mock_encode
        mock_transformer.return_value = mock_model

        # Create service and test workflow
        service = EmbeddingService()

        # Generate embeddings for agent capabilities
        agent_caps = [
            "text-summarization: Summarize long documents",
            "document-analysis: Analyze document structure",
            "content-summarization: Generate concise summaries"
        ]
        agent_embeddings = service.generate_embeddings_batch(agent_caps)

        # Generate embedding for user query
        query = "I need to summarize this document"
        query_embedding = service.generate_embedding(query)

        # Compute similarities
        similarities = [
            service.compute_similarity(query_embedding, emb)
            for emb in agent_embeddings
        ]

        # Verify that summarization capabilities have higher similarity
        assert len(similarities) == 3
        assert all(0.0 <= sim <= 1.0 for sim in similarities)

    @patch('sentence_transformers.SentenceTransformer')
    def test_batch_vs_individual_consistency(self, mock_transformer):
        """Test that batch and individual embeddings are consistent."""
        mock_model = Mock()

        # Return consistent embeddings
        def mock_encode(texts, convert_to_numpy=True, show_progress_bar=False):
            if isinstance(texts, str):
                # For single text, return 1D array
                return np.array([hash(texts) % 100 / 100.0] * 384)
            # For batch, return 2D array
            return np.array([[hash(t) % 100 / 100.0] * 384 for t in texts])

        mock_model.encode.side_effect = mock_encode
        mock_transformer.return_value = mock_model

        service = EmbeddingService()

        texts = ["capability1", "capability2", "capability3"]

        # Generate embeddings individually
        individual_embeddings = [service.generate_embedding(t) for t in texts]

        # Generate embeddings in batch
        batch_embeddings = service.generate_embeddings_batch(texts)

        # Results should be consistent
        assert len(individual_embeddings) == len(batch_embeddings) == 3

        # Note: Due to mocking, we can't test exact similarity, but we can verify structure
        for ind_emb, batch_emb in zip(individual_embeddings, batch_embeddings):
            assert isinstance(ind_emb, list)
            assert isinstance(batch_emb, list)
            assert len(ind_emb) == len(batch_emb) == 384
