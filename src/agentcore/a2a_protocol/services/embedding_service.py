"""
Embedding Service

Generates vector embeddings for semantic capability matching using sentence-transformers.
Implements A2A-016: Semantic Capability Matching.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding service for generating vector representations of text.

    Uses sentence-transformers/all-MiniLM-L6-v2 model for CPU-based embedding generation.
    Generates 384-dimensional vectors (optimized for semantic search).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding service.

        Args:
            model_name: HuggingFace model name for embeddings
        """
        self.model_name = model_name
        self._model: any | None = None
        self.embedding_dim = 384  # all-MiniLM-L6-v2 output dimension

    def _load_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info(
                    f"Embedding model loaded successfully (dim={self.embedding_dim})"
                )
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for semantic matching. "
                    "Install it with: uv add sentence-transformers"
                ) from e
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding vector for a single text input.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector (384-dim)

        Raises:
            ValueError: If text is empty
            RuntimeError: If model fails to generate embedding
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        self._load_model()

        try:
            # Generate embedding
            embedding = self._model.encode(text.strip(), convert_to_numpy=True)

            # Convert to list of floats
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (more efficient than individual calls).

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If texts list is empty
            RuntimeError: If model fails to generate embeddings
        """
        if not texts:
            raise ValueError("Cannot generate embeddings for empty text list")

        # Filter out empty strings
        valid_texts = [t.strip() for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("All provided texts are empty")

        self._load_model()

        try:
            # Batch encode for better performance
            embeddings = self._model.encode(
                valid_texts, convert_to_numpy=True, show_progress_bar=False
            )

            # Convert to list of lists
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {e}") from e

    def compute_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0.0 to 1.0, higher is more similar)

        Raises:
            ValueError: If embeddings have different dimensions
        """
        if len(embedding1) != len(embedding2):
            raise ValueError(
                f"Embedding dimensions must match ({len(embedding1)} vs {len(embedding2)})"
            )

        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # Cosine similarity is in [-1, 1], normalize to [0, 1]
        return float((similarity + 1) / 2)

    def generate_capability_embedding(
        self, capability_name: str, description: str | None = None
    ) -> list[float]:
        """
        Generate embedding for an agent capability.

        Combines capability name and description for richer semantic representation.

        Args:
            capability_name: Name of the capability
            description: Optional description of the capability

        Returns:
            Embedding vector for the capability
        """
        # Combine name and description for better semantic representation
        if description:
            text = f"{capability_name}: {description}"
        else:
            text = capability_name

        return self.generate_embedding(text)


# Global embedding service instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """
    Get global embedding service instance (singleton pattern).

    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
    return _embedding_service
    return _embedding_service
