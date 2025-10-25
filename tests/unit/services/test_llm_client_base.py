"""Unit tests for abstract LLM client base class.

This module tests that the LLMClient abstract base class enforces
the correct contract for provider implementations. It verifies:
- Abstract class cannot be instantiated directly
- Concrete implementations must define all abstract methods
- Type hints are correctly defined
- Method signatures match specification
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agentcore.a2a_protocol.models.llm import LLMRequest, LLMResponse, LLMUsage
from agentcore.a2a_protocol.services.llm_client_base import LLMClient


class TestAbstractLLMClient:
    """Test abstract LLMClient base class contract enforcement."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that LLMClient cannot be instantiated directly.

        The abstract base class should raise TypeError when attempting
        direct instantiation.
        """
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            LLMClient()  # type: ignore[abstract]

    def test_concrete_implementation_requires_all_methods(self) -> None:
        """Test that concrete implementations must define all abstract methods.

        A subclass that doesn't implement all abstract methods should
        raise TypeError on instantiation.
        """

        class IncompleteClient(LLMClient):
            """Incomplete implementation missing required methods."""

            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteClient()  # type: ignore[abstract]

    def test_concrete_implementation_missing_complete(self) -> None:
        """Test that concrete implementation missing complete() cannot be instantiated."""

        class ClientMissingComplete(LLMClient):
            """Implementation missing complete() method."""

            async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
                """Stream implementation."""
                yield "token"

            def _normalize_response(
                self, raw_response: object, request: LLMRequest
            ) -> LLMResponse:
                """Normalize implementation."""
                return LLMResponse(
                    content="test",
                    usage=LLMUsage(
                        prompt_tokens=10, completion_tokens=20, total_tokens=30
                    ),
                    latency_ms=100,
                    provider="test",
                    model="test-model",
                )

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ClientMissingComplete()  # type: ignore[abstract]

    def test_concrete_implementation_missing_stream(self) -> None:
        """Test that concrete implementation missing stream() cannot be instantiated."""

        class ClientMissingStream(LLMClient):
            """Implementation missing stream() method."""

            async def complete(self, request: LLMRequest) -> LLMResponse:
                """Complete implementation."""
                return LLMResponse(
                    content="test",
                    usage=LLMUsage(
                        prompt_tokens=10, completion_tokens=20, total_tokens=30
                    ),
                    latency_ms=100,
                    provider="test",
                    model="test-model",
                )

            def _normalize_response(
                self, raw_response: object, request: LLMRequest
            ) -> LLMResponse:
                """Normalize implementation."""
                return LLMResponse(
                    content="test",
                    usage=LLMUsage(
                        prompt_tokens=10, completion_tokens=20, total_tokens=30
                    ),
                    latency_ms=100,
                    provider="test",
                    model="test-model",
                )

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ClientMissingStream()  # type: ignore[abstract]

    def test_concrete_implementation_missing_normalize(self) -> None:
        """Test that concrete implementation missing _normalize_response() cannot be instantiated."""

        class ClientMissingNormalize(LLMClient):
            """Implementation missing _normalize_response() method."""

            async def complete(self, request: LLMRequest) -> LLMResponse:
                """Complete implementation."""
                return LLMResponse(
                    content="test",
                    usage=LLMUsage(
                        prompt_tokens=10, completion_tokens=20, total_tokens=30
                    ),
                    latency_ms=100,
                    provider="test",
                    model="test-model",
                )

            async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
                """Stream implementation."""
                yield "token"

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ClientMissingNormalize()  # type: ignore[abstract]


class TestConcreteImplementation:
    """Test that a complete concrete implementation can be instantiated and used."""

    def test_complete_implementation_can_be_instantiated(self) -> None:
        """Test that a complete concrete implementation can be instantiated.

        A subclass implementing all abstract methods should instantiate
        successfully.
        """

        class CompleteClient(LLMClient):
            """Complete implementation with all required methods."""

            async def complete(self, request: LLMRequest) -> LLMResponse:
                """Execute completion request."""
                return LLMResponse(
                    content="Test response",
                    usage=LLMUsage(
                        prompt_tokens=10, completion_tokens=20, total_tokens=30
                    ),
                    latency_ms=100,
                    provider="test",
                    model=request.model,
                    trace_id=request.trace_id,
                )

            async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
                """Execute streaming completion."""
                tokens = ["Hello", " ", "World"]
                for token in tokens:
                    yield token

            def _normalize_response(
                self, raw_response: object, request: LLMRequest
            ) -> LLMResponse:
                """Normalize provider response."""
                return LLMResponse(
                    content="Normalized",
                    usage=LLMUsage(
                        prompt_tokens=10, completion_tokens=20, total_tokens=30
                    ),
                    latency_ms=100,
                    provider="test",
                    model=request.model,
                    trace_id=request.trace_id,
                )

        # Should instantiate successfully
        client = CompleteClient()
        assert isinstance(client, LLMClient)
        assert isinstance(client, CompleteClient)

    @pytest.mark.asyncio
    async def test_complete_implementation_complete_method(self) -> None:
        """Test that complete() method works in concrete implementation."""

        class CompleteClient(LLMClient):
            """Complete implementation for testing."""

            async def complete(self, request: LLMRequest) -> LLMResponse:
                """Execute completion request."""
                return LLMResponse(
                    content="Test response",
                    usage=LLMUsage(
                        prompt_tokens=10, completion_tokens=20, total_tokens=30
                    ),
                    latency_ms=100,
                    provider="test",
                    model=request.model,
                    trace_id=request.trace_id,
                )

            async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
                """Execute streaming completion."""
                yield "token"

            def _normalize_response(
                self, raw_response: object, request: LLMRequest
            ) -> LLMResponse:
                """Normalize provider response."""
                return LLMResponse(
                    content="Normalized",
                    usage=LLMUsage(
                        prompt_tokens=10, completion_tokens=20, total_tokens=30
                    ),
                    latency_ms=100,
                    provider="test",
                    model=request.model,
                )

        client = CompleteClient()
        request = LLMRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            trace_id="trace-123",
        )

        response = await client.complete(request)

        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.trace_id == "trace-123"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_complete_implementation_stream_method(self) -> None:
        """Test that stream() method works in concrete implementation."""

        class CompleteClient(LLMClient):
            """Complete implementation for testing."""

            async def complete(self, request: LLMRequest) -> LLMResponse:
                """Execute completion request."""
                return LLMResponse(
                    content="Test response",
                    usage=LLMUsage(
                        prompt_tokens=10, completion_tokens=20, total_tokens=30
                    ),
                    latency_ms=100,
                    provider="test",
                    model=request.model,
                )

            async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
                """Execute streaming completion."""
                tokens = ["Hello", " ", "World", "!"]
                for token in tokens:
                    yield token

            def _normalize_response(
                self, raw_response: object, request: LLMRequest
            ) -> LLMResponse:
                """Normalize provider response."""
                return LLMResponse(
                    content="Normalized",
                    usage=LLMUsage(
                        prompt_tokens=10, completion_tokens=20, total_tokens=30
                    ),
                    latency_ms=100,
                    provider="test",
                    model=request.model,
                )

        client = CompleteClient()
        request = LLMRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        tokens = []
        async for token in client.stream(request):
            tokens.append(token)

        assert tokens == ["Hello", " ", "World", "!"]
        assert "".join(tokens) == "Hello World!"

    def test_complete_implementation_normalize_method(self) -> None:
        """Test that _normalize_response() method works in concrete implementation."""

        class CompleteClient(LLMClient):
            """Complete implementation for testing."""

            async def complete(self, request: LLMRequest) -> LLMResponse:
                """Execute completion request."""
                return LLMResponse(
                    content="Test response",
                    usage=LLMUsage(
                        prompt_tokens=10, completion_tokens=20, total_tokens=30
                    ),
                    latency_ms=100,
                    provider="test",
                    model=request.model,
                )

            async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
                """Execute streaming completion."""
                yield "token"

            def _normalize_response(
                self, raw_response: object, request: LLMRequest
            ) -> LLMResponse:
                """Normalize provider response."""
                return LLMResponse(
                    content=f"Normalized: {raw_response}",
                    usage=LLMUsage(
                        prompt_tokens=5, completion_tokens=15, total_tokens=20
                    ),
                    latency_ms=50,
                    provider="test",
                    model=request.model,
                    trace_id=request.trace_id,
                )

        client = CompleteClient()
        request = LLMRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            trace_id="trace-456",
        )

        raw_response = {"raw": "data"}
        normalized = client._normalize_response(raw_response, request)

        assert isinstance(normalized, LLMResponse)
        assert normalized.content == f"Normalized: {raw_response}"
        assert normalized.model == "test-model"
        assert normalized.trace_id == "trace-456"
        assert normalized.usage.prompt_tokens == 5
        assert normalized.usage.total_tokens == 20


class TestTypeHints:
    """Test that type hints are correctly defined."""

    def test_llm_client_is_abc(self) -> None:
        """Test that LLMClient properly inherits from ABC."""
        from abc import ABC

        assert issubclass(LLMClient, ABC)

    def test_method_signatures_exist(self) -> None:
        """Test that abstract methods have correct signatures."""
        import inspect

        # Get abstract methods
        abstract_methods = {
            name
            for name, method in inspect.getmembers(LLMClient, inspect.isfunction)
            if getattr(method, "__isabstractmethod__", False)
        }

        # Verify expected abstract methods exist
        assert "complete" in abstract_methods
        assert "stream" in abstract_methods
        assert "_normalize_response" in abstract_methods

    def test_complete_signature(self) -> None:
        """Test that complete() has correct signature."""
        import inspect

        sig = inspect.signature(LLMClient.complete)
        params = list(sig.parameters.keys())

        assert params == ["self", "request"]
        # With deferred annotations, annotation is a string
        assert sig.parameters["request"].annotation in (LLMRequest, "LLMRequest")
        # Return annotation is available but checking async return is complex

    def test_stream_signature(self) -> None:
        """Test that stream() has correct signature."""
        import inspect

        sig = inspect.signature(LLMClient.stream)
        params = list(sig.parameters.keys())

        assert params == ["self", "request"]
        # With deferred annotations, annotation is a string
        assert sig.parameters["request"].annotation in (LLMRequest, "LLMRequest")

    def test_normalize_signature(self) -> None:
        """Test that _normalize_response() has correct signature."""
        import inspect

        sig = inspect.signature(LLMClient._normalize_response)
        params = list(sig.parameters.keys())

        assert params == ["self", "raw_response", "request"]
        # With deferred annotations, annotations are strings
        assert sig.parameters["raw_response"].annotation in (object, "object")
        assert sig.parameters["request"].annotation in (LLMRequest, "LLMRequest")
        assert sig.return_annotation in (LLMResponse, "LLMResponse")
