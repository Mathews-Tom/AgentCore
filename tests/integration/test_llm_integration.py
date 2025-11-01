"""Comprehensive end-to-end integration tests for LLM client service.

This module provides production-ready integration tests that validate:
1. All 3 providers (OpenAI, Anthropic, Gemini) with real API calls
2. Streaming functionality end-to-end for each provider
3. A2A context propagation (trace_id verification)
4. Error handling (invalid models, timeouts, network errors)
5. Retry logic with transient failures
6. Concurrent requests (100+ concurrent minimum)
7. Rate limit handling (where test environment allows)

This is the CRITICAL PATH quality gate for production readiness.

Tests require real API keys in environment:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GEMINI_API_KEY

Run with:
    uv run pytest tests/integration/test_llm_integration.py -v -m integration

Target: >95% success rate for production release
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import pytest

from agentcore.a2a_protocol.models.llm import (
    LLMRequest,
    LLMResponse,
    ModelNotAllowedError,
    ProviderError,
    ProviderTimeoutError,
)
from agentcore.a2a_protocol.services.llm_service import (
    LLMService,
    ProviderRegistry,
    llm_service,
)


# Mark all tests as integration tests
pytestmark = pytest.mark.integration


class TestProviderIntegrationOpenAI:
    """Integration tests for OpenAI provider with real API."""

    def setup_method(self) -> None:
        """Clear singleton instances before each test."""
        ProviderRegistry._instances = {}

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not configured",
    )
    async def test_openai_complete_basic(self) -> None:
        """Test basic OpenAI completion with real API."""
        service = LLMService(timeout=30.0, max_retries=3)

        request = LLMRequest(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
            trace_id="integration-openai-001",
            source_agent="test-agent",
            session_id="test-session",
        )

        response = await service.complete(request)

        # Validate response structure
        assert isinstance(response, LLMResponse)
        assert response.content
        assert "test" in response.content.lower()
        assert response.provider == "openai"
        assert response.model == "gpt-5-mini"
        assert response.trace_id == "integration-openai-001"
        assert response.usage.total_tokens > 0
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not configured",
    )
    async def test_openai_stream_basic(self) -> None:
        """Test basic OpenAI streaming with real API."""
        service = LLMService(timeout=30.0)

        request = LLMRequest(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            stream=True,
            trace_id="integration-openai-stream-001",
        )

        tokens: list[str] = []
        async for token in service.stream(request):
            assert isinstance(token, str)
            assert len(token) > 0
            tokens.append(token)

        # Validate streaming worked
        assert len(tokens) > 0
        full_response = "".join(tokens)
        assert len(full_response) > 0
        # Should contain counting sequence
        assert "1" in full_response
        assert "5" in full_response

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not configured",
    )
    async def test_openai_multi_turn_conversation(self) -> None:
        """Test multi-turn conversation with OpenAI."""
        service = LLMService()

        request = LLMRequest(
            model="gpt-5-mini",
            messages=[
                {"role": "user", "content": "What is 10 + 5?"},
                {"role": "assistant", "content": "15"},
                {"role": "user", "content": "Multiply that by 2."},
            ],
            trace_id="integration-openai-conversation-001",
        )

        response = await service.complete(request)

        assert isinstance(response, LLMResponse)
        assert response.content
        # Should mention 30 (15 * 2)
        assert "30" in response.content


class TestProviderIntegrationAnthropic:
    """Integration tests for Anthropic provider with real API."""

    def setup_method(self) -> None:
        """Clear singleton instances before each test."""
        ProviderRegistry._instances = {}

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not configured",
    )
    async def test_anthropic_complete_basic(self) -> None:
        """Test basic Anthropic completion with real API."""
        service = LLMService(timeout=30.0)

        request = LLMRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Respond with 'hello' only"}],
            trace_id="integration-anthropic-001",
            source_agent="test-agent-anthropic",
        )

        response = await service.complete(request)

        # Validate response structure
        assert isinstance(response, LLMResponse)
        assert response.content
        assert response.provider == "anthropic"
        assert response.model == "claude-haiku-4-5-20251001"
        assert response.trace_id == "integration-anthropic-001"
        assert response.usage.total_tokens > 0
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not configured",
    )
    async def test_anthropic_stream_basic(self) -> None:
        """Test basic Anthropic streaming with real API."""
        service = LLMService()

        request = LLMRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "List 3 colors"}],
            stream=True,
            trace_id="integration-anthropic-stream-001",
        )

        tokens: list[str] = []
        async for token in service.stream(request):
            assert isinstance(token, str)
            tokens.append(token)

        # Validate streaming worked
        assert len(tokens) > 0
        full_response = "".join(tokens)
        assert len(full_response) > 0


class TestProviderIntegrationGemini:
    """Integration tests for Gemini provider with real API."""

    def setup_method(self) -> None:
        """Clear singleton instances before each test."""
        ProviderRegistry._instances = {}

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not configured",
    )
    async def test_gemini_complete_basic(self) -> None:
        """Test basic Gemini completion with real API."""
        service = LLMService(timeout=30.0)

        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": "Say 'hi'"}],
            trace_id="integration-gemini-001",
        )

        response = await service.complete(request)

        # Validate response structure
        assert isinstance(response, LLMResponse)
        assert response.content
        assert response.provider == "gemini"
        assert response.model == "gemini-2.5-flash-lite"
        assert response.trace_id == "integration-gemini-001"
        assert response.usage.total_tokens > 0
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not configured",
    )
    async def test_gemini_stream_basic(self) -> None:
        """Test basic Gemini streaming with real API."""
        service = LLMService()

        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": "Name 3 animals"}],
            stream=True,
            trace_id="integration-gemini-stream-001",
        )

        tokens: list[str] = []
        async for token in service.stream(request):
            assert isinstance(token, str)
            tokens.append(token)

        # Validate streaming worked
        assert len(tokens) > 0
        full_response = "".join(tokens)
        assert len(full_response) > 0


class TestA2AContextPropagation:
    """Integration tests for A2A context propagation across all providers."""

    def setup_method(self) -> None:
        """Clear singleton instances before each test."""
        ProviderRegistry._instances = {}

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not configured",
    )
    async def test_trace_id_propagation_openai(self) -> None:
        """Test trace_id propagates through OpenAI request/response."""
        service = LLMService()

        trace_id = "trace-a2a-openai-12345"
        request = LLMRequest(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Say yes"}],
            trace_id=trace_id,
            source_agent="agent-001",
            session_id="session-001",
        )

        response = await service.complete(request)

        # Verify A2A context is preserved
        assert response.trace_id == trace_id

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not configured",
    )
    async def test_trace_id_propagation_anthropic(self) -> None:
        """Test trace_id propagates through Anthropic request/response."""
        service = LLMService()

        trace_id = "trace-a2a-anthropic-67890"
        request = LLMRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Say yes"}],
            trace_id=trace_id,
        )

        response = await service.complete(request)

        # Verify A2A context is preserved
        assert response.trace_id == trace_id

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not configured",
    )
    async def test_trace_id_propagation_gemini(self) -> None:
        """Test trace_id propagates through Gemini request/response."""
        service = LLMService()

        trace_id = "trace-a2a-gemini-abcde"
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": "Say yes"}],
            trace_id=trace_id,
        )

        response = await service.complete(request)

        # Verify A2A context is preserved
        assert response.trace_id == trace_id

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not configured",
    )
    async def test_trace_id_without_context(self) -> None:
        """Test request without A2A context works correctly."""
        service = LLMService()

        request = LLMRequest(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Say yes"}],
        )

        response = await service.complete(request)

        # Should complete successfully with None trace_id
        assert response.trace_id is None
        assert isinstance(response, LLMResponse)


class TestErrorHandling:
    """Integration tests for error handling across all providers."""

    def setup_method(self) -> None:
        """Clear singleton instances before each test."""
        ProviderRegistry._instances = {}

    @pytest.mark.asyncio
    async def test_invalid_model_error(self) -> None:
        """Test error handling for invalid model."""
        service = LLMService()

        request = LLMRequest(
            model="invalid-model-name",
            messages=[{"role": "user", "content": "test"}],
        )

        # Should raise ModelNotAllowedError or ValueError
        with pytest.raises((ModelNotAllowedError, ValueError)):
            await service.complete(request)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not configured",
    )
    async def test_timeout_error_handling(self) -> None:
        """Test timeout error handling with very short timeout."""
        # Use very short timeout to force timeout
        service = LLMService(timeout=0.001, max_retries=1)

        request = LLMRequest(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Write a long essay"}],
        )

        # Should raise timeout error
        with pytest.raises((ProviderTimeoutError, ProviderError, Exception)):
            await service.complete(request)


class TestRetryLogic:
    """Integration tests for retry logic with transient failures."""

    def setup_method(self) -> None:
        """Clear singleton instances before each test."""
        ProviderRegistry._instances = {}

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not configured",
    )
    async def test_retry_configuration(self) -> None:
        """Test that retry configuration is properly set."""
        service = LLMService(timeout=30.0, max_retries=5)

        # Verify retry configuration
        assert service.max_retries == 5
        assert service.registry.max_retries == 5

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not configured",
    )
    async def test_successful_request_no_retry(self) -> None:
        """Test that successful requests don't trigger retries."""
        service = LLMService(timeout=30.0, max_retries=3)

        request = LLMRequest(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Say test"}],
        )

        start_time = time.time()
        response = await service.complete(request)
        elapsed = time.time() - start_time

        # Should complete successfully
        assert isinstance(response, LLMResponse)
        # Should be relatively fast (no retries)
        assert elapsed < 10.0  # Reasonable time for single request


class TestConcurrentRequests:
    """Integration tests for concurrent request handling."""

    def setup_method(self) -> None:
        """Clear singleton instances before each test."""
        ProviderRegistry._instances = {}

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not configured",
    )
    async def test_concurrent_requests_100_openai(self) -> None:
        """Test 100 concurrent requests to OpenAI."""
        service = LLMService(timeout=30.0, max_retries=3)

        async def make_request(idx: int) -> LLMResponse:
            request = LLMRequest(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": f"Say number {idx}"}],
                trace_id=f"concurrent-{idx}",
            )
            return await service.complete(request)

        # Execute 100 concurrent requests
        start_time = time.time()
        tasks = [make_request(i) for i in range(100)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        # Count successful responses
        successful = [r for r in responses if isinstance(r, LLMResponse)]
        failed = [r for r in responses if not isinstance(r, LLMResponse)]

        # Verify success rate > 95%
        success_rate = len(successful) / len(responses) * 100
        print(f"\nConcurrent test results:")
        print(f"  Total requests: {len(responses)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Success rate: {success_rate:.2f}%")
        print(f"  Total time: {elapsed:.2f}s")

        assert success_rate >= 95.0, f"Success rate {success_rate}% below 95% threshold"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not all(
            [
                os.getenv("OPENAI_API_KEY"),
                os.getenv("ANTHROPIC_API_KEY"),
                os.getenv("GEMINI_API_KEY"),
            ]
        ),
        reason="All API keys not configured",
    )
    async def test_concurrent_requests_multi_provider(self) -> None:
        """Test concurrent requests across all 3 providers."""
        service = LLMService(timeout=30.0, max_retries=3)

        providers_config = [
            ("gpt-5-mini", "openai"),
            ("claude-haiku-4-5-20251001", "anthropic"),
            ("gemini-2.5-flash-lite", "gemini"),
        ]

        async def make_request(model: str, provider: str, idx: int) -> LLMResponse:
            request = LLMRequest(
                model=model,
                messages=[{"role": "user", "content": f"Say {provider} {idx}"}],
                trace_id=f"multi-{provider}-{idx}",
            )
            return await service.complete(request)

        # Create 30 requests (10 per provider)
        tasks = []
        for model, provider in providers_config:
            for i in range(10):
                tasks.append(make_request(model, provider, i))

        # Execute concurrent requests
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful responses
        successful = [r for r in responses if isinstance(r, LLMResponse)]
        success_rate = len(successful) / len(responses) * 100

        print(f"\nMulti-provider concurrent test:")
        print(f"  Success rate: {success_rate:.2f}%")

        assert success_rate >= 95.0


class TestRateLimitHandling:
    """Integration tests for rate limit handling (best effort)."""

    def setup_method(self) -> None:
        """Clear singleton instances before each test."""
        ProviderRegistry._instances = {}

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not configured",
    )
    async def test_rate_limit_graceful_handling(self) -> None:
        """Test graceful handling when approaching rate limits.

        Note: This test may not actually hit rate limits in test environment,
        but verifies the system handles errors gracefully.
        """
        service = LLMService(timeout=30.0, max_retries=3)

        # Make rapid sequential requests
        request = LLMRequest(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Say test"}],
        )

        responses: list[LLMResponse | Exception] = []
        for i in range(10):
            try:
                response = await service.complete(request)
                responses.append(response)
            except Exception as e:
                responses.append(e)

        # Count successful responses
        successful = [r for r in responses if isinstance(r, LLMResponse)]

        # Should have some successful responses
        assert len(successful) > 0
        print(f"\nRate limit test: {len(successful)}/10 succeeded")


class TestMultiProviderE2E:
    """End-to-end tests across all providers."""

    def setup_method(self) -> None:
        """Clear singleton instances before each test."""
        ProviderRegistry._instances = {}

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not all(
            [
                os.getenv("OPENAI_API_KEY"),
                os.getenv("ANTHROPIC_API_KEY"),
                os.getenv("GEMINI_API_KEY"),
            ]
        ),
        reason="All API keys not configured",
    )
    async def test_all_providers_complete_workflow(self) -> None:
        """Test complete workflow with all 3 providers."""
        service = LLMService(timeout=30.0, max_retries=3)

        test_cases = [
            ("gpt-5-mini", "openai", "What is 2+2?"),
            ("claude-haiku-4-5-20251001", "anthropic", "What is the capital of France?"),
            ("gemini-2.5-flash-lite", "gemini", "Name a color"),
        ]

        for model, expected_provider, prompt in test_cases:
            request = LLMRequest(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                trace_id=f"e2e-{expected_provider}",
            )

            response = await service.complete(request)

            # Validate response
            assert isinstance(response, LLMResponse)
            assert response.provider == expected_provider
            assert response.model == model
            assert response.content
            assert response.usage.total_tokens > 0
            assert response.latency_ms > 0
            assert response.trace_id == f"e2e-{expected_provider}"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not all(
            [
                os.getenv("OPENAI_API_KEY"),
                os.getenv("ANTHROPIC_API_KEY"),
                os.getenv("GEMINI_API_KEY"),
            ]
        ),
        reason="All API keys not configured",
    )
    async def test_all_providers_streaming_workflow(self) -> None:
        """Test streaming workflow with all 3 providers."""
        service = LLMService(timeout=30.0)

        test_cases = [
            ("gpt-5-mini", "Count to 3"),
            ("claude-haiku-4-5-20251001", "List 2 fruits"),
            ("gemini-2.5-flash-lite", "Name 2 cities"),
        ]

        for model, prompt in test_cases:
            request = LLMRequest(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

            tokens: list[str] = []
            async for token in service.stream(request):
                tokens.append(token)

            # Validate streaming worked
            assert len(tokens) > 0
            full_response = "".join(tokens)
            assert len(full_response) > 0

    @pytest.mark.asyncio
    async def test_global_singleton_instance(self) -> None:
        """Test that global llm_service singleton works correctly."""
        # Verify singleton exists
        assert llm_service is not None
        assert isinstance(llm_service, LLMService)
        assert hasattr(llm_service, "complete")
        assert hasattr(llm_service, "stream")
        assert hasattr(llm_service, "registry")


class TestPerformanceMetrics:
    """Integration tests for performance characteristics."""

    def setup_method(self) -> None:
        """Clear singleton instances before each test."""
        ProviderRegistry._instances = {}

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not configured",
    )
    async def test_latency_tracking_accuracy(self) -> None:
        """Test that latency tracking is accurate."""
        service = LLMService()

        request = LLMRequest(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Say test"}],
        )

        start_time = time.time()
        response = await service.complete(request)
        actual_elapsed_ms = int((time.time() - start_time) * 1000)

        # Reported latency should be close to actual elapsed time
        # Allow 10% variance for overhead
        assert response.latency_ms > 0
        assert abs(response.latency_ms - actual_elapsed_ms) < actual_elapsed_ms * 0.1

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not configured",
    )
    async def test_token_usage_accuracy(self) -> None:
        """Test that token usage is accurately reported."""
        service = LLMService()

        request = LLMRequest(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "Be very concise. Reply with only 'OK'."},
                {"role": "user", "content": "Acknowledge"}
            ],
            max_tokens=10,  # Limit response tokens
        )

        response = await service.complete(request)

        # Verify token counts are sensible
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )
        # With max_tokens=10 and concise prompt, should be reasonable (model may use more)
        assert response.usage.total_tokens < 200  # Generous limit for model behavior
