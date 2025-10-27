"""Pytest-benchmark integration for LLM client service.

This module provides pytest-compatible benchmarks using pytest-benchmark plugin.
These benchmarks can be run as part of the test suite and integrated with CI/CD.

Run with:
    uv run pytest tests/benchmarks/test_llm_benchmarks.py --benchmark-only
    uv run pytest tests/benchmarks/test_llm_benchmarks.py --benchmark-compare
    uv run pytest tests/benchmarks/test_llm_benchmarks.py --benchmark-autosave

For comparison with previous runs:
    uv run pytest tests/benchmarks/test_llm_benchmarks.py --benchmark-compare=0001

To see histogram:
    uv run pytest tests/benchmarks/test_llm_benchmarks.py --benchmark-histogram

These benchmarks focus on microbenchmarks (no API calls) to avoid costs.
For full benchmarks including API calls, use scripts/benchmark_llm.py.
"""

from __future__ import annotations

import pytest

from agentcore.a2a_protocol.models.llm import LLMRequest
from agentcore.a2a_protocol.services.llm_service import (
    MODEL_PROVIDER_MAP,
    ProviderRegistry)


class TestMicrobenchmarks:
    """Microbenchmarks for LLM client service operations (no network calls)."""

    def setup_method(self) -> None:
        """Clear singleton instances before each test."""
        ProviderRegistry._instances = {}

    def test_benchmark_model_lookup(self, benchmark: pytest.BenchmarkFixture) -> None:
        """Benchmark model-to-provider mapping lookup.

        Target: <0.1ms (100 microseconds)
        """

        def lookup() -> str | None:
            return MODEL_PROVIDER_MAP.get("gpt-4.1-mini")

        benchmark(lookup)

    def test_benchmark_provider_selection(
        self, benchmark: pytest.BenchmarkFixture
    ) -> None:
        """Benchmark provider selection from model string.

        Target: <1ms (1000 microseconds)
        """
        registry = ProviderRegistry()

        def select_provider() -> bool:
            try:
                registry.get_provider_for_model("gpt-4.1-mini")
                return True
            except RuntimeError:
                # Expected if API key not configured
                return False

        benchmark(select_provider)

    def test_benchmark_request_validation(
        self, benchmark: pytest.BenchmarkFixture
    ) -> None:
        """Benchmark LLMRequest Pydantic validation.

        Target: <1ms (1000 microseconds)
        """

        def validate_request() -> LLMRequest:
            return LLMRequest(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": "test"}], trace_id="bench-001")

        result = benchmark(validate_request)
        assert result is not None
        assert result.model == "gpt-4.1-mini"

    def test_benchmark_list_available_models(
        self, benchmark: pytest.BenchmarkFixture
    ) -> None:
        """Benchmark listing available models.

        Target: <0.5ms (500 microseconds)
        """
        registry = ProviderRegistry()

        def list_models() -> list[str]:
            return registry.list_available_models()

        result = benchmark(list_models)
        assert isinstance(result, list)
        assert len(result) > 0


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4.1-mini",
        "claude-3-5-haiku-20241022",
        "gemini-2.0-flash-exp",
    ])
class TestModelSpecificBenchmarks:
    """Benchmarks for each supported model."""

    def setup_method(self) -> None:
        """Clear singleton instances before each test."""
        ProviderRegistry._instances = {}

    def test_benchmark_provider_lookup_by_model(
        self, benchmark: pytest.BenchmarkFixture, model: str
    ) -> None:
        """Benchmark provider lookup for specific model.

        Target: <0.5ms per model
        """
        registry = ProviderRegistry()

        def lookup() -> bool:
            try:
                registry.get_provider_for_model(model)
                return True
            except RuntimeError:
                return False

        benchmark(lookup)


# Benchmark groups for easy filtering
pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.microbenchmark,
]
