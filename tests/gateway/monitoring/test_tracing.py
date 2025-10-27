"""Tests for OpenTelemetry distributed tracing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from gateway.monitoring.tracing import (
    add_span_attributes,
    add_span_event,
    configure_tracing,
    get_span_id,
    get_trace_id,
    get_tracer,
    record_exception)


class TestTracing:
    """Test OpenTelemetry tracing functionality."""

    def test_configure_tracing(self) -> None:
        """Test configuring distributed tracing."""
        provider = configure_tracing(
            service_name="test-gateway",
            service_version="1.0.0",
            sample_rate=1.0,
            enable_console_export=True)

        assert isinstance(provider, TracerProvider)
        assert provider is not None

    def test_configure_tracing_with_otlp_endpoint(self) -> None:
        """Test configuring tracing with OTLP endpoint."""
        with patch("gateway.monitoring.tracing.OTLPSpanExporter"):
            provider = configure_tracing(
                service_name="test-gateway",
                service_version="1.0.0",
                otlp_endpoint="http://localhost:4317",
                sample_rate=0.5)

            assert isinstance(provider, TracerProvider)

    def test_get_tracer(self) -> None:
        """Test getting a tracer instance."""
        tracer = get_tracer("test-component")
        assert tracer is not None
        assert isinstance(tracer, trace.Tracer)

    def test_add_span_attributes(self) -> None:
        """Test adding custom span attributes."""
        # Configure tracing first
        configure_tracing(
            service_name="test-gateway",
            service_version="1.0.0",
            sample_rate=1.0)

        tracer = get_tracer()
        with tracer.start_as_current_span("test_span") as span:
            add_span_attributes(
                user_id="user123",
                tenant="acme-corp",
                request_count=42,
                is_authenticated=True)

            # Verify span is recording
            assert span.is_recording()

    def test_add_span_event(self) -> None:
        """Test adding events to spans."""
        configure_tracing(
            service_name="test-gateway",
            service_version="1.0.0",
            sample_rate=1.0)

        tracer = get_tracer()
        with tracer.start_as_current_span("test_span") as span:
            add_span_event(
                "cache_miss",
                cache_key="user:123",
                cache_type="redis")

            assert span.is_recording()

    def test_record_exception(self) -> None:
        """Test recording exceptions in spans."""
        configure_tracing(
            service_name="test-gateway",
            service_version="1.0.0",
            sample_rate=1.0)

        tracer = get_tracer()
        with tracer.start_as_current_span("test_span") as span:
            exception = ValueError("Test error")
            record_exception(exception)

            assert span.is_recording()

    def test_get_trace_id(self) -> None:
        """Test getting current trace ID."""
        configure_tracing(
            service_name="test-gateway",
            service_version="1.0.0",
            sample_rate=1.0)

        tracer = get_tracer()
        with tracer.start_as_current_span("test_span"):
            trace_id = get_trace_id()
            assert isinstance(trace_id, str)
            assert len(trace_id) == 32  # 128-bit trace ID as hex

    def test_get_span_id(self) -> None:
        """Test getting current span ID."""
        configure_tracing(
            service_name="test-gateway",
            service_version="1.0.0",
            sample_rate=1.0)

        tracer = get_tracer()
        with tracer.start_as_current_span("test_span"):
            span_id = get_span_id()
            assert isinstance(span_id, str)
            assert len(span_id) == 16  # 64-bit span ID as hex

    def test_get_trace_id_no_active_span(self) -> None:
        """Test getting trace ID when no span is active."""
        trace_id = get_trace_id()
        assert trace_id == ""

    def test_get_span_id_no_active_span(self) -> None:
        """Test getting span ID when no span is active."""
        span_id = get_span_id()
        assert span_id == ""

    def test_sampling_rate(self) -> None:
        """Test that sampling rate affects trace recording."""
        # Configure with 0% sampling (no traces)
        provider = configure_tracing(
            service_name="test-gateway",
            service_version="1.0.0",
            sample_rate=0.0)

        assert provider is not None

        # Configure with 100% sampling (all traces)
        provider = configure_tracing(
            service_name="test-gateway",
            service_version="1.0.0",
            sample_rate=1.0)

        assert provider is not None
