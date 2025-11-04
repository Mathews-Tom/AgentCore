"""
End-to-end integration tests for Reasoning JSON-RPC API.
Tests complete JSON-RPC request/response flow including:
- Single requests with valid parameters
- Batch requests
- Error scenarios
- Parameter validation
- Response formatting
"""
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from agentcore.a2a_protocol.main import app
from agentcore.a2a_protocol.models.security import Permission, Role, TokenPayload


@pytest.fixture(autouse=True, scope="function")
def reset_module_state():
    """Reset module state before each test to prevent interference from other tests.

    Note: These tests pass when run in isolation but may fail in full suite
    due to test ordering issues. This fixture helps isolate the tests.
    """
    # Reload the app to ensure clean state
    import sys
    import importlib

    # Force reload of app module to clear any cached state
    if 'agentcore.a2a_protocol.main' in sys.modules:
        try:
            # Clear FastAPI app cache/state by reimporting
            app_module = sys.modules['agentcore.a2a_protocol.main']
            importlib.reload(app_module)
        except Exception:
            pass  # If reload fails, tests will still use existing app

    yield
@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def valid_token_payload():
    """Create valid token payload with reasoning:execute permission."""
    return TokenPayload.create(
        subject="agent-test",
        role=Role.AGENT,
        token_type="access",
        expiration_hours=24,
        agent_id="agent-test")


@pytest.fixture
def mock_llm_config():
    """Mock LLM configuration for testing (avoids empty API key issues)."""
    return {
        "api_key": "test-mock-api-key-12345",
        "base_url": "https://api.mock-llm.test/v1",
        "default_model": "gpt-5",
    }
def test_single_request_valid_parameters(client, valid_token_payload, mock_llm_http_client):
    """Test single JSON-RPC request with valid parameters."""
    with patch("agentcore.reasoning.services.reasoning_jsonrpc.security_service") as mock_security:
        mock_security.validate_token.return_value = valid_token_payload  # Mock auth
        response = client.post(
            "/api/v1/jsonrpc",
            json={
                "jsonrpc": "2.0",
                "method": "reasoning.bounded_context",
                "params": {
                    "auth_token": "test-token",
                    "query": "What is 2+2?",
                    "temperature": 0.7,
                    "chunk_size": 8192,
                    "carryover_size": 4096,
                    "max_iterations": 5,
                },
                "id": 1,
            })
        assert response.status_code == 200
        data = response.json()
        # Verify JSON-RPC structure
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert "result" in data
        assert "error" not in data
        # Verify result structure (E2E test with real engine + mocked LLM)
        result = data["result"]
        assert result["success"] is True
        assert result["answer"] == "42"  # Extracted from mock LLM response
        assert result["total_iterations"] == 1
        assert result["total_tokens"] == 500  # From mock LLM usage
        assert result["compute_savings_pct"] > 0  # Real engine calculates this
        assert result["execution_time_ms"] > 0  # Real timing
        assert len(result["iterations"]) == 1
        assert result["carryover_compressions"] == 0
def test_single_request_minimal_parameters(client, valid_token_payload, mock_llm_http_client):
    """Test request with only required parameters."""
    with patch("agentcore.reasoning.services.reasoning_jsonrpc.security_service") as mock_security:
        mock_security.validate_token.return_value = valid_token_payload  # Mock auth
        response = client.post(
            "/api/v1/jsonrpc",
            json={
                "jsonrpc": "2.0",
                "method": "reasoning.bounded_context",
                "params": {
                    "auth_token": "test-token",
                    "query": "Simple question",
                },
                "id": 2,
            })
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["success"] is True
def test_batch_request(client, valid_token_payload, mock_llm_http_client):
    """Test batch JSON-RPC request."""
    with patch("agentcore.reasoning.services.reasoning_jsonrpc.security_service") as mock_security:
        mock_security.validate_token.return_value = valid_token_payload  # Mock auth
        batch_request = [
            {
                "jsonrpc": "2.0",
                "method": "reasoning.bounded_context",
                "params": {"auth_token": "test-token", "query": "Question 1"},
                "id": 1,
            },
            {
                "jsonrpc": "2.0",
                "method": "reasoning.bounded_context",
                "params": {"auth_token": "test-token", "query": "Question 2"},
                "id": 2,
            },
        ]
        response = client.post("/api/v1/jsonrpc", json=batch_request)
        assert response.status_code == 200
        data = response.json()
        # Batch response should be array
        assert isinstance(data, list)
        assert len(data) == 2
        # Verify both responses
        for resp in data:
            assert resp["jsonrpc"] == "2.0"
            assert "result" in resp
            assert resp["result"]["success"] is True
def test_error_invalid_parameters(client):
    """Test error handling for invalid parameters."""
    response = client.post(
        "/api/v1/jsonrpc",
        json={
            "jsonrpc": "2.0",
            "method": "reasoning.bounded_context",
            "params": {
                "query": "Test",
                "chunk_size": 4096,
                "carryover_size": 4096,  # Invalid: equal to chunk_size
            },
            "id": 3,
        })
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert "result" not in data
    assert data["error"]["code"] == -32603  # Internal error
    # Error message wrapped by JSON-RPC handler
def test_error_missing_required_parameter(client):
    """Test error handling for missing required parameter."""
    response = client.post(
        "/api/v1/jsonrpc",
        json={
            "jsonrpc": "2.0",
            "method": "reasoning.bounded_context",
            "params": {},  # Missing required 'query'
            "id": 4,
        })
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32603  # Internal error for ValueError
def test_error_llm_failure(client):
    """Test error handling when LLM client fails."""
    with (
        patch("agentcore.reasoning.services.reasoning_jsonrpc.BoundedContextEngine") as mock_engine_class):
        
        mock_engine = MagicMock()
        mock_engine.reason = AsyncMock(
            side_effect=RuntimeError("LLM service unavailable")
        )
        mock_engine_class.return_value = mock_engine
        response = client.post(
            "/api/v1/jsonrpc",
            json={
                "jsonrpc": "2.0",
                "method": "reasoning.bounded_context",
                "params": {"query": "Test query"},
                "id": 5,
            })
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32603  # Internal error
        # Error message wrapped by JSON-RPC handler
def test_error_invalid_temperature(client):
    """Test error handling for out-of-range temperature."""
    response = client.post(
        "/api/v1/jsonrpc",
        json={
            "jsonrpc": "2.0",
            "method": "reasoning.bounded_context",
            "params": {
                "query": "Test",
                "temperature": 3.0,  # Invalid: > 2.0
            },
            "id": 6,
        })
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32603
def test_error_invalid_max_iterations(client):
    """Test error handling for invalid max_iterations."""
    response = client.post(
        "/api/v1/jsonrpc",
        json={
            "jsonrpc": "2.0",
            "method": "reasoning.bounded_context",
            "params": {
                "query": "Test",
                "max_iterations": 100,  # Invalid: > 50
            },
            "id": 7,
        })
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32603
def test_notification_request(client, valid_token_payload, mock_llm_http_client):
    """Test notification request (no response expected)."""
    with patch("agentcore.reasoning.services.reasoning_jsonrpc.security_service") as mock_security:
        mock_security.validate_token.return_value = valid_token_payload  # Mock auth
        response = client.post(
            "/api/v1/jsonrpc",
            json={
                "jsonrpc": "2.0",
                "method": "reasoning.bounded_context",
                "params": {"auth_token": "test-token", "query": "Notification test"},
                # No 'id' field = notification
            })
        # Notifications execute but return 204 No Content
        # Note: JSON-RPC spec says notifications have no response
        assert response.status_code in (200, 204)
def test_response_format_validation(client, valid_token_payload, mock_llm_http_client):
    """Test that response format matches expected structure."""
    with patch("agentcore.reasoning.services.reasoning_jsonrpc.security_service") as mock_security:
        mock_security.validate_token.return_value = valid_token_payload  # Mock auth
        response = client.post(
            "/api/v1/jsonrpc",
            json={
                "jsonrpc": "2.0",
                "method": "reasoning.bounded_context",
                "params": {"auth_token": "test-token", "query": "Format test"},
                "id": 8,
            })
        data = response.json()
        result = data["result"]
        # Verify all required fields are present
        required_fields = [
            "success",
            "answer",
            "total_iterations",
            "total_tokens",
            "compute_savings_pct",
            "carryover_compressions",
            "execution_time_ms",
            "iterations",
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        # Verify iterations structure
        assert isinstance(result["iterations"], list)
        assert len(result["iterations"]) > 0
        iteration = result["iterations"][0]
        iteration_fields = [
            "iteration",
            "tokens",
            "has_answer",
            "execution_time_ms",
            "carryover_generated",
            "content_preview",
        ]
        for field in iteration_fields:
            assert field in iteration, f"Missing iteration field: {field}"
def test_custom_system_prompt(client, valid_token_payload, mock_llm_http_client):
    """Test request with custom system prompt."""
    with patch("agentcore.reasoning.services.reasoning_jsonrpc.security_service") as mock_security:
        mock_security.validate_token.return_value = valid_token_payload  # Mock auth
        response = client.post(
            "/api/v1/jsonrpc",
            json={
                "jsonrpc": "2.0",
                "method": "reasoning.bounded_context",
                "params": {
                    "auth_token": "test-token",
                    "query": "Test query",
                    "system_prompt": "You are a helpful assistant.",
                },
                "id": 9,
            })
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        result = data["result"]
        assert result["success"] is True
        # E2E test: verify request with system_prompt completes successfully
        assert result["answer"] == "42"  # Extracted from mock LLM response
        assert result["total_iterations"] >= 1
        assert result["total_tokens"] > 0
def test_a2a_context_in_request(client, valid_token_payload, mock_llm_http_client):
    """Test request with A2A context."""
    with patch("agentcore.reasoning.services.reasoning_jsonrpc.security_service") as mock_security:
        mock_security.validate_token.return_value = valid_token_payload  # Mock auth
        response = client.post(
            "/api/v1/jsonrpc",
            json={
                "jsonrpc": "2.0",
                "method": "reasoning.bounded_context",
                "params": {"auth_token": "test-token", "query": "Test query"},
                "id": 10,
                "a2a_context": {
                    "source_agent": "agent-123",
                    "target_agent": "reasoning-agent",
                    "trace_id": "trace-abc",
                    "timestamp": "2025-01-01T00:00:00Z",
                },
            })
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["success"] is True
