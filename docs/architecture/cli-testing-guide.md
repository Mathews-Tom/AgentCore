# CLI Testing Guide

**Document Type:** Testing Guide
**Target Audience:** Developers and QA Engineers
**CLI Version:** 2.0
**Date:** 2025-10-22

---

## Overview

This guide provides comprehensive testing strategies, patterns, and best practices for the AgentCore CLI. The CLI's 4-layer architecture enables testing at multiple levels with clear mocking boundaries.

---

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Pyramid](#test-pyramid)
3. [Test Structure](#test-structure)
4. [Unit Testing by Layer](#unit-testing-by-layer)
5. [Integration Testing](#integration-testing)
6. [End-to-End Testing](#end-to-end-testing)
7. [Mocking Strategies](#mocking-strategies)
8. [Test Fixtures](#test-fixtures)
9. [Coverage Requirements](#coverage-requirements)
10. [Running Tests](#running-tests)
11. [Writing New Tests](#writing-new-tests)
12. [Common Patterns](#common-patterns)
13. [Troubleshooting](#troubleshooting)

---

## Testing Philosophy

### Core Principles

1. **Test at the Right Level**: Mock at the highest level that gives you control
2. **Fast Feedback**: Unit tests should run in milliseconds
3. **Clear Assertions**: One clear assertion per test
4. **Descriptive Names**: Test names describe what they verify
5. **Isolation**: Tests don't depend on each other
6. **Repeatability**: Tests produce same results every run

### Testing Goals

- **90%+ Code Coverage**: All production code should be tested
- **Fast Execution**: Unit tests complete in < 2 seconds
- **Clear Failures**: Failures clearly indicate what broke
- **Documentation**: Tests serve as usage documentation
- **Confidence**: Tests give confidence to refactor

---

## Test Pyramid

The CLI testing follows the standard test pyramid:

```
        /\
       /  \        E2E Tests (5%)
      /────\       - Full stack with live API
     /      \      - Slow (minutes)
    /────────\     - Few tests (smoke tests)
   /          \
  /────────────\   Integration Tests (15%)
 /              \  - Multiple layers together
/────────────────\ - Moderate speed (seconds)
                   - Service + Protocol tests

──────────────────
                   Unit Tests (80%)
                   - Single layer in isolation
                   - Fast (milliseconds)
                   - Many tests (comprehensive)
```

### Distribution

- **Unit Tests (80%):** Test each layer independently
  - CLI Layer: 30%
  - Service Layer: 25%
  - Protocol Layer: 15%
  - Transport Layer: 10%

- **Integration Tests (15%):** Test layer interactions
  - CLI → Service: 10%
  - Service → Protocol → Transport: 5%

- **E2E Tests (5%):** Test full stack
  - Critical user workflows only

---

## Test Structure

### Directory Layout

```
tests/
├── cli/                          # CLI layer tests
│   ├── test_agent_commands.py
│   ├── test_task_commands.py
│   ├── test_session_commands.py
│   ├── test_workflow_commands.py
│   └── test_config_commands.py
├── services/                     # Service layer tests (future)
│   ├── test_agent_service.py
│   ├── test_task_service.py
│   └── test_session_service.py
├── protocol/                     # Protocol layer tests (future)
│   ├── test_jsonrpc_client.py
│   └── test_models.py
├── transport/                    # Transport layer tests (future)
│   └── test_http_transport.py
├── integration/                  # Integration tests
│   └── test_cli_integration.py
└── e2e/                          # End-to-end tests
    └── test_cli_e2e.py
```

### File Naming Convention

- Test files: `test_{module}_name.py`
- Test classes: `Test{Feature}Command` or `Test{Feature}Service`
- Test functions: `test_{command}_{scenario}`

**Examples:**
```python
# tests/cli/test_agent_commands.py
class TestAgentRegisterCommand:
    def test_register_success(self): ...
    def test_register_validation_error(self): ...
    def test_register_json_output(self): ...
```

---

## Unit Testing by Layer

### Layer 1: CLI Layer Tests

**Goal:** Test argument parsing, output formatting, error handling

**Mock:** Service layer

**Location:** `tests/cli/test_{resource}_commands.py`

**Example:**

```python
"""Tests for agent commands."""
from __future__ import annotations
from unittest.mock import Mock, patch
import pytest
from typer.testing import CliRunner

from agentcore_cli.main import app
from agentcore_cli.services.exceptions import ValidationError, OperationError

@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()

@pytest.fixture
def mock_agent_service() -> Mock:
    """Create mock agent service."""
    return Mock()

class TestAgentRegisterCommand:
    """Test suite for agent register command."""

    def test_register_success(
        self,
        runner: CliRunner,
        mock_agent_service: Mock,
    ) -> None:
        """Test successful agent registration."""
        # Setup
        mock_agent_service.register.return_value = "agent-001"

        # Execute
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(
                app,
                ["agent", "register", "--name", "test", "--capabilities", "python"],
            )

        # Assert
        assert result.exit_code == 0
        assert "agent-001" in result.output
        mock_agent_service.register.assert_called_once_with(
            name="test",
            capabilities=["python"],
            cost_per_request=0.01,
        )

    def test_register_validation_error(
        self,
        runner: CliRunner,
        mock_agent_service: Mock,
    ) -> None:
        """Test registration with validation error."""
        # Setup
        mock_agent_service.register.side_effect = ValidationError(
            "Agent name cannot be empty"
        )

        # Execute
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(
                app,
                ["agent", "register", "--name", "", "--capabilities", "python"],
            )

        # Assert
        assert result.exit_code == 2  # Validation error exit code
        assert "Validation error" in result.output
        assert "cannot be empty" in result.output

    def test_register_json_output(
        self,
        runner: CliRunner,
        mock_agent_service: Mock,
    ) -> None:
        """Test registration with JSON output."""
        # Setup
        mock_agent_service.register.return_value = "agent-001"

        # Execute
        with patch(
            "agentcore_cli.commands.agent.get_agent_service",
            return_value=mock_agent_service,
        ):
            result = runner.invoke(
                app,
                [
                    "agent",
                    "register",
                    "--name",
                    "test",
                    "--capabilities",
                    "python",
                    "--json",
                ],
            )

        # Assert
        assert result.exit_code == 0
        assert '"agent_id": "agent-001"' in result.output  # JSON format
```

**Key Points:**
- Mock service layer via `get_agent_service()`
- Use `CliRunner` to invoke commands
- Test exit codes explicitly
- Verify service method calls with `.assert_called_once_with()`
- Test both table and JSON output formats

### Layer 2: Service Layer Tests

**Goal:** Test business logic, validation, API calls

**Mock:** JSON-RPC client

**Location:** `tests/services/test_{resource}_service.py`

**Example:**

```python
"""Tests for agent service."""
from __future__ import annotations
from unittest.mock import Mock
import pytest

from agentcore_cli.services.agent import AgentService
from agentcore_cli.services.exceptions import ValidationError, OperationError

@pytest.fixture
def mock_client() -> Mock:
    """Create mock JSON-RPC client."""
    return Mock()

@pytest.fixture
def agent_service(mock_client: Mock) -> AgentService:
    """Create agent service with mock client."""
    return AgentService(mock_client)

class TestAgentServiceRegister:
    """Test suite for agent service register method."""

    def test_register_success(
        self,
        agent_service: AgentService,
        mock_client: Mock,
    ) -> None:
        """Test successful agent registration."""
        # Setup
        mock_client.call.return_value = {"agent_id": "agent-001"}

        # Execute
        result = agent_service.register(
            name="test-agent",
            capabilities=["python", "analysis"],
            cost_per_request=0.01,
        )

        # Assert
        assert result == "agent-001"
        mock_client.call.assert_called_once_with(
            "agent.register",
            {
                "name": "test-agent",
                "capabilities": ["python", "analysis"],
                "cost_per_request": 0.01,
            },
        )

    def test_register_empty_name_raises_validation_error(
        self,
        agent_service: AgentService,
        mock_client: Mock,
    ) -> None:
        """Test registration with empty name raises ValidationError."""
        # Execute & Assert
        with pytest.raises(ValidationError, match="Agent name cannot be empty"):
            agent_service.register(
                name="",
                capabilities=["python"],
            )

        # Verify client was not called
        mock_client.call.assert_not_called()

    def test_register_no_capabilities_raises_validation_error(
        self,
        agent_service: AgentService,
        mock_client: Mock,
    ) -> None:
        """Test registration without capabilities raises ValidationError."""
        # Execute & Assert
        with pytest.raises(ValidationError, match="At least one capability required"):
            agent_service.register(
                name="test-agent",
                capabilities=[],
            )

    def test_register_api_error_raises_operation_error(
        self,
        agent_service: AgentService,
        mock_client: Mock,
    ) -> None:
        """Test registration API error raises OperationError."""
        # Setup
        mock_client.call.side_effect = Exception("Connection refused")

        # Execute & Assert
        with pytest.raises(OperationError, match="Agent registration failed"):
            agent_service.register(
                name="test-agent",
                capabilities=["python"],
            )
```

**Key Points:**
- Mock JSON-RPC client
- Test business validation (empty name, missing capabilities, etc.)
- Test API call parameters
- Test error handling and translation
- Verify client is NOT called when validation fails

### Layer 3: Protocol Layer Tests

**Goal:** Test JSON-RPC 2.0 compliance, request/response building

**Mock:** HTTP transport

**Location:** `tests/protocol/test_jsonrpc_client.py`

**Example:**

```python
"""Tests for JSON-RPC client."""
from __future__ import annotations
from unittest.mock import Mock
import pytest

from agentcore_cli.protocol.jsonrpc import JsonRpcClient
from agentcore_cli.protocol.models import JsonRpcRequest, JsonRpcResponse
from agentcore_cli.protocol.exceptions import MethodNotFoundError

@pytest.fixture
def mock_transport() -> Mock:
    """Create mock HTTP transport."""
    return Mock()

@pytest.fixture
def client(mock_transport: Mock) -> JsonRpcClient:
    """Create JSON-RPC client with mock transport."""
    return JsonRpcClient(transport=mock_transport, auth_token="test-token")

class TestJsonRpcClientCall:
    """Test suite for JSON-RPC client call method."""

    def test_call_builds_correct_request(
        self,
        client: JsonRpcClient,
        mock_transport: Mock,
    ) -> None:
        """Test call builds JSON-RPC 2.0 compliant request."""
        # Setup
        mock_transport.post.return_value = {
            "jsonrpc": "2.0",
            "result": {"success": True},
            "id": 1,
        }

        # Execute
        client.call("test.method", {"param1": "value1"})

        # Assert
        mock_transport.post.assert_called_once()
        call_args = mock_transport.post.call_args

        # Verify request structure
        request_data = call_args[1]["data"]
        assert request_data["jsonrpc"] == "2.0"
        assert request_data["method"] == "test.method"
        assert "params" in request_data  # CRITICAL: params wrapper
        assert request_data["params"] == {"param1": "value1"}
        assert "id" in request_data

    def test_call_includes_auth_header(
        self,
        client: JsonRpcClient,
        mock_transport: Mock,
    ) -> None:
        """Test call includes authentication header."""
        # Setup
        mock_transport.post.return_value = {
            "jsonrpc": "2.0",
            "result": {},
            "id": 1,
        }

        # Execute
        client.call("test.method", {})

        # Assert
        call_args = mock_transport.post.call_args
        headers = call_args[1]["headers"]
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-token"

    def test_call_handles_error_response(
        self,
        client: JsonRpcClient,
        mock_transport: Mock,
    ) -> None:
        """Test call handles JSON-RPC error response."""
        # Setup
        mock_transport.post.return_value = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32601,
                "message": "Method not found",
            },
            "id": 1,
        }

        # Execute & Assert
        with pytest.raises(MethodNotFoundError, match="Method not found"):
            client.call("unknown.method", {})
```

**Key Points:**
- Mock HTTP transport
- Verify JSON-RPC 2.0 structure (jsonrpc, method, params, id)
- Test params wrapper is present
- Test authentication header injection
- Test error response handling

### Layer 4: Transport Layer Tests

**Goal:** Test HTTP communication, retry logic, connection pooling

**Mock:** requests library

**Location:** `tests/transport/test_http_transport.py`

**Example:**

```python
"""Tests for HTTP transport."""
from __future__ import annotations
from unittest.mock import Mock, patch
import pytest
import requests

from agentcore_cli.transport.http import HttpTransport
from agentcore_cli.transport.exceptions import HttpError, NetworkError, TimeoutError

@pytest.fixture
def transport() -> HttpTransport:
    """Create HTTP transport."""
    return HttpTransport(
        base_url="http://localhost:8001",
        timeout=30,
        retries=3,
    )

class TestHttpTransportPost:
    """Test suite for HTTP transport post method."""

    def test_post_success(self, transport: HttpTransport) -> None:
        """Test successful POST request."""
        with patch("requests.Session.post") as mock_post:
            # Setup
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}
            mock_post.return_value = mock_response

            # Execute
            result = transport.post(
                endpoint="/api/v1/jsonrpc",
                data={"jsonrpc": "2.0", "method": "test"},
                headers={"Content-Type": "application/json"},
            )

            # Assert
            assert result == {"success": True}
            mock_post.assert_called_once()

    def test_post_retries_on_500_error(self, transport: HttpTransport) -> None:
        """Test POST retries on 500 error."""
        with patch("requests.Session.post") as mock_post:
            # Setup: Fail twice, succeed on third attempt
            mock_response_fail = Mock()
            mock_response_fail.status_code = 500

            mock_response_success = Mock()
            mock_response_success.status_code = 200
            mock_response_success.json.return_value = {"success": True}

            mock_post.side_effect = [
                mock_response_fail,
                mock_response_fail,
                mock_response_success,
            ]

            # Execute
            result = transport.post("/api/v1/jsonrpc", {})

            # Assert
            assert result == {"success": True}
            assert mock_post.call_count == 3  # 2 retries + 1 success

    def test_post_raises_http_error_after_max_retries(
        self,
        transport: HttpTransport,
    ) -> None:
        """Test POST raises HttpError after max retries."""
        with patch("requests.Session.post") as mock_post:
            # Setup: Always fail
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_post.return_value = mock_response

            # Execute & Assert
            with pytest.raises(HttpError, match="500"):
                transport.post("/api/v1/jsonrpc", {})

            assert mock_post.call_count == 3  # Max retries

    def test_post_raises_timeout_error(self, transport: HttpTransport) -> None:
        """Test POST raises TimeoutError on timeout."""
        with patch("requests.Session.post") as mock_post:
            # Setup
            mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

            # Execute & Assert
            with pytest.raises(TimeoutError, match="timed out"):
                transport.post("/api/v1/jsonrpc", {})
```

**Key Points:**
- Mock requests.Session.post
- Test retry logic (fail → retry → success)
- Test max retries exhaustion
- Test timeout handling
- Test network error translation

---

## Integration Testing

**Goal:** Test multiple layers working together

**Example:** CLI → Service → Protocol (mock transport)

**Location:** `tests/integration/test_cli_integration.py`

```python
"""Integration tests for CLI."""
from __future__ import annotations
from unittest.mock import Mock, patch
import pytest
from typer.testing import CliRunner

from agentcore_cli.main import app

@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()

@pytest.fixture
def mock_transport() -> Mock:
    """Mock transport layer only."""
    return Mock()

class TestAgentLifecycle:
    """Integration test for complete agent lifecycle."""

    def test_complete_agent_lifecycle(
        self,
        runner: CliRunner,
        mock_transport: Mock,
    ) -> None:
        """Test register → list → info → remove flow."""
        # Setup transport responses
        mock_transport.post.side_effect = [
            # agent.register response
            {
                "jsonrpc": "2.0",
                "result": {"agent_id": "agent-001"},
                "id": 1,
            },
            # agent.list response
            {
                "jsonrpc": "2.0",
                "result": {
                    "agents": [
                        {
                            "agent_id": "agent-001",
                            "name": "test-agent",
                            "status": "active",
                        }
                    ]
                },
                "id": 2,
            },
            # agent.get response
            {
                "jsonrpc": "2.0",
                "result": {
                    "agent_id": "agent-001",
                    "name": "test-agent",
                    "capabilities": ["python"],
                },
                "id": 3,
            },
            # agent.remove response
            {
                "jsonrpc": "2.0",
                "result": {"success": True},
                "id": 4,
            },
        ]

        with patch("agentcore_cli.container.get_transport", return_value=mock_transport):
            # 1. Register agent
            result = runner.invoke(
                app,
                ["agent", "register", "--name", "test-agent", "--capabilities", "python"],
            )
            assert result.exit_code == 0
            assert "agent-001" in result.output

            # 2. List agents
            result = runner.invoke(app, ["agent", "list"])
            assert result.exit_code == 0
            assert "test-agent" in result.output

            # 3. Get agent info
            result = runner.invoke(app, ["agent", "info", "agent-001"])
            assert result.exit_code == 0
            assert "test-agent" in result.output

            # 4. Remove agent
            result = runner.invoke(app, ["agent", "remove", "agent-001", "--force"])
            assert result.exit_code == 0
            assert "success" in result.output.lower()
```

**Key Points:**
- Mock only transport layer
- Test multiple commands in sequence
- Verify full data flow through layers
- Test realistic scenarios

---

## End-to-End Testing

**Goal:** Test full stack with live API server

**Setup:** Requires running API server

**Location:** `tests/e2e/test_cli_e2e.py`

```python
"""End-to-end tests for CLI."""
import pytest
from typer.testing import CliRunner

from agentcore_cli.main import app

pytestmark = pytest.mark.e2e  # Skip unless --run-e2e flag

@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()

class TestE2EAgentWorkflow:
    """E2E test for agent workflow."""

    def test_agent_registration_and_task_execution(self, runner: CliRunner) -> None:
        """Test complete agent registration and task execution."""
        # 1. Register agent
        result = runner.invoke(
            app,
            ["agent", "register", "--name", "e2e-agent", "--capabilities", "testing"],
        )
        assert result.exit_code == 0

        # Extract agent ID from output
        # (implementation depends on output format)
        agent_id = extract_agent_id(result.output)

        # 2. Create task for agent
        result = runner.invoke(
            app,
            ["task", "create", "--description", "E2E test task", "--agent-id", agent_id],
        )
        assert result.exit_code == 0

        # Extract task ID
        task_id = extract_task_id(result.output)

        # 3. Monitor task completion
        # (polling or wait logic)

        # 4. Cleanup
        runner.invoke(app, ["agent", "remove", agent_id, "--force"])
```

**Key Points:**
- No mocking - test against live API
- Test critical user workflows only
- Cleanup after tests
- Mark with `@pytest.mark.e2e`
- Skip by default (slow)

---

## Mocking Strategies

### Mocking Hierarchy

```
CLI Tests:      Mock service layer (get_agent_service)
Service Tests:  Mock JSON-RPC client
Protocol Tests: Mock HTTP transport
Transport Tests: Mock requests.Session
Integration:    Mock transport only
E2E Tests:      No mocking
```

### Mock Patterns

#### Pattern 1: Mock Factory Function

```python
with patch("agentcore_cli.commands.agent.get_agent_service", return_value=mock_service):
    result = runner.invoke(app, ["agent", "register", ...])
```

#### Pattern 2: Mock Object Method

```python
mock_service.register.return_value = "agent-001"
mock_service.register.assert_called_once_with(name="test", capabilities=["python"])
```

#### Pattern 3: Mock Side Effects

```python
# Sequential return values
mock_service.operation.side_effect = ["result1", "result2", "result3"]

# Exception
mock_service.operation.side_effect = ValidationError("Invalid input")
```

#### Pattern 4: Mock with Callback

```python
def custom_behavior(name, capabilities):
    if not name:
        raise ValidationError("Name required")
    return f"agent-{name}"

mock_service.register.side_effect = custom_behavior
```

---

## Test Fixtures

### Common Fixtures

```python
# conftest.py (shared fixtures)

import pytest
from typer.testing import CliRunner
from unittest.mock import Mock

@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()

@pytest.fixture
def mock_agent_service() -> Mock:
    """Create mock agent service."""
    return Mock()

@pytest.fixture
def mock_task_service() -> Mock:
    """Create mock task service."""
    return Mock()

@pytest.fixture
def mock_jsonrpc_client() -> Mock:
    """Create mock JSON-RPC client."""
    return Mock()

@pytest.fixture
def mock_transport() -> Mock:
    """Create mock HTTP transport."""
    return Mock()

@pytest.fixture(autouse=True)
def reset_container():
    """Reset DI container between tests."""
    from agentcore_cli.container import reset_container
    yield
    reset_container()
```

### Fixture Scopes

- `function` (default): New instance per test
- `class`: Shared within test class
- `module`: Shared within test module
- `session`: Shared across all tests

---

## Coverage Requirements

### Minimum Coverage

- **Overall:** 90%
- **Per Module:** 85%
- **Critical Paths:** 100%

### Coverage Report

```bash
# Generate coverage report
uv run pytest --cov=agentcore_cli --cov-report=html

# View report
open htmlcov/index.html

# Terminal summary
uv run pytest --cov=agentcore_cli --cov-report=term-missing
```

### Coverage Exclusions

```python
# Exclude from coverage
if TYPE_CHECKING:  # pragma: no cover
    from typing import ...

def debug_only():  # pragma: no cover
    """Debug helper function."""
    ...
```

---

## Running Tests

### Run All Tests

```bash
uv run pytest
```

### Run Specific Test File

```bash
uv run pytest tests/cli/test_agent_commands.py
```

### Run Specific Test Class

```bash
uv run pytest tests/cli/test_agent_commands.py::TestAgentRegisterCommand
```

### Run Specific Test Function

```bash
uv run pytest tests/cli/test_agent_commands.py::TestAgentRegisterCommand::test_register_success
```

### Run by Marker

```bash
# Run only E2E tests
uv run pytest -m e2e

# Run only integration tests
uv run pytest -m integration

# Skip E2E tests
uv run pytest -m "not e2e"
```

### Run with Verbosity

```bash
# Verbose output
uv run pytest -v

# Very verbose output
uv run pytest -vv

# Show print statements
uv run pytest -s
```

### Run with Coverage

```bash
# Basic coverage
uv run pytest --cov=agentcore_cli

# Coverage with missing lines
uv run pytest --cov=agentcore_cli --cov-report=term-missing

# Coverage HTML report
uv run pytest --cov=agentcore_cli --cov-report=html
```

### Run in Parallel

```bash
# Install pytest-xdist
uv add pytest-xdist --dev

# Run with 4 workers
uv run pytest -n 4
```

---

## Writing New Tests

### Test Template

```python
"""Tests for {module} commands."""
from __future__ import annotations
from unittest.mock import Mock, patch
import pytest
from typer.testing import CliRunner

from agentcore_cli.main import app
from agentcore_cli.services.exceptions import ValidationError, OperationError

@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()

@pytest.fixture
def mock_service() -> Mock:
    """Create mock service."""
    return Mock()

class Test{Feature}Command:
    """Test suite for {feature} command."""

    def test_{command}_success(
        self,
        runner: CliRunner,
        mock_service: Mock,
    ) -> None:
        """Test successful {command} execution."""
        # Setup
        mock_service.operation.return_value = "expected-result"

        # Execute
        with patch(
            "agentcore_cli.commands.{module}.get_{module}_service",
            return_value=mock_service,
        ):
            result = runner.invoke(
                app,
                ["{module}", "{command}", "--arg", "value"],
            )

        # Assert
        assert result.exit_code == 0
        assert "expected-result" in result.output
        mock_service.operation.assert_called_once_with(arg="value")

    def test_{command}_validation_error(
        self,
        runner: CliRunner,
        mock_service: Mock,
    ) -> None:
        """Test {command} with validation error."""
        # Setup
        mock_service.operation.side_effect = ValidationError("Invalid input")

        # Execute
        with patch(
            "agentcore_cli.commands.{module}.get_{module}_service",
            return_value=mock_service,
        ):
            result = runner.invoke(
                app,
                ["{module}", "{command}", "--arg", "invalid"],
            )

        # Assert
        assert result.exit_code == 2
        assert "Validation error" in result.output
        assert "Invalid input" in result.output
```

### Test Checklist

When adding a new command, write tests for:

- [ ] Success scenario
- [ ] Validation error
- [ ] Operation error
- [ ] JSON output format
- [ ] Table output format (if applicable)
- [ ] All CLI arguments
- [ ] Service method called with correct parameters
- [ ] Exit codes (0, 1, 2)

---

## Common Patterns

### Pattern 1: Test Multiple Scenarios

```python
@pytest.mark.parametrize(
    "capabilities,expected",
    [
        (["python"], ["python"]),
        (["python", "go"], ["python", "go"]),
        (["python", "go", "rust"], ["python", "go", "rust"]),
    ],
)
def test_register_with_multiple_capabilities(
    runner: CliRunner,
    mock_service: Mock,
    capabilities: list[str],
    expected: list[str],
) -> None:
    """Test registration with various capabilities."""
    mock_service.register.return_value = "agent-001"

    with patch("commands.agent.get_agent_service", return_value=mock_service):
        result = runner.invoke(
            app,
            ["agent", "register", "--name", "test", "--capabilities", ",".join(capabilities)],
        )

    assert result.exit_code == 0
    mock_service.register.assert_called_once()
    call_args = mock_service.register.call_args
    assert call_args[1]["capabilities"] == expected
```

### Pattern 2: Test Error Messages

```python
def test_register_empty_name_shows_clear_error(
    runner: CliRunner,
    mock_service: Mock,
) -> None:
    """Test empty name shows clear error message."""
    mock_service.register.side_effect = ValidationError("Agent name cannot be empty")

    with patch("commands.agent.get_agent_service", return_value=mock_service):
        result = runner.invoke(
            app,
            ["agent", "register", "--name", "", "--capabilities", "python"],
        )

    assert result.exit_code == 2
    assert "Validation error" in result.output
    assert "Agent name cannot be empty" in result.output
```

### Pattern 3: Test JSON Output

```python
import json

def test_list_json_output(
    runner: CliRunner,
    mock_service: Mock,
) -> None:
    """Test list command with JSON output."""
    mock_service.list_agents.return_value = [
        {"agent_id": "agent-001", "name": "test1"},
        {"agent_id": "agent-002", "name": "test2"},
    ]

    with patch("commands.agent.get_agent_service", return_value=mock_service):
        result = runner.invoke(app, ["agent", "list", "--json"])

    assert result.exit_code == 0

    # Parse JSON output
    output_data = json.loads(result.output)
    assert len(output_data) == 2
    assert output_data[0]["agent_id"] == "agent-001"
    assert output_data[1]["name"] == "test2"
```

---

## Troubleshooting

### Issue: Tests fail with "ModuleNotFoundError"

**Solution:**
```bash
# Ensure in correct directory
cd /path/to/agentcore

# Install in editable mode
uv pip install -e .

# Or run with uv
uv run pytest
```

### Issue: Mocks not working

**Check:**
1. Correct import path in `patch()`
2. Mock return value set before invocation
3. Context manager (`with patch(...)`) used correctly

**Example:**
```python
# ✓ CORRECT
with patch("agentcore_cli.commands.agent.get_agent_service", return_value=mock_service):
    result = runner.invoke(...)

# ✗ WRONG (patch path incorrect)
with patch("agentcore_cli.container.get_agent_service", return_value=mock_service):
    result = runner.invoke(...)
```

### Issue: Tests pass locally but fail in CI

**Common causes:**
1. Environment variables not set
2. Dependencies not installed
3. Race conditions in parallel tests
4. Filesystem differences

**Solution:**
```bash
# Run tests with same environment as CI
uv run pytest --cov=agentcore_cli

# Check for race conditions
uv run pytest --count=10  # Run tests 10 times
```

### Issue: Coverage not counting lines

**Check:**
1. Coverage is running: `pytest --cov=agentcore_cli`
2. Module is imported
3. Lines are actually executed

**Debug:**
```bash
# Generate detailed coverage report
uv run pytest --cov=agentcore_cli --cov-report=html
open htmlcov/index.html  # Check which lines are missing
```

---

## Conclusion

This guide provides comprehensive testing strategies for the AgentCore CLI. By following these patterns and best practices, you can ensure high-quality, maintainable tests that give confidence to refactor and extend the CLI.

**Key Takeaways:**
- Test at the right level (unit, integration, E2E)
- Mock at the highest level that gives you control
- Write descriptive test names
- Aim for 90%+ coverage
- Use fixtures for reusable setup
- Test both success and error paths

---

**Document Version:** 1.0
**Last Updated:** 2025-10-22
**Maintained By:** AgentCore Team
