# CLI Layer v2.0 - Migration Learnings & Best Practices

**Document Type:** Migration Guide
**Ticket:** CLI-R006
**Phase:** 2 - Proof of Concept
**Date:** 2025-10-22
**Audience:** Development Team

---

## Purpose

This document captures learnings from the CLI Layer v2.0 Proof of Concept (POC) implementation to guide Phase 3 full migration. It provides patterns, best practices, and actionable recommendations for migrating remaining commands.

---

## Table of Contents

1. [What Worked Well](#1-what-worked-well)
2. [Challenges & Solutions](#2-challenges--solutions)
3. [Best Practices](#3-best-practices)
4. [Migration Pattern Template](#4-migration-pattern-template)
5. [Testing Strategy](#5-testing-strategy)
6. [Common Pitfalls](#6-common-pitfalls)
7. [Phase 3 Recommendations](#7-phase-3-recommendations)

---

## 1. What Worked Well

### 1.1 Pydantic Models for Protocol Enforcement

**Why it worked:**
- Automatic validation of JSON-RPC requests/responses
- Type safety at runtime
- Self-documenting code
- IDE autocomplete support

**Implementation:**
```python
# protocol/models.py
class JsonRpcRequest(BaseModel):
    jsonrpc: Literal["2.0"] = Field(default="2.0")
    method: str = Field(...)
    params: dict[str, Any] = Field(default_factory=dict)  # CRITICAL: params wrapper
    id: int | str | None = Field(default=None)
```

**Key Insight:** Using `Literal["2.0"]` instead of deprecated `const=True` provides better type checking

**Recommendation:** Use Pydantic models for all structured data in Phase 3

### 1.2 Layer Separation via Dependency Injection

**Why it worked:**
- Each layer testable in isolation
- Easy to mock dependencies
- Clear dependency graph
- Lazy initialization

**Implementation:**
```python
# container.py
@lru_cache(maxsize=1)
def get_transport() -> HttpTransport:
    config = get_config()
    return HttpTransport(...)

def get_agent_service() -> AgentService:
    return AgentService(get_jsonrpc_client())
```

**Key Insight:** Cache infrastructure (transport, client), create services on-demand

**Recommendation:** Follow this caching pattern for all future services

### 1.3 Service Layer Abstraction

**Why it worked:**
- Commands don't know about JSON-RPC
- Business logic centralized
- Easy to change protocol without touching commands
- Reusable for non-CLI interfaces

**Implementation:**
```python
# services/agent.py
class AgentService:
    def register(self, name: str, capabilities: list[str], ...) -> str:
        # Validate business rules
        if not capabilities:
            raise ValidationError("At least one capability required")

        # Call protocol layer
        result = self.client.call("agent.register", {
            "name": name,
            "capabilities": capabilities,
            ...
        })

        # Return domain object
        return result["agent_id"]
```

**Key Insight:** Service layer knows domain logic, not protocol details

**Recommendation:** Keep services protocol-agnostic for maximum reusability

### 1.4 Error Hierarchy with Exit Code Mapping

**Why it worked:**
- Clear error types for different scenarios
- Proper Unix exit codes
- User-friendly error messages
- Easy to test error paths

**Implementation:**
```python
# commands/agent.py
try:
    result = service.operation(...)
except ValidationError as e:
    console.print(f"[red]Validation error:[/red] {e.message}")
    raise typer.Exit(2)  # Usage error
except OperationError as e:
    console.print(f"[red]Operation failed:[/red] {e.message}")
    raise typer.Exit(1)  # General error
```

**Key Insight:** Map exceptions to exit codes at CLI layer only

**Recommendation:** Use this error handling pattern for all commands

### 1.5 Mock-Friendly Testing

**Why it worked:**
- No need for live API server in CLI tests
- Fast test execution (1.94s for 22 tests)
- Easy to test error scenarios
- Clear test assertions

**Implementation:**
```python
# tests/cli/test_agent_commands.py
def test_register_success(runner, mock_agent_service):
    mock_agent_service.register.return_value = "agent-001"

    with patch("commands.agent.get_agent_service", return_value=mock_agent_service):
        result = runner.invoke(app, ["agent", "register", ...])

    assert result.exit_code == 0
    mock_agent_service.register.assert_called_once_with(...)
```

**Key Insight:** Mock at service layer, not transport layer

**Recommendation:** Mock service layer for all CLI tests, mock transport for service tests

---

## 2. Challenges & Solutions

### 2.1 Challenge: Service Layer Caching

**Problem:** Should service instances be cached like transport/client?

**Analysis:**
- **PRO Caching:** Performance optimization, fewer object creations
- **CON Caching:** Risk of state pollution, harder to test

**Solution:** Do NOT cache service instances

**Rationale:**
- Services are lightweight (just hold client reference)
- Transport/client already cached (where overhead exists)
- Avoid state issues
- Easier to test (fresh instance per operation)

**Implementation:**
```python
# ✓ CORRECT: No caching for services
def get_agent_service() -> AgentService:
    return AgentService(get_jsonrpc_client())

# ✗ WRONG: Don't do this
@lru_cache(maxsize=1)  # NO!
def get_agent_service() -> AgentService:
    return AgentService(get_jsonrpc_client())
```

### 2.2 Challenge: Error Translation Complexity

**Problem:** How to map JSON-RPC error codes to domain exceptions?

**Analysis:**
- JSON-RPC has standard error codes (-32700, -32600, etc.)
- Domain needs specific exceptions (AgentNotFoundError, etc.)
- Error messages should be user-friendly

**Solution:** Two-layer error translation

**Implementation:**
```python
# Layer 1: Protocol → Protocol Exceptions (jsonrpc.py)
def _raise_error(self, error: JsonRpcError) -> None:
    if error.code == -32601:
        raise MethodNotFoundError(...)
    elif error.code == -32602:
        raise InvalidParamsError(...)
    # ... etc

# Layer 2: Service → Domain Exceptions (services/agent.py)
try:
    result = self.client.call("agent.get", {"agent_id": agent_id})
except Exception as e:
    if "not found" in str(e).lower():
        raise AgentNotFoundError(f"Agent '{agent_id}' not found")
    raise OperationError(f"Agent retrieval failed: {str(e)}")
```

**Key Insight:** Protocol layer handles protocol errors, service layer handles domain errors

### 2.3 Challenge: Testing Without API Server

**Problem:** CLI tests need to verify behavior without running full API

**Analysis:**
- Integration tests are slow
- Hard to test error scenarios
- Flaky tests due to network issues

**Solution:** Mock at service layer boundary

**Implementation:**
```python
# ✓ CORRECT: Mock service layer
with patch("commands.agent.get_agent_service", return_value=mock_service):
    result = runner.invoke(app, ["agent", "register", ...])

# ✗ WRONG: Don't mock transport (too low-level for CLI tests)
with patch("transport.http.HttpTransport.post", ...):  # NO!
```

**Key Insight:** Mock at highest level that gives you control

### 2.4 Challenge: Configuration Management

**Problem:** How to handle config precedence (CLI args, env vars, files)?

**Analysis:**
- CLI args should override everything
- Env vars should override config files
- Config files should have defaults

**Solution:** Use pydantic_settings with env variable support

**Implementation:**
```python
# container.py
class Config(BaseSettings):
    api: ApiConfig = Field(default_factory=ApiConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)

    model_config = SettingsConfigDict(
        env_prefix="AGENTCORE_",
        env_nested_delimiter="_",
        case_sensitive=False,
    )

# Usage: Environment variables override defaults
# AGENTCORE_API_URL=http://prod.example.com agentcore agent list
```

**Key Insight:** pydantic_settings handles precedence automatically

### 2.5 Challenge: Output Formatting Consistency

**Problem:** Need consistent output across commands (table, JSON, errors)

**Analysis:**
- Different commands have different output structures
- JSON output must be machine-readable
- Table output should be user-friendly
- Error messages need consistent styling

**Solution:** Use rich library with consistent patterns

**Implementation:**
```python
# commands/agent.py
if json_output:
    console.print(json.dumps(result, indent=2))
else:
    console.print(f"[green]✓[/green] Operation successful")
    console.print(f"[bold]Field:[/bold] {value}")

# Error output (consistent across all commands)
console.print(f"[red]Error type:[/red] {error.message}")
```

**Key Insight:** Define output patterns once, reuse everywhere

---

## 3. Best Practices

### 3.1 Code Organization

**Directory Structure:**
```
src/agentcore_cli/
├── commands/          # CLI layer (Typer)
│   ├── agent.py
│   ├── task.py
│   └── ...
├── services/          # Service layer (business logic)
│   ├── agent.py
│   ├── task.py
│   └── ...
├── protocol/          # Protocol layer (JSON-RPC)
│   ├── jsonrpc.py
│   ├── models.py
│   └── exceptions.py
├── transport/         # Transport layer (HTTP)
│   ├── http.py
│   └── exceptions.py
├── container.py       # Dependency injection
└── main.py           # CLI entry point
```

**Principle:** One layer per directory, clear boundaries

### 3.2 Naming Conventions

**Files:**
- Singular nouns: `agent.py`, not `agents.py`
- Match domain concepts: `task.py`, `session.py`
- Exceptions grouped: `exceptions.py` per layer

**Classes:**
- Services: `{Resource}Service` (e.g., `AgentService`)
- Models: `{Purpose}{Type}` (e.g., `JsonRpcRequest`)
- Exceptions: `{Condition}Error` (e.g., `ValidationError`)

**Functions:**
- Commands: Verb phrases (e.g., `register`, `list`, `remove`)
- Services: Domain operations (e.g., `register`, `get`, `search`)
- Container: `get_{resource}_service()` (e.g., `get_agent_service()`)

### 3.3 Type Hints

**Always use type hints:**
```python
# ✓ CORRECT
def register(
    self,
    name: str,
    capabilities: list[str],
    cost_per_request: float = 0.01,
) -> str:
    ...

# ✗ WRONG
def register(self, name, capabilities, cost_per_request=0.01):
    ...
```

**Use modern syntax (Python 3.12+):**
```python
# ✓ CORRECT
list[str]           # Not List[str]
dict[str, Any]      # Not Dict[str, Any]
int | None          # Not Optional[int]

# ✗ WRONG (deprecated)
from typing import List, Dict, Optional
```

### 3.4 Documentation

**Docstrings:**
- All public functions must have docstrings
- Include Args, Returns, Raises sections
- Add usage examples

**Example:**
```python
def register(
    self,
    name: str,
    capabilities: list[str],
    cost_per_request: float = 0.01,
) -> str:
    """Register a new agent.

    Args:
        name: Agent name (must be unique)
        capabilities: List of agent capabilities (at least one required)
        cost_per_request: Cost per request in dollars (default: 0.01)

    Returns:
        Agent ID (string)

    Raises:
        ValidationError: If validation fails
        OperationError: If registration fails

    Example:
        >>> service = AgentService(client)
        >>> agent_id = service.register("analyzer", ["python", "analysis"])
        >>> print(agent_id)
        'agent-001'
    """
```

### 3.5 Error Handling

**Always validate at service layer:**
```python
# ✓ CORRECT
def register(self, name: str, ...) -> str:
    # Validate
    if not name or not name.strip():
        raise ValidationError("Agent name cannot be empty")

    # Call API
    result = self.client.call(...)

    # Validate result
    if "agent_id" not in result:
        raise OperationError("API did not return agent_id")

    return result["agent_id"]
```

**Handle exceptions at CLI layer:**
```python
# ✓ CORRECT
try:
    result = service.register(...)
except ValidationError as e:
    console.print(f"[red]Validation error:[/red] {e.message}")
    raise typer.Exit(2)
except OperationError as e:
    console.print(f"[red]Operation failed:[/red] {e.message}")
    raise typer.Exit(1)
```

---

## 4. Migration Pattern Template

### 4.1 Step-by-Step Migration Process

#### Step 1: Create Service (if doesn't exist)

**File:** `src/agentcore_cli/services/{resource}.py`

```python
"""Service for {resource} operations."""

from __future__ import annotations
from typing import Any

from agentcore_cli.protocol.jsonrpc import JsonRpcClient
from agentcore_cli.services.exceptions import ValidationError, OperationError

class {Resource}Service:
    """Service for {resource} operations.

    Provides business operations for {resource} management:
    - Operation 1
    - Operation 2
    - ...
    """

    def __init__(self, client: JsonRpcClient) -> None:
        """Initialize {resource} service."""
        self.client = client

    def operation(self, param1: str, param2: int = 0) -> ResultType:
        """Perform operation.

        Args:
            param1: Description
            param2: Description (default: 0)

        Returns:
            Result description

        Raises:
            ValidationError: If validation fails
            OperationError: If operation fails
        """
        # 1. Business validation
        if not param1:
            raise ValidationError("param1 cannot be empty")

        # 2. Prepare parameters
        params = {"param1": param1, "param2": param2}

        # 3. Call JSON-RPC method
        try:
            result = self.client.call("{resource}.operation", params)
        except Exception as e:
            raise OperationError(f"Operation failed: {str(e)}")

        # 4. Validate result
        if "expected_field" not in result:
            raise OperationError("API did not return expected field")

        # 5. Return domain result
        return result["expected_field"]
```

#### Step 2: Update Container

**File:** `src/agentcore_cli/container.py`

```python
def get_{resource}_service() -> {Resource}Service:
    """Create {resource} service instance.

    Returns:
        {Resource} service instance
    """
    if "{resource}_service" in _overrides:
        return _overrides["{resource}_service"]

    client = get_jsonrpc_client()
    return {Resource}Service(client)
```

#### Step 3: Migrate Command

**File:** `src/agentcore_cli/commands/{resource}.py`

```python
"""{Resource} management commands."""

from __future__ import annotations
from typing import Annotated

import typer
from rich.console import Console
import json

from agentcore_cli.container import get_{resource}_service
from agentcore_cli.services.exceptions import (
    ValidationError,
    OperationError,
)

app = typer.Typer(
    name="{resource}",
    help="Manage {resource} lifecycle",
    no_args_is_help=True,
)

console = Console()

@app.command()
def operation(
    param1: Annotated[str, typer.Option("--param1", "-p", help="Parameter description")],
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output in JSON format")] = False,
) -> None:
    """Perform operation.

    Examples:
        # Basic usage
        agentcore {resource} operation --param1 value

        # JSON output
        agentcore {resource} operation -p value --json
    """
    try:
        # Get service
        service = get_{resource}_service()

        # Call service
        result = service.operation(param1=param1)

        # Format output
        if json_output:
            console.print(json.dumps({"result": result}, indent=2))
        else:
            console.print(f"[green]✓[/green] Operation successful")
            console.print(f"[bold]Result:[/bold] {result}")

    except ValidationError as e:
        console.print(f"[red]Validation error:[/red] {e.message}")
        raise typer.Exit(2)
    except OperationError as e:
        console.print(f"[red]Operation failed:[/red] {e.message}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)
```

#### Step 4: Write Tests

**File:** `tests/cli/test_{resource}_commands.py`

```python
"""Tests for {resource} commands."""

from __future__ import annotations
from unittest.mock import Mock, patch
import pytest
from typer.testing import CliRunner

from agentcore_cli.main import app
from agentcore_cli.services.exceptions import ValidationError, OperationError

@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()

@pytest.fixture
def mock_service() -> Mock:
    return Mock()

class TestOperationCommand:
    """Test suite for operation command."""

    def test_operation_success(self, runner: CliRunner, mock_service: Mock) -> None:
        """Test successful operation."""
        mock_service.operation.return_value = "expected-result"

        with patch(
            "agentcore_cli.commands.{resource}.get_{resource}_service",
            return_value=mock_service,
        ):
            result = runner.invoke(app, ["{resource}", "operation", "--param1", "value"])

        assert result.exit_code == 0
        assert "Operation successful" in result.output
        mock_service.operation.assert_called_once_with(param1="value")

    def test_operation_validation_error(self, runner: CliRunner, mock_service: Mock) -> None:
        """Test operation with validation error."""
        mock_service.operation.side_effect = ValidationError("param1 cannot be empty")

        with patch(
            "agentcore_cli.commands.{resource}.get_{resource}_service",
            return_value=mock_service,
        ):
            result = runner.invoke(app, ["{resource}", "operation", "--param1", ""])

        assert result.exit_code == 2
        assert "Validation error" in result.output
```

#### Step 5: Update Main App

**File:** `src/agentcore_cli/main.py`

```python
from agentcore_cli.commands import {resource}

app.add_typer({resource}.app, name="{resource}")
```

---

## 5. Testing Strategy

### 5.1 Test Pyramid

**Layer 1: Unit Tests (per layer)**
- **Protocol Layer:** Test JSON-RPC request/response building
- **Service Layer:** Test business logic and validation
- **CLI Layer:** Test argument parsing and output formatting

**Layer 2: Integration Tests**
- Test service layer with real JSON-RPC client (mocked transport)
- Test CLI with real service layer (mocked client)

**Layer 3: E2E Tests**
- Test full stack with live API server
- Run in CI/CD only (slow, flaky)

### 5.2 Test File Organization

```
tests/
├── cli/                    # CLI layer tests
│   ├── test_agent_commands.py
│   ├── test_task_commands.py
│   └── ...
├── services/               # Service layer tests (future)
│   ├── test_agent_service.py
│   ├── test_task_service.py
│   └── ...
├── protocol/               # Protocol layer tests (future)
│   ├── test_jsonrpc_client.py
│   ├── test_models.py
│   └── ...
└── integration/            # Integration tests
    └── test_cli_e2e.py
```

### 5.3 Mocking Strategy

**CLI Tests:**
- Mock: Service layer (`get_{resource}_service()`)
- Reason: Fast, isolated, easy to test error paths

**Service Tests (future):**
- Mock: JSON-RPC client (`JsonRpcClient`)
- Reason: No network, control responses

**Protocol Tests (future):**
- Mock: Transport layer (`HttpTransport`)
- Reason: No HTTP, control network behavior

**Integration Tests:**
- Mock: Nothing (use test API server)
- Reason: Verify full stack

### 5.4 Test Naming Convention

```python
# Pattern: test_{command}_{scenario}
def test_register_success()
def test_register_validation_error()
def test_register_operation_error()
def test_register_json_output()
def test_register_with_custom_params()
```

---

## 6. Common Pitfalls

### 6.1 Pitfall: Mixing Concerns

**DON'T DO THIS:**
```python
# ✗ WRONG: CLI layer knows about JSON-RPC
@app.command()
def register(...):
    request = JsonRpcRequest(method="agent.register", params={...})  # NO!
    response = http_client.post("/api/v1/jsonrpc", request)
```

**DO THIS INSTEAD:**
```python
# ✓ CORRECT: CLI delegates to service
@app.command()
def register(...):
    service = get_agent_service()
    result = service.register(...)
```

### 6.2 Pitfall: Not Validating at Service Layer

**DON'T DO THIS:**
```python
# ✗ WRONG: No validation before API call
def register(self, name: str, ...) -> str:
    result = self.client.call("agent.register", {"name": name, ...})
    return result["agent_id"]
```

**DO THIS INSTEAD:**
```python
# ✓ CORRECT: Validate before calling API
def register(self, name: str, ...) -> str:
    if not name or not name.strip():
        raise ValidationError("Agent name cannot be empty")

    result = self.client.call("agent.register", {"name": name.strip(), ...})

    if "agent_id" not in result:
        raise OperationError("API did not return agent_id")

    return result["agent_id"]
```

### 6.3 Pitfall: Caching Service Instances

**DON'T DO THIS:**
```python
# ✗ WRONG: Caching services can cause state issues
@lru_cache(maxsize=1)
def get_agent_service() -> AgentService:
    return AgentService(get_jsonrpc_client())
```

**DO THIS INSTEAD:**
```python
# ✓ CORRECT: Create service on-demand
def get_agent_service() -> AgentService:
    return AgentService(get_jsonrpc_client())
```

### 6.4 Pitfall: Not Using Type Hints

**DON'T DO THIS:**
```python
# ✗ WRONG: No type hints
def register(self, name, capabilities):
    ...
```

**DO THIS INSTEAD:**
```python
# ✓ CORRECT: Full type hints
def register(self, name: str, capabilities: list[str]) -> str:
    ...
```

### 6.5 Pitfall: Generic Error Messages

**DON'T DO THIS:**
```python
# ✗ WRONG: Generic error
raise OperationError("Operation failed")
```

**DO THIS INSTEAD:**
```python
# ✓ CORRECT: Specific error message
raise OperationError(f"Agent registration failed: {str(e)}")
```

### 6.6 Pitfall: Testing Too Low in Stack

**DON'T DO THIS:**
```python
# ✗ WRONG: CLI tests mocking transport layer
with patch("agentcore_cli.transport.http.HttpTransport.post"):
    result = runner.invoke(app, ["agent", "register", ...])
```

**DO THIS INSTEAD:**
```python
# ✓ CORRECT: CLI tests mocking service layer
with patch("agentcore_cli.commands.agent.get_agent_service", return_value=mock_service):
    result = runner.invoke(app, ["agent", "register", ...])
```

---

## 7. Phase 3 Recommendations

### 7.1 Migration Order

**Week 1:**
1. **Task commands** (3 SP, 0.75 days) - Similar to agent commands
2. **Session commands** (2 SP, 0.5 days) - Similar to agent commands

**Week 2:**
3. **Workflow commands** (2 SP, 0.5 days) - More complex
4. **Config commands** (2 SP, 0.5 days) - Different pattern
5. **Health/Version commands** (1 SP, 0.25 days) - Simple

**Buffer:** 0.5 days for testing and fixes

### 7.2 Team Coordination

**Before Starting Each Command:**
1. Review this guide
2. Check existing patterns (agent.py as reference)
3. Create service if needed
4. Write tests alongside code

**During Development:**
1. Follow template exactly
2. Run tests frequently
3. Keep commits small and focused
4. Document any deviations

**After Completing Each Command:**
1. Run full test suite
2. Update this document if patterns change
3. Get code review
4. Commit with clear message

### 7.3 Testing Checklist

For each migrated command, ensure:

- [ ] Service layer tests (if new service)
  - [ ] Business validation tests
  - [ ] Success path tests
  - [ ] Error path tests
- [ ] CLI tests
  - [ ] Success scenario
  - [ ] Validation error handling
  - [ ] Operation error handling
  - [ ] JSON output format
  - [ ] Table output format
- [ ] Integration test (one per resource)
  - [ ] Full lifecycle test

### 7.4 Code Review Checklist

For each PR, verify:

- [ ] Layer separation maintained
  - [ ] CLI layer: No JSON-RPC knowledge
  - [ ] Service layer: No CLI knowledge
  - [ ] Protocol layer: No business logic
  - [ ] Transport layer: No protocol knowledge
- [ ] Type hints on all functions
- [ ] Docstrings on all public functions
- [ ] Error handling follows pattern
- [ ] Tests pass and cover new code
- [ ] No Mypy errors (strict mode)
- [ ] No Ruff linting issues

### 7.5 Documentation Updates

**During Phase 3:**
- Update this document with new learnings
- Document any deviations from patterns
- Add troubleshooting tips

**After Phase 3:**
- Update README_CLI.md
- Create architecture diagrams
- Write migration retrospective
- Update API documentation

### 7.6 Metrics to Track

**Code Metrics:**
- Lines of code per layer
- Test coverage per layer
- Number of tests per command
- Mypy/Ruff violations

**Quality Metrics:**
- Test pass rate
- Bugs found during migration
- Regressions introduced
- PR review time

**Process Metrics:**
- Time per command migration
- Time per test written
- Code review iterations
- Documentation updates

---

## 8. Quick Reference

### 8.1 File Creation Checklist

When migrating a command, create/update these files:

- [ ] `src/agentcore_cli/services/{resource}.py` (if new)
- [ ] `src/agentcore_cli/commands/{resource}.py`
- [ ] `src/agentcore_cli/container.py` (add get_{resource}_service)
- [ ] `src/agentcore_cli/main.py` (register command)
- [ ] `tests/cli/test_{resource}_commands.py`

### 8.2 Import Patterns

**Service Layer:**
```python
from agentcore_cli.protocol.jsonrpc import JsonRpcClient
from agentcore_cli.services.exceptions import ValidationError, OperationError
```

**CLI Layer:**
```python
import typer
from rich.console import Console
import json
from agentcore_cli.container import get_{resource}_service
from agentcore_cli.services.exceptions import ValidationError, OperationError
```

**Tests:**
```python
from unittest.mock import Mock, patch
import pytest
from typer.testing import CliRunner
from agentcore_cli.main import app
from agentcore_cli.services.exceptions import ValidationError, OperationError
```

### 8.3 Command Template (Minimal)

```python
@app.command()
def operation(
    param: Annotated[str, typer.Option("--param", "-p", help="Param description")],
    json_output: Annotated[bool, typer.Option("--json", "-j")] = False,
) -> None:
    """Command description."""
    try:
        service = get_resource_service()
        result = service.operation(param=param)

        if json_output:
            console.print(json.dumps(result, indent=2))
        else:
            console.print(f"[green]✓[/green] Success")

    except ValidationError as e:
        console.print(f"[red]Validation error:[/red] {e.message}")
        raise typer.Exit(2)
    except OperationError as e:
        console.print(f"[red]Operation failed:[/red] {e.message}")
        raise typer.Exit(1)
```

---

## 9. Conclusion

The POC has validated a clear, repeatable pattern for CLI command migration. By following this guide, Phase 3 should proceed smoothly with minimal issues.

**Key Takeaways:**
1. Follow the 4-layer architecture strictly
2. Mock at the right level (service layer for CLI tests)
3. Validate at service layer, handle errors at CLI layer
4. Use Pydantic models for type safety
5. Write tests alongside code
6. Keep commits small and focused

**Success Criteria:**
- All commands migrated following this pattern
- All tests passing
- No protocol compliance issues
- Clear, maintainable code

**Next Steps:**
- Begin Phase 3 migration with task commands
- Update this document with new learnings
- Track metrics to validate timeline estimates

---

**Document Author:** Claude Code
**Review Status:** Ready for Team Review
**Last Updated:** 2025-10-22
