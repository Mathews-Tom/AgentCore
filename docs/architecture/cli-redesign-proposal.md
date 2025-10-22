# CLI Redesign Proposal

**Date:** 2025-10-22
**Status:** Proposal
**Priority:** High
**Reason:** Current CLI architecture violates A2A protocol compliance

---

## Executive Summary

The AgentCore CLI (`agentcore_cli`) currently sends invalid JSON-RPC 2.0 requests that violate the A2A protocol specification. The root cause is architectural: parameters are passed as direct dictionary properties instead of being wrapped in the `params` object, causing the backend to receive malformed requests.

This document proposes a major CLI redesign to establish proper separation of concerns, ensure A2A protocol compliance, and create a maintainable architecture for future development.

---

## Problem Statement

### Current Architecture Issues

1. **Protocol Violation**
   - CLI sends: `{"jsonrpc": "2.0", "method": "agent.register", "id": 1, "name": "...", "capabilities": [...], ...}`
   - A2A requires: `{"jsonrpc": "2.0", "method": "agent.register", "params": {"name": "...", "capabilities": [...], ...}, "id": 1}`
   - Backend expects proper JSON-RPC 2.0 format with `params` wrapper

2. **Mixing of Concerns**
   - `client.py:86-115`: Builds JSON-RPC request structure
   - `commands/agent.py:105-110`: Builds parameter dictionary
   - No clear boundary between transport layer and application layer

3. **Tight Coupling**
   - Command handlers directly create client instances
   - Configuration loading repeated in every command
   - Error handling duplicated across all commands
   - No abstraction between CLI commands and JSON-RPC protocol

4. **Test Brittleness**
   - Tests mock at wrong level (HTTP responses instead of domain layer)
   - Integration tests depend on exact API implementation details
   - Cannot test command logic independently from transport

### Impact

- **Functional:** All CLI commands currently fail with protocol errors
- **Maintainability:** Adding new commands requires duplicating boilerplate
- **Testing:** Cannot verify command logic without full API integration
- **Extensibility:** Cannot support batch operations, streaming, or alternative transports

---

## Proposed Architecture

### Layer Model

```plaintext
┌────────────────────────────────────────────────┐
│                    CLI Layer (Typer)           │
│  • Argument parsing                            │
│  • User interaction (prompts, confirmations)   │
│  • Output formatting (tables, JSON, errors)    │
│  • Exit code handling                          │
└────────────────────────┬───────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────┐
│                  Service Layer (Facade)        │
│  • AgentService                                │
│  • TaskService                                 │
│  • SessionService                              │
│  • WorkflowService                             │
│  • High-level business operations              │
│  • Parameter validation & transformation       │
│  • Domain error handling                       │
└────────────────────────┬───────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────┐
│              Protocol Layer (JSON-RPC 2.0)     │
│  • JsonRpcClient                               │
│  • Request/Response models (Pydantic)          │
│  • Batch request handling                      │
│  • A2A context management                      │
│  • Protocol-level error translation            │
└────────────────────────┬───────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────┐
│               Transport Layer (HTTP)           │
│  • HttpTransport                               │
│  • Connection pooling                          │
│  • Retry logic with exponential backoff        │
│  • SSL/TLS verification                        │
│  • Timeout handling                            │
│  • Network error translation                   │
└────────────────────────────────────────────────┘
```

### Key Principles

1. **Separation of Concerns**
   - Each layer has a single, well-defined responsibility
   - Layers communicate through clear interfaces
   - No layer skips or bypasses another

2. **Protocol Compliance**
   - Protocol layer enforces JSON-RPC 2.0 specification
   - All requests validated against Pydantic models
   - A2A context automatically managed

3. **Testability**
   - Each layer testable in isolation
   - Service layer can be tested without HTTP
   - Command handlers can be tested without API
   - Protocol validation can be tested independently

4. **Extensibility**
   - New commands only touch CLI and Service layers
   - Alternative transports (WebSocket, gRPC) can replace HTTP layer
   - Batch operations supported at Protocol layer
   - Streaming operations possible at Transport layer

---

## Detailed Design

### 1. Transport Layer

**File:** `src/agentcore_cli/transport/http.py`

```python
from __future__ import annotations

from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class HttpTransport:
    """HTTP transport for JSON-RPC requests.

    Handles:
    - Connection pooling
    - Retry logic with exponential backoff
    - Timeout configuration
    - SSL/TLS verification
    - Network error translation
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        retries: int = 3,
        verify_ssl: bool = True,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.session = self._create_session(retries)

    def post(
        self,
        endpoint: str,
        data: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send HTTP POST request.

        Args:
            endpoint: API endpoint path
            data: JSON payload
            headers: Optional HTTP headers

        Returns:
            Parsed JSON response

        Raises:
            TransportError: Network-level errors
        """
        # Implementation: HTTP POST with retry logic
        ...
```

**Responsibilities:**

- Network communication only
- No knowledge of JSON-RPC protocol
- No knowledge of business logic
- Returns raw HTTP responses

---

### 2. Protocol Layer

**File:** `src/agentcore_cli/protocol/jsonrpc.py`

```python
from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")

class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request model."""
    jsonrpc: str = Field(default="2.0", const=True)
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    id: int | str

class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 response model."""
    jsonrpc: str = Field(default="2.0", const=True)
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None
    id: int | str | None

class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error model."""
    code: int
    message: str
    data: dict[str, Any] | None = None

class JsonRpcClient:
    """JSON-RPC 2.0 client with A2A protocol support.

    Handles:
    - JSON-RPC 2.0 request/response formatting
    - Request validation against Pydantic models
    - Batch request handling
    - A2A context management
    - Protocol-level error translation
    """

    def __init__(
        self,
        transport: HttpTransport,
        auth_token: str | None = None,
    ) -> None:
        self.transport = transport
        self.auth_token = auth_token
        self.request_id = 0

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        a2a_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute JSON-RPC method call.

        Args:
            method: JSON-RPC method name
            params: Method parameters (NOT the entire request)
            a2a_context: Optional A2A context

        Returns:
            Result from JSON-RPC response

        Raises:
            JsonRpcProtocolError: Protocol-level errors
        """
        # Build request using Pydantic model
        request = JsonRpcRequest(
            method=method,
            params=params or {},
            id=self._next_id(),
        )

        # Add A2A context if provided
        if a2a_context:
            request.params["a2a_context"] = a2a_context

        # Validate request structure
        request_data = request.model_dump(exclude_none=True)

        # Build headers
        headers = self._build_headers()

        # Send via transport
        response_data = self.transport.post(
            endpoint="/api/v1/jsonrpc",
            data=request_data,
            headers=headers,
        )

        # Parse and validate response
        response = JsonRpcResponse(**response_data)

        # Handle errors
        if response.error:
            raise JsonRpcProtocolError(response.error)

        return response.result or {}

    def batch_call(
        self,
        requests: list[tuple[str, dict[str, Any] | None]],
    ) -> list[dict[str, Any]]:
        """Execute multiple JSON-RPC calls in a single batch."""
        # Implementation: batch request handling
        ...
```

**Responsibilities:**

- JSON-RPC 2.0 protocol compliance
- Request/response validation
- A2A context management
- No knowledge of business logic
- No direct network access

---

### 3. Service Layer

**File:** `src/agentcore_cli/services/agent.py`

```python
from __future__ import annotations

from typing import Any

from agentcore_cli.protocol.jsonrpc import JsonRpcClient

class AgentService:
    """High-level agent management operations.

    Provides business-focused methods that:
    - Accept domain-specific parameters
    - Validate business rules
    - Transform data for API
    - Handle domain-level errors
    - Abstract away JSON-RPC details
    """

    def __init__(self, client: JsonRpcClient) -> None:
        self.client = client

    def register(
        self,
        name: str,
        capabilities: list[str],
        cost_per_request: float = 0.01,
        requirements: dict[str, Any] | None = None,
    ) -> str:
        """Register a new agent.

        Args:
            name: Agent name
            capabilities: List of capabilities
            cost_per_request: Cost per request in USD
            requirements: Optional requirements dictionary

        Returns:
            Agent ID of registered agent

        Raises:
            ValidationError: Invalid parameters
            AgentAlreadyExistsError: Agent name already registered
            ServiceError: API-level errors
        """
        # Business logic validation
        if not capabilities:
            raise ValidationError("At least one capability is required")

        # Transform data for API
        params = {
            "name": name,
            "capabilities": capabilities,
            "cost_per_request": cost_per_request,
        }
        if requirements:
            params["requirements"] = requirements

        # Call JSON-RPC method
        result = self.client.call("agent.register", params)

        # Extract and validate result
        agent_id = result.get("agent_id")
        if not agent_id:
            raise ServiceError("API did not return agent_id")

        return agent_id

    def list_agents(
        self,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List registered agents."""
        # Implementation: list agents
        ...

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        """Get agent details."""
        # Implementation: get agent
        ...

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent."""
        # Implementation: remove agent
        ...

    def search_agents(
        self,
        capabilities: list[str],
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Search agents by capabilities."""
        # Implementation: search agents
        ...
```

**Responsibilities:**

- Business logic and validation
- Domain-specific error handling
- Data transformation
- No knowledge of JSON-RPC details
- No knowledge of transport

---

### 4. CLI Layer

**File:** `src/agentcore_cli/commands/agent.py`

```python
from __future__ import annotations

import json
import sys
from typing import Annotated, Optional

import typer
from rich.console import Console

from agentcore_cli.container import get_agent_service
from agentcore_cli.exceptions import (
    AgentCoreError,
    AuthenticationError,
    ValidationError,
)
from agentcore_cli.formatters import (
    format_error,
    format_json,
    format_success,
    format_table,
)

app = typer.Typer(
    name="agent",
    help="Manage agent lifecycle and discovery",
    no_args_is_help=True,
)

console = Console()

@app.command()
def register(
    name: Annotated[str, typer.Option("--name", "-n", help="Agent name")],
    capabilities: Annotated[str, typer.Option("--capabilities", "-c", help="Comma-separated capabilities")],
    cost_per_request: Annotated[float, typer.Option("--cost-per-request", help="Cost per request")] = 0.01,
    requirements: Annotated[Optional[str], typer.Option("--requirements", "-r", help="JSON requirements")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="JSON output")] = False,
) -> None:
    """Register a new agent with AgentCore."""
    try:
        # Parse CLI inputs
        cap_list = [c.strip() for c in capabilities.split(",") if c.strip()]

        req_dict = {}
        if requirements:
            try:
                req_dict = json.loads(requirements)
                if not isinstance(req_dict, dict):
                    console.print(format_error("Requirements must be a JSON object"))
                    sys.exit(2)
            except json.JSONDecodeError as e:
                console.print(format_error(f"Invalid JSON: {e}"))
                sys.exit(2)

        # Get service from DI container
        service = get_agent_service()

        # Call service method
        agent_id = service.register(
            name=name,
            capabilities=cap_list,
            cost_per_request=cost_per_request,
            requirements=req_dict,
        )

        # Format output
        if json_output:
            console.print(format_json({"agent_id": agent_id}))
        else:
            console.print(format_success(f"Agent registered: {agent_id}"))
            console.print(f"  Name: {name}")
            console.print(f"  Capabilities: {', '.join(cap_list)}")

    except ValidationError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(2)
    except AuthenticationError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(4)
    except AgentCoreError as e:
        console.print(format_error(str(e)))
        raise typer.Exit(1)
```

**Responsibilities:**

- Argument parsing
- User interaction
- Output formatting
- Exit code handling
- No business logic
- No protocol knowledge

---

### 5. Dependency Injection Container

**File:** `src/agentcore_cli/container.py`

```python
from __future__ import annotations

from functools import lru_cache

from agentcore_cli.config import Config
from agentcore_cli.protocol.jsonrpc import JsonRpcClient
from agentcore_cli.services.agent import AgentService
from agentcore_cli.services.task import TaskService
from agentcore_cli.transport.http import HttpTransport

@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load and cache configuration."""
    return Config.load()

@lru_cache(maxsize=1)
def get_transport() -> HttpTransport:
    """Create and cache HTTP transport."""
    config = get_config()
    return HttpTransport(
        base_url=config.api.url,
        timeout=config.api.timeout,
        retries=config.api.retries,
        verify_ssl=config.api.verify_ssl,
    )

@lru_cache(maxsize=1)
def get_jsonrpc_client() -> JsonRpcClient:
    """Create and cache JSON-RPC client."""
    config = get_config()
    transport = get_transport()
    auth_token = config.auth.token if config.auth.type == "jwt" else None
    return JsonRpcClient(
        transport=transport,
        auth_token=auth_token,
    )

def get_agent_service() -> AgentService:
    """Create agent service instance."""
    client = get_jsonrpc_client()
    return AgentService(client)

def get_task_service() -> TaskService:
    """Create task service instance."""
    client = get_jsonrpc_client()
    return TaskService(client)
```

**Responsibilities:**

- Object creation and wiring
- Configuration management
- Instance caching
- Dependency resolution

---

## Migration Strategy

### Phase 1: Create New Architecture (No Breaking Changes)

**Goal:** Implement new architecture alongside existing code

**Tasks:**

1. Create new directory structure:

   ```plaintext
   src/agentcore_cli/
   ├── transport/
   │   ├── __init__.py
   │   └── http.py
   ├── protocol/
   │   ├── __init__.py
   │   └── jsonrpc.py
   ├── services/
   │   ├── __init__.py
   │   ├── agent.py
   │   ├── task.py
   │   ├── session.py
   │   └── workflow.py
   ├── container.py
   └── [existing files unchanged]
   ```

2. Implement transport layer with tests
3. Implement protocol layer with tests
4. Implement service layer with tests
5. Implement DI container

**Success Criteria:**

- All new layers have 100% test coverage
- New code passes all linting and type checks
- Existing CLI commands still work (unchanged)

**Duration:** 2-3 days

---

### Phase 2: Migrate One Command (Proof of Concept)

**Goal:** Migrate `agent register` command to validate approach

**Tasks:**

1. Create new version of `agent.py` command file
2. Update to use service layer instead of direct client
3. Add integration tests for new command
4. Run side-by-side with old implementation
5. Document differences and lessons learned

**Success Criteria:**

- `agent register` works with new architecture
- Same CLI interface (no user-facing changes)
- Proper JSON-RPC 2.0 format sent to API
- Tests pass for both old and new versions

**Duration:** 1 day

---

### Phase 3: Migrate Remaining Commands

**Goal:** Complete migration of all commands

**Tasks:**

1. Migrate all `agent.*` commands
2. Migrate all `task.*` commands
3. Migrate all `session.*` commands
4. Migrate all `workflow.*` commands
5. Migrate all `config.*` commands
6. Update integration tests
7. Update E2E tests

**Success Criteria:**

- All commands use new architecture
- All tests pass
- No protocol compliance issues
- No regression in functionality

**Duration:** 3-4 days

---

### Phase 4: Remove Old Code and Documentation

**Goal:** Clean up deprecated code and update docs

**Tasks:**

1. Remove old `client.py` (now replaced by transport + protocol)
2. Remove old command implementations
3. Update `CLAUDE.md` with new architecture
4. Update `README.md` with architecture diagrams
5. Create developer guide for adding new commands
6. Update contribution guidelines

**Success Criteria:**

- No dead code in repository
- Documentation reflects new architecture
- Clear guide for contributors
- All tests still pass

**Duration:** 1 day

---

## Implementation Roadmap

### Week 1: Foundation

**Days 1-3: Phase 1**

- [ ] Create transport layer
- [ ] Create protocol layer
- [ ] Create service layer
- [ ] Create DI container
- [ ] Write comprehensive tests

**Days 4-5: Phase 2**

- [ ] Migrate `agent register` command
- [ ] Validate approach
- [ ] Document findings

### Week 2: Migration

**Days 1-4: Phase 3**

- [ ] Migrate all agent commands
- [ ] Migrate all task commands
- [ ] Migrate all session commands
- [ ] Migrate all workflow commands
- [ ] Update all tests

**Day 5: Phase 4**

- [ ] Remove old code
- [ ] Update documentation
- [ ] Final testing

---

## Testing Strategy

### Unit Tests (per layer)

**Transport Layer:**

```python
def test_http_transport_post_success():
    """Test successful HTTP POST request."""

def test_http_transport_retry_on_failure():
    """Test retry logic with exponential backoff."""

def test_http_transport_timeout():
    """Test timeout handling."""
```

**Protocol Layer:**

```python
def test_jsonrpc_request_validation():
    """Test JSON-RPC request model validation."""

def test_jsonrpc_call_builds_correct_request():
    """Test that call() builds proper JSON-RPC request."""

def test_jsonrpc_batch_call():
    """Test batch request handling."""
```

**Service Layer:**

```python
def test_agent_service_register_validation():
    """Test parameter validation in register()."""

def test_agent_service_register_calls_correct_method():
    """Test that register() calls agent.register via client."""

def test_agent_service_handles_api_errors():
    """Test domain error translation."""
```

**CLI Layer:**

```python
def test_agent_register_command_parses_args():
    """Test CLI argument parsing."""

def test_agent_register_command_calls_service():
    """Test that command calls service layer."""

def test_agent_register_command_formats_output():
    """Test output formatting."""
```

---

### Integration Tests

**Test actual JSON-RPC compliance:**

```python
@pytest.mark.integration
def test_cli_sends_proper_jsonrpc_request(mock_api_server):
    """Verify CLI sends JSON-RPC 2.0 compliant request.

    Validates:
    - Request has 'jsonrpc: "2.0"'
    - Request has 'method' field
    - Request has 'params' object (NOT flat dictionary)
    - Request has 'id' field
    """
    # Start mock server that validates request format
    # Execute CLI command
    # Assert request structure matches JSON-RPC 2.0 spec
```

---

## Benefits

### For Users

- **Functional:** All CLI commands work correctly
- **Reliable:** Proper error messages and handling
- **Consistent:** Same patterns across all commands

### For Developers

- **Maintainable:** Clear layer separation
- **Testable:** Each layer tested independently
- **Extensible:** Easy to add new commands
- **Debuggable:** Clear error propagation

### For Architecture

- **Compliant:** Full A2A protocol compliance
- **Scalable:** Can add batch operations, streaming, etc.
- **Flexible:** Can swap transports (WebSocket, gRPC)
- **Robust:** Type-safe with Pydantic models

---

## Risk Mitigation

### Risk: Migration breaks existing functionality

**Mitigation:**

- Implement new architecture alongside old code
- Run both versions in parallel during Phase 2
- Comprehensive test coverage before migration
- Gradual rollout (one command at a time)

### Risk: Performance degradation

**Mitigation:**

- Benchmark before and after migration
- Connection pooling in transport layer
- Caching in DI container
- Monitor performance metrics

### Risk: Increased complexity

**Mitigation:**

- Clear documentation for each layer
- Developer guide for adding commands
- Code examples in documentation
- Consistent patterns across codebase

---

## Success Metrics

### Functional

- [ ] All CLI commands execute successfully
- [ ] All requests are JSON-RPC 2.0 compliant
- [ ] No protocol errors in logs
- [ ] Integration tests pass

### Code Quality

- [ ] 90%+ test coverage maintained
- [ ] No mypy errors (strict mode)
- [ ] No ruff linting issues
- [ ] All Pydantic models validated

### Documentation

- [ ] Architecture documented in CLAUDE.md
- [ ] Developer guide complete
- [ ] API reference updated
- [ ] Migration guide available

---

## Open Questions

1. **Backward Compatibility:** Should we maintain compatibility with any existing CLI scripts?
   - **Recommendation:** No breaking changes to CLI interface, only internal architecture

2. **Configuration:** Should service layer have its own config models?
   - **Recommendation:** Yes, Pydantic models for service-specific config

3. **Error Handling:** Should we create custom exception hierarchy?
   - **Recommendation:** Yes, separate exceptions per layer

4. **Async Support:** Should we support async/await in future?
   - **Recommendation:** Design interfaces to support future async migration

---

## Conclusion

This redesign addresses the fundamental architectural issues in the CLI while maintaining user-facing compatibility. The layered approach ensures A2A protocol compliance, improves testability, and creates a solid foundation for future enhancements.

The migration strategy minimizes risk through incremental rollout and comprehensive testing. The investment in proper architecture will pay dividends in maintainability, extensibility, and developer productivity.

**Recommendation:** Approve and proceed with implementation following the phased approach outlined above.
