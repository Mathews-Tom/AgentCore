# CLI Layer Specification v2.0 (Redesigned)

**Component:** CLI Layer
**Purpose:** Developer-friendly command-line interface for AgentCore with A2A protocol compliance
**Owner:** Platform Team
**Dependencies:** A2A Protocol Layer (JSON-RPC API)
**Created:** 2025-09-30
**Updated:** 2025-10-22
**Status:** Redesign - Implementation Pending
**Version:** 2.0

---

## Overview

The CLI Layer provides a Python-based command-line interface that wraps the AgentCore JSON-RPC 2.0 API with developer-friendly commands. This redesign addresses critical A2A protocol compliance issues through a properly layered architecture with clear separation of concerns.

**Target Users:**

- Developers prototyping agent systems
- DevOps engineers managing agent deployments
- QA teams testing agent workflows
- Technical users preferring CLI over API integration

**Design Philosophy:**

- Mirror familiar CLI patterns (docker, kubectl, git)
- Sensible defaults with explicit overrides
- Progressive disclosure (simple → advanced)
- Machine-readable output for scripting
- **NEW:** A2A protocol compliance through proper layering
- **NEW:** Testability at every layer
- **NEW:** Extensibility for future transport mechanisms

---

## Problem Statement (v1.0)

The initial CLI implementation (v1.0) has critical architectural issues:

1. **Protocol Violation**: Parameters sent as flat dictionary instead of wrapped in `params` object
2. **Mixing of Concerns**: JSON-RPC request building mixed with command logic
3. **Tight Coupling**: Commands directly create clients, no dependency injection
4. **Test Brittleness**: Cannot test command logic without full HTTP integration

**Impact:** All CLI commands currently fail with JSON-RPC protocol errors.

---

## Architecture v2.0

### 4-Layer Architecture

The redesigned CLI follows a strict 4-layer architecture where each layer has a single, well-defined responsibility:

```plaintext
┌────────────────────────────────────────────────┐
│          Layer 1: CLI Layer (Typer)            │
│  • Argument parsing & validation               │
│  • User interaction (prompts, confirmations)   │
│  • Output formatting (tables, JSON, errors)    │
│  • Exit code handling (0, 1, 2, 3, 4)          │
│  • NO business logic                           │
│  • NO protocol knowledge                       │
└────────────────────────┬───────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────┐
│        Layer 2: Service Layer (Facade)         │
│  • AgentService, TaskService, etc.             │
│  • High-level business operations              │
│  • Parameter validation & transformation       │
│  • Domain error handling                       │
│  • NO JSON-RPC knowledge                       │
│  • NO transport knowledge                      │
└────────────────────────┬───────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────┐
│     Layer 3: Protocol Layer (JSON-RPC 2.0)     │
│  • JsonRpcClient                               │
│  • Request/Response models (Pydantic)          │
│  • Batch request handling                      │
│  • A2A context management                      │
│  • Protocol-level error translation            │
│  • NO network operations                       │
│  • NO business logic                           │
└────────────────────────┬───────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────┐
│       Layer 4: Transport Layer (HTTP)          │
│  • HttpTransport                               │
│  • Connection pooling                          │
│  • Retry logic with exponential backoff        │
│  • SSL/TLS verification                        │
│  • Timeout handling                            │
│  • Network error translation                   │
│  • NO JSON-RPC knowledge                       │
└────────────────────────────────────────────────┘
                         │
                         │ HTTPS
                         ▼
┌────────────────────────────────────────────────┐
│         AgentCore API Server                   │
│         (JSON-RPC 2.0 Endpoint)                │
└────────────────────────────────────────────────┘
```

### Layer Responsibilities

#### Layer 1: CLI Layer

**Location:** `src/agentcore_cli/commands/`

**Responsibilities:**
- Parse command-line arguments using Typer
- Validate CLI-level inputs (required args, formats)
- Handle user interaction (prompts, confirmations)
- Format output (table, JSON, tree)
- Map exit codes (0=success, 1=error, 2=usage, 3=connection, 4=auth)
- Delegate to service layer

**NOT Responsible For:**
- Business logic
- Parameter transformation
- JSON-RPC protocol
- Network communication

**Example:**
```python
@app.command()
def register(
    name: str,
    capabilities: str,
    cost_per_request: float = 0.01,
    json_output: bool = False,
) -> None:
    """Register a new agent."""
    # Parse CLI inputs
    cap_list = [c.strip() for c in capabilities.split(",")]

    # Get service from DI container
    service = get_agent_service()

    # Call service method
    agent_id = service.register(
        name=name,
        capabilities=cap_list,
        cost_per_request=cost_per_request,
    )

    # Format output
    if json_output:
        console.print(format_json({"agent_id": agent_id}))
    else:
        console.print(format_success(f"Agent registered: {agent_id}"))
```

#### Layer 2: Service Layer

**Location:** `src/agentcore_cli/services/`

**Responsibilities:**
- Provide high-level business operations
- Validate business rules
- Transform data for API
- Handle domain-level errors
- Abstract JSON-RPC details

**NOT Responsible For:**
- CLI argument parsing
- JSON-RPC protocol formatting
- Network communication
- Output formatting

**Services:**
- `AgentService`: Agent registration, discovery, removal
- `TaskService`: Task creation, execution, monitoring
- `SessionService`: Session management
- `WorkflowService`: Workflow execution
- `ConfigService`: Configuration management

**Example:**
```python
class AgentService:
    def __init__(self, client: JsonRpcClient) -> None:
        self.client = client

    def register(
        self,
        name: str,
        capabilities: list[str],
        cost_per_request: float = 0.01,
        requirements: dict[str, Any] | None = None,
    ) -> str:
        """Register a new agent."""
        # Business validation
        if not capabilities:
            raise ValidationError("At least one capability required")

        # Prepare parameters
        params = {
            "name": name,
            "capabilities": capabilities,
            "cost_per_request": cost_per_request,
        }
        if requirements:
            params["requirements"] = requirements

        # Call JSON-RPC method
        result = self.client.call("agent.register", params)

        # Validate result
        agent_id = result.get("agent_id")
        if not agent_id:
            raise ServiceError("API did not return agent_id")

        return agent_id
```

#### Layer 3: Protocol Layer

**Location:** `src/agentcore_cli/protocol/`

**Responsibilities:**
- Enforce JSON-RPC 2.0 specification
- Validate request/response against Pydantic models
- Manage A2A context (trace_id, source_agent, etc.)
- Handle batch requests
- Translate protocol errors

**NOT Responsible For:**
- Business logic
- Network communication
- CLI concerns

**Models:**
```python
class JsonRpcRequest(BaseModel):
    jsonrpc: str = Field(default="2.0", const=True)
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    id: int | str

class JsonRpcResponse(BaseModel):
    jsonrpc: str = Field(default="2.0", const=True)
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None
    id: int | str | None

class JsonRpcError(BaseModel):
    code: int
    message: str
    data: dict[str, Any] | None = None
```

**Client:**
```python
class JsonRpcClient:
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
        """Execute JSON-RPC method call."""
        # Build request using Pydantic model
        request = JsonRpcRequest(
            method=method,
            params=params or {},  # CRITICAL: params wrapper
            id=self._next_id(),
        )

        # Add A2A context if provided
        if a2a_context:
            request.params["a2a_context"] = a2a_context

        # Validate request
        request_data = request.model_dump(exclude_none=True)

        # Send via transport
        response_data = self.transport.post(
            endpoint="/api/v1/jsonrpc",
            data=request_data,
            headers=self._build_headers(),
        )

        # Parse and validate response
        response = JsonRpcResponse(**response_data)

        # Handle errors
        if response.error:
            raise JsonRpcProtocolError(response.error)

        return response.result or {}
```

#### Layer 4: Transport Layer

**Location:** `src/agentcore_cli/transport/`

**Responsibilities:**
- HTTP communication
- Connection pooling
- Retry logic with exponential backoff
- Timeout handling
- SSL/TLS verification
- Network error translation

**NOT Responsible For:**
- JSON-RPC protocol
- Business logic
- Request/response validation

**Implementation:**
```python
class HttpTransport:
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
        """Send HTTP POST request."""
        response = self.session.post(
            f"{self.base_url}{endpoint}",
            json=data,
            headers=headers,
            timeout=self.timeout,
            verify=self.verify_ssl,
        )

        # Handle HTTP errors
        if response.status_code >= 400:
            raise TransportError(response.status_code, response.text)

        # Return parsed JSON
        return response.json()
```

---

## Dependency Injection

**Location:** `src/agentcore_cli/container.py`

A dependency injection container manages object creation and wiring:

```python
from functools import lru_cache

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
    return JsonRpcClient(transport=transport, auth_token=auth_token)

def get_agent_service() -> AgentService:
    """Create agent service instance."""
    return AgentService(get_jsonrpc_client())
```

**Benefits:**
- Centralized configuration
- Easy mocking for tests
- Clear dependency graph
- Lazy initialization

---

## Technology Stack

**Core Framework:**
- **Language:** Python 3.12+
- **CLI Framework:** Typer (decided)
- **HTTP Client:** requests with retry logic
- **Output Formatting:** rich library
- **Configuration:** TOML (config.toml)
- **Validation:** Pydantic v2

**Distribution:**
- **Package Name:** `agentcore-cli`
- **Entry Point:** `agentcore` command
- **Installation:** `pip install agentcore-cli` or `uv add agentcore-cli`

**Development:**
- **Package Manager:** uv
- **Testing:** pytest with pytest-asyncio
- **Type Checking:** mypy (strict mode)
- **Linting:** ruff

---

## Command Structure

### Top-Level Commands

```bash
agentcore [OPTIONS] COMMAND [ARGS]...

Commands:
  agent      Manage agent lifecycle and discovery
  task       Manage task creation and execution
  session    Manage session state
  workflow   Execute and monitor workflows
  config     Manage CLI configuration
  health     Check API server health
  version    Show CLI version
```

### Agent Commands

```bash
agentcore agent register --name NAME --capabilities CAPS [OPTIONS]
agentcore agent list [--status STATUS] [--limit N] [--json]
agentcore agent info AGENT_ID [--json]
agentcore agent remove AGENT_ID [--force] [--json]
agentcore agent search --capability CAP [--limit N] [--json]
```

### Task Commands

```bash
agentcore task create --description DESC [OPTIONS]
agentcore task list [--status STATUS] [--limit N] [--json]
agentcore task info TASK_ID [--json]
agentcore task cancel TASK_ID [--force] [--json]
agentcore task logs TASK_ID [--follow] [--lines N]
```

### Session Commands

```bash
agentcore session create --name NAME [OPTIONS]
agentcore session list [--state STATE] [--json]
agentcore session info SESSION_ID [--json]
agentcore session delete SESSION_ID [--force] [--json]
agentcore session restore SESSION_ID [--json]
```

### Workflow Commands

```bash
agentcore workflow run --file WORKFLOW.yaml [--json]
agentcore workflow list [--status STATUS] [--json]
agentcore workflow info WORKFLOW_ID [--json]
agentcore workflow stop WORKFLOW_ID [--force] [--json]
```

### Config Commands

```bash
agentcore config show [--global|--project] [--json]
agentcore config set KEY VALUE [--global|--project]
agentcore config get KEY [--global|--project]
agentcore config init [--global|--project]
```

---

## Configuration Management

### Configuration Precedence

1. CLI arguments (highest priority)
2. Environment variables (`AGENTCORE_*`)
3. Project config (`.agentcore.toml`)
4. Global config (`~/.agentcore/config.toml`)
5. Defaults (lowest priority)

### Configuration Schema

```toml
[api]
url = "http://localhost:8001"
timeout = 30
retries = 3
verify_ssl = true

[auth]
type = "jwt"  # "none", "jwt", "api_key"
token = ""

[defaults]
output_format = "table"  # "table", "json", "tree"
limit = 100
```

---

## Error Handling

### Exit Codes

- **0:** Success
- **1:** General error
- **2:** Usage error (invalid arguments)
- **3:** Connection error (cannot reach API)
- **4:** Authentication error (invalid token)

### Error Hierarchy

```python
AgentCoreError (base)
├── ValidationError (exit 2)
├── ConnectionError (exit 3)
│   ├── TimeoutError
│   └── NetworkError
├── AuthenticationError (exit 4)
├── JsonRpcProtocolError (exit 1)
│   ├── InvalidRequestError
│   ├── MethodNotFoundError
│   └── InvalidParamsError
└── ServiceError (exit 1)
    ├── AgentNotFoundError
    ├── TaskNotFoundError
    └── SessionNotFoundError
```

---

## Output Formats

### Table Format (Default)

```
┌─────────────┬──────────────┬────────┬──────────────────────┐
│ Agent ID    │ Name         │ Status │ Capabilities         │
├─────────────┼──────────────┼────────┼──────────────────────┤
│ agent-001   │ analyzer     │ active │ python, analysis     │
│ agent-002   │ tester       │ active │ testing, qa          │
└─────────────┴──────────────┴────────┴──────────────────────┘
```

### JSON Format (--json)

```json
[
  {
    "agent_id": "agent-001",
    "name": "analyzer",
    "status": "active",
    "capabilities": ["python", "analysis"]
  }
]
```

### Tree Format (--tree)

```
agent-001 (analyzer)
├─ status: active
└─ capabilities
   ├─ python
   └─ analysis
```

---

## Testing Strategy

### Unit Tests (per layer)

**Transport Layer:**
```python
def test_http_transport_post_success()
def test_http_transport_retry_on_failure()
def test_http_transport_timeout()
```

**Protocol Layer:**
```python
def test_jsonrpc_request_validation()
def test_jsonrpc_call_builds_correct_request()
def test_jsonrpc_batch_call()
def test_jsonrpc_a2a_context_injection()
```

**Service Layer:**
```python
def test_agent_service_register_validation()
def test_agent_service_register_calls_correct_method()
def test_agent_service_handles_api_errors()
```

**CLI Layer:**
```python
def test_agent_register_command_parses_args()
def test_agent_register_command_calls_service()
def test_agent_register_command_formats_output()
```

### Integration Tests

```python
@pytest.mark.integration
def test_cli_sends_proper_jsonrpc_request(mock_api_server):
    """Verify CLI sends JSON-RPC 2.0 compliant request."""
    # Validates:
    # - Request has 'jsonrpc: "2.0"'
    # - Request has 'method' field
    # - Request has 'params' object (NOT flat dictionary)
    # - Request has 'id' field
```

### E2E Tests

```python
@pytest.mark.e2e
def test_agent_lifecycle(live_api):
    """Test complete agent lifecycle through CLI."""
    # register → list → info → remove
```

---

## Migration Strategy

### Phase 1: Foundation (2-3 days)

**Goal:** Build new architecture alongside existing code

1. Create new directory structure
2. Implement transport layer with tests
3. Implement protocol layer with tests
4. Implement service layer with tests
5. Implement DI container

**Success Criteria:**
- All new layers have 100% test coverage
- New code passes mypy strict mode
- Existing CLI still works

### Phase 2: Proof of Concept (1 day)

**Goal:** Migrate `agent register` command

1. Create new `agent.py` using service layer
2. Run side-by-side with old implementation
3. Validate JSON-RPC 2.0 compliance
4. Document findings

**Success Criteria:**
- `agent register` sends proper JSON-RPC
- Tests pass for both versions

### Phase 3: Full Migration (3-4 days)

**Goal:** Migrate all commands

1. Migrate all agent commands
2. Migrate all task commands
3. Migrate all session commands
4. Migrate all workflow commands
5. Update all tests

**Success Criteria:**
- All commands use new architecture
- All tests pass
- No protocol errors

### Phase 4: Cleanup (1 day)

**Goal:** Remove old code

1. Delete old `client.py`
2. Remove old command implementations
3. Update documentation
4. Final testing

**Success Criteria:**
- No dead code
- Documentation current
- All tests pass

---

## Acceptance Criteria

- [ ] All layers implemented with clear separation
- [ ] JSON-RPC 2.0 compliance verified
- [ ] 90%+ test coverage maintained
- [ ] All commands functional
- [ ] No mypy errors (strict mode)
- [ ] No ruff linting issues
- [ ] Documentation complete
- [ ] Migration guide available

---

## Success Metrics

**Functional:**
- All CLI commands execute successfully
- All requests are JSON-RPC 2.0 compliant
- No protocol errors in logs

**Code Quality:**
- 90%+ test coverage
- Mypy strict mode passes
- Ruff linting passes
- Pydantic models validated

**Architecture:**
- Clear layer separation
- No circular dependencies
- Proper dependency injection
- Extensible design

---

## References

- A2A Protocol Specification: docs/specs/a2a-protocol/spec.md
- JSON-RPC 2.0 Specification: https://www.jsonrpc.org/specification
- CLI Redesign Proposal: docs/architecture/cli-redesign-proposal.md
- Original CLI Spec (v1.0): docs/specs/cli-layer/spec-v1.0.md (archived)

---

## Appendix A: Directory Structure

```
src/agentcore_cli/
├── __init__.py
├── main.py                  # CLI entry point
├── container.py             # DI container
├── config.py                # Configuration models
├── exceptions.py            # Error hierarchy
├── formatters.py            # Output formatting
├── commands/                # Layer 1: CLI
│   ├── __init__.py
│   ├── agent.py
│   ├── task.py
│   ├── session.py
│   ├── workflow.py
│   └── config.py
├── services/                # Layer 2: Service
│   ├── __init__.py
│   ├── agent.py
│   ├── task.py
│   ├── session.py
│   └── workflow.py
├── protocol/                # Layer 3: Protocol
│   ├── __init__.py
│   ├── jsonrpc.py
│   └── models.py
└── transport/               # Layer 4: Transport
    ├── __init__.py
    └── http.py
```

---

## Appendix B: Key Changes from v1.0

| Aspect | v1.0 (Current) | v2.0 (Redesigned) |
|--------|----------------|-------------------|
| **Architecture** | Monolithic client | 4-layer separation |
| **JSON-RPC** | Params as flat dict | Proper params wrapper |
| **Testing** | Integration only | Unit + Integration |
| **DI** | Manual creation | Container-based |
| **Protocol** | No validation | Pydantic models |
| **Extensibility** | Tightly coupled | Pluggable layers |
| **Compliance** | Violates A2A | Fully compliant |

---

**End of Specification v2.0**
