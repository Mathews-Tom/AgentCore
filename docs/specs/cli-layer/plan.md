# CLI Layer Implementation Plan v2.0 (Redesign)

**Component:** CLI Layer Redesign
**Timeline:** 2 weeks (10 working days)
**Team:** 1 senior Python developer
**Total Effort:** 42 story points
**Created:** 2025-09-30
**Updated:** 2025-10-22
**Version:** 2.0
**Status:** Redesign - Ready for Implementation

---

## Executive Summary

The CLI Layer v2.0 redesign addresses critical A2A protocol compliance issues discovered in v1.0 by implementing a properly layered architecture with clear separation of concerns. This redesign ensures JSON-RPC 2.0 compliance while maintaining the developer-friendly interface.

**Business Value:**

- **CRITICAL:** Fixes protocol violation preventing all CLI commands from working
- Maintains developer adoption gains from v1.0
- Establishes extensible architecture for future enhancements
- Enables proper testing at every layer
- **Risk Mitigation:** Phased migration prevents regression

**Technical Approach:**

- 4-layer architecture (CLI → Service → Protocol → Transport)
- Pydantic models for JSON-RPC 2.0 validation
- Dependency injection for testability
- Side-by-side migration to minimize risk
- Comprehensive test coverage at each layer

**Why Redesign?**

v1.0 CLI sends invalid JSON-RPC 2.0 requests:
- **Current:** `{"jsonrpc": "2.0", "method": "agent.register", "id": 1, "name": "...", "capabilities": [...]}`
- **Required:** `{"jsonrpc": "2.0", "method": "agent.register", "params": {"name": "...", "capabilities": [...]}, "id": 1}`

**Impact:** All CLI commands currently fail with protocol errors.

---

## Phase Overview

### Phase 1: Foundation (18 SP, Days 1-3)

**Duration:** 2-3 working days
**Goal:** Build new 4-layer architecture alongside existing code
**Team:** 1 senior developer
**Risk:** Low (no changes to existing code)

**Deliverables:**
- Transport layer implementation
- Protocol layer implementation
- Service layer implementation
- DI container implementation
- Comprehensive unit tests

### Phase 2: Proof of Concept (8 SP, Days 4-5)

**Duration:** 1 working day
**Goal:** Validate architecture with single command migration
**Team:** 1 senior developer
**Risk:** Medium (first integration test)

**Deliverables:**
- Migrated `agent register` command
- Side-by-side validation
- Protocol compliance verification
- Migration learnings documented

### Phase 3: Full Migration (12 SP, Days 6-9)

**Duration:** 3-4 working days
**Goal:** Migrate all commands to new architecture
**Team:** 1 senior developer
**Risk:** Medium (bulk migration)

**Deliverables:**
- All agent commands migrated
- All task commands migrated
- All session commands migrated
- All workflow commands migrated
- All config commands migrated
- Updated integration tests
- Updated E2E tests

### Phase 4: Cleanup (4 SP, Day 10)

**Duration:** 1 working day
**Goal:** Remove old code and finalize documentation
**Team:** 1 senior developer
**Risk:** Low (cleanup only)

**Deliverables:**
- Old client.py removed
- Old command implementations removed
- Documentation updated
- Migration guide published
- Final testing complete

---

## Detailed Phase Breakdown

### Phase 1: Foundation

#### 1.1 Transport Layer (5 SP)

**Ticket:** CLI-R001
**Files:** `src/agentcore_cli/transport/http.py`

**Implementation:**
```python
class HttpTransport:
    """HTTP transport for JSON-RPC requests."""

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        retries: int = 3,
        verify_ssl: bool = True,
    ) -> None:
        # Initialize session with retry strategy

    def post(
        self,
        endpoint: str,
        data: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        # Send HTTP POST request
        # Handle network errors
        # Return parsed JSON
```

**Responsibilities:**
- HTTP communication only
- Connection pooling
- Retry logic with exponential backoff
- SSL/TLS verification
- Network error translation

**Tests:**
- `test_http_transport_post_success()`
- `test_http_transport_retry_on_failure()`
- `test_http_transport_timeout()`
- `test_http_transport_ssl_verification()`
- `test_http_transport_connection_error()`

**Acceptance Criteria:**
- [ ] All HTTP operations work correctly
- [ ] Retry logic tested with 3 attempts
- [ ] SSL verification configurable
- [ ] Network errors properly translated
- [ ] 100% test coverage

#### 1.2 Protocol Layer (7 SP)

**Ticket:** CLI-R002
**Files:**
- `src/agentcore_cli/protocol/jsonrpc.py`
- `src/agentcore_cli/protocol/models.py`

**Implementation:**
```python
class JsonRpcRequest(BaseModel):
    jsonrpc: str = Field(default="2.0", const=True)
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    id: int | str

class JsonRpcClient:
    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        a2a_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Build request with Pydantic validation
        # Add A2A context
        # Call transport layer
        # Parse and validate response
        # Handle protocol errors
```

**Responsibilities:**
- JSON-RPC 2.0 specification enforcement
- Request/response Pydantic validation
- A2A context management
- Batch request handling
- Protocol error translation

**Tests:**
- `test_jsonrpc_request_validation()`
- `test_jsonrpc_call_builds_correct_request()`
- `test_jsonrpc_params_wrapper_present()`
- `test_jsonrpc_a2a_context_injection()`
- `test_jsonrpc_batch_call()`
- `test_jsonrpc_error_handling()`

**Acceptance Criteria:**
- [ ] All requests have proper `params` wrapper
- [ ] Pydantic validates all requests/responses
- [ ] A2A context properly injected
- [ ] Batch requests supported
- [ ] Protocol errors translated
- [ ] 100% test coverage

#### 1.3 Service Layer (4 SP)

**Ticket:** CLI-R003
**Files:**
- `src/agentcore_cli/services/agent.py`
- `src/agentcore_cli/services/task.py`
- `src/agentcore_cli/services/session.py`
- `src/agentcore_cli/services/workflow.py`

**Implementation:**
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
        # Business validation
        # Prepare parameters
        # Call JSON-RPC method
        # Validate result
        # Return domain object
```

**Responsibilities:**
- High-level business operations
- Parameter validation
- Data transformation
- Domain error handling
- Abstract JSON-RPC details

**Tests:**
- `test_agent_service_register_validation()`
- `test_agent_service_register_success()`
- `test_task_service_create_validation()`
- `test_session_service_lifecycle()`
- `test_workflow_service_execution()`

**Acceptance Criteria:**
- [ ] All services implement business operations
- [ ] Parameter validation enforced
- [ ] Domain errors properly raised
- [ ] No JSON-RPC knowledge in services
- [ ] 100% test coverage

#### 1.4 DI Container (2 SP)

**Ticket:** CLI-R004
**File:** `src/agentcore_cli/container.py`

**Implementation:**
```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config.load()

@lru_cache(maxsize=1)
def get_transport() -> HttpTransport:
    config = get_config()
    return HttpTransport(...)

@lru_cache(maxsize=1)
def get_jsonrpc_client() -> JsonRpcClient:
    return JsonRpcClient(get_transport(), ...)

def get_agent_service() -> AgentService:
    return AgentService(get_jsonrpc_client())
```

**Responsibilities:**
- Object creation and wiring
- Configuration management
- Instance caching
- Dependency resolution

**Tests:**
- `test_container_creates_transport()`
- `test_container_creates_client()`
- `test_container_creates_services()`
- `test_container_caching_works()`

**Acceptance Criteria:**
- [ ] All dependencies properly wired
- [ ] Configuration loaded once
- [ ] Instances cached appropriately
- [ ] Easy to mock for tests
- [ ] 100% test coverage

---

### Phase 2: Proof of Concept

#### 2.1 Migrate Agent Register Command (5 SP)

**Ticket:** CLI-R005
**File:** `src/agentcore_cli/commands/agent_v2.py` (temporary)

**Implementation:**
```python
@app.command()
def register(
    name: str,
    capabilities: str,
    cost_per_request: float = 0.01,
    json_output: bool = False,
) -> None:
    # Parse CLI inputs
    cap_list = [c.strip() for c in capabilities.split(",")]

    # Get service from container
    service = get_agent_service()

    # Call service
    agent_id = service.register(name, cap_list, cost_per_request)

    # Format output
    if json_output:
        console.print(format_json({"agent_id": agent_id}))
    else:
        console.print(format_success(f"Agent registered: {agent_id}"))
```

**Tests:**
- `test_agent_register_v2_parses_args()`
- `test_agent_register_v2_calls_service()`
- `test_agent_register_v2_formats_output()`
- `test_agent_register_v2_sends_proper_jsonrpc()` (integration)

**Acceptance Criteria:**
- [ ] Command works with new architecture
- [ ] Sends proper JSON-RPC 2.0 request
- [ ] Side-by-side validation passes
- [ ] All tests pass

#### 2.2 Validation & Documentation (3 SP)

**Ticket:** CLI-R006

**Tasks:**
1. Run side-by-side comparison (old vs new)
2. Capture actual JSON-RPC requests
3. Verify params wrapper present
4. Document learnings
5. Identify any issues for Phase 3

**Deliverables:**
- Validation report
- Protocol compliance proof
- Migration lessons learned
- Phase 3 adjustments (if needed)

**Acceptance Criteria:**
- [ ] JSON-RPC 2.0 compliance verified
- [ ] Side-by-side comparison documented
- [ ] Lessons learned captured
- [ ] Phase 3 plan validated

---

### Phase 3: Full Migration

#### 3.1 Migrate Agent Commands (3 SP)

**Ticket:** CLI-R007
**Files:** `src/agentcore_cli/commands/agent.py`

**Commands to Migrate:**
- `agent register` (already migrated in Phase 2)
- `agent list`
- `agent info`
- `agent remove`
- `agent search`

**Acceptance Criteria:**
- [ ] All agent commands use service layer
- [ ] All tests updated
- [ ] Integration tests pass
- [ ] E2E tests pass

#### 3.2 Migrate Task Commands (3 SP)

**Ticket:** CLI-R008
**Files:** `src/agentcore_cli/commands/task.py`

**Commands to Migrate:**
- `task create`
- `task list`
- `task info`
- `task cancel`
- `task logs`

**Acceptance Criteria:**
- [ ] All task commands use service layer
- [ ] All tests updated
- [ ] Integration tests pass
- [ ] E2E tests pass

#### 3.3 Migrate Session Commands (2 SP)

**Ticket:** CLI-R009
**Files:** `src/agentcore_cli/commands/session.py`

**Commands to Migrate:**
- `session create`
- `session list`
- `session info`
- `session delete`
- `session restore`

**Acceptance Criteria:**
- [ ] All session commands use service layer
- [ ] All tests updated
- [ ] Integration tests pass

#### 3.4 Migrate Workflow Commands (2 SP)

**Ticket:** CLI-R010
**Files:** `src/agentcore_cli/commands/workflow.py`

**Commands to Migrate:**
- `workflow run`
- `workflow list`
- `workflow info`
- `workflow stop`

**Acceptance Criteria:**
- [ ] All workflow commands use service layer
- [ ] All tests updated
- [ ] Integration tests pass

#### 3.5 Migrate Config Commands (2 SP)

**Ticket:** CLI-R011
**Files:** `src/agentcore_cli/commands/config.py`

**Commands to Migrate:**
- `config show`
- `config set`
- `config get`
- `config init`

**Acceptance Criteria:**
- [ ] All config commands use service layer
- [ ] All tests updated
- [ ] Integration tests pass

---

### Phase 4: Cleanup

#### 4.1 Remove Old Code (2 SP)

**Ticket:** CLI-R012

**Files to Remove:**
- `src/agentcore_cli/client.py` (old monolithic client)
- Any temporary v2 files
- Dead code from migration

**Acceptance Criteria:**
- [ ] Old client.py removed
- [ ] No dead code remains
- [ ] All imports updated
- [ ] All tests still pass

#### 4.2 Update Documentation (2 SP)

**Ticket:** CLI-R013

**Documentation to Update:**
- `README_CLI.md` - Update architecture diagrams
- `docs/api/cli.md` - Update API reference
- `CLAUDE.md` - Update development guide
- `CONTRIBUTING.md` - Update contribution guidelines

**New Documentation:**
- `docs/architecture/cli-migration-guide.md`
- `docs/architecture/cli-testing-guide.md`

**Acceptance Criteria:**
- [ ] All docs reflect new architecture
- [ ] Migration guide complete
- [ ] Testing guide complete
- [ ] Examples updated

---

## Testing Strategy

### Unit Tests (per layer)

**Transport Layer (10 tests):**
- HTTP operations
- Retry logic
- Timeout handling
- SSL verification
- Error translation

**Protocol Layer (15 tests):**
- Request validation
- Response validation
- Params wrapper
- A2A context injection
- Batch requests
- Error handling

**Service Layer (20 tests):**
- Business validation
- Method calls
- Error handling
- Data transformation

**CLI Layer (25 tests):**
- Argument parsing
- Service calls
- Output formatting
- Exit codes

**Total Unit Tests:** 70

### Integration Tests (15 tests)

- JSON-RPC 2.0 compliance validation
- End-to-end command execution
- Configuration precedence
- Error propagation

### E2E Tests (10 tests)

- Complete workflows
- Agent lifecycle
- Task lifecycle
- Session management

**Total Test Count:** 95 tests
**Target Coverage:** 90%+

---

## Risk Management

### Risk 1: Migration breaks existing functionality

**Mitigation:**
- Phase 2 proof of concept validates approach
- Side-by-side testing in Phase 2
- Gradual migration in Phase 3
- Comprehensive test coverage

**Contingency:**
- Rollback to old implementation
- Fix issues before proceeding
- Extend timeline if needed

### Risk 2: Performance degradation

**Mitigation:**
- Connection pooling in transport layer
- Caching in DI container
- Benchmark before/after

**Contingency:**
- Performance profiling
- Optimize hot paths
- Adjust caching strategy

### Risk 3: Increased complexity

**Mitigation:**
- Clear documentation
- Code examples
- Developer guide

**Contingency:**
- Simplify where possible
- More documentation
- Training materials

---

## Success Metrics

### Functional
- [ ] All CLI commands execute successfully
- [ ] All requests are JSON-RPC 2.0 compliant
- [ ] No protocol errors in logs
- [ ] Integration tests pass (95+)

### Code Quality
- [ ] 90%+ test coverage maintained
- [ ] No mypy errors (strict mode)
- [ ] No ruff linting issues
- [ ] All Pydantic models validated

### Documentation
- [ ] Architecture documented
- [ ] Migration guide complete
- [ ] Testing guide complete
- [ ] Developer guide updated

### Performance
- [ ] No regression in command latency
- [ ] Connection pooling reduces overhead
- [ ] Memory usage stable

---

## Timeline

```
Week 1: Foundation & POC
├─ Mon-Wed: Phase 1 (Foundation)
│  ├─ Transport layer
│  ├─ Protocol layer
│  ├─ Service layer
│  └─ DI container
└─ Thu-Fri: Phase 2 (POC)
   ├─ Migrate agent register
   └─ Validation & documentation

Week 2: Migration & Cleanup
├─ Mon-Thu: Phase 3 (Migration)
│  ├─ Agent commands
│  ├─ Task commands
│  ├─ Session commands
│  ├─ Workflow commands
│  └─ Config commands
└─ Fri: Phase 4 (Cleanup)
   ├─ Remove old code
   └─ Update documentation
```

---

## Dependencies

**Internal:**
- A2A Protocol Layer (must be running)
- AgentCore API Server (for integration tests)

**External:**
- Python 3.12+
- uv package manager
- pytest, mypy, ruff
- requests library
- Pydantic v2

**Infrastructure:**
- Development environment
- CI/CD pipeline (for automated tests)

---

## Deliverables

### Code
- [ ] Transport layer implementation
- [ ] Protocol layer implementation
- [ ] Service layer implementation
- [ ] DI container implementation
- [ ] All commands migrated
- [ ] Comprehensive test suite

### Documentation
- [ ] Updated specification (spec.md)
- [ ] Updated plan (this document)
- [ ] Migration guide
- [ ] Testing guide
- [ ] Developer guide

### Validation
- [ ] 95+ tests passing
- [ ] 90%+ code coverage
- [ ] JSON-RPC 2.0 compliance verified
- [ ] Performance benchmarks

---

## Appendix A: Ticket Summary

| ID | Title | Phase | SP | Days |
|----|-------|-------|----|----|
| CLI-R001 | Transport Layer | 1 | 5 | 1 |
| CLI-R002 | Protocol Layer | 1 | 7 | 1.5 |
| CLI-R003 | Service Layer | 1 | 4 | 0.5 |
| CLI-R004 | DI Container | 1 | 2 | 0.25 |
| CLI-R005 | Migrate Agent Register | 2 | 5 | 0.75 |
| CLI-R006 | Validation & Docs | 2 | 3 | 0.25 |
| CLI-R007 | Migrate Agent Commands | 3 | 3 | 0.75 |
| CLI-R008 | Migrate Task Commands | 3 | 3 | 0.75 |
| CLI-R009 | Migrate Session Commands | 3 | 2 | 0.5 |
| CLI-R010 | Migrate Workflow Commands | 3 | 2 | 0.5 |
| CLI-R011 | Migrate Config Commands | 3 | 2 | 0.5 |
| CLI-R012 | Remove Old Code | 4 | 2 | 0.5 |
| CLI-R013 | Update Documentation | 4 | 2 | 0.5 |
| **TOTAL** | | | **42** | **10** |

---

## Appendix B: Comparison with v1.0

| Aspect | v1.0 Plan | v2.0 Plan |
|--------|-----------|-----------|
| **Duration** | 4 weeks | 2 weeks |
| **Story Points** | 34 SP | 42 SP |
| **Phases** | 2 sprints | 4 phases |
| **Focus** | New implementation | Redesign + Migration |
| **Risk** | Medium | Low (phased) |
| **Architecture** | Monolithic | 4-layer |
| **Testing** | Integration only | Unit + Integration |

---

**End of Implementation Plan v2.0**
