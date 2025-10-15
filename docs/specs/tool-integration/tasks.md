# Tasks: Tool Integration Framework

**From:** `spec.md` + `plan.md`
**Timeline:** 6 weeks, 3 sprints
**Team:** 2 Backend Engineers, 1 DevOps, 0.5 QA
**Created:** 2025-10-15

## Summary

- **Total tasks:** 31 (1 epic + 30 stories)
- **Estimated effort:** 124 story points
- **Critical path duration:** 5-7 weeks
- **Key risks:**
  1. Docker sandbox security for code execution (TOOL-011)
  2. Distributed rate limiting with Redis (TOOL-023)
  3. External API reliability and rate limits

## Phase Breakdown

### Phase 1: Foundation (Weeks 1-2, 24 story points)

**Goal:** Establish core framework infrastructure with interfaces, registry, and executor
**Deliverable:** Working tool framework with database schema and validation

#### Tasks

**[TOOL-001] Epic: Tool Integration Framework Implementation**

- **Description:** Implement standardized tool integration framework enabling agents to discover, access, and utilize diverse external tools (search, code execution, APIs, databases) through centralized registry, execution engine, and built-in adapters
- **Acceptance:**
  - [ ] All 30 story tasks completed (TOOL-002 through TOOL-031)
  - [ ] Performance targets met:
    - [ ] Framework overhead <100ms per tool call
    - [ ] Tool success rate >95%
    - [ ] Registry lookup <10ms for 1000+ tools
    - [ ] 1000 concurrent executions per instance
  - [ ] Quality gates passed:
    - [ ] 90%+ test coverage on tool framework
    - [ ] All P0 features have integration tests
    - [ ] Security audit passed (sandboxing, credentials)
  - [ ] Production-ready:
    - [ ] Load testing validates 1000 concurrent executions
    - [ ] Rate limiting prevents cost overruns
    - [ ] Monitoring dashboards deployed
- **Effort:** 124 story points (6 weeks)
- **Owner:** Engineering Team
- **Dependencies:** None (Epic)
- **Priority:** P0 (Critical)
- **Type:** Epic

---

**[TOOL-002] Tool Interface and Base Classes**

- **Description:** Create `Tool` abstract base class defining standardized interface for all tools, with async `execute()` method, parameter validation, and result formatting
- **Acceptance:**
  - [ ] `agentcore/tools/base.py` created with Tool ABC
  - [ ] `execute(parameters, context) -> ToolResult` method signature defined
  - [ ] `validate_parameters(parameters) -> tuple[bool, str]` implemented
  - [ ] ExecutionContext model created with user_id, agent_id, trace_id
  - [ ] Mypy strict mode validation passes
  - [ ] Unit tests for interface contract
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-001
- **Priority:** P0 (Blocker)
- **Files:** `src/agentcore/tools/base.py`

**[TOOL-003] Data Models**

- **Description:** Implement Pydantic models for ToolMetadata, ToolParameter, ToolResult with comprehensive validation, serialization, and database compatibility
- **Acceptance:**
  - [ ] ToolMetadata model with tool_id, name, description, parameters, authentication, rate_limit
  - [ ] ToolParameter model with name, type, description, required, default, enum
  - [ ] ToolResult model with success, result, error, execution_time_ms, metadata
  - [ ] All models have comprehensive validation rules
  - [ ] All models have unit tests with edge cases
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-002
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/tools/models.py`

**[TOOL-004] Tool Registry**

- **Description:** Implement ToolRegistry class with in-memory storage, category indexing, search capabilities, and tool discovery methods
- **Acceptance:**
  - [ ] ToolRegistry class with `_tools` dict and `_categories` index
  - [ ] `register(tool: Tool) -> None` method validates and stores tools
  - [ ] `get(tool_id: str) -> Tool | None` method with <10ms lookup
  - [ ] `search(query: str, category: str) -> list[Tool]` with fuzzy matching
  - [ ] `list_by_category(category: str) -> list[Tool]` method
  - [ ] Unit tests for all registry operations
  - [ ] Performance test validates <10ms lookup for 1000+ tools
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-002
- **Priority:** P0 (Blocker)
- **Files:** `src/agentcore/tools/registry.py`

**[TOOL-005] Database Schema**

- **Description:** Create Alembic migration for `tool_executions` table with proper indexes, foreign keys, and JSONB columns for flexible data storage
- **Acceptance:**
  - [ ] Migration creates tool_executions table with columns: execution_id, tool_id, user_id, agent_id, parameters (JSONB), result (JSONB), success, error, execution_time_ms, trace_id, created_at
  - [ ] B-tree indexes on tool_id, user_id, created_at for fast lookups
  - [ ] Composite index on (tool_id, user_id) for user-specific queries
  - [ ] Partial index on (success = false) for error analysis
  - [ ] Migration applies and rolls back successfully
  - [ ] Test migration with sample data
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-003
- **Priority:** P0 (Critical)
- **Files:** `alembic/versions/XXX_add_tool_executions.py`

**[TOOL-006] Tool Executor**

- **Description:** Implement ToolExecutor class managing tool invocation lifecycle with authentication, error handling, logging, and result formatting
- **Acceptance:**
  - [ ] ToolExecutor class with `execute_tool(tool_id, parameters, context)` method
  - [ ] Integration with ToolRegistry for tool lookup
  - [ ] Basic authentication handling via ExecutionContext
  - [ ] Error handling and categorization (auth, validation, timeout, execution)
  - [ ] Tool execution logging to database (tool_executions table)
  - [ ] Trace ID propagation for distributed tracing
  - [ ] Unit tests with mocked tools and database
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-004, TOOL-005
- **Priority:** P0 (Blocker)
- **Files:** `src/agentcore/tools/executor.py`

**[TOOL-007] Parameter Validation Framework**

- **Description:** Implement comprehensive parameter validation using Pydantic schemas with type checking, required field validation, enum validation, and error message formatting
- **Acceptance:**
  - [ ] Pydantic schema validation for all parameter types (string, integer, boolean, array, object)
  - [ ] Required field validation with clear error messages
  - [ ] Type checking with proper type mapping
  - [ ] Enum validation for restricted values
  - [ ] Custom validators for complex rules (e.g., URL format, length limits)
  - [ ] Unit tests for all validation scenarios and edge cases
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-002, TOOL-003
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/tools/validation.py`

**[TOOL-008] Initial Documentation**

- **Description:** Create comprehensive API documentation for tool interfaces, architecture diagrams, and developer guide for adding custom tools
- **Acceptance:**
  - [ ] API documentation for Tool interface, ToolRegistry, ToolExecutor
  - [ ] Architecture diagram showing layered architecture
  - [ ] Developer guide with step-by-step instructions for adding new tools
  - [ ] Code examples for common tool types (search, API, code execution)
  - [ ] Documentation published in `docs/tools/` directory
- **Effort:** 2 story points (1-2 days)
- **Owner:** Tech Lead
- **Dependencies:** TOOL-002, TOOL-004, TOOL-006
- **Priority:** P1 (High)
- **Files:** `docs/tools/README.md`, `docs/tools/architecture.md`, `docs/tools/developer-guide.md`

---

### Phase 2: Built-in Tools (Week 3, 31 story points)

**Goal:** Implement essential tool adapters for search, code execution, and API integration
**Deliverable:** Working implementations of 5 built-in tools with integration tests

#### Tasks

**[TOOL-009] Google Search Tool**

- **Description:** Implement GoogleSearchTool adapter integrating with Google Custom Search API, with result parsing, formatting, and error handling
- **Acceptance:**
  - [ ] GoogleSearchTool class implementing Tool interface
  - [ ] Google Custom Search API integration using httpx
  - [ ] Parameters: query (required), num_results (optional, default 10)
  - [ ] Result parsing and formatting (title, url, snippet)
  - [ ] Authentication via API key from environment/Vault
  - [ ] Rate limiting metadata (100 calls/minute)
  - [ ] Unit tests with mocked API responses
  - [ ] Integration test with real API (staging)
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-002, TOOL-003
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/tools/adapters/search.py`

**[TOOL-010] Wikipedia Search Tool**

- **Description:** Implement WikipediaSearchTool adapter using Wikipedia API for encyclopedia lookups with article summary extraction
- **Acceptance:**
  - [ ] WikipediaSearchTool class implementing Tool interface
  - [ ] Wikipedia API integration (no auth required)
  - [ ] Parameters: query (required), sentences (optional, default 5)
  - [ ] Article summary extraction and formatting
  - [ ] Disambiguation handling (multiple matches)
  - [ ] Unit tests with mocked API responses
  - [ ] Integration test with real Wikipedia API
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-002, TOOL-003
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/tools/adapters/search.py`

**[TOOL-011] Python Execution Tool**

- **Description:** Implement PythonExecutionTool with Docker sandbox for secure code execution, including resource limits, no network access, and result capture
- **Acceptance:**
  - [ ] PythonExecutionTool class implementing Tool interface
  - [ ] Docker sandbox configuration (no network, read-only filesystem except /tmp, 1 CPU, 512MB RAM)
  - [ ] AppArmor/SELinux security profiles
  - [ ] Code execution with timeout enforcement (default: 30s)
  - [ ] Result capture (stdout, stderr, return value)
  - [ ] Error handling for container crashes, timeouts
  - [ ] Docker image build with minimal dependencies
  - [ ] Unit tests with mocked Docker API
  - [ ] Integration tests with real Docker containers
  - [ ] Security penetration testing (attempt sandbox escape)
- **Effort:** 8 story points (5-8 days)
- **Owner:** Backend Engineer + DevOps
- **Dependencies:** TOOL-002, TOOL-003
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/tools/adapters/code.py`, `docker/python-sandbox/Dockerfile`

**[TOOL-012] REST API Tool**

- **Description:** Implement RESTAPITool adapter providing generic HTTP client for external API calls with support for GET, POST, PUT, DELETE and multiple auth methods
- **Acceptance:**
  - [ ] RESTAPITool class implementing Tool interface
  - [ ] HTTP client using httpx with async support
  - [ ] Support for GET, POST, PUT, DELETE methods
  - [ ] Parameters: url, method, headers, body, auth_type
  - [ ] Authentication support: none, api_key, bearer_token, oauth
  - [ ] Response parsing (JSON, text, binary)
  - [ ] Error handling for network errors, timeouts, HTTP errors
  - [ ] Unit tests with mocked HTTP responses
  - [ ] Integration tests with test API endpoints
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-002, TOOL-003
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/tools/adapters/api.py`

**[TOOL-013] File Operations Tool**

- **Description:** Implement FileOperationsTool for read, write, list operations with security restrictions (path validation, size limits)
- **Acceptance:**
  - [ ] FileOperationsTool class implementing Tool interface
  - [ ] Operations: read, write, list_directory
  - [ ] Path validation (prevent directory traversal)
  - [ ] Size limits (max file size: 10MB)
  - [ ] Whitelist of allowed directories
  - [ ] Error handling for permission denied, file not found
  - [ ] Unit tests with temporary test files
  - [ ] Security tests (directory traversal attempts)
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-002, TOOL-003
- **Priority:** P1 (High)
- **Files:** `src/agentcore/tools/adapters/files.py`

**[TOOL-014] Tool Registration on Startup**

- **Description:** Create `register_builtin_tools()` function to auto-register all built-in tools on application startup with configuration via environment variables
- **Acceptance:**
  - [ ] `register_builtin_tools(registry)` function in builtin.py
  - [ ] Auto-registration of all 5 built-in tools
  - [ ] Environment variable configuration (GOOGLE_API_KEY, ENABLE_CODE_EXECUTION, etc.)
  - [ ] Conditional registration based on config (e.g., skip Google if no API key)
  - [ ] Integration with FastAPI startup event
  - [ ] Logging of registered tools on startup
  - [ ] Unit tests for registration logic
- **Effort:** 2 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-009, TOOL-010, TOOL-011, TOOL-012, TOOL-013
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/tools/builtin.py`

**[TOOL-015] Integration Tests for Built-in Tools**

- **Description:** Create comprehensive integration test suite for each built-in tool with real external services (staging/test environments)
- **Acceptance:**
  - [ ] Integration tests for GoogleSearchTool with real API
  - [ ] Integration tests for WikipediaSearchTool with real API
  - [ ] Integration tests for PythonExecutionTool with real Docker
  - [ ] Integration tests for RESTAPITool with test endpoints
  - [ ] Integration tests for FileOperationsTool with temp files
  - [ ] Tests validate error handling scenarios
  - [ ] Tests use testcontainers for Docker/PostgreSQL
  - [ ] Test coverage ≥85% for adapters
- **Effort:** 5 story points (3-5 days)
- **Owner:** QA Engineer
- **Dependencies:** TOOL-014
- **Priority:** P0 (Critical)
- **Files:** `tests/integration/tools/test_google_search.py`, `tests/integration/tools/test_wikipedia.py`, `tests/integration/tools/test_python_executor.py`, `tests/integration/tools/test_rest_api.py`, `tests/integration/tools/test_file_operations.py`

---

### Phase 3: JSON-RPC Integration (Week 4, 24 story points)

**Goal:** Expose tools via A2A protocol with authentication, tracing, and error handling
**Deliverable:** Fully functional JSON-RPC endpoints for tool discovery and execution

#### Tasks

**[TOOL-016] tools.list JSON-RPC Method**

- **Description:** Register `tools.list` JSON-RPC method with `@register_jsonrpc_method` decorator for tool discovery with category filtering
- **Acceptance:**
  - [ ] `tools.list` method registered with JSON-RPC handler
  - [ ] Optional category parameter for filtering
  - [ ] Returns tool metadata (tool_id, name, description, parameters, authentication, rate_limit)
  - [ ] Integration with ToolRegistry
  - [ ] Error handling (invalid category)
  - [ ] Unit tests for method logic
  - [ ] Integration test via HTTP POST to /api/v1/jsonrpc
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-004, TOOL-014
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/tools/jsonrpc.py`

**[TOOL-017] tools.execute JSON-RPC Method**

- **Description:** Register `tools.execute` JSON-RPC method for tool invocation with parameter validation, authentication, and result formatting
- **Acceptance:**
  - [ ] `tools.execute` method registered with JSON-RPC handler
  - [ ] Parameters: tool_id, parameters, context (optional)
  - [ ] Integration with ToolExecutor
  - [ ] Parameter validation before execution
  - [ ] A2A context extraction (trace_id, source_agent)
  - [ ] Error handling with proper JSON-RPC error codes (400, 404, 429, 408, 500)
  - [ ] Returns ToolResult with success, result, error, execution_time_ms
  - [ ] Unit tests with mocked ToolExecutor
  - [ ] Integration tests with real tools
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-006, TOOL-014
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/tools/jsonrpc.py`

**[TOOL-018] tools.search JSON-RPC Method**

- **Description:** Register `tools.search` JSON-RPC method as convenience wrapper around tools.list with query-based search
- **Acceptance:**
  - [ ] `tools.search` method registered with JSON-RPC handler
  - [ ] Parameters: query, category (optional), limit (optional)
  - [ ] Delegates to ToolRegistry.search()
  - [ ] Returns matching tools sorted by relevance
  - [ ] Unit tests for search logic
  - [ ] Integration test via JSON-RPC
- **Effort:** 2 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-016
- **Priority:** P1 (High)
- **Files:** `src/agentcore/tools/jsonrpc.py`

**[TOOL-019] A2A Authentication Integration**

- **Description:** Integrate with A2A authentication to extract user_id/agent_id from JWT and pass to ExecutionContext for RBAC enforcement
- **Acceptance:**
  - [ ] Extract user_id and agent_id from JWT claims
  - [ ] Pass auth information to ExecutionContext
  - [ ] RBAC policy enforcement for tool access (basic implementation)
  - [ ] Authentication errors return 401 Unauthorized
  - [ ] Unit tests for auth extraction and validation
  - [ ] Integration tests with valid/invalid JWTs
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-017
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/tools/auth.py`

**[TOOL-020] Distributed Tracing Support**

- **Description:** Integrate OpenTelemetry for distributed tracing with span creation, trace ID propagation, and tool execution linking
- **Acceptance:**
  - [ ] OpenTelemetry SDK integrated
  - [ ] Spans created for each tool execution
  - [ ] Trace ID propagated via A2A context
  - [ ] Tool executions linked to parent trace (agent request)
  - [ ] Span attributes include tool_id, user_id, success, execution_time_ms
  - [ ] Trace export to OpenTelemetry collector
  - [ ] Integration tests validate trace propagation
- **Effort:** 5 story points (3-5 days)
- **Owner:** DevOps Engineer
- **Dependencies:** TOOL-017
- **Priority:** P1 (High)
- **Files:** `src/agentcore/tools/tracing.py`

**[TOOL-021] Error Categorization**

- **Description:** Define error types (auth, validation, timeout, execution) and map tool errors to JSON-RPC error codes with structured error responses
- **Acceptance:**
  - [ ] Error type enum: AuthError, ValidationError, TimeoutError, ExecutionError, RateLimitError
  - [ ] Mapping from tool errors to JSON-RPC error codes (401, 400, 408, 500, 429)
  - [ ] Structured error responses with error type, message, details
  - [ ] Error handling in ToolExecutor
  - [ ] Unit tests for error categorization
  - [ ] Integration tests for each error type
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-017
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/tools/errors.py`

**[TOOL-022] API Documentation**

- **Description:** Create comprehensive API documentation for all JSON-RPC methods with request/response examples and tool developer guide
- **Acceptance:**
  - [ ] API documentation for tools.list, tools.execute, tools.search
  - [ ] Request/response schema examples
  - [ ] Error response examples for each error code
  - [ ] Tool developer guide with step-by-step instructions
  - [ ] Code examples for adding custom tools
  - [ ] Published in docs/tools/api.md
- **Effort:** 3 story points (2-3 days)
- **Owner:** Tech Lead
- **Dependencies:** TOOL-016, TOOL-017, TOOL-018
- **Priority:** P1 (High)
- **Files:** `docs/tools/api.md`

---

### Phase 4: Advanced Features (Weeks 5-6, 45 story points)

**Goal:** Production hardening with rate limiting, retry, monitoring, and security
**Deliverable:** Production-ready tool framework with monitoring, load testing, and security audit

#### Tasks

**[TOOL-023] Rate Limiting with Redis**

- **Description:** Implement RateLimiter class using Redis with token bucket algorithm for per-tool, per-user rate limits with fail-closed strategy
- **Acceptance:**
  - [ ] RateLimiter class with `check_and_consume(tool_id, user_id, tokens=1)` method
  - [ ] Token bucket algorithm using Redis INCR + EXPIRE (atomic operations)
  - [ ] Per-tool, per-user rate limits configurable in ToolMetadata
  - [ ] Fail-closed strategy: Return 503 if Redis unavailable
  - [ ] Rate limit exceeded returns 429 with retry-after header
  - [ ] Redis key format: `rate_limit:{tool_id}:{user_id}`
  - [ ] TTL of 60 seconds for sliding window
  - [ ] Unit tests with mocked Redis
  - [ ] Integration tests with real Redis via testcontainers
  - [ ] Concurrency tests validate atomic operations
- **Effort:** 8 story points (5-8 days)
- **Owner:** Backend Engineer + DevOps
- **Dependencies:** TOOL-017
- **Priority:** P0 (Critical)
- **Files:** `src/agentcore/tools/rate_limiter.py`

**[TOOL-024] Automatic Retry with Exponential Backoff**

- **Description:** Implement retry logic in ToolExecutor for retryable errors (network, 503, timeout) with exponential backoff (1s, 2s, 4s)
- **Acceptance:**
  - [ ] Detect retryable errors (network errors, 503 responses, timeouts)
  - [ ] Non-retryable errors fail immediately (401, 400, 404)
  - [ ] Exponential backoff: 1s, 2s, 4s, 8s (max 4 retries)
  - [ ] Max retry attempts configurable per tool (default: 3)
  - [ ] Retry attempts logged with trace_id
  - [ ] Success after retry logged separately from first attempt
  - [ ] Unit tests for retry scenarios
  - [ ] Integration tests with simulated failures
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-017, TOOL-021
- **Priority:** P1 (High)
- **Files:** `src/agentcore/tools/retry.py`

**[TOOL-025] Quota Management**

- **Description:** Implement quota tracking for daily/monthly limits with `tools.get_rate_limit_status` endpoint for quota checking
- **Acceptance:**
  - [ ] Daily and monthly quota tracking in Redis
  - [ ] Quota configuration per tool in ToolMetadata
  - [ ] Quota exceeded returns 429 with quota reset time
  - [ ] `tools.get_rate_limit_status` JSON-RPC method
  - [ ] Returns limit, remaining, reset_at for tool and user
  - [ ] Unit tests for quota logic
  - [ ] Integration tests for quota enforcement
- **Effort:** 3 story points (2-3 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-023
- **Priority:** P1 (High)
- **Files:** `src/agentcore/tools/quota.py`

**[TOOL-026] Prometheus Metrics**

- **Description:** Add Prometheus metrics for tool execution monitoring: latency histograms, success/failure counters, rate limit counters
- **Acceptance:**
  - [ ] `tool_execution_duration_seconds` histogram (p50, p95, p99) by tool_id, success
  - [ ] `tool_execution_total` counter by tool_id, success
  - [ ] `rate_limit_exceeded_total` counter by tool_id, user_id
  - [ ] `framework_overhead_seconds` histogram
  - [ ] `tool_registry_size` gauge
  - [ ] Metrics integrated with existing Prometheus endpoint
  - [ ] Unit tests for metric recording
- **Effort:** 5 story points (3-5 days)
- **Owner:** DevOps Engineer
- **Dependencies:** TOOL-017, TOOL-023
- **Priority:** P1 (High)
- **Files:** `src/agentcore/tools/metrics.py`

**[TOOL-027] Grafana Dashboards**

- **Description:** Create Grafana dashboards for tool usage, performance, and cost monitoring with alerts for SLO violations
- **Acceptance:**
  - [ ] Tool Usage Dashboard: executions per tool, success rate, top users
  - [ ] Performance Dashboard: latency heatmap, throughput, framework overhead
  - [ ] Cost Dashboard: API calls per tool, rate limit status, quota usage
  - [ ] Error Dashboard: error rates by type, failed tool executions
  - [ ] Alerts configured for: error rate >10%, timeout rate >20%, auth failures >5
  - [ ] Dashboards exported to k8s/monitoring/tool-dashboards.yaml
- **Effort:** 3 story points (2-3 days)
- **Owner:** DevOps Engineer
- **Dependencies:** TOOL-026
- **Priority:** P1 (High)
- **Files:** `k8s/monitoring/tool-dashboards.yaml`

**[TOOL-028] Load Testing**

- **Description:** Conduct load testing using Locust to validate 1000 concurrent tool executions with mix of tool types (search, code, API)
- **Acceptance:**
  - [ ] Locust test script with 1000 concurrent users
  - [ ] Mix of tool types: 50% search, 30% API, 20% code execution
  - [ ] Sustained load for 1 hour
  - [ ] Success rate >95% maintained under load
  - [ ] p95 latency <500ms (excluding tool execution time)
  - [ ] No resource exhaustion (CPU, memory, connections)
  - [ ] Load test report with metrics and graphs
  - [ ] Identified bottlenecks and optimization recommendations
- **Effort:** 5 story points (3-5 days)
- **Owner:** QA Engineer
- **Dependencies:** TOOL-023, TOOL-024
- **Priority:** P1 (High)
- **Files:** `tests/load/test_tool_concurrency.py`

**[TOOL-029] Security Audit**

- **Description:** Conduct comprehensive security audit covering Docker sandboxing, credential management, RBAC enforcement, and parameter injection
- **Acceptance:**
  - [ ] Docker sandbox penetration testing (escape attempts, privilege escalation)
  - [ ] Credential leak detection in logs and database
  - [ ] RBAC policy validation (unauthorized tool access attempts)
  - [ ] Parameter injection testing (SQL, shell, XSS)
  - [ ] Secret scanning in codebase and Docker images
  - [ ] Security audit report with findings severity (Critical, High, Medium, Low)
  - [ ] All Critical and High findings remediated
  - [ ] Security best practices documented
- **Effort:** 8 story points (5-8 days)
- **Owner:** Security Engineer + Backend
- **Dependencies:** TOOL-011, TOOL-019, TOOL-023
- **Priority:** P0 (Critical)
- **Files:** `docs/tools/security-audit.md`

**[TOOL-030] Performance Optimization**

- **Description:** Profile and optimize framework overhead to meet <100ms target, focusing on registry lookup, parameter validation, and database logging
- **Acceptance:**
  - [ ] Profiling identifies bottlenecks in hot paths
  - [ ] Registry lookup optimized (<10ms for 1000+ tools)
  - [ ] Parameter validation optimized (Pydantic performance tuning)
  - [ ] Database connection pooling tuned (5-20 connections)
  - [ ] Async operations parallelized where possible
  - [ ] Framework overhead <100ms (p95) validated via benchmarks
  - [ ] Performance report with before/after metrics
- **Effort:** 5 story points (3-5 days)
- **Owner:** Backend Engineer
- **Dependencies:** TOOL-023, TOOL-028
- **Priority:** P1 (High)
- **Files:** `src/agentcore/tools/` (various optimization)

**[TOOL-031] Production Runbook**

- **Description:** Create comprehensive production runbook with deployment guide, monitoring setup, incident response procedures, and troubleshooting guide
- **Acceptance:**
  - [ ] Deployment guide (Docker Compose, Kubernetes)
  - [ ] Environment variable configuration reference
  - [ ] Monitoring and alerting setup instructions
  - [ ] Incident response procedures for common scenarios (Redis down, tool failures, rate limit issues)
  - [ ] Troubleshooting guide with symptoms and solutions
  - [ ] Performance tuning recommendations
  - [ ] Backup and recovery procedures
  - [ ] Published in docs/tools/runbook.md
- **Effort:** 3 story points (2-3 days)
- **Owner:** Tech Lead + DevOps
- **Dependencies:** TOOL-026, TOOL-027, TOOL-029
- **Priority:** P1 (High)
- **Files:** `docs/tools/runbook.md`

---

## Critical Path

```plaintext
TOOL-002 → TOOL-003 → TOOL-005 → TOOL-006 → TOOL-017 → TOOL-023 → TOOL-030
  (3d)      (3d)       (3d)       (5d)       (5d)       (8d)       (5d)
                            [32 days / 35 story points]
```

**Bottlenecks:**

- **TOOL-011 (Python Execution Tool):** Docker sandboxing complexity, security requirements (8 SP)
- **TOOL-023 (Rate Limiting with Redis):** Distributed state management, atomic operations (8 SP)
- **TOOL-029 (Security Audit):** Comprehensive testing, remediation required (8 SP)

**Parallel Tracks:**

- **Tool Adapters:** TOOL-009 through TOOL-013 can be developed in parallel after TOOL-002/003
- **JSON-RPC Methods:** TOOL-016, TOOL-018 can proceed in parallel with TOOL-017
- **Monitoring:** TOOL-020, TOOL-026, TOOL-027 can be done by DevOps in parallel with backend work
- **Testing:** TOOL-015, TOOL-028 can run alongside development

---

## Quick Wins (Week 1-2)

1. **[TOOL-002] Tool Interface and Base Classes** - Unblocks all tool development
2. **[TOOL-003] Data Models** - Demonstrates progress, enables validation
3. **[TOOL-004] Tool Registry** - Enables tool discovery immediately

---

## Risk Mitigation

| Task | Risk | Mitigation | Contingency |
|------|------|------------|-------------|
| TOOL-011 | Docker sandbox escape vulnerability | Security profiles (AppArmor, seccomp), penetration testing, read-only filesystem | Disable code execution tools, use alternative sandbox (Pyodide) |
| TOOL-023 | Redis unavailability breaks rate limiting | Redis clustering with failover, fail-closed strategy, health checks | Manual rate limiting via config, temporary disable high-volume tools |
| TOOL-009/010 | External API rate limits exceeded | Client-side rate limiting, caching, multiple API keys | Fallback to alternative APIs, cached results, notify users |
| TOOL-029 | Critical security findings block launch | Early security review (week 4), incremental fixes | Delay launch, disable affected tools, implement workarounds |
| TOOL-028 | Performance targets not met under load | Profile early (week 4), optimize hot paths, allocate week 7 buffer | Relax latency target to <200ms with user approval |

---

## Testing Strategy

### Automated Testing Tasks

- **[TOOL-008] Unit Test Framework** - Integrated into Phase 1 tasks
- **[TOOL-015] Integration Tests** (5 SP) - Week 3
- **[TOOL-028] Load Testing** (5 SP) - Week 5
- **[TOOL-029] Security Audit** (8 SP) - Week 5-6

### Quality Gates

- **90% code coverage** required for tool framework
- **95% tool success rate** under normal load
- **All P0 features** have integration tests
- **Security audit passed** before production deployment
- **Load testing** validates 1000 concurrent executions

### Testing Coverage

| Component | Unit Tests | Integration Tests | Load Tests | Security Tests |
|-----------|------------|-------------------|------------|----------------|
| Framework (base, registry, executor) | ✓ TOOL-002-008 | ✓ TOOL-015 | ✓ TOOL-028 | ✓ TOOL-029 |
| Tool Adapters (search, code, API) | ✓ (included) | ✓ TOOL-015 | ✓ TOOL-028 | ✓ TOOL-029 |
| JSON-RPC Methods | ✓ TOOL-016-018 | ✓ TOOL-015 | ✓ TOOL-028 | ✓ TOOL-029 |
| Rate Limiting | ✓ TOOL-023 | ✓ TOOL-023 | ✓ TOOL-028 | ✓ TOOL-029 |
| Retry Logic | ✓ TOOL-024 | ✓ TOOL-024 | ✓ TOOL-028 | - |

---

## Team Allocation

**Backend Engineers (2 FTE)**

- Core framework (TOOL-002 to TOOL-008)
- Tool adapters (TOOL-009 to TOOL-013)
- JSON-RPC integration (TOOL-016 to TOOL-019, TOOL-021)
- Rate limiting and retry (TOOL-023 to TOOL-025)
- Performance optimization (TOOL-030)

**DevOps Engineer (1 FTE)**

- Docker sandboxing (TOOL-011 support)
- Distributed tracing (TOOL-020)
- Monitoring and dashboards (TOOL-026, TOOL-027)
- Production runbook (TOOL-031)

**QA Engineer (0.5 FTE)**

- Integration testing (TOOL-015)
- Load testing (TOOL-028)
- Security testing support (TOOL-029)

**Security Engineer (consulting)**

- Security audit (TOOL-029)
- Docker security review (TOOL-011)

**Tech Lead (coordination)**

- Documentation (TOOL-008, TOOL-022, TOOL-031)
- Architecture decisions
- Code reviews
- Risk management

---

## Sprint Planning

**2-week sprints, ~40 SP velocity (2 backend engineers)**

| Sprint | Focus | Story Points | Key Deliverables |
|--------|-------|--------------|------------------|
| **Sprint 1** (Weeks 1-2) | Foundation | 24 SP | Tool interface, registry, executor, database schema |
| **Sprint 2** (Week 3) | Built-in Tools | 31 SP | All 5 tool adapters, integration tests |
| **Sprint 3** (Week 4) | JSON-RPC Integration | 24 SP | API endpoints, auth, tracing, error handling |
| **Sprint 4** (Weeks 5-6) | Advanced Features Part 1 | 24 SP | Rate limiting, retry, quotas, metrics |
| **Sprint 5** (Week 6) | Advanced Features Part 2 | 21 SP | Dashboards, load testing, security audit, optimization, runbook |

**Total:** 124 SP across 5 sprints (6 weeks)

---

## Task Import Format

CSV export for project management tools:

```csv
ID,Title,Description,Estimate,Priority,Assignee,Dependencies,Sprint,Type
TOOL-001,Epic: Tool Integration Framework,Implement standardized tool integration,124,P0,Team,,All,Epic
TOOL-002,Tool Interface and Base Classes,Create Tool ABC,3,P0,Backend,TOOL-001,1,Story
TOOL-003,Data Models,Pydantic models,3,P0,Backend,TOOL-002,1,Story
TOOL-004,Tool Registry,Centralized registry,5,P0,Backend,TOOL-002,1,Story
TOOL-005,Database Schema,tool_executions table,3,P0,Backend,TOOL-003,1,Story
TOOL-006,Tool Executor,Invocation lifecycle,5,P0,Backend,"TOOL-004,TOOL-005",1,Story
TOOL-007,Parameter Validation,Pydantic validation,3,P0,Backend,"TOOL-002,TOOL-003",1,Story
TOOL-008,Initial Documentation,API docs,2,P1,Tech Lead,"TOOL-002,TOOL-004,TOOL-006",1,Story
TOOL-009,Google Search Tool,Google API integration,5,P0,Backend,"TOOL-002,TOOL-003",2,Story
TOOL-010,Wikipedia Search Tool,Wikipedia API,3,P0,Backend,"TOOL-002,TOOL-003",2,Story
TOOL-011,Python Execution Tool,Docker sandbox,8,P0,Backend+DevOps,"TOOL-002,TOOL-003",2,Story
TOOL-012,REST API Tool,HTTP client,5,P0,Backend,"TOOL-002,TOOL-003",2,Story
TOOL-013,File Operations Tool,File read/write,3,P1,Backend,"TOOL-002,TOOL-003",2,Story
TOOL-014,Tool Registration,Auto-register on startup,2,P0,Backend,"TOOL-009,TOOL-010,TOOL-011,TOOL-012,TOOL-013",2,Story
TOOL-015,Integration Tests,End-to-end tests,5,P0,QA,TOOL-014,2,Story
TOOL-016,tools.list JSON-RPC,Tool discovery API,3,P0,Backend,"TOOL-004,TOOL-014",3,Story
TOOL-017,tools.execute JSON-RPC,Tool execution API,5,P0,Backend,"TOOL-006,TOOL-014",3,Story
TOOL-018,tools.search JSON-RPC,Search convenience method,2,P1,Backend,TOOL-016,3,Story
TOOL-019,A2A Authentication,JWT integration,3,P0,Backend,TOOL-017,3,Story
TOOL-020,Distributed Tracing,OpenTelemetry,5,P1,DevOps,TOOL-017,3,Story
TOOL-021,Error Categorization,Error types and codes,3,P0,Backend,TOOL-017,3,Story
TOOL-022,API Documentation,JSON-RPC docs,3,P1,Tech Lead,"TOOL-016,TOOL-017,TOOL-018",3,Story
TOOL-023,Rate Limiting,Redis token bucket,8,P0,Backend+DevOps,TOOL-017,4,Story
TOOL-024,Automatic Retry,Exponential backoff,5,P1,Backend,"TOOL-017,TOOL-021",4,Story
TOOL-025,Quota Management,Daily/monthly limits,3,P1,Backend,TOOL-023,4,Story
TOOL-026,Prometheus Metrics,Monitoring metrics,5,P1,DevOps,"TOOL-017,TOOL-023",4,Story
TOOL-027,Grafana Dashboards,Visualization,3,P1,DevOps,TOOL-026,5,Story
TOOL-028,Load Testing,1000 concurrent users,5,P1,QA,"TOOL-023,TOOL-024",5,Story
TOOL-029,Security Audit,Comprehensive audit,8,P0,Security,"TOOL-011,TOOL-019,TOOL-023",5,Story
TOOL-030,Performance Optimization,Meet <100ms target,5,P1,Backend,"TOOL-023,TOOL-028",5,Story
TOOL-031,Production Runbook,Operational guide,3,P1,Tech Lead+DevOps,"TOOL-026,TOOL-027,TOOL-029",5,Story
```

---

## Appendix

**Estimation Method:** Planning Poker with team, Fibonacci scale
**Story Point Scale:** 1 (trivial), 2 (simple), 3 (moderate), 5 (complex), 8 (very complex), 13 (epic-level)
**Velocity Assumption:** 20 SP per backend engineer per 2-week sprint (conservative)

**Definition of Done:**

- [ ] Code implemented and reviewed by 2+ engineers
- [ ] Unit tests written and passing (90%+ coverage)
- [ ] Integration tests passing (where applicable)
- [ ] Documentation updated (docstrings, API docs)
- [ ] Deployed to staging environment
- [ ] Performance validated (meets targets)
- [ ] Security reviewed (for P0 tasks)

**Dependencies:**

- **Docker runtime** for TOOL-011 (code execution sandboxing)
- **Redis** for TOOL-023 (rate limiting state)
- **Secret management service** (Vault/AWS Secrets Manager) for credentials
- **External APIs:** Google Custom Search, Wikipedia API
- **MOD-001 (Modular Agent Core):** Executor module will consume this framework (downstream dependency)

**Technology Stack:**

- Python 3.12+ with asyncio
- FastAPI + JSON-RPC 2.0
- PostgreSQL with asyncpg
- Redis 7+
- Docker 24+
- Pydantic v2
- OpenTelemetry
- Prometheus + Grafana
- pytest-asyncio

**References:**

- Spec: `docs/specs/tool-integration/spec.md`
- Plan: `docs/specs/tool-integration/plan.md`
- Research: `docs/research/multi-tool-integration.md`
