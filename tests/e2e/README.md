# E2E Tests for AgentCore CLI

## Overview

This directory contains end-to-end (E2E) tests for the AgentCore CLI. These tests run the CLI as a subprocess against a real AgentCore API instance (not mocked), validating full workflows and command interactions.

## Requirements

- **AgentCore API running at http://localhost:8001**
- Docker Compose environment up and healthy:
  ```bash
  docker compose -f docker-compose.dev.yml up
  ```
- PostgreSQL and Redis services available

## Running E2E Tests

E2E tests are designed to run **on demand only** (not automatically in CI) due to their dependency on external services:

```bash
# Run all E2E tests
uv run pytest tests/e2e/test_cli_e2e.py -v

# Run specific test class
uv run pytest tests/e2e/test_cli_e2e.py::TestConfiguration -v

# Run with detailed output
uv run pytest tests/e2e/test_cli_e2e.py -v --tb=short
```

## Test Coverage

The E2E test suite covers:

1. **Agent Lifecycle** (4 tests - currently xfail)
   - Agent registration and listing
   - Agent info retrieval
   - Agent search by capability
   - Agent removal

2. **Task Lifecycle** (3 tests - currently xfail)
   - Task creation and status checking
   - Task listing
   - Task cancellation

3. **Session Management** (3 tests - currently xfail)
   - Session save and list
   - Session info retrieval
   - Session resume and delete

4. **Workflow Execution** (2 tests - currently xfail)
   - Workflow creation from YAML file
   - Workflow status monitoring

5. **Configuration Management** (4 tests - **PASSING**)
   - Config show command
   - Config init command
   - Config validate command
   - Environment variable override

6. **Output Formats** (2 tests - currently xfail)
   - JSON output format
   - Table output format

7. **Error Handling** (4 tests - **PASSING**)
   - Connection errors with wrong URL
   - Invalid agent ID errors
   - Missing required arguments
   - Invalid JSON input

## Test Status

**Current Results: 8 passed, 14 xfail**

### Passing Tests (8)
All configuration and error handling tests pass successfully:
- Configuration show, init, validate, env override
- Connection error handling
- Invalid input validation
- Missing argument detection

### Expected Failures (xfail) (14)
The following tests are marked as `xfail` due to **known CLI-API compatibility issues**:

**Root Cause**: The CLI's `agent.register` command sends parameters (`name`, `capabilities`, `cost_per_request`, `requirements`) that don't match the A2A protocol's `agent.register` JSON-RPC method signature, which expects an `agent_card` object.

**Affected Tests**:
- All Agent Lifecycle tests (4)
- All Task Lifecycle tests (3) - depend on agent registration
- All Session Management tests (3) - API compatibility not verified
- All Workflow tests (2) - API compatibility not verified
- All Output Format tests (2) - depend on agent registration

**Known Issue**:
```
✗ JSON-RPC Error -32603: Internal error
Details: {'details': 'Missing required parameter: agent_card'}
```

## Known Issues & Future Work

### CLI-API Compatibility (HIGH PRIORITY)

The CLI needs to be updated to match the A2A protocol's agent registration method:

**Current CLI behavior**:
```python
client.call("agent.register", {
    "name": name,
    "capabilities": cap_list,
    "cost_per_request": cost_per_request,
    "requirements": req_dict,
})
```

**Expected A2A protocol**:
```python
client.call("agent.register", {
    "agent_card": {
        "name": name,
        "capabilities": cap_list,
        "cost_per_request": cost_per_request,
        "requirements": req_dict,
        # ... other AgentCard fields
    }
})
```

**Resolution**: Update `src/agentcore_cli/commands/agent.py` to construct proper `agent_card` object according to A2A protocol specification.

### Session and Workflow API

Session and workflow tests are marked as xfail because their API compatibility has not been verified against the live AgentCore API. Once the agent registration issue is fixed, these tests should be validated and the xfail markers removed if they pass.

## Test Architecture

### Helper Functions

- `run_cli(*args, **kwargs)`: Execute CLI command and return result
- `run_cli_json(*args, **kwargs)`: Execute CLI with `--json` flag and parse output

### Fixtures

- `api_health_check`: Session-scoped health check for API availability
- `test_id`: Unique ID generator for test resource naming
- `cleanup_agents`: Tracks agent IDs for cleanup after tests
- `cleanup_tasks`: Tracks task IDs for cleanup after tests
- `cleanup_sessions`: Tracks session IDs for cleanup after tests

### Test Patterns

1. **Idempotent Tests**: All tests can be run multiple times without side effects
2. **Automatic Cleanup**: Fixtures ensure test resources are removed after tests
3. **Best-Effort Cleanup**: Cleanup continues even if some resources fail to delete
4. **Real API Testing**: No mocks - tests validate against actual running API
5. **Subprocess Execution**: CLI commands run as subprocess for realistic testing

## Maintenance

When updating E2E tests:

1. **Check API Health**: Ensure AgentCore API is running before test execution
2. **Update CLI Signatures**: If CLI command signatures change, update test calls
3. **Verify Cleanup**: Ensure cleanup fixtures properly remove test data
4. **Update xfail Markers**: Remove xfail when CLI-API compatibility is restored
5. **Document Breaking Changes**: Update README with any new known issues

## CI Integration (Future)

E2E tests are currently **excluded from CI** due to:
- Dependency on external Docker Compose environment
- Longer execution time (10-15 seconds)
- Known CLI-API compatibility issues

**Future CI Integration**:
Once CLI-API compatibility is restored, consider:
- Adding E2E tests to a separate CI job
- Using Docker Compose in CI for service orchestration
- Setting appropriate test timeouts
- Configuring proper API URL via environment variables

## Contributing

When adding new E2E tests:

1. Follow existing test patterns (setup → execute → assert → cleanup)
2. Use descriptive test names indicating what is being tested
3. Add proper cleanup fixtures for new resource types
4. Document any new known issues or xfail markers
5. Update this README with new test coverage information
