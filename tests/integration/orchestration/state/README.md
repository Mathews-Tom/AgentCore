# Workflow State Persistence Integration Tests

Integration tests for the PostgreSQL state management system (ORCH-012).

## Test Modes

### SQLite (Default - Fast)

```bash
pytest tests/integration/orchestration/state/
```

- **Database**: SQLite in-memory
- **Speed**: Very fast (< 2 seconds)
- **Coverage**: Basic CRUD operations (6/12 tests)
- **Limitations**: No JSONB operators, no tag filtering

### PostgreSQL (Full Features)

```bash
USE_POSTGRES=1 pytest tests/integration/orchestration/state/
```

- **Database**: PostgreSQL 16 (testcontainer)
- **Speed**: Moderate (~10-15 seconds including container startup)
- **Coverage**: Full test suite (12/12 tests)
- **Features**: JSONB operators, GIN indexes, tag filtering

## Requirements

### SQLite Mode
- `aiosqlite` (already installed)

### PostgreSQL Mode
- `testcontainers[postgres]` (already installed)
- Docker running locally

## Test Status

| Test | SQLite | PostgreSQL |
|------|--------|------------|
| test_create_execution | ✅ Pass | ✅ Pass |
| test_get_execution | ✅ Pass | ✅ Pass |
| test_list_executions_with_filters | ❌ Skip (JSONB) | ✅ Pass |
| test_update_execution_status | ✅ Pass | ✅ Pass |
| test_update_execution_state | ❌ Skip (JSONB) | ✅ Pass |
| test_create_checkpoint | ❌ Skip (JSONB) | ✅ Pass |
| test_state_history | ✅ Pass | ✅ Pass |
| test_delete_execution | ❌ Skip (JSONB) | ✅ Pass |
| test_execution_statistics | ❌ Skip (JSONB) | ✅ Pass |
| test_create_version | ✅ Pass | ✅ Pass |
| test_get_latest_version | ✅ Pass | ✅ Pass |
| test_deprecate_version | ❌ Skip (JSONB) | ✅ Pass |

## CI/CD Usage

### Fast CI (Pull Requests)
```yaml
- name: Run fast integration tests
  run: pytest tests/integration/orchestration/state/
```

### Full CI (Main Branch)
```yaml
- name: Run full integration tests with PostgreSQL
  run: USE_POSTGRES=1 pytest tests/integration/orchestration/state/
```

## Troubleshooting

### Docker Not Running
```
Error: Cannot connect to Docker daemon
```
**Solution**: Start Docker Desktop or Docker daemon

### Container Startup Timeout
```
testcontainers.core.waiting_utils.TimeoutException
```
**Solution**: Increase timeout or check Docker resources

### Port Conflicts
```
Error: Port 5432 already in use
```
**Solution**: testcontainers auto-assigns random ports, this shouldn't happen

## Production Use

In production, the system uses PostgreSQL with full JSONB support:
- GIN indexes for efficient JSON queries
- JSONB containment operators for tag filtering
- Optimized state queries with JSONB path expressions

The SQLite tests validate core functionality, while PostgreSQL tests validate production features.
