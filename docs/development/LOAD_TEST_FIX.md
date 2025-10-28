# Load Test Workflow Fix

## Problem

The `integration-load-test.yml` workflow was failing with **100% request failure rate**:

```
Total Requests:        74,941
Successful Requests:   0
Failed Requests:       74,941
Success Rate:          0.00%
```

## Root Cause Analysis

The load test (`integration_layer_load_test.py`) was testing endpoints that **don't exist** in the current AgentCore implementation:

### Tested Endpoints (Non-Existent)
❌ `/api/v1/integration/llm/complete` - LLM provider routing
❌ `/api/v1/integration/webhooks` - Webhook management
❌ `/api/v1/integration/events/publish` - Event publishing
❌ `/api/v1/integration/storage/upload` - Storage operations

### Actual Available Endpoints
✅ `/api/v1/jsonrpc` - JSON-RPC 2.0 endpoint (A2A Protocol)
✅ `/health` - Health check
✅ `/.well-known/agent.json` - A2A discovery
✅ `/api/v1/ws` - WebSocket endpoint

## Solution

### 1. Fixed Integration Workflow
**File:** `.github/workflows/integration-load-test.yml`

Added conditional execution:
- **Check Job**: Detects if `src/agentcore/integration/` exists
- **Skip Logic**: Only runs if integration layer is implemented
- **Disabled Schedule**: Commented out nightly runs until layer exists

```yaml
jobs:
  check-integration-layer:
    name: Check if Integration Layer exists
    steps:
      - name: Check for integration layer
        run: |
          if [ -d "src/agentcore/integration" ]; then
            echo "exists=true"
          else
            echo "exists=false" - skipping tests
          fi

  load-test:
    needs: check-integration-layer
    if: needs.check-integration-layer.outputs.exists == 'true'
    # ... rest of job
```

### 2. Created New A2A Protocol Workflow
**File:** `.github/workflows/a2a-load-test.yml`

Tests **actual implemented features**:
- Uses `locustfile.py` (A2A protocol tests)
- Tests real JSON-RPC endpoints
- Appropriate targets (100 users, 1m duration)
- Runs on A2A protocol changes

## Files Modified

1. `.github/workflows/integration-load-test.yml`
   - Added existence check
   - Conditional execution
   - Disabled nightly schedule

2. `.github/workflows/a2a-load-test.yml` (new)
   - Tests A2A protocol endpoints
   - Uses correct locustfile
   - Reasonable performance targets

## Testing Approach

### Integration Layer (Future)
When `src/agentcore/integration/` is created:
- Workflow automatically activates
- Tests integration endpoints
- High performance targets (10k+ req/s)

### A2A Protocol (Current)
Active now for A2A protocol testing:
- Tests JSON-RPC endpoints
- WebSocket connections
- Agent registration/discovery
- Task management

## Performance Targets

### Integration Layer (10k+ req/s)
- Throughput: 10,000+ req/s
- P95 Latency: <100ms
- Success Rate: 99.9%+

### A2A Protocol (Baseline)
- Throughput: Baseline measurement
- P95 Latency: <200ms
- Success Rate: 99%+

## Verification

### Before Fix
```bash
Total Requests:    74,941
Failed Requests:   74,941
Success Rate:      0.00%  ❌
```

### After Fix
- Integration workflow: Skipped (layer doesn't exist) ✅
- A2A workflow: Will test actual endpoints ✅
- No false failures ✅

## Related Files

- `tests/load/integration_layer_load_test.py` - Integration tests (future)
- `tests/load/locustfile.py` - A2A protocol tests (active)
- `tests/load/http_load_test.py` - HTTP load tests
- `tests/load/websocket_load_test.py` - WebSocket tests

## Future Work

1. **Integration Layer Implementation**
   - Create `src/agentcore/integration/` directory
   - Implement LLM provider routing
   - Implement webhook system
   - Implement event publishing
   - Implement storage adapters

2. **When Integration Layer Exists**
   - Workflow automatically activates
   - Tests will run on integration changes
   - Nightly schedule can be re-enabled

## Commit

```
fix(workflows): fix load test workflow to skip non-existent integration layer
```

## Impact

✅ No more false workflow failures
✅ Tests only implemented features
✅ Clear path for future integration testing
✅ A2A protocol has dedicated load testing
