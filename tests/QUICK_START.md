# Test Runner Quick Start

## One-Line Commands

### Run all tests

```bash
uv run python tests/run_tests.py
```

### Fast mode (parallel, no coverage)

```bash
uv run python tests/run_tests.py --parallel --no-coverage
```

### Single component

```bash
uv run python tests/run_tests.py --components a2a_protocol
```

### Multiple components

```bash
uv run python tests/run_tests.py --components a2a_protocol,agent_runtime
```

### Save results

```bash
uv run python tests/run_tests.py --json results.json
```

## Common Workflows

### Development (fast feedback)

```bash
# Test what you're working on
uv run python tests/run_tests.py --components a2a_protocol --no-coverage
```

### Pre-commit (quick check)

```bash
# Fast parallel run
uv run python tests/run_tests.py --parallel --no-coverage
```

### CI/CD (full report)

```bash
# Complete test suite with coverage and JSON export
uv run python tests/run_tests.py --parallel --json test-results.json
```

### Coverage analysis

```bash
# Focus on specific component with detailed coverage
uv run python tests/run_tests.py --components a2a_protocol --verbose
```

## Component Names

Available test components:

- `a2a_protocol` - A2A Protocol layer tests
- `agent_runtime` - Agent Runtime tests
- `cli` - CLI tests
- `gateway` - Gateway tests
- `integration` - Integration tests
- `load` - Load/performance tests
- `orchestration` - Orchestration tests

## Exit Codes

- `0` - All tests passed
- `1` - One or more tests failed
- `130` - Interrupted by user (Ctrl+C)

## Need Help?

```bash
uv run python tests/run_tests.py --help
```

See `README_TEST_RUNNER.md` for full documentation.
