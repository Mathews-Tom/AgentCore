# Test Runner Documentation

## Overview

The `run_tests.py` script provides comprehensive test execution with detailed metrics, coverage reporting, and performance analysis for the AgentCore test suite.

## Features

- ✅ **Automated test discovery** - Finds all test components automatically
- 📊 **Detailed metrics** - Pass/fail rates, coverage, timing per component
- ⚡ **Parallel execution** - Run test components in parallel for faster results
- 📈 **Coverage tracking** - Per-component and overall coverage reporting
- 📄 **JSON export** - Save results in machine-readable format
- 🎨 **Beautiful reports** - Clean, readable console output with emojis and formatting

## Quick Start

### Run all tests (sequential)

```bash
uv run python tests/run_tests.py
```

### Run tests in parallel

```bash
uv run python tests/run_tests.py --parallel
```

### Run specific components

```bash
uv run python tests/run_tests.py --components a2a_protocol,agent_runtime
```

### Skip coverage (faster)

```bash
uv run python tests/run_tests.py --no-coverage
```

### Verbose output

```bash
uv run python tests/run_tests.py --verbose
```

### Save results as JSON

```bash
uv run python tests/run_tests.py --json test_results.json
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--components <list>` | Comma-separated list of components to test |
| `--parallel` | Run test components in parallel |
| `--no-coverage` | Skip coverage reporting (faster execution) |
| `--verbose` | Show verbose pytest output |
| `--json <file>` | Save results as JSON to specified file |
| `-h, --help` | Show help message |

## Example Output

```
================================================================================
🧪 AgentCore Test Suite
================================================================================
Start Time: 2025-10-14 14:30:00
Components: 8
Mode: Sequential
Coverage: Enabled
================================================================================

[1/8] Running a2a_protocol... ✅ (3.2s)
[2/8] Running agent_runtime... ✅ (2.8s)
[3/8] Running cli... ✅ (1.5s)
...

================================================================================
📊 Test Results Summary
================================================================================

Component                   Tests     Pass     Fail     Skip      Time   Coverage
------------------------- -------- -------- -------- -------- ---------- ----------
✅ a2a_protocol                181      181        0        0      3.2s      85.3%
✅ agent_runtime               120      118        2        0      2.8s      72.1%
✅ cli                          15       15        0        0      1.5s      91.2%
...
------------------------- -------- -------- -------- -------- ---------- ----------
TOTAL                          450      445        5        0      12.5s      78.9%

================================================================================
📈 Overall Statistics
================================================================================

Total Duration:    12.5s
Success Rate:      🟢 98.9%
Average Coverage:  🟡 78.9%

================================================================================
⚡ Performance Metrics
================================================================================

Fastest Component: cli (1.5s)
Slowest Component: a2a_protocol (3.2s)
Average Duration:  1.6s

================================================================================
📋 Coverage Breakdown
================================================================================

cli                       ██████████████████████████████████████████████░░░  91.2%
a2a_protocol              ██████████████████████████████████████████░░░░░░░  85.3%
agent_runtime             ████████████████████████████████████░░░░░░░░░░░░  72.1%
...

================================================================================
✅ All tests passed!
================================================================================
```

## Component Structure

The test runner automatically discovers test components based on the directory structure in `tests/`:

```plaintext
tests/
├── a2a_protocol/      # A2A protocol tests
├── agent_runtime/     # Agent runtime tests
├── cli/               # CLI tests
├── gateway/           # Gateway tests
├── integration/       # Integration tests
├── load/              # Load/performance tests
├── orchestration/     # Orchestration tests
└── run_tests.py       # This test runner
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Run Tests with Coverage
  run: |
    uv run python tests/run_tests.py --parallel --json test-results.json

- name: Upload Test Results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: test-results.json
```

### Local Development Workflow

```bash
# Quick test run (no coverage, parallel)
uv run python tests/run_tests.py --parallel --no-coverage

# Full test run with coverage
uv run python tests/run_tests.py

# Test specific component while developing
uv run python tests/run_tests.py --components a2a_protocol --verbose
```

## Performance Tips

1. **Use `--parallel`** for faster execution when testing multiple components
2. **Use `--no-coverage`** when you only need pass/fail status
3. **Test specific components** during development to get faster feedback
4. **Use `--json`** to track metrics over time

## Troubleshooting

### Tests timeout

- Default timeout is 5 minutes per component
- Check for hanging tests or infinite loops
- Consider running with `--verbose` to see where it hangs

### Coverage not showing

- Ensure the source path matches: `src/agentcore/<component>/`
- Check that test files follow the pattern `test_*.py`

### Component not found

- Verify the component directory exists in `tests/`
- Check that it contains at least one `test_*.py` file

## Advanced Usage

### Watch Mode (Future Feature)

```bash
# Automatically rerun tests on file changes
uv run python tests/run_tests.py --watch
```

### Custom Coverage Thresholds (Future Feature)

```bash
# Fail if coverage drops below threshold
uv run python tests/run_tests.py --min-coverage 85
```

## Output Format

### Console Output

- ✅ Green checkmark for passed tests
- ❌ Red X for failed tests
- 🟢 Green percentage for good coverage (≥90%)
- 🟡 Yellow percentage for medium coverage (70-89%)
- 🔴 Red percentage for low coverage (<70%)

### JSON Output

```json
{
  "start_time": "2025-10-14T14:30:00",
  "end_time": "2025-10-14T14:30:12",
  "total_duration": 12.5,
  "total_tests": 450,
  "total_passed": 445,
  "total_failed": 5,
  "total_skipped": 0,
  "overall_success_rate": 98.9,
  "overall_coverage": 78.9,
  "components": [...]
}
```

## Contributing

To add new test components:

1. Create a new directory in `tests/`
2. Add test files following the pattern `test_*.py`
3. The test runner will automatically discover it

## Support

For issues or questions:

- Check the main project documentation
- Review pytest configuration in `pytest.ini`
- Check coverage configuration in `.coveragerc`
