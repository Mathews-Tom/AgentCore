#!/usr/bin/env bash
# Full test suite - Python runner with rich display
#
# Wrapper script that calls the Python test runner for better
# visual feedback and progress tracking.

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Python test runner
exec uv run python "$SCRIPT_DIR/test_runner.py" "$@"
