#!/usr/bin/env bash
# Full test suite
#
# Runs all tests including both fast and real integration tests.
# If Docker is not available, skips real integration tests.

set -e

echo "🧪 Running Full Test Suite"
echo "=========================="
echo ""

# Run all tests (requires Docker for full coverage)
if ! docker info > /dev/null 2>&1; then
    echo "⚠️  Docker not running - skipping integration tests marked for real services"
    echo "   Starting Docker Desktop will enable full test coverage"
    echo ""
    ./scripts/test-fast.sh "$@"
else
    echo "Uses: All test modes (fast + real integration)"
    echo "Docker: Available"
    echo ""
    # Override pytest.ini default marker filter to include integration tests
    uv run pytest -m "" "$@"
fi

echo ""
echo "✅ All tests complete!"
