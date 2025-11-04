#!/usr/bin/env bash
# Real integration test suite
#
# Runs integration tests with real PostgreSQL + Redis via testcontainers.
# Requires Docker Desktop running. Slower but provides definitive validation.

set -e

echo "🔬 Running Real Integration Tests"
echo "=================================="
echo ""
echo "Uses: PostgreSQL (testcontainer) + Redis (testcontainer)"
echo "Requires: Docker Desktop running"
echo ""

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Integration tests require Docker."
    echo "   Start Docker Desktop and try again."
    exit 1
fi

# Run integration tests
echo "📦 Starting testcontainers (PostgreSQL + Redis)..."
uv run pytest -m integration -v "$@"

echo ""
echo "✅ Integration tests complete!"
