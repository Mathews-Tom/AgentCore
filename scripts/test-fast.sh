#!/usr/bin/env bash
# Fast test suite for development
#
# Runs fast integration tests using SQLite + fakeredis (no Docker required).
# Perfect for rapid development feedback loop (<5 minutes).

set -e

echo "🚀 Running Fast Test Suite"
echo "=========================="
echo ""
echo "Uses: SQLite (in-memory) + fakeredis"
echo "No Docker required"
echo ""

# Run fast tests only (skip integration tests marked for real services)
uv run pytest -m "not integration and not slow" "$@"

echo ""
echo "✅ Fast tests complete!"
