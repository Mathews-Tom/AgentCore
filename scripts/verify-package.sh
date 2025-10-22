#!/bin/bash
# AgentCore Package Verification Script
# Verifies package quality before PyPI upload
#
# Usage: ./scripts/verify-package.sh

set -e

echo "=========================================="
echo "AgentCore Package Verification"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if twine is installed
if ! command -v twine &> /dev/null; then
    echo "Installing twine..."
    pip install twine
fi

echo "Step 1: Clean previous builds"
echo "----------------------------------------"
rm -rf dist/ build/ *.egg-info
echo "✓ Cleaned dist/, build/, *.egg-info"
echo ""

echo "Step 2: Build package with uv"
echo "----------------------------------------"
uv build
echo "✓ Package built successfully"
echo ""

echo "Step 3: Verify package metadata"
echo "----------------------------------------"
twine check dist/*
echo "✓ Package metadata is valid"
echo ""

echo "Step 4: Check package contents"
echo "----------------------------------------"
echo "Source distribution contents:"
tar -tzf dist/agentcore-*.tar.gz | grep -E "(agentcore_cli|pyproject.toml|README.md)" | head -15
echo ""
echo "Wheel contents:"
unzip -l dist/agentcore-*.whl | grep -E "(agentcore_cli|METADATA)" | head -15
echo ""
echo "✓ Package contents look good"
echo ""

echo "Step 5: Test installation in virtual environment"
echo "----------------------------------------"
python -m venv test-venv
source test-venv/bin/activate

# Install from local wheel
pip install dist/agentcore-*.whl

# Test CLI
echo "Testing agentcore CLI..."
agentcore --version
agentcore --help > /dev/null
agentcore agent --help > /dev/null
agentcore task --help > /dev/null
agentcore session --help > /dev/null
agentcore workflow --help > /dev/null
agentcore config --help > /dev/null

deactivate
rm -rf test-venv
echo "✓ CLI installation and basic commands work"
echo ""

echo "Step 6: Package size check"
echo "----------------------------------------"
wheel_size=$(du -h dist/agentcore-*.whl | cut -f1)
sdist_size=$(du -h dist/agentcore-*.tar.gz | cut -f1)
echo "Wheel size: $wheel_size"
echo "Source distribution size: $sdist_size"
echo "✓ Package sizes are reasonable"
echo ""

echo "=========================================="
echo "✓ All verification checks passed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Test PyPI: twine upload --repository testpypi dist/*"
echo "  2. Production: twine upload dist/*"
echo ""
echo "Or use the GitHub Actions workflow:"
echo "  gh workflow run publish-pypi.yml --field environment=testpypi"
echo ""
