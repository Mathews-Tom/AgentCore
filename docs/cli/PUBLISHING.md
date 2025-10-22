# Publishing Guide for AgentCore

This document describes the process for publishing AgentCore to PyPI (Python Package Index).

## Overview

AgentCore uses a hybrid publishing workflow:
- **`uv`** for building and verification
- **`twine`** for uploading to PyPI
- **GitHub Actions** for automated publishing

## Prerequisites

### Required Tools

```bash
# Install uv (package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install twine (PyPI uploader)
pip install twine

# Optional: GitHub CLI for workflow triggers
brew install gh  # macOS
```

### PyPI Accounts

1. **Test PyPI** (for testing): https://test.pypi.org/account/register/
2. **Production PyPI**: https://pypi.org/account/register/

### API Tokens

1. Go to https://pypi.org/manage/account/token/
2. Create API token with "Upload packages" scope
3. Store securely (you'll need this for manual publishing)

For GitHub Actions, set up trusted publishing (recommended):
1. Go to https://pypi.org/manage/account/publishing/
2. Add pending publisher for `agentcore/agentcore` repository
3. No API tokens needed - uses OIDC authentication

## Publishing Methods

### Method 1: Automated Publishing (Recommended)

Uses GitHub Actions for fully automated publishing.

#### A. Publish to Test PyPI

```bash
# Trigger workflow manually
gh workflow run publish-pypi.yml --field environment=testpypi

# Monitor workflow
gh run watch
```

#### B. Publish to Production PyPI

```bash
# Create and publish a GitHub release
gh release create v0.1.0 --title "AgentCore CLI v0.1.0" --notes "Initial release"

# The workflow automatically triggers on release
```

### Method 2: Manual Publishing

For local testing and development.

#### Step 1: Verify Package Quality

Run the verification script:

```bash
./scripts/verify-package.sh
```

This script:
- Cleans previous builds
- Builds package with `uv build`
- Verifies metadata with `twine check`
- Inspects package contents
- Tests installation in virtual environment
- Validates CLI commands

#### Step 2: Build Package

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build with uv
uv build
```

Output:
```
Built agentcore-0.1.0.tar.gz
Built agentcore-0.1.0-py3-none-any.whl
```

#### Step 3: Upload to Test PyPI

```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# You'll be prompted for credentials
# Username: __token__
# Password: <your-testpypi-api-token>
```

#### Step 4: Test Installation from Test PyPI

```bash
# Create test environment
python -m venv test-env
source test-env/bin/activate

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ agentcore

# Test CLI
agentcore --version
agentcore --help

# Cleanup
deactivate
rm -rf test-env
```

#### Step 5: Upload to Production PyPI

Once verified on Test PyPI:

```bash
# Upload to production PyPI
twine upload dist/*

# You'll be prompted for credentials
# Username: __token__
# Password: <your-pypi-api-token>
```

## Version Management

### Updating Version Number

Edit `pyproject.toml`:

```toml
[project]
name = "agentcore"
version = "0.2.0"  # Update this
```

### Semantic Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.2.0): New features, backward compatible
- **PATCH** (0.1.1): Bug fixes, backward compatible

### Version Checklist

Before bumping version:

- [ ] Update `pyproject.toml` version
- [ ] Update `CHANGELOG.md` with changes
- [ ] Update `docs/cli/README.md` if needed
- [ ] Run full test suite: `uv run pytest`
- [ ] Run verification script: `./scripts/verify-package.sh`
- [ ] Create git tag: `git tag v0.2.0`
- [ ] Push tag: `git push origin v0.2.0`

## Publishing Workflow

### Complete Publishing Checklist

#### Pre-Release

- [ ] All tests passing (`uv run pytest`)
- [ ] Code coverage ≥90% (`uv run pytest --cov`)
- [ ] Type checking passes (`uv run mypy src/`)
- [ ] Linting passes (`uv run ruff check src/`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in pyproject.toml

#### Build & Verify

- [ ] Run verification script: `./scripts/verify-package.sh`
- [ ] Verify package contents manually
- [ ] Test installation locally
- [ ] Test CLI commands work

#### Test PyPI

- [ ] Upload to Test PyPI
- [ ] Install from Test PyPI
- [ ] Test on multiple Python versions (3.12, 3.13)
- [ ] Test on multiple platforms (Linux, macOS, Windows)

#### Production PyPI

- [ ] Create git tag
- [ ] Create GitHub release
- [ ] Upload to PyPI (auto or manual)
- [ ] Verify installation from PyPI
- [ ] Update documentation links

#### Post-Release

- [ ] Announce on GitHub Discussions
- [ ] Update Discord/Slack channels
- [ ] Tweet/social media announcement
- [ ] Monitor PyPI download stats
- [ ] Monitor GitHub issues for bugs

## GitHub Actions Workflow

### Workflow File

Located at `.github/workflows/publish-pypi.yml`

### Triggers

1. **Automatic**: On GitHub release creation
2. **Manual**: Via workflow dispatch with environment selection

### Jobs

1. **build**: Build distribution with `uv build` and verify with `twine check`
2. **publish-testpypi**: Upload to Test PyPI (manual dispatch only)
3. **publish-pypi**: Upload to production PyPI (on release or manual)
4. **verify-installation**: Test installation on multiple platforms

### Using the Workflow

#### Test Publishing

```bash
# Using GitHub CLI
gh workflow run publish-pypi.yml --field environment=testpypi

# Or via GitHub UI
# 1. Go to Actions tab
# 2. Select "Publish to PyPI" workflow
# 3. Click "Run workflow"
# 4. Select "testpypi" environment
# 5. Click "Run workflow"
```

#### Production Publishing

```bash
# Create release (triggers automatic publishing)
gh release create v0.1.0 \
  --title "AgentCore CLI v0.1.0" \
  --notes-file CHANGELOG.md

# Or manually trigger workflow
gh workflow run publish-pypi.yml --field environment=pypi
```

## Troubleshooting

### Build Fails

**Error**: `uv build` fails

**Solution**:
```bash
# Update uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clean and rebuild
rm -rf dist/ build/ *.egg-info
uv build
```

### Package Validation Fails

**Error**: `twine check` reports errors

**Solution**:
```bash
# Check metadata
twine check dist/*

# Verify pyproject.toml is valid
uv run python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"

# Fix metadata errors and rebuild
uv build
```

### Upload Fails - Authentication

**Error**: `401 Unauthorized`

**Solution**:
```bash
# Verify API token is correct
# For manual upload, use:
# Username: __token__
# Password: pypi-... (your API token)

# Or configure in ~/.pypirc
cat > ~/.pypirc << 'EOF'
[pypi]
username = __token__
password = <your-pypi-api-token>

[testpypi]
username = __token__
password = <your-testpypi-api-token>
EOF

chmod 600 ~/.pypirc
```

### Upload Fails - Version Already Exists

**Error**: `File already exists`

**Solution**:
```bash
# You cannot overwrite existing versions on PyPI
# Bump the version in pyproject.toml and rebuild

# For Test PyPI, you can use skip-existing:
twine upload --repository testpypi --skip-existing dist/*
```

### Installation Fails from PyPI

**Error**: Package not found

**Solution**:
```bash
# Wait a few minutes - PyPI indexing can take time

# Verify package exists
curl https://pypi.org/pypi/agentcore/json | jq '.info.version'

# Clear pip cache
pip cache purge

# Try again
pip install agentcore
```

## Security Best Practices

### API Tokens

- ✅ Use scoped tokens (upload only)
- ✅ Store in password manager
- ✅ Use GitHub Secrets for CI/CD
- ✅ Rotate tokens regularly
- ❌ Never commit tokens to git
- ❌ Never share tokens in issues/PRs

### Trusted Publishing (Recommended)

GitHub Actions supports PyPI trusted publishing:

1. No API tokens needed
2. Uses OIDC authentication
3. More secure than API tokens
4. Already configured in `publish-pypi.yml`

## Monitoring

### PyPI Statistics

- Downloads: https://pypistats.org/packages/agentcore
- Project page: https://pypi.org/project/agentcore/
- Release history: https://pypi.org/project/agentcore/#history

### GitHub Release Metrics

- Releases: https://github.com/agentcore/agentcore/releases
- Download counts per release
- Community feedback

## Rollback Procedure

### If Bad Version Published

PyPI doesn't allow deleting versions, but you can:

1. **Yank the version** (doesn't delete, but marks as unavailable):
   ```bash
   # Via PyPI web interface
   # Project settings → Manage → Yank release
   ```

2. **Publish a fixed version** immediately:
   ```bash
   # Bump to 0.1.1 (patch version)
   # Fix the issue
   # Publish new version
   ```

3. **Notify users**:
   - Update GitHub release notes
   - Post in Discord/community channels
   - Create GitHub issue explaining the problem

## References

- [PyPI Documentation](https://packaging.python.org/tutorials/packaging-projects/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [GitHub Actions Publishing](https://docs.github.com/en/actions/publishing-packages)
- [Semantic Versioning](https://semver.org/)

## Support

For publishing issues:
- Create GitHub issue: https://github.com/agentcore/agentcore/issues
- Contact maintainers: team@agentcore.ai

---

**Last Updated**: 2025-10-22
