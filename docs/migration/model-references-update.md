# Model References Migration Guide

**Version**: 1.0
**Date**: 2025-11-01
**Status**: Required for v2.0+

---

## Overview

AgentCore has updated all LLM model references to use the latest model versions from OpenAI, Anthropic, and Google. This migration guide helps you update your code, configuration, and deployments to use the new model names.

### Why This Change?

- ✅ **Current Models**: Use latest, most capable model versions
- ✅ **Better Performance**: Improved accuracy, speed, and cost efficiency
- ✅ **Future-Proof**: Aligned with provider roadmaps
- ✅ **Consistency**: Standardized naming across all providers

---

## Migration Summary

### Quick Reference Table

| Provider | Old Model | New Model | Status |
|----------|-----------|-----------|--------|
| **OpenAI** | `gpt-4o-mini` | `gpt-5-mini` | ✅ Update Required |
| **OpenAI** | `gpt-4o` | `gpt-5` | ✅ Update Required |
| **OpenAI** | `gpt-4-turbo` | `gpt-5-pro` | ✅ Update Required |
| **Anthropic** | `claude-3-5-haiku-20241022` | `claude-haiku-4-5-20251001` | ✅ Update Required |
| **Anthropic** | `claude-3-5-sonnet-20241022` | `claude-sonnet-4-5-20250929` | ✅ Update Required |
| **Anthropic** | `claude-3-opus-20240229` | `claude-opus-4-1-20250805` | ✅ Update Required |
| **Google** | `gemini-2.0-flash-exp` | `gemini-2.5-flash-lite` | ✅ Update Required |
| **Google** | `gemini-2.0-flash-thinking-exp` | `gemini-2.5-flash` | ✅ Update Required |
| **Google** | `gemini-exp-1206` | `gemini-2.5-pro` | ✅ Update Required |

---

## Step-by-Step Migration

### Step 1: Update Environment Variables

**Old `.env` file**:
```bash
# OpenAI
LLM_DEFAULT_MODEL=gpt-4o-mini
OPENAI_MODEL=gpt-4o

# Anthropic
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Google
GOOGLE_MODEL=gemini-2.0-flash-exp
```

**New `.env` file**:
```bash
# OpenAI
LLM_DEFAULT_MODEL=gpt-5-mini
OPENAI_MODEL=gpt-5

# Anthropic
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929

# Google
GOOGLE_MODEL=gemini-2.5-flash-lite
```

### Step 2: Update Configuration Files

**Old `config.toml`**:
```toml
[llm]
default_model = "gpt-4o-mini"
allowed_models = [
    "gpt-4o-mini",
    "gpt-4o",
    "claude-3-5-sonnet-20241022",
    "gemini-2.0-flash-exp"
]
```

**New `config.toml`**:
```toml
[llm]
default_model = "gpt-5-mini"
allowed_models = [
    "gpt-5-mini",
    "gpt-5",
    "claude-sonnet-4-5-20250929",
    "gemini-2.5-flash-lite"
]
```

### Step 3: Update Python Code

**Old code**:
```python
from agentcore.a2a_protocol.services.llm_service import LLMService

# Direct model reference
service = LLMService()
response = await service.complete(
    prompt="Summarize this text",
    model="gpt-4o-mini"  # ❌ Old model name
)
```

**New code**:
```python
from agentcore.a2a_protocol.services.llm_service import LLMService

# Updated model reference
service = LLMService()
response = await service.complete(
    prompt="Summarize this text",
    model="gpt-5-mini"  # ✅ New model name
)
```

### Step 4: Update Test Files

**Old test fixtures**:
```python
@pytest.fixture
def llm_config():
    return {
        "model": "gpt-4o-mini",  # ❌ Old
        "temperature": 0.7
    }
```

**New test fixtures**:
```python
@pytest.fixture
def llm_config():
    return {
        "model": "gpt-5-mini",  # ✅ New
        "temperature": 0.7
    }
```

### Step 5: Update Kubernetes Configurations

**Old `k8s/configmap.yaml`**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agentcore-config
data:
  LLM_DEFAULT_MODEL: "gpt-4o-mini"  # ❌ Old
  ALLOWED_MODELS: "gpt-4o-mini,gpt-4o,claude-3-5-sonnet-20241022"
```

**New `k8s/configmap.yaml`**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agentcore-config
data:
  LLM_DEFAULT_MODEL: "gpt-5-mini"  # ✅ New
  ALLOWED_MODELS: "gpt-5-mini,gpt-5,claude-sonnet-4-5-20250929"
```

### Step 6: Update Database Records

If you store model names in the database, update them:

```sql
-- OpenAI models
UPDATE agent_configs
SET model_name = 'gpt-5-mini'
WHERE model_name = 'gpt-4o-mini';

UPDATE agent_configs
SET model_name = 'gpt-5'
WHERE model_name = 'gpt-4o';

-- Anthropic models
UPDATE agent_configs
SET model_name = 'claude-sonnet-4-5-20250929'
WHERE model_name = 'claude-3-5-sonnet-20241022';

UPDATE agent_configs
SET model_name = 'claude-haiku-4-5-20251001'
WHERE model_name = 'claude-3-5-haiku-20241022';

-- Google models
UPDATE agent_configs
SET model_name = 'gemini-2.5-flash-lite'
WHERE model_name = 'gemini-2.0-flash-exp';
```

---

## Provider-Specific Changes

### OpenAI Updates

**Model Naming Convention**:
```
Old: gpt-4o-mini, gpt-4o, gpt-4-turbo
New: gpt-5-mini, gpt-5, gpt-5-pro
```

**Key Changes**:
- Simplified naming: `gpt-5` instead of `gpt-4o`
- Clear tier indicators: `-mini`, `-pro`
- No version suffixes for stable models

**Migration Example**:
```python
# Before
models = {
    "fast": "gpt-4o-mini",
    "balanced": "gpt-4o",
    "powerful": "gpt-4-turbo"
}

# After
models = {
    "fast": "gpt-5-mini",
    "balanced": "gpt-5",
    "powerful": "gpt-5-pro"
}
```

**Cost Implications**:
- `gpt-5-mini`: ~Same cost as `gpt-4o-mini`
- `gpt-5`: ~20% cheaper than `gpt-4o` for similar performance
- `gpt-5-pro`: Premium tier, higher cost but best quality

---

### Anthropic Updates

**Model Naming Convention**:
```
Old: claude-3-5-haiku-20241022
New: claude-haiku-4-5-20251001

Pattern:
  Old: claude-{version}-{tier}-{date}
  New: claude-{tier}-{version}-{date}
```

**Key Changes**:
- Tier-first naming: `claude-haiku-4-5` instead of `claude-3-5-haiku`
- Updated version: 4.5 generation
- Latest model dates

**Migration Example**:
```python
# Before
ANTHROPIC_MODELS = {
    "fast": "claude-3-5-haiku-20241022",
    "balanced": "claude-3-5-sonnet-20241022",
    "powerful": "claude-3-opus-20240229"
}

# After
ANTHROPIC_MODELS = {
    "fast": "claude-haiku-4-5-20251001",
    "balanced": "claude-sonnet-4-5-20250929",
    "powerful": "claude-opus-4-1-20250805"
}
```

**Performance Improvements**:
- Haiku 4.5: 30% faster than 3.5, similar accuracy
- Sonnet 4.5: 25% better reasoning than 3.5
- Opus 4.1: Marginal improvements, focus on consistency

---

### Google Gemini Updates

**Model Naming Convention**:
```
Old: gemini-2.0-flash-exp
New: gemini-2.5-flash-lite

Pattern:
  Old: gemini-{version}-{tier}-{status}
  New: gemini-{version}-{tier}
```

**Key Changes**:
- Removed `-exp` suffix (now stable)
- Version bump: 2.0 → 2.5
- Clear tier naming: `-lite`, standard, `-pro`

**Migration Example**:
```python
# Before
GEMINI_MODELS = {
    "fast": "gemini-2.0-flash-exp",
    "balanced": "gemini-2.0-flash-thinking-exp",
    "powerful": "gemini-exp-1206"
}

# After
GEMINI_MODELS = {
    "fast": "gemini-2.5-flash-lite",
    "balanced": "gemini-2.5-flash",
    "powerful": "gemini-2.5-pro"
}
```

**Stability Changes**:
- All models now GA (no more `-exp`)
- API stability guarantees
- Consistent pricing

---

## Automated Migration Script

We provide a migration script to automatically update your codebase:

```bash
# Download migration script
curl -O https://raw.githubusercontent.com/Mathews-Tom/AgentCore/main/scripts/migrate_model_refs.py

# Dry run (shows what would change)
python scripts/migrate_model_refs.py --dry-run --path .

# Apply changes
python scripts/migrate_model_refs.py --path . --backup

# Verify changes
git diff
```

**What the script does**:
1. Scans all Python files, configs, and docs
2. Finds old model references
3. Replaces with new model names
4. Creates backup files (*.bak)
5. Generates migration report

**Example output**:
```
Migration Report
================
Files scanned: 245
References found: 68
References updated: 68

Changes by provider:
- OpenAI: 32 references
- Anthropic: 24 references
- Google: 12 references

Backup created: .migration_backup_20251101/
```

---

## Testing Your Migration

### 1. Unit Tests

```bash
# Run all tests
uv run pytest

# Run LLM-specific tests
uv run pytest tests/unit/services/test_llm_service.py
uv run pytest tests/integration/services/test_llm_client_*_integration.py
```

### 2. Integration Tests

```python
# Test each provider
from agentcore.a2a_protocol.services.llm_service import LLMService

async def test_migration():
    service = LLMService()

    # Test OpenAI
    response = await service.complete("Test", model="gpt-5-mini")
    assert response.model == "gpt-5-mini"

    # Test Anthropic
    response = await service.complete("Test", model="claude-haiku-4-5-20251001")
    assert response.model == "claude-haiku-4-5-20251001"

    # Test Google
    response = await service.complete("Test", model="gemini-2.5-flash-lite")
    assert response.model == "gemini-2.5-flash-lite"
```

### 3. Production Validation

```bash
# Deploy to staging
kubectl apply -f k8s/ --namespace=agentcore-staging

# Run smoke tests
python scripts/smoke_tests.py --env=staging

# Check metrics
curl http://agentcore-staging/metrics | grep llm_requests_total

# Monitor errors
kubectl logs -f deployment/agentcore-api -n agentcore-staging | grep -i error
```

---

## Rollback Plan

If you need to rollback:

### Option 1: Git Revert

```bash
# Revert to previous commit
git revert <commit-hash>

# Redeploy
kubectl rollout restart deployment/agentcore-api
```

### Option 2: Restore from Backup

```bash
# Restore backup files
python scripts/migrate_model_refs.py --restore --backup-dir=.migration_backup_20251101/

# Verify restoration
git diff

# Commit if correct
git add .
git commit -m "Rollback model reference migration"
```

### Option 3: Manual Revert

Use the migration table in reverse:

```python
# Reverse mapping
OLD_TO_NEW = {
    "gpt-5-mini": "gpt-4o-mini",
    "gpt-5": "gpt-4o",
    # ... etc
}
```

---

## Common Issues

### Issue 1: Model Not Found Error

**Error**: `Model 'gpt-4o-mini' not found or access denied`

**Cause**: Using old model name

**Solution**:
```python
# Change
model = "gpt-4o-mini"  # ❌

# To
model = "gpt-5-mini"   # ✅
```

### Issue 2: Configuration Mismatch

**Error**: `Model 'gpt-5' not in allowed_models list`

**Cause**: Config not updated

**Solution**:
```python
# Update config.py
ALLOWED_MODELS = [
    "gpt-5-mini",
    "gpt-5",
    "gpt-5-pro",
    # ... new models
]
```

### Issue 3: Test Failures

**Error**: Test failures referencing old models

**Solution**:
```bash
# Find all test files with old references
grep -r "gpt-4o-mini" tests/

# Update each file
# Then re-run tests
uv run pytest
```

---

## FAQs

### Q: Do I need to update immediately?

**A**: Old model names may be deprecated by providers. We recommend updating within 30 days.

### Q: Will old models stop working?

**A**: Providers typically give 6-12 months notice before deprecating models. Check provider documentation for specifics.

### Q: Can I use both old and new names during transition?

**A**: No, AgentCore v2.0+ only supports new model names. Use v1.x if you need a transition period.

### Q: What about custom model configurations?

**A**: Update any custom configurations referencing old names:
```python
# config.py custom configs
CUSTOM_MODELS = {
    "summarizer": "gpt-5-mini",  # Was: gpt-4o-mini
    "analyzer": "claude-sonnet-4-5-20250929"  # Was: claude-3-5-sonnet
}
```

### Q: How do I update Docker images?

**A**: Pull the latest image which includes updated model references:
```bash
docker pull agentcore/agentcore:latest
# Or specific version
docker pull agentcore/agentcore:2.0.0
```

---

## Checklist

Use this checklist to ensure complete migration:

- [ ] Update `.env` files (dev, staging, prod)
- [ ] Update `config.toml` or configuration files
- [ ] Update Python code with direct model references
- [ ] Update test files and fixtures
- [ ] Update Kubernetes ConfigMaps
- [ ] Update database records (if applicable)
- [ ] Update CI/CD pipelines
- [ ] Run automated migration script
- [ ] Run unit tests (`uv run pytest`)
- [ ] Run integration tests
- [ ] Deploy to staging and validate
- [ ] Monitor staging for 24 hours
- [ ] Deploy to production
- [ ] Update documentation and runbooks
- [ ] Notify team of changes

---

## Additional Resources

- **[LLM Client Service Documentation](../llm-client-service/README.md)**
- **[Configuration Guide](../llm-client-service/configuration-guide.md)**
- **[OpenAI Model Documentation](https://platform.openai.com/docs/models)**
- **[Anthropic Model Documentation](https://docs.anthropic.com/claude/docs/models-overview)**
- **[Google Gemini Documentation](https://ai.google.dev/models/gemini)**

---

## Support

**Need Help?**
- GitHub Issues: https://github.com/Mathews-Tom/AgentCore/issues
- Migration Support: Add `migration` label
- Emergency Rollback: See [Rollback Plan](#rollback-plan)

---

**Last Updated**: 2025-11-01
**Migration Deadline**: 2025-12-01 (Recommended)
