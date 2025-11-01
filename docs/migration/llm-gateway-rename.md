# LLM Gateway Migration Guide: Portkey → LLM Gateway

**Version**: 1.0
**Date**: 2025-11-01
**Status**: Required for v2.0+

---

## Overview

AgentCore has renamed the `portkey` module to `llm_gateway` to better reflect its purpose as a multi-provider LLM gateway. This migration guide helps you update imports, references, and configurations to use the new module name.

### Why This Change?

- ✅ **Clear Naming**: "LLM Gateway" better describes functionality
- ✅ **Provider Agnostic**: Not tied to specific third-party service (Portkey)
- ✅ **Consistency**: Aligns with AgentCore naming conventions
- ✅ **Flexibility**: Easier to add new providers without confusion

---

## Migration Summary

### Module Rename Table

| Old Path | New Path | Status |
|----------|----------|--------|
| `agentcore.integration.portkey` | `agentcore.llm_gateway` | ✅ Renamed |
| `agentcore.integration.portkey.client` | `agentcore.llm_gateway.client` | ✅ Renamed |
| `agentcore.integration.portkey.config` | `agentcore.llm_gateway.config` | ✅ Renamed |
| `agentcore.integration.portkey.models` | `agentcore.llm_gateway.models` | ✅ Renamed |
| `agentcore.integration.portkey.provider` | `agentcore.llm_gateway.provider` | ✅ Renamed |
| `agentcore.integration.portkey.failover` | `agentcore.llm_gateway.failover` | ✅ Renamed |
| `agentcore.integration.portkey.cost_tracker` | `agentcore.llm_gateway.cost_tracker` | ✅ Renamed |
| `agentcore.integration.portkey.cache_service` | `agentcore.llm_gateway.cache_service` | ✅ Renamed |
| `tests/integration/portkey` | `tests/integration/llm_gateway` | ✅ Renamed |

---

## Step-by-Step Migration

### Step 1: Update Python Imports

**Before**:
```python
# ❌ Old imports
from agentcore.integration.portkey import PortkeyClient
from agentcore.integration.portkey.config import PortkeyConfig
from agentcore.integration.portkey.provider import ProviderRegistry
from agentcore.integration.portkey.failover import FailoverManager
from agentcore.integration.portkey.cost_tracker import CostTracker
```

**After**:
```python
# ✅ New imports
from agentcore.llm_gateway import LLMGatewayClient
from agentcore.llm_gateway.config import LLMGatewayConfig
from agentcore.llm_gateway.provider import ProviderRegistry
from agentcore.llm_gateway.failover import FailoverManager
from agentcore.llm_gateway.cost_tracker import CostTracker
```

### Step 2: Update Class Names

Some classes have been renamed for consistency:

**Before**:
```python
# ❌ Old class names
from agentcore.integration.portkey import PortkeyClient
from agentcore.integration.portkey.config import PortkeyConfig
from agentcore.integration.portkey.exceptions import PortkeyError

client = PortkeyClient(config=PortkeyConfig())
```

**After**:
```python
# ✅ New class names
from agentcore.llm_gateway import LLMGatewayClient
from agentcore.llm_gateway.config import LLMGatewayConfig
from agentcore.llm_gateway.exceptions import LLMGatewayError

client = LLMGatewayClient(config=LLMGatewayConfig())
```

### Step 3: Update Configuration Files

**Old `config.toml`**:
```toml
[integration.portkey]
enabled = true
cache_ttl_seconds = 3600
cost_tracking_enabled = true
```

**New `config.toml`**:
```toml
[llm_gateway]
enabled = true
cache_ttl_seconds = 3600
cost_tracking_enabled = true
```

**Old environment variables**:
```bash
PORTKEY_CACHE_ENABLED=true
PORTKEY_COST_TRACKING=true
PORTKEY_FAILOVER_ENABLED=true
```

**New environment variables**:
```bash
LLM_GATEWAY_CACHE_ENABLED=true
LLM_GATEWAY_COST_TRACKING=true
LLM_GATEWAY_FAILOVER_ENABLED=true
```

### Step 4: Update Test Imports

**Old test imports**:
```python
# ❌ Old test imports
from tests.integration.portkey.conftest import portkey_client
from agentcore.integration.portkey.models import PortkeyRequest
```

**New test imports**:
```python
# ✅ New test imports
from tests.integration.llm_gateway.conftest import llm_gateway_client
from agentcore.llm_gateway.models import LLMGatewayRequest
```

---

## Detailed Changes by Component

### Client (`client.py`)

**Before**:
```python
from agentcore.integration.portkey import PortkeyClient

class MyService:
    def __init__(self):
        self.client = PortkeyClient(
            api_key="sk-...",
            base_url="https://api.portkey.ai"
        )

    async def complete(self, prompt: str) -> str:
        response = await self.client.complete(
            prompt=prompt,
            model="gpt-5-mini"
        )
        return response.text
```

**After**:
```python
from agentcore.llm_gateway import LLMGatewayClient

class MyService:
    def __init__(self):
        self.client = LLMGatewayClient(
            # Configuration now provider-agnostic
            providers=["openai", "anthropic", "google"]
        )

    async def complete(self, prompt: str) -> str:
        response = await self.client.complete(
            prompt=prompt,
            model="gpt-5-mini"
        )
        return response.text
```

### Configuration (`config.py`)

**Before**:
```python
from agentcore.integration.portkey.config import PortkeyConfig

config = PortkeyConfig(
    api_key="pk-...",
    virtual_key="vk-...",
    cache_enabled=True
)
```

**After**:
```python
from agentcore.llm_gateway.config import LLMGatewayConfig

config = LLMGatewayConfig(
    # Provider keys now specified directly
    openai_api_key="sk-...",
    anthropic_api_key="sk-ant-...",
    cache_enabled=True
)
```

### Provider Registry (`provider.py`)

**Before**:
```python
from agentcore.integration.portkey.provider import ProviderRegistry

registry = ProviderRegistry()
providers = registry.list_providers()
```

**After**:
```python
from agentcore.llm_gateway.provider import ProviderRegistry

registry = ProviderRegistry()
providers = registry.list_providers()
# Same API, just different import path
```

### Failover Manager (`failover.py`)

**Before**:
```python
from agentcore.integration.portkey.failover import FailoverManager, FailoverConfig

failover = FailoverManager(
    config=FailoverConfig(
        primary_provider="portkey",
        fallback_providers=["openai", "anthropic"]
    )
)
```

**After**:
```python
from agentcore.llm_gateway.failover import FailoverManager, FailoverConfig

failover = FailoverManager(
    config=FailoverConfig(
        primary_provider="openai",
        fallback_providers=["anthropic", "google"]
    )
)
```

### Cost Tracking (`cost_tracker.py`)

**Before**:
```python
from agentcore.integration.portkey.cost_tracker import CostTracker

tracker = CostTracker()
total_cost = await tracker.get_total_cost()
```

**After**:
```python
from agentcore.llm_gateway.cost_tracker import CostTracker

tracker = CostTracker()
total_cost = await tracker.get_total_cost()
# Same API
```

### Cache Service (`cache_service.py`)

**Before**:
```python
from agentcore.integration.portkey.cache_service import CacheService

cache = CacheService(redis_url="redis://localhost:6379")
```

**After**:
```python
from agentcore.llm_gateway.cache_service import CacheService

cache = CacheService(redis_url="redis://localhost:6379")
# Same API
```

---

## Complete Example Migration

### Before (Old Portkey Code)

```python
# main.py
from agentcore.integration.portkey import PortkeyClient
from agentcore.integration.portkey.config import PortkeyConfig
from agentcore.integration.portkey.failover import FailoverManager
from agentcore.integration.portkey.cost_tracker import CostTracker

class LLMService:
    def __init__(self):
        # Old Portkey configuration
        self.config = PortkeyConfig(
            api_key=os.getenv("PORTKEY_API_KEY"),
            virtual_key=os.getenv("PORTKEY_VIRTUAL_KEY"),
            cache_enabled=True
        )

        self.client = PortkeyClient(config=self.config)

        self.failover = FailoverManager(
            primary="portkey",
            fallbacks=["openai"]
        )

        self.cost_tracker = CostTracker()

    async def generate(self, prompt: str) -> str:
        try:
            response = await self.client.complete(
                prompt=prompt,
                model="gpt-4o-mini"
            )

            # Track costs
            await self.cost_tracker.track_request(
                provider="portkey",
                model="gpt-4o-mini",
                tokens=response.usage.total_tokens
            )

            return response.text

        except Exception as e:
            # Failover to backup
            return await self.failover.execute_with_fallback(
                prompt=prompt,
                model="gpt-4o-mini"
            )
```

### After (New LLM Gateway Code)

```python
# main.py
from agentcore.llm_gateway import LLMGatewayClient
from agentcore.llm_gateway.config import LLMGatewayConfig
from agentcore.llm_gateway.failover import FailoverManager
from agentcore.llm_gateway.cost_tracker import CostTracker

class LLMService:
    def __init__(self):
        # New provider-agnostic configuration
        self.config = LLMGatewayConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            cache_enabled=True
        )

        self.client = LLMGatewayClient(config=self.config)

        self.failover = FailoverManager(
            primary="openai",
            fallbacks=["anthropic", "google"]
        )

        self.cost_tracker = CostTracker()

    async def generate(self, prompt: str) -> str:
        try:
            response = await self.client.complete(
                prompt=prompt,
                model="gpt-5-mini"  # Updated model name
            )

            # Track costs (same API)
            await self.cost_tracker.track_request(
                provider="openai",
                model="gpt-5-mini",
                tokens=response.usage.total_tokens
            )

            return response.text

        except Exception as e:
            # Failover to backup (same API)
            return await self.failover.execute_with_fallback(
                prompt=prompt,
                model="gpt-5-mini"
            )
```

---

## Automated Migration

Use our migration script to automatically update imports:

```bash
# Download migration script
curl -O https://raw.githubusercontent.com/Mathews-Tom/AgentCore/main/scripts/migrate_llm_gateway.py

# Dry run
python scripts/migrate_llm_gateway.py --dry-run --path .

# Apply changes
python scripts/migrate_llm_gateway.py --path . --backup

# Verify
git diff
```

**What the script does**:
1. Finds all `portkey` imports
2. Replaces with `llm_gateway` imports
3. Updates class names (PortkeyClient → LLMGatewayClient)
4. Updates config variables
5. Renames test directories
6. Creates backups

---

## Testing Your Migration

### Unit Tests

```bash
# Run all tests
uv run pytest

# Run LLM gateway tests specifically
uv run pytest tests/integration/llm_gateway/
```

### Integration Tests

```python
import pytest
from agentcore.llm_gateway import LLMGatewayClient

@pytest.mark.asyncio
async def test_migration():
    """Verify LLM gateway works after migration"""
    client = LLMGatewayClient()

    # Test basic completion
    response = await client.complete(
        prompt="Say hello",
        model="gpt-5-mini"
    )

    assert response.text
    assert response.model == "gpt-5-mini"
    assert response.usage.total_tokens > 0
```

### Manual Verification

```python
# Check imports work
from agentcore.llm_gateway import LLMGatewayClient
from agentcore.llm_gateway.config import LLMGatewayConfig
from agentcore.llm_gateway.provider import ProviderRegistry

# Verify old imports fail
try:
    from agentcore.integration.portkey import PortkeyClient
    raise AssertionError("Old import should fail!")
except ImportError:
    print("✅ Old imports correctly removed")
```

---

## Common Issues

### Issue 1: Import Errors

**Error**: `ModuleNotFoundError: No module named 'agentcore.integration.portkey'`

**Cause**: Old imports still in code

**Solution**:
```bash
# Find all old imports
grep -r "from agentcore.integration.portkey" .

# Update manually or use migration script
python scripts/migrate_llm_gateway.py --path .
```

### Issue 2: Configuration Not Found

**Error**: `KeyError: 'integration.portkey'`

**Cause**: Old config keys in config files

**Solution**:
```toml
# Change
[integration.portkey]  # ❌

# To
[llm_gateway]  # ✅
```

### Issue 3: Class Name Errors

**Error**: `NameError: name 'PortkeyClient' is not defined`

**Cause**: Class name not updated

**Solution**:
```python
# Change
client = PortkeyClient()  # ❌

# To
client = LLMGatewayClient()  # ✅
```

---

## Backward Compatibility

**Note**: AgentCore v2.0+ does NOT provide backward compatibility for the `portkey` module. You must migrate.

If you need to maintain both old and new code during transition:

```python
# Temporary compatibility shim (NOT recommended for production)
try:
    from agentcore.llm_gateway import LLMGatewayClient as Client
except ImportError:
    from agentcore.integration.portkey import PortkeyClient as Client

# Use Client() in your code
client = Client()
```

---

## Database Migration

If you store provider names or module paths in the database:

```sql
-- Update provider references
UPDATE llm_requests
SET provider_module = 'llm_gateway'
WHERE provider_module = 'portkey';

-- Update configuration keys
UPDATE system_config
SET config_key = REPLACE(config_key, 'portkey', 'llm_gateway')
WHERE config_key LIKE '%portkey%';
```

---

## Checklist

- [ ] Update Python imports (`portkey` → `llm_gateway`)
- [ ] Update class names (`PortkeyClient` → `LLMGatewayClient`)
- [ ] Update configuration files (`[integration.portkey]` → `[llm_gateway]`)
- [ ] Update environment variables (`PORTKEY_*` → `LLM_GATEWAY_*`)
- [ ] Update test files and fixtures
- [ ] Update database records (if applicable)
- [ ] Run automated migration script
- [ ] Run unit tests
- [ ] Run integration tests
- [ ] Update documentation
- [ ] Deploy to staging and validate
- [ ] Deploy to production
- [ ] Remove backup files after validation

---

## Rollback

If you need to rollback:

```bash
# Restore from backup
python scripts/migrate_llm_gateway.py --restore --backup-dir=.migration_backup/

# Or revert git commit
git revert <commit-hash>

# Redeploy
kubectl rollout restart deployment/agentcore-api
```

---

## FAQs

### Q: Why was this renamed?

**A**: "LLM Gateway" better describes the module's purpose as a multi-provider LLM gateway, not tied to a specific third-party service.

### Q: Will Portkey integration be removed?

**A**: The module supports all providers (OpenAI, Anthropic, Google) directly. No separate Portkey integration is needed.

### Q: Can I still use Portkey API?

**A**: Yes, but you'll need to configure it as a custom provider. The default providers are OpenAI, Anthropic, and Google.

### Q: Do I need to change my API keys?

**A**: If you were using Portkey virtual keys, switch to direct provider API keys:
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GOOGLE_API_KEY`

### Q: What about cost tracking?

**A**: Cost tracking is still available and now works with all providers, not just through Portkey.

---

## Additional Resources

- **[LLM Gateway Documentation](../llm-client-service/README.md)**
- **[Configuration Guide](../llm-client-service/configuration-guide.md)**
- **[Provider Setup](../llm-client-service/provider-setup.md)**
- **[Cost Tracking Guide](../llm-client-service/cost-tracking.md)**

---

## Support

**Need Help?**
- GitHub Issues: https://github.com/Mathews-Tom/AgentCore/issues
- Add `migration` label for priority support
- Include error messages and stack traces

---

**Last Updated**: 2025-11-01
**Migration Deadline**: 2025-12-01 (Recommended)
