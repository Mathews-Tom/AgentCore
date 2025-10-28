# Model Selection Strategy

## Overview

The ModelSelector provides intelligent, tier-based model selection for the AgentCore LLM Client Service. This feature enables cost optimization by automatically selecting appropriate models based on task complexity and configured tiers, rather than requiring hardcoded model choices.

**Status:** P1 (Nice-to-have) cost optimization feature
**Implementation:** `src/agentcore/a2a_protocol/services/model_selector.py`

## Model Tiers

The ModelSelector organizes models into three tiers based on capability and cost:

### FAST Tier
**Use Case:** Simple tasks requiring low latency and cost-effective processing

**Models (in priority order):**
- `gpt-4.1-mini` (OpenAI) - Primary choice
- `gpt-5-mini` (OpenAI) - Fallback
- `claude-3-5-haiku-20241022` (Anthropic) - Multi-provider option
- `gemini-2.0-flash-exp` (Google) - Multi-provider option

**Characteristics:**
- Lowest cost per token
- Fastest response times
- Suitable for: simple Q&A, classification, basic summarization

### BALANCED Tier
**Use Case:** Moderate complexity tasks requiring good quality with reasonable cost

**Models (in priority order):**
- `gpt-4.1` (OpenAI) - Primary choice
- `claude-3-5-sonnet` (Anthropic) - Multi-provider option
- `gemini-1.5-pro` (Google) - Multi-provider option

**Characteristics:**
- Moderate cost per token
- Balanced quality and speed
- Suitable for: code generation, analysis, moderate reasoning tasks

### PREMIUM Tier
**Use Case:** Complex tasks requiring highest quality regardless of cost

**Models (in priority order):**
- `gpt-5` (OpenAI) - Primary choice
- `claude-3-opus` (Anthropic) - Multi-provider option
- `gemini-2.0-flash-exp` (Google) - Multi-provider option

**Characteristics:**
- Highest cost per token
- Best quality and reasoning capability
- Suitable for: complex reasoning, research, critical decisions

## Complexity Mapping

For simplified API usage, task complexity strings automatically map to tiers:

| Complexity Level | Model Tier | Example Use Cases |
|-----------------|------------|------------------|
| `low` | FAST | Simple Q&A, classification, keyword extraction |
| `medium` | BALANCED | Code generation, summarization, data analysis |
| `high` | PREMIUM | Complex reasoning, research synthesis, critical analysis |

## Usage Examples

### Basic Tier Selection

```python
from agentcore.a2a_protocol.services.model_selector import ModelSelector
from agentcore.a2a_protocol.models.llm import ModelTier

# Create selector (uses default configuration)
selector = ModelSelector()

# Select by tier
fast_model = selector.select_model(ModelTier.FAST)  # "gpt-4.1-mini"
balanced_model = selector.select_model(ModelTier.BALANCED)  # "gpt-4.1"
premium_model = selector.select_model(ModelTier.PREMIUM)  # "gpt-5"
```

### Complexity-Based Selection

```python
from agentcore.a2a_protocol.services.model_selector import ModelSelector

selector = ModelSelector()

# Select by task complexity (simpler API)
model_low = selector.select_model_by_complexity("low")  # "gpt-4.1-mini"
model_medium = selector.select_model_by_complexity("medium")  # "gpt-4.1"
model_high = selector.select_model_by_complexity("high")  # "gpt-5"

# Case-insensitive
model = selector.select_model_by_complexity("Low")  # "gpt-4.1-mini"
```

### Provider Preference

Configure provider preference for multi-provider failover:

```python
from agentcore.a2a_protocol.services.model_selector import ModelSelector

# Prefer OpenAI, fallback to Anthropic, then Gemini
selector = ModelSelector(provider_preference=["openai", "anthropic", "gemini"])

# If OpenAI's gpt-4.1-mini is available, it will be selected
# If not, Claude or Gemini models will be used as fallback
model = selector.select_model(ModelTier.FAST)
```

### Integration with LLMService

```python
from agentcore.a2a_protocol.services.model_selector import ModelSelector
from agentcore.a2a_protocol.services.llm_service import llm_service
from agentcore.a2a_protocol.models.llm import LLMRequest, ModelTier

# Select model based on task complexity
selector = ModelSelector()
model = selector.select_model(ModelTier.FAST)

# Use selected model with LLMService
request = LLMRequest(
    model=model,  # "gpt-4.1-mini"
    messages=[{"role": "user", "content": "What is 2+2?"}],
    temperature=0.7,
)

response = await llm_service.complete(request)
print(response.content)  # "4"
```

### Configuration Validation

Validate that all tiers have available models:

```python
from agentcore.a2a_protocol.services.model_selector import ModelSelector

selector = ModelSelector()

# Check configuration at startup
if not selector.validate_configuration():
    print("WARNING: Some model tiers have no allowed models!")
    # This will log detailed warnings about which tiers are missing
```

## Cost Optimization Guidelines

### 1. Use the Lowest Sufficient Tier

Always start with the FAST tier and escalate only when quality is insufficient:

```python
# GOOD: Start with FAST tier
model = selector.select_model_by_complexity("low")

# AVOID: Using PREMIUM tier for simple tasks
# This wastes resources and increases costs
model = selector.select_model_by_complexity("high")  # Overkill for simple Q&A
```

### 2. Batch Similar Requests

Group similar-complexity requests to reuse the same model:

```python
# GOOD: Batch similar complexity requests
fast_model = selector.select_model(ModelTier.FAST)
for question in simple_questions:
    request = LLMRequest(model=fast_model, messages=[...])
    response = await llm_service.complete(request)

# AVOID: Selecting model for each request
# This adds overhead and may select different models
for question in simple_questions:
    model = selector.select_model(ModelTier.FAST)  # Unnecessary overhead
    request = LLMRequest(model=model, messages=[...])
```

### 3. Monitor Token Usage by Tier

Track which tiers consume the most tokens to identify optimization opportunities:

```python
import logging

# ModelSelector logs selection rationale with tier and model info
# Use structured logging to aggregate by tier:

# Example log output:
# INFO Model selected tier=fast selected_model=gpt-4.1-mini provider=openai
# INFO Model selected tier=premium selected_model=gpt-5 provider=openai

# Analyze logs to identify:
# - Which tiers are used most frequently
# - Whether PREMIUM tier usage is justified
# - Opportunities to downgrade tier without quality loss
```

### 4. Provider Preference for Cost

Configure provider preference based on cost structure:

```python
# If Anthropic offers better pricing for your use case:
selector = ModelSelector(provider_preference=["anthropic", "openai", "gemini"])

# This will prefer Claude models when available:
# FAST tier: claude-3-5-haiku-20241022 instead of gpt-4.1-mini
# BALANCED tier: claude-3-5-sonnet instead of gpt-4.1
```

### 5. Complexity-Based Auto-Escalation

Use task characteristics to automatically select tier:

```python
def get_task_complexity(task: dict) -> str:
    """Determine task complexity based on characteristics."""
    # Simple heuristics:
    if task["type"] == "classification":
        return "low"
    elif task["type"] == "code_generation":
        return "medium"
    elif task["type"] == "research":
        return "high"

    # Token-based heuristics:
    if len(task["prompt"].split()) < 50:
        return "low"
    elif len(task["prompt"].split()) < 200:
        return "medium"
    else:
        return "high"

# Auto-select tier based on task
complexity = get_task_complexity(task)
model = selector.select_model_by_complexity(complexity)
```

## Configuration

### ALLOWED_MODELS

Model selection respects the `ALLOWED_MODELS` configuration in `config.py`:

```python
# config.py
ALLOWED_MODELS: list[str] = [
    "gpt-4.1-mini",  # FAST tier
    "gpt-5-mini",     # FAST tier fallback
    "claude-3-5-haiku-20241022",  # FAST tier multi-provider
    "gemini-2.0-flash-exp",  # FAST tier multi-provider
]
```

**Governance Rules:**
- Only models in `ALLOWED_MODELS` can be selected
- If a tier has no allowed models, `validate_configuration()` returns False
- Selection fails with `ValueError` if no allowed models for tier

### Provider Preference

Provider preference can be configured via constructor:

```python
# Option 1: Hardcode preference
selector = ModelSelector(provider_preference=["openai", "anthropic", "gemini"])

# Option 2: Load from environment/config
import os
preference = os.getenv("LLM_PROVIDER_PREFERENCE", "openai,anthropic,gemini").split(",")
selector = ModelSelector(provider_preference=preference)

# Option 3: No preference (use tier priority order)
selector = ModelSelector()  # Uses TIER_MODEL_MAP priority
```

## Logging and Observability

### Selection Rationale Logs

Every model selection logs detailed rationale:

```python
# INFO level log for successful selection:
{
    "tier": "fast",
    "selected_model": "gpt-4.1-mini",
    "provider": "openai",
    "candidate_count": 4,
    "allowed_count": 4
}

# WARNING level for fallback scenarios:
{
    "tier": "fast",
    "provider_preference": ["openai"],
    "fallback_model": "claude-3-5-haiku-20241022",
    "message": "Provider preference not satisfied, using fallback"
}

# ERROR level for missing models:
{
    "tier": "premium",
    "candidate_models": ["gpt-5", "claude-3-opus", "gemini-2.0-flash-exp"],
    "allowed_models": ["gpt-4.1-mini", "gpt-4.1"],
    "message": "No allowed models for tier 'premium'"
}
```

### Configuration Validation Logs

Validation logs identify configuration issues:

```python
# INFO level for valid configuration:
{
    "tiers": ["fast", "balanced", "premium"],
    "allowed_models": ["gpt-4.1-mini", "gpt-4.1", "gpt-5"],
    "message": "Configuration validation passed"
}

# WARNING level for missing tier mappings:
{
    "tier": "premium",
    "candidate_models": ["gpt-5", "claude-3-opus"],
    "allowed_models": ["gpt-4.1-mini", "gpt-4.1"],
    "message": "Tier has no allowed models"
}

# ERROR level for validation failure:
{
    "allowed_models": ["gpt-4.1-mini"],
    "message": "Configuration validation failed - some tiers have no allowed models"
}
```

## Best Practices

1. **Validate Configuration at Startup**
   ```python
   selector = ModelSelector()
   if not selector.validate_configuration():
       logger.error("Model selector configuration invalid!")
   ```

2. **Use Complexity-Based Selection for Flexibility**
   ```python
   # Easier to change tier mappings without code changes
   model = selector.select_model_by_complexity("medium")
   ```

3. **Configure Provider Preference for Resilience**
   ```python
   # Enables multi-provider failover
   selector = ModelSelector(provider_preference=["openai", "anthropic", "gemini"])
   ```

4. **Monitor Selection Logs for Optimization**
   - Track which tiers are used most frequently
   - Identify opportunities to downgrade tier
   - Detect provider availability issues

5. **Combine with LLM Metrics for Complete Observability**
   ```python
   # ModelSelector logs selection rationale
   # LLMService logs usage statistics
   # Together they provide full cost attribution by tier
   ```

## Error Handling

### No Models Available for Tier

```python
try:
    model = selector.select_model(ModelTier.PREMIUM)
except ValueError as e:
    logger.error(f"Model selection failed: {e}")
    # Fallback to lower tier or fail gracefully
    model = selector.select_model(ModelTier.BALANCED)
```

### Invalid Complexity Level

```python
try:
    model = selector.select_model_by_complexity("ultra")  # Invalid
except ValueError as e:
    logger.error(f"Invalid complexity: {e}")
    # Fallback to default tier
    model = selector.select_model(ModelTier.BALANCED)
```

### Configuration Validation Failure

```python
selector = ModelSelector()
if not selector.validate_configuration():
    # Log warning but don't fail startup
    logger.warning("Some model tiers have no allowed models")
    # Application can still use available tiers
```

## Architecture Notes

**Pattern:** Strategy Pattern
**File:** `src/agentcore/a2a_protocol/services/model_selector.py`
**Tests:** `tests/unit/services/test_model_selector.py` (95% coverage)
**Dependencies:** None (standalone utility class)

The ModelSelector is designed as a standalone utility that can be used independently or integrated with LLMService. It has no dependencies on database, HTTP, or other infrastructure, making it lightweight and easy to test.

## Future Enhancements

Potential improvements for future iterations:

1. **Dynamic Tier Mapping**: Load tier-to-model mappings from configuration files
2. **Cost Tracking**: Track estimated cost per tier selection
3. **Auto-Escalation**: Automatically retry with higher tier if lower tier fails
4. **A/B Testing**: Support multiple model variants per tier for experimentation
5. **Usage-Based Selection**: Select model based on historical success rates

## References

- LLM Client Service Specification: `docs/specs/llm-client-service/spec.md`
- LLM Client Implementation Plan: `docs/specs/llm-client-service/plan.md`
- Ticket: `.sage/tickets/LLM-CLIENT-012.md`
- Model Tier Enum: `src/agentcore/a2a_protocol/models/llm.py`
- LLM Service: `src/agentcore/a2a_protocol/services/llm_service.py`
