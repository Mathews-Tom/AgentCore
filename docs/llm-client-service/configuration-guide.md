# Configuration Guide

Complete guide for configuring the LLM Client Service including provider API keys, model governance, timeout settings, and environment setup.

## Table of Contents

- [Environment Variables](#environment-variables)
- [Provider Configuration](#provider-configuration)
  - [OpenAI Setup](#openai-setup)
  - [Anthropic Setup](#anthropic-setup)
  - [Google Gemini Setup](#google-gemini-setup)
- [Model Governance](#model-governance)
- [Timeout and Retry Configuration](#timeout-and-retry-configuration)
- [Production Configuration](#production-configuration)
- [Docker Configuration](#docker-configuration)
- [Kubernetes Configuration](#kubernetes-configuration)

---

## Environment Variables

All configuration is managed through environment variables or `.env` files.

### Core Settings

```bash
# OpenAI API Key
OPENAI_API_KEY="sk-..."

# Anthropic API Key
ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini API Key
GOOGLE_API_KEY="..."

# Model Governance (JSON array of allowed models)
ALLOWED_MODELS='["gpt-4.1-mini","claude-3-5-haiku-20241022","gemini-2.0-flash-exp"]'

# Default model (must be in ALLOWED_MODELS)
LLM_DEFAULT_MODEL="gpt-4.1-mini"

# Request timeout in seconds (default: 60.0)
LLM_REQUEST_TIMEOUT=60.0

# Maximum retry attempts on transient errors (default: 3)
LLM_MAX_RETRIES=3
```

### Environment File (.env)

Create `.env` file in project root:

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
GOOGLE_API_KEY=your-google-key-here

ALLOWED_MODELS=["gpt-4.1-mini","claude-3-5-haiku-20241022"]
LLM_DEFAULT_MODEL=gpt-4.1-mini

LLM_REQUEST_TIMEOUT=60.0
LLM_MAX_RETRIES=3
```

**Important:** Add `.env` to `.gitignore` to prevent committing API keys.

---

## Provider Configuration

### OpenAI Setup

#### 1. Get API Key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Navigate to API Keys section
3. Create new secret key
4. Copy key (starts with `sk-`)

#### 2. Set Environment Variable

```bash
export OPENAI_API_KEY="sk-..."
```

#### 3. Supported Models

```python
# Available OpenAI models
OPENAI_MODELS = [
    "gpt-4.1",          # Latest GPT-4 Turbo (premium tier)
    "gpt-4.1-mini",     # Cost-effective GPT-4 (fast tier)
    "gpt-5",            # Cutting-edge reasoning (premium tier)
    "gpt-5-mini",       # Balanced performance (balanced tier)
]
```

#### 4. Usage Limits

- Free tier: Limited requests per minute
- Pay-as-you-go: Higher rate limits based on usage
- Enterprise: Custom rate limits

**Rate Limit Headers:**
- `x-ratelimit-limit-requests`
- `x-ratelimit-remaining-requests`
- `x-ratelimit-limit-tokens`

#### 5. Cost Estimation

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| gpt-4.1-mini | $0.15 | $0.60 |
| gpt-4.1 | $3.00 | $12.00 |
| gpt-5-mini | $0.30 | $1.20 |
| gpt-5 | $5.00 | $20.00 |

*(Prices as of 2025-01 - check OpenAI pricing page for current rates)*

---

### Anthropic Setup

#### 1. Get API Key

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Navigate to API Keys
3. Create new API key
4. Copy key (starts with `sk-ant-`)

#### 2. Set Environment Variable

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

#### 3. Supported Models

```python
# Available Anthropic models
ANTHROPIC_MODELS = [
    "claude-3-5-sonnet",           # Highest capability (premium tier)
    "claude-3-5-haiku-20241022",   # Fast and efficient (balanced tier)
    "claude-3-opus",               # Maximum intelligence (premium tier)
]
```

#### 4. Usage Limits

- Default: 50 requests per minute
- Tier-based rate limits increase with usage
- Custom enterprise limits available

#### 5. Cost Estimation

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| claude-3-5-haiku-20241022 | $0.25 | $1.25 |
| claude-3-5-sonnet | $3.00 | $15.00 |
| claude-3-opus | $15.00 | $75.00 |

*(Prices as of 2025-01 - check Anthropic pricing page for current rates)*

#### 6. Key Differences

**Message Format:**
- System messages must be in separate `system` parameter
- Only "user" and "assistant" roles in messages array
- Requires `max_tokens` parameter (no default)

**Response Format:**
- Content in `content[0].text` (not `choices[0].message.content`)
- Token usage: `input_tokens` and `output_tokens`

---

### Google Gemini Setup

#### 1. Get API Key

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Click "Get API Key"
3. Create new API key or use existing
4. Copy key

#### 2. Set Environment Variable

```bash
export GOOGLE_API_KEY="..."
```

#### 3. Supported Models

```python
# Available Gemini models
GEMINI_MODELS = [
    "gemini-2.0-flash-exp",   # Experimental flash model (fast tier)
    "gemini-1.5-pro",         # Advanced reasoning (premium tier)
    "gemini-2.0-flash-exp",       # Fast responses (fast tier)
]
```

#### 4. Usage Limits

- Free tier: 15 requests per minute, 1M tokens per minute
- Paid tier: 360 requests per minute, higher token limits

#### 5. Cost Estimation

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| gemini-2.0-flash-exp | $0.075 | $0.30 |
| gemini-1.5-pro | $1.25 | $5.00 |
| gemini-2.0-flash-exp | Free (limited) | Free (limited) |

*(Prices as of 2025-01 - check Google AI pricing page for current rates)*

---

## Model Governance

Model governance prevents unauthorized model usage and controls costs.

### Configuration

```bash
# Set allowed models (JSON array)
export ALLOWED_MODELS='[
  "gpt-4.1-mini",
  "claude-3-5-haiku-20241022",
  "gemini-2.0-flash-exp"
]'
```

### Governance Rules

1. **Requests for non-allowed models are rejected** with `ModelNotAllowedError`
2. **Governance violations are tracked** in Prometheus metrics
3. **All violations are logged** with `source_agent` for accountability

### Common Governance Patterns

#### 1. Cost Control (Fast Models Only)

```bash
# Only allow cost-effective fast tier models
export ALLOWED_MODELS='[
  "gpt-4.1-mini",
  "gemini-2.0-flash-exp",
  "claude-3-5-haiku-20241022"
]'
```

#### 2. Provider Lock-in (Single Provider)

```bash
# Only allow OpenAI models
export ALLOWED_MODELS='[
  "gpt-4.1-mini",
  "gpt-4.1",
  "gpt-5-mini",
  "gpt-5"
]'
```

#### 3. Tiered Access (Development vs Production)

**Development:**
```bash
export ALLOWED_MODELS='["gpt-4.1-mini"]'
```

**Production:**
```bash
export ALLOWED_MODELS='[
  "gpt-4.1-mini",
  "gpt-5-mini",
  "claude-3-5-haiku-20241022",
  "claude-3-5-sonnet"
]'
```

#### 4. Multi-Provider Strategy

```bash
# Allow one model from each provider
export ALLOWED_MODELS='[
  "gpt-4.1-mini",
  "claude-3-5-haiku-20241022",
  "gemini-2.0-flash-exp"
]'
```

### Monitoring Governance

```python
# Query Prometheus for governance violations
llm_governance_violations_total{model="gpt-3.5-turbo",source_agent="agent-001"}
```

**Grafana Alert:**
```yaml
alert: HighGovernanceViolations
expr: rate(llm_governance_violations_total[5m]) > 10
for: 5m
annotations:
  summary: "High rate of LLM governance violations"
```

---

## Timeout and Retry Configuration

### Request Timeout

Controls maximum time to wait for LLM provider response.

```bash
# Set global timeout (seconds)
export LLM_REQUEST_TIMEOUT=60.0
```

**Timeout Recommendations:**

| Model Tier | Recommended Timeout |
|-----------|-------------------|
| Fast | 30s |
| Balanced | 60s |
| Premium | 120s |

### Retry Configuration

Controls retry behavior on transient errors (rate limits, connection issues).

```bash
# Set maximum retry attempts
export LLM_MAX_RETRIES=3
```

**Retry Behavior:**
- Exponential backoff: 1s, 2s, 4s, 8s, ...
- Only transient errors retried (rate limits, connection errors)
- Terminal errors fail immediately (authentication, bad request)

### Custom Timeout/Retry

```python
from agentcore.a2a_protocol.services.llm_service import LLMService

# Create custom service instance
custom_service = LLMService(
    timeout=120.0,     # 2 minutes
    max_retries=5,     # 5 attempts
)

# Use custom service
response = await custom_service.complete(request)
```

---

## Production Configuration

### Secret Management

**Never hardcode API keys in code or commit to version control.**

#### AWS Secrets Manager

```python
import boto3
import json

def get_secret(secret_name: str) -> dict:
    """Retrieve secret from AWS Secrets Manager."""
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Load secrets
secrets = get_secret('agentcore/llm-api-keys')

os.environ['OPENAI_API_KEY'] = secrets['openai_key']
os.environ['ANTHROPIC_API_KEY'] = secrets['anthropic_key']
os.environ['GOOGLE_API_KEY'] = secrets['google_key']
```

#### HashiCorp Vault

```python
import hvac

def get_vault_secrets() -> dict:
    """Retrieve secrets from HashiCorp Vault."""
    client = hvac.Client(url='https://vault.example.com')
    client.auth.approle.login(role_id='...', secret_id='...')

    secrets = client.secrets.kv.v2.read_secret_version(path='llm-api-keys')
    return secrets['data']['data']

# Load secrets
secrets = get_vault_secrets()
os.environ['OPENAI_API_KEY'] = secrets['openai_key']
```

### Environment-Specific Configuration

#### Development

```bash
# .env.development
ALLOWED_MODELS=["gpt-4.1-mini"]
LLM_DEFAULT_MODEL=gpt-4.1-mini
LLM_REQUEST_TIMEOUT=30.0
LLM_MAX_RETRIES=2
LOG_LEVEL=DEBUG
```

#### Staging

```bash
# .env.staging
ALLOWED_MODELS=["gpt-4.1-mini","claude-3-5-haiku-20241022"]
LLM_DEFAULT_MODEL=gpt-4.1-mini
LLM_REQUEST_TIMEOUT=60.0
LLM_MAX_RETRIES=3
LOG_LEVEL=INFO
```

#### Production

```bash
# .env.production
ALLOWED_MODELS=["gpt-4.1-mini","gpt-5-mini","claude-3-5-haiku-20241022","claude-3-5-sonnet"]
LLM_DEFAULT_MODEL=claude-3-5-haiku-20241022
LLM_REQUEST_TIMEOUT=60.0
LLM_MAX_RETRIES=3
LOG_LEVEL=WARNING
ENABLE_METRICS=true
```

---

## Docker Configuration

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install uv && uv pip install -e .

# Copy application
COPY . .

# Environment variables (override at runtime)
ENV ALLOWED_MODELS='["gpt-4.1-mini"]'
ENV LLM_DEFAULT_MODEL=gpt-4.1-mini
ENV LLM_REQUEST_TIMEOUT=60.0
ENV LLM_MAX_RETRIES=3

# Run application
CMD ["uvicorn", "agentcore.a2a_protocol.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  agentcore:
    build: .
    ports:
      - "8001:8001"
    environment:
      # API Keys (use secrets in production)
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      GOOGLE_API_KEY: ${GOOGLE_API_KEY}

      # Model Governance
      ALLOWED_MODELS: '["gpt-4.1-mini","claude-3-5-haiku-20241022"]'
      LLM_DEFAULT_MODEL: gpt-4.1-mini

      # Timeouts
      LLM_REQUEST_TIMEOUT: 60.0
      LLM_MAX_RETRIES: 3

      # Logging
      LOG_LEVEL: INFO

    # Use Docker secrets for production
    secrets:
      - openai_key
      - anthropic_key
      - google_key

secrets:
  openai_key:
    file: ./secrets/openai_key.txt
  anthropic_key:
    file: ./secrets/anthropic_key.txt
  google_key:
    file: ./secrets/google_key.txt
```

---

## Kubernetes Configuration

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-client-config
data:
  ALLOWED_MODELS: '["gpt-4.1-mini","claude-3-5-haiku-20241022","gemini-2.0-flash-exp"]'
  LLM_DEFAULT_MODEL: "gpt-4.1-mini"
  LLM_REQUEST_TIMEOUT: "60.0"
  LLM_MAX_RETRIES: "3"
  LOG_LEVEL: "INFO"
```

### Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: llm-api-keys
type: Opaque
stringData:
  OPENAI_API_KEY: "sk-..."
  ANTHROPIC_API_KEY: "sk-ant-..."
  GOOGLE_API_KEY: "..."
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentcore
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agentcore
  template:
    metadata:
      labels:
        app: agentcore
    spec:
      containers:
      - name: agentcore
        image: agentcore:latest
        ports:
        - containerPort: 8001
        envFrom:
        - configMapRef:
            name: llm-client-config
        - secretRef:
            name: llm-api-keys
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

### Create Secrets from Files

```bash
# Create secret from environment file
kubectl create secret generic llm-api-keys \
  --from-env-file=.env.production

# Create secret from literals
kubectl create secret generic llm-api-keys \
  --from-literal=OPENAI_API_KEY=sk-... \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-... \
  --from-literal=GOOGLE_API_KEY=...
```

---

## Configuration Validation

### Startup Validation

```python
from agentcore.a2a_protocol.config import settings

def validate_configuration():
    """Validate LLM client configuration at startup."""

    # Check API keys are set
    providers = []
    if settings.OPENAI_API_KEY:
        providers.append("OpenAI")
    if settings.ANTHROPIC_API_KEY:
        providers.append("Anthropic")
    if settings.GOOGLE_API_KEY:
        providers.append("Gemini")

    if not providers:
        raise RuntimeError("No LLM provider API keys configured")

    print(f"Configured providers: {', '.join(providers)}")

    # Check allowed models
    if not settings.ALLOWED_MODELS:
        raise RuntimeError("ALLOWED_MODELS is empty")

    print(f"Allowed models: {settings.ALLOWED_MODELS}")

    # Check default model is allowed
    if settings.LLM_DEFAULT_MODEL not in settings.ALLOWED_MODELS:
        raise RuntimeError(
            f"Default model '{settings.LLM_DEFAULT_MODEL}' not in ALLOWED_MODELS"
        )

    # Check timeout is reasonable
    if settings.LLM_REQUEST_TIMEOUT < 10.0:
        print(f"Warning: LLM_REQUEST_TIMEOUT is very low ({settings.LLM_REQUEST_TIMEOUT}s)")

# Run at startup
validate_configuration()
```

---

## Best Practices

1. **Never commit API keys** - Use environment variables and secret management
2. **Implement model governance** - Define ALLOWED_MODELS to control costs
3. **Use separate keys** - Different keys for dev/staging/production
4. **Monitor usage** - Track requests and tokens via Prometheus metrics
5. **Set appropriate timeouts** - Balance between reliability and user experience
6. **Rotate keys regularly** - Follow provider security recommendations
7. **Use secrets management** - AWS Secrets Manager, HashiCorp Vault, etc.
8. **Validate configuration** - Check settings at application startup
9. **Log governance violations** - Track unauthorized model access attempts
10. **Test configuration** - Verify API keys work before deployment

---

## Troubleshooting Configuration

See [Troubleshooting Guide](./troubleshooting-guide.md) for common configuration issues and solutions.
