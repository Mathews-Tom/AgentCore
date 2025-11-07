# SDK Generation Guide

This guide explains how to generate client SDKs for the AgentCore Gateway API using the OpenAPI specification.

## Prerequisites

- Docker (recommended) or Java 11+ (for openapi-generator)
- Access to AgentCore Gateway API (running locally or deployed)

## Quick Start

### 1. Download OpenAPI Specification

```bash
# From running gateway
curl http://localhost:8080/openapi.json > openapi.json

# Or from repository
cp src/gateway/openapi.json openapi.json
```

### 2. Generate SDK Using Docker

We provide a convenient script for generating SDKs:

```bash
# Generate Python SDK
./scripts/generate-sdk.sh python

# Generate JavaScript/TypeScript SDK
./scripts/generate-sdk.sh typescript-axios

# Generate Java SDK
./scripts/generate-sdk.sh java

# Generate Go SDK
./scripts/generate-sdk.sh go
```

## Supported Languages

### Python SDK

**Generator:** `python`

**Output:** `./generated-sdks/python/`

**Installation:**
```bash
cd generated-sdks/python
pip install -e .
```

**Usage:**
```python
from agentcore_sdk import ApiClient, Configuration
from agentcore_sdk.api import AuthenticationApi

# Configure client
config = Configuration(host="http://localhost:8080")
client = ApiClient(configuration=config)
auth_api = AuthenticationApi(client)

# Authenticate
token_response = auth_api.create_token({
    "grant_type": "password",
    "username": "user",
    "password": "user123"
})

# Use access token
config.access_token = token_response.access_token
```

### TypeScript/JavaScript SDK

**Generator:** `typescript-axios`

**Output:** `./generated-sdks/typescript/`

**Installation:**
```bash
cd generated-sdks/typescript
npm install
npm run build
```

**Usage:**
```typescript
import { Configuration, AuthenticationApi } from 'agentcore-sdk';

// Configure client
const config = new Configuration({
  basePath: 'http://localhost:8080',
});
const authApi = new AuthenticationApi(config);

// Authenticate
const tokenResponse = await authApi.createToken({
  grant_type: 'password',
  username: 'user',
  password: 'user123',
});

// Use access token
config.accessToken = tokenResponse.data.access_token;
```

### Java SDK

**Generator:** `java`

**Output:** `./generated-sdks/java/`

**Installation:**
```bash
cd generated-sdks/java
mvn clean install
```

**Usage:**
```java
import ai.agentcore.client.ApiClient;
import ai.agentcore.client.api.AuthenticationApi;
import ai.agentcore.client.model.TokenRequest;

// Configure client
ApiClient client = new ApiClient();
client.setBasePath("http://localhost:8080");
AuthenticationApi authApi = new AuthenticationApi(client);

// Authenticate
TokenRequest request = new TokenRequest();
request.setGrantType("password");
request.setUsername("user");
request.setPassword("user123");

TokenResponse response = authApi.createToken(request);

// Use access token
client.setAccessToken(response.getAccessToken());
```

### Go SDK

**Generator:** `go`

**Output:** `./generated-sdks/go/`

**Installation:**
```bash
cd generated-sdks/go
go mod init agentcore-sdk
go mod tidy
```

**Usage:**
```go
package main

import (
    "context"
    "fmt"
    agentcore "agentcore-sdk"
)

func main() {
    cfg := agentcore.NewConfiguration()
    cfg.Host = "http://localhost:8080"
    client := agentcore.NewAPIClient(cfg)

    // Authenticate
    ctx := context.Background()
    tokenReq := agentcore.TokenRequest{
        GrantType: "password",
        Username:  agentcore.PtrString("user"),
        Password:  agentcore.PtrString("user123"),
    }

    tokenResp, _, err := client.AuthenticationApi.CreateToken(ctx).
        TokenRequest(tokenReq).Execute()
    if err != nil {
        panic(err)
    }

    // Use access token
    cfg.AddDefaultHeader("Authorization", fmt.Sprintf("Bearer %s", tokenResp.AccessToken))
}
```

## Advanced Configuration

### Custom Generator Configuration

Create a configuration file for fine-tuned SDK generation:

**Python Example** (`python-config.yaml`):
```yaml
packageName: agentcore_sdk
projectName: agentcore-sdk
packageVersion: 1.0.0
packageUrl: https://github.com/agentcore/agentcore-sdk-python
library: asyncio
generateSourceCodeOnly: false
```

**TypeScript Example** (`typescript-config.yaml`):
```yaml
npmName: @agentcore/sdk
npmVersion: 1.0.0
supportsES6: true
withInterfaces: true
useSingleRequestParameter: true
```

### Generate with Custom Config

```bash
docker run --rm \
  -v ${PWD}:/local \
  openapitools/openapi-generator-cli generate \
  -i /local/openapi.json \
  -g python \
  -o /local/generated-sdks/python \
  -c /local/python-config.yaml
```

## Manual Installation (Without Docker)

### Install OpenAPI Generator

```bash
# macOS
brew install openapi-generator

# Linux/Windows
# Download from: https://github.com/OpenAPITools/openapi-generator
```

### Generate SDK

```bash
openapi-generator generate \
  -i openapi.json \
  -g python \
  -o ./generated-sdks/python \
  --additional-properties=packageName=agentcore_sdk
```

## SDK Features

All generated SDKs include:

- **Type Safety:** Strong typing for requests and responses
- **Authentication:** Built-in JWT bearer token support
- **Error Handling:** Proper exception handling for API errors
- **Async Support:** Asynchronous operations where available
- **Documentation:** Inline documentation from OpenAPI spec
- **Examples:** Code examples for common operations

## Testing Generated SDKs

### Python
```bash
cd generated-sdks/python
pip install -e ".[test]"
pytest tests/
```

### TypeScript
```bash
cd generated-sdks/typescript
npm install
npm test
```

### Java
```bash
cd generated-sdks/java
mvn test
```

### Go
```bash
cd generated-sdks/go
go test ./...
```

## Publishing SDKs

### Python (PyPI)

```bash
cd generated-sdks/python
python -m build
twine upload dist/*
```

### TypeScript (npm)

```bash
cd generated-sdks/typescript
npm publish --access public
```

### Java (Maven Central)

```bash
cd generated-sdks/java
mvn deploy
```

### Go (GitHub)

```bash
cd generated-sdks/go
git tag v1.0.0
git push origin v1.0.0
```

## Troubleshooting

### Issue: Generated code has compilation errors

**Solution:** Ensure OpenAPI spec is valid:
```bash
docker run --rm -v ${PWD}:/local \
  openapitools/openapi-generator-cli validate \
  -i /local/openapi.json
```

### Issue: Missing authentication in generated SDK

**Solution:** Verify security schemes in OpenAPI spec:
```bash
curl http://localhost:8080/openapi.json | jq '.components.securitySchemes'
```

### Issue: SDK doesn't match deployed API

**Solution:** Regenerate from latest OpenAPI spec:
```bash
curl http://localhost:8080/openapi.json > openapi.json
./scripts/generate-sdk.sh <language>
```

## Further Resources

- [OpenAPI Generator Documentation](https://openapi-generator.tech/docs/usage)
- [OpenAPI Specification](https://spec.openapis.org/oas/v3.1.0)
- [AgentCore API Documentation](http://localhost:8080/docs)
- [GitHub: OpenAPI Generator](https://github.com/OpenAPITools/openapi-generator)

## Support

For SDK generation issues:
- Open GitHub Issue: https://github.com/agentcore/agentcore/issues
- Email: api-support@agentcore.ai
- Documentation: https://docs.agentcore.ai
