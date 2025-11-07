# AgentCore API - Getting Started Guide

Welcome to AgentCore! This guide will help you get started with the AgentCore Gateway API in under 10 minutes.

## Quick Navigation

- [Prerequisites](#prerequisites)
- [5-Minute Quickstart](#5-minute-quickstart)
- [Authentication](#authentication)
- [Common Operations](#common-operations)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [Next Steps](#next-steps)

## Prerequisites

- **Development Environment:** Any HTTP client (curl, Postman, Insomnia, or programming language)
- **AgentCore Gateway:** Running locally (http://localhost:8080) or deployed instance
- **Optional:** jq for JSON parsing in command-line examples

## 5-Minute Quickstart

### Step 1: Verify Gateway is Running

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2025-10-18T10:30:00Z"
}
```

### Step 2: Get Access Token

```bash
curl -X POST http://localhost:8080/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "password",
    "username": "user",
    "password": "user123"
  }'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "scope": null
}
```

**Save the access token** - you'll need it for subsequent requests!

### Step 3: Make Authenticated Request

```bash
# Set token as environment variable
export ACCESS_TOKEN="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."

# Get current user info
curl http://localhost:8080/auth/me \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

Response:
```json
{
  "id": "00000000-0000-0000-0000-000000000002",
  "username": "user",
  "email": "user@agentcore.ai",
  "roles": ["user"],
  "is_active": true
}
```

**Congratulations!** You've successfully authenticated and made your first API call.

## Authentication

### Available Methods

#### 1. Username/Password (Development)

Best for: Initial testing, development environments

```bash
curl -X POST http://localhost:8080/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "password",
    "username": "user",
    "password": "user123"
  }'
```

**Demo Accounts:**
- User: `user` / `user123` (roles: user)
- Admin: `admin` / `admin123` (roles: admin, user)
- Service: `service` / `service123` (roles: service)

#### 2. OAuth 2.0 (Production)

Best for: Production applications, web/mobile apps

**Supported Providers:**
- Google OAuth 2.0
- GitHub OAuth 2.0
- Microsoft OAuth 2.0

**Flow:**

1. List available providers:
```bash
curl http://localhost:8080/oauth/providers
```

2. Redirect user to authorization URL:
```
http://localhost:8080/oauth/authorize/google?scope=user:read%20user:write
```

3. Handle callback and receive tokens automatically

4. Use access token in API requests

#### 3. Service Account (Machine-to-Machine)

Best for: Backend services, automation, CI/CD

```bash
curl -X POST http://localhost:8080/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "client_credentials",
    "client_id": "service",
    "client_secret": "service123"
  }'
```

### Token Management

#### Token Expiration
- **Access Token:** 60 minutes
- **Refresh Token:** 7 days

#### Refresh Access Token

```bash
curl -X POST http://localhost:8080/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "YOUR_REFRESH_TOKEN"
  }'
```

#### Logout

```bash
curl -X POST http://localhost:8080/auth/logout \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

## Common Operations

### Health Check

```bash
# Quick health check
curl http://localhost:8080/health

# Readiness check (for load balancers)
curl http://localhost:8080/ready

# Liveness check (for Kubernetes)
curl http://localhost:8080/live
```

### Session Management

#### List Active Sessions

```bash
curl http://localhost:8080/auth/sessions \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

#### Delete Specific Session

```bash
curl -X DELETE http://localhost:8080/auth/sessions/SESSION_ID \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

### OAuth Scopes

#### List Available Scopes

```bash
# All scopes
curl http://localhost:8080/oauth/scopes

# Filter by resource
curl "http://localhost:8080/oauth/scopes?resource=agent"
```

## Error Handling

### Standard Error Format

All errors follow a consistent structure:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description",
    "details": {
      "field": "specific_field",
      "reason": "why it failed"
    },
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2025-10-18T10:30:00Z"
  }
}
```

### Common Error Codes

| Status | Code | Meaning | Solution |
|--------|------|---------|----------|
| 401 | `UNAUTHORIZED` | Missing or invalid token | Authenticate or refresh token |
| 403 | `FORBIDDEN` | Insufficient permissions | Check user roles and scopes |
| 404 | `NOT_FOUND` | Resource doesn't exist | Verify resource ID |
| 422 | `VALIDATION_ERROR` | Invalid request data | Check request parameters |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests | Wait and retry with backoff |
| 503 | `SERVICE_UNAVAILABLE` | Backend service down | Check system health, retry |

### Rate Limiting

All endpoints are rate limited:

**Rate Limits:**
- Per IP: 1,000 requests/minute
- Per User: 5,000 requests/minute
- Per Endpoint: 100 requests/minute

**Response Headers:**
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 750
X-RateLimit-Reset: 1697631600
```

**Handling Rate Limits:**

```python
import time
import requests

def make_request_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)

        if response.status_code == 429:
            retry_after = int(response.headers.get('X-RateLimit-Reset', 60))
            print(f"Rate limited. Retrying after {retry_after}s...")
            time.sleep(retry_after)
            continue

        return response

    raise Exception("Max retries exceeded")
```

## Best Practices

### 1. Security

**DO:**
- ✅ Store tokens securely (environment variables, secure storage)
- ✅ Use HTTPS in production
- ✅ Implement token refresh before expiration
- ✅ Logout when done to invalidate sessions
- ✅ Use OAuth for production web/mobile apps

**DON'T:**
- ❌ Hardcode credentials in source code
- ❌ Share tokens between applications
- ❌ Store tokens in browser localStorage (use httpOnly cookies)
- ❌ Ignore token expiration

### 2. Performance

**DO:**
- ✅ Reuse access tokens until expiration
- ✅ Implement connection pooling
- ✅ Use compression for large payloads
- ✅ Cache responses when appropriate
- ✅ Handle rate limits gracefully

**DON'T:**
- ❌ Request new token for every API call
- ❌ Make unnecessary requests
- ❌ Ignore retry-after headers

### 3. Error Handling

**DO:**
- ✅ Log request IDs for debugging
- ✅ Implement exponential backoff for retries
- ✅ Handle network timeouts
- ✅ Validate responses before using
- ✅ Monitor error rates

**DON'T:**
- ❌ Retry immediately on errors
- ❌ Ignore error details
- ❌ Hard fail on temporary errors

## Language-Specific Examples

### Python

```python
import requests

class AgentCoreClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.access_token = None

    def authenticate(self, username, password):
        response = requests.post(
            f"{self.base_url}/auth/token",
            json={
                "grant_type": "password",
                "username": username,
                "password": password,
            }
        )
        response.raise_for_status()
        data = response.json()
        self.access_token = data["access_token"]
        return data

    def get_current_user(self):
        response = requests.get(
            f"{self.base_url}/auth/me",
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        response.raise_for_status()
        return response.json()

# Usage
client = AgentCoreClient()
client.authenticate("user", "user123")
user = client.get_current_user()
print(f"Logged in as: {user['username']}")
```

### JavaScript/TypeScript

```typescript
class AgentCoreClient {
  private baseUrl: string;
  private accessToken: string | null = null;

  constructor(baseUrl: string = "http://localhost:8080") {
    this.baseUrl = baseUrl;
  }

  async authenticate(username: string, password: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/auth/token`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        grant_type: "password",
        username,
        password,
      }),
    });

    if (!response.ok) {
      throw new Error(`Authentication failed: ${response.statusText}`);
    }

    const data = await response.json();
    this.accessToken = data.access_token;
  }

  async getCurrentUser(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/auth/me`, {
      headers: { Authorization: `Bearer ${this.accessToken}` },
    });

    if (!response.ok) {
      throw new Error(`Request failed: ${response.statusText}`);
    }

    return response.json();
  }
}

// Usage
const client = new AgentCoreClient();
await client.authenticate("user", "user123");
const user = await client.getCurrentUser();
console.log(`Logged in as: ${user.username}`);
```

### Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
)

type AgentCoreClient struct {
    BaseURL     string
    AccessToken string
}

func NewClient(baseURL string) *AgentCoreClient {
    return &AgentCoreClient{BaseURL: baseURL}
}

func (c *AgentCoreClient) Authenticate(username, password string) error {
    body, _ := json.Marshal(map[string]string{
        "grant_type": "password",
        "username":   username,
        "password":   password,
    })

    resp, err := http.Post(
        c.BaseURL+"/auth/token",
        "application/json",
        bytes.NewBuffer(body),
    )
    if err != nil {
        return err
    }
    defer resp.Body.Close()

    var result map[string]interface{}
    json.NewDecoder(resp.Body).Decode(&result)
    c.AccessToken = result["access_token"].(string)
    return nil
}

func (c *AgentCoreClient) GetCurrentUser() (map[string]interface{}, error) {
    req, _ := http.NewRequest("GET", c.BaseURL+"/auth/me", nil)
    req.Header.Set("Authorization", "Bearer "+c.AccessToken)

    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var user map[string]interface{}
    json.NewDecoder(resp.Body).Decode(&user)
    return user, nil
}

func main() {
    client := NewClient("http://localhost:8080")
    client.Authenticate("user", "user123")
    user, _ := client.GetCurrentUser()
    fmt.Printf("Logged in as: %s\n", user["username"])
}
```

## Next Steps

### 1. Explore API Documentation

- **Interactive Docs:** http://localhost:8080/docs (Swagger UI)
- **Alternative Docs:** http://localhost:8080/redoc (ReDoc)
- **OpenAPI Spec:** http://localhost:8080/openapi.json

### 2. Generate SDK

Generate a client library in your preferred language:

```bash
./scripts/generate-sdk.sh python
./scripts/generate-sdk.sh typescript-axios
./scripts/generate-sdk.sh java
```

See [SDK_GENERATION.md](./SDK_GENERATION.md) for details.

### 3. Advanced Topics

- **WebSocket Communication:** Real-time agent monitoring
- **Server-Sent Events:** Live status updates
- **Rate Limiting Strategies:** Optimize API usage
- **Circuit Breakers:** Handle backend failures gracefully

### 4. Production Deployment

- **Security Hardening:** TLS 1.3, security headers
- **Monitoring:** Prometheus metrics, health checks
- **High Availability:** Load balancing, auto-scaling
- **Performance:** Connection pooling, caching

## Support

**Need Help?**
- **Documentation:** https://docs.agentcore.ai
- **API Reference:** http://localhost:8080/docs
- **GitHub Issues:** https://github.com/agentcore/agentcore/issues
- **Email Support:** api-support@agentcore.ai

**Found a Bug?**
Please open an issue on GitHub with:
- API endpoint and method
- Request/response examples
- Expected vs actual behavior
- Request ID from error response

## Quick Reference

### Base URLs

- **Local Development:** http://localhost:8080
- **Development:** https://api-dev.agentcore.ai
- **Staging:** https://api-staging.agentcore.ai
- **Production:** https://api.agentcore.ai

### Authentication

```bash
# Get token
POST /auth/token

# Refresh token
POST /auth/refresh

# Logout
POST /auth/logout

# Get current user
GET /auth/me

# List sessions
GET /auth/sessions
```

### OAuth

```bash
# List providers
GET /oauth/providers

# Start OAuth flow
GET /oauth/authorize/{provider}

# List scopes
GET /oauth/scopes
```

### Health

```bash
# Health check
GET /health

# Readiness
GET /ready

# Liveness
GET /live
```

---

**Happy Building!** 🚀

For more examples and advanced usage, check out our [API Documentation](http://localhost:8080/docs).
