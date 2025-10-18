"""
OpenAPI metadata and configuration for comprehensive API documentation.

Provides tags, descriptions, security schemes, and external documentation links.
"""

from __future__ import annotations

# OpenAPI tags for endpoint grouping
OPENAPI_TAGS = [
    {
        "name": "health",
        "description": "Health check and system status endpoints. No authentication required.",
    },
    {
        "name": "authentication",
        "description": """
JWT-based authentication endpoints for token generation, refresh, and session management.

**Supported Grant Types:**
- `password`: Username/password authentication for user accounts
- `client_credentials`: Service account authentication for machine-to-machine
- `refresh_token`: Token refresh for long-lived sessions

**Security:** All endpoints except `/auth/token` require Bearer token authentication.
        """,
    },
    {
        "name": "oauth",
        "description": """
OAuth 2.0/3.0 authentication flows with enterprise SSO support.

**Supported Providers:**
- Google OAuth 2.0
- GitHub OAuth 2.0
- Microsoft OAuth 2.0 (Azure AD)
- Enterprise SAML 2.0
- Enterprise LDAP

**Features:**
- PKCE (Proof Key for Code Exchange) for enhanced security
- Automatic state management and CSRF protection
- Scope-based authorization
- Session management with JWT tokens
        """,
    },
    {
        "name": "realtime",
        "description": """
Real-time communication endpoints for WebSocket and Server-Sent Events (SSE).

**WebSocket:** Bidirectional communication for agent monitoring and control
**SSE:** Server-to-client streaming for real-time updates and notifications

**Use Cases:**
- Real-time agent status updates
- Task execution progress tracking
- System event notifications
- Live log streaming

**Authentication:** Requires valid JWT token via query parameter or Authorization header.
        """,
    },
]

# OpenAPI license and contact information
OPENAPI_LICENSE = {
    "name": "Apache 2.0",
    "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
}

OPENAPI_CONTACT = {
    "name": "AgentCore API Support",
    "url": "https://agentcore.ai/support",
    "email": "api-support@agentcore.ai",
}

# External documentation links
OPENAPI_EXTERNAL_DOCS = {
    "description": "Complete AgentCore Documentation",
    "url": "https://docs.agentcore.ai",
}

# OpenAPI description with markdown formatting
OPENAPI_DESCRIPTION = """
# AgentCore API Gateway

High-performance API gateway for AgentCore providing unified entry point for all external interactions with the agentic AI platform.

## Features

- **High Performance:** 60,000+ requests/second with <5ms routing latency
- **Enterprise Security:** JWT/OAuth authentication, rate limiting, DDoS protection
- **Real-time Communication:** WebSocket and SSE support for 10,000+ concurrent connections
- **Comprehensive Middleware:** CORS, validation, compression, security headers
- **Developer Experience:** Auto-generated OpenAPI docs, code examples, interactive explorer

## Quick Start

### 1. Authentication

Get an access token using username/password:

```bash
curl -X POST "http://localhost:8080/auth/token" \\
  -H "Content-Type: application/json" \\
  -d '{
    "grant_type": "password",
    "username": "user",
    "password": "user123"
  }'
```

### 2. Use Token

Include the token in subsequent requests:

```bash
curl -X GET "http://localhost:8080/auth/me" \\
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### 3. OAuth Login

For OAuth authentication, redirect users to:

```
GET /oauth/authorize/{provider}?scope=user:read&redirect_after_login=/dashboard
```

## Rate Limiting

All API endpoints are rate limited to ensure fair usage:

- **Per IP:** 1,000 requests/minute
- **Per User:** 5,000 requests/minute
- **Per Endpoint:** 100 requests/minute

Rate limit information is included in response headers:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Time when rate limit resets (Unix timestamp)

## Error Handling

All errors follow a consistent format:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 1000,
      "window": 60,
      "retry_after": 30
    },
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2025-10-18T10:30:00Z"
  }
}
```

## Versioning

API versioning is implemented through URL paths:
- Current version: `/api/v1/`
- Legacy support: Previous versions maintained for 6 months

## Support

- **Documentation:** https://docs.agentcore.ai
- **Support Email:** api-support@agentcore.ai
- **GitHub Issues:** https://github.com/agentcore/agentcore/issues
"""

# Security schemes for OpenAPI
OPENAPI_SECURITY_SCHEMES = {
    "BearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": """
JWT Bearer token authentication.

**How to get a token:**
1. Call `POST /auth/token` with username/password
2. Or use OAuth flow via `GET /oauth/authorize/{provider}`
3. Include token in Authorization header: `Authorization: Bearer YOUR_TOKEN`

**Token Lifetime:**
- Access tokens expire in 60 minutes
- Refresh tokens expire in 7 days
- Use `POST /auth/refresh` to get new access token

**Token Format:**
Tokens are RS256-signed JWTs containing:
- `sub`: User ID
- `username`: Username
- `roles`: User roles for authorization
- `session_id`: Session identifier
- `scope`: OAuth scopes (if applicable)
- `exp`: Expiration timestamp
        """,
    },
    "OAuth2": {
        "type": "oauth2",
        "description": "OAuth 2.0 authentication with multiple providers",
        "flows": {
            "authorizationCode": {
                "authorizationUrl": "/oauth/authorize/{provider}",
                "tokenUrl": "/oauth/callback/{provider}",
                "scopes": {
                    "user:read": "Read user information",
                    "user:write": "Modify user information",
                    "agent:read": "Read agent information",
                    "agent:write": "Create and modify agents",
                    "agent:execute": "Execute agent workflows",
                    "task:read": "Read task information",
                    "task:write": "Create and modify tasks",
                    "admin:read": "Read administrative information",
                    "admin:write": "Perform administrative operations",
                },
            },
            "clientCredentials": {
                "tokenUrl": "/oauth/token/client_credentials",
                "scopes": {
                    "service:read": "Service account read access",
                    "service:write": "Service account write access",
                },
            },
        },
    },
}

# Servers configuration
OPENAPI_SERVERS = [
    {
        "url": "http://localhost:8080",
        "description": "Local development server",
    },
    {
        "url": "https://api-dev.agentcore.ai",
        "description": "Development environment",
    },
    {
        "url": "https://api-staging.agentcore.ai",
        "description": "Staging environment",
    },
    {
        "url": "https://api.agentcore.ai",
        "description": "Production environment",
    },
]
