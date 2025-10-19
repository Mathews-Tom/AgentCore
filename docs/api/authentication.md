# Authentication Guide

AgentCore API Gateway supports multiple authentication methods including JWT tokens, OAuth 2.0, and enterprise SSO integration.

## Table of Contents

- [JWT Authentication](#jwt-authentication)
- [OAuth 2.0 Authentication](#oauth-20-authentication)
- [Enterprise SSO](#enterprise-sso)
- [Session Management](#session-management)
- [Security Best Practices](#security-best-practices)

## JWT Authentication

### Getting Started

The simplest way to authenticate is using username/password to obtain a JWT token:

```bash
curl -X POST "http://localhost:8080/auth/token" \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "password",
    "username": "user",
    "password": "user123",
    "scope": "user:read user:write"
  }'
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "scope": "user:read user:write"
}
```

### Using the Access Token

Include the access token in the `Authorization` header of subsequent requests:

```bash
curl -X GET "http://localhost:8080/auth/me" \
  -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### Token Lifecycle

- **Access Token Lifetime:** 60 minutes
- **Refresh Token Lifetime:** 7 days
- **Token Format:** RS256-signed JWT

### Refreshing Tokens

Before the access token expires, use the refresh token to obtain a new one:

```bash
curl -X POST "http://localhost:8080/auth/refresh" \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."
  }'
```

### Service Account Authentication

For machine-to-machine authentication, use the client credentials flow:

```bash
curl -X POST "http://localhost:8080/auth/token" \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "client_credentials",
    "client_id": "service",
    "client_secret": "service123",
    "scope": "service:read service:write"
  }'
```

## OAuth 2.0 Authentication

OAuth 2.0 provides third-party authentication through popular providers.

### Supported Providers

- Google OAuth 2.0
- GitHub OAuth 2.0
- Microsoft OAuth 2.0 (Azure AD)

### Authorization Flow

#### Step 1: Initiate OAuth Flow

Redirect the user to the OAuth authorization endpoint:

```
GET /oauth/authorize/{provider}?scope=user:read&redirect_after_login=/dashboard
```

Example:

```
https://api.agentcore.ai/oauth/authorize/google?scope=user:read user:write&redirect_after_login=/dashboard
```

#### Step 2: User Authorization

The user is redirected to the OAuth provider (Google, GitHub, etc.) to authorize the application.

#### Step 3: Callback

After authorization, the provider redirects back to:

```
GET /oauth/callback/{provider}?code=AUTHORIZATION_CODE&state=STATE
```

The gateway automatically:
1. Validates the state parameter (CSRF protection)
2. Exchanges the authorization code for tokens
3. Retrieves user information
4. Creates a session and JWT token
5. Redirects to the application with the access token

#### Step 4: Access Protected Resources

Use the JWT token as described in the JWT Authentication section.

### PKCE Support

All OAuth flows support PKCE (Proof Key for Code Exchange) for enhanced security. The gateway automatically generates and validates PKCE challenges.

### Available Scopes

OAuth scopes control what permissions are granted:

| Scope | Description |
|-------|-------------|
| `user:read` | Read user information |
| `user:write` | Modify user information |
| `agent:read` | Read agent information |
| `agent:write` | Create and modify agents |
| `agent:execute` | Execute agent workflows |
| `task:read` | Read task information |
| `task:write` | Create and modify tasks |
| `admin:read` | Read administrative information |
| `admin:write` | Perform administrative operations |

### List Available Providers

```bash
curl -X GET "http://localhost:8080/oauth/providers"
```

**Response:**

```json
{
  "providers": [
    {
      "provider": "google",
      "name": "Google",
      "authorize_url": "/oauth/authorize/google"
    },
    {
      "provider": "github",
      "name": "Github",
      "authorize_url": "/oauth/authorize/github"
    }
  ]
}
```

## Enterprise SSO

AgentCore supports enterprise Single Sign-On (SSO) through SAML 2.0 and LDAP.

### SAML 2.0 Configuration

Contact your administrator to configure SAML integration with your identity provider.

### LDAP Authentication

LDAP authentication is configured through environment variables:

```bash
GATEWAY_SSO_LDAP_ENABLED=true
GATEWAY_LDAP_SERVER_URI=ldaps://ldap.example.com
GATEWAY_LDAP_BIND_DN=cn=admin,dc=example,dc=com
GATEWAY_LDAP_BIND_PASSWORD=secret
GATEWAY_LDAP_BASE_DN=dc=example,dc=com
```

## Session Management

### List Active Sessions

View all active sessions for the current user:

```bash
curl -X GET "http://localhost:8080/auth/sessions" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Response:**

```json
{
  "sessions": [
    {
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "created_at": "2025-10-18T10:00:00Z",
      "expires_at": "2025-10-19T10:00:00Z",
      "last_activity": "2025-10-18T10:30:00Z",
      "ip_address": "192.168.1.100",
      "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }
  ]
}
```

### Delete Session

Revoke a specific session:

```bash
curl -X DELETE "http://localhost:8080/auth/sessions/{session_id}" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Logout

Logout and invalidate the current session:

```bash
curl -X POST "http://localhost:8080/auth/logout" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## Security Best Practices

### Token Storage

- **Never** store tokens in localStorage (vulnerable to XSS)
- Use httpOnly cookies for web applications
- Store tokens in secure storage for mobile apps
- Use environment variables or secure vaults for server-side applications

### Token Transmission

- Always use HTTPS/TLS for token transmission
- Never include tokens in URL query parameters
- Use the `Authorization` header for bearer tokens

### Token Rotation

- Implement automatic token refresh before expiration
- Rotate refresh tokens on use
- Invalidate old tokens after refresh

### Session Security

- Monitor active sessions regularly
- Revoke sessions from unknown devices
- Implement session timeout and inactivity policies

### CSRF Protection

- OAuth flows include automatic state validation
- Verify state parameters in callbacks
- Use PKCE for enhanced security

### Rate Limiting

- Token endpoints are rate limited to prevent brute force attacks
- Failed authentication attempts count against rate limits
- Implement exponential backoff for retries

## Error Handling

### Common Authentication Errors

#### 401 Unauthorized

```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or expired token",
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2025-10-18T10:30:00Z"
  }
}
```

**Solution:** Obtain a new token or refresh the existing one.

#### 403 Forbidden

```json
{
  "error": {
    "code": "FORBIDDEN",
    "message": "Insufficient permissions",
    "details": {
      "required_roles": ["admin"],
      "user_roles": ["user"]
    },
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2025-10-18T10:30:00Z"
  }
}
```

**Solution:** Request appropriate scopes/roles or contact administrator.

#### 429 Too Many Requests

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many authentication attempts",
    "details": {
      "retry_after": 60
    },
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2025-10-18T10:30:00Z"
  }
}
```

**Solution:** Wait for the specified `retry_after` period before retrying.

## Demo Credentials

For development and testing purposes:

| Username | Password | Roles |
|----------|----------|-------|
| `admin` | `admin123` | admin, user |
| `user` | `user123` | user |
| `service` | `service123` | service |

**Warning:** These credentials are for development only. Never use them in production.

## Next Steps

- [Rate Limiting Guide](rate-limiting.md)
- [Real-time Communication](realtime.md)
- [Error Handling](errors.md)
- [Code Examples](../examples/)
