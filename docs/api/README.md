# AgentCore API Documentation

Complete documentation for AgentCore API Gateway including guides, examples, and reference materials.

## Quick Links

- **[Getting Started](getting-started.md)** - 10-minute quick start guide
- **[Interactive API Docs](http://localhost:8001/docs)** - Swagger UI for testing endpoints
- **[API Reference](http://localhost:8001/redoc)** - Complete API reference documentation

## Documentation Structure

### Guides

Comprehensive guides covering all aspects of the API:

- **[Getting Started](getting-started.md)** - Quick start guide for new developers
- **[Authentication](authentication.md)** - JWT, OAuth, and SSO authentication
- **[Rate Limiting](rate-limiting.md)** - Rate limiting policies and best practices
- **[Real-time Communication](realtime.md)** - WebSocket and SSE usage
- **[Error Handling](errors.md)** - Error codes and troubleshooting

### Code Examples

Ready-to-use client examples in multiple languages:

- **[Python Client](../examples/python-client.py)** - Full-featured Python SDK with retry logic
- **[JavaScript Client](../examples/javascript-client.js)** - Browser and Node.js compatible
- **[cURL Examples](../examples/curl-examples.sh)** - Command-line examples for all endpoints

### API Reference

- **[OpenAPI Specification](http://localhost:8001/openapi.json)** - Machine-readable API spec
- **[Swagger UI](http://localhost:8001/docs)** - Interactive API explorer
- **[ReDoc](http://localhost:8001/redoc)** - Alternative documentation viewer

## Features

### Authentication & Authorization

- **JWT Tokens:** RS256-signed tokens with automatic refresh
- **OAuth 2.0:** Integration with Google, GitHub, Microsoft
- **Enterprise SSO:** SAML 2.0 and LDAP support
- **RBAC:** Role-based access control with scopes
- **Session Management:** Track and manage active sessions

### Performance & Scalability

- **60,000+ req/sec:** High-performance async architecture
- **<5ms latency:** p95 routing overhead
- **10,000+ connections:** Concurrent WebSocket connections
- **Horizontal scaling:** Stateless design for linear scaling
- **Connection pooling:** Optimized resource utilization

### Security

- **Rate Limiting:** Per IP, user, and endpoint limits
- **DDoS Protection:** Automatic IP blocking with burst detection
- **Input Validation:** Protection against XSS, SQL injection, path traversal
- **Security Headers:** HSTS, CSP, X-Frame-Options, etc.
- **TLS 1.3:** Modern encryption standards

### Real-time Communication

- **WebSocket:** Bidirectional communication for interactive apps
- **Server-Sent Events:** Server-to-client streaming for updates
- **Event Subscriptions:** Subscribe to specific event types
- **Automatic Reconnection:** Client-side reconnection logic
- **Heartbeat Monitoring:** Keep connections alive

### Observability

- **Prometheus Metrics:** Request rates, latency, error rates
- **Structured Logging:** JSON logs with request correlation
- **Distributed Tracing:** Request tracking across services
- **Health Checks:** Liveness and readiness probes
- **Rate Limit Headers:** Real-time limit information

### Developer Experience

- **OpenAPI 3.0:** Auto-generated API documentation
- **Interactive Docs:** Test endpoints directly from browser
- **Code Examples:** Python, JavaScript, cURL examples
- **<10min Onboarding:** Quick start guide for rapid integration
- **Clear Errors:** Actionable error messages with request IDs

## Getting Started

### Prerequisites

- Python 3.12+ or Node.js 18+ (for client examples)
- HTTP client (curl, Postman, or browser)
- Basic REST API knowledge

### 1. Quick Start (3 minutes)

```bash
# Authenticate
curl -X POST "http://localhost:8001/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"grant_type":"password","username":"user","password":"user123"}'

# Save the access_token from response

# Make authenticated request
curl -X GET "http://localhost:8001/auth/me" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### 2. Explore Interactive Docs

Visit http://localhost:8001/docs to:
- Browse all available endpoints
- Test API calls directly from browser
- View request/response examples
- See authentication requirements

### 3. Choose Your Client

Download pre-built clients:

```bash
# Python
curl -O https://raw.githubusercontent.com/agentcore/agentcore/main/docs/examples/python-client.py

# JavaScript
curl -O https://raw.githubusercontent.com/agentcore/agentcore/main/docs/examples/javascript-client.js

# cURL
curl -O https://raw.githubusercontent.com/agentcore/agentcore/main/docs/examples/curl-examples.sh
```

## Core Concepts

### Authentication Flow

1. **Obtain Token:** POST `/auth/token` with credentials
2. **Use Token:** Include in `Authorization: Bearer TOKEN` header
3. **Refresh Token:** Use refresh token before expiration
4. **Session Management:** Track and revoke sessions

### Rate Limiting

All endpoints are rate limited:
- **Per IP:** 1,000 requests/minute
- **Per User:** 5,000 requests/minute (authenticated)
- **Per Endpoint:** 100 requests/minute

Check `X-RateLimit-*` headers for current usage.

### Error Handling

All errors follow standard format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {},
    "request_id": "uuid",
    "timestamp": "ISO8601"
  }
}
```

Use `request_id` when contacting support.

## API Endpoints

### Authentication

- `POST /auth/token` - Obtain JWT tokens
- `POST /auth/refresh` - Refresh access token
- `GET /auth/me` - Get current user
- `GET /auth/sessions` - List active sessions
- `DELETE /auth/sessions/{id}` - Revoke session
- `POST /auth/logout` - Logout and invalidate session

### OAuth 2.0

- `GET /oauth/authorize/{provider}` - Initiate OAuth flow
- `GET /oauth/callback/{provider}` - OAuth callback
- `GET /oauth/providers` - List available providers
- `GET /oauth/scopes` - List available scopes
- `POST /oauth/token/client_credentials` - Client credentials flow

### Health & Monitoring

- `GET /health` - Health check
- `GET /health/ready` - Readiness check
- `GET /metrics` - Prometheus metrics

### Real-time

- `GET /realtime/ws` - WebSocket connection
- `GET /realtime/sse` - Server-Sent Events stream

## Demo Credentials

For development and testing:

| Username | Password | Roles |
|----------|----------|-------|
| `admin` | `admin123` | admin, user |
| `user` | `user123` | user |
| `service` | `service123` | service |

**Warning:** Development credentials only. Never use in production.

## Best Practices

### 1. Token Management

```python
# Check expiration and refresh
if token_expires_soon():
    refresh_token()
```

### 2. Rate Limit Handling

```python
# Check headers and backoff
remaining = response.headers.get('X-RateLimit-Remaining')
if int(remaining) < 10:
    implement_backoff()
```

### 3. Error Handling

```python
# Extract request_id for support
if response.status_code >= 500:
    request_id = response.json()['error']['request_id']
    log_error(request_id)
```

### 4. Retry Logic

```python
# Exponential backoff
@retry(stop=stop_after_attempt(3))
def api_call():
    return client.get('/endpoint')
```

### 5. Security

- Never store tokens in localStorage
- Always use HTTPS in production
- Implement token rotation
- Monitor active sessions

## Troubleshooting

### Common Issues

**401 Unauthorized**
- Token expired → Use refresh token
- Invalid token → Re-authenticate
- Missing header → Add Authorization header

**429 Too Many Requests**
- Rate limit exceeded → Wait for X-RateLimit-Reset
- Too many requests → Implement client-side rate limiting

**503 Service Unavailable**
- Service down → Check /health endpoint
- Maintenance → Check status page
- Overload → Implement retry with backoff

## Support

### Resources

- **Documentation:** https://docs.agentcore.ai
- **API Reference:** http://localhost:8001/redoc
- **Interactive Docs:** http://localhost:8001/docs
- **OpenAPI Spec:** http://localhost:8001/openapi.json

### Contact

- **Email:** api-support@agentcore.ai
- **Discord:** https://discord.gg/agentcore
- **GitHub Issues:** https://github.com/agentcore/agentcore/issues
- **Status Page:** https://status.agentcore.ai

## Contributing

Found an issue or want to contribute?

1. Check [existing issues](https://github.com/agentcore/agentcore/issues)
2. Open a new issue with details
3. Submit a pull request with fix

## License

AgentCore is licensed under AGPL-3.0. See [LICENSE](../../LICENSE) for details.

## Changelog

### Version 0.1.0 (2025-10-18)

**Initial Release:**
- JWT authentication with RS256 signing
- OAuth 2.0 integration (Google, GitHub, Microsoft)
- Rate limiting with multiple algorithms
- WebSocket and SSE support
- DDoS protection
- Comprehensive OpenAPI documentation
- Python, JavaScript, and cURL examples

---

**Last Updated:** 2025-10-18
**API Version:** 0.1.0
**Documentation Version:** 1.0.0
