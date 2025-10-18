# Getting Started with AgentCore API Gateway

Get up and running with AgentCore API Gateway in less than 10 minutes.

## Prerequisites

- Python 3.12+ or Node.js 18+ (for client examples)
- Basic understanding of REST APIs
- HTTP client (curl, Postman, or browser)

## Quick Start

### 1. Access the API

The API is available at:
- **Local Development:** http://localhost:8080
- **Production:** https://api.agentcore.ai

### 2. Authenticate

Get an access token using demo credentials:

```bash
curl -X POST "http://localhost:8080/auth/token" \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "password",
    "username": "user",
    "password": "user123"
  }'
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "scope": null
}
```

**Copy the `access_token` for the next step.**

### 3. Make Your First API Call

Get your user information:

```bash
curl -X GET "http://localhost:8080/auth/me" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Response:**

```json
{
  "id": "00000000-0000-0000-0000-000000000002",
  "username": "user",
  "email": "user@agentcore.ai",
  "roles": ["user"],
  "is_active": true,
  "metadata": {},
  "created_at": "2025-10-01T10:00:00Z",
  "updated_at": "2025-10-18T10:30:00Z"
}
```

**Congratulations! You've made your first API call.**

## Next Steps

### Explore the Interactive API Docs

Visit http://localhost:8080/docs to explore all available endpoints in an interactive Swagger UI.

Features:
- Try API calls directly from browser
- See request/response examples
- View authentication requirements
- Test with your own credentials

### Choose Your Client Library

#### Python

```bash
# Download the example client
curl -O https://raw.githubusercontent.com/agentcore/agentcore/main/docs/examples/python-client.py

# Install dependencies
pip install requests

# Use the client
python
>>> from python_client import AgentCoreClient
>>> client = AgentCoreClient(username="user", password="user123")
>>> user = client.get_current_user()
>>> print(user["username"])
user
```

#### JavaScript/Node.js

```bash
# Download the example client
curl -O https://raw.githubusercontent.com/agentcore/agentcore/main/docs/examples/javascript-client.js

# Use with Node.js
node
> const AgentCoreClient = require('./javascript-client.js');
> const client = new AgentCoreClient({ username: 'user', password: 'user123' });
> client.getCurrentUser().then(user => console.log(user.username));
user
```

#### cURL

```bash
# Download examples
curl -O https://raw.githubusercontent.com/agentcore/agentcore/main/docs/examples/curl-examples.sh

# Run examples
bash curl-examples.sh
```

## Core Concepts

### Authentication

AgentCore supports multiple authentication methods:

1. **JWT Tokens** - Most common for API access
2. **OAuth 2.0** - For third-party integrations (Google, GitHub, Microsoft)
3. **Enterprise SSO** - SAML 2.0 and LDAP for enterprise deployments

**Learn more:** [Authentication Guide](authentication.md)

### Rate Limiting

All API endpoints are rate limited:
- **Per IP:** 1,000 requests/minute
- **Per User:** 5,000 requests/minute
- **Per Endpoint:** 100 requests/minute

Rate limit information is included in response headers.

**Learn more:** [Rate Limiting Guide](rate-limiting.md)

### Real-time Communication

For real-time updates, use WebSocket or Server-Sent Events (SSE):

```javascript
// WebSocket example
const ws = new WebSocket('ws://localhost:8080/realtime/ws?token=YOUR_TOKEN');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Event:', data);
};
```

**Learn more:** [Real-time Communication Guide](realtime.md)

### Error Handling

All errors follow a standard format:

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

**Learn more:** [Error Handling Guide](errors.md)

## Common Use Cases

### Use Case 1: User Authentication Flow

```python
from python_client import AgentCoreClient

# Create client
client = AgentCoreClient()

# Authenticate
tokens = client.authenticate('user', 'user123')
print(f"Access token: {tokens['access_token']}")

# Access protected resources
user = client.get_current_user()
print(f"Logged in as: {user['username']}")

# Logout
client.logout()
```

### Use Case 2: OAuth Social Login

```javascript
// Redirect user to OAuth provider
window.location.href = '/oauth/authorize/google?scope=user:read&redirect_after_login=/dashboard';

// After callback, user is authenticated and redirected to /dashboard
// Token is available in URL parameter or cookie
```

### Use Case 3: Service Account Integration

```bash
# Authenticate service account
curl -X POST "http://localhost:8080/auth/token" \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "client_credentials",
    "client_id": "service",
    "client_secret": "service123"
  }'

# Use token for automated operations
```

### Use Case 4: Real-time Monitoring

```javascript
const client = new AgentCoreClient({ username: 'user', password: 'user123' });

const ws = await client.connectWebSocket({
  onMessage: (data) => {
    if (data.event === 'agent.status_changed') {
      console.log(`Agent ${data.data.agent_id} status: ${data.data.status}`);
    }
  }
});

// Subscribe to events
ws.send(JSON.stringify({
  type: 'subscribe',
  channels: ['agent.status_changed', 'agent.task_completed']
}));
```

## Best Practices

### 1. Store Tokens Securely

**Never:**
- Store tokens in localStorage (vulnerable to XSS)
- Include tokens in URLs
- Commit tokens to version control

**Instead:**
- Use httpOnly cookies for web apps
- Use secure storage for mobile apps
- Use environment variables for server-side apps

### 2. Implement Token Refresh

```python
# Check token expiration and refresh
if datetime.utcnow() >= client.token_expires_at - timedelta(minutes=5):
    client.refresh_access_token()
```

### 3. Handle Rate Limits

```python
# Respect rate limit headers
remaining = response.headers.get('X-RateLimit-Remaining')
if int(remaining) < 10:
    print('Warning: Approaching rate limit')
    # Implement backoff
```

### 4. Implement Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def api_call():
    return client.get('/auth/me')
```

### 5. Use Structured Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    user = client.get_current_user()
    logger.info(f"User {user['username']} authenticated successfully")
except Exception as e:
    logger.error(f"API error: {e}")
```

## Demo Credentials

For testing and development:

| Username | Password | Roles |
|----------|----------|-------|
| `admin` | `admin123` | admin, user |
| `user` | `user123` | user |
| `service` | `service123` | service |

**Important:** These credentials are for development only. Never use them in production.

## Troubleshooting

### Issue: 401 Unauthorized

**Problem:** Token is invalid or expired

**Solution:**
```bash
# Refresh token
curl -X POST "http://localhost:8080/auth/refresh" \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "YOUR_REFRESH_TOKEN"}'
```

### Issue: 429 Too Many Requests

**Problem:** Rate limit exceeded

**Solution:**
```bash
# Check rate limit headers
curl -I -X GET "http://localhost:8080/auth/me" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Wait for X-RateLimit-Reset time
```

### Issue: Connection Refused

**Problem:** API gateway is not running

**Solution:**
```bash
# Check if service is running
curl http://localhost:8080/health

# Start the gateway
docker compose up -d
```

## Next Steps

Now that you're familiar with the basics:

1. **Explore Documentation:**
   - [Authentication Guide](authentication.md)
   - [Rate Limiting Guide](rate-limiting.md)
   - [Real-time Communication](realtime.md)
   - [Error Handling](errors.md)

2. **Try Examples:**
   - [Python Client](../examples/python-client.py)
   - [JavaScript Client](../examples/javascript-client.js)
   - [cURL Examples](../examples/curl-examples.sh)

3. **Interactive API Docs:**
   - Visit http://localhost:8080/docs
   - Try all endpoints in Swagger UI

4. **Build Your Integration:**
   - Choose your preferred language
   - Implement authentication
   - Add error handling
   - Deploy to production

## Support

- **Documentation:** https://docs.agentcore.ai
- **Support Email:** api-support@agentcore.ai
- **GitHub Issues:** https://github.com/agentcore/agentcore/issues
- **Community Discord:** https://discord.gg/agentcore

Welcome to AgentCore! Happy coding!
