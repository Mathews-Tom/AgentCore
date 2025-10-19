# Error Handling Guide

AgentCore API Gateway uses standardized error responses with actionable guidance.

## Error Response Format

All errors follow a consistent JSON structure:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "additional": "context-specific information"
    },
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2025-10-18T10:30:00Z"
  }
}
```

## HTTP Status Codes

| Status | Meaning | Description |
|--------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 204 | No Content | Request successful, no content returned |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required or invalid |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource conflict (duplicate, etc.) |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Unexpected server error |
| 502 | Bad Gateway | Backend service error |
| 503 | Service Unavailable | Service temporarily unavailable |

## Error Codes

### Authentication Errors (401)

#### UNAUTHORIZED

Missing or invalid authentication credentials.

```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Authentication required",
    "details": {
      "reason": "Missing or invalid authorization header"
    }
  }
}
```

**Solution:** Include valid Bearer token in Authorization header.

#### TOKEN_EXPIRED

JWT token has expired.

```json
{
  "error": {
    "code": "TOKEN_EXPIRED",
    "message": "Access token has expired",
    "details": {
      "expired_at": "2025-10-18T09:30:00Z"
    }
  }
}
```

**Solution:** Use refresh token to obtain new access token.

#### INVALID_TOKEN

JWT token is malformed or signature is invalid.

```json
{
  "error": {
    "code": "INVALID_TOKEN",
    "message": "Token validation failed",
    "details": {
      "reason": "Invalid signature"
    }
  }
}
```

**Solution:** Obtain a new token from `/auth/token` endpoint.

### Authorization Errors (403)

#### FORBIDDEN

User authenticated but lacks required permissions.

```json
{
  "error": {
    "code": "FORBIDDEN",
    "message": "Insufficient permissions",
    "details": {
      "required_roles": ["admin"],
      "user_roles": ["user"]
    }
  }
}
```

**Solution:** Request appropriate roles or contact administrator.

#### INSUFFICIENT_SCOPE

OAuth token lacks required scopes.

```json
{
  "error": {
    "code": "INSUFFICIENT_SCOPE",
    "message": "Token does not have required scopes",
    "details": {
      "required_scopes": ["agent:write"],
      "token_scopes": ["agent:read"]
    }
  }
}
```

**Solution:** Re-authorize with required scopes.

### Validation Errors (400)

#### VALIDATION_ERROR

Request validation failed.

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "field": "grant_type",
      "issue": "Value must be one of: password, client_credentials"
    }
  }
}
```

**Solution:** Fix request parameters according to error details.

#### INVALID_GRANT

OAuth grant validation failed.

```json
{
  "error": {
    "code": "INVALID_GRANT",
    "message": "Invalid grant credentials",
    "details": {
      "grant_type": "password",
      "reason": "Invalid username or password"
    }
  }
}
```

**Solution:** Verify credentials and retry.

### Rate Limiting Errors (429)

#### RATE_LIMIT_EXCEEDED

Request rate limit exceeded.

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 1000,
      "window": 60,
      "retry_after": 30,
      "policy": "client_ip"
    }
  }
}
```

**Solution:** Wait for `retry_after` seconds before retrying.

### Resource Errors (404)

#### NOT_FOUND

Requested resource does not exist.

```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "Resource not found",
    "details": {
      "resource": "session",
      "id": "invalid-session-id"
    }
  }
}
```

**Solution:** Verify resource ID and retry.

### Server Errors (500-503)

#### INTERNAL_ERROR

Unexpected server error.

```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An unexpected error occurred",
    "details": {
      "trace_id": "770e8400-e29b-41d4-a716-446655440000"
    }
  }
}
```

**Solution:** Report error with `request_id` and `trace_id` to support.

#### SERVICE_UNAVAILABLE

Service temporarily unavailable.

```json
{
  "error": {
    "code": "SERVICE_UNAVAILABLE",
    "message": "Service temporarily unavailable",
    "details": {
      "service": "authentication",
      "retry_after": 60
    }
  }
}
```

**Solution:** Retry after specified delay with exponential backoff.

## Error Handling Best Practices

### 1. Always Check Status Codes

```python
import requests

response = requests.post(url, json=data)

if response.status_code == 200:
    # Success
    result = response.json()
elif response.status_code == 401:
    # Refresh token and retry
    refresh_auth()
    response = requests.post(url, json=data)
elif response.status_code == 429:
    # Rate limited, backoff and retry
    retry_after = int(response.headers.get('Retry-After', 60))
    time.sleep(retry_after)
    response = requests.post(url, json=data)
else:
    # Log error
    error = response.json()
    logging.error(f"API error: {error}")
```

### 2. Extract Request ID for Support

```python
if response.status_code >= 500:
    error_data = response.json()
    request_id = error_data['error']['request_id']
    logging.error(f"Server error - Request ID: {request_id}")
    # Report to support with request_id
```

### 3. Implement Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def api_call_with_retry():
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
```

### 4. Handle Specific Error Codes

```python
def handle_api_error(response):
    if response.status_code < 400:
        return response.json()

    error = response.json()['error']
    code = error['code']

    if code == 'TOKEN_EXPIRED':
        # Refresh token
        refresh_token()
        return retry_request()
    elif code == 'RATE_LIMIT_EXCEEDED':
        # Backoff
        retry_after = error['details']['retry_after']
        time.sleep(retry_after)
        return retry_request()
    elif code == 'INSUFFICIENT_SCOPE':
        # Re-authorize with required scopes
        required_scopes = error['details']['required_scopes']
        reauthorize(required_scopes)
        return retry_request()
    else:
        raise Exception(f"API error: {error['message']}")
```

## Debugging

### Request ID Tracking

Every error includes a unique `request_id` for tracking:

```bash
curl -X GET "http://localhost:8080/auth/me" \
  -H "Authorization: Bearer invalid_token"
```

```json
{
  "error": {
    "code": "INVALID_TOKEN",
    "message": "Token validation failed",
    "request_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

Use `request_id` when contacting support for faster resolution.

### Trace ID for Distributed Tracing

Server errors include `trace_id` for distributed tracing:

```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "details": {
      "trace_id": "770e8400-e29b-41d4-a716-446655440000"
    }
  }
}
```

## Common Issues

### Issue: Constant 401 Errors

**Cause:** Expired or invalid token

**Solution:**
1. Check token expiration: `exp` claim in JWT
2. Refresh token using `/auth/refresh`
3. Re-authenticate if refresh token expired

### Issue: 403 on Previously Working Endpoint

**Cause:** User permissions changed or scope insufficient

**Solution:**
1. Check error details for required permissions
2. Verify user roles haven't changed
3. Re-authenticate with required scopes

### Issue: Intermittent 503 Errors

**Cause:** Service scaling or maintenance

**Solution:**
1. Implement retry with exponential backoff
2. Check status page for maintenance
3. Contact support if persistent

## Next Steps

- [Authentication Guide](authentication.md)
- [Rate Limiting Guide](rate-limiting.md)
- [Real-time Communication](realtime.md)
- [Code Examples](../examples/)
