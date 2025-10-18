"""
OpenAPI examples for request/response models.

Provides realistic examples for API documentation and interactive testing.
"""

from __future__ import annotations

# Authentication examples
TOKEN_REQUEST_EXAMPLES = {
    "password_grant": {
        "summary": "Username/Password Authentication",
        "description": "Standard user authentication with username and password",
        "value": {
            "grant_type": "password",
            "username": "user",
            "password": "user123",
            "scope": "user:read user:write agent:read agent:execute",
        },
    },
    "client_credentials": {
        "summary": "Service Account Authentication",
        "description": "Machine-to-machine authentication for service accounts",
        "value": {
            "grant_type": "client_credentials",
            "client_id": "service",
            "client_secret": "service123",
            "scope": "service:read service:write",
        },
    },
    "admin_login": {
        "summary": "Administrator Login",
        "description": "Admin user authentication with full permissions",
        "value": {
            "grant_type": "password",
            "username": "admin",
            "password": "admin123",
            "scope": "admin:read admin:write",
        },
    },
}

TOKEN_RESPONSE_EXAMPLE = {
    "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIwMDAwMDAwMC0wMDAwLTAwMDAtMDAwMC0wMDAwMDAwMDAwMDIiLCJ1c2VybmFtZSI6InVzZXIiLCJyb2xlcyI6WyJ1c2VyIl0sInNlc3Npb25faWQiOiI1NTBlODQwMC1lMjliLTQxZDQtYTcxNi00NDY2NTU0NDAwMDAiLCJpYXQiOjE3MjkzMjAwMDAsImV4cCI6MTcyOTMyMzYwMCwic2NvcGUiOiJ1c2VyOnJlYWQgdXNlcjp3cml0ZSIsImp0aSI6Ijc3MGU4NDAwLWUyOWItNDFkNC1hNzE2LTQ0NjY1NTQ0MDAwMCJ9.signature",
    "token_type": "Bearer",
    "expires_in": 3600,
    "refresh_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIwMDAwMDAwMC0wMDAwLTAwMDAtMDAwMC0wMDAwMDAwMDAwMDIiLCJzZXNzaW9uX2lkIjoiNTUwZTg0MDAtZTI5Yi00MWQ0LWE3MTYtNDQ2NjU1NDQwMDAwIiwiaWF0IjoxNzI5MzIwMDAwLCJleHAiOjE3Mjk5MjQ4MDAsImp0aSI6Ijg4MGU4NDAwLWUyOWItNDFkNC1hNzE2LTQ0NjY1NTQ0MDAwMCIsInRva2VuX3R5cGUiOiJyZWZyZXNoIn0.signature",
    "scope": "user:read user:write",
}

USER_EXAMPLE = {
    "id": "00000000-0000-0000-0000-000000000002",
    "username": "user",
    "email": "user@agentcore.ai",
    "roles": ["user"],
    "is_active": True,
    "metadata": {},
    "created_at": "2025-10-01T10:00:00Z",
    "updated_at": "2025-10-18T10:30:00Z",
}

SESSION_LIST_EXAMPLE = {
    "sessions": [
        {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "created_at": "2025-10-18T10:00:00Z",
            "expires_at": "2025-10-19T10:00:00Z",
            "last_activity": "2025-10-18T10:30:00Z",
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        },
        {
            "session_id": "660e8400-e29b-41d4-a716-446655440000",
            "created_at": "2025-10-17T14:00:00Z",
            "expires_at": "2025-10-18T14:00:00Z",
            "last_activity": "2025-10-18T09:45:00Z",
            "ip_address": "10.0.1.50",
            "user_agent": "AgentCore Python SDK/1.0.0",
        },
    ]
}

# OAuth examples
OAUTH_PROVIDERS_EXAMPLE = {
    "providers": [
        {
            "provider": "google",
            "name": "Google",
            "authorize_url": "/oauth/authorize/google",
        },
        {
            "provider": "github",
            "name": "Github",
            "authorize_url": "/oauth/authorize/github",
        },
        {
            "provider": "microsoft",
            "name": "Microsoft",
            "authorize_url": "/oauth/authorize/microsoft",
        },
    ]
}

OAUTH_SCOPES_EXAMPLE = {
    "scopes": [
        {
            "scope": "user:read",
            "description": "Read user information",
            "resource": "user",
        },
        {
            "scope": "user:write",
            "description": "Modify user information",
            "resource": "user",
        },
        {
            "scope": "agent:read",
            "description": "Read agent information",
            "resource": "agent",
        },
        {
            "scope": "agent:write",
            "description": "Create and modify agents",
            "resource": "agent",
        },
        {
            "scope": "agent:execute",
            "description": "Execute agent workflows",
            "resource": "agent",
        },
        {
            "scope": "task:read",
            "description": "Read task information",
            "resource": "task",
        },
        {
            "scope": "task:write",
            "description": "Create and modify tasks",
            "resource": "task",
        },
        {
            "scope": "admin:read",
            "description": "Read administrative information",
            "resource": "admin",
        },
        {
            "scope": "admin:write",
            "description": "Perform administrative operations",
            "resource": "admin",
        },
    ]
}

# Error response examples
ERROR_RESPONSE_EXAMPLES = {
    "unauthorized": {
        "summary": "Unauthorized - Missing or Invalid Token",
        "description": "Authentication required but token is missing or invalid",
        "value": {
            "error": {
                "code": "UNAUTHORIZED",
                "message": "Authentication required",
                "details": {
                    "reason": "Missing or invalid authorization header",
                },
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2025-10-18T10:30:00Z",
            }
        },
    },
    "forbidden": {
        "summary": "Forbidden - Insufficient Permissions",
        "description": "User authenticated but lacks required permissions",
        "value": {
            "error": {
                "code": "FORBIDDEN",
                "message": "Insufficient permissions",
                "details": {
                    "required_roles": ["admin"],
                    "user_roles": ["user"],
                },
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2025-10-18T10:30:00Z",
            }
        },
    },
    "rate_limit": {
        "summary": "Rate Limit Exceeded",
        "description": "Too many requests from client",
        "value": {
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Rate limit exceeded",
                "details": {
                    "limit": 1000,
                    "window": 60,
                    "retry_after": 30,
                },
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2025-10-18T10:30:00Z",
            }
        },
    },
    "validation_error": {
        "summary": "Validation Error",
        "description": "Request validation failed",
        "value": {
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {
                    "field": "grant_type",
                    "issue": "Value must be one of: password, client_credentials, refresh_token",
                },
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2025-10-18T10:30:00Z",
            }
        },
    },
    "not_found": {
        "summary": "Resource Not Found",
        "description": "Requested resource does not exist",
        "value": {
            "error": {
                "code": "NOT_FOUND",
                "message": "Resource not found",
                "details": {
                    "resource": "session",
                    "id": "invalid-session-id",
                },
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2025-10-18T10:30:00Z",
            }
        },
    },
    "internal_error": {
        "summary": "Internal Server Error",
        "description": "Unexpected server error occurred",
        "value": {
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {
                    "trace_id": "770e8400-e29b-41d4-a716-446655440000",
                },
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2025-10-18T10:30:00Z",
            }
        },
    },
}

# Health check examples
HEALTH_CHECK_EXAMPLE = {
    "status": "healthy",
    "timestamp": "2025-10-18T10:30:00Z",
    "version": "0.1.0",
    "uptime": 86400,
    "components": {
        "redis": "healthy",
        "database": "healthy",
        "jwt": "healthy",
    },
}

READINESS_CHECK_EXAMPLE = {
    "ready": True,
    "checks": {
        "redis": {"status": "ready", "latency_ms": 1.2},
        "session_manager": {"status": "ready", "active_sessions": 42},
        "jwt_manager": {"status": "ready", "key_expires_in_days": 75},
    },
}
