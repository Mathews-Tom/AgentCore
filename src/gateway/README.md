# Gateway Layer

High-performance API gateway for AgentCore built on FastAPI, providing a unified entry point for all external interactions.

## Features

- **FastAPI Application**: Async-first web framework with automatic OpenAPI documentation
- **Health Endpoints**: `/health`, `/ready`, `/live` for monitoring and load balancing
- **Middleware Stack**:
  - Request/response logging with distributed tracing
  - CORS support for web applications
  - Prometheus metrics collection
- **Configuration**: Environment-based settings via Pydantic
- **Production Ready**: Gunicorn/Uvicorn deployment support

## Quick Start

```bash
# Run development server
uv run uvicorn gateway.main:app --host 0.0.0.0 --port 8080 --reload

# Run tests
uv run pytest tests/gateway/ -v

# Access API documentation (DEBUG mode only)
open http://localhost:8080/docs
```

## Architecture

```
src/gateway/
├── __init__.py          # Package initialization
├── main.py              # FastAPI application
├── config.py            # Configuration settings
├── middleware/          # Middleware components
│   ├── logging.py       # Request/response logging
│   ├── cors.py          # CORS configuration
│   └── metrics.py       # Prometheus metrics
├── routes/              # Route handlers
│   └── health.py        # Health check endpoints
└── models/              # Pydantic models
    └── health.py        # Health response models
```

## Configuration

Set environment variables with `GATEWAY_` prefix:

```bash
GATEWAY_DEBUG=false
GATEWAY_HOST=0.0.0.0
GATEWAY_PORT=8080
GATEWAY_ALLOWED_ORIGINS=["http://localhost:3000"]
GATEWAY_ENABLE_METRICS=true
GATEWAY_LOG_LEVEL=INFO
```

## Endpoints

- `GET /health` - Health check with version info
- `GET /ready` - Readiness probe for load balancers
- `GET /live` - Liveness probe for orchestrators
- `GET /metrics` - Prometheus metrics (if enabled)
- `GET /docs` - OpenAPI documentation (DEBUG mode)

## Next Steps (Future Tickets)

- GATE-002: JWT Authentication System
- GATE-003: OAuth 3.0 Integration
- GATE-004: Rate Limiting & Security
- GATE-007: Backend Service Routing

## Development

This is the foundation layer (GATE-001). Future enhancements will add:
- Authentication and authorization
- Request routing to backend services
- Rate limiting and advanced security
- WebSocket and SSE support
