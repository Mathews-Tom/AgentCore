# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentCore is an open-source orchestration framework for agentic AI systems implementing Google's A2A (Agent2Agent) protocol v0.2. It provides JSON-RPC 2.0 compliant infrastructure for agent communication, discovery, task management, and real-time messaging via WebSocket/SSE.

**Stack:** Python 3.12+, FastAPI, PostgreSQL, Redis, Pydantic, SQLAlchemy (async), Alembic

## Development Commands

### Environment Setup

```bash
# Install dependencies (using uv package manager)
uv add <package-name>

# Execute commands/scripts
uv run <command>
```

### Running the Application

```bash
# Development server with hot reload (port 8001)
uv run uvicorn agentcore.a2a_protocol.main:app --host 0.0.0.0 --port 8001 --reload

# Using Docker Compose (includes PostgreSQL + Redis)
docker compose -f docker-compose.dev.yml up

# API docs available at http://localhost:8001/docs (DEBUG mode only)
```

### Testing

```bash
# Run all tests with coverage (requires 90%+ coverage)
uv run pytest

# Run specific test file
uv run pytest tests/integration/test_agent_lifecycle.py

# Run single test function
uv run pytest tests/integration/test_agent_lifecycle.py::test_agent_registration

# Integration tests only
uv run pytest tests/integration/

# Load tests using Locust
uv run locust -f tests/load/locustfile.py
```

### Database Migrations

```bash
# Create new migration
uv run alembic revision --autogenerate -m "description"

# Apply migrations
uv run alembic upgrade head

# Rollback one migration
uv run alembic downgrade -1

# Show current version
uv run alembic current
```

### Code Quality

```bash
# Run Ruff linter
uv run ruff check src/ tests/

# Auto-fix linting issues
uv run ruff check --fix src/ tests/

# Type checking with mypy (strict mode enabled)
uv run mypy src/
```

## Architecture Overview

### Core Components

**JSON-RPC Handler (`services/jsonrpc_handler.py`):**

- `JsonRpcProcessor`: Central request/response processor with method registry
- Supports single and batch requests, notifications, middleware, A2A context handling
- Built-in methods: `rpc.ping`, `rpc.methods`, `rpc.version`
- Global instance: `jsonrpc_processor`

**A2A Protocol Models (`models/`):**

- `jsonrpc.py`: JSON-RPC 2.0 models (request, response, error codes, A2A context)
- `agent.py`: AgentCard specification for discovery and capabilities
- `task.py`: Task lifecycle and artifact models
- `events.py`: Event streaming models for real-time updates
- `security.py`: Authentication and authorization models

**Services Layer (`services/`):**

- `agent_manager.py`: Agent registration, discovery, health monitoring
- `task_manager.py`: Task creation, execution tracking, artifact handling
- `message_router.py`: Intelligent routing based on agent capabilities
- `security_service.py`: JWT authentication, RBAC authorization
- `event_manager.py`: Real-time event streaming via SSE/WebSocket
- Each manager has a corresponding `*_jsonrpc.py` file that registers JSON-RPC methods

**Database (`database/`):**

- `models.py`: SQLAlchemy ORM models (AgentRecord, TaskRecord, ArtifactRecord)
- `repositories.py`: Data access patterns (AgentRepository, TaskRepository)
- `connection.py`: Async database connection management with `init_db()`, `close_db()`, `get_session()`
- Uses PostgreSQL with async driver (asyncpg)

**Routers (`routers/`):**

- `jsonrpc.py`: POST `/api/v1/jsonrpc` - main JSON-RPC endpoint
- `websocket.py`: WebSocket endpoint for bidirectional communication
- `wellknown.py`: `/.well-known/agent.json` - A2A discovery endpoint
- `health.py`: Health check and readiness endpoints

### Configuration

All settings in `config.py` using Pydantic Settings (loads from `.env`):

- Database: `DATABASE_URL` or individual `POSTGRES_*` variables
- Redis: `REDIS_URL` or `REDIS_CLUSTER_URLS`
- Security: `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `JWT_EXPIRATION_HOURS`
- A2A Protocol: `A2A_PROTOCOL_VERSION`, `MAX_CONCURRENT_CONNECTIONS`, `MESSAGE_TIMEOUT_SECONDS`
- Monitoring: `ENABLE_METRICS`, `LOG_LEVEL`

### Key Design Patterns

**Method Registration:**

```python
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method

@register_jsonrpc_method("agent.register")
async def handle_agent_register(request: JsonRpcRequest) -> Dict[str, Any]:
    # Implementation
    return result
```

**Database Access:**

```python
from agentcore.a2a_protocol.database import get_session

async with get_session() as session:
    repo = AgentRepository(session)
    agent = await repo.get_by_id(agent_id)
```

**A2A Context:**
All JSON-RPC requests can include `a2a_context` with `trace_id`, `source_agent`, `target_agent`, `session_id`, `timestamp` for distributed tracing and agent routing.

## Important Notes

- **Async-first**: All I/O operations use asyncio (database, HTTP, WebSocket)
- **Type safety**: Mypy strict mode enabled - never use `any`, always provide proper types
- **A2A Compliance**: Follow Google's A2A protocol v0.2 specification
- **Testing**: pytest-asyncio with 90%+ coverage requirement, testcontainers for Redis
- **Security**: JWT auth required for agent operations, validate all inputs with Pydantic
- **Error handling**: Use JSON-RPC error codes from `JsonRpcErrorCode` enum
- **Database**: All repository methods are async, use context managers for sessions
- **Method registration**: Import service modules in `main.py` to auto-register JSON-RPC methods

## Common Patterns

### Adding a New JSON-RPC Method

1. Define Pydantic models for params/result if needed
2. Create handler in appropriate `services/*_jsonrpc.py` file
3. Use `@register_jsonrpc_method("namespace.method")` decorator
4. Import the service module in `main.py` to auto-register
5. Add integration test in `tests/integration/`

### Creating Database Migration

1. Update models in `database/models.py`
2. Run `uv run alembic revision --autogenerate -m "description"`
3. Review generated migration in `alembic/versions/`
4. Apply with `uv run alembic upgrade head`
5. Commit migration file with code changes

### Adding Agent Capability

1. Update AgentCard model in `models/agent.py`
2. Modify agent registration logic in `services/agent_manager.py`
3. Update capability matching in `services/message_router.py`
4. Add validation in agent registration JSON-RPC handler
5. Write integration test for new capability

## Deployment

**Local Development:** Use `docker-compose.dev.yml` for full stack (app, PostgreSQL, Redis)

**Production:** See `k8s/` directory for Kubernetes manifests (deployment, service, configmap, secret, HPA, servicemonitor)

**Environment Variables:** Use `.env` file locally, Kubernetes secrets in production
