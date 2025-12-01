# AgentCore

Production-ready orchestration framework for agentic AI systems implementing Google's A2A (Agent-to-Agent) protocol v0.2.

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-8000%2B-brightgreen)](./tests)

---

## Features

### A2A Protocol Implementation

- **JSON-RPC 2.0** compliant API for standardized agent communication
- **Agent discovery** via `/.well-known/agent.json` endpoints
- **Task coordination** with full lifecycle management
- **Real-time messaging** via WebSocket and Server-Sent Events (SSE)
- **Distributed tracing** with A2A context propagation

### Agent Runtime

- **Chain-of-Thought (CoT)** reasoning engine
- **ReAct** pattern for iterative reasoning and action
- **Autonomous execution** mode for complex multi-step tasks
- **Multi-tool integration** with API connectors, code execution, and file operations
- **Plugin system** with version management and validation
- **State persistence** with backup and recovery
- **Sandbox execution** with security profiles

### Memory Service

- **Hierarchical memory** with working, episodic, and semantic layers
- **Entity-Centric Learning (ECL)** pipeline for knowledge extraction
- **Graph-based memory** with Neo4j integration
- **Hybrid search** combining vector similarity and graph traversal
- **MEMify optimization** for memory consolidation and pruning
- **Context compression** and expansion for efficient token usage

### LLM Client Service

- **Multi-provider support**: OpenAI, Anthropic, Google Gemini
- **Intelligent model selection** based on task requirements
- **Automatic failover** with provider health monitoring
- **Cost tracking** and budget management
- **Response caching** for optimization

### ACE (Adaptive Capability Engine)

- **Performance monitoring** with baseline tracking
- **Capability evaluation** and fitness scoring
- **Intervention engine** with trigger-based decisions
- **Playbook management** for automated responses
- **Delta generation** for capability improvements

### DSPy Optimization

- **MIPROv2** and **GEPA** optimization algorithms
- **MLflow integration** for experiment tracking
- **A/B testing framework** for prompt variants
- **Continuous learning** with drift detection
- **GPU acceleration** support

### Coordination Service

- **Multi-agent coordination** with consensus protocols
- **Workflow orchestration** with graph-based execution
- **CQRS pattern** with event sourcing
- **Saga pattern** for distributed transactions
- **Parallel execution** with dependency resolution

### Integration Layer

- **Cloud storage**: S3, GCS, Azure Blob
- **Database connectors**: PostgreSQL with async support
- **Webhook management** with delivery tracking
- **Resilience patterns**: Circuit breaker, bulkhead, timeout
- **Security**: Credential management, compliance scanning

### Training System

- **GRPO** (Group Relative Policy Optimization)
- **Trajectory recording** and replay
- **Custom reward registry**
- **Credit assignment** algorithms
- **Job management** with budget controls

### CLI

- **4-layer architecture**: CLI, Service, Protocol, Transport
- **Agent management**: register, list, info, remove
- **Task operations**: create, list, status tracking
- **Session management**: create, list, pause
- **Workflow control**: start, monitor, manage
- **Configuration**: show, set, validate

---

## Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 14+
- Redis 7+
- Neo4j 5+ (optional, for graph memory)

### Installation

```bash
# Clone the repository
git clone https://github.com/Mathews-Tom/AgentCore.git
cd AgentCore

# Install dependencies using uv
uv sync

# Set up environment
cp .env.test.template .env
```

### Running with Docker Compose

```bash
# Start all services
docker compose -f docker-compose.dev.yml up

# API available at http://localhost:8001
# API docs at http://localhost:8001/docs
```

### Running Locally

```bash
# Start database and Redis
docker compose -f docker-compose.dev.yml up postgres redis

# Apply migrations
uv run alembic upgrade head

# Start development server
uv run uvicorn agentcore.a2a_protocol.main:app --host 0.0.0.0 --port 8001 --reload
```

### CLI Usage

```bash
# Register an agent
agentcore agent register --name "my-agent" --capabilities "text_generation,analysis"

# List agents
agentcore agent list

# Create a task
agentcore task create --description "Analyze customer feedback"

# Start a session
agentcore session create --name "analysis-session"
```

---

## Architecture

```plaintext
src/agentcore/
├── a2a_protocol/          # A2A protocol implementation
│   ├── models/            # Pydantic models (AgentCard, Task, Event)
│   ├── services/          # Business logic and JSON-RPC handlers
│   ├── routers/           # FastAPI endpoints
│   └── database/          # PostgreSQL models and repositories
│
├── agent_runtime/         # Agent execution engine
│   ├── engines/           # CoT, ReAct, Autonomous engines
│   ├── services/          # Lifecycle, tools, state management
│   └── tools/             # Built-in and custom tool support
│
├── ace/                   # Adaptive Capability Engine
│   ├── capability/        # Evaluation and scoring
│   ├── intervention/      # Trigger-based decisions
│   └── monitors/          # Performance tracking
│
├── dspy_optimization/     # DSPy optimization framework
│   ├── algorithms/        # MIPROv2, GEPA
│   ├── learning/          # Continuous learning, drift detection
│   └── tracking/          # MLflow integration
│
├── gateway/               # API gateway services
├── integration/           # External service integrations
├── orchestration/         # Workflow orchestration (CQRS, Saga)
├── reasoning/             # Reasoning strategies
└── training/              # RL training system

src/agentcore_cli/         # Command-line interface
├── commands/              # CLI command handlers
├── services/              # Service layer
├── protocol/              # JSON-RPC client
└── transport/             # HTTP transport
```

---

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov-report=html

# Run specific test categories
uv run pytest tests/unit/
uv run pytest tests/integration/
uv run pytest tests/cli/

# Load testing
uv run locust -f tests/load/locustfile.py
```

---

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run linter
uv run ruff check src/ tests/

# Auto-fix issues
uv run ruff check --fix src/ tests/

# Type checking
uv run mypy src/
```

### Database Migrations

```bash
# Create migration
uv run alembic revision --autogenerate -m "description"

# Apply migrations
uv run alembic upgrade head

# Rollback
uv run alembic downgrade -1
```

---

## Configuration

All settings via environment variables or `.env`:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/agentcore

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256

# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
```

---

## Documentation

- [API Reference](./docs/api/) - JSON-RPC method specifications
- [Architecture](./docs/architecture/) - System design documents
- [CLI Reference](./docs/cli/) - Command-line interface guide
- [Deployment](./docs/deployment/) - Production deployment guides
- [Security](./docs/security/) - Security audit and compliance

---

## License

AGPL-3.0 License - see [LICENSE](./LICENSE) for details.
