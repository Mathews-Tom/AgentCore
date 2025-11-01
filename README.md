# AgentCore - Production-Ready Agentic AI Framework

**AgentCore** is an open-source orchestration framework for building, training, and deploying intelligent agent systems at scale. Built on Google's A2A (Agent-to-Agent) protocol v0.2, it provides enterprise-grade infrastructure for multi-agent collaboration, automated prompt optimization, and cross-platform LLM integration.

[![Tests](https://img.shields.io/badge/tests-4598%20passing-brightgreen)](./tests)
[![Coverage](https://img.shields.io/badge/coverage-94.2%25-brightgreen)](./htmlcov)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

---

## 🎯 Key Features

### 🤖 Agent-to-Agent Protocol (A2A)
- **JSON-RPC 2.0** compliant API for standardized agent communication
- **Agent discovery** via `.well-known/agent.json` endpoints
- **Task coordination** with lifecycle management (create, assign, execute, complete)
- **Real-time messaging** via WebSocket and Server-Sent Events (SSE)
- **Distributed tracing** with A2A context propagation

### 🧠 DSPy Optimization Framework
- **Automated prompt optimization** using MIPROv2, BootstrapFewShot, and GEPA algorithms
- **Continuous learning pipeline** with online optimization and drift detection
- **A/B testing framework** for prompt variant comparison
- **MLflow integration** for experiment tracking and model versioning
- **Performance analytics** with ROI tracking and improvement recommendations
- **GPU acceleration** support for high-throughput optimization
- **Custom algorithm plugins** for domain-specific optimization strategies

### 🔌 LLM Gateway
- **Multi-provider support**: OpenAI (GPT-5), Anthropic (Claude 4.5), Google (Gemini 2.5)
- **Intelligent routing** with automatic failover and load balancing
- **Cost tracking** and budget management
- **Rate limit handling** with exponential backoff
- **Caching** for duplicate request optimization
- **Model selection** based on task requirements and constraints

### 🎭 Agent Runtime
- **Chain-of-Thought (CoT)** reasoning engine
- **Multi-tool integration** with external API connectors
- **Memory management** with context preservation
- **Error recovery** and retry mechanisms
- **Bounded context** for consistent agent behavior
- **State persistence** and checkpointing

### 📊 Orchestration & Workflows
- **Graph-based workflow** planning with topological sorting
- **Event-driven architecture** with 15k+ events/sec throughput
- **Parallel execution** with automatic dependency resolution
- **Performance monitoring** with Prometheus metrics
- **Health checks** and circuit breakers

### 🔐 Security & Compliance
- **JWT authentication** with role-based access control (RBAC)
- **PII detection** and data anonymization
- **Audit logging** for compliance tracking
- **Secure credential management** via Kubernetes secrets
- **Network isolation** with service mesh integration

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- PostgreSQL 14+
- Redis 7+
- (Optional) MLflow server for experiment tracking
- (Optional) GPU for accelerated optimization

### Installation

```bash
# Clone the repository
git clone https://github.com/Mathews-Tom/AgentCore.git
cd AgentCore

# Install dependencies using uv
uv sync

# Set up environment variables
cp .env.test.template .env
# Edit .env with your configuration
```

### Running with Docker Compose

```bash
# Start all services (PostgreSQL, Redis, AgentCore)
docker compose -f docker-compose.dev.yml up

# API available at http://localhost:8001
# API docs at http://localhost:8001/docs (DEBUG mode only)
```

### Running Locally

```bash
# Start database and Redis
docker compose -f docker-compose.dev.yml up postgres redis

# Source environment variables
set -a && source .env && set +a

# Run migrations
uv run alembic upgrade head

# Start development server
uv run uvicorn agentcore.a2a_protocol.main:app --host 0.0.0.0 --port 8001 --reload
```

---

## 📖 Documentation

### Core Documentation
- **[Architecture Overview](./docs/architecture/)** - System design and component interactions
- **[API Reference](./docs/api/)** - JSON-RPC method specifications
- **[Deployment Guide](./docs/deployment/)** - Production deployment with Kubernetes
- **[Development Guide](./CLAUDE.md)** - Contributing and development workflow

### DSPy Optimization
- **[DSPy Specification](./docs/specs/dspy-optimization/spec.md)** - Detailed requirements and design
- **[DSPy Implementation Plan](./docs/specs/dspy-optimization/plan.md)** - Architecture and integration
- **[DSPy User Guide](./docs/dspy-optimization/user-guide.md)** - Getting started with optimization
- **[DSPy Best Practices](./docs/dspy-optimization/best-practices.md)** - Tips and patterns

### LLM Integration
- **[LLM Client Service](./docs/llm-client-service/README.md)** - Multi-provider LLM integration
- **[Configuration Guide](./docs/llm-client-service/configuration-guide.md)** - Provider setup
- **[Model Selection](./docs/specs/llm-client-service/spec.md)** - Intelligent model routing

### Research & Future Work
- **[Parallax & OpenEnv Analysis](./docs/research/parallax-openenv-analysis.md)** - Future enhancements
- **[Integration Proposal](./docs/research/parallax-openenv-integration-proposal.md)** - Technical specifications
- **[Implementation Roadmap](./docs/research/parallax-openenv-roadmap.md)** - Execution plan

---

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/agentcore --cov-report=html

# Run specific test categories
uv run pytest -m unit              # Unit tests only
uv run pytest -m integration       # Integration tests
uv run pytest -m performance       # Performance benchmarks
uv run pytest -m slow              # Long-running tests

# Run specific test file
uv run pytest tests/unit/test_llm_service.py

# Run with verbose output
uv run pytest -v

# Performance benchmarks
uv run pytest -m performance --durations=20
```

### Test Coverage

Current test coverage: **94.2%** (Target: >90%)

- **Unit Tests**: 2,847 tests
- **Integration Tests**: 1,312 tests
- **Performance Tests**: 312 tests
- **E2E Tests**: 127 tests

**Total**: 4,598 tests passing

---

## 🏗️ Architecture

### Component Overview

```
agentcore/
├── a2a_protocol/           # A2A protocol implementation
│   ├── models/             # Pydantic models (AgentCard, Task, Event, etc.)
│   ├── services/           # Business logic (AgentManager, TaskManager, etc.)
│   ├── routers/            # FastAPI endpoints
│   ├── database/           # PostgreSQL models and repositories
│   └── metrics/            # Prometheus metrics collection
│
├── agent_runtime/          # Agent execution engine
│   ├── engines/            # CoT reasoning, planning
│   ├── services/           # Tool integration, memory management
│   └── config/             # Runtime configuration
│
├── dspy_optimization/      # DSPy optimization framework
│   ├── algorithms/         # MIPROv2, GEPA, BootstrapFewShot
│   ├── analytics/          # Performance analytics, ROI tracking
│   ├── learning/           # Online learning, drift detection
│   ├── monitoring/         # Metrics collection, baselines
│   ├── plugins/            # Custom algorithm registry
│   ├── testing/            # A/B testing, experiments
│   └── tracking/           # MLflow integration
│
├── llm_gateway/            # Multi-provider LLM integration
│   ├── providers/          # OpenAI, Anthropic, Gemini clients
│   ├── models.py           # Request/response models
│   ├── failover.py         # Automatic failover logic
│   ├── cost_tracker.py     # Cost monitoring
│   └── cache_service.py    # Response caching
│
├── orchestration/          # Workflow orchestration
│   ├── engines/            # Graph planning, execution
│   ├── events/             # Event streaming
│   └── performance/        # Benchmarking, optimization
│
├── gateway/                # API gateway services
│   ├── routing.py          # Request routing
│   ├── health_monitor.py   # Health checks
│   └── load_balancer.py    # Load distribution
│
└── integration/            # External service integrations
    ├── api/                # API connectors
    ├── storage/            # Blob storage (Azure, S3)
    └── security/           # Credential management
```

### Technology Stack

**Backend**:
- Python 3.12+ with asyncio
- FastAPI for HTTP/WebSocket APIs
- PostgreSQL 14+ (async via asyncpg)
- Redis 7+ for caching and pub/sub

**Optimization**:
- DSPy for prompt optimization
- MLflow for experiment tracking
- SciPy for statistical analysis
- NetworkX for graph algorithms

**Observability**:
- Prometheus for metrics
- Grafana for dashboards
- OpenTelemetry for distributed tracing
- Structured logging with structlog

**Deployment**:
- Kubernetes for orchestration
- Docker for containerization
- Alembic for database migrations
- GitHub Actions for CI/CD

---

## 🎓 Example Usage

### Registering an Agent

```python
import httpx

# Agent registration
agent_card = {
    "agent_id": "my-agent-001",
    "name": "My Agent",
    "description": "A sample intelligent agent",
    "capabilities": ["text_generation", "data_analysis"],
    "version": "1.0.0",
    "owner": "team@example.com"
}

request = {
    "jsonrpc": "2.0",
    "method": "agent.register",
    "params": {"agent_card": agent_card},
    "id": "1"
}

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8001/api/v1/jsonrpc",
        json=request
    )
    print(response.json())
```

### Running DSPy Optimization

```python
from agentcore.dspy_optimization.pipeline import DSPyOptimizationPipeline
from agentcore.dspy_optimization.models import OptimizationRequest

# Initialize pipeline
pipeline = DSPyOptimizationPipeline()

# Configure optimization
request = OptimizationRequest(
    target="prompt_template",
    objective="maximize_accuracy",
    algorithm="miprov2",
    evaluation_metric="f1_score",
    max_iterations=100
)

# Run optimization
result = await pipeline.optimize(request)
print(f"Optimized prompt: {result.optimized_prompt}")
print(f"Performance improvement: {result.improvement_percentage}%")
```

### Creating and Executing Tasks

```python
# Create task
task_definition = {
    "description": "Analyze customer feedback and extract insights",
    "requirements": ["text_analysis", "sentiment_detection"],
    "priority": "high",
    "timeout_seconds": 300
}

create_request = {
    "jsonrpc": "2.0",
    "method": "task.create",
    "params": {
        "task_definition": task_definition,
        "auto_assign": True
    },
    "id": "2"
}

response = await client.post(
    "http://localhost:8001/api/v1/jsonrpc",
    json=create_request
)
execution_id = response.json()["result"]["execution_id"]

# Monitor task progress via WebSocket
async with client.websocket_connect("ws://localhost:8001/ws") as ws:
    subscribe_msg = {
        "jsonrpc": "2.0",
        "method": "event.subscribe",
        "params": {
            "event_types": ["task.updated", "task.completed"],
            "filters": {"execution_id": execution_id}
        },
        "id": "3"
    }
    await ws.send_json(subscribe_msg)

    # Receive real-time updates
    async for message in ws:
        event = message.json()
        print(f"Task update: {event['event_type']}")
        if event["event_type"] == "task.completed":
            break
```

---

## 🔬 Performance Benchmarks

### Optimization Performance
- **Graph Planning**: <1s for 1000+ node workflows
- **Event Processing**: 15,000+ events/second
- **DSPy Optimization**: <5 minutes for typical prompts
- **LLM Latency**: P95 <500ms (with caching)

### Scalability
- **Concurrent Agents**: 1000+ agents per instance
- **Task Throughput**: 500+ tasks/second
- **WebSocket Connections**: 10,000+ concurrent
- **Database**: 100,000+ ops/second with PostgreSQL

### Resource Usage
- **Memory**: ~500MB baseline, scales with workload
- **CPU**: 2-4 cores typical, scales with parallelism
- **GPU** (optional): 1x GPU for DSPy acceleration

---

## 🛠️ Development

### Setting Up Development Environment

```bash
# Install development dependencies
uv sync --dev

# Run linting
uv run ruff check src/ tests/

# Auto-fix linting issues
uv run ruff check --fix src/ tests/

# Type checking
uv run mypy src/

# Run tests with coverage
uv run pytest --cov=src/agentcore --cov-report=html
```

### Code Quality Standards
- **Test Coverage**: Minimum 90% required
- **Type Safety**: Mypy strict mode enabled
- **Linting**: Ruff with project-specific rules
- **Formatting**: Black-compatible (via Ruff)
- **Documentation**: Docstrings for all public APIs

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure all tests pass (`uv run pytest`)
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📦 Deployment

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Verify deployment
kubectl get pods -n agentcore
kubectl get svc -n agentcore

# Check logs
kubectl logs -f deployment/agentcore-api -n agentcore

# Scale deployment
kubectl scale deployment/agentcore-api --replicas=3 -n agentcore
```

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/agentcore
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agentcore
POSTGRES_USER=agentcore
POSTGRES_PASSWORD=secure_password

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=secure_password

# Security
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...

# MLflow (Optional)
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=agentcore-optimization

# Monitoring
ENABLE_METRICS=true
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO
```

---

## 📊 Monitoring & Observability

### Prometheus Metrics
- `agentcore_agent_count`: Number of registered agents
- `agentcore_task_count`: Total tasks by status
- `agentcore_llm_requests_total`: LLM API requests
- `agentcore_llm_latency_seconds`: LLM response latency
- `agentcore_optimization_runs_total`: DSPy optimization runs
- `agentcore_event_throughput`: Event processing rate

### Grafana Dashboards
- Agent health and activity
- Task execution metrics
- LLM usage and costs
- DSPy optimization performance
- System resource utilization

### Distributed Tracing
All requests include A2A context with:
- `trace_id`: End-to-end request tracking
- `source_agent`: Originating agent
- `target_agent`: Destination agent
- `session_id`: User session tracking
- `timestamp`: Request timestamp

---

## 🤝 Community & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Mathews-Tom/AgentCore/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/Mathews-Tom/AgentCore/discussions)
- **Documentation**: [Full documentation](./docs/)
- **Examples**: [Sample implementations](./examples/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Google A2A Protocol** - Standardized agent communication
- **DSPy Framework** - Prompt optimization algorithms
- **Stanford NLP** - Research and innovation
- **FastAPI** - Modern Python web framework
- **Pydantic** - Data validation and settings management

---

## 🗺️ Roadmap

### Q1 2025
- ✅ DSPy optimization framework integration
- ✅ Multi-provider LLM gateway with cost tracking
- ✅ Production-ready deployment configurations
- 🔄 OpenEnv training environment integration (planned)

### Q2 2025
- 🔮 Parallax POC for self-hosted LLM inference
- 🔮 Enhanced monitoring and alerting
- 🔮 Advanced workflow orchestration patterns
- 🔮 Plugin marketplace for custom algorithms

### Q3-Q4 2025
- 🔮 Multi-region deployment support
- 🔮 Advanced security features (SAML, SSO)
- 🔮 Graph-based memory and knowledge management
- 🔮 Enterprise SLA guarantees

---

**Built with ❤️ by the AgentCore team**
