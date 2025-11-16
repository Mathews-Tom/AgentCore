# Memory Service Production Deployment Guide

This document provides comprehensive instructions for deploying the AgentCore Memory Service in local development and production Kubernetes environments. It covers infrastructure setup, database initialization, mock removal, and validation procedures.

## Table of Contents

1. [Local Development Deployment](#local-development-deployment)
2. [Production Kubernetes Deployment](#production-kubernetes-deployment)
3. [Remove Memify Mock Implementation](#remove-memify-mock-implementation)
4. [Validation and Testing](#validation-and-testing)
5. [Configuration Reference](#configuration-reference)
6. [Resource Requirements](#resource-requirements)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Next Steps Checklist](#next-steps-checklist)

---

## Local Development Deployment

### Prerequisites

- Docker and Docker Compose installed
- Python 3.12+ with `uv` package manager
- Git repository cloned

### Docker Compose Setup

The development environment uses `docker-compose.dev.yml` which includes all required infrastructure:

```bash
# Start all services
docker compose -f docker-compose.dev.yml up -d

# Verify services are healthy
docker compose -f docker-compose.dev.yml ps
```

**Services included:**
- **PostgreSQL (pgvector)**: `localhost:5432` - Primary relational storage
- **Redis**: `localhost:6379` - Working memory cache
- **Qdrant**: `localhost:6333` (HTTP), `localhost:6334` (gRPC) - Vector embeddings
- **Neo4j**: `localhost:7474` (HTTP), `localhost:7687` (Bolt) - Knowledge graph
- **A2A Protocol**: `localhost:8001` - Application server

### Environment Configuration

Create a `.env` file in the project root:

```bash
# Database
DATABASE_URL=postgresql://agentcore:password@localhost:5432/agentcore
POSTGRES_USER=agentcore
POSTGRES_PASSWORD=password
POSTGRES_DB=agentcore

# Redis
REDIS_URL=redis://localhost:6379/0

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_GRPC_URL=http://localhost:6334
QDRANT_COLLECTION_NAME=memory_vectors
QDRANT_VECTOR_SIZE=1536
QDRANT_DISTANCE=Cosine
QDRANT_TIMEOUT=30
QDRANT_MAX_RETRIES=3

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=agentcore
NEO4J_MAX_CONNECTION_LIFETIME=3600
NEO4J_MAX_CONNECTION_POOL_SIZE=50
NEO4J_CONNECTION_ACQUISITION_TIMEOUT=60
NEO4J_ENCRYPTED=false

# Security
JWT_SECRET_KEY=dev-secret-key-change-in-production
DEBUG=true
```

### Qdrant Collection Creation

The Qdrant collection is automatically created on application startup via `StorageBackendService.initialize()`. Manual creation (if needed):

```bash
# Using curl
curl -X PUT http://localhost:6333/collections/memory_vectors \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 1536,
      "distance": "Cosine"
    },
    "optimizers_config": {
      "memmap_threshold": 20000
    },
    "replication_factor": 1
  }'

# Verify collection
curl http://localhost:6333/collections/memory_vectors
```

### Neo4j Schema Initialization

The GraphOptimizer automatically creates required indexes. For manual schema setup:

```cypher
// Connect via Neo4j Browser at http://localhost:7474
// Username: neo4j, Password: password

// Create constraints for Entity nodes
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;

// Create indexes for common queries
CREATE INDEX entity_type_idx IF NOT EXISTS
FOR (e:Entity) ON (e.entity_type);

CREATE INDEX entity_agent_idx IF NOT EXISTS
FOR (e:Entity) ON (e.agent_id);

CREATE INDEX entity_created_idx IF NOT EXISTS
FOR (e:Entity) ON (e.created_at);

// Verify indexes
SHOW INDEXES;
```

### Start Development Server

```bash
# Install dependencies
uv sync

# Run database migrations
uv run alembic upgrade head

# Start development server
uv run uvicorn agentcore.a2a_protocol.main:app --host 0.0.0.0 --port 8001 --reload
```

---

## Production Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.25+)
- `kubectl` configured
- Helm (optional, for chart-based deployment)
- Storage class with dynamic provisioning

### Deploy Namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

### Deploy Qdrant Vector Database

```bash
# Deploy Qdrant StatefulSet and Service
kubectl apply -f k8s/qdrant/deployment.yaml
kubectl apply -f k8s/qdrant/service.yaml

# Verify deployment
kubectl -n agentcore get pods -l app=qdrant
kubectl -n agentcore get svc qdrant

# Check Qdrant health
kubectl -n agentcore exec -it qdrant-0 -- wget -qO- http://localhost:6333/health
```

**Qdrant StatefulSet Configuration:**
- Image: `qdrant/qdrant:v1.7.0`
- Replicas: 1 (scale as needed for HA)
- Storage: 20Gi PVC (adjust based on vector count)
- Memory: 2Gi request / 4Gi limit
- CPU: 1000m request / 2000m limit

### Deploy Neo4j Graph Database

```bash
# Create Neo4j secret (update password!)
kubectl apply -f k8s/neo4j/secret.yaml

# Deploy Neo4j StatefulSet and Service
kubectl apply -f k8s/neo4j/statefulset.yaml
kubectl apply -f k8s/neo4j/service.yaml

# Verify deployment
kubectl -n agentcore get pods -l app=neo4j
kubectl -n agentcore get svc neo4j

# Check Neo4j health (after initialization)
kubectl -n agentcore exec -it neo4j-0 -- cypher-shell -u neo4j -p password "RETURN 1"
```

**Neo4j StatefulSet Configuration:**
- Image: `neo4j:5.15-community`
- Plugins: APOC 5.15.0, GDS 2.5.0 (installed via init container)
- Storage: 50Gi data, 10Gi logs, 10Gi import, 1Gi plugins
- Memory: 8Gi request/limit (4G heap, 2G page cache)
- CPU: 2000m request / 4000m limit
- Database: `agentcore`

**Important:** The Neo4j secret contains default credentials. **Change in production:**

```bash
# Generate secure password
NEW_PASSWORD=$(openssl rand -base64 32)

# Update secret
kubectl -n agentcore create secret generic neo4j-secret \
  --from-literal=auth="neo4j/${NEW_PASSWORD}" \
  --from-literal=password="${NEW_PASSWORD}" \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart Neo4j to apply
kubectl -n agentcore rollout restart statefulset neo4j
```

### Deploy Application ConfigMap and Secrets

Update `k8s/configmap.yaml` with memory service settings:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agentcore-config
  namespace: agentcore
data:
  # ... existing config ...

  # Qdrant Configuration
  QDRANT_URL: "http://qdrant.agentcore.svc.cluster.local:6333"
  QDRANT_GRPC_URL: "http://qdrant.agentcore.svc.cluster.local:6334"
  QDRANT_COLLECTION_NAME: "memory_vectors"
  QDRANT_VECTOR_SIZE: "1536"
  QDRANT_DISTANCE: "Cosine"
  QDRANT_TIMEOUT: "30"
  QDRANT_MAX_RETRIES: "3"

  # Neo4j Configuration
  NEO4J_URI: "bolt://neo4j.agentcore.svc.cluster.local:7687"
  NEO4J_USER: "neo4j"
  NEO4J_DATABASE: "agentcore"
  NEO4J_MAX_CONNECTION_LIFETIME: "3600"
  NEO4J_MAX_CONNECTION_POOL_SIZE: "50"
  NEO4J_CONNECTION_ACQUISITION_TIMEOUT: "60"
  NEO4J_ENCRYPTED: "false"
```

Update `k8s/secret.yaml` with sensitive data:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: agentcore-secret
  namespace: agentcore
type: Opaque
stringData:
  # ... existing secrets ...
  NEO4J_PASSWORD: "your-secure-password"
  QDRANT_API_KEY: ""  # Empty for local deployment
```

### Deploy Application

```bash
# Apply configuration
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Verify deployment
kubectl -n agentcore get pods -l app=agentcore
kubectl -n agentcore logs -l app=agentcore --tail=100
```

---

## Remove Memify Mock Implementation

The current `memory.run_memify` JSON-RPC method returns mock results. Follow these steps to wire up the actual `GraphOptimizer`.

### Step 1: Verify Dependencies

Dependencies are already declared in `pyproject.toml`:
- `neo4j>=5.15.0`
- `croniter>=6.0.0`

Sync dependencies if needed:

```bash
uv sync
```

### Step 2: Update StorageBackendService Access

The `StorageBackendService` already initializes the Neo4j driver. No changes needed to `main.py` - the lifespan already calls:

```python
# main.py:87-88 (already implemented)
await initialize_storage_backend()
logger.info("Memory storage backends initialized (PostgreSQL, Qdrant, Neo4j)")
```

### Step 3: Replace Mock in memory_jsonrpc.py

Update the `handle_memory_run_memify` function in `src/agentcore/a2a_protocol/services/memory_jsonrpc.py`:

**Current (mock implementation, lines 1020-1068):**

```python
# Note: In production, this would connect to actual Neo4j driver
# For now, return mock optimization results
# This demonstrates the API contract without requiring Neo4j

from uuid import uuid4

optimization_id = f"opt-{uuid4()}"
start_time = datetime.now(UTC)

# Mock optimization results (in production, would call GraphOptimizer.optimize())
# ...

result = RunMemifyResult(
    optimization_id=optimization_id,
    entities_analyzed=0,  # Would be populated from actual optimization
    entities_merged=0,
    relationships_pruned=0,
    patterns_detected=0,
    consolidation_accuracy=1.0,
    duplicate_rate=0.0,
    duration_seconds=duration,
    scheduled_job_id=scheduled_job_id,
    next_run=next_run,
)
```

**Replace with actual implementation:**

```python
@register_jsonrpc_method("memory.run_memify")
async def handle_memory_run_memify(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Trigger graph optimization (memify) operation.

    [... docstring unchanged ...]
    """
    try:
        params_dict = request.params or {}
        if isinstance(params_dict, list):
            raise ValueError("Parameters must be an object, not an array")

        # Validate parameters
        params = RunMemifyParams(**params_dict)

        # Get storage backend to access Neo4j driver
        storage_backend = _get_storage_backend()

        # Create GraphOptimizer with Neo4j driver
        optimizer = GraphOptimizer(
            driver=storage_backend.neo4j_driver,
            similarity_threshold=params.similarity_threshold,
            min_access_count=params.min_access_count,
            batch_size=params.batch_size,
        )

        # Run optimization
        metrics = await optimizer.optimize()

        # Track scheduling if requested
        scheduled_job_id = None
        next_run = None
        if params.schedule_cron:
            from croniter import croniter

            if not croniter.is_valid(params.schedule_cron):
                raise ValueError(f"Invalid cron expression: {params.schedule_cron}")

            scheduled_job_id = optimizer.schedule_optimization(params.schedule_cron)
            next_run_dt = optimizer.get_next_scheduled_run()
            next_run = next_run_dt.isoformat() if next_run_dt else None

        result = RunMemifyResult(
            optimization_id=metrics.optimization_id,
            entities_analyzed=metrics.entities_analyzed,
            entities_merged=metrics.entities_merged,
            relationships_pruned=metrics.low_value_edges_removed,
            patterns_detected=metrics.patterns_detected,
            consolidation_accuracy=metrics.consolidation_accuracy,
            duplicate_rate=metrics.duplicate_rate,
            duration_seconds=metrics.duration_seconds,
            scheduled_job_id=scheduled_job_id,
            next_run=next_run,
        )

        logger.info(
            "Memify optimization completed",
            optimization_id=result.optimization_id,
            entities_analyzed=result.entities_analyzed,
            entities_merged=result.entities_merged,
            relationships_pruned=result.relationships_pruned,
            patterns_detected=result.patterns_detected,
            duration_seconds=result.duration_seconds,
            consolidation_accuracy=result.consolidation_accuracy,
            duplicate_rate=result.duplicate_rate,
            scheduled=scheduled_job_id is not None,
            method="memory.run_memify",
        )

        return result.model_dump(mode="json")

    except ValidationError as e:
        logger.error("Run memify validation failed", error=str(e))
        raise ValueError(f"Parameter validation failed: {e}") from e
```

### Step 4: Verify Import

Ensure the GraphOptimizer import is already present (line 34):

```python
from agentcore.a2a_protocol.services.memory.graph_optimizer import GraphOptimizer
```

---

## Validation and Testing

### Unit Tests

Run the complete memory service test suite:

```bash
# All memory service tests
uv run pytest tests/unit/services/test_memory_jsonrpc.py -v

# GraphOptimizer tests specifically
uv run pytest tests/unit/services/test_graph_optimizer.py -v

# Test with coverage
uv run pytest tests/unit/services/ -k memory --cov=src/agentcore/a2a_protocol/services/memory --cov-report=term-missing
```

### Integration Tests

Test full stack with actual databases:

```bash
# Start infrastructure
docker compose -f docker-compose.dev.yml up -d

# Wait for services to be healthy
docker compose -f docker-compose.dev.yml ps

# Run integration tests
uv run pytest tests/integration/ -k memory -v

# Test hybrid memory architecture
uv run pytest tests/integration/test_hybrid_memory.py -v
```

### Manual Verification Commands

#### 1. Test Memory Store

```bash
curl -X POST http://localhost:8001/api/v1/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "memory.store",
    "params": {
      "content": "Test memory for deployment validation",
      "memory_layer": "episodic",
      "agent_id": "test-agent-001",
      "session_id": "test-session-001",
      "keywords": ["test", "deployment", "validation"],
      "embedding": [0.1, 0.2, 0.3]
    },
    "id": 1
  }'
```

Expected response:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "memory_id": "mem-...",
    "memory_layer": "episodic",
    "timestamp": "...",
    "success": true,
    "message": "Memory stored successfully"
  },
  "id": 1
}
```

#### 2. Test Memory Retrieve

```bash
curl -X POST http://localhost:8001/api/v1/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "memory.retrieve",
    "params": {
      "agent_id": "test-agent-001",
      "session_id": "test-session-001",
      "limit": 10
    },
    "id": 2
  }'
```

#### 3. Test Memify Optimization

```bash
curl -X POST http://localhost:8001/api/v1/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "memory.run_memify",
    "params": {
      "similarity_threshold": 0.90,
      "min_access_count": 2,
      "batch_size": 100
    },
    "id": 3
  }'
```

Expected response (with actual GraphOptimizer):
```json
{
  "jsonrpc": "2.0",
  "result": {
    "optimization_id": "opt-...",
    "entities_analyzed": 0,
    "entities_merged": 0,
    "relationships_pruned": 0,
    "patterns_detected": 0,
    "consolidation_accuracy": 1.0,
    "duplicate_rate": 0.0,
    "duration_seconds": 0.05,
    "scheduled_job_id": null,
    "next_run": null
  },
  "id": 3
}
```

#### 4. Verify Qdrant Collection

```bash
# Check collection info
curl http://localhost:6333/collections/memory_vectors

# Count vectors
curl http://localhost:6333/collections/memory_vectors/points/count
```

#### 5. Verify Neo4j Connection

```bash
# Connect via cypher-shell
docker exec -it $(docker ps -qf "name=neo4j") cypher-shell -u neo4j -p password

# Check entity count
MATCH (e:Entity) RETURN count(e) as entity_count;

# Check indexes
SHOW INDEXES;
```

### Performance Validation

Test GraphOptimizer performance targets:

```bash
# Run performance benchmark
uv run pytest tests/performance/test_graph_optimizer_perf.py -v

# Expected: <5s per 1000 entities
# Expected: 90%+ consolidation accuracy
# Expected: <5% duplicate rate after optimization
```

### Health Check Endpoints

```bash
# Application health
curl http://localhost:8001/api/v1/health

# Readiness (includes DB connectivity)
curl http://localhost:8001/api/v1/ready

# Prometheus metrics
curl http://localhost:8001/metrics | grep memory
```

---

## Configuration Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| **PostgreSQL** |||
| `DATABASE_URL` | - | Full connection URL |
| `POSTGRES_USER` | `agentcore` | Database user |
| `POSTGRES_PASSWORD` | - | Database password |
| `POSTGRES_DB` | `agentcore` | Database name |
| **Redis** |||
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| **Qdrant** |||
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP API URL |
| `QDRANT_GRPC_URL` | `http://localhost:6334` | Qdrant gRPC API URL |
| `QDRANT_API_KEY` | - | API key (optional) |
| `QDRANT_COLLECTION_NAME` | `memory_vectors` | Vector collection name |
| `QDRANT_VECTOR_SIZE` | `1536` | Embedding dimension (OpenAI ada-002) |
| `QDRANT_DISTANCE` | `Cosine` | Distance metric (Cosine, Euclid, Dot) |
| `QDRANT_TIMEOUT` | `30` | Request timeout (seconds) |
| `QDRANT_MAX_RETRIES` | `3` | Max retry attempts |
| **Neo4j** |||
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | - | Neo4j password |
| `NEO4J_DATABASE` | `agentcore` | Neo4j database name |
| `NEO4J_MAX_CONNECTION_LIFETIME` | `3600` | Max connection lifetime (seconds) |
| `NEO4J_MAX_CONNECTION_POOL_SIZE` | `50` | Connection pool size |
| `NEO4J_CONNECTION_ACQUISITION_TIMEOUT` | `60` | Connection acquisition timeout (seconds) |
| `NEO4J_ENCRYPTED` | `false` | Enable TLS encryption |
| **Application** |||
| `DEBUG` | `false` | Enable debug mode |
| `JWT_SECRET_KEY` | - | JWT signing key |
| `LOG_LEVEL` | `INFO` | Logging level |
| `ENABLE_METRICS` | `true` | Enable Prometheus metrics |

---

## Resource Requirements

### Local Development (Docker Compose)

| Service | Memory | CPU | Storage |
|---------|--------|-----|---------|
| PostgreSQL (pgvector) | 1GB | 0.5 | 10GB |
| Redis | 512MB | 0.25 | 1GB |
| Qdrant | 2GB | 1 | 20GB |
| Neo4j | 4GB | 1 | 50GB |
| Application | 1GB | 0.5 | - |
| **Total** | **8.5GB** | **3.25** | **81GB** |

### Production Kubernetes (per component)

| Component | Memory Request/Limit | CPU Request/Limit | Storage |
|-----------|---------------------|-------------------|---------|
| PostgreSQL | 4Gi / 8Gi | 2000m / 4000m | 100Gi |
| Redis | 1Gi / 2Gi | 500m / 1000m | 10Gi |
| Qdrant | 2Gi / 4Gi | 1000m / 2000m | 20Gi |
| Neo4j | 8Gi / 8Gi | 2000m / 4000m | 70Gi total |
| Application | 1Gi / 2Gi | 500m / 1000m | - |

### Scaling Recommendations

- **1-10K memories**: Single replica for all databases
- **10K-100K memories**: Increase Qdrant/Neo4j storage, consider read replicas
- **100K+ memories**: Qdrant distributed mode, Neo4j causal cluster
- **High throughput**: Horizontal scaling of application pods (HPA configured)

---

## Troubleshooting Guide

### Common Issues

#### 1. Storage Backend Not Initialized

**Symptom:** `RuntimeError: StorageBackendService not initialized`

**Solution:**
```bash
# Check application logs
kubectl -n agentcore logs -l app=agentcore --tail=100 | grep -i storage

# Verify Qdrant is reachable
kubectl -n agentcore exec -it deploy/agentcore -- curl http://qdrant:6333/health

# Verify Neo4j is reachable
kubectl -n agentcore exec -it deploy/agentcore -- nc -zv neo4j 7687
```

#### 2. Qdrant Collection Not Found

**Symptom:** Vector search fails with collection not found

**Solution:**
```bash
# Check if collection exists
curl http://localhost:6333/collections/memory_vectors

# Create manually if needed
curl -X PUT http://localhost:6333/collections/memory_vectors \
  -H "Content-Type: application/json" \
  -d '{"vectors": {"size": 1536, "distance": "Cosine"}}'

# Restart application to reinitialize
kubectl -n agentcore rollout restart deployment agentcore
```

#### 3. Neo4j Connection Timeout

**Symptom:** `ServiceUnavailable: Connection timed out`

**Solution:**
```bash
# Check Neo4j pod status
kubectl -n agentcore get pods -l app=neo4j

# Check Neo4j logs
kubectl -n agentcore logs neo4j-0

# Verify Bolt port is exposed
kubectl -n agentcore exec -it neo4j-0 -- netstat -tlnp | grep 7687

# Test connection from app pod
kubectl -n agentcore exec -it deploy/agentcore -- python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://neo4j:7687', auth=('neo4j', 'password'))
with driver.session() as session:
    print(session.run('RETURN 1').single()[0])
"
```

#### 4. GDS Plugin Not Available

**Symptom:** `Unknown function 'gds.similarity.cosine'`

**Solution:**
```bash
# Check if GDS plugin is installed
kubectl -n agentcore exec -it neo4j-0 -- ls -la /plugins/ | grep gds

# Verify plugin is loaded
kubectl -n agentcore exec -it neo4j-0 -- cypher-shell -u neo4j -p password \
  "RETURN gds.version() AS version"

# If not loaded, reinstall
kubectl -n agentcore delete pod neo4j-0
# Init container will reinstall plugins on restart
```

#### 5. Memory Store Fallback to In-Memory

**Symptom:** Logs show `using in-memory storage`

**Solution:**
Check initialization order:
1. Verify databases are healthy before app starts
2. Check `depends_on` in docker-compose.dev.yml
3. Review application startup logs for initialization errors

```bash
docker compose -f docker-compose.dev.yml logs a2a-protocol | grep -i "initialized\|failed\|fallback"
```

#### 6. High Memory Usage in GraphOptimizer

**Symptom:** OOMKilled during optimization

**Solution:**
```bash
# Reduce batch size
curl -X POST http://localhost:8001/api/v1/jsonrpc \
  -d '{
    "jsonrpc": "2.0",
    "method": "memory.run_memify",
    "params": {
      "batch_size": 50  # Reduce from default 100
    },
    "id": 1
  }'

# Increase Neo4j memory limits
kubectl -n agentcore patch statefulset neo4j -p '{
  "spec": {"template": {"spec": {"containers": [{"name": "neo4j", "resources": {"limits": {"memory": "12Gi"}}}]}}}
}'
```

### Diagnostic Commands

```bash
# Full system status
docker compose -f docker-compose.dev.yml ps

# Application logs
docker compose -f docker-compose.dev.yml logs -f a2a-protocol

# Database connectivity test
docker compose -f docker-compose.dev.yml exec a2a-protocol python -c "
import asyncio
from agentcore.a2a_protocol.services.memory.storage_backend import get_storage_backend
backend = get_storage_backend()
print('Storage backend initialized:', backend._initialized)
"

# Qdrant cluster status
curl http://localhost:6333/cluster

# Neo4j database status
docker exec $(docker ps -qf "name=neo4j") cypher-shell -u neo4j -p password "CALL dbms.listDatabases()"
```

---

## Next Steps Checklist

### Pre-Deployment

- [ ] Review and update all environment variables in `.env` or Kubernetes secrets
- [ ] Change default Neo4j password (`password` -> secure password)
- [ ] Configure Qdrant API key for production
- [ ] Set `DEBUG=false` for production
- [ ] Generate secure `JWT_SECRET_KEY`
- [ ] Review resource limits based on expected workload

### Infrastructure Setup

- [ ] Deploy Kubernetes namespace (`k8s/namespace.yaml`)
- [ ] Deploy PostgreSQL with pgvector extension
- [ ] Deploy Redis cluster/sentinel for HA
- [ ] Deploy Qdrant StatefulSet (`k8s/qdrant/`)
- [ ] Deploy Neo4j StatefulSet with GDS plugin (`k8s/neo4j/`)
- [ ] Configure persistent storage with appropriate storage class
- [ ] Set up backup/restore procedures for all databases

### Application Changes

- [ ] Update `memory_jsonrpc.py` to remove mock implementation (see Step 3)
- [ ] Run `uv sync` to ensure all dependencies are installed
- [ ] Run all unit tests: `uv run pytest tests/unit/services/test_memory_jsonrpc.py`
- [ ] Run integration tests with real databases
- [ ] Verify GraphOptimizer performance (<5s per 1000 entities)
- [ ] Test error handling and fallback behavior

### Validation

- [ ] Store test memory via JSON-RPC
- [ ] Retrieve memory with semantic search
- [ ] Run Memify optimization successfully
- [ ] Verify Qdrant collection has vectors
- [ ] Verify Neo4j has entities and relationships
- [ ] Check application logs for no warnings/errors
- [ ] Monitor resource utilization during optimization

### Production Hardening

- [ ] Configure TLS for all database connections
- [ ] Set up monitoring dashboards (Prometheus + Grafana)
- [ ] Configure alerting for storage backends
- [ ] Implement backup automation
- [ ] Document disaster recovery procedures
- [ ] Load test with expected production volume
- [ ] Set up log aggregation (ELK/Loki)

### Post-Deployment

- [ ] Monitor error rates via Prometheus metrics
- [ ] Track memory service performance (query times, optimization duration)
- [ ] Review duplicate rates after Memify runs
- [ ] Schedule regular Memify optimizations (e.g., nightly)
- [ ] Plan capacity for growth (storage, memory, CPU)

---

## References

- [GraphOptimizer Implementation](../../src/agentcore/a2a_protocol/services/memory/graph_optimizer.py)
- [StorageBackendService](../../src/agentcore/a2a_protocol/services/memory/storage_backend.py)
- [Memory JSON-RPC Methods](../../src/agentcore/a2a_protocol/services/memory_jsonrpc.py)
- [Docker Compose Development](../../docker-compose.dev.yml)
- [Kubernetes Manifests](../../k8s/)
- [Neo4j GDS Documentation](https://neo4j.com/docs/graph-data-science/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
