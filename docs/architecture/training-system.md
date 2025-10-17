# Training System Architecture

**Version:** 1.0
**Last Updated:** 2025-10-17
**Component:** Flow-Based Optimization (Training Infrastructure)

---

## Overview

The AgentCore training system implements GRPO (Group Refined Policy Optimization) for reinforcement learning of agent policies. The architecture follows a distributed, queue-based design with asynchronous job processing.

---

## System Architecture

### High-Level Components

```mermaid
graph TB
    Client[Client Applications]
    API[Training API<br/>FastAPI + JSON-RPC 2.0]
    Auth[Authentication Service<br/>JWT]
    Queue[Job Queue<br/>Redis]
    DB[(Database<br/>PostgreSQL)]
    Workers[Worker Pool<br/>GRPO Trainers]
    Metrics[Metrics<br/>Prometheus]
    Storage[Model Storage<br/>S3]

    Client -->|JSON-RPC| API
    API --> Auth
    API --> Queue
    API --> DB
    Queue --> Workers
    Workers --> DB
    Workers --> Storage
    API --> Metrics
    Workers --> Metrics

    style API fill:#4CAF50
    style Workers fill:#2196F3
    style Queue fill:#FF9800
    style DB fill:#9C27B0
```

### Component Details

#### Training API (FastAPI)
- **Technology:** Python 3.12+, FastAPI, Pydantic
- **Protocol:** JSON-RPC 2.0
- **Port:** 8001
- **Responsibilities:**
  - Request validation and authentication
  - Job creation and queueing
  - Status queries and job management
  - Rate limiting and budget enforcement
  - Metrics export

#### Worker Pool
- **Technology:** Python 3.12+, async/await
- **Deployment:** Kubernetes Deployment with HPA
- **Responsibilities:**
  - Job polling from Redis queue
  - Trajectory collection via agent execution
  - Reward computation with custom functions
  - Policy gradient calculation (GRPO algorithm)
  - Model checkpoint creation
  - Cost tracking and budget enforcement

#### Redis Queue
- **Technology:** Redis Cluster
- **Purpose:** Job queue and caching
- **Queue Structure:**
  - `training_jobs` - Pending job queue (FIFO)
  - `job:{job_id}:status` - Job status cache
  - `job:{job_id}:metrics` - Real-time metrics

#### PostgreSQL Database
- **Technology:** PostgreSQL 15 with asyncpg
- **Schema:**
  - `training_jobs` - Job metadata and configuration
  - `trajectories` - Agent execution trajectories
  - `trajectory_steps` - Individual trajectory steps
  - `checkpoints` - Model checkpoints metadata
  - `training_metrics` - Time-series metrics

---

## Training Workflow

### Job Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Queued: Submit Job
    Queued --> Running: Worker Picks Up
    Running --> Completed: Success
    Running --> Failed: Error
    Running --> Cancelled: User Cancels
    Completed --> [*]
    Failed --> [*]
    Cancelled --> [*]

    Running --> Running: Checkpoint Created

    note right of Queued
        Job in Redis queue
        Waiting for worker
    end note

    note right of Running
        Worker executing GRPO
        Collecting trajectories
        Computing rewards
        Updating policy
    end note

    note right of Completed
        Final checkpoint saved
        Metrics exported
        Resources released
    end note
```

### GRPO Training Loop

```mermaid
flowchart TD
    Start([Start Training Job]) --> Init[Initialize Policy θ]
    Init --> IterStart{Iteration < N?}

    IterStart -->|Yes| Sample[Sample Batch of Queries]
    Sample --> Collect[Collect K Trajectories<br/>per Query]

    Collect --> Reward[Compute Rewards<br/>R = r₀ + γr₁ + γ²r₂ + ...]
    Reward --> Normalize[Normalize Rewards<br/>within Group]

    Normalize --> Advantage[Compute Advantages<br/>A = (R - mean(R)) / std(R)]
    Advantage --> Gradient[Compute Policy Gradient<br/>∇θ = E[A · ∇log π(a|s)]

    Gradient --> Update[Update Policy<br/>θ ← θ + α∇θ]
    Update --> Budget{Budget OK?}

    Budget -->|Yes| Checkpoint{Checkpoint<br/>Interval?}
    Budget -->|No| Cancel[Cancel Job]

    Checkpoint -->|Yes| Save[Save Checkpoint]
    Checkpoint -->|No| IterStart
    Save --> IterStart

    IterStart -->|No| Final[Save Final Checkpoint]
    Final --> End([Training Complete])
    Cancel --> End

    style Start fill:#4CAF50
    style End fill:#4CAF50
    style Cancel fill:#f44336
    style Update fill:#2196F3
    style Reward fill:#FF9800
```

---

## Data Flow

### Request Processing Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API
    participant R as Redis
    participant D as Database
    participant W as Worker
    participant S as Storage

    C->>A: POST /jsonrpc<br/>training.start_grpo
    A->>A: Validate JWT
    A->>A: Validate Config
    A->>D: Create Job Record
    D-->>A: Job ID
    A->>R: Enqueue Job
    A-->>C: Job Created (queued)

    W->>R: Poll Queue
    R-->>W: Job Data
    W->>D: Update Status (running)

    loop Training Iterations
        W->>W: Collect Trajectories
        W->>W: Compute Rewards
        W->>W: Update Policy
        W->>D: Store Metrics

        alt Checkpoint Interval
            W->>S: Save Checkpoint
            W->>D: Store Checkpoint Metadata
        end

        alt Budget Exceeded
            W->>D: Update Status (cancelled)
            W->>W: Stop Training
        end
    end

    W->>S: Save Final Checkpoint
    W->>D: Update Status (completed)

    C->>A: GET /jsonrpc<br/>training.get_status
    A->>D: Query Job Status
    D-->>A: Status + Metrics
    A-->>C: Status Response
```

### Trajectory Collection Flow

```mermaid
flowchart LR
    Query[Query:<br/>User Task] --> Agent[Agent Executor]

    Agent --> Step1[Step 1:<br/>State s₀<br/>Action a₀<br/>Result r₀]
    Step1 --> Step2[Step 2:<br/>State s₁<br/>Action a₁<br/>Result r₁]
    Step2 --> StepN[Step N:<br/>State sₙ<br/>Action aₙ<br/>Result rₙ]

    StepN --> Trajectory[Complete Trajectory:<br/>τ = {s₀,a₀,r₀,...,sₙ,aₙ,rₙ}]
    Trajectory --> Reward[Reward Function:<br/>R(τ) → [0, 1]]

    Reward --> Storage[(Database:<br/>trajectory_steps<br/>trajectories)]

    style Query fill:#4CAF50
    style Trajectory fill:#2196F3
    style Reward fill:#FF9800
    style Storage fill:#9C27B0
```

---

## Database Schema

### Entity Relationship Diagram

```mermaid
erDiagram
    TRAINING_JOBS ||--o{ TRAJECTORIES : generates
    TRAINING_JOBS ||--o{ CHECKPOINTS : creates
    TRAINING_JOBS ||--o{ TRAINING_METRICS : records
    TRAJECTORIES ||--o{ TRAJECTORY_STEPS : contains

    TRAINING_JOBS {
        uuid job_id PK
        string agent_id
        enum status
        jsonb config
        decimal budget_usd
        decimal cost_usd
        timestamp created_at
        timestamp started_at
        timestamp completed_at
    }

    TRAJECTORIES {
        uuid trajectory_id PK
        uuid job_id FK
        string query
        boolean success
        float reward
        float normalized_reward
        float advantage
        int execution_time_ms
        timestamp created_at
    }

    TRAJECTORY_STEPS {
        uuid step_id PK
        uuid trajectory_id FK
        int step_number
        jsonb state
        jsonb action
        jsonb result
        int duration_ms
        timestamp timestamp
    }

    CHECKPOINTS {
        uuid checkpoint_id PK
        uuid job_id FK
        int iteration
        string storage_path
        float validation_accuracy
        boolean is_best
        timestamp created_at
    }

    TRAINING_METRICS {
        uuid metric_id PK
        uuid job_id FK
        int iteration
        float train_loss
        float validation_accuracy
        float avg_reward
        int trajectories_count
        timestamp recorded_at
    }
```

---

## Deployment Architecture

### Kubernetes Architecture

```mermaid
graph TB
    subgraph "Ingress Layer"
        Ingress[Ingress Controller<br/>NGINX]
    end

    subgraph "Application Layer"
        API1[Training API Pod 1]
        API2[Training API Pod 2]
        API3[Training API Pod 3]
        Worker1[Worker Pod 1]
        Worker2[Worker Pod 2]
        WorkerN[Worker Pod N]
    end

    subgraph "Data Layer"
        Redis[Redis Cluster<br/>3 Masters + 3 Replicas]
        PG[PostgreSQL<br/>StatefulSet + PVC]
    end

    subgraph "Monitoring"
        Prom[Prometheus]
        Graf[Grafana]
    end

    subgraph "Storage"
        S3[S3 Bucket<br/>Model Checkpoints]
    end

    Ingress --> API1
    Ingress --> API2
    Ingress --> API3

    API1 --> Redis
    API2 --> Redis
    API3 --> Redis

    API1 --> PG
    API2 --> PG
    API3 --> PG

    Redis --> Worker1
    Redis --> Worker2
    Redis --> WorkerN

    Worker1 --> PG
    Worker2 --> PG
    WorkerN --> PG

    Worker1 --> S3
    Worker2 --> S3
    WorkerN --> S3

    API1 -.->|metrics| Prom
    Worker1 -.->|metrics| Prom
    Prom --> Graf

    style Ingress fill:#4CAF50
    style API1 fill:#2196F3
    style API2 fill:#2196F3
    style API3 fill:#2196F3
    style Worker1 fill:#FF9800
    style Worker2 fill:#FF9800
    style WorkerN fill:#FF9800
    style Redis fill:#f44336
    style PG fill:#9C27B0
```

---

## Scaling Strategy

### Horizontal Pod Autoscaling

**API Pods:**
- Min replicas: 2
- Max replicas: 10
- Scale up: CPU > 70% or Request rate > 100 req/s
- Scale down: CPU < 30% and Request rate < 20 req/s

**Worker Pods:**
- Min replicas: 5
- Max replicas: 50
- Scale up: Queue length > 100 or CPU > 80%
- Scale down: Queue length < 10 and CPU < 30%

**Database:**
- Vertical scaling (increase resources)
- Read replicas for query offloading
- Connection pooling with PgBouncer

**Redis:**
- Cluster mode with 3 masters + 3 replicas
- Auto-failover enabled
- Horizontal scaling by adding shards

---

## Security Architecture

### Authentication Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API
    participant Auth as Auth Service
    participant JWT as JWT Store

    C->>Auth: Login (credentials)
    Auth->>Auth: Validate Credentials
    Auth->>JWT: Generate JWT
    JWT-->>Auth: JWT Token
    Auth-->>C: JWT Token

    C->>A: Request + JWT
    A->>A: Validate JWT Signature
    A->>A: Check Expiration
    A->>A: Verify Permissions

    alt Valid JWT
        A->>A: Process Request
        A-->>C: Success Response
    else Invalid JWT
        A-->>C: 401 Unauthorized
    end
```

### Permission Model

| Permission | Description | Required For |
|------------|-------------|--------------|
| `training:start` | Start training jobs | `training.start_grpo` |
| `training:view` | View job status | `training.get_status` |
| `training:cancel` | Cancel running jobs | `training.cancel` |
| `training:evaluate` | Run evaluations | `training.evaluate` |
| `data:export` | Export trajectories | `training.export_trajectories` |

---

## Monitoring and Observability

### Metrics Architecture

```mermaid
graph LR
    API[API Pods] -->|/metrics| Prom[Prometheus]
    Workers[Worker Pods] -->|/metrics| Prom
    Redis[Redis Exporter] -->|/metrics| Prom
    PG[PostgreSQL Exporter] -->|/metrics| Prom

    Prom --> Graf[Grafana Dashboards]
    Prom --> Alert[AlertManager]

    Alert -->|PagerDuty| PD[On-Call]
    Alert -->|Slack| Slack[#incidents]

    style Prom fill:#FF9800
    style Graf fill:#2196F3
    style Alert fill:#f44336
```

### Key Metrics

**Job Metrics:**
- `training_jobs_active` - Active jobs count
- `training_jobs_completed_total` - Completed jobs counter
- `training_jobs_failed_total` - Failed jobs counter
- `training_job_duration_seconds` - Job duration histogram
- `training_iteration_duration_seconds` - Iteration duration histogram

**System Metrics:**
- `http_requests_total` - API request counter
- `http_request_duration_seconds` - Request latency histogram
- `redis_queue_length` - Queue depth gauge
- `pg_connections_active` - Database connections gauge
- `worker_cpu_usage_percent` - Worker CPU utilization

---

## Cost Optimization

### Budget Enforcement Flow

```mermaid
flowchart TD
    Start([Job Starts]) --> Track[Track API Costs]
    Track --> Check{Cost Check<br/>Every Iteration}

    Check -->|Cost < Budget| Continue[Continue Training]
    Check -->|Cost ≥ Budget| Stop[Stop Training]

    Continue --> Warn{Cost > 75%?}
    Warn -->|Yes| Alert[Send Warning Alert]
    Warn -->|No| Track
    Alert --> Track

    Stop --> Save[Save Checkpoint]
    Save --> Update[Update Status:<br/>budget_exceeded]
    Update --> End([Job Cancelled])

    style Start fill:#4CAF50
    style End fill:#f44336
    style Alert fill:#FF9800
```

---

## Disaster Recovery

### Backup Strategy

```mermaid
graph TB
    subgraph "Primary"
        PG[(PostgreSQL<br/>Primary)]
        Redis[(Redis<br/>Cluster)]
    end

    subgraph "Backups"
        S3Daily[S3: Daily Backups<br/>Retention: 30 days]
        S3Hourly[S3: Hourly Snapshots<br/>Retention: 7 days]
        PGReplica[(PostgreSQL<br/>Read Replica)]
    end

    PG -->|pg_dump| S3Daily
    PG -->|WAL streaming| PGReplica
    Redis -->|BGSAVE| S3Hourly

    PGReplica -.->|Failover| PG

    style PG fill:#9C27B0
    style PGReplica fill:#9C27B0
    style Redis fill:#f44336
```

**RPO (Recovery Point Objective):** 1 hour
**RTO (Recovery Time Objective):** 15 minutes

---

## Performance Characteristics

### Throughput

| Metric | Target | Current |
|--------|--------|---------|
| Job submission rate | 10 jobs/min | 15 jobs/min |
| Trajectory generation | 100 traj/s | 120 traj/s |
| Database writes | 1000 TPS | 800 TPS |
| API latency (p95) | < 500ms | 350ms |

### Resource Requirements

**Per API Pod:**
- CPU: 2 cores (request), 4 cores (limit)
- Memory: 4 GB (request), 8 GB (limit)

**Per Worker Pod:**
- CPU: 4 cores (request), 8 cores (limit)
- Memory: 8 GB (request), 16 GB (limit)

**Database:**
- CPU: 8 cores
- Memory: 32 GB
- Storage: 500 GB SSD

**Redis:**
- CPU: 2 cores per node
- Memory: 8 GB per node
- Storage: 20 GB per node

---

## Technology Stack Summary

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| API Framework | FastAPI | 0.110+ | JSON-RPC endpoints |
| Language | Python | 3.12+ | Core implementation |
| Database | PostgreSQL | 15+ | Data persistence |
| Queue | Redis | 7.0+ | Job queue |
| Container | Docker | 24+ | Containerization |
| Orchestration | Kubernetes | 1.28+ | Deployment |
| Monitoring | Prometheus | 2.45+ | Metrics collection |
| Visualization | Grafana | 10.0+ | Dashboards |
| Storage | S3 | - | Checkpoint storage |

---

## Further Reading

- [Training API Reference](../api/training-api.md)
- [Developer Guide](../guides/training-agents.md)
- [Operational Runbook](../ops/training-runbook.md)
- [Custom Rewards Guide](../guides/custom_rewards.md)

---

**Document Version:** 1.0
**Last Reviewed:** 2025-10-17
**Next Review:** 2025-11-17
