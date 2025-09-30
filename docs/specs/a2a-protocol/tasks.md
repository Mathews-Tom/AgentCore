# Tasks: A2A Protocol Layer

**From:** `spec.md` + `plan.md`
**Timeline:** 6 weeks, 3 sprints
**Team:** 2-3 developers (1 senior, 1-2 mid-level)
**Created:** 2025-09-27

## Summary

- Total tasks: 15
- Estimated effort: 65 story points
- Critical path duration: 6 weeks
- Key risks: Protocol complexity, JSON-RPC edge cases, WebSocket scaling

## Phase Breakdown

### Phase 1: Foundation (Sprint 1, 13 story points)

**Goal:** Establish core FastAPI application with JSON-RPC 2.0 support
**Deliverable:** Working JSON-RPC endpoint with basic agent registration

#### Tasks

**[A2A-001] Setup FastAPI Application Structure** (Completed)

- **Description:** Initialize FastAPI app with proper async configuration, project structure, and development environment
- **Acceptance:**
  - [x] FastAPI app runs on localhost:8001
  - [x] Uvicorn/Gunicorn configuration working
  - [x] Basic health check endpoint responds
  - [x] Docker container builds successfully
- **Effort:** 3 story points (2-3 days)
- **Owner:** Senior Developer
- **Dependencies:** None
- **Priority:** P0 (Blocker)

**[A2A-002] Implement JSON-RPC 2.0 Core** (Completed)

- **Description:** Create JSON-RPC 2.0 message parsing, validation, and response handling
- **Acceptance:**
  - [x] Parses valid JSON-RPC 2.0 requests
  - [x] Handles batch requests correctly
  - [x] Returns proper error codes for malformed requests
  - [x] Supports notifications (no response)
- **Effort:** 5 story points (3-5 days)
- **Owner:** Senior Developer
- **Dependencies:** A2A-001
- **Priority:** P0 (Critical)

**[A2A-003] Basic Agent Card Management** (Completed)

- **Description:** Implement AgentCard data model with CRUD operations and validation
- **Acceptance:**
  - [x] AgentCard Pydantic model with all required fields
  - [x] Agent registration endpoint working
  - [x] Agent listing and lookup functionality
  - [x] Validation for capabilities and requirements
- **Effort:** 5 story points (3-5 days)
- **Owner:** Mid-level Developer
- **Dependencies:** A2A-002
- **Priority:** P0 (Critical)

### Phase 2: Protocol Features (Sprint 2, 21 story points)

**Goal:** Complete task management and message routing capabilities
**Deliverable:** Full task lifecycle with agent assignment and message routing

#### Tasks

**[A2A-004] Task Management System** (Completed)

- **Description:** Implement TaskDefinition and TaskExecution models with complete lifecycle management
- **Acceptance:**
  - [x] TaskDefinition model with dependencies and requirements
  - [x] Task state transitions (pending→running→completed/failed)
  - [x] Task assignment to capable agents
  - [x] Dependency tracking and validation
- **Effort:** 8 story points (5-8 days)
- **Owner:** Senior Developer
- **Dependencies:** A2A-003
- **Priority:** P0 (Critical)

**[A2A-005] Message Envelope & Routing** (Completed)

- **Description:** Create intelligent message routing based on agent capabilities and load balancing
- **Acceptance:**
  - [x] MessageEnvelope with headers and metadata
  - [x] Capability-based routing algorithm
  - [x] Load balancing across agent instances
  - [x] Message queuing for offline agents
- **Effort:** 5 story points (3-5 days)
- **Owner:** Mid-level Developer
- **Dependencies:** A2A-003
- **Priority:** P1 (High)
- **Notes:** Full routing service with 4 strategies (round-robin, least-loaded, random, capability-match), circuit breaker, message queuing with priority, and 8 JSON-RPC methods

**[A2A-006] Protocol Security Layer** (Completed)

- **Description:** Implement JWT authentication, request signing, and rate limiting
- **Acceptance:**
  - [x] JWT token generation and validation for agents
  - [x] Request signing with RSA keys
  - [x] Rate limiting per agent (1000 req/min)
  - [x] Input validation and sanitization
- **Effort:** 8 story points (5-8 days)
- **Owner:** Senior Developer
- **Dependencies:** A2A-001
- **Priority:** P0 (Critical)
- **Notes:** Full security service with JWT (HS256/RSA256), RSA request signing with replay attack prevention, rate limiting with sliding window, RBAC with 3 roles (agent/service/admin), input validation, and 11 JSON-RPC methods

### Phase 3: Advanced Features (Sprint 3, 31 story points)

**Goal:** Production-ready features with monitoring and real-time capabilities
**Deliverable:** Complete A2A protocol implementation with WebSocket support

#### Tasks

**[A2A-007] Event System & Notifications** (Completed)

- **Description:** Async event publishing with WebSocket support for real-time notifications
- **Acceptance:**
  - [x] Event publishing for task status changes
  - [x] WebSocket connections for real-time updates
  - [x] Event subscription management
  - [x] Dead letter queue for failed events
- **Effort:** 5 story points (3-5 days)
- **Owner:** Mid-level Developer
- **Dependencies:** A2A-004
- **Priority:** P1 (High)
- **Notes:** Full event system with 17 event types (agent, task, routing, system), WebSocket endpoint `/api/v1/ws/events`, event subscriptions with filters and TTL, dead letter queue with retry mechanism, event history replay (1000 events), event hooks for custom processing, and 9 JSON-RPC methods

**[A2A-008] Health Monitoring & Discovery** (Completed)

- **Description:** Agent health checks, service discovery, and circuit breaker implementation
- **Acceptance:**
  - [x] Agent health check endpoints
  - [x] Service discovery integration
  - [x] Circuit breaker for failing agents
  - [x] Metrics collection (response time, success rate)
- **Effort:** 5 story points (3-5 days)
- **Owner:** Mid-level Developer
- **Dependencies:** A2A-003
- **Priority:** P1 (High)
- **Notes:** Full health monitoring service with periodic health checks (60s interval), consecutive failure tracking (threshold 3), automatic agent status updates (ACTIVE/ERROR), health metrics persistence in database with response time tracking, 8 JSON-RPC methods (health.check_agent, health.check_all, health.get_history, health.get_unhealthy, health.get_stats, discovery.find_agents, discovery.get_agent, discovery.list_capabilities). Circuit breaker already implemented in message_router.py from A2A-005. Enhanced health endpoints with database connectivity checks.

**[A2A-009] PostgreSQL Integration** (Completed)

- **Description:** Replace in-memory storage with PostgreSQL for persistence
- **Acceptance:**
  - [x] Database schema migrations
  - [x] Agent and task persistence
  - [x] Connection pooling with pgBouncer
  - [x] Database performance optimization
- **Effort:** 8 story points (5-8 days)
- **Owner:** Senior Developer
- **Dependencies:** A2A-004
- **Priority:** P0 (Critical)
- **Notes:** Full PostgreSQL integration with SQLAlchemy async (asyncpg driver), 9 database models (AgentDB, TaskDB, AgentHealthMetricDB, MessageQueueDB, EventSubscriptionDB, SecurityTokenDB, RateLimitDB, AgentPublicKeyDB), Alembic migrations with initial schema (revision 001), Repository pattern (AgentRepository, TaskRepository, HealthMetricRepository) for database operations, connection pooling (size 10, max overflow 20, timeout 30s, recycle 3600s), database health checks integrated in FastAPI lifespan and health endpoints, comprehensive indexes (GIN for JSON capabilities, composite indexes for common queries), enum types for AgentStatus and TaskStatus, JSON columns for flexible metadata storage.

**[A2A-010] Integration Testing & Performance** (Completed)

- **Description:** Comprehensive testing suite and performance validation
- **Acceptance:**
  - [x] Integration tests for all endpoints
  - [x] Load testing for 1000+ concurrent connections
  - [x] Performance optimization and profiling
  - [x] API documentation with OpenAPI
- **Effort:** 8 story points (5-8 days)
- **Owner:** Senior Developer
- **Dependencies:** A2A-009
- **Priority:** P1 (High)
- **Notes:** Comprehensive test suite with pytest-asyncio, integration tests for JSON-RPC core (9 tests covering protocol compliance, batch requests, notifications), agent lifecycle tests (7 tests), task lifecycle tests (8 tests), load testing with Locust (simulates 1000+ concurrent connections with realistic workload patterns), pytest.ini configured with 80% coverage threshold, FastAPI automatic OpenAPI documentation at `/docs` and `/redoc`, test fixtures for database sessions and async HTTP clients. Performance optimizations: uvloop for better async performance, connection pooling, database query optimization with indexes.

**[A2A-011] Production Deployment** (Completed)

- **Description:** Docker hardening, monitoring setup, and production configuration
- **Acceptance:**
  - [x] Hardened Docker images
  - [x] Prometheus metrics integration
  - [x] Production-ready logging
  - [x] Kubernetes deployment manifests
- **Effort:** 5 story points (3-5 days)
- **Owner:** Senior Developer
- **Dependencies:** A2A-010
- **Priority:** P1 (High)
- **Notes:** Hardened multi-stage Dockerfile with non-root user (UID 1000), minimal attack surface (python:3.12-slim), security labels, proper healthchecks (30s interval, 30s start period), production CMD with 4 uvicorn workers and uvloop. Kubernetes manifests in k8s/ directory: namespace, configmap for environment variables, secrets for sensitive data, deployment with 3 replicas, rolling updates, pod anti-affinity, resource limits, security context (non-root, read-only filesystem, seccomp, dropped capabilities), init container for database migrations, liveness/readiness probes. HPA for autoscaling (3-10 replicas based on CPU 70%/memory 80%), ClusterIP service, headless service for StatefulSet-like discovery, ServiceAccount with RBAC, ServiceMonitor for Prometheus scraping. Prometheus instrumentation with prometheus-fastapi-instrumentator exposing /metrics endpoint. Structlog for production-ready JSON logging.

## Critical Path

```text
A2A-001 → A2A-002 → A2A-003 → A2A-004 → A2A-009 → A2A-010
  (3d)      (5d)      (5d)      (8d)      (8d)      (8d)
                            [41 days total]
```

**Bottlenecks:**

- A2A-004: Task management complexity (highest risk)
- A2A-009: Database integration and performance
- A2A-006: Security implementation complexity

**Parallel Tracks:**

- Security: A2A-006 (can start after A2A-001)
- Routing: A2A-005 (parallel with A2A-004)
- Monitoring: A2A-008 (parallel with A2A-007)

## Quick Wins (Week 1-2)

1. **[A2A-001] FastAPI Setup** - Unblocks all development
2. **[A2A-002] JSON-RPC Core** - Validates protocol feasibility
3. **[A2A-003] Agent Registration** - Demonstrates core functionality

## Risk Mitigation

| Task | Risk | Mitigation | Contingency |
|------|------|------------|-------------|
| A2A-004 | Task dependency complexity | Start with simple linear dependencies | Implement advanced dependencies in v2 |
| A2A-006 | Security implementation delays | Use existing JWT libraries | Simplified auth for initial release |
| A2A-009 | Database performance issues | Early load testing | Redis fallback for high-frequency data |

## Testing Strategy

### Automated Testing Tasks

- **[A2A-012] Unit Test Framework** (3 SP) - Sprint 1
- **[A2A-013] Integration Tests** (5 SP) - Sprint 2-3
- **[A2A-014] Load Testing** (5 SP) - Sprint 3
- **[A2A-015] Security Testing** (3 SP) - Sprint 3

### Quality Gates

- 90% code coverage required
- All JSON-RPC edge cases tested
- 1000+ concurrent connection validation
- Security audit passed

## Team Allocation

**Senior Developer (1 FTE)**

- Core protocol implementation (A2A-001, A2A-002, A2A-004, A2A-006)
- Database integration (A2A-009)
- Performance optimization (A2A-010)

**Mid-level Developer (1-2 FTE)**

- Agent management (A2A-003)
- Message routing (A2A-005)
- Event system (A2A-007, A2A-008)

## Sprint Planning

**2-week sprints, 20-25 SP velocity per developer**

| Sprint | Focus | Story Points | Key Deliverables |
|--------|-------|--------------|------------------|
| Sprint 1 | Foundation | 13 SP | FastAPI app, JSON-RPC, agent registration |
| Sprint 2 | Core Features | 21 SP | Task management, routing, security |
| Sprint 3 | Production | 31 SP | Events, monitoring, database, testing |

## Task Import Format

CSV export for project management tools:

```csv
ID,Title,Description,Estimate,Priority,Assignee,Dependencies,Sprint
A2A-001,Setup FastAPI App,Initialize FastAPI application...,3,P0,Senior Dev,,1
A2A-002,JSON-RPC Core,Implement JSON-RPC 2.0 parsing...,5,P0,Senior Dev,A2A-001,1
A2A-003,Agent Card Management,Create AgentCard model and CRUD...,5,P0,Mid-level Dev,A2A-002,1
A2A-004,Task Management,Implement task lifecycle...,8,P0,Senior Dev,A2A-003,2
A2A-005,Message Routing,Create intelligent routing...,5,P1,Mid-level Dev,A2A-003,2
A2A-006,Security Layer,JWT auth and request signing...,8,P0,Senior Dev,A2A-001,2
A2A-007,Event System,Async events and WebSocket...,5,P1,Mid-level Dev,A2A-004,3
A2A-008,Health Monitoring,Agent health checks...,5,P1,Mid-level Dev,A2A-003,3
A2A-009,PostgreSQL Integration,Database persistence...,8,P0,Senior Dev,A2A-004,3
A2A-010,Integration Testing,Testing and performance...,8,P1,Senior Dev,A2A-009,3
A2A-011,Production Deployment,Docker and monitoring...,5,P1,Senior Dev,A2A-010,3
```

## Appendix

**Estimation Method:** Planning Poker with A2A protocol expertise
**Story Point Scale:** Fibonacci (1,2,3,5,8,13,21)
**Definition of Done:**

- Code reviewed and approved
- Unit tests written and passing (90% coverage)
- Integration tests validate JSON-RPC compliance
- Documentation updated
- Security review completed
- Deployed to staging environment
