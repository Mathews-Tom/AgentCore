# Gateway Layer - Technical Breakdown

**Created:** 2025-09-27
**Sources:** [spec](docs/specs/gateway-layer/spec.md) | [plan](docs/specs/gateway-layer/plan.md) | [tasks](docs/specs/gateway-layer/tasks.md)

---

## Quick Reference

```yaml
complexity: Medium
risk_level: Medium
team_size: 2-3 engineers
duration: 6 weeks
dependencies:
  - FastAPI framework (high-performance async web serving)
  - Redis (rate limiting and session management)
  - OAuth 3.0 providers (enterprise SSO)
  - Load balancer (production deployment)
```

## Component Overview

**Purpose:** High-performance, secure API gateway built on FastAPI serving as the unified entry point for all external interactions with AgentCore, handling HTTP/WebSocket endpoints, middleware processing, request routing, authentication, and cross-cutting concerns.

**Key Capabilities:**

- FastAPI-based HTTP endpoints with automatic OpenAPI documentation generation
- WebSocket and Server-Sent Events support for real-time agent communication
- JWT authentication with OAuth 3.0 and enterprise SSO integration
- Intelligent request routing and load balancing across backend services
- Comprehensive middleware for rate limiting, security, and monitoring

**Success Metrics:**

- **Performance:** 60,000+ requests per second per instance
- **Latency:** <5ms p95 latency for API gateway routing overhead
- **Availability:** 99.99% uptime with automatic failover and load balancing
- **Security:** Zero security breaches, 100% request authentication/authorization

## System Context

```mermaid
graph LR
    A[External Clients] -->|HTTP/WebSocket| B[Gateway Layer]
    B -->|Agent Requests| C[Agent Runtime]
    B -->|Workflow API| D[Orchestration Engine]
    B -->|Integration API| E[Integration Layer]
    B -->|Protocol API| F[A2A Protocol]
    B -->|Optimization API| G[DSPy Engine]
    H[OAuth Providers] -->|Authentication| B
    I[Enterprise SSO] -->|SAML/LDAP| B
    B -->|Sessions| J[(Redis)]
    B -->|Metrics| K[Monitoring Stack]
```

**Integration Points:**

1. **Upstream:** External applications, mobile clients, web interfaces, enterprise systems
2. **Downstream:** All AgentCore backend services for request routing and processing
3. **Security:** OAuth 3.0 providers, enterprise identity systems, API key management

## Architecture Design

### Component Structure

```mermaid
graph TD
    subgraph "Gateway Layer"
        API[FastAPI Application]
        WS[WebSocket Manager]
        SSE[Server-Sent Events]

        subgraph "Authentication"
            JWT[JWT Handler]
            OAUTH[OAuth 3.0]
            SSO[Enterprise SSO]
            APIKEY[API Key Auth]
        end

        subgraph "Middleware Stack"
            RATE[Rate Limiter]
            CORS[CORS Handler]
            VALIDATE[Request Validator]
            TRANSFORM[Response Transform]
            AUDIT[Audit Logger]
        end

        subgraph "Request Router"
            ROUTE[Service Router]
            BALANCE[Load Balancer]
            HEALTH[Health Monitor]
            CIRCUIT[Circuit Breaker]
        end
    end

    API --> JWT
    API --> RATE
    API --> CORS
    API --> VALIDATE

    WS --> JWT
    WS --> RATE

    ROUTE --> BALANCE
    BALANCE --> HEALTH
    HEALTH --> CIRCUIT

    API --> ROUTE
    WS --> ROUTE
    SSE --> ROUTE
```

### Key Modules

**Module 1: Authentication & Authorization**

- **Responsibility:** JWT token management, OAuth 3.0 flows, enterprise SSO integration
- **Dependencies:** OAuth providers, Redis for session state, enterprise identity systems
- **Complexity:** High

**Module 2: Request Routing Engine**

- **Responsibility:** Service discovery, load balancing, health checks, circuit breakers
- **Dependencies:** Service registry, health check endpoints, backend services
- **Complexity:** Medium

**Module 3: Middleware Pipeline**

- **Responsibility:** Rate limiting, CORS, validation, transformation, audit logging
- **Dependencies:** Redis for distributed state, monitoring systems
- **Complexity:** Medium

**Module 4: Real-time Communication**

- **Responsibility:** WebSocket connection management, Server-Sent Events streaming
- **Dependencies:** Event streaming system, connection pooling
- **Complexity:** Medium

## Interface Contracts

### REST API

```yaml
POST /api/v1/auth/token:
  Request:
    - grant_type: enum ["password", "client_credentials", "authorization_code"]
    - username: string (conditional)
    - password: string (conditional)
    - client_id: string (required)
    - client_secret: string (conditional)
    - scope: string (optional)
  Response:
    - access_token: string (JWT)
    - token_type: "Bearer"
    - expires_in: int (seconds)
    - refresh_token: string (optional)
    - scope: string
  Errors:
    - 400: Invalid request parameters
    - 401: Authentication failed
    - 429: Rate limit exceeded

GET /api/v1/health:
  Response:
    - status: "healthy" | "degraded" | "unhealthy"
    - timestamp: ISO8601
    - services: object (backend service status)
    - version: string
  Errors:
    - 503: Service unhealthy

POST /api/v1/agents/{id}/tasks:
  Headers:
    - Authorization: Bearer <jwt_token>
    - Content-Type: application/json
    - X-Request-ID: UUID (optional)
  Request:
    - task_type: string (required)
    - input_data: object (required)
    - priority: enum ["low", "normal", "high"] (default: "normal")
  Response:
    - task_id: UUID
    - status: "created"
    - agent_id: string
    - estimated_completion: ISO8601
  Errors:
    - 401: Unauthorized
    - 404: Agent not found
    - 422: Invalid task data
    - 429: Rate limit exceeded
    - 503: Backend service unavailable
```

### WebSocket Protocol

```yaml
Connection: /ws/agents/{agent_id}
Protocol: JSON messages with authentication
Authentication: JWT token in query parameter or header
Connection Flow:
  1. Client connects with JWT token
  2. Server validates token and establishes session
  3. Client subscribes to events (task updates, status changes)
  4. Server streams real-time updates
  5. Heartbeat every 30 seconds for connection health

Message Format:
  - type: "subscribe" | "unsubscribe" | "event" | "heartbeat"
  - timestamp: ISO8601
  - data: object (type-specific payload)
  - request_id: UUID (for correlation)
```

### Server-Sent Events

```yaml
Endpoint: /api/v1/events/stream
Authentication: JWT token in Authorization header
Headers:
  - Content-Type: text/event-stream
  - Cache-Control: no-cache
  - Connection: keep-alive

Event Format:
  - event: string (event type)
  - data: JSON object
  - id: string (for replay)
  - retry: int (reconnection time in ms)

Supported Events:
  - agent.status_changed
  - task.completed
  - workflow.started
  - system.alert
```

### Error Response Format

```yaml
Error Response:
  - error:
      code: string (machine-readable error code)
      message: string (human-readable description)
      details: object (additional context)
      request_id: UUID (for correlation)
      timestamp: ISO8601
      documentation_url: string (optional)

Common Error Codes:
  - INVALID_REQUEST: Malformed request data
  - AUTHENTICATION_FAILED: Invalid or expired credentials
  - AUTHORIZATION_DENIED: Insufficient permissions
  - RATE_LIMIT_EXCEEDED: Too many requests
  - SERVICE_UNAVAILABLE: Backend service down
  - VALIDATION_ERROR: Request validation failed
```

## Implementation Details

### Technology Stack

**Web Framework:** FastAPI 0.104+ with Uvicorn/Gunicorn for production deployment
**Authentication:** OAuth 3.0 with enterprise SSO support, JWT with RSA-256 signing
**Rate Limiting:** Redis-based distributed rate limiting with sliding windows
**Load Balancing:** Nginx or cloud load balancer with health-check based routing
**Monitoring:** Prometheus metrics with Grafana dashboards
**Rationale:** FastAPI provides industry-leading async performance with 60,000+ req/sec capability, OAuth 3.0 offers enhanced security over 2.0, Redis enables sub-millisecond rate limiting decisions

### Design Patterns

1. **Gateway Pattern** - Unified entry point for all external API access
2. **Middleware Chain** - Composable request/response processing pipeline
3. **Circuit Breaker** - Fail-fast protection for backend service failures
4. **Rate Limiting** - Token bucket algorithm with Redis for distributed enforcement
5. **Observer Pattern** - Real-time event streaming to connected clients

### Configuration

```yaml
# Environment variables
AUTH_SECRET_KEY: ${VAULT_JWT_SECRET}
OAUTH_CLIENT_ID: ${OAUTH_CLIENT_ID}
OAUTH_CLIENT_SECRET: ${VAULT_OAUTH_SECRET}
REDIS_URL: redis://redis-cluster:6379
DATABASE_URL: postgresql://gateway:${DB_PASSWORD}@postgres:5432/gateway
BACKEND_SERVICES: agent-runtime:8001,orchestration:8002,integration:8003
RATE_LIMIT_REQUESTS_PER_MINUTE: 1000
MAX_WEBSOCKET_CONNECTIONS: 10000
JWT_EXPIRATION_HOURS: 24
CORS_ALLOWED_ORIGINS: https://app.agentcore.ai,https://dashboard.agentcore.ai
```

## Testing Strategy

### Unit Tests (Target: 95% coverage)

**Critical Paths:**

- JWT token generation, validation, and expiration handling
- OAuth 3.0 authorization flows and token exchange
- Rate limiting algorithms and distributed enforcement
- Request routing and load balancing logic
- WebSocket connection management and message handling

**Tools:** pytest with asyncio support, pytest-cov for coverage, httpx for async HTTP testing

### Integration Tests

**Scenarios:**

1. End-to-end authentication flows with real OAuth providers
2. Request routing to backend services with health monitoring
3. WebSocket connection lifecycle with authentication
4. Rate limiting enforcement across multiple instances
5. Circuit breaker activation and recovery

**Tools:** pytest-asyncio, testcontainers for Redis/PostgreSQL, websocket-client for WebSocket testing

### End-to-End Tests

**User Flows:**

1. **API Authentication:** Client registration → Token request → API access → Token refresh
2. **Real-time Communication:** WebSocket connection → Event subscription → Real-time updates
3. **Request Processing:** Request → Authentication → Routing → Backend processing → Response
4. **Failure Handling:** Backend failure → Circuit breaker → Graceful degradation → Recovery

**Tools:** pytest-playwright for browser testing, k6 for load testing

### Performance Tests

**Load Scenarios:**

- 60,000+ HTTP requests per second with authentication
- 10,000+ concurrent WebSocket connections
- Rate limiting under high traffic (burst and sustained)
- Backend service failover and recovery

**SLA Targets:**

- API response latency: <5ms p95 routing overhead
- WebSocket connection establishment: <100ms p95
- Authentication latency: <50ms p95 for token validation
- Rate limiting decision: <1ms p95

**Tools:** k6 for HTTP/WebSocket load testing, Grafana for real-time monitoring

### Security Tests

- OAuth flow security and token validation bypass attempts
- JWT token tampering and replay attack prevention
- Rate limiting bypass through distributed requests
- WebSocket authentication and authorization validation
- OWASP Top 10 API security testing with ZAP proxy

## Operational Concerns

### Infrastructure

```yaml
Compute:
  - Container: 4 vCPU, 8GB RAM (WebSocket connection overhead)
  - Auto-scaling: 3-10 instances based on request rate and connections
  - Health checks: /health, /ready, /metrics endpoints

Storage:
  - Redis Cluster: 3 masters for rate limiting and sessions
  - PostgreSQL: 50GB for audit logs and configuration
  - Logs: 30-day retention with structured JSON format

Networking:
  - Load balancer: Layer 7 with sticky sessions for WebSocket
  - TLS 1.3: All external communication with HSTS headers
  - CDN: Static asset delivery and edge caching
```

### Monitoring

**Metrics:**

- Gateway KPIs: request_rate, response_latency, error_rate, authentication_success_rate
- WebSocket metrics: connection_count, message_throughput, connection_duration
- Rate limiting: requests_blocked, limit_violations, quota_utilization
- Infrastructure: CPU, memory, Redis performance, backend service health

**Alerts:**

- Response latency >100ms p95 for 5 minutes
- Error rate >1% for 2 minutes
- WebSocket connections >8000 (80% capacity)
- Rate limiting blocking >10% of requests
- Backend service circuit breaker open

**Dashboards:**

- API Gateway Health: Real-time performance and availability metrics
- Authentication Analytics: Login success rates, OAuth flow performance
- Traffic Analysis: Request patterns, geographic distribution, rate limiting

### Security

**Transport Security:** TLS 1.3 with HSTS, certificate pinning for critical endpoints
**Authentication:** JWT with RSA-256 signing, OAuth 3.0 with PKCE for enhanced security
**Authorization:** RBAC with fine-grained permissions, tenant isolation
**Data Protection:**

- Request/response logging with PII redaction
- Audit trails for all authentication and authorization events
- Rate limiting to prevent abuse and DDoS attacks

**Compliance:**

- GDPR compliance for user data handling
- SOC2 Type II controls for enterprise customers
- OAuth 3.0 security best practices implementation

### Scaling Strategy

**Horizontal:**

- Stateless gateway design with session state in Redis
- Auto-scaling based on CPU utilization and connection count
- Geographic distribution with latency-based routing

**Vertical:**

- Connection pooling optimization for backend services
- Redis connection multiplexing and pipelining
- WebSocket connection batching and compression

## Risk Analysis

### Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| OAuth 3.0 provider integration complexity | High | Medium | Extensive testing, fallback to OAuth 2.0, provider redundancy |
| WebSocket connection memory leaks | Medium | Medium | Connection lifecycle monitoring, automatic cleanup, memory limits |
| Rate limiting Redis cluster split-brain | High | Low | Proper cluster configuration, monitoring, graceful degradation |
| Backend service discovery failures | High | Medium | Service mesh integration, health check redundancy, circuit breakers |
| TLS certificate expiration | Medium | Low | Automated renewal, monitoring, backup certificates |

### Dependency Risks

| Dependency | Risk | Mitigation |
|------------|------|------------|
| FastAPI framework | Breaking changes in updates | Pin versions, automated testing, gradual upgrades |
| OAuth 3.0 providers | Service availability and reliability | Multiple provider support, fallback authentication |
| Redis cluster | Data loss or performance degradation | Proper persistence, monitoring, backup strategies |

### Business Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Performance targets not met under load | High | Low | Early load testing, performance optimization, horizontal scaling |
| Security vulnerabilities in authentication | High | Low | Regular security audits, penetration testing, automated scanning |

## Development Workflow

### Local Setup

```bash
# Clone and setup environment
git clone <repo>
cd agentcore
uv sync

# Start infrastructure dependencies
docker-compose -f docker-compose.dev.yml up -d redis postgres

# Run database migrations
uv run alembic upgrade head

# Start gateway layer in development mode
uv run uvicorn src.gateway_layer.main:app --reload --port 8080

# Verify setup
curl http://localhost:8080/health
curl http://localhost:8080/api/v1/openapi.json
```

### Code Quality

- **Linting:** Ruff with security rules, async/await pattern validation
- **Type Safety:** mypy strict mode with FastAPI type annotations
- **Pre-commit:** Black formatting, isort, security scanning with bandit
- **Documentation:** Automatic OpenAPI generation, comprehensive docstrings

### Deployment Pipeline

```yaml
Stages:
  1. Build: UV package build, Docker multi-stage build with security scanning
  2. Test: Unit, integration, security tests (95% coverage required)
  3. OAuth Testing: Integration with real OAuth providers
  4. Performance Testing: Load testing with 60,000+ req/sec validation
  5. Deploy Staging: Automated deployment with smoke tests
  6. Security Scanning: OWASP ZAP, dependency vulnerability scanning
  7. Deploy Production: Manual approval with blue-green deployment
```

## Implementation Checklist

**Phase 1: Core Gateway (Weeks 1-2)**

- [ ] FastAPI application with proper async configuration and production setup
- [ ] JWT authentication system with RSA-256 signing and key rotation
- [ ] OAuth 3.0 integration with enterprise SSO support
- [ ] Basic request routing and middleware pipeline
- [ ] Health check endpoints and monitoring integration

**Phase 2: Advanced Features (Weeks 3-4)**

- [ ] Redis-based distributed rate limiting with sliding window algorithms
- [ ] WebSocket and Server-Sent Events support with authentication
- [ ] Request/response middleware for CORS, validation, and transformation
- [ ] Backend service routing with intelligent load balancing
- [ ] Circuit breaker patterns for backend service failures

**Phase 3: Production Features (Weeks 5-6)**

- [ ] Load balancing and health checks with service discovery
- [ ] Performance monitoring and observability with Prometheus metrics
- [ ] Security hardening with TLS 1.3 and comprehensive input validation
- [ ] API documentation and developer portal with interactive explorer
- [ ] Production deployment automation with monitoring and alerting

## References

**Internal:**

- [Gateway Layer Specification](docs/specs/gateway-layer/spec.md)
- [Implementation Plan](docs/specs/gateway-layer/plan.md)
- [Task Breakdown](docs/specs/gateway-layer/tasks.md)

**External:**

- [FastAPI Production Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [OAuth 3.0 Specification](https://datatracker.ietf.org/doc/draft-ietf-oauth-v2-1/)
- [Redis Rate Limiting Patterns](https://redis.io/docs/latest/develop/interact/rate-limiting/)
- [WebSocket Security Best Practices](https://devcenter.heroku.com/articles/websocket-security)
- [API Gateway Security Guidelines](https://owasp.org/www-project-api-security/)
