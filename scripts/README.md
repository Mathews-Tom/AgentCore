# Development Scripts

This directory contains scripts for local testing and development.

## Migration Testing

### `test-migrations.sh`

Tests database migrations locally before pushing to CI/CD.

**Prerequisites:**

```bash
# Start PostgreSQL
docker run -d --name postgres-test -p 5432:5432 \
  -e POSTGRES_PASSWORD=agentcore \
  -e POSTGRES_USER=agentcore \
  -e POSTGRES_DB=agentcore_test \
  pgvector/pgvector:pg15

# Start Redis
docker run -d --name redis-test -p 6379:6379 redis:7
```

**Usage:**

```bash
./scripts/test-migrations.sh
```

**What it does:**

1. Checks PostgreSQL and Redis are running
2. Recreates test database
3. Runs all migrations
4. Verifies all enum types exist
5. Tests idempotency by running migrations again

**Cleanup:**

```bash
docker stop postgres-test redis-test
docker rm postgres-test redis-test
```

## Quick Start

```bash
# One-time setup
chmod +x scripts/*.sh

# Before pushing changes
./scripts/test-migrations.sh
```

## See Also

- [Local Workflow Testing Guide](../.docs/LOCAL_WORKFLOW_TESTING.md) - Comprehensive guide
- [Docker Compose Testing](../docker-compose.test.yml) - Full environment testing
