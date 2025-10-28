# Local GitHub Workflow Testing

This guide explains how to test GitHub Actions workflows locally before pushing changes.

## Method 1: Using `act` (Recommended)

[`act`](https://github.com/nektos/act) runs GitHub Actions locally using Docker.

### Installation

**macOS:**
```bash
brew install act
```

**Linux:**
```bash
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

### Basic Usage

```bash
# List all workflows
act -l

# Run a specific workflow
act -W .github/workflows/integration-load-test.yml

# Run a specific job
act -j load-test

# Dry run (show what would be executed)
act -n

# Run with specific event (push, pull_request, etc.)
act push

# Use specific runner image
act -P ubuntu-latest=catthehacker/ubuntu:act-latest
```

### Configuration for AgentCore

Create `.actrc` in the repository root:

```bash
# Use larger Docker images with more tools
-P ubuntu-latest=catthehacker/ubuntu:full-latest

# Pass environment variables
--env-file .env.test

# Reuse containers for faster iterations
--reuse

# Show verbose output
-v
```

### Testing Migration Workflows

```bash
# Test the integration-load-test workflow (includes migrations)
act -j load-test -W .github/workflows/integration-load-test.yml

# Test with secrets (create .secrets file)
act -j load-test --secret-file .secrets
```

### Limitations of `act`

- Some GitHub-specific features may not work (e.g., `github.rest` API calls)
- Services (PostgreSQL, Redis) work but may be slower
- Some actions may require specific runner images
- Resource-intensive workflows may be slow on local machines

---

## Method 2: Docker Compose (Most Accurate)

This method replicates the exact CI environment locally.

### Setup

Create `docker-compose.test.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_USER: agentcore
      POSTGRES_PASSWORD: agentcore
      POSTGRES_DB: agentcore_test
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agentcore"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  test-runner:
    build:
      context: .
      dockerfile: Dockerfile.test
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql://agentcore:agentcore@postgres:5432/agentcore_test
      REDIS_URL: redis://redis:6379
    volumes:
      - .:/app
    working_dir: /app
```

Create `Dockerfile.test`:

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync

CMD ["/bin/bash"]
```

### Usage

```bash
# Start services
docker compose -f docker-compose.test.yml up -d postgres redis

# Run migrations
docker compose -f docker-compose.test.yml run --rm test-runner \
  bash -c "uv run alembic upgrade head"

# Run tests
docker compose -f docker-compose.test.yml run --rm test-runner \
  bash -c "uv run pytest"

# Cleanup
docker compose -f docker-compose.test.yml down -v
```

---

## Method 3: Local Test Script (Fastest for Iterations)

Create a script that mimics workflow steps without Docker overhead.

### Create `scripts/test-migrations.sh`

```bash
#!/bin/bash
set -e

echo "🧪 Testing Database Migrations Locally"
echo "======================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
export DATABASE_URL="${DATABASE_URL:-postgresql://agentcore:agentcore@localhost:5432/agentcore_test}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379}"

# Check if PostgreSQL is running
echo -e "\n${YELLOW}Checking PostgreSQL...${NC}"
if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo -e "${RED}❌ PostgreSQL is not running on localhost:5432${NC}"
    echo "Start it with: docker run -d --name postgres-test -p 5432:5432 -e POSTGRES_PASSWORD=agentcore -e POSTGRES_USER=agentcore -e POSTGRES_DB=agentcore_test pgvector/pgvector:pg15"
    exit 1
fi
echo -e "${GREEN}✅ PostgreSQL is running${NC}"

# Check if Redis is running
echo -e "\n${YELLOW}Checking Redis...${NC}"
if ! redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
    echo -e "${RED}❌ Redis is not running on localhost:6379${NC}"
    echo "Start it with: docker run -d --name redis-test -p 6379:6379 redis:7"
    exit 1
fi
echo -e "${GREEN}✅ Redis is running${NC}"

# Drop and recreate test database
echo -e "\n${YELLOW}Recreating test database...${NC}"
psql -h localhost -U agentcore -d postgres -c "DROP DATABASE IF EXISTS agentcore_test;" 2>/dev/null || true
psql -h localhost -U agentcore -d postgres -c "CREATE DATABASE agentcore_test;" 2>/dev/null || true
echo -e "${GREEN}✅ Database recreated${NC}"

# Run migrations
echo -e "\n${YELLOW}Running Alembic migrations...${NC}"
if uv run alembic upgrade head; then
    echo -e "${GREEN}✅ Migrations successful${NC}"
else
    echo -e "${RED}❌ Migrations failed${NC}"
    exit 1
fi

# Verify migration state
echo -e "\n${YELLOW}Verifying migration state...${NC}"
CURRENT_REV=$(uv run alembic current 2>/dev/null | grep -oE '[a-f0-9]{12}' | head -1)
echo "Current revision: $CURRENT_REV"

# Check enum types exist
echo -e "\n${YELLOW}Verifying enum types...${NC}"
ENUM_TYPES=$(psql -h localhost -U agentcore -d agentcore_test -t -c "SELECT typname FROM pg_type WHERE typtype = 'e' ORDER BY typname;")
echo "Found enum types:"
echo "$ENUM_TYPES"

EXPECTED_ENUMS=("agentstatus" "sessionpriority" "sessionstate" "taskstatus" "workflowstatus")
for enum in "${EXPECTED_ENUMS[@]}"; do
    if echo "$ENUM_TYPES" | grep -q "$enum"; then
        echo -e "${GREEN}✅ $enum${NC}"
    else
        echo -e "${RED}❌ Missing: $enum${NC}"
        exit 1
    fi
done

# Test idempotency - run migrations again
echo -e "\n${YELLOW}Testing migration idempotency...${NC}"
if uv run alembic upgrade head; then
    echo -e "${GREEN}✅ Migrations are idempotent${NC}"
else
    echo -e "${RED}❌ Migrations failed on second run (not idempotent!)${NC}"
    exit 1
fi

echo -e "\n${GREEN}🎉 All migration tests passed!${NC}"
```

Make it executable:
```bash
chmod +x scripts/test-migrations.sh
```

### Usage

```bash
# Start services manually
docker run -d --name postgres-test -p 5432:5432 \
  -e POSTGRES_PASSWORD=agentcore \
  -e POSTGRES_USER=agentcore \
  -e POSTGRES_DB=agentcore_test \
  pgvector/pgvector:pg15

docker run -d --name redis-test -p 6379:6379 redis:7

# Run the test script
./scripts/test-migrations.sh

# Cleanup
docker stop postgres-test redis-test
docker rm postgres-test redis-test
```

---

## Method 4: Pre-Push Hook (Automatic)

Create `.git/hooks/pre-push` to automatically test before pushing:

```bash
#!/bin/bash

echo "Running pre-push migration tests..."

# Run migration test script
if [ -f "scripts/test-migrations.sh" ]; then
    if ! ./scripts/test-migrations.sh; then
        echo "❌ Migration tests failed. Push aborted."
        echo "Fix the issues or use 'git push --no-verify' to skip."
        exit 1
    fi
fi

echo "✅ Pre-push tests passed!"
exit 0
```

Make it executable:
```bash
chmod +x .git/hooks/pre-push
```

---

## Recommended Workflow

### For Quick Iterations
1. Use Docker containers for PostgreSQL/Redis
2. Run `scripts/test-migrations.sh`
3. Test specific changes with `uv run pytest tests/...`

### For Full Workflow Validation
1. Use `act` to run the entire workflow
2. Review output for any differences from CI

### Before Major Changes
1. Use Docker Compose for full environment replication
2. Test all workflows end-to-end
3. Verify performance benchmarks

---

## Common Issues and Solutions

### Issue: "Port already in use"
```bash
# Find and stop conflicting containers
docker ps
docker stop <container-id>

# Or use different ports in docker-compose.test.yml
```

### Issue: "Permission denied" on scripts
```bash
chmod +x scripts/*.sh
```

### Issue: `act` workflows fail with "runner not found"
```bash
# Use specific runner images
act -P ubuntu-latest=catthehacker/ubuntu:full-latest
```

### Issue: Database connection failures
```bash
# Wait for services to be healthy
docker compose -f docker-compose.test.yml up -d
sleep 10  # Wait for health checks
```

---

## Performance Tips

1. **Reuse containers**: Don't recreate PostgreSQL/Redis between test runs
2. **Use volumes**: Mount code as volume for faster iterations
3. **Parallel testing**: Run tests in parallel with `pytest -n auto`
4. **Cache dependencies**: Use `uv cache` to speed up dependency installation
5. **Incremental migrations**: Test only the latest migration with `alembic upgrade +1`

---

## Integration with Development Workflow

Add to your development routine:

```bash
# Before committing migration changes
./scripts/test-migrations.sh

# Before pushing to remote
act -j load-test

# Before creating PR
docker compose -f docker-compose.test.yml up --abort-on-container-exit test-runner
```
