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
    echo ""
    echo "Start it with:"
    echo "  docker run -d --name postgres-test -p 5432:5432 \\"
    echo "    -e POSTGRES_PASSWORD=agentcore \\"
    echo "    -e POSTGRES_USER=agentcore \\"
    echo "    -e POSTGRES_DB=agentcore_test \\"
    echo "    pgvector/pgvector:pg15"
    exit 1
fi
echo -e "${GREEN}✅ PostgreSQL is running${NC}"

# Check if Redis is running
echo -e "\n${YELLOW}Checking Redis...${NC}"
if ! redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
    echo -e "${RED}❌ Redis is not running on localhost:6379${NC}"
    echo ""
    echo "Start it with:"
    echo "  docker run -d --name redis-test -p 6379:6379 redis:7"
    exit 1
fi
echo -e "${GREEN}✅ Redis is running${NC}"

# Drop and recreate test database
echo -e "\n${YELLOW}Recreating test database...${NC}"
PGPASSWORD=agentcore psql -h localhost -U agentcore -d postgres -c "DROP DATABASE IF EXISTS agentcore_test;" 2>/dev/null || true
PGPASSWORD=agentcore psql -h localhost -U agentcore -d postgres -c "CREATE DATABASE agentcore_test;" 2>/dev/null || true
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
ENUM_TYPES=$(PGPASSWORD=agentcore psql -h localhost -U agentcore -d agentcore_test -t -c "SELECT typname FROM pg_type WHERE typtype = 'e' ORDER BY typname;")
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
