# Development Documentation

This directory contains documentation for developers working on AgentCore.

## Contents

### Testing & CI/CD

- **[LOCAL_WORKFLOW_TESTING.md](LOCAL_WORKFLOW_TESTING.md)** - Guide for testing GitHub Actions workflows locally before pushing
  - Using `act` to simulate GitHub Actions
  - Docker Compose for full environment replication
  - Quick test scripts for migrations
  - Recommended development workflow

- **[LOAD_TEST_FIX.md](LOAD_TEST_FIX.md)** - Context on load test workflow fixes
  - Why integration tests were failing (100% failure rate)
  - How workflows were fixed to skip non-existent features
  - New A2A protocol load testing workflow

- **[MIGRATION_FIXES_SUMMARY.md](MIGRATION_FIXES_SUMMARY.md)** - Database migration idempotency fixes
  - Enum creation issues (5 enums fixed)
  - Index creation idempotency (12 indexes fixed)
  - JSON vs JSONB type issues
  - Testing methodology and verification

## Quick Links

### For Contributors

- Start here: [LOCAL_WORKFLOW_TESTING.md](LOCAL_WORKFLOW_TESTING.md)
- Before migrations: Run `./scripts/test-migrations.sh`
- Main docs: [../README.md](../README.md)

### Related Files

- `scripts/test-migrations.sh` - Migration testing script
- `docker-compose.test.yml` - Test environment
- `.actrc.example` - GitHub Actions local runner config

## Development Workflow

1. **Make Changes** - Code, migrations, etc.
2. **Test Locally** - Use scripts from LOCAL_WORKFLOW_TESTING.md
3. **Verify** - Ensure tests pass
4. **Commit** - Small, logical commits
5. **Push** - CI/CD will run automatically

## Getting Help

- Check [LOCAL_WORKFLOW_TESTING.md](LOCAL_WORKFLOW_TESTING.md) for local testing
- See [MIGRATION_FIXES_SUMMARY.md](MIGRATION_FIXES_SUMMARY.md) for migration patterns
- Review [LOAD_TEST_FIX.md](LOAD_TEST_FIX.md) for load testing context
