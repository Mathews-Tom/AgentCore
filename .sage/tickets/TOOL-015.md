# TOOL-015: Integration Tests for Built-in Tools

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story

## Description
Create comprehensive integration test suite for each built-in tool with real external services (staging/test environments)

## Acceptance Criteria
- [ ] Integration tests for GoogleSearchTool with real API
- [ ] Integration tests for WikipediaSearchTool with real API
- [ ] Integration tests for PythonExecutionTool with real Docker
- [ ] Integration tests for RESTAPITool with test endpoints
- [ ] Integration tests for FileOperationsTool with temp files
- [ ] Tests validate error handling scenarios
- [ ] Tests use testcontainers for Docker/PostgreSQL
- [ ] Test coverage ≥85% for adapters

## Dependencies
#TOOL-014

## Context
**Specs:** docs/specs/tool-integration/spec.md
**Plans:** docs/specs/tool-integration/plan.md
**Tasks:** docs/specs/tool-integration/tasks.md

## Effort
**Story Points:** 5
**Estimated Duration:** 5 days
**Sprint:** 2

## Implementation Details
**Owner:** QA Engineer
**Files:** tests/integration/tools/test_google_search.py, tests/integration/tools/test_wikipedia.py, tests/integration/tools/test_python_executor.py, tests/integration/tools/test_rest_api.py, tests/integration/tools/test_file_operations.py
