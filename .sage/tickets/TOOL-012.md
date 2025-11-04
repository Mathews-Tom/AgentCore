# TOOL-012: REST API Tool

**State:** UNPROCESSED
**Priority:** P0
**Type:** Story

## Description
Implement RESTAPITool adapter providing generic HTTP client for external API calls with support for GET, POST, PUT, DELETE and multiple auth methods

## Acceptance Criteria
- [ ] RESTAPITool class implementing Tool interface
- [ ] HTTP client using httpx with async support
- [ ] Support for GET, POST, PUT, DELETE methods
- [ ] Parameters: url, method, headers, body, auth_type
- [ ] Authentication support: none, api_key, bearer_token, oauth
- [ ] Response parsing (JSON, text, binary)
- [ ] Error handling for network errors, timeouts, HTTP errors
- [ ] Unit tests with mocked HTTP responses
- [ ] Integration tests with test API endpoints

## Dependencies
#TOOL-002, #TOOL-003

## Context
**Specs:** docs/specs/tool-integration/spec.md
**Plans:** docs/specs/tool-integration/plan.md
**Tasks:** docs/specs/tool-integration/tasks.md

## Effort
**Story Points:** 5
**Estimated Duration:** 5 days
**Sprint:** 2

## Implementation Details
**Owner:** Backend Engineer
**Files:** src/agentcore/tools/adapters/api.py
