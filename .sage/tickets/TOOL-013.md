# TOOL-013: File Operations Tool

**State:** UNPROCESSED
**Priority:** P1
**Type:** Story

## Description
Implement FileOperationsTool for read, write, list operations with security restrictions (path validation, size limits)

## Acceptance Criteria
- [ ] FileOperationsTool class implementing Tool interface
- [ ] Operations: read, write, list_directory
- [ ] Path validation (prevent directory traversal)
- [ ] Size limits (max file size: 10MB)
- [ ] Whitelist of allowed directories
- [ ] Error handling for permission denied, file not found
- [ ] Unit tests with temporary test files
- [ ] Security tests (directory traversal attempts)

## Dependencies
#TOOL-002, #TOOL-003

## Context
**Specs:** docs/specs/tool-integration/spec.md
**Plans:** docs/specs/tool-integration/plan.md
**Tasks:** docs/specs/tool-integration/tasks.md

## Effort
**Story Points:** 3
**Estimated Duration:** 3 days
**Sprint:** 2

## Implementation Details
**Owner:** Backend Engineer
**Files:** src/agentcore/tools/adapters/files.py
