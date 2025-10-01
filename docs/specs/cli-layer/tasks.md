# Tasks: CLI Layer

**From:** `spec.md` + `plan.md`
**Timeline:** 4 weeks, 2 sprints
**Team:** 1 senior Python developer
**Created:** 2025-09-30

## Summary

- Total tasks: 12
- Estimated effort: 34 story points
- Critical path duration: 4 weeks
- Key risks: Session API dependency, CLI framework choice, output formatting complexity

## Phase Breakdown

### Sprint 1: Core Framework (Week 1-2, 18 story points)

**Goal:** Establish CLI framework with basic agent and task commands
**Deliverable:** Working CLI that can register agents and create tasks

#### Tasks

**[CLI-001] Project Setup & CLI Framework**

- **Description:** Initialize Python project structure, choose and configure CLI framework (Click vs Typer)
- **Acceptance:**
  - [ ] Python package structure created (src/agentcore_cli/)
  - [ ] CLI framework chosen and configured (Typer recommended)
  - [ ] pyproject.toml with dependencies configured
  - [ ] Entry point `agentcore` command working
  - [ ] Basic --help and --version flags functional
  - [ ] Development environment setup (pytest, black, mypy, ruff)
  - [ ] GitHub Actions CI pipeline configured
- **Effort:** 3 story points (2-3 days)
- **Owner:** Senior Python Developer
- **Dependencies:** None
- **Priority:** P0 (Blocker)
- **Notes:** Framework decision critical - Typer recommended for type safety and modern syntax. Evaluate both Click and Typer in first day.

**[CLI-002] JSON-RPC Client Implementation**

- **Description:** Build robust JSON-RPC 2.0 client with connection management, retry logic, and error handling
- **Acceptance:**
  - [ ] AgentCoreClient class with call() method
  - [ ] JSON-RPC 2.0 request/response handling
  - [ ] Connection pooling with requests.Session
  - [ ] Retry logic with exponential backoff (3 retries default)
  - [ ] Timeout configuration (30s default)
  - [ ] SSL/TLS verification support
  - [ ] Error translation (JSON-RPC errors → user-friendly messages)
  - [ ] Unit tests with 95%+ coverage
- **Effort:** 5 story points (3-5 days)
- **Owner:** Senior Python Developer
- **Dependencies:** CLI-001
- **Priority:** P0 (Critical)
- **Notes:** Core infrastructure component. Must handle all JSON-RPC edge cases (batch requests, notifications, errors). Consider using existing JSON-RPC client library vs custom implementation.

**[CLI-003] Configuration Management**

- **Description:** Implement multi-level configuration system with file loading, environment variables, and precedence logic
- **Acceptance:**
  - [ ] Config class with Pydantic models
  - [ ] Load global config (~/.agentcore/config.yaml)
  - [ ] Load project config (./.agentcore.yaml)
  - [ ] Environment variable support (AGENTCORE_* prefix)
  - [ ] Configuration precedence: CLI args > env > project > global > defaults
  - [ ] Config validation with helpful error messages
  - [ ] `agentcore config init` command to generate template
  - [ ] `agentcore config show` command to display current config
  - [ ] `agentcore config validate` command to check syntax
  - [ ] Unit tests for config loading and merging
- **Effort:** 5 story points (3-5 days)
- **Owner:** Senior Python Developer
- **Dependencies:** CLI-001
- **Priority:** P0 (Critical)
- **Notes:** Config system must handle missing files gracefully, provide sensible defaults, and support environment variable substitution in YAML (e.g., ${AGENTCORE_TOKEN}).

**[CLI-004] Agent Commands**

- **Description:** Implement all agent management commands (register, list, info, remove, search)
- **Acceptance:**
  - [ ] `agentcore agent register` command with required/optional flags
  - [ ] `agentcore agent list` command with status filtering
  - [ ] `agentcore agent info <id>` command showing full details
  - [ ] `agentcore agent remove <id>` command with confirmation prompt
  - [ ] `agentcore agent search --capability <cap>` command
  - [ ] Input validation for all commands
  - [ ] Error handling with helpful suggestions
  - [ ] Default output format (table) implemented
  - [ ] --json flag support for all commands
  - [ ] Unit tests for each command (90%+ coverage)
  - [ ] Integration tests with mock API responses
- **Effort:** 5 story points (3-5 days)
- **Owner:** Senior Python Developer
- **Dependencies:** CLI-002, CLI-003
- **Priority:** P0 (Critical)
- **Notes:** First user-facing commands. UX is critical - error messages must be helpful, defaults must be sensible. Follow docker/kubectl patterns for familiarity.

### Sprint 2: Advanced Features (Week 3-4, 16 story points)

**Goal:** Add session management, workflow support, and polished output formatting
**Deliverable:** Full-featured CLI with excellent developer experience

#### Tasks

**[CLI-005] Task Commands**

- **Description:** Implement task lifecycle management commands (create, status, list, cancel, result, retry)
- **Acceptance:**
  - [ ] `agentcore task create` command with task definition
  - [ ] `agentcore task status <id>` command
  - [ ] `agentcore task status <id> --watch` for real-time updates
  - [ ] `agentcore task list` with filtering (status, date range)
  - [ ] `agentcore task cancel <id>` command
  - [ ] `agentcore task result <id>` command with artifact retrieval
  - [ ] `agentcore task retry <id>` command for failed tasks
  - [ ] Progress indicators for long-running operations
  - [ ] Watch mode with rich live display
  - [ ] Unit and integration tests (90%+ coverage)
- **Effort:** 5 story points (3-5 days)
- **Owner:** Senior Python Developer
- **Dependencies:** CLI-002, CLI-003
- **Priority:** P0 (Critical)
- **Notes:** Watch mode is key feature - use rich.live.Live for smooth updates. Handle Ctrl+C gracefully (exit watch, don't cancel task).

**[CLI-006] Session Commands**

- **Description:** Implement session save/resume commands for long-running workflows
- **Acceptance:**
  - [ ] `agentcore session save` command with name, description, tags
  - [ ] `agentcore session resume <id>` command with progress indicator
  - [ ] `agentcore session list` command with filtering
  - [ ] `agentcore session info <id>` showing full session details
  - [ ] `agentcore session delete <id>` command with confirmation
  - [ ] `agentcore session export <id>` for debugging (JSON output)
  - [ ] Session metadata support (custom key-value pairs)
  - [ ] Unit tests with mock session API
  - [ ] Integration tests when session API available
- **Effort:** 5 story points (3-5 days)
- **Owner:** Senior Python Developer
- **Dependencies:** CLI-002, CLI-003, A2A-019 (Session Resumption API)
- **Priority:** P1 (High)
- **Notes:** **DEPENDENCY RISK:** Session API (A2A-019) must be implemented first. Can mock API for CLI development, but integration requires real API. Coordinate with A2A Protocol team.

**[CLI-007] Workflow Commands**

- **Description:** Implement workflow creation and execution commands
- **Acceptance:**
  - [ ] `agentcore workflow create --file <yaml>` command
  - [ ] `agentcore workflow execute <id>` command with optional --watch
  - [ ] `agentcore workflow status <id>` showing task progress
  - [ ] `agentcore workflow list` command
  - [ ] `agentcore workflow visualize <id>` (ASCII graph)
  - [ ] `agentcore workflow pause <id>` command
  - [ ] `agentcore workflow resume <id>` command
  - [ ] YAML workflow definition validation
  - [ ] Unit tests for workflow parsing and validation
- **Effort:** 3 story points (2-3 days)
- **Owner:** Senior Python Developer
- **Dependencies:** CLI-002, CLI-003
- **Priority:** P1 (High)
- **Notes:** Workflow visualization can be simple ASCII graph initially. Future: export to Mermaid/Graphviz formats.

**[CLI-008] Output Formatters**

- **Description:** Implement multiple output formats (JSON, table, tree) with rich formatting
- **Acceptance:**
  - [ ] JSON formatter (--json flag for all commands)
  - [ ] Table formatter using rich.table.Table (default)
  - [ ] Tree formatter using rich.tree.Tree (--tree flag)
  - [ ] Color support with auto-detection (disable for piped output)
  - [ ] Timestamp formatting options (--timestamps flag)
  - [ ] Column selection for table output (--columns flag)
  - [ ] Pagination support for large outputs (--limit flag)
  - [ ] Unit tests for each formatter (90%+ coverage)
  - [ ] Snapshot tests for output format regression
- **Effort:** 3 story points (2-3 days)
- **Owner:** Senior Python Developer
- **Dependencies:** CLI-001
- **Priority:** P1 (High)
- **Notes:** Rich library provides excellent formatting out-of-the-box. Focus on consistent UX across all commands. Consider adding CSV format for data export.

## Testing & Quality Assurance Tasks

**[CLI-009] Unit Test Suite**

- **Description:** Comprehensive unit tests for all modules
- **Acceptance:**
  - [ ] 90%+ overall code coverage
  - [ ] 100% coverage for CLI command parsing
  - [ ] 95%+ coverage for JSON-RPC client
  - [ ] 95%+ coverage for configuration management
  - [ ] 90%+ coverage for formatters
  - [ ] All error paths tested
  - [ ] Mock-based tests for API interactions
  - [ ] Pytest fixtures for common test scenarios
  - [ ] Coverage report generated in CI/CD
- **Effort:** Included in above tasks (each task includes unit tests)
- **Owner:** Senior Python Developer
- **Dependencies:** All implementation tasks
- **Priority:** P0 (Critical)
- **Notes:** Unit tests should be written alongside implementation (TDD approach). Coverage is enforced in CI pipeline.

**[CLI-010] Integration Test Suite**

- **Description:** Integration tests with mock AgentCore API
- **Acceptance:**
  - [ ] Mock API server fixture with realistic responses
  - [ ] Test all command combinations
  - [ ] Test configuration precedence (CLI > env > file)
  - [ ] Test error handling for various API responses
  - [ ] Test output format correctness (JSON, table, tree)
  - [ ] Test watch mode and progress indicators
  - [ ] Test interactive prompts (confirmation, wizards)
  - [ ] Integration tests run in CI/CD
- **Effort:** Included in CLI-004 through CLI-008
- **Owner:** Senior Python Developer
- **Dependencies:** CLI-002, CLI-003
- **Priority:** P1 (High)
- **Notes:** Mock API should match AgentCore JSON-RPC spec exactly. Use pytest-mock for mocking requests.

**[CLI-011] E2E Test Suite (Optional)**

- **Description:** End-to-end tests against real AgentCore instance
- **Acceptance:**
  - [ ] Docker Compose test environment (CLI + AgentCore)
  - [ ] Test full workflows (register agent → create task → check status)
  - [ ] Test session save/resume flow
  - [ ] Test workflow creation and execution
  - [ ] Cleanup after tests (remove test agents/tasks)
  - [ ] E2E tests run on demand (not in CI due to complexity)
- **Effort:** 2 story points (1-2 days) - **OPTIONAL**
- **Owner:** Senior Python Developer
- **Dependencies:** CLI-001 through CLI-008, AgentCore running instance
- **Priority:** P2 (Nice to have)
- **Notes:** E2E tests are valuable but not critical for MVP. Can be added post-launch. Requires Docker Compose environment with AgentCore + PostgreSQL + Redis.

## Documentation Tasks

**[CLI-012] Documentation & Distribution**

- **Description:** Complete documentation and prepare for PyPI distribution
- **Acceptance:**
  - [ ] README.md with installation instructions
  - [ ] Command reference documentation (auto-generated from --help)
  - [ ] Configuration guide with examples
  - [ ] Troubleshooting guide with common errors
  - [ ] CHANGELOG.md for version history
  - [ ] PyPI package metadata (description, keywords, classifiers)
  - [ ] GitHub Actions workflow for PyPI publishing
  - [ ] Test PyPI deployment successful
  - [ ] Production PyPI deployment successful
- **Effort:** 2 story points (1-2 days)
- **Owner:** Senior Python Developer
- **Dependencies:** All implementation tasks completed
- **Priority:** P1 (High)
- **Notes:** Documentation is critical for adoption. Auto-generate command reference from Typer help strings. Include animated GIFs for watch mode demo.

## Critical Path

```text
CLI-001 → CLI-002 → CLI-004 → CLI-005 → CLI-008 → CLI-012
  (3d)      (5d)      (5d)      (5d)      (3d)      (2d)
                  [23 days total / ~4.5 weeks]

Parallel tracks:
- CLI-003 (config) can run parallel with CLI-002
- CLI-006 (session) blocked by A2A-019 API, can develop with mocks
- CLI-007 (workflow) can run parallel with CLI-005
```

**Bottlenecks:**
- CLI-002: JSON-RPC client is foundational (blocks CLI-004, CLI-005, CLI-006, CLI-007)
- CLI-006: Session commands blocked by A2A-019 API availability
- CLI-008: Output formatting needed for polished UX (blocks launch)

**Parallel Tracks:**
- Configuration: CLI-003 (parallel with CLI-002)
- Session commands: CLI-006 (can mock API, integrate later)
- Workflow commands: CLI-007 (parallel with CLI-005)

## Quick Wins (Week 1)

1. **[CLI-001] CLI Framework** - Establishes foundation, enables parallel work
2. **[CLI-002] JSON-RPC Client** - Core infrastructure, unblocks all commands
3. **[CLI-003] Configuration** - Critical UX feature, early completion important

## Risk Mitigation

| Task | Risk | Mitigation | Contingency |
|------|------|------------|-------------|
| CLI-001 | Wrong framework choice (Click vs Typer) | Evaluate both in 1 day, choose Typer for type safety | Can migrate if needed, but Typer strongly recommended |
| CLI-002 | JSON-RPC complexity (batch requests, errors) | Use proven requests library, extensive testing | Consider using existing JSON-RPC client library (e.g., jsonrpcclient) |
| CLI-006 | Session API not ready (A2A-019 dependency) | Mock API responses, integrate when available | Delay session commands to v0.2 if API not ready |
| CLI-008 | Rich library performance issues | Profile early, optimize hot paths, lazy import | Fallback to simpler tabulate library for tables |

## Testing Strategy

### Automated Testing Tasks

**Unit Tests (included in each task)**
- Pytest framework
- 90%+ coverage requirement
- Run in CI/CD on every commit
- Coverage report published

**Integration Tests (included in CLI-004 through CLI-008)**
- Mock API server
- Test command combinations
- Run in CI/CD

**E2E Tests (optional, CLI-011)**
- Docker Compose environment
- Run on demand
- Not in CI pipeline

### Quality Gates

- 90%+ code coverage required for merge
- All tests passing (unit + integration)
- Mypy type checking passing (strict mode)
- Ruff linting passing (no warnings)
- Black formatting passing

## Team Allocation

**Senior Python Developer (1 FTE)**
- All CLI implementation tasks (CLI-001 through CLI-012)
- Unit and integration testing
- Documentation writing
- PyPI distribution

**Optional Support:**
- Technical writer for documentation (0.25 FTE)
- QA engineer for E2E testing (0.25 FTE)

## Sprint Planning

**2-week sprints, 17 SP velocity (1 developer)**

| Sprint | Focus | Story Points | Key Deliverables |
|--------|-------|--------------|------------------|
| Sprint 1 | Core Framework | 18 SP | CLI framework, JSON-RPC client, config, agent commands |
| Sprint 2 | Advanced Features | 16 SP | Task/session/workflow commands, formatters, docs |

## Task Import Format

CSV export for project management tools:

```csv
ID,Title,Description,Estimate,Priority,Assignee,Dependencies,Sprint
CLI-001,CLI Framework,Initialize project and choose CLI framework,3,P0,Senior Python Dev,,1
CLI-002,JSON-RPC Client,Build robust JSON-RPC 2.0 client,5,P0,Senior Python Dev,CLI-001,1
CLI-003,Configuration,Multi-level config management,5,P0,Senior Python Dev,CLI-001,1
CLI-004,Agent Commands,Implement agent management commands,5,P0,Senior Python Dev,CLI-002;CLI-003,1
CLI-005,Task Commands,Implement task lifecycle commands,5,P0,Senior Python Dev,CLI-002;CLI-003,2
CLI-006,Session Commands,Implement session save/resume,5,P1,Senior Python Dev,CLI-002;CLI-003;A2A-019,2
CLI-007,Workflow Commands,Implement workflow management,3,P1,Senior Python Dev,CLI-002;CLI-003,2
CLI-008,Output Formatters,Multiple output formats (JSON/table/tree),3,P1,Senior Python Dev,CLI-001,2
CLI-009,Unit Tests,Comprehensive unit test suite,0,P0,Senior Python Dev,All impl tasks,1-2
CLI-010,Integration Tests,Integration tests with mock API,0,P1,Senior Python Dev,CLI-002;CLI-003,2
CLI-011,E2E Tests (Optional),End-to-end tests with Docker,2,P2,Senior Python Dev,All impl tasks,2
CLI-012,Documentation,Complete docs and PyPI distribution,2,P1,Senior Python Dev,All impl tasks,2
```

## Appendix

**Estimation Method:** Planning Poker with CLI development expertise
**Story Point Scale:** Fibonacci (1,2,3,5,8,13,21)
**Definition of Done:**
- Code reviewed and approved
- Unit tests written and passing (90% coverage)
- Integration tests passing
- Type checking passing (mypy strict)
- Linting passing (ruff)
- Formatting passing (black)
- Documentation updated
- Tested manually with real AgentCore instance

**Technology Stack:**
- Python 3.12+
- Typer (CLI framework)
- Requests (HTTP client)
- Pydantic (validation)
- Rich (output formatting)
- PyYAML (config parsing)
- Pytest (testing)

**Distribution:**
- PyPI package: `agentcore-cli`
- Entry point: `agentcore` command
- Installation: `pip install agentcore-cli` or `uv add agentcore-cli`

**Success Metrics:**
- Time-to-first-task < 5 minutes (vs hours for API)
- PyPI downloads > 100/week within first month
- GitHub stars > 50 within first month
- Zero critical bugs reported in first month
- 90%+ user satisfaction (survey)

---

**End of Tasks Document**
