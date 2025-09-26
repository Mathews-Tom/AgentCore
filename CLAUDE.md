# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentCore is an open-source orchestration framework for agentic AI systems built on Google's A2A (Agent2Agent) protocol. The project focuses on unified autonomous agents, collaborative workflows, and cross-platform LLM integration.

## Architecture

AgentCore follows a layered architecture with these core components:

- **A2A Protocol Layer**: JSON-RPC 2.0 implementation with agent cards and task management
- **Agent Runtime Layer**: Multi-philosophy support (ReAct, Chain-of-Thought, Multi-Agent, Autonomous)
- **Orchestration Engine**: Supervisor, hierarchical, network, and custom orchestration patterns
- **Gateway Layer**: FastAPI-based API gateway with middleware and routing
- **Integration Layer**: Portkey Gateway for LLM orchestration, monitoring, and storage adapters

## Development Commands

This project uses UV for package management. Key commands:

- **Setup**: `uv sync` (installs dependencies)
- **Add dependencies**: `uv add <package>` (dev: `uv add --dev <package>`)
- **Run scripts**: `uv run <command>`
- **Lint**: `uv run ruff check .` (autofix: `uv run ruff check . --fix`)
- **Format**: `uv run ruff format .`
- **Test all**: `uv run pytest -q`
- **Single test**: `uv run pytest -q path/to/test_file.py::TestClass::test_case`
- **Test by pattern**: `uv run pytest -q -k "substring"`

## Code Standards

- **Imports**: stdlib, third-party, local; absolute imports; no wildcards
- **Types**: Strict typing with no `Any`; use `TypedDict`, `Protocol`, `Literal`
- **Naming**: `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE` constants
- **Error handling**: Fail fast; raise specific exceptions; no bare `except`; never swallow errors
- **Testing**: No mocks of core logic; prefer integration tests; deterministic tests
- **Config**: No hardcoded model names; store in `config.toml`; read via env vars
- **Logging**: Structured logs; no `print`; include error context; avoid PII

## Key Design Principles

- **A2A Protocol Native**: First-class support for emerging agent communication standards
- **Multi-Philosophy Support**: Native support for different agentic AI approaches (ReAct, CoT, Multi-agent, Autonomous)
- **Modern Python Ecosystem**: UV for project management, FastAPI for APIs, Portkey for LLM orchestration
- **Enterprise-Grade Features**: Built-in observability, security, scalability, and vendor neutrality

## Documentation Structure

- `docs/agentcore-architecture-and-development-plan.md`: Comprehensive architecture and development plan with diagrams and implementation details
- `docs/agentcore-strategic-roadmap.md`: Strategic phased development roadmap
- `AGENTS.md`: Condensed engineering guide with commands and code standards


## Important Notes

- Project is in early planning/design phase with no actual implementation yet
- Architecture focuses on agent-specialized features (decision lineage, systematic optimization, cross-platform compatibility)
- Focus on agent-specialized orchestration rather than general workflow management
- Built around A2A protocol compliance for cross-platform agent interoperability
- Project follows strict "no bullshit code" principles - no fallbacks, mocks, templates, error swallowing, or graceful degradation
- Use `fd` for file searching on macOS (not `find`)
- Development environment uses `uv` package manager exclusively