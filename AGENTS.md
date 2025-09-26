# AGENTS.md — AgentCore Engineering Guide

- Environment: macOS; package manager `uv`; search with `fd`.
- Setup: `uv sync` (installs deps); dev tools: `uv add --dev ruff pytest`.
- Build: No build step yet; run via `uv run ...`.
- Lint: `uv run ruff check .`; autofix: `uv run ruff check . --fix`.
- Format: `uv run ruff format .`.
- Tests (all): `uv run pytest -q`.
- Single test (nodeid): `uv run pytest -q path/to/test_file.py::TestClass::test_case`.
- Single test (pattern): `uv run pytest -q -k "substring"`.
- Imports: stdlib, third-party, local; absolute imports; no wildcard.
- Formatting: keep lines ≤ 100 chars; one statement per line; trailing commas allowed.
- Types: no `Any`; strict typing; use `TypedDict`, `Protocol`, `Literal`; annotate public APIs.
- Naming: `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE` constants; explicit names.
- Errors: fail fast; raise specific exceptions; no bare `except`; never swallow errors.
- Testing discipline: no mocks of core logic; prefer integration paths; deterministic tests.
- Config: no hardcoded model names; store in `config.toml`; read via env vars.
- Logging: structured logs; no `print`; include error context; avoid PII.
- Tools: do not use `timeout`; prefer `fd` for search; use `uv run` to execute scripts.
- Cursor/Copilot: no `.cursor/rules`, `.cursorrules`, or Copilot rules found.
- Enforcement: use `bs-check`/`bs-enforce` if present; block fallbacks, templates, error swallowing.

- Docs: see `docs/agentcore-architecture-and-development-plan.md` (architecture & development plan) and `docs/agentcore-strategic-roadmap.md` (strategic roadmap).
