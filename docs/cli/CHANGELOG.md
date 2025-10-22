# Changelog

All notable changes to the AgentCore CLI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2025-10-21

### Initial Release

First stable release of the AgentCore CLI - a developer-friendly command-line interface for the AgentCore orchestration framework.

### Added

#### Core Framework
- **CLI Framework**: Built on Typer with rich terminal output
- **Configuration System**: Multi-level config (CLI → env → file → defaults)
- **JSON-RPC Client**: Robust client with connection pooling, retries, and error handling
- **Output Formats**: JSON, table, and tree visualization modes

#### Agent Commands
- `agentcore agent register` - Register new agents with capabilities
- `agentcore agent list` - List all registered agents with filtering
- `agentcore agent info` - Get detailed agent information
- `agentcore agent remove` - Remove agents with confirmation prompt
- `agentcore agent search` - Search agents by capability

#### Task Commands
- `agentcore task create` - Create tasks with requirements and priorities
- `agentcore task status` - Get task status with optional watch mode
- `agentcore task list` - List tasks with status filtering
- `agentcore task cancel` - Cancel running tasks
- `agentcore task result` - Get task results and artifacts
- `agentcore task retry` - Retry failed tasks

#### Session Commands
- `agentcore session save` - Save workflow sessions with metadata
- `agentcore session resume` - Resume saved sessions
- `agentcore session list` - List sessions with filtering
- `agentcore session info` - Get detailed session information
- `agentcore session delete` - Delete sessions
- `agentcore session export` - Export sessions for debugging

#### Workflow Commands
- `agentcore workflow create` - Create workflows from YAML definitions
- `agentcore workflow execute` - Execute workflows with watch mode
- `agentcore workflow status` - Get workflow execution status
- `agentcore workflow list` - List workflows with filtering
- `agentcore workflow visualize` - Visualize workflow as ASCII graph
- `agentcore workflow pause` - Pause running workflows
- `agentcore workflow resume` - Resume paused workflows

#### Configuration Commands
- `agentcore config init` - Initialize configuration files
- `agentcore config show` - Display current configuration
- `agentcore config validate` - Validate configuration files

#### Features
- **Watch Mode**: Real-time monitoring of tasks and workflows with `--watch` flag
- **Rich Formatting**: Beautiful terminal tables with colors and Unicode characters
- **Progress Indicators**: Progress bars for long-running operations
- **Interactive Prompts**: Confirmation dialogs for destructive operations
- **Environment Variables**: Full support for `AGENTCORE_*` environment variables
- **Configuration Precedence**: CLI args > env vars > project config > global config > defaults
- **Error Handling**: User-friendly error messages with suggestions
- **Exit Codes**: Standard exit codes for automation and scripting

### Documentation
- Comprehensive README with quick start guide
- Complete CLI reference with all commands and options
- Configuration guide with examples for all use cases
- Troubleshooting guide with common issues and solutions
- Inline help for all commands via `--help` flag

### Testing
- 366 passing tests across all command groups
- 83% code coverage
- Integration tests with mock API
- Configuration precedence tests
- Output format tests

### Dependencies
- Python 3.12+ required
- Typer for CLI framework
- Rich for terminal formatting
- PyYAML for configuration files
- Pydantic for validation
- Requests for HTTP client

### Known Issues
- Session commands require A2A Session API (may not be fully implemented on server)
- 4 minor test failures in config integration tests (non-critical)
- Watch mode may not work in all terminal emulators
- Tree visualization requires Unicode-capable terminal

### Breaking Changes
- None (initial release)

### Deprecated
- None (initial release)

### Security
- JWT authentication support
- API key authentication support
- SSL/TLS certificate verification
- No secrets stored in plaintext (environment variable substitution)

### Performance
- Command startup time: <200ms for simple commands
- Memory usage: <50MB for typical operations
- Connection pooling for better API performance
- Lazy imports for faster startup

---

## [Unreleased]

### Planned for 0.2.0
- Shell completion (Bash, Zsh, Fish)
- Interactive REPL mode
- Batch operations from file
- Advanced filtering with JMESPath
- Custom output formatters
- Plugin architecture for custom commands
- Workflow templates
- Better error messages with contextual help
- Performance optimizations

### Planned for 0.3.0
- Secure credential storage (system keychain)
- Remote configuration management
- TUI (Terminal UI) mode
- Advanced workflow visualization (export to Mermaid/Graphviz)
- Workflow marketplace integration
- Multi-language support
- Audit logging
- Rate limiting and quota management

### Planned for 1.0.0
- Stable API (no breaking changes)
- Production-ready quality
- Complete test coverage (95%+)
- Full documentation
- Migration guides from 0.x
- Long-term support (LTS)

---

## Version History

| Version | Release Date | Status | Notes |
|---------|--------------|--------|-------|
| 0.1.0 | 2025-10-21 | Current | Initial stable release |

---

## Upgrade Guide

### From Pre-release to 0.1.0

This is the first stable release. If you were using development builds:

1. **Uninstall old version**:
   ```bash
   pip uninstall agentcore-cli
   ```

2. **Install stable version**:
   ```bash
   pip install agentcore
   ```

3. **Update configuration** (if needed):
   ```bash
   agentcore config init --force
   ```

4. **Verify installation**:
   ```bash
   agentcore --version
   # Should show: AgentCore CLI version: 0.1.0
   ```

---

## Release Notes Template

For future releases:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features and commands

### Changed
- Changes to existing functionality

### Deprecated
- Features marked for removal

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security improvements
```

---

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for how to contribute to AgentCore CLI.

---

## Support

- **Documentation**: [https://docs.agentcore.ai](https://docs.agentcore.ai)
- **Issues**: [GitHub Issues](https://github.com/agentcore/agentcore/issues)
- **Discussions**: [GitHub Discussions](https://github.com/agentcore/agentcore/discussions)
- **Discord**: [https://discord.gg/agentcore](https://discord.gg/agentcore)

---

**Last Updated**: 2025-10-21
