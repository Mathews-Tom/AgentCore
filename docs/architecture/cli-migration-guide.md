# CLI v1.0 to v2.0 Migration Guide

**Document Type:** User Migration Guide
**Target Audience:** CLI Users and Developers
**CLI Version:** v1.0 → v2.0
**Date:** 2025-10-22

---

## Overview

This guide helps users and developers migrate from CLI v1.0 to v2.0. The redesigned CLI maintains the same command structure and user interface while fixing critical A2A protocol compliance issues through a new 4-layer architecture.

**Good News:** Most commands remain unchanged at the user level. The migration primarily affects internal architecture.

---

## What Changed

### Architecture (Internal)

**v1.0 (Old):**
- Monolithic client design
- JSON-RPC parameters sent incorrectly (flat dictionary)
- Mixed concerns (protocol + business logic)
- Difficult to test

**v2.0 (New):**
- 4-layer architecture (CLI → Service → Protocol → Transport)
- JSON-RPC 2.0 compliant (proper `params` wrapper)
- Clear separation of concerns
- Testable at every layer

### User Interface (External)

**v1.0 and v2.0:**
- Command structure: **UNCHANGED**
- Command arguments: **UNCHANGED**
- Output format: **UNCHANGED**
- Configuration: **UNCHANGED**

**Result:** Existing scripts and workflows will continue to work without modification.

---

## Breaking Changes

### None for Users

There are **NO breaking changes** for CLI users. All commands work exactly as before:

```bash
# These commands work identically in v1.0 and v2.0
agentcore agent register --name analyzer --capabilities "python,analysis"
agentcore agent list
agentcore task create --description "Run tests"
```

### For Developers (If Extending CLI)

If you've extended the CLI with custom commands:

1. **Service Layer Abstraction**: Commands should now use service layer instead of direct client access
2. **Error Handling**: Use new typed exception hierarchy
3. **Testing**: Mock at service layer instead of transport layer

See [CLI Migration Learnings](cli-migration-learnings.md) for developer patterns.

---

## Migration Steps

### For CLI Users

**No action required.** Simply update to v2.0:

```bash
# Pull latest changes
git pull origin main

# Install dependencies
uv sync

# Verify CLI works
agentcore --version
agentcore health
```

### For Developers Extending CLI

Follow these steps to update custom commands:

#### Step 1: Update Imports

**Old (v1.0):**
```python
from agentcore_cli.client import AgentCoreClient
```

**New (v2.0):**
```python
from agentcore_cli.container import get_agent_service
from agentcore_cli.services.exceptions import ValidationError, OperationError
```

#### Step 2: Update Command Structure

**Old (v1.0):**
```python
@app.command()
def my_command(name: str):
    client = AgentCoreClient(base_url="http://localhost:8001")
    result = client.call_method("my.method", name=name)
    print(result)
```

**New (v2.0):**
```python
@app.command()
def my_command(
    name: Annotated[str, typer.Option("--name", "-n")],
    json_output: Annotated[bool, typer.Option("--json", "-j")] = False,
) -> None:
    """Command description."""
    try:
        service = get_my_service()
        result = service.my_operation(name=name)

        if json_output:
            console.print(json.dumps(result, indent=2))
        else:
            console.print(f"[green]✓[/green] Success: {result}")

    except ValidationError as e:
        console.print(f"[red]Validation error:[/red] {e.message}")
        raise typer.Exit(2)
    except OperationError as e:
        console.print(f"[red]Operation failed:[/red] {e.message}")
        raise typer.Exit(1)
```

#### Step 3: Create Service Layer (If Needed)

If your custom command needs business logic:

```python
# services/my_service.py
from agentcore_cli.protocol.jsonrpc import JsonRpcClient
from agentcore_cli.services.exceptions import ValidationError, OperationError

class MyService:
    def __init__(self, client: JsonRpcClient) -> None:
        self.client = client

    def my_operation(self, name: str) -> str:
        # Validate
        if not name:
            raise ValidationError("Name cannot be empty")

        # Call API
        result = self.client.call("my.method", {"name": name})

        # Return result
        return result["id"]
```

#### Step 4: Update Tests

**Old (v1.0):**
```python
def test_my_command():
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {...}
        result = runner.invoke(app, ["my-command", "--name", "test"])
        assert result.exit_code == 0
```

**New (v2.0):**
```python
def test_my_command(mock_my_service):
    mock_my_service.my_operation.return_value = "result-001"

    with patch("commands.my_command.get_my_service", return_value=mock_my_service):
        result = runner.invoke(app, ["my-command", "--name", "test"])

    assert result.exit_code == 0
    assert "Success" in result.output
    mock_my_service.my_operation.assert_called_once_with(name="test")
```

---

## Command Reference (v1.0 vs v2.0)

All commands maintain the same interface. This table confirms compatibility:

| Command | v1.0 | v2.0 | Status |
|---------|------|------|--------|
| `agent register` | ✓ | ✓ | **Compatible** |
| `agent list` | ✓ | ✓ | **Compatible** |
| `agent info` | ✓ | ✓ | **Compatible** |
| `agent remove` | ✓ | ✓ | **Compatible** |
| `agent search` | ✓ | ✓ | **Compatible** |
| `task create` | ✓ | ✓ | **Compatible** |
| `task list` | ✓ | ✓ | **Compatible** |
| `task info` | ✓ | ✓ | **Compatible** |
| `task cancel` | ✓ | ✓ | **Compatible** |
| `task logs` | ✓ | ✓ | **Compatible** |
| `session create` | ✓ | ✓ | **Compatible** |
| `session list` | ✓ | ✓ | **Compatible** |
| `session info` | ✓ | ✓ | **Compatible** |
| `session delete` | ✓ | ✓ | **Compatible** |
| `workflow run` | ✓ | ✓ | **Compatible** |
| `workflow list` | ✓ | ✓ | **Compatible** |
| `workflow info` | ✓ | ✓ | **Compatible** |
| `workflow stop` | ✓ | ✓ | **Compatible** |
| `config show` | ✓ | ✓ | **Compatible** |
| `config set` | ✓ | ✓ | **Compatible** |
| `config get` | ✓ | ✓ | **Compatible** |
| `config init` | ✓ | ✓ | **Compatible** |

---

## Configuration Changes

### Configuration Format

**Status:** UNCHANGED

Configuration files (`.agentcore.toml` and `~/.agentcore/config.toml`) use the same format:

```toml
[api]
url = "http://localhost:8001"
timeout = 30
retries = 3
verify_ssl = true

[auth]
type = "jwt"
token = ""

[defaults]
output_format = "table"
limit = 100
```

### Environment Variables

**Status:** UNCHANGED

Environment variables use the same `AGENTCORE_` prefix:

```bash
export AGENTCORE_API_URL="http://localhost:8001"
export AGENTCORE_AUTH_TYPE="jwt"
export AGENTCORE_AUTH_TOKEN="your-token"
```

---

## Examples: Old vs New (Behind the Scenes)

While the CLI interface remains the same, here's what changed internally:

### Agent Registration

**User Command (Identical):**
```bash
agentcore agent register --name analyzer --capabilities "python,analysis"
```

**v1.0 Request (INCORRECT):**
```json
{
  "jsonrpc": "2.0",
  "method": "agent.register",
  "id": 1,
  "name": "analyzer",
  "capabilities": ["python", "analysis"]
}
```
**Issue:** Parameters mixed with protocol fields (violates JSON-RPC 2.0)

**v2.0 Request (CORRECT):**
```json
{
  "jsonrpc": "2.0",
  "method": "agent.register",
  "params": {
    "name": "analyzer",
    "capabilities": ["python", "analysis"]
  },
  "id": 1
}
```
**Fix:** Parameters properly wrapped in `params` object (compliant with JSON-RPC 2.0)

---

## Benefits of v2.0

### For Users

1. **Reliability**: Fixed protocol compliance issues that caused API errors
2. **Performance**: Connection pooling and retry logic improve reliability
3. **Better Errors**: Clearer error messages with proper exit codes
4. **Same Interface**: No need to relearn commands

### For Developers

1. **Testability**: Easy to test commands without full API server
2. **Maintainability**: Clear separation of concerns
3. **Extensibility**: Easy to add new commands and services
4. **Type Safety**: Full type hints with Pydantic validation
5. **Documentation**: Self-documenting code with type hints and docstrings

---

## Troubleshooting Migration Issues

### Issue: Commands fail with protocol errors

**Symptom:**
```
Error: Invalid JSON-RPC request
```

**Cause:** Old client code still in use

**Solution:**
```bash
# Ensure you're using v2.0
git pull origin main
uv sync
agentcore --version  # Should show v2.0
```

### Issue: Custom commands not working

**Symptom:**
```
ModuleNotFoundError: No module named 'agentcore_cli.client'
```

**Cause:** Custom commands using old imports

**Solution:**
Update imports to use service layer (see "For Developers" section above)

### Issue: Tests failing after update

**Symptom:**
```
AttributeError: Mock object has no attribute 'call_method'
```

**Cause:** Tests mocking old client interface

**Solution:**
Update tests to mock service layer instead (see "Update Tests" section above)

---

## Rollback Procedure

If you encounter critical issues with v2.0:

```bash
# Rollback to v1.0
git checkout <v1.0-tag>
uv sync

# Verify
agentcore --version
```

**Note:** v1.0 has known protocol compliance issues. Please report v2.0 issues so we can fix them.

---

## Testing Your Migration

### Smoke Tests

Run these commands to verify v2.0 works correctly:

```bash
# 1. Check version
agentcore --version

# 2. Check health
agentcore health

# 3. Test configuration
agentcore config show

# 4. Test agent operations
agentcore agent list
agentcore agent register --name test-agent --capabilities "testing"
agentcore agent info <agent-id>
agentcore agent remove <agent-id> --force

# 5. Test task operations
agentcore task list
agentcore task create --description "Test task"
agentcore task info <task-id>

# 6. Test JSON output
agentcore agent list --json
agentcore task list --json
```

### Integration Tests

If you have custom scripts:

```bash
# Run your existing scripts
./scripts/deploy.sh
./scripts/test-workflow.sh

# Verify output matches expectations
```

---

## Getting Help

### Documentation

- **CLI Usage Guide:** [README_CLI.md](../../README_CLI.md)
- **Developer Guide:** [cli-migration-learnings.md](cli-migration-learnings.md)
- **Testing Guide:** [cli-testing-guide.md](cli-testing-guide.md)
- **Architecture Spec:** [docs/specs/cli-layer/spec.md](../specs/cli-layer/spec.md)

### Support Channels

- **GitHub Issues:** https://github.com/your-org/agentcore/issues
- **Discord:** https://discord.gg/agentcore
- **Email:** support@agentcore.example.com

### Reporting Issues

When reporting migration issues, include:

1. CLI version: `agentcore --version`
2. Command that failed: Full command with arguments
3. Error message: Complete error output
4. Configuration: `agentcore config show --json`
5. Environment: OS, Python version, installation method

---

## Timeline

- **v1.0 Support:** Ends 2025-12-31
- **v2.0 Release:** 2025-10-22 (current)
- **Deprecation:** v1.0 will be deprecated after 2 months (2025-12-22)
- **Migration Window:** 2 months to migrate custom extensions

---

## FAQ

### Q: Do I need to change my existing scripts?

**A:** No. All commands work identically. Scripts using `agentcore` commands will work without modification.

### Q: Do I need to update my configuration files?

**A:** No. Configuration format is unchanged.

### Q: What if I have custom CLI extensions?

**A:** You'll need to update custom commands to use the new service layer. See "For Developers" section.

### Q: Will v1.0 continue to work?

**A:** Yes, v1.0 will continue to work until end of support (2025-12-31). However, it has known protocol compliance issues.

### Q: How do I know if I'm using v2.0?

**A:** Run `agentcore --version`. Version should be 2.0 or higher.

### Q: Can I use v1.0 and v2.0 side by side?

**A:** No. Install one version at a time. Use virtual environments if needed for testing.

### Q: What are the main benefits of upgrading?

**A:** Fixed protocol compliance, better error handling, improved reliability, better testing, same user interface.

### Q: Is there any downtime during migration?

**A:** No. Migration is instant (just update code). The API server doesn't need changes.

---

## Conclusion

The CLI v2.0 migration is **seamless for users** and **straightforward for developers**. The same commands work identically while benefiting from a robust, protocol-compliant architecture.

**Key Takeaways:**
- ✓ No breaking changes for CLI users
- ✓ Same command structure and interface
- ✓ Fixed critical protocol compliance issues
- ✓ Improved reliability and error handling
- ✓ Better developer experience for extensions

**Recommended Action:** Update to v2.0 immediately to benefit from protocol compliance fixes and improved reliability.

---

**Document Version:** 1.0
**Last Updated:** 2025-10-22
**Maintained By:** AgentCore Team
