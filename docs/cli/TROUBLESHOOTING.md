# AgentCore CLI Troubleshooting Guide

Common issues, error messages, and solutions for the AgentCore CLI.

---

## Table of Contents

- [Installation Issues](#installation-issues)
- [Connection Issues](#connection-issues)
- [Authentication Issues](#authentication-issues)
- [Configuration Issues](#configuration-issues)
- [Command Execution Issues](#command-execution-issues)
- [Performance Issues](#performance-issues)
- [Output and Formatting Issues](#output-and-formatting-issues)
- [Error Reference](#error-reference)
- [Debugging Tips](#debugging-tips)
- [Getting Help](#getting-help)

---

## Installation Issues

### Issue: Command not found after installation

**Error**:
```bash
$ agentcore --version
agentcore: command not found
```

**Causes**:
- CLI not installed
- Python scripts directory not in PATH
- Wrong Python environment

**Solutions**:

1. **Verify installation**:
   ```bash
   pip list | grep agentcore
   ```

2. **Reinstall**:
   ```bash
   pip install --force-reinstall agentcore
   ```

3. **Check PATH**:
   ```bash
   # Find Python scripts directory
   python -m site --user-base

   # Add to PATH (add to ~/.bashrc or ~/.zshrc)
   export PATH="$PATH:$(python -m site --user-base)/bin"
   ```

4. **Use full path**:
   ```bash
   python -m agentcore_cli.main --version
   ```

---

### Issue: Import errors or missing dependencies

**Error**:
```
ModuleNotFoundError: No module named 'typer'
```

**Solution**:
```bash
# Reinstall with dependencies
pip install --upgrade agentcore

# Or install specific missing dependency
pip install typer rich pyyaml
```

---

### Issue: Python version incompatibility

**Error**:
```
ERROR: This package requires Python 3.12 or higher
```

**Solution**:
```bash
# Check Python version
python --version

# Use Python 3.12+ explicitly
python3.12 -m pip install agentcore
```

---

## Connection Issues

### Issue: Cannot connect to AgentCore API

**Error**:
```
Error: Cannot connect to AgentCore API at http://localhost:8001
Connection refused
```

**Causes**:
- AgentCore server not running
- Wrong API URL
- Network/firewall issues

**Solutions**:

1. **Check if server is running**:
   ```bash
   curl http://localhost:8001/health
   ```

2. **Verify API URL**:
   ```bash
   agentcore config show
   ```

3. **Update API URL**:
   ```bash
   # Via environment variable
   export AGENTCORE_API_URL="http://localhost:8001"

   # Or update config
   agentcore config init --force
   ```

4. **Check firewall**:
   ```bash
   # Test connectivity
   telnet localhost 8001
   nc -zv localhost 8001
   ```

---

### Issue: SSL/TLS certificate verification failed

**Error**:
```
Error: SSL certificate verification failed
CERTIFICATE_VERIFY_FAILED
```

**Causes**:
- Self-signed certificate
- Expired certificate
- Invalid CA bundle

**Solutions**:

1. **Disable SSL verification** (development only):
   ```yaml
   # .agentcore.yaml
   api:
     verify_ssl: false
   ```

2. **Provide custom CA bundle**:
   ```yaml
   api:
     verify_ssl: true
     ca_bundle: /path/to/ca-bundle.crt
   ```

3. **Update certificates**:
   ```bash
   # macOS
   brew install ca-certificates

   # Ubuntu/Debian
   sudo apt-get install ca-certificates
   ```

---

### Issue: Request timeout

**Error**:
```
Error: Request timeout after 30 seconds
```

**Causes**:
- Server overloaded
- Network latency
- Large response payload

**Solutions**:

1. **Increase timeout**:
   ```yaml
   # .agentcore.yaml
   api:
     timeout: 60  # seconds
   ```

2. **Check server health**:
   ```bash
   curl http://localhost:8001/health
   ```

3. **Reduce payload size**:
   ```bash
   # Limit results
   agentcore task list --limit 10
   ```

---

## Authentication Issues

### Issue: Authentication failed (401)

**Error**:
```
Error: Authentication failed (401 Unauthorized)
Invalid or expired token
```

**Causes**:
- No token provided
- Expired JWT token
- Invalid token format

**Solutions**:

1. **Check token**:
   ```bash
   echo $AGENTCORE_TOKEN
   ```

2. **Set token**:
   ```bash
   export AGENTCORE_TOKEN="your-jwt-token"
   ```

3. **Verify token in config**:
   ```bash
   agentcore config show --sources
   ```

4. **Decode JWT to check expiration**:
   ```bash
   # Using jq (install: brew install jq)
   echo $AGENTCORE_TOKEN | cut -d. -f2 | base64 -d | jq .exp
   ```

---

### Issue: Permission denied (403)

**Error**:
```
Error: Permission denied (403 Forbidden)
Insufficient permissions for this operation
```

**Causes**:
- Token lacks required permissions
- Role-based access control (RBAC) restrictions

**Solutions**:

1. **Check token permissions**:
   ```bash
   # Decode JWT payload
   echo $AGENTCORE_TOKEN | cut -d. -f2 | base64 -d | jq .
   ```

2. **Request elevated permissions** from your administrator

3. **Use different credentials** with appropriate permissions

---

## Configuration Issues

### Issue: Invalid configuration file

**Error**:
```
Error: Invalid configuration file
yaml.parser.ParserError: while parsing a block mapping
```

**Causes**:
- Invalid YAML syntax
- Incorrect indentation
- Missing quotes

**Solutions**:

1. **Validate YAML syntax**:
   ```bash
   # Use yamllint (install: pip install yamllint)
   yamllint .agentcore.yaml
   ```

2. **Regenerate config**:
   ```bash
   agentcore config init --force
   ```

3. **Check common YAML issues**:
   ```yaml
   # ❌ BAD - mixing tabs and spaces
   api:
   	url: http://localhost:8001  # tab here

   # ✅ GOOD - consistent spaces
   api:
     url: http://localhost:8001

   # ❌ BAD - unquoted special characters
   auth:
     token: ${TOKEN}  # needs quotes if $ is literal

   # ✅ GOOD - properly quoted
   auth:
     token: "${TOKEN}"
   ```

---

### Issue: Environment variable not substituted

**Error**:
```
Token value is literally '${AGENTCORE_TOKEN}' instead of actual token
```

**Causes**:
- Environment variable not exported
- Wrong variable name
- Shell not expanding variables

**Solutions**:

1. **Export variable**:
   ```bash
   export AGENTCORE_TOKEN="your-token"
   ```

2. **Verify variable is set**:
   ```bash
   echo $AGENTCORE_TOKEN
   env | grep AGENTCORE
   ```

3. **Check config shows actual value**:
   ```bash
   agentcore config show
   ```

---

### Issue: Configuration file not found

**Warning**:
```
Warning: Configuration file not found: ./.agentcore.yaml
Using default values
```

**Solutions**:

1. **Create config file**:
   ```bash
   agentcore config init
   ```

2. **Use global config**:
   ```bash
   agentcore config init --global
   ```

3. **Specify config path**:
   ```bash
   agentcore --config /path/to/config.yaml agent list
   ```

---

## Command Execution Issues

### Issue: Required argument missing

**Error**:
```
Error: Missing option '--name' / '-n'
Try 'agentcore agent register --help' for help
```

**Solution**:
```bash
# Check required arguments
agentcore agent register --help

# Provide required arguments
agentcore agent register --name "my-agent" --capabilities "python"
```

---

### Issue: Invalid argument value

**Error**:
```
Error: Invalid value for '--priority': invalid choice: urgent
(choose from low, medium, high, critical)
```

**Solution**:
```bash
# Use valid value
agentcore task create --priority high

# Check valid values
agentcore task create --help
```

---

### Issue: JSON parsing error

**Error**:
```
Error: Invalid JSON in --requirements
Expecting ',' delimiter: line 1 column 25 (char 24)
```

**Causes**:
- Invalid JSON syntax
- Unescaped quotes
- Missing braces

**Solutions**:

1. **Validate JSON**:
   ```bash
   # Use jq to validate
   echo '{"memory": "512MB"}' | jq .
   ```

2. **Escape quotes properly**:
   ```bash
   # ❌ BAD
   agentcore task create --requirements {"memory": "512MB"}

   # ✅ GOOD
   agentcore task create --requirements '{"memory": "512MB"}'
   ```

3. **Use file input** for complex JSON:
   ```bash
   # Save to file
   cat > requirements.json << 'EOF'
   {
     "memory": "512MB",
     "cpu": "0.5",
     "capabilities": ["python", "testing"]
   }
   EOF

   # Read from file
   agentcore task create --requirements "$(cat requirements.json)"
   ```

---

### Issue: Task/Agent/Workflow not found

**Error**:
```
Error: Task not found: task-invalid-id
```

**Causes**:
- Incorrect ID
- Resource was deleted
- Wrong environment

**Solutions**:

1. **Verify ID**:
   ```bash
   # List resources to get correct ID
   agentcore task list
   agentcore agent list
   ```

2. **Check API environment**:
   ```bash
   agentcore config show
   ```

3. **Search by name**:
   ```bash
   agentcore agent search --capability "python"
   ```

---

## Performance Issues

### Issue: Slow command execution

**Symptoms**:
- Commands take > 5 seconds to run
- Startup delay
- Network lag

**Solutions**:

1. **Enable caching**:
   ```yaml
   performance:
     cache_config: true
     cache_ttl: 300
   ```

2. **Reduce timeout**:
   ```yaml
   api:
     timeout: 10  # Fail fast
   ```

3. **Use JSON output** (faster):
   ```bash
   agentcore agent list --json
   ```

4. **Limit results**:
   ```bash
   agentcore task list --limit 10
   ```

---

### Issue: High memory usage

**Symptoms**:
- CLI consumes > 500MB RAM
- System slowdown

**Solutions**:

1. **Enable lazy imports**:
   ```yaml
   performance:
     lazy_imports: true
   ```

2. **Use streaming for large responses**:
   ```bash
   agentcore task list --limit 100 | head -20
   ```

3. **Clear cache**:
   ```bash
   rm -rf ~/.agentcore/cache/
   ```

---

## Output and Formatting Issues

### Issue: Garbled table output

**Symptoms**:
- Table borders look broken
- Unicode characters displayed incorrectly

**Causes**:
- Terminal doesn't support Unicode
- Locale not set correctly

**Solutions**:

1. **Check locale**:
   ```bash
   locale
   export LANG=en_US.UTF-8
   ```

2. **Use JSON output**:
   ```bash
   agentcore agent list --json
   ```

3. **Disable colors**:
   ```bash
   export AGENTCORE_COLOR=false
   agentcore agent list
   ```

---

### Issue: No color output

**Symptoms**:
- Output is plain text
- No colored terminal output

**Solutions**:

1. **Enable colors**:
   ```bash
   export AGENTCORE_COLOR=true
   ```

2. **Check if piping to file** (colors auto-disabled):
   ```bash
   # This will have no colors
   agentcore agent list > output.txt

   # Force colors
   AGENTCORE_COLOR=true agentcore agent list > output.txt
   ```

---

### Issue: Watch mode not updating

**Symptoms**:
- `--watch` flag doesn't show updates
- Screen frozen

**Solutions**:

1. **Check interval**:
   ```bash
   agentcore task status <id> --watch --interval 2
   ```

2. **Verify task is still running**:
   ```bash
   # In another terminal
   agentcore task status <id>
   ```

3. **Press Ctrl+C and retry**:
   ```bash
   # Stop watch mode
   # Press Ctrl+C

   # Try again
   agentcore task status <id> --watch
   ```

---

## Error Reference

### Exit Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 0 | Success | Command completed successfully |
| 1 | General error | API error, validation failed |
| 2 | Usage error | Invalid arguments, missing options |
| 3 | Connection error | Cannot connect to API |
| 4 | Authentication error | Invalid token, expired credentials |
| 5 | Timeout error | Request took too long |
| 130 | Interrupted | User pressed Ctrl+C |

### HTTP Status Codes

| Code | Meaning | CLI Error Message |
|------|---------|-------------------|
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication failed |
| 403 | Forbidden | Permission denied |
| 404 | Not Found | Resource not found |
| 408 | Request Timeout | Request timeout |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error occurred |
| 502 | Bad Gateway | Gateway error |
| 503 | Service Unavailable | Service temporarily unavailable |

---

## Debugging Tips

### Enable Verbose Mode

Get detailed debug information:

```bash
agentcore --verbose agent list
```

**Output**:
```
[DEBUG] Loading configuration from: ./.agentcore.yaml
[DEBUG] API URL: http://localhost:8001
[DEBUG] Sending request: POST /api/v1/jsonrpc
[DEBUG] Request payload: {"jsonrpc": "2.0", "method": "agent.list", ...}
[DEBUG] Response status: 200
[DEBUG] Response time: 0.234s
...
```

### Check Effective Configuration

See what configuration is actually being used:

```bash
agentcore config show --sources
```

### Test API Connectivity

Verify API is accessible:

```bash
# Health check
curl http://localhost:8001/health

# JSON-RPC ping
curl -X POST http://localhost:8001/api/v1/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"rpc.ping","id":1}'
```

### Validate JSON

Test JSON syntax before using in commands:

```bash
echo '{"memory": "512MB"}' | jq .
```

### Check Logs

Enable logging to file:

```yaml
logging:
  level: DEBUG
  file: /tmp/agentcore-cli.log
```

Then check logs:
```bash
tail -f /tmp/agentcore-cli.log
```

### Use JSON Output for Debugging

JSON output includes full error details:

```bash
agentcore agent register \
  --name "test" \
  --capabilities "invalid" \
  --json
```

**Output**:
```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32603,
    "message": "Validation error",
    "data": {
      "field": "capabilities",
      "error": "Invalid capability: invalid",
      "valid_values": ["python", "testing", "analysis"]
    }
  },
  "id": 1
}
```

### Network Debugging

Use `curl` to test API directly:

```bash
# Test connection
curl -v http://localhost:8001/health

# Test authentication
curl -v http://localhost:8001/api/v1/jsonrpc \
  -H "Authorization: Bearer $AGENTCORE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"rpc.ping","id":1}'
```

### Environment Debugging

Check all AgentCore environment variables:

```bash
env | grep AGENTCORE
printenv | grep AGENTCORE
```

---

## Getting Help

### Command Help

```bash
# General help
agentcore --help

# Command help
agentcore agent --help

# Subcommand help
agentcore agent register --help
```

### Check Version

```bash
agentcore --version
```

### Configuration Help

```bash
# Show current config
agentcore config show

# Validate config
agentcore config validate
```

### Community Support

- **Documentation**: [https://docs.agentcore.ai](https://docs.agentcore.ai)
- **GitHub Issues**: [https://github.com/agentcore/agentcore/issues](https://github.com/agentcore/agentcore/issues)
- **Discord Community**: [https://discord.gg/agentcore](https://discord.gg/agentcore)
- **Stack Overflow**: Tag `agentcore`

### Report a Bug

When reporting issues, include:

1. **CLI version**: `agentcore --version`
2. **Python version**: `python --version`
3. **Operating system**: `uname -a` (Linux/Mac) or `ver` (Windows)
4. **Configuration**: `agentcore config show` (remove sensitive data)
5. **Full error message**: Including stack trace
6. **Steps to reproduce**: Exact commands you ran
7. **Expected vs actual behavior**

**Template**:

```markdown
## Bug Report

**CLI Version**: 0.1.0
**Python Version**: 3.12.0
**OS**: macOS 14.0

**Steps to Reproduce**:
1. Run `agentcore agent register --name "test"`
2. Error occurs

**Expected**: Agent registered successfully

**Actual**: Error: Connection refused

**Full Error**:
```
Error: Cannot connect to AgentCore API at http://localhost:8001
Connection refused
```

**Configuration**:
```yaml
api:
  url: http://localhost:8001
  timeout: 30
```
\```

### Enterprise Support

For enterprise support, contact: support@agentcore.ai

---

## Quick Diagnostics Script

Save this script to quickly diagnose common issues:

```bash
#!/bin/bash
# agentcore-diagnostics.sh

echo "AgentCore CLI Diagnostics"
echo "========================="
echo

echo "1. CLI Version:"
agentcore --version
echo

echo "2. Python Version:"
python --version
echo

echo "3. Installation:"
pip show agentcore
echo

echo "4. Configuration:"
agentcore config show
echo

echo "5. API Connectivity:"
curl -s http://localhost:8001/health || echo "FAILED"
echo

echo "6. Environment Variables:"
env | grep AGENTCORE
echo

echo "7. Config File:"
cat .agentcore.yaml 2>/dev/null || echo "No project config"
echo

echo "Diagnostics complete!"
```

Run it:
```bash
bash agentcore-diagnostics.sh
```

---

**Last Updated**: 2025-10-21
**Version**: 0.1.0
