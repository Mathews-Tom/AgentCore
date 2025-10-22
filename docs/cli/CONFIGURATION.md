# AgentCore CLI Configuration Guide

Complete guide to configuring the AgentCore CLI for different environments and use cases.

---

## Table of Contents

- [Configuration Hierarchy](#configuration-hierarchy)
- [Configuration File](#configuration-file)
- [Environment Variables](#environment-variables)
- [CLI Arguments](#cli-arguments)
- [Configuration Examples](#configuration-examples)
- [Advanced Configuration](#advanced-configuration)
- [Validation and Troubleshooting](#validation-and-troubleshooting)

---

## Configuration Hierarchy

AgentCore CLI uses a multi-level configuration system with the following precedence (highest to lowest):

```
1. CLI Arguments     (e.g., --api-url http://localhost:8001)
2. Environment Vars  (e.g., AGENTCORE_API_URL)
3. Project Config    (./.agentcore.yaml in current directory)
4. Global Config     (~/.agentcore/config.yaml in home directory)
5. Built-in Defaults (hardcoded fallback values)
```

This allows you to:
- Set global defaults in `~/.agentcore/config.yaml`
- Override per-project in `./.agentcore.yaml`
- Override per-command with environment variables
- Override per-execution with CLI flags

---

## Configuration File

### File Locations

**Global Configuration**:
- Path: `~/.agentcore/config.yaml`
- Purpose: User-wide defaults
- Created by: `agentcore config init --global`

**Project Configuration**:
- Path: `./.agentcore.yaml` (current directory)
- Purpose: Project-specific settings
- Created by: `agentcore config init`

### File Format

Configuration files use YAML format with the following structure:

```yaml
# AgentCore CLI Configuration

# API Connection Settings
api:
  url: http://localhost:8001
  timeout: 30
  retries: 3
  verify_ssl: true

# Authentication Settings
auth:
  type: jwt  # jwt | api_key | none
  token: ${AGENTCORE_TOKEN}
  # api_key: ${AGENTCORE_API_KEY}

# Output Preferences
output:
  format: table  # json | table | tree
  color: true
  timestamps: false
  verbose: false

# Default Values for Commands
defaults:
  task:
    priority: medium
    timeout: 3600
  agent:
    cost_per_request: 0.01
  workflow:
    max_retries: 3
```

### Creating Configuration Files

**Initialize global configuration**:
```bash
agentcore config init --global
```

This creates `~/.agentcore/config.yaml` with default values.

**Initialize project configuration**:
```bash
agentcore config init
```

This creates `./.agentcore.yaml` in the current directory.

**Force overwrite existing config**:
```bash
agentcore config init --force
```

---

## Environment Variables

All configuration options can be set via environment variables with the `AGENTCORE_` prefix.

### API Configuration

```bash
# API endpoint URL
export AGENTCORE_API_URL="http://localhost:8001"

# Request timeout in seconds
export AGENTCORE_API_TIMEOUT="30"

# Number of retry attempts
export AGENTCORE_API_RETRIES="3"

# SSL/TLS verification
export AGENTCORE_VERIFY_SSL="true"

# Custom CA bundle path
export AGENTCORE_CA_BUNDLE="/path/to/ca-bundle.crt"
```

### Authentication

```bash
# JWT token authentication
export AGENTCORE_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# API key authentication
export AGENTCORE_API_KEY="ak_1234567890abcdef"

# Authentication type
export AGENTCORE_AUTH_TYPE="jwt"  # jwt | api_key | none
```

### Output Preferences

```bash
# Output format
export AGENTCORE_OUTPUT_FORMAT="json"  # json | table | tree

# Enable/disable colors
export AGENTCORE_COLOR="true"

# Show timestamps
export AGENTCORE_TIMESTAMPS="true"

# Verbose output
export AGENTCORE_VERBOSE="false"
```

### Command Defaults

```bash
# Default task priority
export AGENTCORE_TASK_PRIORITY="medium"

# Default task timeout
export AGENTCORE_TASK_TIMEOUT="3600"

# Default agent cost
export AGENTCORE_AGENT_COST="0.01"
```

### Environment Variable Substitution

Configuration files support environment variable substitution using `${VAR_NAME}` syntax:

```yaml
auth:
  token: ${AGENTCORE_TOKEN}
  api_key: ${AGENTCORE_API_KEY}

api:
  url: ${AGENTCORE_API_URL:-http://localhost:8001}  # with default
```

---

## CLI Arguments

Command-line arguments override all other configuration sources.

### Global Flags

```bash
# Override API URL
agentcore --api-url https://production.agentcore.com agent list

# Force JSON output
agentcore agent list --json

# Enable verbose mode
agentcore --verbose task status <task-id>
```

### Per-Command Options

Each command has specific options that can override config defaults:

```bash
# Override default priority
agentcore task create \
  --type "urgent-fix" \
  --priority critical

# Override default cost
agentcore agent register \
  --name "expensive-agent" \
  --cost-per-request 0.50
```

---

## Configuration Examples

### Example 1: Development Environment

**File**: `./.agentcore.yaml`

```yaml
# Development configuration
api:
  url: http://localhost:8001
  timeout: 60  # Longer timeout for debugging
  verify_ssl: false  # Self-signed certs OK in dev

auth:
  type: none  # No auth required in dev

output:
  format: table
  color: true
  verbose: true  # Show debug info

defaults:
  task:
    priority: low
    timeout: 7200  # 2 hours for development tasks
```

### Example 2: Production Environment

**File**: `~/.agentcore/config.yaml`

```yaml
# Production configuration
api:
  url: https://agentcore.example.com
  timeout: 30
  retries: 3
  verify_ssl: true

auth:
  type: jwt
  token: ${AGENTCORE_TOKEN}  # From environment

output:
  format: json  # Machine-readable for logging
  color: false  # No colors in logs
  timestamps: true  # Include timestamps
  verbose: false

defaults:
  task:
    priority: medium
    timeout: 3600
  agent:
    cost_per_request: 0.01
```

### Example 3: CI/CD Pipeline

**Environment Variables** (in CI config):

```bash
# .gitlab-ci.yml or .github/workflows/main.yml
export AGENTCORE_API_URL="https://ci.agentcore.example.com"
export AGENTCORE_TOKEN="${CI_AGENTCORE_TOKEN}"
export AGENTCORE_OUTPUT_FORMAT="json"
export AGENTCORE_COLOR="false"
export AGENTCORE_VERBOSE="true"
```

**Usage in scripts**:

```bash
#!/bin/bash
set -e

# Register CI agent
AGENT_ID=$(agentcore agent register \
  --name "ci-agent-${CI_JOB_ID}" \
  --capabilities "ci,testing,build" \
  --json | jq -r '.result.agent_id')

# Run tests
TASK_ID=$(agentcore task create \
  --type "ci-test" \
  --description "Run test suite" \
  --json | jq -r '.result.task_id')

# Wait for completion
while true; do
  STATUS=$(agentcore task status $TASK_ID --json | jq -r '.result.status')
  if [[ "$STATUS" == "completed" ]]; then
    break
  elif [[ "$STATUS" == "failed" ]]; then
    exit 1
  fi
  sleep 5
done

# Cleanup
agentcore agent remove $AGENT_ID --force
```

### Example 4: Multi-Environment Setup

**Directory structure**:

```
project/
├── .agentcore.yaml          # Default (development)
├── .agentcore.staging.yaml  # Staging environment
└── .agentcore.prod.yaml     # Production environment
```

**Switching environments**:

```bash
# Development (default)
agentcore agent list

# Staging
cp .agentcore.staging.yaml .agentcore.yaml
agentcore agent list

# Production
cp .agentcore.prod.yaml .agentcore.yaml
agentcore agent list

# Or use environment variables
AGENTCORE_API_URL=https://staging.example.com agentcore agent list
```

### Example 5: Team Collaboration

**Shared project config** (`.agentcore.yaml`, committed to git):

```yaml
# Shared team configuration
api:
  url: ${AGENTCORE_API_URL:-http://localhost:8001}
  timeout: 30
  verify_ssl: true

output:
  format: table
  color: true

defaults:
  task:
    priority: medium
  agent:
    cost_per_request: 0.01
```

**Personal overrides** (`.agentcore.local.yaml`, git-ignored):

```yaml
# Personal overrides (not committed)
api:
  url: http://192.168.1.100:8001  # Personal dev server

auth:
  token: ${MY_PERSONAL_TOKEN}

output:
  verbose: true  # I like verbose output
```

**Load personal config**:

```bash
# Merge configs manually
cat .agentcore.yaml .agentcore.local.yaml > /tmp/merged.yaml
agentcore --config /tmp/merged.yaml agent list
```

---

## Advanced Configuration

### Custom CA Certificates

For self-signed or custom CA certificates:

```yaml
api:
  verify_ssl: true
  ca_bundle: /path/to/custom-ca-bundle.crt
```

Or via environment:

```bash
export AGENTCORE_CA_BUNDLE="/path/to/ca-bundle.crt"
```

### Connection Pooling

Configure HTTP connection pooling for better performance:

```yaml
api:
  pool_connections: 10
  pool_maxsize: 20
  pool_block: false
```

### Proxy Configuration

Configure HTTP/HTTPS proxy:

```yaml
api:
  proxy:
    http: http://proxy.example.com:8080
    https: https://proxy.example.com:8080
    no_proxy: "localhost,127.0.0.1"
```

Or via standard environment variables:

```bash
export HTTP_PROXY="http://proxy.example.com:8080"
export HTTPS_PROXY="https://proxy.example.com:8080"
export NO_PROXY="localhost,127.0.0.1"
```

### Custom Headers

Add custom headers to all API requests:

```yaml
api:
  headers:
    X-Custom-Header: "custom-value"
    X-Request-ID: "${REQUEST_ID}"
```

### Logging Configuration

Configure logging behavior:

```yaml
logging:
  level: INFO  # DEBUG | INFO | WARNING | ERROR | CRITICAL
  file: /var/log/agentcore-cli.log
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  max_bytes: 10485760  # 10MB
  backup_count: 5
```

### Performance Tuning

Optimize for performance:

```yaml
# Fast startup (minimal imports)
performance:
  lazy_imports: true
  cache_config: true
  cache_ttl: 300  # 5 minutes

# Aggressive timeouts
api:
  timeout: 10
  retries: 1
  connect_timeout: 5
  read_timeout: 10
```

---

## Validation and Troubleshooting

### Validate Configuration

Check if your configuration is valid:

```bash
agentcore config validate
```

**Output** (success):
```
✓ Configuration is valid

Config file: ./.agentcore.yaml
Schema version: 1.0
All settings validated successfully
```

**Output** (error):
```
✗ Configuration validation failed

Error in .agentcore.yaml:
  Line 5: Invalid value for 'api.timeout': must be between 1 and 300
  Line 12: Unknown field 'auth.invalid_field'
  Line 18: 'output.format' must be one of: json, table, tree

Fix these errors and run 'agentcore config validate' again.
```

### Show Effective Configuration

See the merged configuration from all sources:

```bash
agentcore config show
```

**Show with sources**:
```bash
agentcore config show --sources
```

**Output**:
```
Current Configuration (with sources)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

API:
  url: http://localhost:8001
    Source: Project config (./.agentcore.yaml)
  timeout: 30
    Source: Global config (~/.agentcore/config.yaml)
  verify_ssl: false
    Source: CLI argument (--verify-ssl=false)

Authentication:
  token: eyJhbGc...
    Source: Environment (AGENTCORE_TOKEN)
```

### Debug Configuration Issues

Enable verbose mode to see config loading:

```bash
agentcore --verbose config show
```

**Output**:
```
[DEBUG] Loading global config: ~/.agentcore/config.yaml
[DEBUG] Loading project config: ./.agentcore.yaml
[DEBUG] Loading environment variables: 3 found
[DEBUG] Applying CLI arguments: 0 overrides
[DEBUG] Final merged configuration:
...
```

### Common Issues

**Issue**: Configuration file not found

```
Warning: Configuration file not found: ./.agentcore.yaml
Using default values.
```

**Solution**:
```bash
agentcore config init
```

---

**Issue**: Invalid YAML syntax

```
Error: Failed to parse configuration file
yaml.parser.ParserError: while parsing a block mapping
  in ".agentcore.yaml", line 5, column 3
expected <block end>, but found ':'
```

**Solution**: Fix YAML syntax (check indentation, quotes, etc.)

---

**Issue**: Environment variable not substituted

```yaml
auth:
  token: ${AGENTCORE_TOKEN}  # Not working
```

**Solution**: Ensure environment variable is exported:
```bash
export AGENTCORE_TOKEN="your-token"
agentcore config show  # Verify it's set
```

---

**Issue**: SSL certificate verification failed

```
Error: SSL certificate verification failed
```

**Solution**:
```yaml
# Temporarily disable (not recommended for production)
api:
  verify_ssl: false

# Or provide custom CA bundle
api:
  verify_ssl: true
  ca_bundle: /path/to/ca-bundle.crt
```

---

## Configuration Best Practices

### Security

1. **Never commit secrets to git**:
   ```yaml
   # ❌ BAD - hardcoded token
   auth:
     token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

   # ✅ GOOD - from environment
   auth:
     token: ${AGENTCORE_TOKEN}
   ```

2. **Use `.gitignore`**:
   ```
   .agentcore.local.yaml
   .agentcore.*.yaml
   ```

3. **Verify SSL in production**:
   ```yaml
   api:
     verify_ssl: true  # Always in production
   ```

### Organization

1. **Global defaults**:
   - Put common settings in `~/.agentcore/config.yaml`
   - API URLs for different environments
   - Personal preferences (output format, colors)

2. **Project-specific**:
   - Put project settings in `./.agentcore.yaml`
   - Commit to git for team sharing
   - Use environment variable substitution

3. **Per-command overrides**:
   - Use CLI flags for one-off changes
   - Use environment variables in CI/CD

### Maintenance

1. **Document your config**:
   ```yaml
   # Production API endpoint - DO NOT CHANGE
   api:
     url: https://prod.agentcore.example.com
   ```

2. **Validate regularly**:
   ```bash
   agentcore config validate
   ```

3. **Version your config** (in comments):
   ```yaml
   # Configuration version: 1.0
   # Last updated: 2025-10-21
   # Owner: team@example.com
   ```

---

## Configuration Schema Reference

Complete schema for `.agentcore.yaml`:

```yaml
# API Configuration
api:
  url: string              # API endpoint URL
  timeout: integer         # Request timeout (1-300 seconds)
  retries: integer         # Retry attempts (0-10)
  verify_ssl: boolean      # Verify SSL certificates
  ca_bundle: string        # Path to CA bundle (optional)
  proxy:                   # Proxy configuration (optional)
    http: string
    https: string
    no_proxy: string
  headers:                 # Custom headers (optional)
    X-Custom: string

# Authentication Configuration
auth:
  type: enum               # jwt | api_key | none
  token: string            # JWT token (if type=jwt)
  api_key: string          # API key (if type=api_key)

# Output Configuration
output:
  format: enum             # json | table | tree
  color: boolean           # Enable colored output
  timestamps: boolean      # Show timestamps
  verbose: boolean         # Verbose output

# Default Values
defaults:
  task:
    priority: enum         # low | medium | high | critical
    timeout: integer       # Task timeout in seconds
  agent:
    cost_per_request: float  # Default cost per request
  workflow:
    max_retries: integer   # Workflow retry attempts

# Logging Configuration (optional)
logging:
  level: enum              # DEBUG | INFO | WARNING | ERROR | CRITICAL
  file: string             # Log file path
  format: string           # Log format string
  max_bytes: integer       # Max log file size
  backup_count: integer    # Number of backups

# Performance Configuration (optional)
performance:
  lazy_imports: boolean    # Lazy load heavy libraries
  cache_config: boolean    # Cache parsed config
  cache_ttl: integer       # Cache TTL in seconds
```

---

**Last Updated**: 2025-10-21
**Version**: 0.1.0
