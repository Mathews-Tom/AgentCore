# Python Sandbox Docker Image

Secure Docker container for Python code execution with resource limits and security profiles.

## Features

- **Minimal base image**: python:3.12-slim
- **Non-root user**: Runs as `sandbox` user (UID 1000)
- **Network isolation**: No network access
- **Read-only filesystem**: Except `/tmp/sandbox`
- **Resource limits**: 512MB RAM, 1 CPU
- **Security hardening**: AppArmor profile, no privilege escalation

## Building the Image

```bash
cd docker/python-sandbox
docker build -t agentcore-python-sandbox .
```

## Testing the Image

```bash
# Test basic execution
docker run --rm agentcore-python-sandbox python3 -c "print('Hello from sandbox')"

# Test with security restrictions
docker run --rm \
  --memory=512m \
  --cpus=1 \
  --network=none \
  --read-only \
  --tmpfs /tmp/sandbox:rw,size=100m,mode=1777 \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --user=sandbox \
  agentcore-python-sandbox \
  python3 -c "print('Secure execution')"
```

## Security Profile

The AppArmor profile (`apparmor-profile`) provides additional security restrictions:

```bash
# Load the AppArmor profile
sudo apparmor_parser -r -W apparmor-profile

# Run container with AppArmor profile
docker run --rm \
  --security-opt apparmor=agentcore-python-sandbox \
  agentcore-python-sandbox \
  python3 -c "print('Protected execution')"
```

## Usage in ExecutePythonTool

The Docker image is automatically used by `ExecutePythonTool` when Docker is available:

```python
from agentcore.agent_runtime.tools.builtin.code_execution_tools import ExecutePythonTool
from agentcore.agent_runtime.tools.base import ExecutionContext

tool = ExecutePythonTool(use_docker=True)

context = ExecutionContext(user_id="user123", agent_id="agent456")

result = await tool.execute(
    parameters={"code": "print('Hello, World!')", "timeout": 30},
    context=context
)
```

## Resource Limits

- **Memory**: 512MB (no swap)
- **CPU**: 1 core
- **Disk**: 100MB in /tmp/sandbox
- **Network**: Disabled
- **Execution Time**: Configurable timeout (default: 30s, max: 300s)

## Security Considerations

1. **No network access**: Containers cannot make external network requests
2. **Read-only root**: Filesystem is read-only except /tmp/sandbox
3. **No privilege escalation**: Cannot gain additional privileges
4. **Capabilities dropped**: All Linux capabilities are dropped
5. **Non-root user**: Runs as unprivileged `sandbox` user
6. **Resource limits**: Prevents resource exhaustion attacks

## Maintenance

### Updating Dependencies

To add Python packages to the sandbox:

```dockerfile
# In Dockerfile, uncomment and modify:
RUN pip install --no-cache-dir numpy pandas
```

### Security Updates

Rebuild the image regularly to include security updates:

```bash
docker build --no-cache --pull -t agentcore-python-sandbox .
```

## Troubleshooting

### Container Not Starting

Check Docker daemon is running:
```bash
docker info
```

### Permission Denied Errors

Ensure user has Docker permissions:
```bash
sudo usermod -aG docker $USER
```

### AppArmor Profile Issues

Check AppArmor is enabled:
```bash
sudo aa-status
```

Load profile:
```bash
sudo apparmor_parser -r apparmor-profile
```
