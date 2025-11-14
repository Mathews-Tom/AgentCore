# Python Sandbox Docker Image for Secure Code Execution
# Implements TOOL-011 requirements: minimal dependencies, security hardening

FROM python:3.12-slim

# Install minimal dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for code execution
RUN useradd -m -u 1000 -s /bin/bash sandbox && \
    mkdir -p /tmp/sandbox && \
    chown -R sandbox:sandbox /tmp/sandbox

# Set working directory
WORKDIR /tmp/sandbox

# Install minimal Python packages (if needed)
# RUN pip install --no-cache-dir <packages>

# Switch to non-root user
USER sandbox

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command (will be overridden)
CMD ["python3"]
