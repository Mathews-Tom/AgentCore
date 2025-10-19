# Multi-stage Docker build for AgentCore A2A Protocol Layer
# Hardened for production deployment
FROM python:3.12-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install UV package manager
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.12-slim

# Security labels
LABEL maintainer="AgentCore Team"
LABEL version="0.1.0"
LABEL description="A2A Protocol Layer - Production"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libpq5 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && groupadd -r -g 1000 agentcore \
    && useradd -r -u 1000 -g agentcore -d /app -s /sbin/nologin agentcore

# Set working directory
WORKDIR /app

# Copy virtual environment and application from builder
COPY --from=builder --chown=agentcore:agentcore /app/.venv /app/.venv
COPY --from=builder --chown=agentcore:agentcore /app/src /app/src
COPY --from=builder --chown=agentcore:agentcore /app/alembic /app/alembic
COPY --from=builder --chown=agentcore:agentcore /app/alembic.ini /app/

# Set up environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBUG=False \
    LOG_LEVEL=INFO

# Switch to non-root user
USER agentcore

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/api/v1/health/live || exit 1

# Expose port
EXPOSE 8001

# Start the application with production settings
CMD ["uvicorn", "agentcore.a2a_protocol.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--log-level", "info", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]