"""
Production Deployment Configuration

Optimized settings for 60,000+ req/sec throughput with Gunicorn + Uvicorn workers.
"""

from __future__ import annotations

import multiprocessing
import os


class DeploymentConfig:
    """Production deployment configuration."""

    # Worker Configuration
    WORKERS = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
    """Number of worker processes (recommended: CPU cores * 2 + 1)"""

    WORKER_CLASS = "uvicorn.workers.UvicornWorker"
    """Worker class for async support"""

    WORKER_CONNECTIONS = 1000
    """Max concurrent connections per worker"""

    # Uvicorn Worker Options (passed via environment)
    UVICORN_LOOP = os.getenv("UVICORN_LOOP", "uvloop")
    """Event loop implementation (uvloop is faster than asyncio)"""

    UVICORN_HTTP = os.getenv("UVICORN_HTTP", "httptools")
    """HTTP protocol implementation (httptools is faster than h11)"""

    UVICORN_LIFESPAN = "on"
    """Enable lifespan events"""

    UVICORN_BACKLOG = 4096
    """Socket backlog size (increased from default 2048)"""

    UVICORN_LIMIT_CONCURRENCY = None
    """Limit concurrent connections (None = unlimited)"""

    UVICORN_LIMIT_MAX_REQUESTS = None
    """Max requests before worker restart (None = unlimited)"""

    UVICORN_TIMEOUT_KEEP_ALIVE = 5
    """Keep-alive timeout in seconds"""

    UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN = 15
    """Graceful shutdown timeout in seconds"""

    # HTTP/2 Configuration
    HTTP2_ENABLED = os.getenv("HTTP2_ENABLED", "true").lower() == "true"
    """Enable HTTP/2 support"""

    HTTP2_MAX_CONCURRENT_STREAMS = 100
    """Max concurrent HTTP/2 streams per connection"""

    HTTP2_MAX_HEADER_LIST_SIZE = 16384
    """Max HTTP/2 header list size in bytes"""

    # Performance Tuning
    KEEPALIVE_TIMEOUT = 5
    """TCP keep-alive timeout in seconds"""

    BACKLOG = 4096
    """Listen socket backlog"""

    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10 MB
    """Maximum request body size"""

    REQUEST_TIMEOUT = 30
    """Request processing timeout in seconds"""

    # Connection Pooling
    CONNECTION_POOL_MAX_SIZE = 1000
    """Maximum connections in pool"""

    CONNECTION_POOL_MAX_KEEPALIVE = 500
    """Maximum keep-alive connections"""

    CONNECTION_POOL_KEEPALIVE_EXPIRY = 30.0
    """Keep-alive connection expiry in seconds"""

    # Response Caching
    CACHE_ENABLED = True
    """Enable response caching"""

    CACHE_TTL = 300
    """Default cache TTL in seconds (5 minutes)"""

    CACHE_MAX_SIZE = 10000
    """Maximum cached responses"""

    # Resource Limits
    WORKER_MAX_REQUESTS = 10000
    """Restart worker after N requests (prevents memory leaks)"""

    WORKER_MAX_REQUESTS_JITTER = 1000
    """Jitter for max_requests to prevent thundering herd"""

    WORKER_TIMEOUT = 30
    """Worker process timeout in seconds"""

    GRACEFUL_TIMEOUT = 15
    """Graceful shutdown timeout"""

    # OS Tuning Recommendations (apply via sysctl)
    OS_TUNING = {
        "net.core.somaxconn": 65535,  # Increase listen backlog
        "net.ipv4.tcp_max_syn_backlog": 8192,  # SYN flood protection
        "net.ipv4.ip_local_port_range": "1024 65535",  # Ephemeral port range
        "net.ipv4.tcp_tw_reuse": 1,  # Reuse TIME_WAIT sockets
        "net.ipv4.tcp_fin_timeout": 15,  # Faster FIN timeout
        "net.ipv4.tcp_keepalive_time": 300,  # Keep-alive probe interval
        "net.ipv4.tcp_keepalive_probes": 3,  # Keep-alive probe count
        "net.ipv4.tcp_keepalive_intvl": 15,  # Keep-alive probe interval
        "fs.file-max": 2097152,  # Max open files system-wide
        # Set via ulimit: ulimit -n 65535 (max open files per process)
    }

    @classmethod
    def get_uvicorn_env(cls) -> dict[str, str]:
        """
        Get environment variables for uvicorn worker configuration.

        Returns:
            Dictionary of environment variables
        """
        return {
            "UVICORN_LOOP": cls.UVICORN_LOOP,
            "UVICORN_HTTP": cls.UVICORN_HTTP,
            "UVICORN_LIFESPAN": cls.UVICORN_LIFESPAN,
            "UVICORN_BACKLOG": str(cls.UVICORN_BACKLOG),
            "UVICORN_TIMEOUT_KEEP_ALIVE": str(cls.UVICORN_TIMEOUT_KEEP_ALIVE),
        }

    @classmethod
    def get_gunicorn_command(cls) -> str:
        """
        Generate Gunicorn command line for production deployment.

        Returns:
            Gunicorn command string
        """
        return (
            f"gunicorn gateway.main:app "
            f"--workers {cls.WORKERS} "
            f"--worker-class {cls.WORKER_CLASS} "
            f"--worker-connections {cls.WORKER_CONNECTIONS} "
            f"--backlog {cls.BACKLOG} "
            f"--timeout {cls.WORKER_TIMEOUT} "
            f"--graceful-timeout {cls.GRACEFUL_TIMEOUT} "
            f"--max-requests {cls.WORKER_MAX_REQUESTS} "
            f"--max-requests-jitter {cls.WORKER_MAX_REQUESTS_JITTER} "
            f"--keep-alive {cls.KEEPALIVE_TIMEOUT} "
            f"--reuse-port "
            f"--config src/gateway/gunicorn.conf.py "
            f"--bind 0.0.0.0:8080"
        )

    @classmethod
    def print_deployment_guide(cls) -> None:
        """Print production deployment guide."""
        print("=" * 80)
        print("AgentCore Gateway - Production Deployment Guide")
        print("=" * 80)
        print()
        print("1. System Tuning (run as root):")
        print("   " + "-" * 76)
        for key, value in cls.OS_TUNING.items():
            print(f"   sysctl -w {key}={value}")
        print()
        print("2. File Limits:")
        print("   " + "-" * 76)
        print("   ulimit -n 65535  # Max open files")
        print()
        print("3. Environment Variables:")
        print("   " + "-" * 76)
        for key, value in cls.get_uvicorn_env().items():
            print(f"   export {key}={value}")
        print()
        print("4. Start Gateway:")
        print("   " + "-" * 76)
        print(f"   {cls.get_gunicorn_command()}")
        print()
        print("5. Performance Validation:")
        print("   " + "-" * 76)
        print("   uv run locust -f tests/load/locustfile.py --host http://localhost:8080")
        print()
        print("Expected Performance:")
        print(f"   - Workers: {cls.WORKERS}")
        print(f"   - Connections per worker: {cls.WORKER_CONNECTIONS}")
        print(f"   - Total capacity: ~{cls.WORKERS * cls.WORKER_CONNECTIONS} concurrent connections")
        print("   - Target throughput: 60,000+ req/sec")
        print("   - Latency (p95): <5ms routing overhead")
        print()
        print("=" * 80)


if __name__ == "__main__":
    DeploymentConfig.print_deployment_guide()
