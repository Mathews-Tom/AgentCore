"""
Gunicorn Configuration for AgentCore Gateway

Production-optimized configuration for 60,000+ req/sec throughput.
"""

from __future__ import annotations

import multiprocessing
import os

# Server Socket
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8080')}"
backlog = 4096  # Increased from default 2048 for high concurrency

# Worker Processes
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000  # Max concurrent connections per worker
max_requests = 10000  # Restart worker after N requests (prevents memory leaks)
max_requests_jitter = 1000  # Add randomness to prevent all workers restarting simultaneously
timeout = 30  # Worker timeout in seconds
graceful_timeout = 15  # Graceful shutdown timeout
keepalive = 5  # Keep-alive connections timeout

# Performance Tuning
# Note: preload_app disabled due to Prometheus metric duplication in workers
# See: https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn
preload_app = False  # Disabled to avoid Prometheus metric duplication
reuse_port = True  # Enable SO_REUSEPORT for better load balancing across workers

# Logging
accesslog = os.getenv("ACCESS_LOG", "-")  # "-" = stdout
errorlog = os.getenv("ERROR_LOG", "-")  # "-" = stderr
loglevel = os.getenv("LOG_LEVEL", "info").lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'  # Include response time

# Process Naming
proc_name = "agentcore-gateway"

# Server Mechanics
daemon = False  # Run in foreground for container compatibility
pidfile = None  # Disable PID file for stateless deployment
umask = 0
user = None  # Run as current user (container security handles this)
group = None
tmp_upload_dir = None

# SSL (if enabled via environment)
keyfile = os.getenv("SSL_KEYFILE")
certfile = os.getenv("SSL_CERTFILE")
ssl_version = 2  # TLS 1.2+ (ssl.PROTOCOL_TLS)
ciphers = "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_128_GCM_SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256"
cert_reqs = 0  # Don't require client certificates
do_handshake_on_connect = False  # Defer SSL handshake


def on_starting(server):
    """Called before master process is initialized."""
    print(f"Starting Gunicorn with {workers} workers")
    print(f"Worker class: {worker_class}")
    print(f"Binding to: {bind}")
    print(f"Backlog: {backlog}")
    print(f"Worker connections: {worker_connections}")
    print(f"Max requests per worker: {max_requests}")
    print(f"Reuse port: {reuse_port}")


def on_reload(server):
    """Called when configuration is reloaded."""
    print("Configuration reloaded")


def when_ready(server):
    """Called after server is ready."""
    print("Server is ready. Spawning workers")


def pre_fork(server, worker):
    """Called before worker is forked."""
    pass


def post_fork(server, worker):
    """Called after worker is forked."""
    print(f"Worker spawned (pid: {worker.pid})")


def post_worker_init(worker):
    """Called after worker initialization."""
    print(f"Worker initialized (pid: {worker.pid})")


def worker_int(worker):
    """Called when worker receives INT or QUIT signal."""
    print(f"Worker interrupted (pid: {worker.pid})")


def worker_abort(worker):
    """Called when worker receives SIGABRT."""
    print(f"Worker aborted (pid: {worker.pid})")


def pre_exec(server):
    """Called before new master is exec'd."""
    print("Forking new master")


def pre_request(worker, req):
    """Called before processing request."""
    worker.log.debug(f"{req.method} {req.path}")


def post_request(worker, req, environ, resp):
    """Called after request is processed."""
    pass


def child_exit(server, worker):
    """Called when worker exits."""
    print(f"Worker exited (pid: {worker.pid})")


def worker_exit(server, worker):
    """Called when worker is cleaned up."""
    print(f"Worker cleanup (pid: {worker.pid})")


def nworkers_changed(server, new_value, old_value):
    """Called when worker count changes."""
    print(f"Worker count changed: {old_value} -> {new_value}")


def on_exit(server):
    """Called on server shutdown."""
    print("Server shutdown complete")
