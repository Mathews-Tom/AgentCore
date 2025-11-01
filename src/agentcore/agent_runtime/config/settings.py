"""Agent Runtime configuration settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Agent Runtime Layer configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application Settings
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    agent_runtime_port: int = 8002

    # Database Configuration
    database_url: str = "postgresql://agentcore:dev@localhost:5432/agentcore_dev"

    # Redis Configuration
    redis_url: str = "redis://localhost:6379"

    # Kubernetes Configuration
    kubernetes_namespace: str = "agentcore-runtime"
    kubernetes_service_host: str | None = None

    # Container Configuration
    container_registry: str = "agentcore.azurecr.io"
    agent_image_base: str = "agentcore/hardened-python:3.12"
    agent_image_registry: str = "localhost:5000"

    # Agent Runtime Limits
    max_concurrent_agents: int = 1000
    agent_startup_timeout: int = 30
    tool_execution_timeout: int = 60
    checkpoint_interval: int = 300
    resource_cleanup_interval: int = 60

    # Security Configuration
    security_profile_default: Literal["minimal", "standard", "privileged"] = "standard"
    seccomp_profile_path: str = "/app/security/seccomp/agent-restricted.json"

    # Resource Limits (defaults per agent)
    default_memory_limit_mb: int = 512
    default_cpu_limit: float = 1.0
    default_execution_time_seconds: int = 300
    default_storage_quota_mb: int = 1024

    # Sandbox Security Configuration
    sandbox_enabled: bool = True
    sandbox_strict_mode: bool = True
    sandbox_default_timeout_seconds: int = 30
    sandbox_default_memory_mb: int = 256
    sandbox_default_cpu_percent: float = 50.0
    sandbox_max_processes: int = 10
    sandbox_max_file_descriptors: int = 50
    sandbox_max_network_requests: int = 100
    sandbox_workspace_root: str = "/tmp/agentcore/sandboxes"
    sandbox_audit_log_dir: str = "/var/log/agentcore/audit"
    sandbox_audit_retention_days: int = 90
    sandbox_audit_max_logs_memory: int = 10000

    # Monitoring Configuration
    enable_metrics: bool = True
    metrics_port: int = 9090

    # Portkey LLM Configuration
    portkey_api_key: str = ""
    portkey_base_url: str = "https://api.portkey.ai"
    portkey_virtual_key: str = ""
    default_llm_model: str = "gpt-5"
    llm_fallback_models: list[str] = ["gpt-5-mini", "gpt-5-mini"]
    llm_temperature: float = 0.7
    llm_max_tokens: int = 500
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 3
    llm_cache_enabled: bool = True

    @property
    def is_kubernetes(self) -> bool:
        """Check if running in Kubernetes environment."""
        return self.kubernetes_service_host is not None


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
