"""
Security and Privacy Module for DSPy Optimization

Provides model encryption, data privacy, access control, and audit trails.
"""

from __future__ import annotations

from agentcore.dspy_optimization.security.encryption import (
    ModelEncryption,
    EncryptionConfig,
)
from agentcore.dspy_optimization.security.privacy import (
    PrivacyManager,
    PrivacyConfig,
    PIIDetector,
)
from agentcore.dspy_optimization.security.access_control import (
    AccessController,
    AccessConfig,
    SecurityRole,
    SecurityPermission,
)
from agentcore.dspy_optimization.security.audit import (
    AuditLogger,
    AuditConfig,
    AuditEvent,
    AuditEventType,
)
from agentcore.dspy_optimization.security.compliance import (
    ComplianceValidator,
    ComplianceConfig,
    ComplianceStandard,
    ComplianceReport,
)

__all__ = [
    "ModelEncryption",
    "EncryptionConfig",
    "PrivacyManager",
    "PrivacyConfig",
    "PIIDetector",
    "AccessController",
    "AccessConfig",
    "SecurityRole",
    "SecurityPermission",
    "AuditLogger",
    "AuditConfig",
    "AuditEvent",
    "AuditEventType",
    "ComplianceValidator",
    "ComplianceConfig",
    "ComplianceStandard",
    "ComplianceReport",
]
