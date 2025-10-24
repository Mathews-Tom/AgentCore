"""
Security and compliance module for the Integration Layer.

Provides comprehensive security features including:
- Credential encryption and rotation
- Audit logging and compliance trails
- Data residency and privacy controls
- Security scanning and validation
"""

from agentcore.integration.security.credential_manager import (
    CredentialManager,
    EncryptedCredential,
    CredentialType,
    CredentialStatus,
)
from agentcore.integration.security.audit_logger import (
    AuditLogger,
    AuditEvent,
    AuditAction,
    AuditOutcome,
)
from agentcore.integration.security.compliance import (
    ComplianceManager,
    ComplianceFramework,
    DataRegion,
    DataResidencyConfig,
    DataClassification,
    PIIPattern,
)
from agentcore.integration.security.scanner import (
    SecurityScanner,
    SecurityPolicy,
    SecurityIssue,
    SecurityIssueType,
    SecuritySeverity,
)

__all__ = [
    # Credential Management
    "CredentialManager",
    "EncryptedCredential",
    "CredentialType",
    "CredentialStatus",
    # Audit Logging
    "AuditLogger",
    "AuditEvent",
    "AuditAction",
    "AuditOutcome",
    # Compliance
    "ComplianceManager",
    "ComplianceFramework",
    "DataRegion",
    "DataResidencyConfig",
    "DataClassification",
    "PIIPattern",
    # Security Scanning
    "SecurityScanner",
    "SecurityPolicy",
    "SecurityIssue",
    "SecurityIssueType",
    "SecuritySeverity",
]
