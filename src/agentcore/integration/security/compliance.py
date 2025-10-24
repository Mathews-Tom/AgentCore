"""Compliance controls and data residency management.

Provides GDPR/CCPA compliance support, data residency enforcement,
PII detection and redaction, and data classification.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class DataClassification(str, Enum):
    """Data classification levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"
    PHI = "phi"  # Protected Health Information


class DataRegion(str, Enum):
    """Supported data residency regions."""

    US_EAST = "us-east"
    US_WEST = "us-west"
    EU_WEST = "eu-west"
    EU_CENTRAL = "eu-central"
    ASIA_PACIFIC = "asia-pacific"
    CANADA = "canada"
    AUSTRALIA = "australia"
    GLOBAL = "global"


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""

    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci-dss"


class DataResidencyConfig(BaseModel):
    """Data residency configuration.

    Defines where data can be stored and processed based on
    compliance requirements and customer preferences.
    """

    allowed_regions: list[DataRegion] = Field(
        description="Regions where data can be stored",
    )
    primary_region: DataRegion = Field(
        description="Primary data storage region",
    )
    replication_regions: list[DataRegion] = Field(
        default_factory=list,
        description="Regions for data replication",
    )
    enforce_strict: bool = Field(
        default=True,
        description="Strict enforcement (reject operations outside allowed regions)",
    )
    compliance_frameworks: list[ComplianceFramework] = Field(
        default_factory=list,
        description="Compliance frameworks to enforce",
    )


class PIIPattern(BaseModel):
    """PII detection pattern.

    Defines regex patterns and detection rules for identifying
    personally identifiable information in data.
    """

    name: str = Field(
        description="Pattern name (e.g., 'email', 'ssn', 'phone')",
    )
    pattern: str = Field(
        description="Regex pattern for detection",
    )
    classification: DataClassification = Field(
        description="Data classification when matched",
    )
    redaction_mask: str = Field(
        default="***",
        description="Mask to use when redacting",
    )


class ComplianceManager:
    """Compliance controls and data residency manager.

    Provides GDPR/CCPA compliance support, data residency enforcement,
    PII detection and redaction, and data classification.
    """

    # Default PII patterns
    DEFAULT_PII_PATTERNS = [
        PIIPattern(
            name="email",
            pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            classification=DataClassification.PII,
            redaction_mask="***@***.***",
        ),
        PIIPattern(
            name="ssn",
            pattern=r"\b\d{3}-\d{2}-\d{4}\b",
            classification=DataClassification.PII,
            redaction_mask="***-**-****",
        ),
        PIIPattern(
            name="phone_us",
            pattern=r"(?<![a-zA-Z0-9-])(?:\d{3}[-.\s]\d{3}[-.\s]\d{4}|\d{3}[-.\s]\d{4})(?![a-zA-Z0-9-])",
            classification=DataClassification.PII,
            redaction_mask="***-****",
        ),
        PIIPattern(
            name="credit_card",
            pattern=r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            classification=DataClassification.PII,
            redaction_mask="****-****-****-****",
        ),
        PIIPattern(
            name="ip_address",
            pattern=r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            classification=DataClassification.INTERNAL,
            redaction_mask="***.***.***.***",
        ),
    ]

    def __init__(
        self,
        residency_config: DataResidencyConfig | None = None,
        custom_pii_patterns: list[PIIPattern] | None = None,
    ) -> None:
        """Initialize compliance manager.

        Args:
            residency_config: Data residency configuration
            custom_pii_patterns: Additional PII detection patterns
        """
        self._residency_config = residency_config
        self._pii_patterns = self.DEFAULT_PII_PATTERNS.copy()

        if custom_pii_patterns:
            self._pii_patterns.extend(custom_pii_patterns)

        # Compile regex patterns
        self._compiled_patterns: dict[str, re.Pattern[str]] = {
            p.name: re.compile(p.pattern) for p in self._pii_patterns
        }

        logger.info(
            "compliance_manager_initialized",
            has_residency_config=residency_config is not None,
            pii_pattern_count=len(self._pii_patterns),
        )

    def detect_pii(self, text: str) -> dict[str, list[str]]:
        """Detect PII in text.

        Args:
            text: Text to scan for PII

        Returns:
            Dictionary mapping pattern names to list of matches
        """
        matches: dict[str, list[str]] = {}

        for pattern in self._pii_patterns:
            compiled = self._compiled_patterns[pattern.name]
            pattern_matches = compiled.findall(text)

            if pattern_matches:
                matches[pattern.name] = pattern_matches

        if matches:
            logger.warning(
                "pii_detected",
                pattern_types=list(matches.keys()),
                match_counts={k: len(v) for k, v in matches.items()},
            )

        return matches

    def redact_pii(self, text: str, patterns: list[str] | None = None) -> str:
        """Redact PII from text.

        Args:
            text: Text to redact
            patterns: Specific patterns to redact (None = all patterns)

        Returns:
            Text with PII redacted
        """
        redacted = text

        patterns_to_use = (
            [p for p in self._pii_patterns if p.name in patterns]
            if patterns
            else self._pii_patterns
        )

        for pattern in patterns_to_use:
            compiled = self._compiled_patterns[pattern.name]
            matches = compiled.findall(redacted)

            if matches:
                for match in matches:
                    redacted = redacted.replace(match, pattern.redaction_mask)

        return redacted

    def classify_data(self, data: dict[str, Any]) -> DataClassification:
        """Classify data based on content analysis.

        Args:
            data: Data to classify

        Returns:
            Data classification level
        """
        # Convert data to string for analysis
        data_str = str(data)

        # Check keys explicitly for credential patterns
        credential_keys = {"password", "secret", "token", "api_key", "credential", "key"}
        has_credential_key = any(
            any(cred_key in k.lower() for cred_key in credential_keys)
            for k in data.keys()
        )

        # Check for PII
        pii_matches = self.detect_pii(data_str)

        if pii_matches:
            # Check for PHI patterns
            if any("medical" in k or "health" in k for k in data.keys()):
                return DataClassification.PHI

            return DataClassification.PII

        # Check for credential-like patterns in keys (higher priority)
        if has_credential_key:
            return DataClassification.RESTRICTED

        # Check for internal patterns
        if any(k in data_str.lower() for k in ["internal", "confidential", "private"]):
            return DataClassification.CONFIDENTIAL

        return DataClassification.INTERNAL

    def validate_region(self, region: DataRegion) -> bool:
        """Validate if operation in region is allowed.

        Args:
            region: Region to validate

        Returns:
            True if region is allowed

        Raises:
            ValueError: If strict enforcement and region not allowed
        """
        if not self._residency_config:
            return True

        allowed = (
            region in self._residency_config.allowed_regions
            or region == DataRegion.GLOBAL
        )

        if not allowed:
            logger.warning(
                "data_residency_violation",
                region=region.value,
                allowed_regions=[r.value for r in self._residency_config.allowed_regions],
            )

            if self._residency_config.enforce_strict:
                raise ValueError(
                    f"Data operation in region {region.value} not allowed. "
                    f"Allowed regions: {[r.value for r in self._residency_config.allowed_regions]}"
                )

        return allowed

    def get_allowed_regions(self) -> list[DataRegion]:
        """Get list of allowed data regions.

        Returns:
            List of allowed regions
        """
        if not self._residency_config:
            return [r for r in DataRegion]

        return self._residency_config.allowed_regions

    def check_compliance(
        self,
        framework: ComplianceFramework,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Check data compliance with framework requirements.

        Args:
            framework: Compliance framework to check
            data: Data to validate

        Returns:
            Compliance check result with violations and recommendations
        """
        violations: list[str] = []
        recommendations: list[str] = []

        # Detect PII
        data_str = str(data)
        pii_matches = self.detect_pii(data_str)

        if framework == ComplianceFramework.GDPR:
            # GDPR requires explicit consent for PII processing
            if pii_matches:
                violations.append("PII detected without consent verification")
                recommendations.append("Ensure user consent is obtained for PII processing")

            # Check for right to erasure support
            if "user_id" not in data and pii_matches:
                violations.append("No user_id for right to erasure")
                recommendations.append("Include user_id to support data deletion requests")

        elif framework == ComplianceFramework.CCPA:
            # CCPA requires disclosure of data collection
            if pii_matches:
                recommendations.append("Ensure privacy notice discloses PII collection")

            # Check for opt-out mechanism
            if pii_matches and "consent" not in data:
                violations.append("No opt-out mechanism for PII")
                recommendations.append("Implement opt-out mechanism for data sale")

        elif framework == ComplianceFramework.HIPAA:
            # HIPAA requires PHI protection
            classification = self.classify_data(data)
            if classification == DataClassification.PHI:
                violations.append("PHI detected - ensure HIPAA safeguards")
                recommendations.append("Encrypt PHI at rest and in transit")

        elif framework == ComplianceFramework.SOC2:
            # SOC2 requires security controls
            if pii_matches:
                recommendations.append("Ensure SOC2 security controls for PII")

        elif framework == ComplianceFramework.PCI_DSS:
            # PCI DSS for credit card data
            if "credit_card" in pii_matches:
                violations.append("Credit card data detected - PCI DSS required")
                recommendations.append("Implement PCI DSS controls for cardholder data")

        result = {
            "framework": framework.value,
            "compliant": len(violations) == 0,
            "violations": violations,
            "recommendations": recommendations,
            "pii_detected": bool(pii_matches),
            "pii_types": list(pii_matches.keys()) if pii_matches else [],
        }

        logger.info(
            "compliance_check_completed",
            framework=framework.value,
            compliant=result["compliant"],
            violation_count=len(violations),
            pii_detected=result["pii_detected"],
        )

        return result

    def sanitize_logs(self, log_data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize log data by redacting PII.

        Args:
            log_data: Log data to sanitize

        Returns:
            Sanitized log data with PII redacted
        """
        sanitized = {}

        for key, value in log_data.items():
            if isinstance(value, str):
                sanitized[key] = self.redact_pii(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_logs(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize_logs(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value

        return sanitized

    def add_pii_pattern(self, pattern: PIIPattern) -> None:
        """Add custom PII detection pattern.

        Args:
            pattern: PII pattern to add
        """
        self._pii_patterns.append(pattern)
        self._compiled_patterns[pattern.name] = re.compile(pattern.pattern)

        logger.info(
            "pii_pattern_added",
            pattern_name=pattern.name,
            classification=pattern.classification.value,
        )

    def generate_compliance_report(self) -> dict[str, Any]:
        """Generate compliance configuration report.

        Returns:
            Compliance report with current configuration
        """
        report = {
            "data_residency": {
                "configured": self._residency_config is not None,
                "allowed_regions": (
                    [r.value for r in self._residency_config.allowed_regions]
                    if self._residency_config
                    else []
                ),
                "primary_region": (
                    self._residency_config.primary_region.value
                    if self._residency_config
                    else None
                ),
                "strict_enforcement": (
                    self._residency_config.enforce_strict if self._residency_config else False
                ),
                "compliance_frameworks": (
                    [f.value for f in self._residency_config.compliance_frameworks]
                    if self._residency_config
                    else []
                ),
            },
            "pii_detection": {
                "pattern_count": len(self._pii_patterns),
                "patterns": [
                    {
                        "name": p.name,
                        "classification": p.classification.value,
                    }
                    for p in self._pii_patterns
                ],
            },
        }

        logger.info(
            "compliance_report_generated",
            has_residency_config=report["data_residency"]["configured"],
            pii_pattern_count=report["pii_detection"]["pattern_count"],
        )

        return report
