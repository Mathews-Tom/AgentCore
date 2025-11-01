"""
Training Data Privacy

Provides PII detection, data anonymization, and secure data handling.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class PIIType(str, Enum):
    """Types of personally identifiable information"""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"


class AnonymizationStrategy(str, Enum):
    """Data anonymization strategies"""

    REDACT = "redact"  # Replace with [REDACTED]
    HASH = "hash"  # Replace with hash
    TOKENIZE = "tokenize"  # Replace with token
    MASK = "mask"  # Partial masking


class PrivacyConfig(BaseModel):
    """Configuration for privacy controls"""

    enable_pii_detection: bool = Field(
        default=True, description="Enable PII detection"
    )
    anonymization_strategy: AnonymizationStrategy = Field(
        default=AnonymizationStrategy.REDACT, description="Default anonymization strategy"
    )
    hash_salt: str = Field(
        default="dspy_privacy_salt", description="Salt for hashing"
    )
    enable_data_retention: bool = Field(
        default=True, description="Enable data retention policies"
    )
    retention_days: int = Field(
        default=90, description="Days to retain training data"
    )


@dataclass
class PIIDetection:
    """PII detection result"""

    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float


@dataclass
class AnonymizationResult:
    """Result of data anonymization"""

    original_data: str
    anonymized_data: str
    detections: list[PIIDetection]
    strategy: AnonymizationStrategy
    anonymized_at: datetime


class PIIDetector:
    """
    PII detection service.

    Detects various types of personally identifiable information in text.
    """

    # Regex patterns for PII detection
    PATTERNS = {
        PIIType.EMAIL: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        PIIType.PHONE: r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b",
        PIIType.SSN: r"\b\d{3}-\d{2}-\d{4}\b",
        PIIType.CREDIT_CARD: r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        PIIType.IP_ADDRESS: r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }

    def __init__(self):
        self.logger = structlog.get_logger()
        self._compiled_patterns = {
            pii_type: re.compile(pattern)
            for pii_type, pattern in self.PATTERNS.items()
        }

    def detect(self, text: str) -> list[PIIDetection]:
        """
        Detect PII in text.

        Args:
            text: Text to scan for PII

        Returns:
            List of detected PII instances
        """
        detections: list[PIIDetection] = []

        for pii_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                detection = PIIDetection(
                    pii_type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=self._calculate_confidence(pii_type, match.group()),
                )
                detections.append(detection)

        self.logger.info("pii_detection_completed", detections_count=len(detections))
        return detections

    def _calculate_confidence(self, pii_type: PIIType, value: str) -> float:
        """Calculate confidence score for PII detection"""
        # Basic confidence calculation
        if pii_type == PIIType.EMAIL and "@" in value and "." in value:
            return 0.95
        elif pii_type == PIIType.SSN and len(value.replace("-", "")) == 9:
            return 0.9
        elif pii_type == PIIType.PHONE and len(value.replace("-", "").replace(" ", "")) >= 10:
            return 0.85
        elif pii_type == PIIType.CREDIT_CARD:
            return 0.8
        elif pii_type == PIIType.IP_ADDRESS:
            return 0.75
        return 0.7

    def has_pii(self, text: str) -> bool:
        """Check if text contains PII"""
        return len(self.detect(text)) > 0


class PrivacyManager:
    """
    Privacy management service.

    Provides PII detection, data anonymization, and secure data handling.
    """

    def __init__(self, config: PrivacyConfig | None = None):
        self.config = config or PrivacyConfig()
        self.logger = structlog.get_logger()
        self.pii_detector = PIIDetector()

        # Statistics
        self._privacy_stats = {
            "texts_scanned": 0,
            "pii_detected": 0,
            "texts_anonymized": 0,
            "data_retained": 0,
        }

        self.logger.info(
            "privacy_manager_initialized",
            pii_detection=self.config.enable_pii_detection,
            strategy=self.config.anonymization_strategy,
        )

    def scan_for_pii(self, data: str | dict[str, Any]) -> list[PIIDetection]:
        """
        Scan data for PII.

        Args:
            data: Text or dictionary to scan

        Returns:
            List of detected PII instances
        """
        if not self.config.enable_pii_detection:
            return []

        text = self._extract_text(data)
        detections = self.pii_detector.detect(text)

        self._privacy_stats["texts_scanned"] += 1
        self._privacy_stats["pii_detected"] += len(detections)

        return detections

    def anonymize(
        self,
        data: str | dict[str, Any],
        strategy: AnonymizationStrategy | None = None,
    ) -> AnonymizationResult:
        """
        Anonymize data by removing or masking PII.

        Args:
            data: Data to anonymize
            strategy: Anonymization strategy (uses config default if not provided)

        Returns:
            AnonymizationResult with anonymized data
        """
        strategy = strategy or self.config.anonymization_strategy
        original_text = self._extract_text(data)
        detections = self.pii_detector.detect(original_text)

        anonymized_text = original_text
        offset = 0

        # Sort detections by position
        sorted_detections = sorted(detections, key=lambda d: d.start)

        for detection in sorted_detections:
            start = detection.start + offset
            end = detection.end + offset
            original_value = anonymized_text[start:end]

            # Apply anonymization strategy
            if strategy == AnonymizationStrategy.REDACT:
                replacement = f"[REDACTED_{detection.pii_type.value.upper()}]"
            elif strategy == AnonymizationStrategy.HASH:
                replacement = self._hash_value(original_value)
            elif strategy == AnonymizationStrategy.TOKENIZE:
                replacement = f"[TOKEN_{detection.pii_type.value.upper()}_{hash(original_value) % 10000}]"
            elif strategy == AnonymizationStrategy.MASK:
                replacement = self._mask_value(original_value, detection.pii_type)
            else:
                replacement = "[REDACTED]"

            anonymized_text = (
                anonymized_text[:start] + replacement + anonymized_text[end:]
            )
            offset += len(replacement) - len(original_value)

        self._privacy_stats["texts_anonymized"] += 1

        self.logger.info(
            "data_anonymized",
            detections_count=len(detections),
            strategy=strategy,
        )

        return AnonymizationResult(
            original_data=original_text,
            anonymized_data=anonymized_text,
            detections=detections,
            strategy=strategy,
            anonymized_at=datetime.now(UTC),
        )

    def _extract_text(self, data: str | dict[str, Any]) -> str:
        """Extract text from various data types"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            return str(data)
        else:
            return str(data)

    def _hash_value(self, value: str) -> str:
        """Hash a value using SHA-256"""
        salted = f"{self.config.hash_salt}{value}"
        hashed = hashlib.sha256(salted.encode()).hexdigest()
        return f"[HASH_{hashed[:16]}]"

    def _mask_value(self, value: str, pii_type: PIIType) -> str:
        """Mask a value based on PII type"""
        if pii_type == PIIType.EMAIL:
            parts = value.split("@")
            if len(parts) == 2:
                username = parts[0]
                domain = parts[1]
                masked_username = username[0] + "*" * (len(username) - 1)
                return f"{masked_username}@{domain}"
        elif pii_type == PIIType.PHONE:
            clean = value.replace("-", "").replace(" ", "").replace("(", "").replace(")", "")
            if len(clean) >= 10:
                return f"***-***-{clean[-4:]}"
        elif pii_type == PIIType.CREDIT_CARD:
            clean = value.replace("-", "").replace(" ", "")
            if len(clean) >= 4:
                return f"****-****-****-{clean[-4:]}"

        # Default masking
        if len(value) > 4:
            return "*" * (len(value) - 4) + value[-4:]
        return "*" * len(value)

    def validate_data_privacy(self, data: str | dict[str, Any]) -> dict[str, Any]:
        """
        Validate data privacy compliance.

        Args:
            data: Data to validate

        Returns:
            Validation result with compliance status
        """
        detections = self.scan_for_pii(data)
        has_pii = len(detections) > 0

        return {
            "compliant": not has_pii,
            "pii_detected": has_pii,
            "detection_count": len(detections),
            "detections": [
                {
                    "type": d.pii_type.value,
                    "confidence": d.confidence,
                    "position": (d.start, d.end),
                }
                for d in detections
            ],
            "recommendation": (
                "Anonymize data before storage"
                if has_pii
                else "Data is privacy-compliant"
            ),
        }

    def should_retain_data(self, created_at: datetime) -> bool:
        """
        Check if data should be retained based on retention policy.

        Args:
            created_at: Data creation timestamp

        Returns:
            True if data should be retained
        """
        if not self.config.enable_data_retention:
            return True

        age_days = (datetime.now(UTC) - created_at).days
        return age_days <= self.config.retention_days

    def get_privacy_stats(self) -> dict[str, Any]:
        """Get privacy statistics"""
        return {
            **self._privacy_stats,
            "pii_detection_rate": (
                self._privacy_stats["pii_detected"] / self._privacy_stats["texts_scanned"]
                if self._privacy_stats["texts_scanned"] > 0
                else 0.0
            ),
            "anonymization_rate": (
                self._privacy_stats["texts_anonymized"] / self._privacy_stats["texts_scanned"]
                if self._privacy_stats["texts_scanned"] > 0
                else 0.0
            ),
        }
