"""
Tests for training data privacy
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agentcore.dspy_optimization.security.privacy import (
    AnonymizationStrategy,
    PIIDetector,
    PIIType,
    PrivacyConfig,
    PrivacyManager,
)


class TestPrivacyConfig:
    """Tests for PrivacyConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = PrivacyConfig()
        assert config.enable_pii_detection is True
        assert config.anonymization_strategy == AnonymizationStrategy.REDACT
        assert config.enable_data_retention is True
        assert config.retention_days == 90


class TestPIIDetector:
    """Tests for PIIDetector"""

    @pytest.fixture
    def detector(self) -> PIIDetector:
        """Create PII detector"""
        return PIIDetector()

    def test_detect_email(self, detector: PIIDetector):
        """Test email detection"""
        text = "Contact me at john.doe@example.com for details"
        detections = detector.detect(text)

        assert len(detections) == 1
        assert detections[0].pii_type == PIIType.EMAIL
        assert detections[0].value == "john.doe@example.com"

    def test_detect_phone(self, detector: PIIDetector):
        """Test phone number detection"""
        text = "Call me at 123-456-7890 or (555) 123-4567"
        detections = detector.detect(text)

        assert len(detections) >= 1
        assert any(d.pii_type == PIIType.PHONE for d in detections)

    def test_detect_ssn(self, detector: PIIDetector):
        """Test SSN detection"""
        text = "My SSN is 123-45-6789"
        detections = detector.detect(text)

        assert len(detections) == 1
        assert detections[0].pii_type == PIIType.SSN
        assert detections[0].value == "123-45-6789"

    def test_detect_credit_card(self, detector: PIIDetector):
        """Test credit card detection"""
        text = "Card number: 1234-5678-9012-3456"
        detections = detector.detect(text)

        assert len(detections) == 1
        assert detections[0].pii_type == PIIType.CREDIT_CARD

    def test_detect_ip_address(self, detector: PIIDetector):
        """Test IP address detection"""
        text = "Server IP: 192.168.1.1"
        detections = detector.detect(text)

        assert len(detections) == 1
        assert detections[0].pii_type == PIIType.IP_ADDRESS
        assert detections[0].value == "192.168.1.1"

    def test_detect_multiple_pii_types(self, detector: PIIDetector):
        """Test detection of multiple PII types"""
        text = "Email: john@example.com, Phone: 123-456-7890, SSN: 123-45-6789"
        detections = detector.detect(text)

        assert len(detections) >= 3
        detected_types = {d.pii_type for d in detections}
        assert PIIType.EMAIL in detected_types
        assert PIIType.PHONE in detected_types
        assert PIIType.SSN in detected_types

    def test_has_pii_true(self, detector: PIIDetector):
        """Test has_pii returns True"""
        text = "Contact: john@example.com"
        assert detector.has_pii(text) is True

    def test_has_pii_false(self, detector: PIIDetector):
        """Test has_pii returns False"""
        text = "This is a clean text without PII"
        assert detector.has_pii(text) is False

    def test_confidence_scores(self, detector: PIIDetector):
        """Test confidence scores are calculated"""
        text = "Email: john@example.com"
        detections = detector.detect(text)

        assert len(detections) == 1
        assert 0.0 <= detections[0].confidence <= 1.0
        assert detections[0].confidence > 0.8  # High confidence for email


class TestPrivacyManager:
    """Tests for PrivacyManager"""

    @pytest.fixture
    def manager(self) -> PrivacyManager:
        """Create privacy manager"""
        return PrivacyManager()

    def test_initialization(self, manager: PrivacyManager):
        """Test privacy manager initialization"""
        assert manager.config.enable_pii_detection is True
        assert manager.pii_detector is not None

    def test_scan_for_pii_string(self, manager: PrivacyManager):
        """Test scanning string for PII"""
        text = "Contact me at john@example.com"
        detections = manager.scan_for_pii(text)

        assert len(detections) == 1
        assert detections[0].pii_type == PIIType.EMAIL

    def test_scan_for_pii_dict(self, manager: PrivacyManager):
        """Test scanning dictionary for PII"""
        data = {"email": "john@example.com", "name": "John Doe"}
        detections = manager.scan_for_pii(data)

        assert len(detections) >= 1

    def test_anonymize_redact(self, manager: PrivacyManager):
        """Test anonymization with REDACT strategy"""
        text = "Contact me at john@example.com"
        result = manager.anonymize(text, AnonymizationStrategy.REDACT)

        assert "john@example.com" not in result.anonymized_data
        assert "[REDACTED_EMAIL]" in result.anonymized_data
        assert len(result.detections) == 1
        assert result.strategy == AnonymizationStrategy.REDACT

    def test_anonymize_hash(self, manager: PrivacyManager):
        """Test anonymization with HASH strategy"""
        text = "Contact me at john@example.com"
        result = manager.anonymize(text, AnonymizationStrategy.HASH)

        assert "john@example.com" not in result.anonymized_data
        assert "[HASH_" in result.anonymized_data
        assert result.strategy == AnonymizationStrategy.HASH

    def test_anonymize_tokenize(self, manager: PrivacyManager):
        """Test anonymization with TOKENIZE strategy"""
        text = "Contact me at john@example.com"
        result = manager.anonymize(text, AnonymizationStrategy.TOKENIZE)

        assert "john@example.com" not in result.anonymized_data
        assert "[TOKEN_EMAIL_" in result.anonymized_data
        assert result.strategy == AnonymizationStrategy.TOKENIZE

    def test_anonymize_mask(self, manager: PrivacyManager):
        """Test anonymization with MASK strategy"""
        text = "Email: john.doe@example.com"
        result = manager.anonymize(text, AnonymizationStrategy.MASK)

        assert "john.doe@example.com" not in result.anonymized_data
        assert result.strategy == AnonymizationStrategy.MASK

    def test_anonymize_multiple_pii(self, manager: PrivacyManager):
        """Test anonymizing multiple PII instances"""
        text = "Email: john@example.com, Phone: 123-456-7890"
        result = manager.anonymize(text)

        assert "john@example.com" not in result.anonymized_data
        assert "123-456-7890" not in result.anonymized_data
        assert len(result.detections) >= 2

    def test_anonymize_preserves_non_pii(self, manager: PrivacyManager):
        """Test that anonymization preserves non-PII content"""
        text = "Hello, contact me at john@example.com for details"
        result = manager.anonymize(text)

        assert "Hello" in result.anonymized_data
        assert "contact me at" in result.anonymized_data
        assert "for details" in result.anonymized_data

    def test_validate_data_privacy_compliant(self, manager: PrivacyManager):
        """Test privacy validation for compliant data"""
        text = "This is clean data without PII"
        validation = manager.validate_data_privacy(text)

        assert validation["compliant"] is True
        assert validation["pii_detected"] is False
        assert validation["detection_count"] == 0

    def test_validate_data_privacy_non_compliant(self, manager: PrivacyManager):
        """Test privacy validation for non-compliant data"""
        text = "Contact: john@example.com"
        validation = manager.validate_data_privacy(text)

        assert validation["compliant"] is False
        assert validation["pii_detected"] is True
        assert validation["detection_count"] > 0
        assert len(validation["detections"]) > 0
        assert "recommendation" in validation

    def test_should_retain_data_within_retention(self, manager: PrivacyManager):
        """Test data retention check within retention period"""
        created_at = datetime.now(UTC) - timedelta(days=30)
        assert manager.should_retain_data(created_at) is True

    def test_should_retain_data_outside_retention(self, manager: PrivacyManager):
        """Test data retention check outside retention period"""
        created_at = datetime.now(UTC) - timedelta(days=100)
        assert manager.should_retain_data(created_at) is False

    def test_should_retain_data_on_boundary(self, manager: PrivacyManager):
        """Test data retention on boundary"""
        created_at = datetime.now(UTC) - timedelta(days=90)
        assert manager.should_retain_data(created_at) is True

    def test_get_privacy_stats(self, manager: PrivacyManager):
        """Test getting privacy statistics"""
        manager.scan_for_pii("Contact: john@example.com")
        manager.anonymize("Email: jane@example.com")

        stats = manager.get_privacy_stats()

        assert stats["texts_scanned"] >= 1
        assert stats["pii_detected"] >= 1
        assert stats["texts_anonymized"] >= 1
        assert "pii_detection_rate" in stats
        assert "anonymization_rate" in stats

    def test_email_masking(self, manager: PrivacyManager):
        """Test email masking"""
        masked = manager._mask_value("john.doe@example.com", PIIType.EMAIL)
        assert masked.startswith("j")
        assert "@example.com" in masked
        assert "*" in masked

    def test_phone_masking(self, manager: PrivacyManager):
        """Test phone number masking"""
        masked = manager._mask_value("123-456-7890", PIIType.PHONE)
        assert masked.endswith("7890")
        assert "***" in masked

    def test_credit_card_masking(self, manager: PrivacyManager):
        """Test credit card masking"""
        masked = manager._mask_value("1234-5678-9012-3456", PIIType.CREDIT_CARD)
        assert masked.endswith("3456")
        assert "****" in masked


class TestPrivacyIntegration:
    """Integration tests for privacy"""

    def test_end_to_end_anonymization(self):
        """Test end-to-end anonymization workflow"""
        manager = PrivacyManager()

        # Original data with PII
        original = "User john@example.com called from 123-456-7890"

        # Validate (should fail)
        validation = manager.validate_data_privacy(original)
        assert validation["compliant"] is False

        # Anonymize
        result = manager.anonymize(original)
        assert "john@example.com" not in result.anonymized_data
        assert "123-456-7890" not in result.anonymized_data

        # Validate anonymized data (should pass)
        validation_after = manager.validate_data_privacy(result.anonymized_data)
        assert validation_after["compliant"] is True

    def test_privacy_config_disabled_detection(self):
        """Test privacy manager with detection disabled"""
        config = PrivacyConfig(enable_pii_detection=False)
        manager = PrivacyManager(config)

        text = "Contact: john@example.com"
        detections = manager.scan_for_pii(text)

        assert len(detections) == 0

    def test_different_anonymization_strategies(self):
        """Test all anonymization strategies"""
        manager = PrivacyManager()
        text = "Email: john@example.com"

        strategies = [
            AnonymizationStrategy.REDACT,
            AnonymizationStrategy.HASH,
            AnonymizationStrategy.TOKENIZE,
            AnonymizationStrategy.MASK,
        ]

        results = [manager.anonymize(text, strategy) for strategy in strategies]

        # All should remove original PII
        for result in results:
            assert "john@example.com" not in result.anonymized_data

        # All should produce different outputs
        anonymized_texts = [r.anonymized_data for r in results]
        assert len(set(anonymized_texts)) == len(strategies)
