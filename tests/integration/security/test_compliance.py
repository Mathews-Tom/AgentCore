"""Unit tests for compliance manager.

Tests PII detection, data residency, and compliance checks.
"""

from __future__ import annotations

import pytest

from agentcore.integration.security.compliance import (
    ComplianceFramework,
    ComplianceManager,
    DataClassification,
    DataRegion,
    DataResidencyConfig,
    PIIPattern,
)


class TestComplianceManager:
    """Test compliance manager functionality."""

    def test_initialization(self) -> None:
        """Test compliance manager initialization."""
        manager = ComplianceManager()
        assert manager is not None

    def test_initialization_with_residency_config(self) -> None:
        """Test initialization with data residency configuration."""
        config = DataResidencyConfig(
            allowed_regions=[DataRegion.US_EAST, DataRegion.US_WEST],
            primary_region=DataRegion.US_EAST,
            compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.SOC2],
        )

        manager = ComplianceManager(residency_config=config)
        assert manager is not None

    def test_detect_pii_email(self) -> None:
        """Test detecting email addresses."""
        manager = ComplianceManager()

        text = "Contact john.doe@example.com for more info"
        matches = manager.detect_pii(text)

        assert "email" in matches
        assert "john.doe@example.com" in matches["email"]

    def test_detect_pii_ssn(self) -> None:
        """Test detecting Social Security Numbers."""
        manager = ComplianceManager()

        text = "SSN: 123-45-6789"
        matches = manager.detect_pii(text)

        assert "ssn" in matches
        assert "123-45-6789" in matches["ssn"]

    def test_detect_pii_phone(self) -> None:
        """Test detecting phone numbers."""
        manager = ComplianceManager()

        text = "Call me at 555-123-4567 or 5551234567"
        matches = manager.detect_pii(text)

        assert "phone_us" in matches
        assert len(matches["phone_us"]) >= 1

    def test_detect_pii_credit_card(self) -> None:
        """Test detecting credit card numbers."""
        manager = ComplianceManager()

        text = "Card: 4532-1234-5678-9012"
        matches = manager.detect_pii(text)

        assert "credit_card" in matches
        assert "4532-1234-5678-9012" in matches["credit_card"]

    def test_detect_pii_ip_address(self) -> None:
        """Test detecting IP addresses."""
        manager = ComplianceManager()

        text = "Server at 192.168.1.1"
        matches = manager.detect_pii(text)

        assert "ip_address" in matches
        assert "192.168.1.1" in matches["ip_address"]

    def test_detect_pii_multiple_types(self) -> None:
        """Test detecting multiple PII types."""
        manager = ComplianceManager()

        text = "Contact john@example.com at 555-1234 or use SSN 123-45-6789"
        matches = manager.detect_pii(text)

        assert "email" in matches
        assert "phone_us" in matches
        assert "ssn" in matches
        assert len(matches) == 3

    def test_detect_pii_no_pii(self) -> None:
        """Test text with no PII."""
        manager = ComplianceManager()

        text = "This is a regular text without any sensitive data"
        matches = manager.detect_pii(text)

        assert len(matches) == 0

    def test_redact_pii_email(self) -> None:
        """Test redacting email addresses."""
        manager = ComplianceManager()

        text = "Contact john.doe@example.com for info"
        redacted = manager.redact_pii(text)

        assert "john.doe@example.com" not in redacted
        assert "***@***.***" in redacted

    def test_redact_pii_ssn(self) -> None:
        """Test redacting SSN."""
        manager = ComplianceManager()

        text = "SSN: 123-45-6789"
        redacted = manager.redact_pii(text)

        assert "123-45-6789" not in redacted
        assert "***-**-****" in redacted

    def test_redact_pii_all_patterns(self) -> None:
        """Test redacting all PII patterns."""
        manager = ComplianceManager()

        text = "Email: user@example.com Phone: 555-1234 SSN: 123-45-6789"
        redacted = manager.redact_pii(text)

        # All PII should be redacted
        assert "user@example.com" not in redacted
        assert "555-1234" not in redacted
        assert "123-45-6789" not in redacted

    def test_redact_pii_specific_patterns(self) -> None:
        """Test redacting only specific PII patterns."""
        manager = ComplianceManager()

        text = "Email: user@example.com SSN: 123-45-6789"
        redacted = manager.redact_pii(text, patterns=["email"])

        # Only email should be redacted
        assert "user@example.com" not in redacted
        assert "123-45-6789" in redacted

    def test_classify_data_pii(self) -> None:
        """Test classifying data containing PII."""
        manager = ComplianceManager()

        data = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": 30,
        }

        classification = manager.classify_data(data)
        assert classification == DataClassification.PII

    def test_classify_data_phi(self) -> None:
        """Test classifying data as PHI (Protected Health Information)."""
        manager = ComplianceManager()

        data = {
            "patient": "John Doe",
            "email": "john@example.com",
            "medical_condition": "diabetes",
        }

        classification = manager.classify_data(data)
        assert classification == DataClassification.PHI

    def test_classify_data_restricted(self) -> None:
        """Test classifying data as restricted."""
        manager = ComplianceManager()

        data = {
            "api_key": "sk-1234567890",
            "secret": "my-secret-value",
        }

        classification = manager.classify_data(data)
        assert classification == DataClassification.RESTRICTED

    def test_classify_data_confidential(self) -> None:
        """Test classifying data as confidential."""
        manager = ComplianceManager()

        data = {
            "internal_memo": "This is confidential information",
            "private_notes": "Not for public release",
        }

        classification = manager.classify_data(data)
        assert classification == DataClassification.CONFIDENTIAL

    def test_classify_data_internal(self) -> None:
        """Test classifying data as internal."""
        manager = ComplianceManager()

        data = {
            "status": "active",
            "count": 42,
        }

        classification = manager.classify_data(data)
        assert classification == DataClassification.INTERNAL

    def test_validate_region_no_config(self) -> None:
        """Test region validation with no configuration."""
        manager = ComplianceManager()

        # Without config, all regions should be allowed
        assert manager.validate_region(DataRegion.US_EAST)
        assert manager.validate_region(DataRegion.EU_WEST)

    def test_validate_region_allowed(self) -> None:
        """Test validating allowed region."""
        config = DataResidencyConfig(
            allowed_regions=[DataRegion.US_EAST, DataRegion.US_WEST],
            primary_region=DataRegion.US_EAST,
        )
        manager = ComplianceManager(residency_config=config)

        assert manager.validate_region(DataRegion.US_EAST)
        assert manager.validate_region(DataRegion.US_WEST)

    def test_validate_region_not_allowed_strict(self) -> None:
        """Test validating non-allowed region with strict enforcement."""
        config = DataResidencyConfig(
            allowed_regions=[DataRegion.US_EAST],
            primary_region=DataRegion.US_EAST,
            enforce_strict=True,
        )
        manager = ComplianceManager(residency_config=config)

        with pytest.raises(ValueError, match="not allowed"):
            manager.validate_region(DataRegion.EU_WEST)

    def test_validate_region_not_allowed_non_strict(self) -> None:
        """Test validating non-allowed region without strict enforcement."""
        config = DataResidencyConfig(
            allowed_regions=[DataRegion.US_EAST],
            primary_region=DataRegion.US_EAST,
            enforce_strict=False,
        )
        manager = ComplianceManager(residency_config=config)

        # Should return False but not raise
        assert not manager.validate_region(DataRegion.EU_WEST)

    def test_validate_region_global_always_allowed(self) -> None:
        """Test that GLOBAL region is always allowed."""
        config = DataResidencyConfig(
            allowed_regions=[DataRegion.US_EAST],
            primary_region=DataRegion.US_EAST,
            enforce_strict=True,
        )
        manager = ComplianceManager(residency_config=config)

        assert manager.validate_region(DataRegion.GLOBAL)

    def test_get_allowed_regions_no_config(self) -> None:
        """Test getting allowed regions with no configuration."""
        manager = ComplianceManager()

        regions = manager.get_allowed_regions()
        assert len(regions) > 0
        assert DataRegion.US_EAST in regions
        assert DataRegion.EU_WEST in regions

    def test_get_allowed_regions_with_config(self) -> None:
        """Test getting allowed regions with configuration."""
        config = DataResidencyConfig(
            allowed_regions=[DataRegion.US_EAST, DataRegion.US_WEST],
            primary_region=DataRegion.US_EAST,
        )
        manager = ComplianceManager(residency_config=config)

        regions = manager.get_allowed_regions()
        assert len(regions) == 2
        assert DataRegion.US_EAST in regions
        assert DataRegion.US_WEST in regions
        assert DataRegion.EU_WEST not in regions

    def test_check_compliance_gdpr_no_violations(self) -> None:
        """Test GDPR compliance check with no violations."""
        manager = ComplianceManager()

        data = {
            "user_id": "user-123",
            "status": "active",
        }

        result = manager.check_compliance(ComplianceFramework.GDPR, data)
        assert result["framework"] == "gdpr"
        assert result["compliant"]
        assert len(result["violations"]) == 0

    def test_check_compliance_gdpr_pii_without_consent(self) -> None:
        """Test GDPR compliance check with PII but no consent."""
        manager = ComplianceManager()

        data = {
            "email": "user@example.com",
            "name": "John Doe",
        }

        result = manager.check_compliance(ComplianceFramework.GDPR, data)
        assert result["framework"] == "gdpr"
        assert not result["compliant"]
        assert result["pii_detected"]
        assert len(result["violations"]) > 0

    def test_check_compliance_gdpr_no_user_id_for_erasure(self) -> None:
        """Test GDPR compliance check for right to erasure."""
        manager = ComplianceManager()

        data = {
            "email": "user@example.com",
        }

        result = manager.check_compliance(ComplianceFramework.GDPR, data)
        assert not result["compliant"]
        assert any("user_id" in v.lower() for v in result["violations"])

    def test_check_compliance_ccpa(self) -> None:
        """Test CCPA compliance check."""
        manager = ComplianceManager()

        data = {
            "email": "user@example.com",
        }

        result = manager.check_compliance(ComplianceFramework.CCPA, data)
        assert result["framework"] == "ccpa"
        assert result["pii_detected"]

    def test_check_compliance_hipaa(self) -> None:
        """Test HIPAA compliance check."""
        manager = ComplianceManager()

        data = {
            "patient": "John Doe",
            "email": "john@example.com",
            "medical_diagnosis": "diabetes",
        }

        result = manager.check_compliance(ComplianceFramework.HIPAA, data)
        assert result["framework"] == "hipaa"
        # Should detect PHI
        assert len(result["violations"]) > 0

    def test_check_compliance_pci_dss_credit_card(self) -> None:
        """Test PCI DSS compliance with credit card data."""
        manager = ComplianceManager()

        data = {
            "card_number": "4532-1234-5678-9012",
        }

        result = manager.check_compliance(ComplianceFramework.PCI_DSS, data)
        assert result["framework"] == "pci-dss"
        assert not result["compliant"]
        assert any("credit card" in v.lower() for v in result["violations"])

    def test_sanitize_logs_simple(self) -> None:
        """Test sanitizing log data."""
        manager = ComplianceManager()

        log_data = {
            "message": "User john@example.com logged in",
            "ip": "192.168.1.1",
        }

        sanitized = manager.sanitize_logs(log_data)
        assert "john@example.com" not in sanitized["message"]
        assert "***@***.***" in sanitized["message"]

    def test_sanitize_logs_nested(self) -> None:
        """Test sanitizing nested log data."""
        manager = ComplianceManager()

        log_data = {
            "user": {
                "email": "user@example.com",
                "phone": "555-1234",
            },
            "action": "login",
        }

        sanitized = manager.sanitize_logs(log_data)
        assert "user@example.com" not in str(sanitized)
        assert sanitized["action"] == "login"

    def test_sanitize_logs_with_lists(self) -> None:
        """Test sanitizing log data with lists."""
        manager = ComplianceManager()

        log_data = {
            "users": [
                {"email": "user1@example.com"},
                {"email": "user2@example.com"},
            ],
        }

        sanitized = manager.sanitize_logs(log_data)
        assert "user1@example.com" not in str(sanitized)
        assert "user2@example.com" not in str(sanitized)

    def test_add_pii_pattern(self) -> None:
        """Test adding custom PII pattern."""
        manager = ComplianceManager()

        # Add custom pattern for employee ID
        pattern = PIIPattern(
            name="employee_id",
            pattern=r"\bEMP-\d{6}\b",
            classification=DataClassification.INTERNAL,
            redaction_mask="EMP-******",
        )

        manager.add_pii_pattern(pattern)

        # Test detection
        text = "Employee EMP-123456 accessed the system"
        matches = manager.detect_pii(text)

        assert "employee_id" in matches
        assert "EMP-123456" in matches["employee_id"]

    def test_add_pii_pattern_redaction(self) -> None:
        """Test redaction with custom PII pattern."""
        manager = ComplianceManager()

        pattern = PIIPattern(
            name="custom_id",
            pattern=r"\bCUST-\d{4}\b",
            classification=DataClassification.PII,
            redaction_mask="CUST-****",
        )

        manager.add_pii_pattern(pattern)

        text = "Customer CUST-1234 made a purchase"
        redacted = manager.redact_pii(text)

        assert "CUST-1234" not in redacted
        assert "CUST-****" in redacted

    def test_generate_compliance_report_no_config(self) -> None:
        """Test generating compliance report without configuration."""
        manager = ComplianceManager()

        report = manager.generate_compliance_report()

        assert "data_residency" in report
        assert "pii_detection" in report
        assert not report["data_residency"]["configured"]
        assert report["pii_detection"]["pattern_count"] > 0

    def test_generate_compliance_report_with_config(self) -> None:
        """Test generating compliance report with configuration."""
        config = DataResidencyConfig(
            allowed_regions=[DataRegion.US_EAST, DataRegion.US_WEST],
            primary_region=DataRegion.US_EAST,
            compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.SOC2],
        )

        manager = ComplianceManager(residency_config=config)
        report = manager.generate_compliance_report()

        assert report["data_residency"]["configured"]
        assert len(report["data_residency"]["allowed_regions"]) == 2
        assert report["data_residency"]["primary_region"] == "us-east"
        assert "gdpr" in report["data_residency"]["compliance_frameworks"]
        assert "soc2" in report["data_residency"]["compliance_frameworks"]

    def test_generate_compliance_report_with_custom_patterns(self) -> None:
        """Test compliance report includes custom patterns."""
        custom_pattern = PIIPattern(
            name="custom",
            pattern=r"\bCUSTOM-\d+\b",
            classification=DataClassification.PII,
        )

        manager = ComplianceManager(custom_pii_patterns=[custom_pattern])
        report = manager.generate_compliance_report()

        # Should include default + custom patterns
        assert report["pii_detection"]["pattern_count"] > 5
        pattern_names = [p["name"] for p in report["pii_detection"]["patterns"]]
        assert "custom" in pattern_names

    def test_initialization_with_custom_patterns(self) -> None:
        """Test initialization with custom PII patterns."""
        custom_patterns = [
            PIIPattern(
                name="custom1",
                pattern=r"\bPAT1-\d+\b",
                classification=DataClassification.PII,
            ),
            PIIPattern(
                name="custom2",
                pattern=r"\bPAT2-\d+\b",
                classification=DataClassification.CONFIDENTIAL,
            ),
        ]

        manager = ComplianceManager(custom_pii_patterns=custom_patterns)

        # Should detect custom patterns
        text = "PAT1-123 and PAT2-456"
        matches = manager.detect_pii(text)

        assert "custom1" in matches
        assert "custom2" in matches
