"""Tests for reasoning configuration management."""

from __future__ import annotations

import pytest

from agentcore.a2a_protocol.config import Settings


class TestReasoningConfig:
    """Test suite for reasoning configuration settings."""

    def test_default_reasoning_values(self) -> None:
        """Test that default reasoning configuration values are set correctly."""
        settings = Settings()

        assert settings.REASONING_MAX_ITERATIONS == 5
        assert settings.REASONING_CHUNK_SIZE == 8192
        assert settings.REASONING_CARRYOVER_SIZE == 4096
        assert settings.REASONING_DEFAULT_TEMPERATURE == 0.7
        assert settings.REASONING_ENABLE_METRICS is True
        assert settings.REASONING_ENABLE_TRACING is True
        assert settings.REASONING_INPUT_SANITIZATION is True

    def test_reasoning_env_var_loading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading reasoning configuration from environment variables."""
        monkeypatch.setenv("REASONING_MAX_ITERATIONS", "10")
        monkeypatch.setenv("REASONING_CHUNK_SIZE", "16384")
        monkeypatch.setenv("REASONING_CARRYOVER_SIZE", "8192")
        monkeypatch.setenv("REASONING_DEFAULT_TEMPERATURE", "0.5")
        monkeypatch.setenv("REASONING_ENABLE_METRICS", "false")
        monkeypatch.setenv("REASONING_ENABLE_TRACING", "false")
        monkeypatch.setenv("REASONING_INPUT_SANITIZATION", "false")

        settings = Settings()

        assert settings.REASONING_MAX_ITERATIONS == 10
        assert settings.REASONING_CHUNK_SIZE == 16384
        assert settings.REASONING_CARRYOVER_SIZE == 8192
        assert settings.REASONING_DEFAULT_TEMPERATURE == 0.5
        assert settings.REASONING_ENABLE_METRICS is False
        assert settings.REASONING_ENABLE_TRACING is False
        assert settings.REASONING_INPUT_SANITIZATION is False

    def test_reasoning_max_iterations_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test max_iterations must be between 1-50."""
        # Test minimum boundary
        monkeypatch.setenv("REASONING_MAX_ITERATIONS", "1")
        settings = Settings()
        assert settings.REASONING_MAX_ITERATIONS == 1

        # Test maximum boundary
        monkeypatch.setenv("REASONING_MAX_ITERATIONS", "50")
        settings = Settings()
        assert settings.REASONING_MAX_ITERATIONS == 50

        # Test below minimum
        monkeypatch.setenv("REASONING_MAX_ITERATIONS", "0")
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            Settings()

        # Test above maximum
        monkeypatch.setenv("REASONING_MAX_ITERATIONS", "51")
        with pytest.raises(ValueError, match="less than or equal to 50"):
            Settings()

    def test_reasoning_chunk_size_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test chunk_size must be between 1024-32768."""
        # Test minimum boundary
        monkeypatch.setenv("REASONING_CHUNK_SIZE", "1024")
        settings = Settings()
        assert settings.REASONING_CHUNK_SIZE == 1024

        # Test maximum boundary
        monkeypatch.setenv("REASONING_CHUNK_SIZE", "32768")
        settings = Settings()
        assert settings.REASONING_CHUNK_SIZE == 32768

        # Test below minimum
        monkeypatch.setenv("REASONING_CHUNK_SIZE", "1023")
        with pytest.raises(ValueError, match="greater than or equal to 1024"):
            Settings()

        # Test above maximum
        monkeypatch.setenv("REASONING_CHUNK_SIZE", "32769")
        with pytest.raises(ValueError, match="less than or equal to 32768"):
            Settings()

    def test_reasoning_carryover_size_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test carryover_size must be between 512-16384."""
        # Test minimum boundary
        monkeypatch.setenv("REASONING_CARRYOVER_SIZE", "512")
        settings = Settings()
        assert settings.REASONING_CARRYOVER_SIZE == 512

        # Test maximum boundary
        monkeypatch.setenv("REASONING_CARRYOVER_SIZE", "16384")
        settings = Settings()
        assert settings.REASONING_CARRYOVER_SIZE == 16384

        # Test below minimum
        monkeypatch.setenv("REASONING_CARRYOVER_SIZE", "511")
        with pytest.raises(ValueError, match="greater than or equal to 512"):
            Settings()

        # Test above maximum
        monkeypatch.setenv("REASONING_CARRYOVER_SIZE", "16385")
        with pytest.raises(ValueError, match="less than or equal to 16384"):
            Settings()

    def test_reasoning_temperature_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test temperature must be between 0.0-2.0."""
        # Test minimum boundary
        monkeypatch.setenv("REASONING_DEFAULT_TEMPERATURE", "0.0")
        settings = Settings()
        assert settings.REASONING_DEFAULT_TEMPERATURE == 0.0

        # Test maximum boundary
        monkeypatch.setenv("REASONING_DEFAULT_TEMPERATURE", "2.0")
        settings = Settings()
        assert settings.REASONING_DEFAULT_TEMPERATURE == 2.0

        # Test below minimum
        monkeypatch.setenv("REASONING_DEFAULT_TEMPERATURE", "-0.1")
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            Settings()

        # Test above maximum
        monkeypatch.setenv("REASONING_DEFAULT_TEMPERATURE", "2.1")
        with pytest.raises(ValueError, match="less than or equal to 2"):
            Settings()

    def test_reasoning_config_independence(self) -> None:
        """Test that reasoning config is independent of other settings."""
        settings = Settings()

        # Verify reasoning settings exist alongside other settings
        assert hasattr(settings, "DEBUG")
        assert hasattr(settings, "DATABASE_URL")
        assert hasattr(settings, "REASONING_MAX_ITERATIONS")
        assert hasattr(settings, "REASONING_CHUNK_SIZE")

        # Verify other settings don't interfere
        assert settings.DEBUG in (True, False)
        assert isinstance(settings.REASONING_MAX_ITERATIONS, int)

    def test_reasoning_config_completeness(self) -> None:
        """Test all required reasoning configuration fields are present."""
        settings = Settings()

        required_fields = [
            "REASONING_MAX_ITERATIONS",
            "REASONING_CHUNK_SIZE",
            "REASONING_CARRYOVER_SIZE",
            "REASONING_DEFAULT_TEMPERATURE",
            "REASONING_ENABLE_METRICS",
            "REASONING_ENABLE_TRACING",
            "REASONING_INPUT_SANITIZATION",
        ]

        for field in required_fields:
            assert hasattr(
                settings, field
            ), f"Missing required reasoning config field: {field}"
            assert getattr(settings, field) is not None

    def test_reasoning_boolean_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test boolean reasoning configuration values."""
        # Test string "true"
        monkeypatch.setenv("REASONING_ENABLE_METRICS", "true")
        settings = Settings()
        assert settings.REASONING_ENABLE_METRICS is True

        # Test string "1"
        monkeypatch.setenv("REASONING_ENABLE_TRACING", "1")
        settings = Settings()
        assert settings.REASONING_ENABLE_TRACING is True

        # Test string "false"
        monkeypatch.setenv("REASONING_INPUT_SANITIZATION", "false")
        settings = Settings()
        assert settings.REASONING_INPUT_SANITIZATION is False

        # Test string "0"
        monkeypatch.setenv("REASONING_ENABLE_METRICS", "0")
        settings = Settings()
        assert settings.REASONING_ENABLE_METRICS is False
