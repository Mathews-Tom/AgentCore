"""Unit tests for ModelSelector runtime model selection.

Tests cover:
- Tier-based model selection (FAST, BALANCED, PREMIUM)
- Complexity-based model selection (low, medium, high)
- Provider preference ordering
- Fallback logic when preferred model not available
- Configuration validation (all tiers mapped)
- Error handling (no models available, invalid complexity)
- Selection rationale logging

Target: 100% code coverage per CLAUDE.md requirements.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agentcore.a2a_protocol.models.llm import ModelTier
from agentcore.a2a_protocol.services.model_selector import (
    COMPLEXITY_TIER_MAP,
    TIER_MODEL_MAP,
    ModelSelector)


class TestModelSelector:
    """Test suite for ModelSelector class."""

    def test_init_with_no_preference(self) -> None:
        """Test ModelSelector initialization with no provider preference."""
        selector = ModelSelector()

        assert selector.provider_preference is None
        assert selector.logger.name == "agentcore.a2a_protocol.services.model_selector"

    def test_init_with_provider_preference(self) -> None:
        """Test ModelSelector initialization with provider preference."""
        preference = ["openai", "anthropic", "gemini"]
        selector = ModelSelector(provider_preference=preference)

        assert selector.provider_preference == preference

    def test_select_model_fast_tier(self) -> None:
        """Test selecting model for FAST tier."""
        selector = ModelSelector()

        # Mock ALLOWED_MODELS to include FAST tier models
        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            mock_settings.ALLOWED_MODELS = [
                "gpt-5-mini",
                "claude-haiku-4-5-20251001",
                "gemini-2.5-flash-lite",
            ]

            model = selector.select_model(ModelTier.FAST)

            # Should return first allowed model from TIER_MODEL_MAP[FAST]
            assert model == "gpt-5-mini"

    def test_select_model_balanced_tier(self) -> None:
        """Test selecting model for BALANCED tier."""
        selector = ModelSelector()

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            mock_settings.ALLOWED_MODELS = [
                "gpt-5.1",
                "claude-haiku-4-5-20251001",
                "gemini-2.5-flash",
            ]

            model = selector.select_model(ModelTier.BALANCED)

            assert model == "gpt-5.1"

    def test_select_model_premium_tier(self) -> None:
        """Test selecting model for PREMIUM tier."""
        selector = ModelSelector()

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            mock_settings.ALLOWED_MODELS = [
                "gpt-5-pro",
                "claude-opus-4-1-20250805",
                "gemini-2.5-pro",
            ]

            model = selector.select_model(ModelTier.PREMIUM)

            # Any premium tier model is valid
            assert model in TIER_MODEL_MAP[ModelTier.PREMIUM]

    def test_select_model_with_provider_preference_openai(self) -> None:
        """Test model selection with OpenAI provider preference."""
        selector = ModelSelector(provider_preference=["openai", "anthropic", "gemini"])

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            # Make OpenAI model allowed
            mock_settings.ALLOWED_MODELS = [
                "gpt-5-mini",
                "claude-haiku-4-5-20251001",
            ]

            model = selector.select_model(ModelTier.FAST)

            # Should prefer OpenAI model (gpt-5-mini)
            assert model == "gpt-5-mini"

    def test_select_model_with_provider_preference_anthropic(self) -> None:
        """Test model selection with Anthropic provider preference."""
        selector = ModelSelector(provider_preference=["anthropic", "openai", "gemini"])

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            # Make both OpenAI and Anthropic models allowed
            mock_settings.ALLOWED_MODELS = [
                "gpt-5-mini",
                "claude-haiku-4-5-20251001",
            ]

            model = selector.select_model(ModelTier.FAST)

            # Should prefer Anthropic model (claude-haiku-4-5-20251001)
            assert model == "claude-haiku-4-5-20251001"

    def test_select_model_fallback_when_preferred_not_available(self) -> None:
        """Test fallback to non-preferred model when preferred provider not available."""
        # Prefer OpenAI but only Anthropic model is allowed
        selector = ModelSelector(provider_preference=["openai"])

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            # Only Anthropic model allowed
            mock_settings.ALLOWED_MODELS = ["claude-haiku-4-5-20251001"]

            model = selector.select_model(ModelTier.FAST)

            # Should fallback to Anthropic model
            assert model == "claude-haiku-4-5-20251001"

    def test_select_model_no_allowed_models_raises_error(self) -> None:
        """Test error when no models allowed for tier."""
        selector = ModelSelector()

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            # No models allowed
            mock_settings.ALLOWED_MODELS = []

            with pytest.raises(ValueError) as exc_info:
                selector.select_model(ModelTier.FAST)

            assert "No allowed models for tier 'fast'" in str(exc_info.value)

    def test_select_model_by_complexity_low(self) -> None:
        """Test selecting model by 'low' complexity."""
        selector = ModelSelector()

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            mock_settings.ALLOWED_MODELS = ["gpt-5-mini"]

            model = selector.select_model_by_complexity("low")

            # low → FAST → gpt-5-mini
            assert model == "gpt-5-mini"

    def test_select_model_by_complexity_medium(self) -> None:
        """Test selecting model by 'medium' complexity."""
        selector = ModelSelector()

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            mock_settings.ALLOWED_MODELS = ["gpt-5.1"]

            model = selector.select_model_by_complexity("medium")

            # medium → BALANCED → gpt-5.1
            assert model == "gpt-5.1"

    def test_select_model_by_complexity_high(self) -> None:
        """Test selecting model by 'high' complexity."""
        selector = ModelSelector()

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            # Allow all premium tier models
            mock_settings.ALLOWED_MODELS = [
                "gpt-5-pro",
                "claude-opus-4-1-20250805",
                "gemini-2.5-pro",
            ]

            model = selector.select_model_by_complexity("high")

            # high → PREMIUM → any premium model
            assert model in TIER_MODEL_MAP[ModelTier.PREMIUM]

    def test_select_model_by_complexity_case_insensitive(self) -> None:
        """Test complexity selection is case-insensitive."""
        selector = ModelSelector()

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            mock_settings.ALLOWED_MODELS = ["gpt-5-mini"]

            # Test uppercase
            model_upper = selector.select_model_by_complexity("LOW")
            assert model_upper == "gpt-5-mini"

            # Test mixed case
            model_mixed = selector.select_model_by_complexity("Low")
            assert model_mixed == "gpt-5-mini"

    def test_select_model_by_complexity_invalid_raises_error(self) -> None:
        """Test error when invalid complexity provided."""
        selector = ModelSelector()

        with pytest.raises(ValueError) as exc_info:
            selector.select_model_by_complexity("invalid")

        assert "Invalid complexity level 'invalid'" in str(exc_info.value)
        assert "Valid levels: ['low', 'medium', 'high']" in str(exc_info.value)

    def test_validate_configuration_all_tiers_valid(self) -> None:
        """Test configuration validation when all tiers have allowed models."""
        selector = ModelSelector()

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            # All tier models allowed - one from each tier
            mock_settings.ALLOWED_MODELS = [
                "gpt-5-mini",  # FAST
                "gpt-5.1",  # BALANCED
                "gpt-5-pro",  # PREMIUM
            ]

            is_valid = selector.validate_configuration()

            assert is_valid is True

    def test_validate_configuration_missing_tier(self) -> None:
        """Test configuration validation when tier has no allowed models."""
        selector = ModelSelector()

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            # Only FAST and BALANCED tiers have models, PREMIUM missing
            mock_settings.ALLOWED_MODELS = [
                "gpt-5-mini",  # FAST
                "gpt-5.1",  # BALANCED
                # PREMIUM tier has no allowed models
            ]

            is_valid = selector.validate_configuration()

            assert is_valid is False

    def test_validate_configuration_no_allowed_models(self) -> None:
        """Test configuration validation when no models allowed."""
        selector = ModelSelector()

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            mock_settings.ALLOWED_MODELS = []

            is_valid = selector.validate_configuration()

            assert is_valid is False

    def test_filter_by_preference_with_single_preference(self) -> None:
        """Test _filter_by_preference with single provider preference."""
        selector = ModelSelector(provider_preference=["openai"])

        models = ["claude-haiku-4-5-20251001", "gpt-5-mini", "gemini-2.5-flash-lite"]
        sorted_models = selector._filter_by_preference(models)

        # OpenAI model should be first
        assert sorted_models[0] == "gpt-5-mini"

    def test_filter_by_preference_with_multiple_preferences(self) -> None:
        """Test _filter_by_preference with multiple provider preferences."""
        selector = ModelSelector(provider_preference=["anthropic", "gemini", "openai"])

        models = ["gpt-5-mini", "gemini-2.5-flash-lite", "claude-haiku-4-5-20251001"]
        sorted_models = selector._filter_by_preference(models)

        # Should be sorted: anthropic, gemini, openai
        assert sorted_models[0] == "claude-haiku-4-5-20251001"  # anthropic
        assert sorted_models[1] == "gemini-2.5-flash-lite"  # gemini
        assert sorted_models[2] == "gpt-5-mini"  # openai

    def test_filter_by_preference_no_preference(self) -> None:
        """Test _filter_by_preference with no provider preference."""
        selector = ModelSelector(provider_preference=None)

        models = ["claude-haiku-4-5-20251001", "gpt-5-mini", "gemini-2.5-flash-lite"]
        sorted_models = selector._filter_by_preference(models)

        # Should return models unchanged
        assert sorted_models == models

    def test_filter_by_preference_empty_preference(self) -> None:
        """Test _filter_by_preference with empty provider preference list."""
        selector = ModelSelector(provider_preference=[])

        models = ["claude-haiku-4-5-20251001", "gpt-5-mini", "gemini-2.5-flash-lite"]
        sorted_models = selector._filter_by_preference(models)

        # Should return models unchanged
        assert sorted_models == models

    def test_complexity_tier_map_completeness(self) -> None:
        """Test COMPLEXITY_TIER_MAP has all expected complexity levels."""
        assert "low" in COMPLEXITY_TIER_MAP
        assert "medium" in COMPLEXITY_TIER_MAP
        assert "high" in COMPLEXITY_TIER_MAP

        assert COMPLEXITY_TIER_MAP["low"] == ModelTier.FAST
        assert COMPLEXITY_TIER_MAP["medium"] == ModelTier.BALANCED
        assert COMPLEXITY_TIER_MAP["high"] == ModelTier.PREMIUM

    def test_tier_model_map_completeness(self) -> None:
        """Test TIER_MODEL_MAP has all ModelTier values."""
        assert ModelTier.FAST in TIER_MODEL_MAP
        assert ModelTier.BALANCED in TIER_MODEL_MAP
        assert ModelTier.PREMIUM in TIER_MODEL_MAP

        # Each tier should have at least one model
        assert len(TIER_MODEL_MAP[ModelTier.FAST]) > 0
        assert len(TIER_MODEL_MAP[ModelTier.BALANCED]) > 0
        assert len(TIER_MODEL_MAP[ModelTier.PREMIUM]) > 0

    def test_tier_model_map_fast_tier_models(self) -> None:
        """Test FAST tier contains expected fast models."""
        fast_models = TIER_MODEL_MAP[ModelTier.FAST]

        # Should include small models
        assert "gpt-5-mini" in fast_models
        assert "claude-haiku-4-5-20251001" in fast_models
        assert "gemini-2.5-flash-lite" in fast_models

    def test_tier_model_map_balanced_tier_models(self) -> None:
        """Test BALANCED tier contains expected balanced models."""
        balanced_models = TIER_MODEL_MAP[ModelTier.BALANCED]

        # Should include medium models
        assert "gpt-5.1" in balanced_models
        assert "claude-sonnet-4-5-20250929" in balanced_models
        assert "gemini-2.5-flash" in balanced_models

    def test_tier_model_map_premium_tier_models(self) -> None:
        """Test PREMIUM tier contains expected premium models."""
        premium_models = TIER_MODEL_MAP[ModelTier.PREMIUM]

        # Should include large models
        assert "gpt-5-pro" in premium_models
        assert "claude-opus-4-1-20250805" in premium_models
        assert "gemini-2.5-pro" in premium_models

    def test_select_model_logs_selection_rationale(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that select_model logs selection rationale."""
        selector = ModelSelector()

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            mock_settings.ALLOWED_MODELS = ["gpt-5-mini"]

            selector.select_model(ModelTier.FAST)

            # Check that selection was logged
            assert any("Model selected" in record.message for record in caplog.records)

    def test_select_model_logs_error_when_no_models(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that select_model logs error when no models available."""
        selector = ModelSelector()

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            mock_settings.ALLOWED_MODELS = []

            with pytest.raises(ValueError):
                selector.select_model(ModelTier.FAST)

            # Check that error was logged
            assert any("No allowed models for tier" in record.message for record in caplog.records)

    def test_validate_configuration_logs_warnings(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that validate_configuration logs warnings for missing tiers."""
        selector = ModelSelector()

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            # Only FAST tier has models
            mock_settings.ALLOWED_MODELS = ["gpt-5-mini"]

            selector.validate_configuration()

            # Check that warnings were logged for BALANCED and PREMIUM tiers
            warning_messages = [
                record.message for record in caplog.records if record.levelname == "WARNING"
            ]
            assert len(warning_messages) >= 2  # At least BALANCED and PREMIUM

    def test_select_model_with_gpt_5_mini_fallback(self) -> None:
        """Test FAST tier can select gpt-5-mini if gpt-5-mini not available."""
        selector = ModelSelector()

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            # Only gpt-5-mini allowed (gpt-5-mini not in list)
            mock_settings.ALLOWED_MODELS = ["gpt-5-mini"]

            model = selector.select_model(ModelTier.FAST)

            # Should select gpt-5-mini as fallback
            assert model == "gpt-5-mini"

    def test_select_model_respects_allowed_models_order(self) -> None:
        """Test that select_model respects TIER_MODEL_MAP priority order."""
        selector = ModelSelector()

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            # All FAST tier models allowed
            mock_settings.ALLOWED_MODELS = [
                "gemini-2.5-flash-lite",
                "claude-haiku-4-5-20251001",
                "gpt-5-mini",
                "gpt-5-mini",
            ]

            model = selector.select_model(ModelTier.FAST)

            # Should select first model from TIER_MODEL_MAP[FAST]
            # which is gpt-5-mini per configuration
            assert model == "gpt-5-mini"

    def test_provider_preference_with_partial_match(self) -> None:
        """Test provider preference when only some preferred providers available."""
        # Prefer openai > anthropic > gemini
        selector = ModelSelector(provider_preference=["openai", "anthropic", "gemini"])

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            # Only Anthropic and Gemini models allowed (no OpenAI)
            mock_settings.ALLOWED_MODELS = [
                "claude-haiku-4-5-20251001",
                "gemini-2.5-flash-lite",
            ]

            model = selector.select_model(ModelTier.FAST)

            # Should select Anthropic (next in preference after OpenAI)
            assert model == "claude-haiku-4-5-20251001"

    def test_filter_by_preference_logs_debug_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that _filter_by_preference logs debug information."""
        import logging

        caplog.set_level(logging.DEBUG)

        selector = ModelSelector(provider_preference=["openai", "anthropic"])

        models = ["claude-haiku-4-5-20251001", "gpt-5-mini"]
        selector._filter_by_preference(models)

        # Check that debug log was created
        debug_messages = [
            record.message for record in caplog.records if record.levelname == "DEBUG"
        ]
        assert any("Models filtered by provider preference" in msg for msg in debug_messages)

    def test_select_model_by_complexity_logs_debug_mapping(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that select_model_by_complexity logs complexity-to-tier mapping."""
        import logging

        caplog.set_level(logging.DEBUG)

        selector = ModelSelector()

        with patch("agentcore.a2a_protocol.services.model_selector.settings") as mock_settings:
            mock_settings.ALLOWED_MODELS = ["gpt-5-mini"]

            selector.select_model_by_complexity("low")

            # Check that debug log was created for complexity mapping
            debug_messages = [
                record.message for record in caplog.records if record.levelname == "DEBUG"
            ]
            assert any("Complexity mapped to tier" in msg for msg in debug_messages)

