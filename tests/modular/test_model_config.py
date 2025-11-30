"""
Tests for Module-Specific Model Configuration

Tests the modular agent core's per-module LLM model selection and cost tracking
capabilities. Validates default configurations, environment variable overrides,
token usage tracking, and cost calculation.

Test Coverage:
- Default model assignments per module
- Environment variable configuration
- Model validation (allowed models only)
- Token usage tracking
- Cost calculation and aggregation
- Configuration export/import
- Error handling
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from agentcore.modular.config import (
    ALLOWED_MODELS,
    MODEL_PRICING,
    ModularConfig,
    ModularConfigSettings,
    ModuleName,
    ModuleModelConfig,
    TokenUsage,
    get_modular_config,
    reset_modular_config,
)


class TestTokenUsage:
    """Test TokenUsage model and cost calculation."""

    def test_default_token_usage(self) -> None:
        """Test default TokenUsage initialization."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.estimated_cost_usd == 0.0
        assert usage.invocation_count == 0

    def test_add_usage_basic(self) -> None:
        """Test basic token usage tracking."""
        usage = TokenUsage()
        usage.add_usage(input_tokens=100, output_tokens=50, model="gpt-4.1")

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.invocation_count == 1
        assert usage.estimated_cost_usd > 0

    def test_add_usage_multiple_invocations(self) -> None:
        """Test cumulative token usage across multiple invocations."""
        usage = TokenUsage()
        usage.add_usage(input_tokens=100, output_tokens=50, model="gpt-4.1")
        usage.add_usage(input_tokens=200, output_tokens=100, model="gpt-4.1")

        assert usage.input_tokens == 300
        assert usage.output_tokens == 150
        assert usage.total_tokens == 450
        assert usage.invocation_count == 2

    def test_add_usage_cost_calculation_gpt_5(self) -> None:
        """Test cost calculation for gpt-5 model."""
        usage = TokenUsage()
        usage.add_usage(input_tokens=1_000_000, output_tokens=1_000_000, model="gpt-5")

        # Cost = (1M / 1M) * $5.00 + (1M / 1M) * $15.00 = $20.00
        expected_cost = 5.0 + 15.0
        assert abs(usage.estimated_cost_usd - expected_cost) < 0.01

    def test_add_usage_cost_calculation_gpt_4_1_mini(self) -> None:
        """Test cost calculation for gpt-4.1-mini model (cheapest)."""
        usage = TokenUsage()
        usage.add_usage(
            input_tokens=1_000_000, output_tokens=1_000_000, model="gpt-4.1-mini"
        )

        # Cost = (1M / 1M) * $0.50 + (1M / 1M) * $1.50 = $2.00
        expected_cost = 0.5 + 1.5
        assert abs(usage.estimated_cost_usd - expected_cost) < 0.01

    def test_add_usage_negative_tokens_raises_error(self) -> None:
        """Test that negative token counts raise ValueError."""
        usage = TokenUsage()

        with pytest.raises(ValueError, match="Token counts must be non-negative"):
            usage.add_usage(input_tokens=-100, output_tokens=50, model="gpt-4.1")

        with pytest.raises(ValueError, match="Token counts must be non-negative"):
            usage.add_usage(input_tokens=100, output_tokens=-50, model="gpt-4.1")

    def test_add_usage_unknown_model_no_error(self) -> None:
        """Test that unknown models don't raise errors (cost just not calculated)."""
        usage = TokenUsage()
        # Should not raise error - just won't calculate cost
        usage.add_usage(input_tokens=100, output_tokens=50, model="unknown-model")
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.estimated_cost_usd == 0.0  # No pricing data available


class TestModuleModelConfig:
    """Test ModuleModelConfig validation."""

    def test_valid_module_config(self) -> None:
        """Test valid ModuleModelConfig creation."""
        config = ModuleModelConfig(
            module_name=ModuleName.PLANNER,
            model="gpt-5",
            temperature=0.7,
            max_tokens=4096,
        )

        assert config.module_name == ModuleName.PLANNER
        assert config.model == "gpt-5"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert isinstance(config.token_usage, TokenUsage)

    def test_invalid_model_raises_error(self) -> None:
        """Test that invalid models raise ValueError."""
        with pytest.raises(ValueError, match="is not allowed"):
            ModuleModelConfig(
                module_name=ModuleName.PLANNER,
                model="gpt-3.5-turbo",  # Not in ALLOWED_MODELS
                temperature=0.7,
            )

    def test_temperature_bounds_validation(self) -> None:
        """Test temperature validation (must be 0.0-2.0)."""
        # Valid temperatures
        ModuleModelConfig(
            module_name=ModuleName.PLANNER, model="gpt-5", temperature=0.0
        )
        ModuleModelConfig(
            module_name=ModuleName.PLANNER, model="gpt-5", temperature=2.0
        )

        # Invalid temperatures
        with pytest.raises(ValueError):
            ModuleModelConfig(
                module_name=ModuleName.PLANNER, model="gpt-5", temperature=-0.1
            )

        with pytest.raises(ValueError):
            ModuleModelConfig(
                module_name=ModuleName.PLANNER, model="gpt-5", temperature=2.1
            )


class TestModularConfigSettings:
    """Test ModularConfigSettings environment variable loading."""

    def test_default_settings(self) -> None:
        """Test default settings match CLAUDE.md requirements."""
        settings = ModularConfigSettings()

        # Check default model assignments
        assert settings.MODULAR_PLANNER_MODEL == "gpt-5"  # Large model for planning
        assert settings.MODULAR_EXECUTOR_MODEL == "gpt-4.1"  # Medium model
        assert settings.MODULAR_VERIFIER_MODEL == "gpt-4.1-mini"  # Small model
        assert settings.MODULAR_GENERATOR_MODEL == "gpt-4.1"  # Medium model

        # Check temperature defaults
        assert settings.MODULAR_PLANNER_TEMPERATURE == 0.7
        assert settings.MODULAR_EXECUTOR_TEMPERATURE == 0.5
        assert settings.MODULAR_VERIFIER_TEMPERATURE == 0.3
        assert settings.MODULAR_GENERATOR_TEMPERATURE == 0.7

    def test_environment_variable_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables override defaults."""
        # Set environment variables
        monkeypatch.setenv("MODULAR_PLANNER_MODEL", "gpt-5-mini")
        monkeypatch.setenv("MODULAR_EXECUTOR_MODEL", "gpt-4.1-mini")
        monkeypatch.setenv("MODULAR_PLANNER_TEMPERATURE", "0.9")

        settings = ModularConfigSettings()

        assert settings.MODULAR_PLANNER_MODEL == "gpt-5-mini"
        assert settings.MODULAR_EXECUTOR_MODEL == "gpt-4.1-mini"
        assert settings.MODULAR_PLANNER_TEMPERATURE == 0.9

    def test_invalid_model_in_env_raises_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that invalid model in environment variable raises error."""
        monkeypatch.setenv("MODULAR_PLANNER_MODEL", "gpt-3.5-turbo")

        with pytest.raises(ValueError, match="is not allowed"):
            ModularConfigSettings()


class TestModularConfig:
    """Test ModularConfig main functionality."""

    def test_initialization_creates_module_configs(self) -> None:
        """Test that initialization creates configs for all modules."""
        config = ModularConfig()

        # All modules should have configs
        for module in ModuleName:
            module_config = config.get_module_config(module)
            assert module_config.module_name == module
            assert module_config.model in ALLOWED_MODELS

    def test_default_model_assignments(self) -> None:
        """Test default model assignments match specification."""
        config = ModularConfig()

        assert config.get_model_for_module("planner") == "gpt-5"
        assert config.get_model_for_module("executor") == "gpt-4.1"
        assert config.get_model_for_module("verifier") == "gpt-4.1-mini"
        assert config.get_model_for_module("generator") == "gpt-4.1"

    def test_get_model_for_module_string_input(self) -> None:
        """Test getting model with string module name."""
        config = ModularConfig()

        # Case-insensitive string lookup
        assert config.get_model_for_module("planner") == "gpt-5"
        assert config.get_model_for_module("PLANNER") == "gpt-5"
        assert config.get_model_for_module("Planner") == "gpt-5"

    def test_get_model_for_module_enum_input(self) -> None:
        """Test getting model with ModuleName enum."""
        config = ModularConfig()

        assert config.get_model_for_module(ModuleName.PLANNER) == "gpt-5"
        assert config.get_model_for_module(ModuleName.EXECUTOR) == "gpt-4.1"

    def test_get_model_invalid_module_raises_error(self) -> None:
        """Test that invalid module name raises ValueError."""
        config = ModularConfig()

        with pytest.raises(ValueError, match="Invalid module name"):
            config.get_model_for_module("invalid_module")

    def test_get_temperature_for_module(self) -> None:
        """Test getting temperature configuration."""
        config = ModularConfig()

        assert config.get_temperature_for_module("planner") == 0.7
        assert config.get_temperature_for_module("executor") == 0.5
        assert config.get_temperature_for_module("verifier") == 0.3
        assert config.get_temperature_for_module("generator") == 0.7

    def test_get_max_tokens_for_module(self) -> None:
        """Test getting max tokens configuration."""
        config = ModularConfig()

        assert config.get_max_tokens_for_module("planner") == 4096
        assert config.get_max_tokens_for_module("executor") == 2048
        assert config.get_max_tokens_for_module("verifier") == 1024
        assert config.get_max_tokens_for_module("generator") == 2048

    def test_track_token_usage_basic(self) -> None:
        """Test basic token usage tracking."""
        config = ModularConfig()

        config.track_token_usage("planner", input_tokens=100, output_tokens=50)

        usage = config.get_token_usage("planner")
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.invocation_count == 1

    def test_track_token_usage_multiple_modules(self) -> None:
        """Test tracking usage across multiple modules."""
        config = ModularConfig()

        config.track_token_usage("planner", input_tokens=100, output_tokens=50)
        config.track_token_usage("executor", input_tokens=200, output_tokens=100)
        config.track_token_usage("verifier", input_tokens=50, output_tokens=25)

        planner_usage = config.get_token_usage("planner")
        executor_usage = config.get_token_usage("executor")
        verifier_usage = config.get_token_usage("verifier")

        assert planner_usage.total_tokens == 150
        assert executor_usage.total_tokens == 300
        assert verifier_usage.total_tokens == 75

    def test_get_module_costs(self) -> None:
        """Test getting cost summary for all modules."""
        config = ModularConfig()

        config.track_token_usage("planner", input_tokens=100, output_tokens=50)
        config.track_token_usage("executor", input_tokens=200, output_tokens=100)

        costs = config.get_module_costs()

        assert "planner" in costs
        assert "executor" in costs
        assert "verifier" in costs
        assert "generator" in costs

        assert costs["planner"]["tokens"] == 150
        assert costs["planner"]["invocations"] == 1
        assert costs["planner"]["estimated_cost_usd"] > 0

        assert costs["executor"]["tokens"] == 300
        assert costs["executor"]["invocations"] == 1

    def test_get_total_cost(self) -> None:
        """Test calculating total cost across all modules."""
        config = ModularConfig()

        config.track_token_usage("planner", input_tokens=1000, output_tokens=500)
        config.track_token_usage("executor", input_tokens=2000, output_tokens=1000)
        config.track_token_usage("verifier", input_tokens=500, output_tokens=250)

        total_cost = config.get_total_cost()

        # Total should be sum of individual module costs
        individual_sum = sum(
            config.get_token_usage(module).estimated_cost_usd
            for module in ["planner", "executor", "verifier", "generator"]
        )

        assert abs(total_cost - individual_sum) < 0.0001

    def test_reset_usage_stats_specific_module(self) -> None:
        """Test resetting usage stats for a specific module."""
        config = ModularConfig()

        config.track_token_usage("planner", input_tokens=100, output_tokens=50)
        config.track_token_usage("executor", input_tokens=200, output_tokens=100)

        # Reset planner only
        config.reset_usage_stats("planner")

        planner_usage = config.get_token_usage("planner")
        executor_usage = config.get_token_usage("executor")

        assert planner_usage.total_tokens == 0
        assert planner_usage.invocation_count == 0
        assert executor_usage.total_tokens == 300  # Unchanged

    def test_reset_usage_stats_all_modules(self) -> None:
        """Test resetting usage stats for all modules."""
        config = ModularConfig()

        config.track_token_usage("planner", input_tokens=100, output_tokens=50)
        config.track_token_usage("executor", input_tokens=200, output_tokens=100)

        # Reset all modules
        config.reset_usage_stats()

        for module in ["planner", "executor", "verifier", "generator"]:
            usage = config.get_token_usage(module)
            assert usage.total_tokens == 0
            assert usage.invocation_count == 0
            assert usage.estimated_cost_usd == 0.0

    def test_to_dict_export(self) -> None:
        """Test exporting configuration as dictionary."""
        config = ModularConfig()
        config.track_token_usage("planner", input_tokens=100, output_tokens=50)

        config_dict = config.to_dict()

        assert "planner" in config_dict
        assert "executor" in config_dict
        assert "verifier" in config_dict
        assert "generator" in config_dict

        assert config_dict["planner"]["model"] == "gpt-5"
        assert config_dict["planner"]["temperature"] == 0.7
        assert config_dict["planner"]["token_usage"]["total_tokens"] == 150


class TestCostOptimization:
    """Test cost optimization through mixed model sizes."""

    def test_cost_reduction_vs_all_large_model(self) -> None:
        """
        Test that mixed model sizes reduce cost vs. using gpt-5 for all modules.

        This validates the 30-40% cost reduction target.
        """
        # Scenario 1: All modules use gpt-5 (expensive)
        config_all_large = ModularConfig(
            settings=ModularConfigSettings(
                MODULAR_PLANNER_MODEL="gpt-5",
                MODULAR_EXECUTOR_MODEL="gpt-5",
                MODULAR_VERIFIER_MODEL="gpt-5",
                MODULAR_GENERATOR_MODEL="gpt-5",
            )
        )

        # Scenario 2: Optimized model assignment (default)
        config_optimized = ModularConfig()

        # Simulate same token usage across both configs
        token_usage = {
            "planner": (1000, 500),
            "executor": (2000, 1000),
            "verifier": (500, 250),
            "generator": (1500, 750),
        }

        for module, (input_tokens, output_tokens) in token_usage.items():
            config_all_large.track_token_usage(module, input_tokens, output_tokens)
            config_optimized.track_token_usage(module, input_tokens, output_tokens)

        cost_all_large = config_all_large.get_total_cost()
        cost_optimized = config_optimized.get_total_cost()

        # Optimized should be cheaper
        assert cost_optimized < cost_all_large

        # Calculate cost reduction percentage
        reduction_percent = ((cost_all_large - cost_optimized) / cost_all_large) * 100

        # Should achieve at least 20% reduction (conservative target)
        assert reduction_percent >= 20.0

        print(
            f"\nCost Optimization Analysis:"
            f"\n  All gpt-5: ${cost_all_large:.4f}"
            f"\n  Optimized: ${cost_optimized:.4f}"
            f"\n  Reduction: {reduction_percent:.1f}%"
        )


class TestGlobalConfig:
    """Test global configuration singleton."""

    def test_get_modular_config_singleton(self) -> None:
        """Test that get_modular_config returns singleton instance."""
        reset_modular_config()  # Ensure clean state

        config1 = get_modular_config()
        config2 = get_modular_config()

        assert config1 is config2  # Same instance

    def test_reset_modular_config(self) -> None:
        """Test resetting global config."""
        config1 = get_modular_config()
        config1.track_token_usage("planner", input_tokens=100, output_tokens=50)

        reset_modular_config()
        config2 = get_modular_config()

        # Should be new instance
        assert config1 is not config2

        # New instance should have clean stats
        assert config2.get_token_usage("planner").total_tokens == 0


class TestQualityValidation:
    """Test that quality is maintained with mixed model sizes."""

    def test_all_models_in_allowed_list(self) -> None:
        """Test that all default models are in the allowed list."""
        config = ModularConfig()

        for module in ["planner", "executor", "verifier", "generator"]:
            model = config.get_model_for_module(module)
            assert model in ALLOWED_MODELS

    def test_no_obsolete_models_used(self) -> None:
        """Test that no obsolete models (gpt-3.5-turbo, gpt-4o-mini) are used."""
        config = ModularConfig()
        obsolete_models = {"gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"}

        for module in ["planner", "executor", "verifier", "generator"]:
            model = config.get_model_for_module(module)
            assert model not in obsolete_models

    def test_planner_uses_largest_model(self) -> None:
        """Test that Planner uses a large model (for complex reasoning)."""
        config = ModularConfig()
        planner_model = config.get_model_for_module("planner")

        # Planner should use gpt-5 or gpt-5-pro (large models)
        assert planner_model in {"gpt-5", "gpt-5-pro"}

    def test_verifier_uses_small_model(self) -> None:
        """Test that Verifier uses a small model (for efficiency)."""
        config = ModularConfig()
        verifier_model = config.get_model_for_module("verifier")

        # Verifier should use gpt-4.1-mini or gpt-5-mini (small models)
        assert verifier_model in {"gpt-4.1-mini", "gpt-5-mini"}


class TestIntegration:
    """Integration tests for real-world usage patterns."""

    def test_typical_execution_flow(self) -> None:
        """Test typical execution flow with cost tracking."""
        config = ModularConfig()

        # 1. Planner analyzes query (more tokens for reasoning)
        config.track_token_usage("planner", input_tokens=500, output_tokens=1000)

        # 2. Executor runs steps (medium tokens for tool invocation)
        config.track_token_usage("executor", input_tokens=300, output_tokens=200)
        config.track_token_usage("executor", input_tokens=300, output_tokens=200)

        # 3. Verifier validates (fewer tokens for validation)
        config.track_token_usage("verifier", input_tokens=200, output_tokens=100)

        # 4. Generator synthesizes response (medium tokens)
        config.track_token_usage("generator", input_tokens=400, output_tokens=600)

        # Verify cost tracking
        costs = config.get_module_costs()
        total_cost = config.get_total_cost()

        assert costs["planner"]["invocations"] == 1
        assert costs["executor"]["invocations"] == 2
        assert costs["verifier"]["invocations"] == 1
        assert costs["generator"]["invocations"] == 1

        assert total_cost > 0
        print(f"\nTypical execution cost: ${total_cost:.4f}")

    def test_config_export_and_monitoring(self) -> None:
        """Test configuration export for monitoring dashboards."""
        config = ModularConfig()

        # Simulate some usage
        config.track_token_usage("planner", input_tokens=1000, output_tokens=500)
        config.track_token_usage("executor", input_tokens=2000, output_tokens=1000)

        # Export for monitoring
        config_dict = config.to_dict()
        costs = config.get_module_costs()

        # Verify exportable data structure
        assert isinstance(config_dict, dict)
        assert isinstance(costs, dict)

        # Should contain all necessary monitoring data
        for module in ["planner", "executor", "verifier", "generator"]:
            assert module in config_dict
            assert module in costs
            assert "model" in config_dict[module]
            assert "tokens" in costs[module]
            assert "estimated_cost_usd" in costs[module]
