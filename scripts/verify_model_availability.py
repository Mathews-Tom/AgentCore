#!/usr/bin/env python3
"""Verify LLM model availability across all providers and tiers.

This script tests the availability of all models configured in TIER_MODEL_MAP:
- 3 Tiers: FAST, BALANCED, PREMIUM
- 3 Providers: OpenAI, Anthropic, Gemini

For each model, it performs a simple API call to verify:
1. Authentication works
2. Model is accessible
3. Basic completion functionality works

The script gracefully handles missing API keys by skipping those providers.

Usage:
    uv run python scripts/verify_model_availability.py

Environment Variables:
    OPENAI_API_KEY - Required for OpenAI models (gpt-5-mini, gpt-5, gpt-5-pro)
    ANTHROPIC_API_KEY - Required for Anthropic models (claude-*)
    GEMINI_API_KEY - Required for Google models (gemini-*)

Exit codes:
    0: All models with configured API keys verified successfully
    1: Some models failed verification

Example Output:
    ================================================================================
    LLM Model Availability Verification
    ================================================================================

    API Key Status:
      ✓ OPENAI_API_KEY
      ✓ ANTHROPIC_API_KEY
      ✓ GEMINI_API_KEY

    Verifying 9 models from TIER_MODEL_MAP...

    Verifying openai/gpt-5-mini (fast)... ✓ Available
    Verifying openai/gpt-5 (balanced)... ✓ Available
    Verifying openai/gpt-5-pro (premium)... ✓ Available
    [... 6 more models ...]

    ✓ All configured models verified successfully!

Notes:
    - Models are loaded automatically from TIER_MODEL_MAP in model_selector.py
    - gpt-5-pro uses the v1/responses endpoint and requires longer timeout (60s)
    - Script requires python-dotenv to load .env file
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from agentcore.a2a_protocol.models.llm import LLMRequest, ModelTier, Provider
from agentcore.a2a_protocol.services.llm_client_anthropic import LLMClientAnthropic
from agentcore.a2a_protocol.services.llm_client_gemini import LLMClientGemini
from agentcore.a2a_protocol.services.llm_client_openai import LLMClientOpenAI
from agentcore.a2a_protocol.services.model_selector import TIER_MODEL_MAP

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


@dataclass
class ModelConfig:
    """Model configuration for verification."""

    provider: Provider
    tier: ModelTier
    model: str


def get_models_to_verify() -> list[ModelConfig]:
    """Generate list of models to verify from TIER_MODEL_MAP.

    Returns:
        List of ModelConfig objects extracted from TIER_MODEL_MAP
    """
    models: list[ModelConfig] = []

    # Map model prefixes to providers
    provider_map = {
        "gpt-": Provider.OPENAI,
        "claude-": Provider.ANTHROPIC,
        "gemini-": Provider.GEMINI,
    }

    for tier, model_list in TIER_MODEL_MAP.items():
        for model in model_list:
            # Determine provider from model name prefix
            provider = None
            for prefix, prov in provider_map.items():
                if model.startswith(prefix):
                    provider = prov
                    break

            if provider:
                models.append(ModelConfig(provider=provider, tier=tier, model=model))

    return models


@dataclass
class VerificationResult:
    """Result of model verification."""

    provider: Provider
    tier: ModelTier
    model: str
    available: bool
    error: str | None = None


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header() -> None:
    """Print script header."""
    print(f"\n{Colors.BOLD}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}LLM Model Availability Verification{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}\n")


def print_summary(results: list[VerificationResult]) -> None:
    """Print verification summary."""
    total = len(results)
    available = sum(1 for r in results if r.available)
    unavailable = total - available

    print(f"\n{Colors.BOLD}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}Verification Summary{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}\n")
    print(f"Total models tested: {total}")
    print(
        f"Available: {Colors.GREEN}{available}{Colors.RESET} "
        f"({available / total * 100:.1f}%)"
    )

    if unavailable > 0:
        print(
            f"Unavailable: {Colors.RED}{unavailable}{Colors.RESET} "
            f"({unavailable / total * 100:.1f}%)"
        )

        print(f"\n{Colors.YELLOW}Unavailable models:{Colors.RESET}")
        for result in results:
            if not result.available:
                print(
                    f"  {Colors.RED}✗{Colors.RESET} "
                    f"{result.provider.value}/{result.model} ({result.tier.value})"
                )
                if result.error:
                    print(f"    Error: {result.error}")

    # Provider breakdown
    print(f"\n{Colors.BOLD}By Provider:{Colors.RESET}")
    for provider in Provider:
        provider_results = [r for r in results if r.provider == provider]
        provider_available = sum(1 for r in provider_results if r.available)
        provider_total = len(provider_results)

        status = (
            f"{Colors.GREEN}✓{Colors.RESET}"
            if provider_available == provider_total
            else f"{Colors.YELLOW}⚠{Colors.RESET}"
        )
        print(
            f"  {status} {provider.value}: "
            f"{provider_available}/{provider_total} available"
        )

    # Tier breakdown
    print(f"\n{Colors.BOLD}By Tier:{Colors.RESET}")
    for tier in ModelTier:
        tier_results = [r for r in results if r.tier == tier]
        tier_available = sum(1 for r in tier_results if r.available)
        tier_total = len(tier_results)

        status = (
            f"{Colors.GREEN}✓{Colors.RESET}"
            if tier_available == tier_total
            else f"{Colors.YELLOW}⚠{Colors.RESET}"
        )
        print(f"  {status} {tier.value}: {tier_available}/{tier_total} available")


def check_api_keys() -> dict[Provider, bool]:
    """Check which API keys are configured.

    Returns:
        Dictionary mapping Provider to availability status
    """
    return {
        Provider.OPENAI: bool(os.getenv("OPENAI_API_KEY")),
        Provider.ANTHROPIC: bool(os.getenv("ANTHROPIC_API_KEY")),
        Provider.GEMINI: bool(os.getenv("GEMINI_API_KEY")),
    }


async def verify_openai_model(model: str) -> tuple[bool, str | None]:
    """Verify OpenAI model availability.

    Args:
        model: Model identifier to verify

    Returns:
        Tuple of (available, error_message)
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return False, "OPENAI_API_KEY not configured"

    try:
        # Reasoning models (gpt-5-pro) need longer timeout
        timeout = 60.0 if model in ("gpt-5-pro", "o1-pro", "o3-mini") else 10.0

        client = LLMClientOpenAI(api_key=api_key, timeout=timeout, max_retries=1)
        request = LLMRequest(
            model=model,
            messages=[{"role": "user", "content": "Say 'OK' only."}],
        )
        response = await client.complete(request)

        # Check if response has content
        if not response.content:
            return False, f"Model returned empty content (usage: {response.usage.total_tokens} tokens)"

        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


async def verify_anthropic_model(model: str) -> tuple[bool, str | None]:
    """Verify Anthropic model availability.

    Args:
        model: Model identifier to verify

    Returns:
        Tuple of (available, error_message)
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return False, "ANTHROPIC_API_KEY not configured"

    try:
        client = LLMClientAnthropic(api_key=api_key, timeout=10.0, max_retries=1)
        request = LLMRequest(
            model=model,
            messages=[{"role": "user", "content": "Say 'OK' only."}],
        )
        response = await client.complete(request)
        return bool(response.content), None
    except Exception as e:
        return False, str(e)


async def verify_gemini_model(model: str) -> tuple[bool, str | None]:
    """Verify Gemini model availability.

    Args:
        model: Model identifier to verify

    Returns:
        Tuple of (available, error_message)
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return False, "GEMINI_API_KEY not configured"

    try:
        client = LLMClientGemini(api_key=api_key, timeout=10.0, max_retries=1)
        request = LLMRequest(
            model=model,
            messages=[{"role": "user", "content": "Say 'OK' only."}],
        )
        response = await client.complete(request)
        return bool(response.content), None
    except Exception as e:
        return False, str(e)


async def verify_model(config: ModelConfig) -> VerificationResult:
    """Verify a single model's availability.

    Args:
        config: Model configuration to verify

    Returns:
        Verification result
    """
    print(
        f"Verifying {Colors.BLUE}{config.provider.value}{Colors.RESET}/"
        f"{config.model} ({config.tier.value})...",
        end=" ",
    )

    try:
        if config.provider == Provider.OPENAI:
            available, error = await verify_openai_model(config.model)
        elif config.provider == Provider.ANTHROPIC:
            available, error = await verify_anthropic_model(config.model)
        elif config.provider == Provider.GEMINI:
            available, error = await verify_gemini_model(config.model)
        else:
            available, error = False, f"Unknown provider: {config.provider}"

        if available:
            print(f"{Colors.GREEN}✓ Available{Colors.RESET}")
        else:
            print(f"{Colors.RED}✗ Unavailable{Colors.RESET}")
            if error:
                print(f"  Error: {error}")

        return VerificationResult(
            provider=config.provider,
            tier=config.tier,
            model=config.model,
            available=available,
            error=error,
        )
    except Exception as e:
        print(f"{Colors.RED}✗ Error{Colors.RESET}")
        print(f"  Exception: {e}")
        return VerificationResult(
            provider=config.provider,
            tier=config.tier,
            model=config.model,
            available=False,
            error=str(e),
        )


async def main() -> int:
    """Main verification entry point.

    Returns:
        Exit code (0=success, 1=failures)
    """
    print_header()

    # Check API keys
    api_key_status = check_api_keys()

    print(f"{Colors.BOLD}API Key Status:{Colors.RESET}")
    for provider, available in api_key_status.items():
        status = (
            f"{Colors.GREEN}✓{Colors.RESET}"
            if available
            else f"{Colors.RED}✗{Colors.RESET}"
        )
        key_name = f"{provider.value.upper()}_API_KEY"
        print(f"  {status} {key_name}")

    available_providers = [p for p, status in api_key_status.items() if status]
    if not available_providers:
        print(
            f"\n{Colors.RED}Error: No API keys configured. "
            f"Please set at least one API key.{Colors.RESET}\n"
        )
        return 1

    print()

    # Get models to verify from TIER_MODEL_MAP
    all_models = get_models_to_verify()
    models_to_verify = [m for m in all_models if api_key_status[m.provider]]
    skipped_models = [m for m in all_models if not api_key_status[m.provider]]

    if skipped_models:
        print(
            f"{Colors.YELLOW}Skipping {len(skipped_models)} models "
            f"(API keys not configured){Colors.RESET}\n"
        )

    # Verify available models
    print(
        f"{Colors.BOLD}Verifying {len(models_to_verify)} models "
        f"from TIER_MODEL_MAP...{Colors.RESET}\n"
    )
    results: list[VerificationResult] = []

    for config in models_to_verify:
        result = await verify_model(config)
        results.append(result)

    # Print summary
    print_summary(results)

    # Determine exit code
    if not results:
        print(
            f"\n{Colors.YELLOW}{Colors.BOLD}⚠ No models could be verified "
            f"(no API keys configured){Colors.RESET}\n"
        )
        return 1

    all_available = all(r.available for r in results)
    if all_available:
        print(
            f"\n{Colors.GREEN}{Colors.BOLD}✓ All configured models verified successfully!"
            f"{Colors.RESET}\n"
        )
        return 0
    else:
        print(
            f"\n{Colors.YELLOW}{Colors.BOLD}⚠ Some models are unavailable"
            f"{Colors.RESET}\n"
        )
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
    sys.exit(exit_code)
