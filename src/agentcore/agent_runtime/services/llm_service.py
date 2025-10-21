"""LLM service using Portkey AI Gateway."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class LLMConfig(BaseModel):
    """LLM configuration via Portkey."""

    portkey_api_key: str = Field(..., description="Portkey API key")
    portkey_base_url: str = Field(
        default="https://api.portkey.ai",
        description="Portkey gateway URL",
    )
    default_model: str = Field(default="gpt-4.1", description="Default LLM model")
    fallback_models: list[str] = Field(
        default_factory=lambda: ["gpt-4.1-mini"],
        description="Fallback models for resilience",
    )
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=500, ge=1, le=8000)
    timeout_seconds: int = Field(default=30, ge=5, le=300)
    max_retries: int = Field(default=3, ge=0, le=10)
    cache_enabled: bool = Field(default=True, description="Enable Portkey caching")


class LLMResponse(BaseModel):
    """Response from LLM."""

    content: str = Field(description="Generated text content")
    model: str = Field(description="Model used")
    tokens_used: int = Field(description="Total tokens consumed")
    finish_reason: str = Field(description="Reason for completion")
    cached: bool = Field(default=False, description="Whether response was cached")


class PortkeyLLMService:
    """LLM service using Portkey AI Gateway."""

    def __init__(self, config: LLMConfig):
        """
        Initialize Portkey LLM service.

        Args:
            config: LLM configuration
        """
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=self.config.portkey_base_url,
            headers={
                "x-portkey-api-key": self.config.portkey_api_key,
                "Content-Type": "application/json",
            },
            timeout=self.config.timeout_seconds,
        )

        logger.info(
            "portkey_llm_service_initialized",
            base_url=self.config.portkey_base_url,
            default_model=self.config.default_model,
            cache_enabled=self.config.cache_enabled,
        )

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate completion via Portkey.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            model: Model to use (defaults to config)
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional Portkey-specific parameters

        Returns:
            LLM response

        Raises:
            httpx.HTTPError: If request fails
        """
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build payload
        model_to_use = model or self.config.default_model
        payload = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature or self.config.default_temperature,
            "max_tokens": max_tokens or self.config.default_max_tokens,
        }
        payload.update(kwargs)

        # Portkey-specific headers for routing
        headers = {}
        if self.config.fallback_models:
            headers["x-portkey-fallback"] = ",".join(self.config.fallback_models)
        if self.config.max_retries:
            headers["x-portkey-retry"] = str(self.config.max_retries)
        if self.config.cache_enabled:
            headers["x-portkey-cache"] = "simple"

        logger.debug(
            "portkey_llm_request",
            model=model_to_use,
            temperature=payload["temperature"],
            max_tokens=payload["max_tokens"],
            cache_enabled=self.config.cache_enabled,
        )

        try:
            # Make request
            response = await self.client.post(
                "/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()

            # Parse response
            data = response.json()
            choice = data["choices"][0]

            cache_status = response.headers.get("x-portkey-cache-status", "MISS")
            cached = cache_status == "HIT"

            llm_response = LLMResponse(
                content=choice["message"]["content"],
                model=data.get("model", model_to_use),
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
                finish_reason=choice.get("finish_reason", "stop"),
                cached=cached,
            )

            logger.info(
                "portkey_llm_response",
                model=llm_response.model,
                tokens=llm_response.tokens_used,
                cached=cached,
                finish_reason=llm_response.finish_reason,
            )

            return llm_response

        except httpx.HTTPError as e:
            logger.error(
                "portkey_llm_request_failed",
                error=str(e),
                model=model_to_use,
            )
            raise

    async def stream_complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream completion via Portkey.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            model: Model to use
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Yields:
            Content chunks

        Raises:
            httpx.HTTPError: If request fails
        """
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        model_to_use = model or self.config.default_model
        payload = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature or self.config.default_temperature,
            "stream": True,
        }
        payload.update(kwargs)

        logger.debug(
            "portkey_llm_stream_request",
            model=model_to_use,
            temperature=payload["temperature"],
        )

        try:
            async with self.client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            if content := chunk["choices"][0]["delta"].get("content"):
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError) as e:
                            logger.warning(
                                "portkey_stream_parse_error",
                                error=str(e),
                                line=line,
                            )
                            continue

        except httpx.HTTPError as e:
            logger.error(
                "portkey_llm_stream_failed",
                error=str(e),
                model=model_to_use,
            )
            raise

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
        logger.info("portkey_llm_service_closed")


# Global singleton
_llm_service: PortkeyLLMService | None = None


def get_llm_service(config: LLMConfig | None = None) -> PortkeyLLMService:
    """
    Get global LLM service instance.

    Args:
        config: Optional LLM configuration (used for initialization)

    Returns:
        Global LLM service instance

    Raises:
        RuntimeError: If service not initialized and no config provided
    """
    global _llm_service

    if _llm_service is None:
        if config is None:
            raise RuntimeError(
                "LLM service not initialized. Call initialize_llm_service() first."
            )
        _llm_service = PortkeyLLMService(config)

    return _llm_service


def initialize_llm_service(config: LLMConfig) -> PortkeyLLMService:
    """
    Initialize global LLM service.

    Args:
        config: LLM configuration

    Returns:
        Initialized LLM service
    """
    global _llm_service
    _llm_service = PortkeyLLMService(config)
    return _llm_service


async def cleanup_llm_service() -> None:
    """Cleanup global LLM service."""
    global _llm_service
    if _llm_service is not None:
        await _llm_service.close()
        _llm_service = None
