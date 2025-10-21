"""
DDoS Protection

Advanced DDoS protection mechanisms using rate limiting and pattern detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import redis.asyncio as aioredis
import structlog

from gateway.middleware.rate_limiter import RateLimiter, RateLimitPolicy, RateLimitType

logger = structlog.get_logger()


class ThreatLevel(str, Enum):
    """DDoS threat levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DDoSConfig:
    """DDoS protection configuration."""

    # Global rate limits (applies to entire gateway)
    global_requests_per_second: int = 10000
    global_requests_per_minute: int = 500000

    # Per-IP rate limits
    ip_requests_per_second: int = 100
    ip_requests_per_minute: int = 1000

    # Burst detection
    burst_threshold_multiplier: float = 5.0  # 5x normal rate triggers burst detection
    burst_window_seconds: int = 10

    # Pattern detection
    suspicious_user_agent_patterns: list[str] | None = None
    blocked_user_agents: list[str] | None = None

    # Auto-blocking
    enable_auto_blocking: bool = True
    auto_block_duration_seconds: int = 3600  # 1 hour
    auto_block_threshold: int = 10  # Number of violations before auto-block

    def __post_init__(self) -> None:
        """Initialize default patterns if not provided."""
        if self.suspicious_user_agent_patterns is None:
            self.suspicious_user_agent_patterns = [
                "bot",
                "crawler",
                "spider",
                "scraper",
                "curl",
                "wget",
                "python-requests",
            ]
        if self.blocked_user_agents is None:
            self.blocked_user_agents = []


@dataclass
class DDoSThreatAssessment:
    """DDoS threat assessment result."""

    threat_level: ThreatLevel
    is_blocked: bool
    reasons: list[str]
    metadata: dict[str, Any]


class DDoSProtector:
    """
    DDoS protection system.

    Provides multi-layered protection against DDoS attacks:
    1. Global rate limiting
    2. Per-IP rate limiting
    3. Burst detection
    4. Pattern-based detection
    5. Auto-blocking of repeat offenders
    """

    def __init__(
        self,
        rate_limiter: RateLimiter,
        config: DDoSConfig | None = None,
    ) -> None:
        """
        Initialize DDoS protector.

        Args:
            rate_limiter: RateLimiter instance to use
            config: DDoS protection configuration (uses defaults if None)
        """
        self.rate_limiter = rate_limiter
        self.config = config or DDoSConfig()

    async def assess_threat(
        self,
        client_ip: str,
        user_agent: str | None = None,
        endpoint: str | None = None,
    ) -> DDoSThreatAssessment:
        """
        Assess DDoS threat level for a request.

        Args:
            client_ip: Client IP address
            user_agent: User-Agent header value
            endpoint: Request endpoint/path

        Returns:
            DDoSThreatAssessment with threat level and blocking decision
        """
        threat_level = ThreatLevel.NONE
        is_blocked = False
        reasons: list[str] = []
        metadata: dict[str, Any] = {}

        # 1. Check if IP is already blocked
        if await self._is_ip_blocked(client_ip):
            return DDoSThreatAssessment(
                threat_level=ThreatLevel.CRITICAL,
                is_blocked=True,
                reasons=["IP address is blocked"],
                metadata={"client_ip": client_ip},
            )

        # 2. Check user agent patterns
        if user_agent:
            ua_threat, ua_reasons = self._check_user_agent(user_agent)
            if ua_threat != ThreatLevel.NONE:
                threat_level = max(
                    threat_level, ua_threat, key=lambda x: list(ThreatLevel).index(x)
                )
                reasons.extend(ua_reasons)
                metadata["user_agent"] = user_agent

        # 3. Check global rate limits
        global_check = await self._check_global_limits()
        if not global_check.allowed:
            threat_level = ThreatLevel.HIGH
            is_blocked = True
            reasons.append("Global rate limit exceeded")
            metadata["global_limit"] = {
                "limit": global_check.limit,
                "retry_after": global_check.retry_after,
            }

        # 4. Check per-IP rate limits
        ip_check = await self._check_ip_limits(client_ip)
        if not ip_check.allowed:
            threat_level = max(
                threat_level,
                ThreatLevel.MEDIUM,
                key=lambda x: list(ThreatLevel).index(x),
            )
            is_blocked = True
            reasons.append("IP rate limit exceeded")
            metadata["ip_limit"] = {
                "limit": ip_check.limit,
                "retry_after": ip_check.retry_after,
            }

            # Record violation for auto-blocking
            if self.config.enable_auto_blocking:
                await self._record_violation(client_ip)

        # 5. Check for burst traffic
        if await self._detect_burst(client_ip):
            threat_level = max(
                threat_level,
                ThreatLevel.MEDIUM,
                key=lambda x: list(ThreatLevel).index(x),
            )
            reasons.append("Burst traffic detected")
            metadata["burst_detected"] = True

        # 6. Auto-block if threshold exceeded
        if self.config.enable_auto_blocking and not is_blocked:
            violations = await self._get_violation_count(client_ip)
            if violations >= self.config.auto_block_threshold:
                await self._block_ip(client_ip)
                threat_level = ThreatLevel.CRITICAL
                is_blocked = True
                reasons.append(f"Auto-blocked after {violations} violations")
                metadata["auto_blocked"] = True

        return DDoSThreatAssessment(
            threat_level=threat_level,
            is_blocked=is_blocked,
            reasons=reasons,
            metadata=metadata,
        )

    async def _check_global_limits(self) -> Any:
        """Check global rate limits."""
        # Check requests per second
        policy_per_second = RateLimitPolicy(
            limit=self.config.global_requests_per_second,
            window_seconds=1,
        )

        result = await self.rate_limiter.check_rate_limit(
            limit_type=RateLimitType.GLOBAL,
            identifier="gateway",
            policy=policy_per_second,
        )

        if not result.allowed:
            return result

        # Check requests per minute
        policy_per_minute = RateLimitPolicy(
            limit=self.config.global_requests_per_minute,
            window_seconds=60,
        )

        return await self.rate_limiter.check_rate_limit(
            limit_type=RateLimitType.GLOBAL,
            identifier="gateway",
            policy=policy_per_minute,
        )

    async def _check_ip_limits(self, client_ip: str) -> Any:
        """Check per-IP rate limits."""
        # Check requests per second
        policy_per_second = RateLimitPolicy(
            limit=self.config.ip_requests_per_second,
            window_seconds=1,
        )

        result = await self.rate_limiter.check_rate_limit(
            limit_type=RateLimitType.CLIENT_IP,
            identifier=client_ip,
            policy=policy_per_second,
        )

        if not result.allowed:
            return result

        # Check requests per minute
        policy_per_minute = RateLimitPolicy(
            limit=self.config.ip_requests_per_minute,
            window_seconds=60,
        )

        return await self.rate_limiter.check_rate_limit(
            limit_type=RateLimitType.CLIENT_IP,
            identifier=client_ip,
            policy=policy_per_minute,
        )

    def _check_user_agent(self, user_agent: str) -> tuple[ThreatLevel, list[str]]:
        """
        Check user agent for suspicious patterns.

        Args:
            user_agent: User-Agent header value

        Returns:
            Tuple of (ThreatLevel, reasons)
        """
        user_agent_lower = user_agent.lower()
        reasons: list[str] = []

        # Check blocked user agents
        for blocked in self.config.blocked_user_agents or []:
            if blocked.lower() in user_agent_lower:
                reasons.append(f"Blocked user agent: {blocked}")
                return ThreatLevel.CRITICAL, reasons

        # Check suspicious patterns
        for pattern in self.config.suspicious_user_agent_patterns or []:
            if pattern.lower() in user_agent_lower:
                reasons.append(f"Suspicious user agent pattern: {pattern}")
                return ThreatLevel.LOW, reasons

        return ThreatLevel.NONE, reasons

    async def _detect_burst(self, client_ip: str) -> bool:
        """
        Detect burst traffic from IP.

        Args:
            client_ip: Client IP address

        Returns:
            True if burst detected, False otherwise
        """
        burst_limit = int(
            self.config.ip_requests_per_second * self.config.burst_threshold_multiplier
        )

        policy = RateLimitPolicy(
            limit=burst_limit,
            window_seconds=self.config.burst_window_seconds,
        )

        result = await self.rate_limiter.check_rate_limit(
            limit_type=RateLimitType.CLIENT_IP,
            identifier=f"{client_ip}:burst",
            policy=policy,
        )

        return not result.allowed

    async def _is_ip_blocked(self, client_ip: str) -> bool:
        """
        Check if IP is blocked.

        Args:
            client_ip: Client IP address

        Returns:
            True if IP is blocked, False otherwise
        """
        key = f"ddos:blocked:{client_ip}"
        exists = await self.rate_limiter.client.exists(key)
        return bool(exists)

    async def _block_ip(self, client_ip: str) -> None:
        """
        Block an IP address.

        Args:
            client_ip: Client IP address to block
        """
        key = f"ddos:blocked:{client_ip}"
        await self.rate_limiter.client.setex(
            key,
            self.config.auto_block_duration_seconds,
            "1",
        )

        logger.warning(
            "IP address blocked",
            client_ip=client_ip,
            duration_seconds=self.config.auto_block_duration_seconds,
        )

    async def _record_violation(self, client_ip: str) -> None:
        """
        Record a rate limit violation.

        Args:
            client_ip: Client IP address
        """
        key = f"ddos:violations:{client_ip}"
        await self.rate_limiter.client.incr(key)
        await self.rate_limiter.client.expire(key, 3600)  # 1 hour window

    async def _get_violation_count(self, client_ip: str) -> int:
        """
        Get violation count for IP.

        Args:
            client_ip: Client IP address

        Returns:
            Number of violations
        """
        key = f"ddos:violations:{client_ip}"
        count = await self.rate_limiter.client.get(key)
        return int(count) if count else 0

    async def unblock_ip(self, client_ip: str) -> bool:
        """
        Manually unblock an IP address.

        Args:
            client_ip: Client IP address to unblock

        Returns:
            True if IP was unblocked, False if not found
        """
        key = f"ddos:blocked:{client_ip}"
        deleted = await self.rate_limiter.client.delete(key)

        if deleted:
            logger.info("IP address unblocked", client_ip=client_ip)

        return bool(deleted)

    async def get_blocked_ips(self) -> list[str]:
        """
        Get list of currently blocked IPs.

        Returns:
            List of blocked IP addresses
        """
        keys = await self.rate_limiter.client.keys("ddos:blocked:*")
        return [key.decode("utf-8").replace("ddos:blocked:", "") for key in keys]

    async def get_ip_statistics(self, client_ip: str) -> dict[str, Any]:
        """
        Get statistics for an IP address.

        Args:
            client_ip: Client IP address

        Returns:
            Dictionary with IP statistics
        """
        is_blocked = await self._is_ip_blocked(client_ip)
        violations = await self._get_violation_count(client_ip)

        # Get current rate limit info
        limit_info = await self.rate_limiter.get_limit_info(
            RateLimitType.CLIENT_IP,
            client_ip,
        )

        return {
            "client_ip": client_ip,
            "is_blocked": is_blocked,
            "violation_count": violations,
            "rate_limit_info": limit_info,
        }
