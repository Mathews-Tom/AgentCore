"""XSS (Cross-Site Scripting) prevention tests.

Tests various XSS attack patterns to ensure proper input sanitization.
"""

import pytest
from pydantic import BaseModel, Field, ValidationError, field_validator


class UserInput(BaseModel):
    """Model for user input with XSS validation."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)

    @field_validator("name", "description")
    @classmethod
    def validate_no_xss(cls, v: str | None) -> str | None:
        """Validate against XSS patterns."""
        if v is None:
            return v

        dangerous_patterns = [
            "<script",
            "</script>",
            "javascript:",
            "onerror=",
            "onload=",
            "onclick=",
            "onmouseover=",
            "onfocus=",
            "<iframe",
            "</iframe>",
            "<object",
            "<embed",
            "<img",
            "<svg",
            "<body",
            "<input",
            "eval(",
            "expression(",
            "vbscript:",
            "data:text/html",
        ]

        v_lower = v.lower()
        for pattern in dangerous_patterns:
            if pattern in v_lower:
                raise ValueError(f"Invalid input: contains dangerous pattern '{pattern}'")

        return v


# XSS attack patterns
XSS_PATTERNS = [
    # Basic script injection
    "<script>alert('XSS')</script>",
    "<script>alert(document.cookie)</script>",
    # Event handlers
    "<img src=x onerror=alert('XSS')>",
    "<body onload=alert('XSS')>",
    "<input onfocus=alert('XSS') autofocus>",
    # JavaScript protocol
    "<a href='javascript:alert(1)'>Click</a>",
    # Encoded attacks
    "<script>alert(String.fromCharCode(88,83,83))</script>",
    # SVG-based XSS
    "<svg onload=alert('XSS')>",
    "<svg><script>alert('XSS')</script></svg>",
    # iframe injection
    "<iframe src='javascript:alert(1)'>",
    # Object/Embed
    "<object data='javascript:alert(1)'>",
    "<embed src='javascript:alert(1)'>",
    # HTML5 attacks
    "<video src=x onerror=alert('XSS')>",
    "<audio src=x onerror=alert('XSS')>",
    # Style-based XSS
    "<style>*{background:url('javascript:alert(1)')}</style>",
    # Expression-based (IE)
    "<div style='width:expression(alert(1))'>",
    # Base64 encoded
    "data:text/html;base64,PHNjcmlwdD5hbGVydCgnWFNTJyk8L3NjcmlwdD4=",
    # Polyglot XSS
    "javascript:/*--></title></style></textarea></script></xmp><svg/onload='+/\"/+/onmouseover=1/+/[*/[]/+alert(1)//'>",
]


@pytest.mark.parametrize("payload", XSS_PATTERNS)
def test_xss_prevention(payload: str) -> None:
    """Test that XSS patterns are rejected."""
    with pytest.raises(ValidationError):
        UserInput(name=payload)


def test_valid_user_input() -> None:
    """Test that valid user input is accepted."""
    valid_inputs = [
        ("John Doe", "Regular user"),
        ("Agent-123", "Test agent description"),
        ("User_2023", "New user profile"),
        ("Admin", "System administrator"),
    ]

    for name, desc in valid_inputs:
        user = UserInput(name=name, description=desc)
        assert user.name == name
        assert user.description == desc


def test_special_characters_allowed() -> None:
    """Test that safe special characters are allowed."""
    # These should be safe and allowed
    safe_inputs = [
        "User @ Company",
        "John & Jane",
        "Price: $100",
        "Math: 2 + 2 = 4",
        "Email: user@example.com",
    ]

    for name in safe_inputs:
        user = UserInput(name=name)
        assert user.name == name


def test_length_limits() -> None:
    """Test that length limits are enforced."""
    with pytest.raises(ValidationError):
        UserInput(name="a" * 201)

    with pytest.raises(ValidationError):
        UserInput(name="valid", description="a" * 1001)


def test_empty_name() -> None:
    """Test that empty name is rejected."""
    with pytest.raises(ValidationError):
        UserInput(name="")


def test_none_description() -> None:
    """Test that None description is allowed."""
    user = UserInput(name="Test User", description=None)
    assert user.description is None


def test_case_insensitive_detection() -> None:
    """Test that XSS detection is case-insensitive."""
    patterns = [
        "<SCRIPT>alert('XSS')</SCRIPT>",
        "<ScRiPt>alert('XSS')</sCrIpT>",
        "<img SRC=x ONERROR=alert('XSS')>",
    ]

    for pattern in patterns:
        with pytest.raises(ValidationError):
            UserInput(name=pattern)
