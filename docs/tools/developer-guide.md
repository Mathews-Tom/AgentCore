# Tool Integration Framework - Developer Guide

**Version:** 1.0.0
**Last Updated:** 2025-01-13
**Audience:** Developers building custom tools for AgentCore

## Table of Contents

- [Getting Started](#getting-started)
- [Creating Your First Tool](#creating-your-first-tool)
- [Parameter Validation](#parameter-validation)
- [Error Handling](#error-handling)
- [Authentication](#authentication)
- [Testing Your Tool](#testing-your-tool)
- [Best Practices](#best-practices)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- Python 3.12+
- AgentCore development environment
- Understanding of async/await patterns
- Familiarity with Pydantic v2

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/agentcore.git
cd agentcore

# Install dependencies
uv sync

# Run tests to verify setup
uv run pytest tests/agent_runtime/tools/
```

### Project Structure

```
src/agentcore/agent_runtime/tools/
├── __init__.py
├── base.py                    # Tool interface and ExecutionContext
├── executor.py                # Tool execution orchestrator
├── registry.py                # Tool discovery and management
├── registration.py            # Built-in tool registration
└── builtin/                   # Built-in tool implementations
    ├── __init__.py
    ├── search_tools.py       # Search tool implementations
    ├── code_execution_tools.py # Code execution tools
    ├── api_tools.py          # API client tools
    └── utility_tools.py      # Utility tools
```

## Creating Your First Tool

### Step 1: Define Tool Metadata

Create a new file `src/agentcore/agent_runtime/tools/builtin/weather_tools.py`:

```python
"""Weather tool implementations."""

import time
from typing import Any

import httpx
import structlog

from agentcore.agent_runtime.models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolExecutionStatus,
    ToolParameter,
    ToolResult,
)
from agentcore.agent_runtime.tools.base import ExecutionContext, Tool

logger = structlog.get_logger()


class OpenWeatherMapTool(Tool):
    """Tool for fetching weather data from OpenWeatherMap API."""

    def __init__(self, api_key: str | None = None):
        """Initialize OpenWeatherMap tool.

        Args:
            api_key: OpenWeatherMap API key (defaults to env var)
        """
        metadata = ToolDefinition(
            tool_id="openweathermap",
            name="OpenWeatherMap",
            description="Get current weather, forecasts, and historical data",
            version="1.0.0",
            category=ToolCategory.API_CLIENT,
            parameters={
                "location": ToolParameter(
                    name="location",
                    type="string",
                    description="City name or 'lat,lon' coordinates",
                    required=True,
                    min_length=2,
                    max_length=100,
                ),
                "units": ToolParameter(
                    name="units",
                    type="string",
                    description="Temperature units",
                    required=False,
                    default="metric",
                    enum=["metric", "imperial", "standard"],
                ),
                "forecast_days": ToolParameter(
                    name="forecast_days",
                    type="integer",
                    description="Number of forecast days (0 for current only)",
                    required=False,
                    default=0,
                    min_value=0,
                    max_value=7,
                ),
            },
            auth_method=AuthMethod.API_KEY,
            auth_config={
                "api_key_header": "X-API-Key",
                "api_key_param": "appid",
            },
            timeout_seconds=10,
            is_retryable=True,
            is_idempotent=True,
            max_retries=3,
            rate_limits={
                "calls_per_minute": 60,
                "calls_per_day": 1000,
            },
            cost_per_execution=0.0001,  # $0.0001 per call
            tags=["weather", "forecast", "api"],
            metadata={
                "api_version": "2.5",
                "provider": "OpenWeatherMap",
                "documentation": "https://openweathermap.org/api",
            },
        )
        super().__init__(metadata)
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute weather lookup.

        Args:
            parameters: Tool parameters (location, units, forecast_days)
            context: Execution context with user/agent identifiers

        Returns:
            ToolResult with weather data or error information
        """
        start_time = time.time()

        # Validate parameters
        is_valid, validation_error = await self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                error=validation_error,
                error_type="ValidationError",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        try:
            # Extract parameters
            location = parameters["location"]
            units = parameters.get("units", "metric")
            forecast_days = parameters.get("forecast_days", 0)

            self.logger.info(
                "weather_lookup_started",
                location=location,
                units=units,
                forecast_days=forecast_days,
                request_id=context.request_id,
            )

            # Fetch weather data
            if forecast_days == 0:
                # Current weather only
                weather_data = await self._fetch_current(location, units)
            else:
                # Weather + forecast
                current = await self._fetch_current(location, units)
                forecast = await self._fetch_forecast(location, units, forecast_days)
                weather_data = {
                    "current": current,
                    "forecast": forecast,
                }

            execution_time_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "weather_lookup_completed",
                location=location,
                execution_time_ms=execution_time_ms,
                request_id=context.request_id,
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.SUCCESS,
                result=weather_data,
                execution_time_ms=execution_time_ms,
                metadata={
                    "location": location,
                    "units": units,
                    "data_source": "OpenWeatherMap API v2.5",
                },
            )

        except httpx.HTTPStatusError as e:
            execution_time_ms = (time.time() - start_time) * 1000
            error_msg = f"API request failed: {e.response.status_code}"

            self.logger.error(
                "weather_lookup_failed",
                error=error_msg,
                status_code=e.response.status_code,
                location=location,
                request_id=context.request_id,
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                error=error_msg,
                error_type="HTTPError",
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000

            self.logger.error(
                "weather_lookup_failed",
                error=str(e),
                error_type=type(e).__name__,
                location=location,
                request_id=context.request_id,
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=execution_time_ms,
            )

    async def _fetch_current(self, location: str, units: str) -> dict[str, Any]:
        """Fetch current weather data.

        Args:
            location: City name or coordinates
            units: Temperature units

        Returns:
            Current weather data

        Raises:
            httpx.HTTPStatusError: If API request fails
        """
        async with httpx.AsyncClient() as client:
            params = {
                "q": location,
                "units": units,
                "appid": self.api_key,
            }

            response = await client.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=self.metadata.timeout_seconds,
            )
            response.raise_for_status()

            data = response.json()

            # Transform to standardized format
            return {
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "conditions": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"],
                "wind_direction": data["wind"].get("deg"),
                "timestamp": data["dt"],
            }

    async def _fetch_forecast(
        self,
        location: str,
        units: str,
        days: int,
    ) -> list[dict[str, Any]]:
        """Fetch weather forecast.

        Args:
            location: City name or coordinates
            units: Temperature units
            days: Number of forecast days

        Returns:
            List of daily forecasts

        Raises:
            httpx.HTTPStatusError: If API request fails
        """
        async with httpx.AsyncClient() as client:
            params = {
                "q": location,
                "units": units,
                "cnt": days * 8,  # 8 data points per day (3-hour intervals)
                "appid": self.api_key,
            }

            response = await client.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=self.metadata.timeout_seconds,
            )
            response.raise_for_status()

            data = response.json()

            # Group by day and return daily summaries
            forecasts = []
            for item in data["list"][::8]:  # One per day
                forecasts.append({
                    "date": item["dt_txt"].split()[0],
                    "temperature": item["main"]["temp"],
                    "conditions": item["weather"][0]["description"],
                    "humidity": item["main"]["humidity"],
                    "wind_speed": item["wind"]["speed"],
                })

            return forecasts[:days]
```

### Step 2: Register Your Tool

Add registration in `src/agentcore/agent_runtime/tools/registration.py`:

```python
from agentcore.agent_runtime.tools.builtin.weather_tools import OpenWeatherMapTool

def register_builtin_tools(registry: "ToolRegistry") -> None:
    """Register all built-in tools with the registry."""
    # Existing tools...

    # Weather tools
    weather_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if weather_api_key:
        await registry.register(OpenWeatherMapTool(api_key=weather_api_key))
    else:
        logger.warning("OpenWeatherMap API key not found, tool not registered")
```

### Step 3: Test Your Tool

Create `tests/agent_runtime/tools/test_weather_tools.py`:

```python
"""Tests for weather tool implementations."""

import pytest
from unittest.mock import AsyncMock, patch

from agentcore.agent_runtime.models.tool_integration import (
    ToolExecutionStatus,
)
from agentcore.agent_runtime.tools.base import ExecutionContext
from agentcore.agent_runtime.tools.builtin.weather_tools import OpenWeatherMapTool


class TestOpenWeatherMapTool:
    """Test cases for OpenWeatherMapTool."""

    @pytest.fixture
    def tool(self):
        """Create tool instance with test API key."""
        return OpenWeatherMapTool(api_key="test_api_key")

    @pytest.fixture
    def context(self):
        """Create execution context."""
        return ExecutionContext(
            user_id="test_user",
            agent_id="test_agent",
            request_id="test_request",
        )

    @pytest.mark.asyncio
    async def test_tool_initialization(self, tool):
        """Test tool initialization and metadata."""
        assert tool.metadata.tool_id == "openweathermap"
        assert tool.metadata.name == "OpenWeatherMap"
        assert tool.metadata.category.value == "api_client"
        assert len(tool.metadata.parameters) == 3

    @pytest.mark.asyncio
    async def test_parameter_validation_success(self, tool):
        """Test successful parameter validation."""
        params = {
            "location": "London",
            "units": "metric",
            "forecast_days": 3,
        }

        is_valid, error = await tool.validate_parameters(params)
        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_parameter_validation_missing_required(self, tool):
        """Test validation fails when required parameter missing."""
        params = {"units": "metric"}  # Missing 'location'

        is_valid, error = await tool.validate_parameters(params)
        assert is_valid is False
        assert "location" in error
        assert "Missing required parameter" in error

    @pytest.mark.asyncio
    async def test_parameter_validation_invalid_enum(self, tool):
        """Test validation fails for invalid enum value."""
        params = {
            "location": "London",
            "units": "fahrenheit",  # Invalid: not in enum
        }

        is_valid, error = await tool.validate_parameters(params)
        assert is_valid is False
        assert "units" in error
        assert "must be one of" in error

    @pytest.mark.asyncio
    async def test_execute_current_weather_success(self, tool, context):
        """Test successful current weather execution."""
        mock_response = {
            "main": {
                "temp": 15.5,
                "feels_like": 14.2,
                "humidity": 72,
                "pressure": 1013,
            },
            "weather": [{"description": "cloudy"}],
            "wind": {"speed": 5.2, "deg": 180},
            "dt": 1704067200,
        }

        with patch.object(tool, "_fetch_current", return_value=mock_response):
            params = {"location": "London", "units": "metric", "forecast_days": 0}

            result = await tool.execute(params, context)

            assert result.status == ToolExecutionStatus.SUCCESS
            assert result.result == mock_response
            assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_execute_with_forecast(self, tool, context):
        """Test execution with forecast days."""
        mock_current = {"temperature": 15.5}
        mock_forecast = [
            {"date": "2025-01-14", "temperature": 16.0},
            {"date": "2025-01-15", "temperature": 14.5},
        ]

        with patch.object(tool, "_fetch_current", return_value=mock_current), \
             patch.object(tool, "_fetch_forecast", return_value=mock_forecast):

            params = {"location": "London", "units": "metric", "forecast_days": 2}

            result = await tool.execute(params, context)

            assert result.status == ToolExecutionStatus.SUCCESS
            assert "current" in result.result
            assert "forecast" in result.result
            assert len(result.result["forecast"]) == 2

    @pytest.mark.asyncio
    async def test_execute_validation_failure(self, tool, context):
        """Test execution with validation failure."""
        params = {}  # Missing required 'location'

        result = await tool.execute(params, context)

        assert result.status == ToolExecutionStatus.FAILED
        assert result.error_type == "ValidationError"
        assert "location" in result.error

    @pytest.mark.asyncio
    async def test_execute_api_error(self, tool, context):
        """Test execution with API error."""
        import httpx

        with patch.object(
            tool,
            "_fetch_current",
            side_effect=httpx.HTTPStatusError(
                "Not found",
                request=AsyncMock(),
                response=AsyncMock(status_code=404),
            ),
        ):
            params = {"location": "InvalidCity", "units": "metric"}

            result = await tool.execute(params, context)

            assert result.status == ToolExecutionStatus.FAILED
            assert result.error_type == "HTTPError"
            assert "404" in result.error
```

## Parameter Validation

### Supported Parameter Types

The framework supports comprehensive parameter validation:

```python
ToolParameter(
    name="string_param",
    type="string",
    required=True,
    min_length=5,
    max_length=100,
    pattern=r"^[a-zA-Z0-9_-]+$"  # Regex validation
)

ToolParameter(
    name="integer_param",
    type="integer",
    min_value=0,
    max_value=100
)

ToolParameter(
    name="number_param",
    type="number",  # int or float
    min_value=0.0,
    max_value=1.0
)

ToolParameter(
    name="boolean_param",
    type="boolean"
)

ToolParameter(
    name="array_param",
    type="array",
    min_length=1,
    max_length=10
)

ToolParameter(
    name="object_param",
    type="object"
)

ToolParameter(
    name="enum_param",
    type="string",
    enum=["option1", "option2", "option3"]
)
```

### Custom Validation Logic

Add custom validation in your `execute()` method:

```python
async def execute(self, parameters: dict[str, Any], context: ExecutionContext) -> ToolResult:
    # Framework validation
    is_valid, error = await self.validate_parameters(parameters)
    if not is_valid:
        return ToolResult(...)

    # Custom business logic validation
    if parameters["start_date"] > parameters["end_date"]:
        return ToolResult(
            status=ToolExecutionStatus.FAILED,
            error="start_date must be before end_date",
            error_type="BusinessLogicError",
            ...
        )

    # Continue with execution
    ...
```

## Error Handling

### Error Handling Pattern

Always handle errors gracefully and return structured error information:

```python
async def execute(self, parameters: dict[str, Any], context: ExecutionContext) -> ToolResult:
    start_time = time.time()

    try:
        # Validation
        is_valid, error = await self.validate_parameters(parameters)
        if not is_valid:
            return self._error_result(
                context,
                "ValidationError",
                error,
                start_time
            )

        # Business logic
        result_data = await self._perform_operation(parameters)

        # Success
        return ToolResult(
            request_id=context.request_id,
            tool_id=self.metadata.tool_id,
            status=ToolExecutionStatus.SUCCESS,
            result=result_data,
            execution_time_ms=(time.time() - start_time) * 1000
        )

    except ValueError as e:
        return self._error_result(context, "ValueError", str(e), start_time)

    except httpx.HTTPError as e:
        return self._error_result(context, "HTTPError", str(e), start_time)

    except Exception as e:
        # Catch-all for unexpected errors
        self.logger.error(
            "unexpected_error",
            error=str(e),
            error_type=type(e).__name__,
            tool_id=self.metadata.tool_id,
            request_id=context.request_id
        )
        return self._error_result(context, type(e).__name__, str(e), start_time)

def _error_result(
    self,
    context: ExecutionContext,
    error_type: str,
    error_msg: str,
    start_time: float
) -> ToolResult:
    """Create error result with consistent format."""
    return ToolResult(
        request_id=context.request_id,
        tool_id=self.metadata.tool_id,
        status=ToolExecutionStatus.FAILED,
        error=error_msg,
        error_type=error_type,
        execution_time_ms=(time.time() - start_time) * 1000
    )
```

## Authentication

### Supported Auth Methods

Configure authentication in `ToolDefinition`:

```python
# API Key authentication
ToolDefinition(
    ...,
    auth_method=AuthMethod.API_KEY,
    auth_config={
        "api_key_header": "X-API-Key",      # Header name
        "api_key_param": "api_key",         # Query param name
        "api_key_env": "MY_SERVICE_API_KEY" # Environment variable
    }
)

# Bearer token
ToolDefinition(
    ...,
    auth_method=AuthMethod.BEARER_TOKEN,
    auth_config={
        "token_header": "Authorization",
        "token_env": "MY_SERVICE_TOKEN"
    }
)

# OAuth2
ToolDefinition(
    ...,
    auth_method=AuthMethod.OAUTH2,
    auth_config={
        "token_url": "https://api.example.com/oauth/token",
        "client_id_env": "OAUTH_CLIENT_ID",
        "client_secret_env": "OAUTH_CLIENT_SECRET",
        "scopes": ["read", "write"]
    }
)
```

### Handling Credentials in Tool

```python
async def execute(self, parameters: dict[str, Any], context: ExecutionContext) -> ToolResult:
    # Get API key from config or environment
    api_key = self.api_key or os.getenv(self.metadata.auth_config.get("api_key_env"))

    if not api_key:
        return ToolResult(
            status=ToolExecutionStatus.FAILED,
            error="API key not configured",
            error_type="AuthenticationError",
            ...
        )

    # Use API key in request
    headers = {"Authorization": f"Bearer {api_key}"}
    response = await client.get(url, headers=headers)
    ...
```

## Testing Your Tool

### Unit Test Structure

```python
import pytest
from unittest.mock import AsyncMock, patch

class TestYourTool:
    """Test suite for YourTool."""

    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return YourTool()

    @pytest.fixture
    def context(self):
        """Create execution context."""
        return ExecutionContext(
            user_id="test_user",
            agent_id="test_agent"
        )

    @pytest.mark.asyncio
    async def test_initialization(self, tool):
        """Test tool initialization."""
        assert tool.metadata.tool_id == "your_tool"
        assert len(tool.metadata.parameters) > 0

    @pytest.mark.asyncio
    async def test_parameter_validation_success(self, tool):
        """Test successful parameter validation."""
        params = {"param1": "value1"}
        is_valid, error = await tool.validate_parameters(params)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_execute_success(self, tool, context):
        """Test successful execution."""
        params = {"param1": "value1"}

        # Mock external dependencies
        with patch.object(tool, "_external_call", return_value="result"):
            result = await tool.execute(params, context)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result == "result"

    @pytest.mark.asyncio
    async def test_execute_error(self, tool, context):
        """Test execution with error."""
        params = {"param1": "value1"}

        # Mock error scenario
        with patch.object(tool, "_external_call", side_effect=ValueError("Error")):
            result = await tool.execute(params, context)

        assert result.status == ToolExecutionStatus.FAILED
        assert result.error_type == "ValueError"
```

### Integration Testing

```python
@pytest.mark.integration
async def test_tool_integration():
    """Test tool in full execution pipeline."""
    from agentcore.agent_runtime.tools.executor import ToolExecutor
    from agentcore.agent_runtime.tools.registry import ToolRegistry

    # Setup
    registry = ToolRegistry()
    await registry.register(YourTool())
    executor = ToolExecutor(registry=registry)

    # Execute
    context = ExecutionContext(user_id="test_user", agent_id="test_agent")
    result = await executor.execute_tool(
        tool_id="your_tool",
        parameters={"param1": "value1"},
        context=context
    )

    # Verify
    assert result.is_success
    assert result.result is not None
```

## Best Practices

### 1. Use Async/Await Consistently

```python
# Good
async def execute(self, parameters, context):
    result = await self._async_operation()
    return ToolResult(...)

async def _async_operation(self):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Bad: Blocking calls in async function
async def execute(self, parameters, context):
    result = requests.get(url)  # Blocking!
    return ToolResult(...)
```

### 2. Structured Logging

```python
# Good: Structured logging with context
self.logger.info(
    "operation_completed",
    tool_id=self.metadata.tool_id,
    request_id=context.request_id,
    user_id=context.user_id,
    execution_time_ms=execution_time,
    result_size=len(result)
)

# Bad: Unstructured string logging
self.logger.info(f"Operation completed in {execution_time}ms")
```

### 3. Resource Cleanup

```python
# Good: Use context managers
async def execute(self, parameters, context):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        # Client automatically closed

# Bad: Manual cleanup (easy to forget on error paths)
async def execute(self, parameters, context):
    client = httpx.AsyncClient()
    response = await client.get(url)
    await client.aclose()  # Might not run on exception
```

### 4. Idempotency

Make tools idempotent when possible:

```python
# Good: Idempotent operation
async def execute(self, parameters, context):
    # GET request is idempotent
    response = await client.get(url)

# For non-idempotent operations, document clearly
ToolDefinition(
    ...,
    is_idempotent=False,  # POST/PUT/DELETE operations
    description="Creates a new resource (non-idempotent)"
)
```

### 5. Error Messages

Provide actionable error messages:

```python
# Good: Specific, actionable error
return ToolResult(
    status=ToolExecutionStatus.FAILED,
    error="API key 'OPENWEATHER_API_KEY' not found in environment. "
          "Set the environment variable or pass api_key parameter.",
    error_type="ConfigurationError"
)

# Bad: Vague error
return ToolResult(
    status=ToolExecutionStatus.FAILED,
    error="Authentication failed",
    error_type="Error"
)
```

## Advanced Topics

### Rate Limiting

Configure rate limits in tool definition:

```python
ToolDefinition(
    ...,
    rate_limits={
        "calls_per_second": 10,
        "calls_per_minute": 60,
        "calls_per_hour": 1000,
        "calls_per_day": 10000
    }
)
```

### Cost Tracking

Track execution costs:

```python
ToolDefinition(
    ...,
    cost_per_execution=0.01,  # $0.01 per call
    metadata={
        "cost_currency": "USD",
        "cost_breakdown": {
            "api_call": 0.008,
            "data_processing": 0.002
        }
    }
)
```

### Streaming Results

For long-running operations, consider streaming:

```python
async def execute_stream(
    self,
    parameters: dict[str, Any],
    context: ExecutionContext
) -> AsyncIterator[dict[str, Any]]:
    """Stream results as they become available."""
    async for chunk in self._process_data(parameters):
        yield {
            "type": "chunk",
            "data": chunk,
            "timestamp": datetime.now().isoformat()
        }

    yield {
        "type": "complete",
        "total_chunks": self.chunk_count
    }
```

## Troubleshooting

### Common Issues

**1. Tool not found in registry**

```python
# Check registration
registry = ToolRegistry()
tools = await registry.list()
print([t.metadata.tool_id for t in tools])

# Re-register if missing
await registry.register(YourTool())
```

**2. Parameter validation failing unexpectedly**

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check actual vs expected types
params = {"param1": 123}
is_valid, error = await tool.validate_parameters(params)
print(f"Valid: {is_valid}, Error: {error}")
```

**3. Execution timeout**

```python
# Increase timeout
ToolDefinition(
    ...,
    timeout_seconds=60  # Increase from default 30
)

# Or handle long operations differently
async def execute(self, parameters, context):
    try:
        result = await asyncio.wait_for(
            self._long_operation(),
            timeout=self.metadata.timeout_seconds
        )
    except asyncio.TimeoutError:
        return ToolResult(
            status=ToolExecutionStatus.TIMEOUT,
            error="Operation exceeded timeout",
            ...
        )
```

## Next Steps

1. **Read the Architecture Guide:** Understand the framework design ([architecture.md](./architecture.md))
2. **Study Built-in Tools:** Review existing implementations in `src/agentcore/agent_runtime/tools/builtin/`
3. **Write Tests:** Comprehensive test coverage is required for all tools
4. **Submit Pull Request:** Follow the contribution guidelines in CLAUDE.md

## Additional Resources

- [README.md](./README.md) - Framework overview
- [architecture.md](./architecture.md) - Architecture documentation
- [Tool Integration Specification](../specs/tool-integration/spec.md)
- [CLAUDE.md](../../CLAUDE.md) - Development guidelines
