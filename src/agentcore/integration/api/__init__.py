"""API Integration Framework.

RESTful API connector framework with authentication, rate limiting, retry logic,
and response transformation. Supports multiple authentication schemes and provides
comprehensive error handling.

Example usage:
    ```python
    from agentcore.integration.api import (
        APIClient,
        APIConfig,
        AuthConfig,
        AuthScheme,
        load_api_config,
    )

    # Load configuration from environment
    config = load_api_config("my-api", base_url="https://api.example.com")

    # Or create configuration manually
    config = APIConfig(
        name="my-api",
        base_url="https://api.example.com",
        auth=AuthConfig(
            scheme=AuthScheme.BEARER,
            token="your-token",
        ),
    )

    # Create client
    async with APIClient(config) as client:
        # Make requests
        response = await client.get("/users")
        print(response.body)

        response = await client.post("/users", body={"name": "John"})
        print(response.status_code)
    ```

For custom connectors:
    ```python
    from agentcore.integration.api import APIConnector, APIConfig

    class MyAPIConnector(APIConnector):
        async def _perform_health_check(self) -> bool:
            client = self.get_client()
            response = await client.get("/health")
            return response.status_code == 200

    # Use connector
    config = load_api_config("my-api")
    async with MyAPIConnector(config) as connector:
        client = connector.get_client()
        response = await client.get("/data")
    ```
"""

from agentcore.integration.api.client import APIClient
from agentcore.integration.api.config import load_api_config
from agentcore.integration.api.connector import (
    APIConnector,
    ConnectorStatus,
    HealthCheckResult,
    RestAPIConnector,
    get_connector,
    list_connectors,
    register_connector,
)
from agentcore.integration.api.exceptions import (
    APIAuthenticationError,
    APIAuthorizationError,
    APIConfigurationError,
    APIConnectionError,
    APIError,
    APINotFoundError,
    APIRateLimitError,
    APIServerError,
    APITimeoutError,
    APITransformationError,
    APIValidationError,
)
from agentcore.integration.api.models import (
    APIConfig,
    APIRequest,
    APIResponse,
    AuthConfig,
    AuthScheme,
    HTTPMethod,
    RateLimitConfig,
    ResponseTransformation,
    RetryConfig,
    TransformationRule,
)
from agentcore.integration.api.rate_limiter import RateLimiter, get_rate_limiter
from agentcore.integration.api.transformer import (
    ResponseTransformer,
    get_default_transformer,
)

__all__ = [
    # Client
    "APIClient",
    # Configuration
    "APIConfig",
    "AuthConfig",
    "AuthScheme",
    "RateLimitConfig",
    "RetryConfig",
    "load_api_config",
    # Models
    "APIRequest",
    "APIResponse",
    "HTTPMethod",
    "ResponseTransformation",
    "TransformationRule",
    # Connector
    "APIConnector",
    "RestAPIConnector",
    "ConnectorStatus",
    "HealthCheckResult",
    "register_connector",
    "get_connector",
    "list_connectors",
    # Rate Limiting
    "RateLimiter",
    "get_rate_limiter",
    # Transformation
    "ResponseTransformer",
    "get_default_transformer",
    # Exceptions
    "APIError",
    "APIConnectionError",
    "APITimeoutError",
    "APIAuthenticationError",
    "APIAuthorizationError",
    "APIRateLimitError",
    "APIValidationError",
    "APINotFoundError",
    "APIServerError",
    "APITransformationError",
    "APIConfigurationError",
]
