"""
Tests for OpenAPI documentation completeness.

Verifies that all API endpoints have comprehensive documentation including
examples, descriptions, and proper response specifications.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gateway.main import create_app


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def openapi_schema(client: TestClient) -> dict:
    """Get OpenAPI schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


def test_openapi_metadata(openapi_schema: dict) -> None:
    """Test that OpenAPI metadata is complete."""
    assert "AgentCore" in openapi_schema["info"]["title"]
    assert "version" in openapi_schema["info"]
    assert "description" in openapi_schema["info"]
    assert openapi_schema["info"]["license"]["name"] == "Apache 2.0"
    assert "contact" in openapi_schema["info"]


def test_security_schemes_defined(openapi_schema: dict) -> None:
    """Test that security schemes are properly defined."""
    components = openapi_schema["components"]
    assert "securitySchemes" in components

    security_schemes = components["securitySchemes"]
    assert "BearerAuth" in security_schemes
    assert security_schemes["BearerAuth"]["type"] == "http"
    assert security_schemes["BearerAuth"]["scheme"] == "bearer"
    assert security_schemes["BearerAuth"]["bearerFormat"] == "JWT"

    assert "OAuth2" in security_schemes
    assert security_schemes["OAuth2"]["type"] == "oauth2"


def test_servers_defined(openapi_schema: dict) -> None:
    """Test that servers are defined."""
    assert "servers" in openapi_schema
    servers = openapi_schema["servers"]
    assert len(servers) > 0

    server_urls = [s["url"] for s in servers]
    assert "http://localhost:8080" in server_urls


def test_tags_defined(openapi_schema: dict) -> None:
    """Test that tags are defined with descriptions."""
    assert "tags" in openapi_schema
    tags = openapi_schema["tags"]

    tag_names = [t["name"] for t in tags]
    assert "health" in tag_names
    assert "authentication" in tag_names
    assert "oauth" in tag_names

    # All tags should have descriptions
    for tag in tags:
        assert "description" in tag
        assert len(tag["description"]) > 0


def test_auth_token_endpoint_documented(openapi_schema: dict) -> None:
    """Test that /auth/token endpoint is well-documented."""
    paths = openapi_schema["paths"]
    assert "/auth/token" in paths

    endpoint = paths["/auth/token"]["post"]

    # Check summary and description
    assert "summary" in endpoint
    assert "description" in endpoint
    assert len(endpoint["description"]) > 100  # Comprehensive description

    # Check responses
    assert "responses" in endpoint
    responses = endpoint["responses"]

    # Should have success and error responses
    assert "200" in responses
    assert "400" in responses
    assert "401" in responses
    assert "429" in responses

    # Success response should have example
    assert "content" in responses["200"]
    content = responses["200"]["content"]["application/json"]
    assert "example" in content or "examples" in content


def test_auth_refresh_endpoint_documented(openapi_schema: dict) -> None:
    """Test that /auth/refresh endpoint is well-documented."""
    paths = openapi_schema["paths"]
    assert "/auth/refresh" in paths

    endpoint = paths["/auth/refresh"]["post"]

    # Check documentation
    assert "summary" in endpoint
    assert "description" in endpoint
    assert len(endpoint["description"]) > 50

    # Check responses
    responses = endpoint["responses"]
    assert "200" in responses
    assert "401" in responses


def test_auth_me_endpoint_documented(openapi_schema: dict) -> None:
    """Test that /auth/me endpoint is well-documented."""
    paths = openapi_schema["paths"]
    assert "/auth/me" in paths

    endpoint = paths["/auth/me"]["get"]

    # Check documentation
    assert "summary" in endpoint
    assert "description" in endpoint

    # Check responses with examples
    responses = endpoint["responses"]
    assert "200" in responses
    assert "401" in responses

    # Should have example response
    success_response = responses["200"]
    assert "content" in success_response


def test_oauth_authorize_endpoint_documented(openapi_schema: dict) -> None:
    """Test that OAuth authorize endpoint is well-documented."""
    paths = openapi_schema["paths"]
    assert "/oauth/authorize/{provider}" in paths

    endpoint = paths["/oauth/authorize/{provider}"]["get"]

    # Check comprehensive documentation
    assert "summary" in endpoint
    assert "description" in endpoint
    assert len(endpoint["description"]) > 100

    # Check parameters - note that path parameters may be separate
    params = endpoint.get("parameters", [])
    param_names = [p["name"] for p in params]

    # Either in parameters or query params
    assert "scope" in param_names or any("scope" in str(p) for p in params)


def test_oauth_providers_endpoint_documented(openapi_schema: dict) -> None:
    """Test that OAuth providers endpoint is well-documented."""
    paths = openapi_schema["paths"]
    assert "/oauth/providers" in paths

    endpoint = paths["/oauth/providers"]["get"]

    # Check documentation
    assert "summary" in endpoint
    assert "description" in endpoint

    # Should have response documented
    responses = endpoint["responses"]
    assert "200" in responses


def test_health_endpoint_documented(openapi_schema: dict) -> None:
    """Test that health check endpoint is well-documented."""
    paths = openapi_schema["paths"]
    assert "/health" in paths

    endpoint = paths["/health"]["get"]

    # Check documentation
    assert "summary" in endpoint
    assert "description" in endpoint
    assert len(endpoint["description"]) > 100  # Should include monitoring examples

    # Check response examples
    responses = endpoint["responses"]
    assert "200" in responses
    success_response = responses["200"]
    assert "content" in success_response


def test_ready_endpoint_documented(openapi_schema: dict) -> None:
    """Test that readiness endpoint is well-documented."""
    paths = openapi_schema["paths"]
    assert "/ready" in paths

    endpoint = paths["/ready"]["get"]

    # Check documentation
    assert "summary" in endpoint
    assert "description" in endpoint

    # Should document Kubernetes integration
    description = endpoint["description"]
    assert "kubernetes" in description.lower() or "readiness" in description.lower()

    # Check both success and error responses
    responses = endpoint["responses"]
    assert "200" in responses
    assert "503" in responses


def test_all_endpoints_have_summaries(openapi_schema: dict) -> None:
    """Test that all endpoints have summaries."""
    paths = openapi_schema["paths"]

    for path, methods in paths.items():
        for method, endpoint in methods.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                assert "summary" in endpoint, f"{method.upper()} {path} missing summary"


def test_all_endpoints_have_descriptions(openapi_schema: dict) -> None:
    """Test that all endpoints have descriptions."""
    paths = openapi_schema["paths"]

    # Skip paths that are typically minimal (metrics, favicon, etc.)
    skip_paths = ["/metrics"]

    for path, methods in paths.items():
        if path in skip_paths:
            continue

        for method, endpoint in methods.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                assert (
                    "description" in endpoint
                ), f"{method.upper()} {path} missing description"


def test_protected_endpoints_have_security(openapi_schema: dict) -> None:
    """Test that protected endpoints specify security requirements."""
    paths = openapi_schema["paths"]

    # Public endpoints (no auth required)
    public_endpoints = [
        ("/health", "get"),
        ("/ready", "get"),
        ("/live", "get"),
        ("/auth/token", "post"),
        ("/oauth/providers", "get"),
        ("/oauth/scopes", "get"),
        ("/oauth/authorize/{provider}", "get"),
        ("/oauth/callback/{provider}", "get"),
    ]

    for path, methods in paths.items():
        for method, endpoint in methods.items():
            if method not in ["get", "post", "put", "delete", "patch"]:
                continue

            is_public = (path, method) in public_endpoints

            if not is_public and not path.startswith("/openapi"):
                # Protected endpoints should have security requirements
                # Either on endpoint level or global level
                has_security = "security" in endpoint
                assert has_security or "security" in openapi_schema, (
                    f"{method.upper()} {path} should specify security requirements"
                )


def test_error_responses_documented(openapi_schema: dict) -> None:
    """Test that endpoints document common error responses."""
    paths = openapi_schema["paths"]

    # Endpoints that don't need error documentation
    skip_endpoints = ["/health", "/live", "/metrics-info", "/openapi.json"]

    for path, methods in paths.items():
        # Skip certain paths
        if any(path.startswith(skip) or path == skip for skip in skip_endpoints):
            continue

        for method, endpoint in methods.items():
            if method not in ["get", "post", "put", "delete", "patch"]:
                continue

            responses = endpoint.get("responses", {})

            # Most endpoints should document error cases
            has_error_responses = any(
                code.startswith("4") or code.startswith("5") for code in responses
            )
            if not has_error_responses:
                # Some endpoints may legitimately not have error responses
                # but most protected endpoints should
                is_protected = "security" in endpoint or path.startswith("/auth")
                if is_protected and path != "/auth/token":
                    assert False, (
                        f"{method.upper()} {path} should document error responses"
                    )


def test_external_docs_defined(openapi_schema: dict) -> None:
    """Test that external documentation is defined."""
    assert "externalDocs" in openapi_schema
    ext_docs = openapi_schema["externalDocs"]
    assert "description" in ext_docs
    assert "url" in ext_docs


def test_response_examples_present(openapi_schema: dict) -> None:
    """Test that key endpoints have response examples."""
    paths = openapi_schema["paths"]

    # Key endpoints that should have examples
    key_endpoints = [
        ("/auth/token", "post"),
        ("/auth/me", "get"),
        ("/health", "get"),
        ("/oauth/providers", "get"),
    ]

    for path, method in key_endpoints:
        if path in paths and method in paths[path]:
            endpoint = paths[path][method]
            responses = endpoint.get("responses", {})

            # Check 200 response has example
            if "200" in responses:
                success_response = responses["200"]
                if "content" in success_response:
                    content = success_response["content"]
                    if "application/json" in content:
                        json_content = content["application/json"]
                        has_example = "example" in json_content or "examples" in json_content
                        assert has_example, (
                            f"{method.upper()} {path} should have response example"
                        )


def test_deprecated_endpoints_marked(openapi_schema: dict) -> None:
    """Test that deprecated endpoints are properly marked."""
    paths = openapi_schema["paths"]

    for path, methods in paths.items():
        for method, endpoint in methods.items():
            if method not in ["get", "post", "put", "delete", "patch"]:
                continue

            # If deprecated, should be clearly marked
            if endpoint.get("deprecated", False):
                # Should have description explaining what to use instead
                assert "description" in endpoint
                description = endpoint["description"].lower()
                assert (
                    "deprecated" in description or "use" in description
                ), f"{method.upper()} {path} marked deprecated but missing migration info"
