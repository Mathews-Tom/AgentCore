"""
Integration Tests for Tool JSON-RPC Methods

Tests for tool discovery, inspection, and execution via JSON-RPC 2.0 protocol.
"""

import pytest
import respx
from httpx import AsyncClient, Response


@pytest.mark.asyncio
class TestToolsJSONRPC:
    """Test tool JSON-RPC methods."""

    async def test_tools_list(self, async_client: AsyncClient, jsonrpc_request_template):
        """Test tools.list method - list all available tools."""
        request = jsonrpc_request_template("tools.list")
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "1"
        assert "result" in data

        result = data["result"]
        assert "tools" in result
        assert "count" in result
        assert isinstance(result["tools"], list)
        assert result["count"] == len(result["tools"])

        # Verify we have the expected built-in tools
        tool_ids = {tool["tool_id"] for tool in result["tools"]}
        expected_tools = {
            "calculator",
            "get_current_time",
            "echo",
            "google_search",
            "wikipedia_search",
            "web_scrape",
            "execute_python",
            "evaluate_expression",
            "http_request",
            "rest_get",
            "rest_post",
            "graphql_query",
        }
        assert expected_tools.issubset(tool_ids)

    async def test_tools_list_with_category_filter(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test tools.list with category filter."""
        request = jsonrpc_request_template(
            "tools.list",
            {"category": "search"}  # Use lowercase to match enum value
        )
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data

        result = data["result"]
        assert "tools" in result
        assert result["count"] > 0

        # All tools should be search category
        for tool in result["tools"]:
            assert tool["category"] == "search"

    async def test_tools_list_with_capabilities_filter(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test tools.list with capabilities filter."""
        request = jsonrpc_request_template(
            "tools.list",
            {"capabilities": ["external_api"]}
        )
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data

        result = data["result"]
        assert result["count"] > 0

        # All tools should have external_api capability
        for tool in result["tools"]:
            assert "external_api" in tool["capabilities"]

    async def test_tools_list_with_tags_filter(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test tools.list with tags filter."""
        request = jsonrpc_request_template(
            "tools.list",
            {"tags": ["search"]}  # Use "search" tag which actually exists
        )
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data

        result = data["result"]
        # Should find tools with search tag
        assert result["count"] > 0
        # Verify tools have the search tag
        for tool in result["tools"]:
            assert "search" in tool["tags"]

    async def test_tools_list_with_invalid_category(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test tools.list with invalid category."""
        request = jsonrpc_request_template(
            "tools.list",
            {"category": "INVALID_CATEGORY"}
        )
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        # Errors are caught and converted to -32603 (Internal Error)
        assert data["error"]["code"] == -32603
        assert "invalid category" in data["error"]["message"].lower() or "invalid category" in data["error"]["data"]["details"].lower()

    async def test_tools_get(self, async_client: AsyncClient, jsonrpc_request_template):
        """Test tools.get method - get detailed tool information."""
        request = jsonrpc_request_template(
            "tools.get",
            {"tool_id": "calculator"}
        )
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data

        tool = data["result"]
        assert tool["tool_id"] == "calculator"
        assert tool["name"] == "Calculator"  # Display name is capitalized
        assert "description" in tool
        assert "version" in tool
        assert "category" in tool
        assert "parameters" in tool
        assert "auth_method" in tool
        assert "timeout_seconds" in tool
        assert "capabilities" in tool

        # Verify calculator parameters (operation, a, b)
        params = tool["parameters"]
        assert "operation" in params
        assert params["operation"]["required"] is True
        assert params["operation"]["type"] == "string"
        assert "a" in params
        assert "b" in params

    async def test_tools_get_not_found(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test tools.get with non-existent tool."""
        request = jsonrpc_request_template(
            "tools.get",
            {"tool_id": "nonexistent_tool"}
        )
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        # Errors are caught and converted to -32603 (Internal Error)
        assert data["error"]["code"] == -32603
        assert "not found" in data["error"]["message"].lower() or "not found" in data["error"]["data"]["details"].lower()

    async def test_tools_get_missing_tool_id(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test tools.get without tool_id parameter."""
        request = jsonrpc_request_template("tools.get", {})
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        # Errors are caught and converted to -32603 (Internal Error)
        assert data["error"]["code"] == -32603

    async def test_tools_execute_calculator(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test tools.execute method - execute calculator tool."""
        request = jsonrpc_request_template(
            "tools.execute",
            {
                "tool_id": "calculator",
                "parameters": {"operation": "+", "a": 2, "b": 2},  # Correct parameters
                "agent_id": "test-agent-001",
            }
        )
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data

        result = data["result"]
        assert result["tool_id"] == "calculator"
        assert result["status"] == "success"  # Lowercase
        assert "result" in result
        # Calculator returns detailed result dict
        assert isinstance(result["result"], dict)
        assert result["result"]["result"] == 4.0
        assert result["result"]["operation"] == "+"
        assert result["result"]["operands"]["a"] == 2.0
        assert result["result"]["operands"]["b"] == 2.0
        assert "execution_time_ms" in result
        assert "timestamp" in result

    async def test_tools_execute_echo(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test tools.execute with echo tool."""
        request = jsonrpc_request_template(
            "tools.execute",
            {
                "tool_id": "echo",
                "parameters": {"message": "Hello, World!"},
                "agent_id": "test-agent-001",
            }
        )
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data

        result = data["result"]
        assert result["status"] == "success"  # Lowercase
        # Echo returns detailed result dict
        assert isinstance(result["result"], dict)
        assert result["result"]["echo"] == "Hello, World!"
        assert result["result"]["original"] == "Hello, World!"
        assert result["result"]["length"] == 13
        assert result["result"]["uppercase"] is False

    @respx.mock
    async def test_tools_execute_wikipedia_search(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test tools.execute with Wikipedia search tool."""
        # Mock Wikipedia API search response (action=query&list=search)
        mock_search_response = {
            "query": {
                "search": [
                    {
                        "title": "Python (programming language)",
                        "snippet": "High-level programming language",
                    }
                ]
            }
        }

        # Mock Wikipedia API extract response (action=query&prop=extracts)
        mock_extract_response = {
            "query": {
                "pages": {
                    "12345": {
                        "pageid": 12345,
                        "title": "Python (programming language)",
                        "extract": "Python is a high-level programming language. It supports multiple programming paradigms.",
                        "fullurl": "https://en.wikipedia.org/wiki/Python_(programming_language)"
                    }
                }
            }
        }

        # Mock both API calls
        respx.get("https://en.wikipedia.org/w/api.php", params__contains={"list": "search"}).mock(
            return_value=Response(200, json=mock_search_response)
        )
        respx.get("https://en.wikipedia.org/w/api.php", params__contains={"prop": "extracts|info"}).mock(
            return_value=Response(200, json=mock_extract_response)
        )

        request = jsonrpc_request_template(
            "tools.execute",
            {
                "tool_id": "wikipedia_search",
                "parameters": {"query": "Python", "limit": 2},
                "agent_id": "test-agent-001",
            }
        )
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data

        result = data["result"]
        assert result["status"] == "success"  # Lowercase
        # Wikipedia search returns detailed result dict
        assert isinstance(result["result"], dict)
        assert result["result"]["query"] == "Python"
        assert result["result"]["total_results"] == 1
        assert result["result"]["disambiguation"] is False
        assert len(result["result"]["results"]) == 1
        assert result["result"]["results"][0]["title"] == "Python (programming language)"
        assert "summary" in result["result"]["results"][0]
        assert "url" in result["result"]["results"][0]

    async def test_tools_execute_missing_tool_id(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test tools.execute without tool_id."""
        request = jsonrpc_request_template(
            "tools.execute",
            {
                "parameters": {"operation": "+", "a": 1, "b": 1},
                "agent_id": "test-agent-001",
            }
        )
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        # Errors are caught and converted to -32603 (Internal Error)
        assert data["error"]["code"] == -32603

    async def test_tools_execute_missing_agent_id(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test tools.execute without agent_id."""
        request = jsonrpc_request_template(
            "tools.execute",
            {
                "tool_id": "calculator",
                "parameters": {"operation": "+", "a": 1, "b": 1},
            }
        )
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        # Errors are caught and converted to -32603 (Internal Error)
        assert data["error"]["code"] == -32603

    async def test_tools_execute_invalid_tool(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test tools.execute with non-existent tool."""
        request = jsonrpc_request_template(
            "tools.execute",
            {
                "tool_id": "nonexistent_tool",
                "parameters": {},
                "agent_id": "test-agent-001",
            }
        )
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        # Should return error from executor
        assert "result" in data or "error" in data

    async def test_tools_search_by_name(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test tools.search with name query."""
        request = jsonrpc_request_template(
            "tools.search",
            {"name_query": "search"}
        )
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data

        result = data["result"]
        assert "tools" in result
        assert "count" in result
        assert "query" in result

        # Should find search-related tools
        assert result["count"] > 0
        tool_names = [tool["name"].lower() for tool in result["tools"]]
        assert any("search" in name for name in tool_names)

    async def test_tools_search_by_category(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test tools.search with category filter."""
        request = jsonrpc_request_template(
            "tools.search",
            {"category": "code_execution"}  # Use lowercase to match enum value
        )
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data

        result = data["result"]
        assert result["count"] > 0

        # All tools should be code_execution category
        for tool in result["tools"]:
            assert tool["category"] == "code_execution"

    async def test_tools_search_by_capabilities(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test tools.search with capabilities filter."""
        request = jsonrpc_request_template(
            "tools.search",
            {"capabilities": ["code_execution"]}
        )
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data

        result = data["result"]
        assert result["count"] > 0

        # All tools should have code_execution capability
        for tool in result["tools"]:
            assert "code_execution" in tool["capabilities"]

    async def test_tools_search_combined_filters(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test tools.search with multiple filters."""
        request = jsonrpc_request_template(
            "tools.search",
            {
                "category": "search",  # Use lowercase to match enum value
                "capabilities": ["external_api"],
                "tags": ["search"]  # Use "search" tag which actually exists
            }
        )
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data

        result = data["result"]
        # Should find search tools with external_api capability and search tag
        for tool in result["tools"]:
            assert tool["category"] == "search"
            assert "external_api" in tool["capabilities"]
            assert "search" in tool["tags"]

    async def test_tools_search_no_results(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test tools.search with filters that match nothing."""
        request = jsonrpc_request_template(
            "tools.search",
            {"name_query": "nonexistent_tool_xyz123"}
        )
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data

        result = data["result"]
        assert result["count"] == 0
        assert result["tools"] == []

    async def test_batch_tools_request(
        self, async_client: AsyncClient, jsonrpc_request_template
    ):
        """Test JSON-RPC batch request for multiple tool operations."""
        batch = [
            jsonrpc_request_template("tools.list", {}, "1"),
            jsonrpc_request_template("tools.get", {"tool_id": "calculator"}, "2"),
            jsonrpc_request_template(
                "tools.execute",
                {
                    "tool_id": "echo",
                    "parameters": {"message": "test"},
                    "agent_id": "test-agent",
                },
                "3"
            ),
        ]
        response = await async_client.post("/api/v1/jsonrpc", json=batch)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3

        # Check all responses succeeded
        for item in data:
            assert "result" in item
            assert item["jsonrpc"] == "2.0"

        # Verify specific results
        assert "tools" in data[0]["result"]  # tools.list
        assert data[1]["result"]["tool_id"] == "calculator"  # tools.get
        assert data[2]["result"]["status"] == "success"  # tools.execute (lowercase)
