"""Native Tool ABC implementations for search and web retrieval tools.

This module provides Tool ABC implementations for web search, Wikipedia queries,
and web scraping. These are native implementations that directly inherit from
Tool ABC (not legacy function-based tools).

Migration from: agent_runtime/tools/search_tools.py
Status: Stage 3 - Native Migration
"""

import re
import time
from typing import Any

import httpx

from ...models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolExecutionStatus,
    ToolParameter,
    ToolResult,
)
from ..base import ExecutionContext, Tool


class GoogleSearchTool(Tool):
    """Google search tool for web search operations.

    Performs Google searches and returns relevant results. In production,
    this would integrate with Google Custom Search API. Currently returns
    mock results for testing and development.
    """

    def __init__(self):
        """Initialize Google search tool with metadata."""
        metadata = ToolDefinition(
            tool_id="google_search",
            name="Google Search",
            description="Search Google and return relevant results",
            version="1.0.0",
            category=ToolCategory.SEARCH,
            parameters={
                "query": ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                    min_length=1,
                    max_length=500,
                ),
                "num_results": ToolParameter(
                    name="num_results",
                    type="number",
                    description="Number of results to return",
                    required=False,
                    default=10,
                    min_value=1,
                    max_value=100,
                ),
            },
            auth_method=AuthMethod.NONE,
            is_retryable=True,
            max_retries=3,
            timeout_seconds=30,
            is_idempotent=True,
            capabilities=["web_search", "external_api"],
            tags=["search", "google", "web"],
        )
        super().__init__(metadata)

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute Google search.

        Args:
            parameters: Dictionary with keys:
                - query: str - Search query
                - num_results: int - Number of results (default: 10)
            context: Execution context

        Returns:
            ToolResult with search results
        """
        start_time = time.time()

        try:
            query = parameters["query"]
            num_results = int(parameters.get("num_results", 10))

            # Validate parameters
            if not query or not query.strip():
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error="Query cannot be empty",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            if num_results < 1 or num_results > 100:
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error="num_results must be between 1 and 100",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            self.logger.info(
                "google_search_executing",
                query=query,
                num_results=num_results,
            )

            # Note: In production, this would use Google Custom Search API
            # For now, return mock structure for testing
            results = [
                {
                    "title": f"Result {i+1} for {query}",
                    "url": f"https://example.com/result{i+1}",
                    "snippet": f"This is a snippet for result {i+1} about {query}",
                }
                for i in range(min(num_results, 10))
            ]

            execution_time_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "google_search_completed",
                query=query,
                result_count=len(results),
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.SUCCESS,
                result={
                    "query": query,
                    "total_results": len(results),
                    "results": results,
                    "search_time_ms": 0.5,
                    "provider": "Google Custom Search API (Mock)",
                },
                error=None,
                execution_time_ms=execution_time_ms,
                metadata={
                    "trace_id": context.trace_id,
                    "agent_id": context.agent_id,
                },
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "google_search_error",
                error=str(e),
                parameters=parameters,
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                result={},
                error=str(e),
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )


class WikipediaSearchTool(Tool):
    """Wikipedia search tool for knowledge retrieval.

    Searches Wikipedia and returns article summaries using the Wikipedia API.
    """

    def __init__(self):
        """Initialize Wikipedia search tool with metadata."""
        metadata = ToolDefinition(
            tool_id="wikipedia_search",
            name="Wikipedia Search",
            description="Search Wikipedia and return article summaries",
            version="1.0.0",
            category=ToolCategory.SEARCH,
            parameters={
                "query": ToolParameter(
                    name="query",
                    type="string",
                    description="Wikipedia search query",
                    required=True,
                    min_length=1,
                    max_length=300,
                ),
                "limit": ToolParameter(
                    name="limit",
                    type="number",
                    description="Maximum number of results",
                    required=False,
                    default=5,
                    min_value=1,
                    max_value=20,
                ),
            },
            auth_method=AuthMethod.NONE,
            is_retryable=True,
            max_retries=3,
            timeout_seconds=30,
            is_idempotent=True,
            capabilities=["knowledge_retrieval", "external_api"],
            tags=["search", "wikipedia", "knowledge", "encyclopedia"],
        )
        super().__init__(metadata)

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute Wikipedia search.

        Args:
            parameters: Dictionary with keys:
                - query: str - Search query
                - limit: int - Maximum number of results (default: 5)
            context: Execution context

        Returns:
            ToolResult with Wikipedia search results
        """
        start_time = time.time()

        try:
            query = parameters["query"]
            limit = int(parameters.get("limit", 5))

            # Validate parameters
            if not query or not query.strip():
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error="Query cannot be empty",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            if limit < 1 or limit > 20:
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error="limit must be between 1 and 20",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            self.logger.info(
                "wikipedia_search_executing",
                query=query,
                limit=limit,
            )

            # Use Wikipedia API
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Search for articles
                search_params = {
                    "action": "opensearch",
                    "search": query,
                    "limit": limit,
                    "format": "json",
                }

                response = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params=search_params,
                )
                response.raise_for_status()

                search_data = response.json()

                if len(search_data) < 4:
                    execution_time_ms = (time.time() - start_time) * 1000
                    return ToolResult(
                        request_id=context.request_id,
                        tool_id=self.metadata.tool_id,
                        status=ToolExecutionStatus.FAILED,
                        result={},
                        error="Invalid response from Wikipedia API",
                        execution_time_ms=execution_time_ms,
                        metadata={"trace_id": context.trace_id},
                    )

                titles = search_data[1]
                descriptions = search_data[2]
                urls = search_data[3]

                results = []
                for i in range(len(titles)):
                    results.append(
                        {
                            "title": titles[i],
                            "description": descriptions[i],
                            "url": urls[i],
                        }
                    )

                execution_time_ms = (time.time() - start_time) * 1000

                self.logger.info(
                    "wikipedia_search_completed",
                    query=query,
                    result_count=len(results),
                )

                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.SUCCESS,
                    result={
                        "query": query,
                        "total_results": len(results),
                        "results": results,
                        "api_url": "https://en.wikipedia.org/w/api.php",
                    },
                    error=None,
                    execution_time_ms=execution_time_ms,
                    metadata={
                        "trace_id": context.trace_id,
                        "agent_id": context.agent_id,
                    },
                )

        except httpx.HTTPError as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "wikipedia_search_error",
                error=str(e),
                parameters=parameters,
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                result={},
                error=f"Wikipedia API error: {str(e)}",
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "wikipedia_search_error",
                error=str(e),
                parameters=parameters,
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                result={},
                error=str(e),
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )


class WebScrapeTool(Tool):
    """Web scraping tool for content extraction from URLs.

    Fetches content from a web URL and optionally extracts plain text from HTML.
    """

    def __init__(self):
        """Initialize web scrape tool with metadata."""
        metadata = ToolDefinition(
            tool_id="web_scrape",
            name="Web Scrape",
            description="Scrape content from a web URL",
            version="1.0.0",
            category=ToolCategory.DATA_PROCESSING,
            parameters={
                "url": ToolParameter(
                    name="url",
                    type="string",
                    description="URL to scrape",
                    required=True,
                    min_length=10,
                    max_length=2000,
                ),
                "extract_text": ToolParameter(
                    name="extract_text",
                    type="boolean",
                    description="Extract plain text from HTML",
                    required=False,
                    default=True,
                ),
            },
            auth_method=AuthMethod.NONE,
            is_retryable=True,
            max_retries=3,
            timeout_seconds=60,
            is_idempotent=True,
            capabilities=["web_scraping", "content_extraction"],
            tags=["scraping", "web", "content", "extraction"],
        )
        super().__init__(metadata)

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute web scraping.

        Args:
            parameters: Dictionary with keys:
                - url: str - URL to scrape
                - extract_text: bool - Extract plain text from HTML (default: True)
            context: Execution context

        Returns:
            ToolResult with scraped content
        """
        start_time = time.time()

        try:
            url = parameters["url"]
            extract_text = parameters.get("extract_text", True)

            # Validate URL
            if not url or not url.strip():
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error="URL cannot be empty",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            self.logger.info(
                "web_scrape_executing",
                url=url,
                extract_text=extract_text,
            )

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")

                if extract_text and "text/html" in content_type:
                    # Simple text extraction (in production, would use BeautifulSoup)
                    html_content = response.text
                    # Remove script and style tags (very basic)
                    text_content = re.sub(
                        r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL
                    )
                    text_content = re.sub(
                        r"<style[^>]*>.*?</style>", "", text_content, flags=re.DOTALL
                    )
                    text_content = re.sub(r"<[^>]+>", " ", text_content)
                    text_content = re.sub(r"\s+", " ", text_content).strip()

                    execution_time_ms = (time.time() - start_time) * 1000

                    self.logger.info(
                        "web_scrape_completed",
                        url=url,
                        content_length=len(text_content),
                    )

                    return ToolResult(
                        request_id=context.request_id,
                        tool_id=self.metadata.tool_id,
                        status=ToolExecutionStatus.SUCCESS,
                        result={
                            "url": url,
                            "content_type": content_type,
                            "text": text_content[:5000],  # Limit to 5000 chars
                            "length": len(text_content),
                            "truncated": len(text_content) > 5000,
                        },
                        error=None,
                        execution_time_ms=execution_time_ms,
                        metadata={
                            "trace_id": context.trace_id,
                            "agent_id": context.agent_id,
                        },
                    )
                else:
                    execution_time_ms = (time.time() - start_time) * 1000

                    self.logger.info(
                        "web_scrape_completed",
                        url=url,
                        content_length=len(response.text),
                    )

                    return ToolResult(
                        request_id=context.request_id,
                        tool_id=self.metadata.tool_id,
                        status=ToolExecutionStatus.SUCCESS,
                        result={
                            "url": url,
                            "content_type": content_type,
                            "content": response.text[:5000],  # Limit to 5000 chars
                            "length": len(response.text),
                            "truncated": len(response.text) > 5000,
                        },
                        error=None,
                        execution_time_ms=execution_time_ms,
                        metadata={
                            "trace_id": context.trace_id,
                            "agent_id": context.agent_id,
                        },
                    )

        except httpx.HTTPError as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "web_scrape_error",
                url=url,
                error=str(e),
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                result={},
                error=f"Failed to scrape URL: {str(e)}",
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "web_scrape_error",
                error=str(e),
                parameters=parameters,
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                result={},
                error=str(e),
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )
