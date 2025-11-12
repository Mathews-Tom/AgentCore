"""Native Tool ABC implementations for search and web retrieval tools.

This module provides Tool ABC implementations for web search, Wikipedia queries,
and web scraping. These are native implementations that directly inherit from
Tool ABC (not legacy function-based tools).

Migration from: agent_runtime/tools/search_tools.py
Status: Stage 3 - Native Migration
"""

import os
import re
import time
from typing import Any

import httpx

from ...config.settings import get_settings
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

    Integrates with Google Custom Search API to perform web searches and
    return relevant results with title, URL, and snippet.

    Requires Google API key and Custom Search Engine ID from environment
    or settings. Falls back to mock results if API credentials not configured.
    """

    def __init__(self, api_key: str | None = None, cse_id: str | None = None):
        """Initialize Google search tool with metadata.

        Args:
            api_key: Google API key (optional, reads from settings if not provided)
            cse_id: Google Custom Search Engine ID (optional, reads from settings)
        """
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
                    max_value=10,
                ),
            },
            auth_method=AuthMethod.API_KEY,
            is_retryable=True,
            max_retries=3,
            timeout_seconds=30,
            is_idempotent=True,
            capabilities=["web_search", "external_api"],
            tags=["search", "google", "web"],
            rate_limits={"calls_per_minute": 100},
        )
        super().__init__(metadata)

        # Get API credentials from settings if not provided
        settings = get_settings()
        self.api_key = api_key or settings.google_api_key or os.getenv("GOOGLE_API_KEY", "")
        self.cse_id = cse_id or settings.google_cse_id or os.getenv("GOOGLE_CSE_ID", "")
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute Google search.

        Args:
            parameters: Dictionary with keys:
                - query: str - Search query
                - num_results: int - Number of results (default: 10, max: 10)
            context: Execution context

        Returns:
            ToolResult with search results including title, URL, and snippet
        """
        start_time = time.time()

        try:
            # Validate parameters
            is_valid, error = await self.validate_parameters(parameters)
            if not is_valid:
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    error=error,
                    error_type="ValidationError",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            query = parameters["query"]
            num_results = int(parameters.get("num_results", 10))

            self.logger.info(
                "google_search_executing",
                query=query,
                num_results=num_results,
                has_api_key=bool(self.api_key),
            )

            # Check if API credentials are configured
            if not self.api_key or not self.cse_id:
                self.logger.warning(
                    "google_api_not_configured",
                    message="Google API credentials not configured, using mock results",
                )
                results = self._get_mock_results(query, num_results)
                execution_time_ms = (time.time() - start_time) * 1000

                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.SUCCESS,
                    result={
                        "query": query,
                        "total_results": len(results),
                        "results": results,
                        "search_time_ms": execution_time_ms,
                        "provider": "Google Custom Search API (Mock)",
                    },
                    execution_time_ms=execution_time_ms,
                    metadata={
                        "trace_id": context.trace_id,
                        "agent_id": context.agent_id,
                        "using_mock": True,
                    },
                )

            # Call Google Custom Search API
            try:
                results = await self._call_google_api(query, num_results)
                execution_time_ms = (time.time() - start_time) * 1000

                self.logger.info(
                    "google_search_completed",
                    query=query,
                    result_count=len(results),
                    execution_time_ms=execution_time_ms,
                )

                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.SUCCESS,
                    result={
                        "query": query,
                        "total_results": len(results),
                        "results": results,
                        "search_time_ms": execution_time_ms,
                        "provider": "Google Custom Search API",
                    },
                    execution_time_ms=execution_time_ms,
                    metadata={
                        "trace_id": context.trace_id,
                        "agent_id": context.agent_id,
                        "using_mock": False,
                    },
                )

            except httpx.HTTPStatusError as e:
                execution_time_ms = (time.time() - start_time) * 1000
                error_msg = f"Google API error: {e.response.status_code} - {e.response.text}"
                self.logger.error(
                    "google_api_http_error",
                    status_code=e.response.status_code,
                    error=str(e),
                )

                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    error=error_msg,
                    error_type="HttpError",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            except httpx.TimeoutException as e:
                execution_time_ms = (time.time() - start_time) * 1000
                error_msg = f"Google API timeout: {str(e)}"
                self.logger.error("google_api_timeout", error=str(e))

                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.TIMEOUT,
                    error=error_msg,
                    error_type="TimeoutError",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "google_search_error",
                error=str(e),
                error_type=type(e).__name__,
                parameters=parameters,
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )

    async def _call_google_api(
        self,
        query: str,
        num_results: int,
    ) -> list[dict[str, Any]]:
        """Call Google Custom Search API.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of search results with title, url, snippet

        Raises:
            httpx.HTTPStatusError: On API HTTP errors
            httpx.TimeoutException: On API timeout
        """
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": min(num_results, 10),  # API max is 10 per request
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

        # Parse and format results
        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })

        return results

    def _get_mock_results(self, query: str, num_results: int) -> list[dict[str, Any]]:
        """Get mock search results for testing.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of mock search results
        """
        return [
            {
                "title": f"Result {i+1} for {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a snippet for result {i+1} about {query}",
            }
            for i in range(min(num_results, 10))
        ]


class WikipediaSearchTool(Tool):
    """Wikipedia search tool for knowledge retrieval.

    Searches Wikipedia and returns article summaries using the Wikipedia API.
    Supports article summary extraction with configurable sentence count and
    disambiguation handling for queries with multiple matches.
    """

    def __init__(self):
        """Initialize Wikipedia search tool with metadata."""
        metadata = ToolDefinition(
            tool_id="wikipedia_search",
            name="Wikipedia Search",
            description="Search Wikipedia and return article summaries with configurable detail",
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
                "sentences": ToolParameter(
                    name="sentences",
                    type="integer",
                    description="Number of sentences to include in article summary",
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
        self.api_base_url = "https://en.wikipedia.org/w/api.php"

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute Wikipedia search with article summary extraction.

        Args:
            parameters: Dictionary with keys:
                - query: str - Search query
                - sentences: int - Number of sentences in summary (default: 5)
            context: Execution context

        Returns:
            ToolResult with Wikipedia article summaries and metadata
        """
        start_time = time.time()

        try:
            # Validate parameters using base class validation
            is_valid, error = await self.validate_parameters(parameters)
            if not is_valid:
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    error=error,
                    error_type="ValidationError",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            query = parameters["query"]
            sentences = int(parameters.get("sentences", 5))

            self.logger.info(
                "wikipedia_search_executing",
                query=query,
                sentences=sentences,
            )

            # Search and extract summaries
            result_data = await self._search_and_extract(query, sentences)

            execution_time_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "wikipedia_search_completed",
                query=query,
                result_count=len(result_data.get("results", [])),
                disambiguation=result_data.get("disambiguation", False),
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.SUCCESS,
                result=result_data,
                execution_time_ms=execution_time_ms,
                metadata={
                    "trace_id": context.trace_id,
                    "agent_id": context.agent_id,
                },
            )

        except httpx.HTTPStatusError as e:
            execution_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Wikipedia API HTTP error: {e.response.status_code}"
            self.logger.error(
                "wikipedia_api_http_error",
                status_code=e.response.status_code,
                error=str(e),
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                error=error_msg,
                error_type="HttpError",
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )

        except httpx.TimeoutException as e:
            execution_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Wikipedia API timeout: {str(e)}"
            self.logger.error("wikipedia_api_timeout", error=str(e))

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.TIMEOUT,
                error=error_msg,
                error_type="TimeoutError",
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "wikipedia_search_error",
                error=str(e),
                error_type=type(e).__name__,
                parameters=parameters,
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )

    async def _search_and_extract(
        self,
        query: str,
        sentences: int,
    ) -> dict[str, Any]:
        """Search Wikipedia and extract article summaries.

        Args:
            query: Search query
            sentences: Number of sentences to extract from each article

        Returns:
            Dictionary with search results, summaries, and disambiguation info
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            # First, search for the article
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 1,
                "format": "json",
            }

            search_response = await client.get(self.api_base_url, params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()

            search_results = search_data.get("query", {}).get("search", [])

            if not search_results:
                # No results found
                return {
                    "query": query,
                    "total_results": 0,
                    "results": [],
                    "disambiguation": False,
                }

            # Get the page title from search results
            page_title = search_results[0]["title"]

            # Check if it's a disambiguation page
            is_disambiguation = "(disambiguation)" in page_title.lower()

            if is_disambiguation:
                # Get disambiguation options
                return await self._handle_disambiguation(client, page_title, query)

            # Extract article summary
            extract_params = {
                "action": "query",
                "prop": "extracts|info",
                "exintro": True,
                "explaintext": True,
                "exsentences": sentences,
                "inprop": "url",
                "titles": page_title,
                "format": "json",
            }

            extract_response = await client.get(self.api_base_url, params=extract_params)
            extract_response.raise_for_status()
            extract_data = extract_response.json()

            pages = extract_data.get("query", {}).get("pages", {})

            if not pages:
                return {
                    "query": query,
                    "total_results": 0,
                    "results": [],
                    "disambiguation": False,
                }

            # Get the first (and only) page
            page = next(iter(pages.values()))

            result = {
                "title": page.get("title", ""),
                "summary": page.get("extract", ""),
                "url": page.get("fullurl", ""),
                "page_id": page.get("pageid"),
            }

            return {
                "query": query,
                "total_results": 1,
                "results": [result],
                "disambiguation": False,
            }

    async def _handle_disambiguation(
        self,
        client: httpx.AsyncClient,
        page_title: str,
        original_query: str,
    ) -> dict[str, Any]:
        """Handle disambiguation pages by extracting alternative options.

        Args:
            client: HTTP client
            page_title: Title of the disambiguation page
            original_query: Original search query

        Returns:
            Dictionary with disambiguation options
        """
        # Get links from the disambiguation page
        links_params = {
            "action": "query",
            "prop": "links",
            "titles": page_title,
            "pllimit": 10,
            "format": "json",
        }

        links_response = await client.get(self.api_base_url, params=links_params)
        links_response.raise_for_status()
        links_data = links_response.json()

        pages = links_data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        links = page.get("links", [])

        # Format disambiguation options
        options = []
        for link in links[:10]:  # Limit to top 10 options
            title = link.get("title", "")
            # Skip Wikipedia meta pages
            if not title.startswith("Wikipedia:") and not title.startswith("Help:"):
                options.append({
                    "title": title,
                    "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                })

        return {
            "query": original_query,
            "total_results": len(options),
            "results": options,
            "disambiguation": True,
            "disambiguation_page": page_title,
        }


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
