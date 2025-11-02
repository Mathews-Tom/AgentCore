"""Search tool adapters for web search and knowledge retrieval."""

import asyncio
import json
from typing import Any

import httpx
import structlog

from ..models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolParameter,
)
from ..services.tool_registry import ToolRegistry

logger = structlog.get_logger()


async def google_search(query: str, num_results: int = 10) -> dict[str, Any]:
    """
    Search Google and return results.

    Args:
        query: Search query
        num_results: Number of results to return (1-100)

    Returns:
        Dictionary with search results
    """
    # Note: In production, this would use Google Custom Search API
    # For now, we'll return a mock structure
    logger.info(
        "google_search_called",
        query=query,
        num_results=num_results,
    )

    # Mock response structure (in production, would call Google API)
    return {
        "query": query,
        "total_results": num_results,
        "results": [
            {
                "title": f"Result {i+1} for {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a snippet for result {i+1} about {query}",
            }
            for i in range(min(num_results, 10))
        ],
        "search_time_ms": 0.5,
    }


async def wikipedia_search(query: str, limit: int = 5) -> dict[str, Any]:
    """
    Search Wikipedia and return article summaries.

    Args:
        query: Search query
        limit: Maximum number of results (1-20)

    Returns:
        Dictionary with Wikipedia results
    """
    logger.info(
        "wikipedia_search_called",
        query=query,
        limit=limit,
    )

    try:
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
                return {
                    "query": query,
                    "results": [],
                    "error": "Invalid response from Wikipedia API",
                }

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

            return {
                "query": query,
                "total_results": len(results),
                "results": results,
            }

    except httpx.HTTPError as e:
        logger.error(
            "wikipedia_search_failed",
            query=query,
            error=str(e),
        )
        return {
            "query": query,
            "results": [],
            "error": f"Wikipedia API error: {str(e)}",
        }


async def web_scrape(url: str, extract_text: bool = True) -> dict[str, Any]:
    """
    Scrape content from a web URL.

    Args:
        url: URL to scrape
        extract_text: Whether to extract plain text (vs HTML)

    Returns:
        Dictionary with scraped content
    """
    logger.info(
        "web_scrape_called",
        url=url,
        extract_text=extract_text,
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            if extract_text and "text/html" in content_type:
                # Simple text extraction (in production, would use BeautifulSoup)
                html_content = response.text
                # Remove script and style tags (very basic)
                import re

                text_content = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL)
                text_content = re.sub(r"<style[^>]*>.*?</style>", "", text_content, flags=re.DOTALL)
                text_content = re.sub(r"<[^>]+>", " ", text_content)
                text_content = re.sub(r"\s+", " ", text_content).strip()

                return {
                    "url": url,
                    "content_type": content_type,
                    "text": text_content[:5000],  # Limit to 5000 chars
                    "length": len(text_content),
                }
            else:
                return {
                    "url": url,
                    "content_type": content_type,
                    "content": response.text[:5000],  # Limit to 5000 chars
                    "length": len(response.text),
                }

    except httpx.HTTPError as e:
        logger.error(
            "web_scrape_failed",
            url=url,
            error=str(e),
        )
        raise ValueError(f"Failed to scrape URL: {str(e)}")


def register_search_tools(registry: ToolRegistry) -> None:
    """
    Register search tools with the tool registry.

    Args:
        registry: ToolRegistry instance
    """
    # Google Search tool
    google_search_def = ToolDefinition(
        tool_id="google_search",
        name="google_search",
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
        timeout_seconds=30,
        is_idempotent=True,
        capabilities=["web_search", "external_api"],
        tags=["search", "google", "web"],
        metadata={
            "provider": "Google Custom Search API",
            "rate_limit": "100 queries per day",
        },
    )
    registry.register_tool(google_search_def, google_search)

    # Wikipedia Search tool
    wikipedia_search_def = ToolDefinition(
        tool_id="wikipedia_search",
        name="wikipedia_search",
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
        timeout_seconds=30,
        is_idempotent=True,
        capabilities=["knowledge_retrieval", "external_api"],
        tags=["search", "wikipedia", "knowledge", "encyclopedia"],
        metadata={
            "provider": "Wikipedia API",
            "api_url": "https://en.wikipedia.org/w/api.php",
        },
    )
    registry.register_tool(wikipedia_search_def, wikipedia_search)

    # Web Scrape tool
    web_scrape_def = ToolDefinition(
        tool_id="web_scrape",
        name="web_scrape",
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
        timeout_seconds=60,
        is_idempotent=True,
        capabilities=["web_scraping", "content_extraction"],
        tags=["scraping", "web", "content", "extraction"],
        metadata={
            "max_content_length": "5000 characters",
        },
    )
    registry.register_tool(web_scrape_def, web_scrape)

    logger.info(
        "search_tools_registered",
        tools=["google_search", "wikipedia_search", "web_scrape"],
    )
