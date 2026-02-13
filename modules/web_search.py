import logging
from typing import List, Optional
from duckduckgo_search import DDGS

logger = logging.getLogger("main")


def search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return formatted results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        A formatted string of search results, or an error message.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No results found for: {query}"

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = r.get("body", "No description")
            href = r.get("href", "")
            formatted.append(f"{i}. {title}\n   {body}\n   {href}")

        return "\n\n".join(formatted)

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Search error: {e}"


def search_news(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo news for a query.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        A formatted string of news results, or an error message.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))

        if not results:
            return f"No news found for: {query}"

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = r.get("body", "No description")
            source = r.get("source", "Unknown")
            date = r.get("date", "")
            url = r.get("url", "")
            formatted.append(
                f"{i}. [{source}] {title}\n   {body}\n   {date}\n   {url}"
            )

        return "\n\n".join(formatted)

    except Exception as e:
        logger.error(f"News search failed: {e}")
        return f"News search error: {e}"


def get_answer(query: str) -> str:
    """Get an instant answer from DuckDuckGo (if available).

    Args:
        query: The question or query string.

    Returns:
        The instant answer text, or a fallback web search.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.answers(query))

        if results:
            answer = results[0]
            text = answer.get("text", "")
            source = answer.get("source", "")
            url = answer.get("url", "")
            if text:
                return f"{text}\n(Source: {source} - {url})"

        # Fall back to regular search if no instant answer
        return search(query, max_results=3)

    except Exception as e:
        logger.error(f"Answer lookup failed: {e}")
        return f"Answer error: {e}"
