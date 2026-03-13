"""Search backend abstraction — Exa and OpenAI web search."""

from __future__ import annotations

import os
from typing import Protocol

from pydantic import BaseModel


class SearchResult(BaseModel):
    """A single web search result."""

    title: str
    url: str
    content: str


class SearchBackend(Protocol):
    """Protocol for web search backends."""

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]: ...


WEB_SEARCH_TOOL_SCHEMA: dict = {
    "name": "web_search",
    "description": (
        "Search the web for current information about a company, "
        "its operations, subsidiaries, or assets."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "num_results": {
                "type": "integer",
                "default": 5,
                "description": "Number of results to return (1-10)",
            },
        },
        "required": ["query"],
    },
}


class ExaSearch:
    """Web search using the Exa API."""

    def __init__(self, api_key: str) -> None:
        from exa_py import Exa

        self.client = Exa(api_key=api_key)

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        response = self.client.search_and_contents(
            query,
            num_results=min(num_results, 10),
            type="auto",
            text=True,
        )
        return [
            SearchResult(
                title=r.title or "",
                url=r.url or "",
                content=r.text or "",
            )
            for r in response.results
        ]


def get_search_backend(provider: str, model: str) -> SearchBackend | None:
    """Resolve provider string to a SearchBackend, or None for OpenAI built-in.

    "auto" uses the model slug to decide: openai/ models → None, else → Exa.
    """
    if provider == "openai":
        return None

    if provider == "auto":
        model_provider = model.split("/", 1)[0] if "/" in model else model
        if model_provider == "openai":
            return None
        # Fall through to Exa for non-OpenAI models

    # provider == "exa" or auto resolved to Exa
    api_key = os.environ.get("EXA_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "EXA_API_KEY environment variable is required when search provider is "
            f"'{provider}'. Set it in .env or your environment."
        )
    return ExaSearch(api_key=api_key)
