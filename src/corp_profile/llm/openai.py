"""OpenAI LLM provider adapter."""

from __future__ import annotations

from pydantic import BaseModel

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]

from ..profile import CompanyProfile


class EnrichmentResponse(BaseModel):
    """Expected response shape from all enrichment stages."""

    profile: CompanyProfile
    changes: list[str] = []


class OpenAIProvider:
    """LLM provider using OpenAI Chat Completions and Responses APIs."""

    def __init__(self, model: str) -> None:
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required for OpenAIProvider. "
                "Install it with: uv sync --extra llm"
            )
        self.model = model
        self.client = OpenAI()

    def complete(
        self,
        messages: list[dict],
        json_mode: bool = False,
        web_search: bool = False,
    ) -> str:
        if web_search:
            return self._complete_with_search(messages, json_mode)

        kwargs: dict = {"model": self.model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _complete_with_search(
        self, messages: list[dict], json_mode: bool
    ) -> str:
        """Use OpenAI Responses API with web search tool.

        Uses responses.parse() with a Pydantic model for structured output.
        This works with web search (unlike json_object mode which OpenAI blocks).
        The SDK handles schema generation with additionalProperties automatically.
        """
        kwargs: dict = {
            "model": self.model,
            "input": messages,
            "tools": [{"type": "web_search_preview"}],
        }
        if json_mode:
            kwargs["text_format"] = EnrichmentResponse

        response = self.client.responses.parse(**kwargs)
        return response.output_text
