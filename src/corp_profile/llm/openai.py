"""OpenAI LLM provider adapter."""

from __future__ import annotations

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]


class OpenAIProvider:
    """LLM provider using OpenAI Chat Completions and Responses APIs."""

    def __init__(self, model: str) -> None:
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required for OpenAIProvider. "
                "Install it with: uv sync --extra openai"
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
        """Use OpenAI Responses API with web search tool."""
        kwargs: dict = {
            "model": self.model,
            "input": messages,
            "tools": [{"type": "web_search_preview"}],
        }
        if json_mode:
            kwargs["text"] = {"format": {"type": "json_object"}}

        response = self.client.responses.create(**kwargs)
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        return content.text
        return ""
