"""OpenAI LLM provider adapter."""

from __future__ import annotations

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]

from . import EnrichmentResponse


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
            # Use Pydantic structured output — the SDK enforces the schema
            # server-side, guaranteeing valid JSON matching EnrichmentResponse.
            kwargs["response_format"] = EnrichmentResponse
            response = self.client.beta.chat.completions.parse(**kwargs)
        else:
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
            "max_output_tokens": 16384,
        }
        if json_mode:
            kwargs["text_format"] = EnrichmentResponse

        try:
            response = self.client.responses.parse(**kwargs)
        except Exception:
            # Truncated or malformed JSON — fall back to create() so
            # _apply_stage can handle the error gracefully.
            response = self.client.responses.create(**kwargs)
        return response.output_text

    def complete_with_tools(
        self,
        messages: list[dict],
        search_backend,
        json_mode: bool = False,
        max_iterations: int = 10,
    ) -> str:
        """Run a tool-use loop using OpenAI function calling with Exa search."""
        import json as _json
        from ..search import WEB_SEARCH_TOOL_SCHEMA

        tools = [{
            "type": "function",
            "function": {
                "name": WEB_SEARCH_TOOL_SCHEMA["name"],
                "description": WEB_SEARCH_TOOL_SCHEMA["description"],
                "parameters": WEB_SEARCH_TOOL_SCHEMA["parameters"],
            },
        }]

        conv_messages = list(messages)

        kwargs: dict = {
            "model": self.model,
            "messages": conv_messages,
            "tools": tools,
        }
        if json_mode:
            kwargs["response_format"] = EnrichmentResponse

        for _ in range(max_iterations):
            if json_mode:
                response = self.client.beta.chat.completions.parse(**kwargs)
            else:
                response = self.client.chat.completions.create(**kwargs)

            choice = response.choices[0]
            msg = choice.message

            if not msg.tool_calls:
                return msg.content or ""

            # Append assistant message with tool calls
            conv_messages.append(msg)

            # Execute each tool call
            for tc in msg.tool_calls:
                if tc.function.name == "web_search":
                    args = _json.loads(tc.function.arguments)
                    query = args.get("query", "")
                    num = args.get("num_results", 5)
                    results = search_backend.search(query, num)
                    result_text = "\n\n".join(
                        f"### {r.title}\nURL: {r.url}\n{r.content}"
                        for r in results
                    )
                    conv_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text or "No results found.",
                    })

            kwargs["messages"] = conv_messages

        return ""
