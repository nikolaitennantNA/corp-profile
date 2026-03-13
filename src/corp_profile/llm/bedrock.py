"""Amazon Bedrock LLM provider adapter."""

from __future__ import annotations

import json

try:
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]

from . import EnrichmentResponse


class BedrockProvider:
    """LLM provider using Amazon Bedrock Converse API."""

    def __init__(self, model: str, region: str | None = None, profile: str | None = None) -> None:
        if boto3 is None:
            raise ImportError(
                "The 'boto3' package is required for BedrockProvider. "
                "Install it with: uv sync --extra llm"
            )
        self.model = model
        session = boto3.Session(
            region_name=region,
            profile_name=profile,
        )
        self.client = session.client("bedrock-runtime")

    def complete(
        self,
        messages: list[dict],
        json_mode: bool = False,
        web_search: bool = False,
    ) -> str:
        if web_search:
            raise NotImplementedError(
                "Bedrock does not support web search. "
                "Set CORPPROFILE_WEB_SEARCH_MODEL to an OpenAI model slug."
            )

        system_parts: list[dict] = []
        converse_messages: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append({"text": msg["content"]})
            else:
                converse_messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}],
                })

        kwargs: dict = {
            "modelId": self.model,
            "messages": converse_messages,
            "inferenceConfig": {"maxTokens": 16384},
        }
        if system_parts:
            kwargs["system"] = system_parts

        if json_mode:
            # Use tool-use to enforce the EnrichmentResponse schema.
            # Force the model to call the tool, guaranteeing structured output.
            schema = EnrichmentResponse.model_json_schema()
            kwargs["toolConfig"] = {
                "tools": [{
                    "toolSpec": {
                        "name": "enrichment_response",
                        "description": "Return the enriched company profile and list of changes made.",
                        "inputSchema": {"json": schema},
                    }
                }],
                "toolChoice": {"tool": {"name": "enrichment_response"}},
            }

        response = self.client.converse(**kwargs)

        # Extract result: tool-use input (json_mode) or plain text
        for block in response["output"]["message"]["content"]:
            if "toolUse" in block:
                return json.dumps(block["toolUse"]["input"])
        return response["output"]["message"]["content"][0]["text"]

    def complete_with_tools(
        self,
        messages: list[dict],
        search_backend,
        json_mode: bool = False,
        max_iterations: int = 10,
    ) -> str:
        """Run a tool-use loop: LLM calls search → we execute → return results → repeat."""
        from ..search import WEB_SEARCH_TOOL_SCHEMA

        # Build Bedrock tool config
        tools = [{
            "toolSpec": {
                "name": WEB_SEARCH_TOOL_SCHEMA["name"],
                "description": WEB_SEARCH_TOOL_SCHEMA["description"],
                "inputSchema": {"json": WEB_SEARCH_TOOL_SCHEMA["parameters"]},
            }
        }]

        # Build initial messages
        system_parts: list[dict] = []
        converse_messages: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append({"text": msg["content"]})
            else:
                converse_messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}],
                })

        if json_mode:
            schema = EnrichmentResponse.model_json_schema()
            tools.append({
                "toolSpec": {
                    "name": "enrichment_response",
                    "description": "Return the enriched company profile and list of changes made.",
                    "inputSchema": {"json": schema},
                }
            })

        kwargs: dict = {
            "modelId": self.model,
            "messages": converse_messages,
            "inferenceConfig": {"maxTokens": 16384},
            "toolConfig": {"tools": tools},
        }
        if system_parts:
            kwargs["system"] = system_parts

        for _ in range(max_iterations):
            response = self.client.converse(**kwargs)
            content_blocks = response["output"]["message"]["content"]

            # Check for tool calls
            tool_calls = [b for b in content_blocks if "toolUse" in b]
            if not tool_calls:
                # No tool calls — return text
                for block in content_blocks:
                    if "text" in block:
                        return block["text"]
                return ""

            # Append assistant message
            converse_messages.append({
                "role": "assistant",
                "content": content_blocks,
            })

            # Process each tool call
            tool_results = []
            for tc in tool_calls:
                tool_use = tc["toolUse"]
                if tool_use["name"] == "web_search":
                    query = tool_use["input"].get("query", "")
                    num = tool_use["input"].get("num_results", 5)
                    results = search_backend.search(query, num)
                    result_text = "\n\n".join(
                        f"### {r.title}\nURL: {r.url}\n{r.content}"
                        for r in results
                    )
                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool_use["toolUseId"],
                            "content": [{"text": result_text or "No results found."}],
                        }
                    })
                elif tool_use["name"] == "enrichment_response":
                    return json.dumps(tool_use["input"])

            converse_messages.append({
                "role": "user",
                "content": tool_results,
            })
            kwargs["messages"] = converse_messages

        # Hit max iterations — return whatever we have
        return ""
