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
