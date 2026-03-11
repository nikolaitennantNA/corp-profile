"""LLM provider abstraction with slug-based model selection."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ..profile import CompanyProfile


class EnrichmentResponse(BaseModel):
    """Expected response shape from all enrichment stages.

    Shared across providers so each can enforce this schema via its
    native structured-output mechanism (OpenAI parsed completions,
    Bedrock tool-use, etc.).
    """

    profile: CompanyProfile
    changes: list[str] = []


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers. Implementations must support complete()."""

    def complete(
        self,
        messages: list[dict],
        json_mode: bool = False,
        web_search: bool = False,
    ) -> str: ...


def parse_model_slug(slug: str) -> tuple[str, str]:
    """Parse 'provider/model' slug into (provider, model) tuple."""
    parts = slug.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid model slug '{slug}'. Expected 'provider/model' format."
        )
    return parts[0], parts[1]


def get_provider(slug: str, *, aws_region: str | None = None, aws_profile: str | None = None) -> LLMProvider:
    """Instantiate an LLMProvider from a model slug like 'openai/gpt-5'."""
    provider_name, model_name = parse_model_slug(slug)
    if provider_name == "openai":
        from .openai import OpenAIProvider
        return OpenAIProvider(model=model_name)
    elif provider_name == "bedrock":
        from .bedrock import BedrockProvider
        return BedrockProvider(model=model_name, region=aws_region, profile=aws_profile)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
