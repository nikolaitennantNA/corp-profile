"""LLM-powered profile enrichment pipeline."""

from __future__ import annotations

import json
import os

from pydantic import BaseModel

from .llm import get_provider
from .profile import CompanyProfile


class EnrichConfig(BaseModel):
    """Configuration for LLM enrichment."""

    model: str  # slug like "openai/gpt-5"
    web_search: bool = False
    web_search_model: str | None = None  # defaults to model if None

    @classmethod
    def from_env(cls) -> EnrichConfig:
        """Load config from environment variables."""
        model = os.environ.get("CORPPROFILE_LLM_MODEL")
        if not model:
            raise RuntimeError(
                "CORPPROFILE_LLM_MODEL not set. "
                "Set it to a slug like 'openai/gpt-5' or 'bedrock/anthropic.claude-3-sonnet'."
            )
        web_search = os.environ.get("CORPPROFILE_WEB_SEARCH", "false").lower() == "true"
        web_search_model = os.environ.get("CORPPROFILE_WEB_SEARCH_MODEL") or None
        return cls(model=model, web_search=web_search, web_search_model=web_search_model)


from . import prompts


def enrich_profile(
    profile: CompanyProfile, config: EnrichConfig
) -> tuple[CompanyProfile, list[str]]:
    """Run the two-stage enrichment pipeline on a profile.

    Returns (enriched_profile, list_of_changes).
    """
    provider = get_provider(config.model)
    all_changes: list[str] = []

    # Stage 1: Clean & validate
    clean_messages = [
        {"role": "system", "content": prompts.CLEAN_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
    ]
    clean_raw = provider.complete(clean_messages, json_mode=True)
    clean_data = json.loads(clean_raw)
    if "changes" in clean_data:
        all_changes.extend(clean_data["changes"])
    if "profile" in clean_data:
        profile = CompanyProfile.model_validate(clean_data["profile"])

    # Stage 2a (optional): Web search for structured discovery
    if config.web_search:
        search_slug = config.web_search_model or config.model
        search_provider = get_provider(search_slug)
        search_messages = [
            {"role": "system", "content": prompts.WEB_SEARCH_ENRICH_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
        ]
        search_raw = search_provider.complete(
            search_messages, json_mode=True, web_search=True
        )
        search_data = json.loads(search_raw)
        if "changes" in search_data:
            all_changes.extend(search_data["changes"])
        if "profile" in search_data:
            profile = CompanyProfile.model_validate(search_data["profile"])

    # Stage 2b: Enrich descriptions and context
    enrich_messages = [
        {"role": "system", "content": prompts.ENRICH_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
    ]
    enrich_raw = provider.complete(enrich_messages, json_mode=True)
    enrich_data = json.loads(enrich_raw)
    if "changes" in enrich_data:
        all_changes.extend(enrich_data["changes"])
    if "profile" in enrich_data:
        profile = CompanyProfile.model_validate(enrich_data["profile"])

    return profile, all_changes
