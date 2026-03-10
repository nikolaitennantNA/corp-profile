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

    def _apply_stage(
        raw: str, stage_name: str
    ) -> CompanyProfile | None:
        """Parse LLM response, extract changes, return updated profile or None."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            all_changes.append(f"[{stage_name}] WARNING: LLM returned invalid JSON, skipping")
            return None
        if "changes" in data:
            all_changes.extend(data["changes"])
        if "profile" in data:
            try:
                return CompanyProfile.model_validate(data["profile"])
            except Exception:
                all_changes.append(
                    f"[{stage_name}] WARNING: LLM returned invalid profile data, skipping"
                )
        return None

    # Stage 1: Clean & validate
    clean_messages = [
        {"role": "system", "content": prompts.CLEAN_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
    ]
    clean_raw = provider.complete(clean_messages, json_mode=True)
    profile = _apply_stage(clean_raw, "clean") or profile

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
        profile = _apply_stage(search_raw, "web_search") or profile

    # Stage 2b: Enrich descriptions and context
    enrich_messages = [
        {"role": "system", "content": prompts.ENRICH_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
    ]
    enrich_raw = provider.complete(enrich_messages, json_mode=True)
    profile = _apply_stage(enrich_raw, "enrich") or profile

    return profile, all_changes
