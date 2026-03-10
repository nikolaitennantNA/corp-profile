"""LLM-powered profile enrichment pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel

from .llm import get_provider
from .profile import CompanyProfile


def _load_config() -> dict:
    """Load config.toml from project root."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    for candidate in [Path("config.toml"), Path(__file__).resolve().parents[3] / "config.toml"]:
        if candidate.exists():
            with open(candidate, "rb") as f:
                return tomllib.load(f)
    return {}


class EnrichConfig(BaseModel):
    """Configuration for LLM enrichment."""

    model: str  # slug like "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0"
    web_search: bool = False
    web_search_model: str | None = None  # defaults to model if None
    aws_region: str | None = None
    aws_profile: str | None = None

    @classmethod
    def load(cls) -> EnrichConfig:
        """Load config from config.toml, with env var overrides.

        Priority: env vars > config.toml > defaults.
        Secrets (API keys) stay in .env. App config lives in config.toml.
        """
        cfg = _load_config()
        llm = cfg.get("llm", {})
        aws = cfg.get("aws", {})

        model = os.environ.get("CORPPROFILE_LLM_MODEL") or llm.get("model")
        if not model:
            raise RuntimeError(
                "LLM model not configured. Set model in [llm] in config.toml, "
                "or set CORPPROFILE_LLM_MODEL env var."
            )

        web_search_env = os.environ.get("CORPPROFILE_WEB_SEARCH")
        if web_search_env is not None:
            web_search = web_search_env.lower() == "true"
        else:
            web_search = bool(llm.get("web_search", False))

        web_search_model = (
            os.environ.get("CORPPROFILE_WEB_SEARCH_MODEL")
            or llm.get("web_search_model")
            or None
        )

        aws_region = (
            os.environ.get("AWS_DEFAULT_REGION")
            or aws.get("region")
            or None
        )
        aws_profile = (
            os.environ.get("AWS_PROFILE")
            or aws.get("profile")
            or None
        )

        return cls(
            model=model,
            web_search=web_search,
            web_search_model=web_search_model,
            aws_region=aws_region,
            aws_profile=aws_profile,
        )

    @classmethod
    def from_env(cls) -> EnrichConfig:
        """Alias for load() — kept for backwards compatibility."""
        return cls.load()


from . import prompts


def enrich_profile(
    profile: CompanyProfile, config: EnrichConfig
) -> tuple[CompanyProfile, list[str]]:
    """Run the two-stage enrichment pipeline on a profile.

    Returns (enriched_profile, list_of_changes).
    """
    provider = get_provider(config.model, aws_region=config.aws_region, aws_profile=config.aws_profile)
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
        search_provider = get_provider(search_slug, aws_region=config.aws_region, aws_profile=config.aws_profile)
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
