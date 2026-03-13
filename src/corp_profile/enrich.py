"""LLM-powered profile enrichment pipeline."""

from __future__ import annotations

import json

from .config import EnrichConfig, WebConfig, load_config
from .llm import get_provider
from .profile import CompanyProfile

from . import prompts


def enrich_profile(
    profile: CompanyProfile,
    config: EnrichConfig,
    web_config: WebConfig | None = None,
) -> tuple[CompanyProfile, list[str]]:
    """Run the enrichment pipeline on a profile.

    Pipeline ordering:
      --enrich only:  clean → enrich → refine
      --web:          clean → web search → refine (enrich skipped)

    Returns (enriched_profile, list_of_changes).
    """
    provider = get_provider(config.model, aws_region=config.aws_region, aws_profile=config.aws_profile)
    all_changes: list[str] = []

    def _strip_markdown_fences(text: str) -> str:
        """Strip ```json ... ``` wrappers that LLMs often add."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _apply_stage(
        raw: str, stage_name: str
    ) -> CompanyProfile | None:
        """Parse LLM response, extract changes, return updated profile or None."""
        try:
            data = json.loads(_strip_markdown_fences(raw))
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

    # Stage 1: Clean & validate (always runs)
    clean_messages = [
        {"role": "system", "content": prompts.CLEAN_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
    ]
    clean_raw = provider.complete(clean_messages, json_mode=True)
    profile = _apply_stage(clean_raw, "clean") or profile

    # Stage 2: Web search OR LLM-only enrich (never both)
    if web_config:
        # Web search replaces enrich — real data with citations
        search_slug = web_config.model
        search_provider = get_provider(
            search_slug,
            aws_region=web_config.aws_region or config.aws_region,
            aws_profile=web_config.aws_profile or config.aws_profile,
        )
        search_messages = [
            {"role": "system", "content": prompts.WEB_SEARCH_ENRICH_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
        ]
        search_raw = search_provider.complete(
            search_messages, json_mode=True, web_search=True
        )
        profile = _apply_stage(search_raw, "web_search") or profile
    else:
        # LLM-only enrich (best effort, no web)
        enrich_messages = [
            {"role": "system", "content": prompts.ENRICH_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
        ]
        enrich_raw = provider.complete(enrich_messages, json_mode=True)
        profile = _apply_stage(enrich_raw, "enrich") or profile

    # Stage 3: Refine guestimator estimates
    if profile.material_asset_types:
        refine_messages = [
            {"role": "system", "content": prompts.REFINE_ESTIMATES_SYSTEM},
            {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
        ]
        refine_raw = provider.complete(refine_messages, json_mode=True)
        try:
            refine_data = json.loads(_strip_markdown_fences(refine_raw))
        except json.JSONDecodeError:
            all_changes.append("[refine_estimates] WARNING: LLM returned invalid JSON, skipping")
            refine_data = {}
        if refine_data:
            from .profile import refine_estimates
            profile = refine_estimates(profile, refine_data)
            all_changes.append("Refined guestimator asset count estimates via LLM")

    return profile, all_changes
