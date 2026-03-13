"""Research command — build a CompanyProfile from scratch via LLM + web search."""

from __future__ import annotations

import json

from .config import EnrichConfig, ResearchConfig, load_config
from .llm import get_provider
from .profile import CompanyProfile
from .search import get_search_backend


def research_profile(
    identifier: str | None = None,
    name: str | None = None,
    seed: CompanyProfile | None = None,
    config: ResearchConfig | None = None,
) -> tuple[CompanyProfile, list[str]]:
    """Build a CompanyProfile from scratch via LLM + web search.

    Pipeline:
      1. Research: LLM + web search tool-use loop → initial CompanyProfile
      2. Clean: fix names, normalize jurisdictions, dedup (existing stage)

    Returns (profile, list_of_changes).
    """
    if config is None:
        config = ResearchConfig.load()

    # Validate we have at least one identifier
    if seed:
        has_id = bool(
            seed.issuer_id
            or seed.isin_list
            or seed.lei
        )
        if not has_id and not identifier:
            raise ValueError(
                "Seed profile must contain at least one of: "
                "issuer_id, isin_list, or lei (or pass identifier argument)."
            )

    all_changes: list[str] = []

    # --- Stage 1: Research via LLM + web search ---
    from . import prompts

    research_provider = get_provider(
        config.model,
        aws_region=config.aws_region,
        aws_profile=config.aws_profile,
    )
    backend = get_search_backend(config.provider, model=config.model)

    # Build user message with identifier + optional context
    user_parts = []
    if identifier:
        user_parts.append(f"Company identifier: {identifier}")
    if name:
        user_parts.append(f"Company name: {name}")
    if seed:
        user_parts.append(f"Partial profile (fill gaps):\n{json.dumps(seed.model_dump(), default=str)}")
    if not user_parts:
        raise ValueError("Must provide identifier, name, or seed.")

    messages = [
        {"role": "system", "content": prompts.RESEARCH_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]

    if backend:
        raw = research_provider.complete_with_tools(
            messages, search_backend=backend, json_mode=True,
        )
    else:
        # OpenAI built-in web search
        raw = research_provider.complete(messages, json_mode=True, web_search=True)

    # Parse research result
    def _strip_markdown_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    try:
        data = json.loads(_strip_markdown_fences(raw))
    except json.JSONDecodeError:
        raise RuntimeError("Research stage returned invalid JSON from LLM.")

    if "changes" in data:
        all_changes.extend(data["changes"])
    if "profile" in data:
        profile = CompanyProfile.model_validate(data["profile"])
    else:
        raise RuntimeError("Research stage did not return a profile.")

    # Ensure estimated_asset_count equals sum of material_asset_types
    if profile.material_asset_types:
        type_sum = sum(t.count for t in profile.material_asset_types if t.count is not None)
        if type_sum > 0:
            profile.estimated_asset_count = type_sum

    # --- Stage 2: Clean ---
    try:
        enrich_config = EnrichConfig.load()
    except RuntimeError:
        # If no enrich config, use research model for clean
        enrich_config = EnrichConfig(
            model=config.model,
            aws_region=config.aws_region,
            aws_profile=config.aws_profile,
        )

    clean_provider = get_provider(
        enrich_config.model,
        aws_region=enrich_config.aws_region,
        aws_profile=enrich_config.aws_profile,
    )
    clean_messages = [
        {"role": "system", "content": prompts.CLEAN_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
    ]
    clean_raw = clean_provider.complete(clean_messages, json_mode=True)
    try:
        clean_data = json.loads(_strip_markdown_fences(clean_raw))
        if "changes" in clean_data:
            all_changes.extend(clean_data["changes"])
        if "profile" in clean_data:
            profile = CompanyProfile.model_validate(clean_data["profile"])
    except (json.JSONDecodeError, Exception):
        all_changes.append("[clean] WARNING: LLM returned invalid JSON, skipping")

    return profile, all_changes
