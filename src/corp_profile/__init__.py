"""corp-profile: Build rich company context documents from corp-graph Postgres."""

from __future__ import annotations

from .profile import (
    CompanyProfile,
    build_context_document,
    build_profile,
    build_profile_from_dict,
    build_profile_from_file,
    save_profile,
)

__all__ = [
    # High-level API
    "run",
    "research",
    # Core
    "CompanyProfile",
    "build_context_document",
    "build_profile",
    "build_profile_from_dict",
    "build_profile_from_file",
    "save_profile",
    # Config (lazy)
    "EnrichConfig",
    "PipelineConfig",
    "ResearchConfig",
    "WebConfig",
    # LLM pipeline (lazy)
    "enrich_profile",
    "research_profile",
]


def run(
    identifier: str | None = None,
    *,
    from_file: str | None = None,
    enrich: bool = False,
    web: bool = False,
    enrich_config: object | None = None,
    web_config: object | None = None,
) -> tuple[CompanyProfile, str]:
    """Build a company profile and context document.

    This is the main entry point — equivalent to the CLI commands:
        python -m corp_profile build <identifier>
        python -m corp_profile build --from-file <path> --enrich --web

    Modes:
        - No flags: build profile from DB/file only, no LLM.
        - enrich=True: clean → LLM enrich → refine estimates.
        - web=True: clean → web search → refine estimates.
          Web replaces enrich (real data beats LLM hallucination).
          Works with OpenAI (built-in search) or Bedrock + Exa.

    Args:
        identifier: ISIN, LEI, issuer_id, or company name (requires DB).
        from_file: Path to JSON file (no DB needed).
        enrich: Run LLM-only enrichment (clean + enrich + refine).
        web: Run web search enrichment (clean + web + refine).
            Replaces enrich — setting both is the same as web=True.
        enrich_config: Optional EnrichConfig override (model for clean/refine).
            Loaded from config.toml / env vars if not provided.
        web_config: Optional WebConfig override (model + search provider).
            Loaded from config.toml / env vars if not provided.

    Returns:
        (profile, context_document) tuple.
    """
    if not identifier and not from_file:
        raise ValueError("Provide identifier or from_file")

    if from_file:
        profile = build_profile_from_file(from_file)
    else:
        profile = build_profile(identifier)

    # If no enrichment requested, try to return a cached enriched version
    if not web and not enrich:
        cached = _load_cached(profile.issuer_id)
        if cached:
            return cached, build_context_document(cached)
        return profile, build_context_document(profile)

    # Run enrichment
    from .config import EnrichConfig as _EnrichConfig, WebConfig as _WebConfig
    from .enrich import enrich_profile as _enrich

    ec = enrich_config or _EnrichConfig.load()
    wc = web_config or (_WebConfig.load() if web else None)
    profile, _changes = _enrich(profile, ec, web_config=wc)

    # Cache the enriched profile
    _save_cached(profile)

    return profile, build_context_document(profile)


def research(
    identifier: str | None = None,
    *,
    name: str | None = None,
    seed: str | CompanyProfile | None = None,
    config: object | None = None,
) -> tuple[CompanyProfile, str]:
    """Research a company from scratch via LLM + web search.

    Equivalent to:
        python -m corp_profile research <identifier>
        python -m corp_profile research --seed <path>

    Args:
        identifier: ISIN, LEI, issuer_id, or company name.
        name: Company name hint for search accuracy.
        seed: Path to JSON file or CompanyProfile to seed research.
        config: Optional ResearchConfig override. Loaded from
            config.toml / env vars if not provided.

    Returns:
        (profile, context_document) tuple.
    """
    from .config import ResearchConfig as _ResearchConfig
    from .research import research_profile as _research

    rc = config or _ResearchConfig.load()

    seed_profile = None
    if isinstance(seed, str):
        seed_profile = build_profile_from_file(seed)
        if not identifier:
            identifier = seed_profile.issuer_id or (
                seed_profile.isin_list[0] if seed_profile.isin_list else None
            )
    elif isinstance(seed, CompanyProfile):
        seed_profile = seed
        if not identifier:
            identifier = seed_profile.issuer_id or (
                seed_profile.isin_list[0] if seed_profile.isin_list else None
            )

    profile, _changes = _research(
        identifier=identifier,
        name=name,
        seed=seed_profile,
        config=rc,
    )

    # Cache the researched profile
    _save_cached(profile)

    return profile, build_context_document(profile)


def _save_cached(profile: CompanyProfile) -> None:
    """Best-effort cache of an enriched profile to DB."""
    if not profile.issuer_id:
        return
    try:
        from .db import get_connection, save_cached_profile
        with get_connection() as conn:
            save_cached_profile(conn, profile.issuer_id, profile.model_dump())
    except Exception:
        pass  # DB not available — skip caching silently


def _load_cached(issuer_id: str) -> CompanyProfile | None:
    """Load cached enriched profile from DB, or None."""
    if not issuer_id:
        return None
    try:
        from .db import get_connection, load_cached_profile
        with get_connection() as conn:
            data = load_cached_profile(conn, issuer_id)
            if data:
                return CompanyProfile.model_validate(data)
    except Exception:
        pass
    return None


# Lazy-load LLM-dependent exports to avoid pulling in openai/bedrock/exa
# when callers only need build_profile + build_context_document.
_LAZY_IMPORTS = {
    "EnrichConfig": (".config", "EnrichConfig"),
    "PipelineConfig": (".config", "PipelineConfig"),
    "ResearchConfig": (".config", "ResearchConfig"),
    "WebConfig": (".config", "WebConfig"),
    "enrich_profile": (".enrich", "enrich_profile"),
    "research_profile": (".research", "research_profile"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(module_path, __package__)
        return getattr(mod, attr)
    raise AttributeError(f"module 'corp_profile' has no attribute {name!r}")
