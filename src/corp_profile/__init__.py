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

    Args:
        identifier: ISIN, LEI, issuer_id, or company name (requires DB).
        from_file: Path to JSON file (no DB needed).
        enrich: Run LLM enrichment (clean + enrich + refine).
        web: Run web search enrichment (implies enrich).
        enrich_config: Optional EnrichConfig override. Loaded from
            config.toml / env vars if not provided.
        web_config: Optional WebConfig override. Loaded from
            config.toml / env vars if not provided.

    Returns:
        (profile, context_document) tuple.
    """
    if not identifier and not from_file:
        raise ValueError("Provide identifier or from_file")

    if web:
        enrich = True

    if from_file:
        profile = build_profile_from_file(from_file)
    else:
        profile = build_profile(identifier)

    if enrich:
        from .config import EnrichConfig as _EnrichConfig, WebConfig as _WebConfig
        from .enrich import enrich_profile as _enrich

        ec = enrich_config or _EnrichConfig.load()
        wc = web_config or (_WebConfig.load() if web else None)
        profile, _changes = _enrich(profile, ec, web_config=wc)

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

    return profile, build_context_document(profile)


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
