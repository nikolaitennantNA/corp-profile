"""corp-profile: Build rich company context documents from corp-graph Postgres."""

from .profile import (
    CompanyProfile,
    build_context_document,
    build_profile,
    build_profile_from_dict,
    build_profile_from_file,
    save_profile,
)

__all__ = [
    "CompanyProfile",
    "EnrichConfig",
    "build_context_document",
    "build_profile",
    "build_profile_from_dict",
    "build_profile_from_file",
    "enrich_profile",
    "save_profile",
]


def __getattr__(name: str):
    """Lazy-load enrichment exports to avoid importing LLM deps eagerly."""
    if name == "EnrichConfig":
        from .enrich import EnrichConfig
        return EnrichConfig
    if name == "enrich_profile":
        from .enrich import enrich_profile
        return enrich_profile
    raise AttributeError(f"module 'corp_profile' has no attribute {name!r}")
