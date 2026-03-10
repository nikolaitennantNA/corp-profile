"""corp-profile: Build rich company context documents from corp-graph Postgres."""

from .enrich import EnrichConfig, enrich_profile
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
