"""corp-profile: Build rich company context documents from corp-graph Postgres."""

from .profile import CompanyProfile, build_context_document, build_profile

__all__ = ["CompanyProfile", "build_context_document", "build_profile"]
