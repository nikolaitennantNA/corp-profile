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
