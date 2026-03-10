"""Tests for enrichment pipeline."""

import json
from unittest.mock import MagicMock, patch

from corp_profile.enrich import EnrichConfig
from corp_profile.profile import CompanyProfile


class TestEnrichConfig:
    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("CORPPROFILE_LLM_MODEL", "openai/gpt-5")
        monkeypatch.setenv("CORPPROFILE_WEB_SEARCH", "false")
        config = EnrichConfig.from_env()
        assert config.model == "openai/gpt-5"
        assert config.web_search is False
        assert config.web_search_model is None

    def test_from_env_with_search(self, monkeypatch):
        monkeypatch.setenv("CORPPROFILE_LLM_MODEL", "bedrock/anthropic.claude-3-sonnet")
        monkeypatch.setenv("CORPPROFILE_WEB_SEARCH", "true")
        monkeypatch.setenv("CORPPROFILE_WEB_SEARCH_MODEL", "openai/gpt-4o")
        config = EnrichConfig.from_env()
        assert config.model == "bedrock/anthropic.claude-3-sonnet"
        assert config.web_search is True
        assert config.web_search_model == "openai/gpt-4o"

    def test_from_env_missing(self, monkeypatch):
        monkeypatch.delenv("CORPPROFILE_LLM_MODEL", raising=False)
        import pytest
        with pytest.raises(RuntimeError, match="CORPPROFILE_LLM_MODEL"):
            EnrichConfig.from_env()

    def test_direct_construction(self):
        config = EnrichConfig(model="openai/gpt-5")
        assert config.web_search is False
