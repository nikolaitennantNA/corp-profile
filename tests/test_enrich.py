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
        # web_search_model comes from config.toml [llm] if not set in env
        assert config.web_search_model == "openai/gpt-5-mini"

    def test_from_env_with_search(self, monkeypatch):
        monkeypatch.setenv("CORPPROFILE_LLM_MODEL", "bedrock/anthropic.claude-3-sonnet")
        monkeypatch.setenv("CORPPROFILE_WEB_SEARCH", "true")
        monkeypatch.setenv("CORPPROFILE_WEB_SEARCH_MODEL", "openai/gpt-4o")
        config = EnrichConfig.from_env()
        assert config.model == "bedrock/anthropic.claude-3-sonnet"
        assert config.web_search is True
        assert config.web_search_model == "openai/gpt-4o"

    def test_from_env_missing_falls_back_to_config_toml(self, monkeypatch):
        monkeypatch.delenv("CORPPROFILE_LLM_MODEL", raising=False)
        # load() reads from config.toml which has model set
        config = EnrichConfig.load()
        assert config.model == "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
        assert config.aws_region == "us-east-2"

    def test_env_overrides_config_toml(self, monkeypatch):
        monkeypatch.setenv("CORPPROFILE_LLM_MODEL", "openai/gpt-5")
        config = EnrichConfig.load()
        assert config.model == "openai/gpt-5"

    def test_direct_construction(self):
        config = EnrichConfig(model="openai/gpt-5")
        assert config.web_search is False


from corp_profile.enrich import enrich_profile


class TestEnrichProfile:
    def _make_mock_provider(self, responses: list[str]) -> MagicMock:
        """Create a mock LLM provider returning canned responses."""
        provider = MagicMock()
        provider.complete = MagicMock(side_effect=responses)
        return provider

    @patch("corp_profile.enrich.get_provider")
    def test_clean_only(self, mock_get_provider, sample_profile):
        cleaned_data = sample_profile.model_dump()
        cleaned_data["legal_name"] = "Acme Corporation"
        clean_response = json.dumps({
            "profile": cleaned_data,
            "changes": ["Fixed company name: Acme Corp -> Acme Corporation"],
        })
        enrich_response = json.dumps({
            "profile": cleaned_data,
            "changes": [],
        })
        refine_response = json.dumps({
            "material_asset_types": [{"type": "Manufacturing Facility", "count": 45}],
            "estimated_asset_count": 150,
        })
        mock_provider = self._make_mock_provider([clean_response, enrich_response, refine_response])
        mock_get_provider.return_value = mock_provider

        config = EnrichConfig(model="openai/gpt-5")
        result, changes = enrich_profile(sample_profile, config)

        assert result.legal_name == "Acme Corporation"
        assert "Fixed company name" in changes[0]
        assert mock_provider.complete.call_count == 3

    @patch("corp_profile.enrich.get_provider")
    def test_with_web_search(self, mock_get_provider, sample_profile):
        profile_data = sample_profile.model_dump()
        clean_response = json.dumps({"profile": profile_data, "changes": []})
        search_response = json.dumps({"profile": profile_data, "changes": []})
        enrich_response = json.dumps({"profile": profile_data, "changes": []})

        refine_response = json.dumps({
            "material_asset_types": [{"type": "Manufacturing Facility", "count": 45}],
            "estimated_asset_count": 150,
        })
        mock_main = self._make_mock_provider([clean_response, enrich_response, refine_response])
        mock_search = self._make_mock_provider([search_response])

        def pick_provider(slug, **kwargs):
            if slug == "openai/gpt-4o":
                return mock_search
            return mock_main

        mock_get_provider.side_effect = pick_provider

        config = EnrichConfig(
            model="bedrock/claude-3",
            web_search=True,
            web_search_model="openai/gpt-4o",
        )
        result, changes = enrich_profile(sample_profile, config)

        # Main provider called for clean + enrich + refine
        assert mock_main.complete.call_count == 3
        # Search provider called once with web_search=True
        mock_search.complete.assert_called_once()
        search_call = mock_search.complete.call_args
        assert search_call.kwargs.get("web_search") is True or search_call[1].get("web_search") is True

    @patch("corp_profile.enrich.get_provider")
    def test_web_search_defaults_to_main_model(self, mock_get_provider, sample_profile):
        profile_data = sample_profile.model_dump()
        response = json.dumps({"profile": profile_data, "changes": []})
        mock_provider = self._make_mock_provider([response, response, response, response])
        mock_get_provider.return_value = mock_provider

        config = EnrichConfig(model="openai/gpt-5", web_search=True)
        enrich_profile(sample_profile, config)

        # Should call get_provider with main model for search too
        calls = [c.args[0] for c in mock_get_provider.call_args_list]
        assert all(s == "openai/gpt-5" for s in calls)
