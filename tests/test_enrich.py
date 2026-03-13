"""Tests for enrichment pipeline."""

import json
from unittest.mock import MagicMock, patch

from corp_profile.config import EnrichConfig, WebConfig
from corp_profile.profile import CompanyProfile


class TestEnrichConfig:
    def test_load_from_toml(self):
        config = EnrichConfig.load()
        assert config.model == "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
        assert config.aws_region == "us-east-2"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("CORPPROFILE_ENRICH_MODEL", "openai/gpt-5")
        config = EnrichConfig.load()
        assert config.model == "openai/gpt-5"

    def test_deprecated_env_var(self, monkeypatch):
        monkeypatch.setenv("CORPPROFILE_LLM_MODEL", "openai/gpt-5")
        monkeypatch.delenv("CORPPROFILE_ENRICH_MODEL", raising=False)
        config = EnrichConfig.load()
        assert config.model == "openai/gpt-5"

    def test_direct_construction(self):
        config = EnrichConfig(model="openai/gpt-5")
        assert config.aws_region is None


from corp_profile.enrich import enrich_profile


class TestEnrichProfile:
    def _make_mock_provider(self, responses: list[str]) -> MagicMock:
        """Create a mock LLM provider returning canned responses."""
        provider = MagicMock()
        provider.complete = MagicMock(side_effect=responses)
        return provider

    @patch("corp_profile.enrich.get_provider")
    def test_enrich_only(self, mock_get_provider, sample_profile):
        """--enrich without --web: runs clean → enrich → refine."""
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
        # clean + enrich + refine = 3 calls
        assert mock_provider.complete.call_count == 3

    @patch("corp_profile.enrich.get_provider")
    def test_with_web_search_skips_enrich(self, mock_get_provider, sample_profile):
        """--web: runs clean → web search → refine (enrich SKIPPED)."""
        profile_data = sample_profile.model_dump()
        clean_response = json.dumps({"profile": profile_data, "changes": []})
        search_response = json.dumps({"profile": profile_data, "changes": ["Found info via web"]})
        refine_response = json.dumps({
            "material_asset_types": [{"type": "Manufacturing Facility", "count": 45}],
            "estimated_asset_count": 150,
        })

        mock_main = self._make_mock_provider([clean_response, refine_response])
        mock_search = self._make_mock_provider([search_response])

        def pick_provider(slug, **kwargs):
            if slug == "openai/gpt-4o":
                return mock_search
            return mock_main

        mock_get_provider.side_effect = pick_provider

        config = EnrichConfig(model="bedrock/claude-3")
        web_config = WebConfig(model="openai/gpt-4o", provider="openai")
        result, changes = enrich_profile(sample_profile, config, web_config=web_config)

        # Main provider: clean + refine = 2 calls (enrich skipped!)
        assert mock_main.complete.call_count == 2
        # Search provider called once with web_search=True
        mock_search.complete.assert_called_once()
        search_call = mock_search.complete.call_args
        assert search_call.kwargs.get("web_search") is True or search_call[1].get("web_search") is True
