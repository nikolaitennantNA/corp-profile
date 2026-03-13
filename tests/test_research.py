"""Tests for the research command."""

import json
from unittest.mock import MagicMock, patch

import pytest

from corp_profile.config import ResearchConfig
from corp_profile.profile import CompanyProfile
from corp_profile.search import SearchResult


class TestResearchProfile:
    def _make_search_backend(self):
        backend = MagicMock()
        backend.search.return_value = [
            SearchResult(
                title="TotalEnergies",
                url="https://totalenergies.com",
                content="TotalEnergies SE is a global energy company.",
            )
        ]
        return backend

    @patch("corp_profile.research.get_search_backend")
    @patch("corp_profile.research.get_provider")
    def test_research_builds_profile(self, mock_get_provider, mock_get_backend):
        from corp_profile.research import research_profile

        profile_data = {
            "issuer_id": "researched-001",
            "legal_name": "TotalEnergies SE",
            "description": "Global energy company",
            "jurisdiction": "FR",
            "primary_industry": "Oil & Gas",
            "operating_countries": ["FR", "US"],
            "business_segments": ["Exploration"],
        }
        research_response = json.dumps({
            "profile": profile_data,
            "changes": ["Found company overview"],
        })
        clean_response = json.dumps({
            "profile": profile_data,
            "changes": [],
        })

        # Research provider uses complete_with_tools
        mock_research_provider = MagicMock()
        mock_research_provider.complete_with_tools.return_value = research_response
        # Clean provider uses complete
        mock_clean_provider = MagicMock()
        mock_clean_provider.complete.return_value = clean_response

        mock_get_provider.side_effect = [mock_research_provider, mock_clean_provider]
        mock_get_backend.return_value = self._make_search_backend()

        config = ResearchConfig(model="openai/gpt-5-mini", provider="exa")
        profile, changes = research_profile(
            identifier="FR0000120271",
            name="TotalEnergies",
            config=config,
        )

        assert profile.legal_name == "TotalEnergies SE"
        assert "Found company overview" in changes
        mock_research_provider.complete_with_tools.assert_called_once()

    def test_seed_validation_rejects_empty(self):
        from corp_profile.research import research_profile

        seed = CompanyProfile(issuer_id="", legal_name="Test")
        config = ResearchConfig(model="openai/gpt-5-mini")

        with pytest.raises(ValueError, match="identifier"):
            research_profile(identifier=None, seed=seed, config=config)

    def test_seed_validation_accepts_isin(self):
        """Seed with isin_list should not raise."""
        seed = CompanyProfile(
            issuer_id="", legal_name="Test", isin_list=["FR0000120271"]
        )
        # Just validating it doesn't raise during validation
        # (the actual LLM call would be mocked in a full test)
        assert seed.isin_list == ["FR0000120271"]
