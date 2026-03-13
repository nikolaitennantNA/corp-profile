"""Tests for search backend abstraction."""

import pytest
from unittest.mock import MagicMock, patch

from corp_profile.search import (
    ExaSearch,
    SearchResult,
    get_search_backend,
    WEB_SEARCH_TOOL_SCHEMA,
)


class TestSearchResult:
    def test_create(self):
        r = SearchResult(title="Test", url="https://example.com", content="hello")
        assert r.title == "Test"


class TestGetSearchBackend:
    def test_exa(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        backend = get_search_backend("exa", model="bedrock/claude")
        assert isinstance(backend, ExaSearch)

    def test_openai_returns_none(self):
        backend = get_search_backend("openai", model="openai/gpt-5")
        assert backend is None

    def test_auto_openai_model_returns_none(self):
        backend = get_search_backend("auto", model="openai/gpt-5-mini")
        assert backend is None

    def test_auto_bedrock_model_returns_exa(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        backend = get_search_backend("auto", model="bedrock/claude")
        assert isinstance(backend, ExaSearch)

    def test_exa_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="EXA_API_KEY"):
            get_search_backend("exa", model="bedrock/claude")


class TestExaSearch:
    @patch("exa_py.Exa")
    def test_search_returns_results(self, mock_exa_cls, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        mock_result = MagicMock()
        mock_result.title = "TotalEnergies"
        mock_result.url = "https://totalenergies.com"
        mock_result.text = "TotalEnergies is a global energy company."

        mock_client = MagicMock()
        mock_client.search_and_contents.return_value.results = [mock_result]
        mock_exa_cls.return_value = mock_client

        backend = ExaSearch(api_key="test-key")
        results = backend.search("TotalEnergies company overview")

        assert len(results) == 1
        assert results[0].title == "TotalEnergies"
        assert results[0].content == "TotalEnergies is a global energy company."


class TestToolSchema:
    def test_has_required_fields(self):
        assert WEB_SEARCH_TOOL_SCHEMA["name"] == "web_search"
        params = WEB_SEARCH_TOOL_SCHEMA["parameters"]
        assert "query" in params["properties"]
