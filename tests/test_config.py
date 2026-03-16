"""Tests for pipeline configuration classes."""

import pytest

from corp_profile.config import (
    EnrichConfig,
    PipelineConfig,
    ResearchConfig,
    WebConfig,
    load_config,
)


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.enrich is False
        assert cfg.web is False

    def test_load_from_toml(self):
        cfg = PipelineConfig.load()
        # config.toml has enrich=false, web=false
        assert cfg.enrich is False
        assert cfg.web is False

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("CORPPROFILE_ENRICH", "true")
        cfg = PipelineConfig.load()
        assert cfg.enrich is True


class TestEnrichConfig:
    def test_load_from_toml(self):
        cfg = EnrichConfig.load()
        assert cfg.model == "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
        assert cfg.aws_region == "us-east-2"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("CORPPROFILE_ENRICH_MODEL", "openai/gpt-5")
        cfg = EnrichConfig.load()
        assert cfg.model == "openai/gpt-5"

    def test_direct_construction(self):
        cfg = EnrichConfig(model="openai/gpt-5")
        assert cfg.aws_region is None


class TestResearchConfig:
    def test_load_from_toml(self):
        cfg = ResearchConfig.load()
        assert cfg.model == "openai/gpt-5-mini"
        assert cfg.provider == "exa"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("CORPPROFILE_RESEARCH_MODEL", "bedrock/claude-opus")
        cfg = ResearchConfig.load()
        assert cfg.model == "bedrock/claude-opus"


class TestWebConfig:
    def test_load_from_toml(self):
        cfg = WebConfig.load()
        assert cfg.model == "bedrock/us.anthropic.claude-sonnet-4-6-v1"
        assert cfg.provider == "exa"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("CORPPROFILE_WEB_MODEL", "openai/gpt-5")
        cfg = WebConfig.load()
        assert cfg.model == "openai/gpt-5"
