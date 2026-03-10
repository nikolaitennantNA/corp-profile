"""Tests for LLM provider layer."""

from corp_profile.llm import parse_model_slug, get_provider


def test_parse_slug_openai():
    provider, model = parse_model_slug("openai/gpt-5")
    assert provider == "openai"
    assert model == "gpt-5"


def test_parse_slug_bedrock():
    provider, model = parse_model_slug("bedrock/anthropic.claude-3-sonnet")
    assert provider == "bedrock"
    assert model == "anthropic.claude-3-sonnet"


def test_parse_slug_invalid():
    import pytest
    with pytest.raises(ValueError, match="Invalid model slug"):
        parse_model_slug("no-slash-here")


def test_parse_slug_with_multiple_slashes():
    provider, model = parse_model_slug("bedrock/us.anthropic/claude-3-sonnet")
    assert provider == "bedrock"
    assert model == "us.anthropic/claude-3-sonnet"


def test_get_provider_unknown():
    import pytest
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("fakeprovider/model-1")
