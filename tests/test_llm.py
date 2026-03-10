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


from unittest.mock import MagicMock, patch


class TestOpenAIProvider:
    @patch("corp_profile.llm.openai.OpenAI")
    def test_complete_basic(self, mock_openai_cls):
        from corp_profile.llm.openai import OpenAIProvider

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"result": "ok"}'))]
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(model="gpt-5")
        result = provider.complete([{"role": "user", "content": "hello"}])
        assert result == '{"result": "ok"}'
        mock_client.chat.completions.create.assert_called_once()

    @patch("corp_profile.llm.openai.OpenAI")
    def test_complete_json_mode(self, mock_openai_cls):
        from corp_profile.llm.openai import OpenAIProvider

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="{}"))]
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(model="gpt-5")
        provider.complete([{"role": "user", "content": "hi"}], json_mode=True)

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("response_format") == {"type": "json_object"}

    @patch("corp_profile.llm.openai.OpenAI")
    def test_complete_with_web_search(self, mock_openai_cls):
        from corp_profile.llm.openai import OpenAIProvider

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_text = MagicMock()
        mock_text.type = "output_text"
        mock_text.text = '{"found": true}'
        mock_msg = MagicMock()
        mock_msg.type = "message"
        mock_msg.content = [mock_text]
        mock_response = MagicMock()
        mock_response.output = [mock_msg]
        mock_client.responses.create.return_value = mock_response

        provider = OpenAIProvider(model="gpt-5")
        result = provider.complete(
            [{"role": "user", "content": "search for Acme Corp"}],
            web_search=True,
        )
        assert result == '{"found": true}'
        mock_client.responses.create.assert_called_once()


def test_get_provider_openai():
    with patch("corp_profile.llm.openai.OpenAI"):
        provider = get_provider("openai/gpt-5")
        assert provider.model == "gpt-5"


class TestBedrockProvider:
    @patch("corp_profile.llm.bedrock.boto3")
    def test_complete_basic(self, mock_boto3):
        from corp_profile.llm.bedrock import BedrockProvider

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": '{"result": "ok"}'}]
                }
            }
        }

        provider = BedrockProvider(model="anthropic.claude-3-sonnet")
        result = provider.complete([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ])
        assert result == '{"result": "ok"}'

    @patch("corp_profile.llm.bedrock.boto3")
    def test_complete_separates_system(self, mock_boto3):
        from corp_profile.llm.bedrock import BedrockProvider

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "{}"}]}}
        }

        provider = BedrockProvider(model="anthropic.claude-3-sonnet")
        provider.complete([
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "hi"},
        ])

        call_kwargs = mock_client.converse.call_args.kwargs
        assert call_kwargs["system"] == [{"text": "Be concise."}]
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    @patch("corp_profile.llm.bedrock.boto3")
    def test_web_search_not_supported(self, mock_boto3):
        from corp_profile.llm.bedrock import BedrockProvider
        import pytest

        mock_boto3.client.return_value = MagicMock()
        provider = BedrockProvider(model="anthropic.claude-3-sonnet")
        with pytest.raises(NotImplementedError):
            provider.complete(
                [{"role": "user", "content": "hi"}], web_search=True
            )


def test_get_provider_bedrock():
    with patch("corp_profile.llm.bedrock.boto3"):
        provider = get_provider("bedrock/anthropic.claude-3-sonnet")
        assert provider.model == "anthropic.claude-3-sonnet"
