"""Tests for provider tool-use loops."""

import json
from unittest.mock import MagicMock, patch, call

import pytest


class TestBedrockToolLoop:
    def _make_search_backend(self, results_per_call=None):
        """Create a mock search backend."""
        from corp_profile.search import SearchResult

        if results_per_call is None:
            results_per_call = [[
                SearchResult(title="Test", url="https://example.com", content="Test content")
            ]]

        backend = MagicMock()
        backend.search = MagicMock(side_effect=results_per_call)
        return backend

    @patch("corp_profile.llm.bedrock.boto3")
    def test_tool_loop_executes_search(self, mock_boto3):
        from corp_profile.llm.bedrock import BedrockProvider

        # First call: model requests a web search
        search_response = {
            "output": {"message": {"content": [
                {"toolUse": {
                    "toolUseId": "call-1",
                    "name": "web_search",
                    "input": {"query": "TotalEnergies overview", "num_results": 3},
                }}
            ]}},
            "stopReason": "tool_use",
        }
        # Second call: model returns final text
        final_response = {
            "output": {"message": {"content": [
                {"text": '{"profile": {}, "changes": []}'}
            ]}},
            "stopReason": "end_turn",
        }

        mock_client = MagicMock()
        mock_client.converse.side_effect = [search_response, final_response]
        mock_boto3.Session.return_value.client.return_value = mock_client

        provider = BedrockProvider(model="test-model", region="us-east-1")
        backend = self._make_search_backend()

        result = provider.complete_with_tools(
            messages=[{"role": "user", "content": "Research TotalEnergies"}],
            search_backend=backend,
        )

        # Search was executed
        backend.search.assert_called_once_with("TotalEnergies overview", 3)
        # Two Bedrock calls: tool request + final response
        assert mock_client.converse.call_count == 2
        assert '{"profile"' in result

    @patch("corp_profile.llm.bedrock.boto3")
    def test_tool_loop_max_iterations(self, mock_boto3):
        from corp_profile.llm.bedrock import BedrockProvider

        # Model keeps requesting searches forever
        search_response = {
            "output": {"message": {"content": [
                {"toolUse": {
                    "toolUseId": "call-1",
                    "name": "web_search",
                    "input": {"query": "search", "num_results": 3},
                }}
            ]}},
            "stopReason": "tool_use",
        }

        mock_client = MagicMock()
        mock_client.converse.return_value = search_response
        mock_boto3.Session.return_value.client.return_value = mock_client

        provider = BedrockProvider(model="test-model", region="us-east-1")
        backend = self._make_search_backend(
            results_per_call=[
                [MagicMock(title="r", url="u", content="c")]
            ] * 15
        )

        # Should stop at max_iterations
        provider.complete_with_tools(
            messages=[{"role": "user", "content": "test"}],
            search_backend=backend,
            max_iterations=3,
        )
        assert mock_client.converse.call_count == 3


class TestOpenAIToolLoop:
    @patch("corp_profile.llm.openai.OpenAI")
    def test_tool_loop_executes_search(self, mock_openai_cls):
        from corp_profile.llm.openai import OpenAIProvider
        from corp_profile.search import SearchResult

        # First call: model requests a search
        tool_call = MagicMock()
        tool_call.id = "call-1"
        tool_call.function.name = "web_search"
        tool_call.function.arguments = '{"query": "TotalEnergies", "num_results": 3}'

        first_msg = MagicMock()
        first_msg.content = None
        first_msg.tool_calls = [tool_call]
        first_response = MagicMock()
        first_response.choices = [MagicMock(message=first_msg, finish_reason="tool_calls")]

        # Second call: model returns text
        second_msg = MagicMock()
        second_msg.content = '{"profile": {}, "changes": []}'
        second_msg.tool_calls = None
        second_response = MagicMock()
        second_response.choices = [MagicMock(message=second_msg, finish_reason="stop")]

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [first_response, second_response]
        mock_openai_cls.return_value = mock_client

        provider = OpenAIProvider(model="gpt-5")
        backend = MagicMock()
        backend.search.return_value = [
            SearchResult(title="Test", url="https://example.com", content="content")
        ]

        result = provider.complete_with_tools(
            messages=[{"role": "user", "content": "test"}],
            search_backend=backend,
        )

        backend.search.assert_called_once_with("TotalEnergies", 3)
        assert mock_client.chat.completions.create.call_count == 2
