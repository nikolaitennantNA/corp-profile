# Data Independence & LLM Enrichment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make corp-profile runnable without corp-graph DB and add optional LLM-powered profile enrichment.

**Architecture:** Layered pipeline — pluggable data sources (DB, JSON file, dict) feed into `CompanyProfile`, which can optionally pass through a two-stage LLM enricher (clean → enrich). Provider-agnostic LLM layer with slug-based model selection (`provider/model`).

**Tech Stack:** Python 3.13+, Pydantic v2, psycopg3, openai SDK, boto3, pytest

---

## Chunk 1: Test Infrastructure & Data Independence

### Task 1: Set up test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add pytest dev dependency**

In `pyproject.toml`, add after `[build-system]`:

```toml
[dependency-groups]
dev = ["pytest>=8.0"]
```

- [ ] **Step 2: Create test fixtures**

Create `tests/__init__.py` (empty) and `tests/conftest.py`:

```python
"""Shared test fixtures for corp-profile."""

import pytest

from corp_profile import CompanyProfile


SAMPLE_PROFILE_DATA = {
    "issuer_id": "ISS-001",
    "legal_name": "Acme Corp",
    "lei": "529900T8BM49AURSDO55",
    "jurisdiction": "US",
    "isin_list": ["US0378331005"],
    "all_isins": ["US0378331005", "US0378331013"],
    "aliases": ["Acme Corporation", "ACME"],
    "description": "A diversified industrial company.",
    "primary_industry": "Industrials",
    "operating_countries": ["US", "GB", "DE"],
    "business_segments": ["Manufacturing", "Services"],
    "subsidiaries": [
        {
            "issuer_id": "ISS-002",
            "legal_name": "Acme UK Ltd",
            "jurisdiction": "GB",
            "lei": "213800ABCDEF123456",
            "ownership_percentage": 100,
            "rel_type": "DIRECT",
        }
    ],
    "existing_assets": [
        {
            "asset_name": "Acme Plant Alpha",
            "address": "123 Industrial Rd, Chicago, IL",
            "latitude": 41.8781,
            "longitude": -87.6298,
            "naturesense_asset_type": "Manufacturing Facility",
            "capacity": 50000,
            "capacity_units": "tonnes/year",
            "status": "Operating",
        }
    ],
    "discovered_assets": [],
    "estimated_asset_count": 150,
    "material_asset_types": [{"type": "Manufacturing Facility", "count": 45}],
}


@pytest.fixture
def sample_profile_data() -> dict:
    """Return a complete sample profile dict."""
    return SAMPLE_PROFILE_DATA.copy()


@pytest.fixture
def sample_profile() -> CompanyProfile:
    """Return a CompanyProfile instance from sample data."""
    return CompanyProfile.model_validate(SAMPLE_PROFILE_DATA)
```

- [ ] **Step 3: Verify pytest runs**

Run: `uv run pytest tests/ -v`
Expected: 0 tests collected, no errors

- [ ] **Step 4: Commit**

```bash
git add tests/ pyproject.toml
git commit -m "chore: add pytest infrastructure and test fixtures"
```

---

### Task 2: `build_profile_from_dict` (TDD)

**Files:**
- Create: `tests/test_profile_io.py`
- Modify: `src/corp_profile/profile.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_profile_io.py`:

```python
"""Tests for profile I/O: from_dict, from_file, save."""

from corp_profile.profile import build_profile_from_dict, CompanyProfile


def test_build_profile_from_dict(sample_profile_data):
    profile = build_profile_from_dict(sample_profile_data)
    assert isinstance(profile, CompanyProfile)
    assert profile.issuer_id == "ISS-001"
    assert profile.legal_name == "Acme Corp"
    assert profile.lei == "529900T8BM49AURSDO55"
    assert len(profile.subsidiaries) == 1
    assert len(profile.existing_assets) == 1


def test_build_profile_from_dict_minimal():
    data = {"issuer_id": "ISS-999", "legal_name": "Minimal Co"}
    profile = build_profile_from_dict(data)
    assert profile.issuer_id == "ISS-999"
    assert profile.subsidiaries == []
    assert profile.description == ""


def test_build_profile_from_dict_invalid():
    import pytest
    with pytest.raises(Exception):
        build_profile_from_dict({"not_a_field": "bad"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_profile_io.py -v`
Expected: FAIL — `cannot import name 'build_profile_from_dict'`

- [ ] **Step 3: Implement `build_profile_from_dict`**

In `src/corp_profile/profile.py`, add after `build_context_document` (end of file):

```python
def build_profile_from_dict(data: dict) -> CompanyProfile:
    """Build a CompanyProfile from a plain dict (no DB required)."""
    return CompanyProfile.model_validate(data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_profile_io.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_profile_io.py src/corp_profile/profile.py
git commit -m "feat: add build_profile_from_dict for DB-free profile construction"
```

---

### Task 3: `save_profile` and `build_profile_from_file` (TDD)

**Files:**
- Modify: `tests/test_profile_io.py`
- Modify: `src/corp_profile/profile.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_profile_io.py`:

```python
import json
from pathlib import Path

from corp_profile.profile import save_profile, build_profile_from_file


def test_save_profile(sample_profile, tmp_path):
    out = tmp_path / "profile.json"
    save_profile(sample_profile, str(out))
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["issuer_id"] == "ISS-001"
    assert data["legal_name"] == "Acme Corp"


def test_build_profile_from_file(sample_profile, tmp_path):
    out = tmp_path / "profile.json"
    save_profile(sample_profile, str(out))
    loaded = build_profile_from_file(str(out))
    assert loaded.issuer_id == sample_profile.issuer_id
    assert loaded.legal_name == sample_profile.legal_name
    assert loaded.subsidiaries == sample_profile.subsidiaries


def test_round_trip(sample_profile, tmp_path):
    out = tmp_path / "profile.json"
    save_profile(sample_profile, str(out))
    loaded = build_profile_from_file(str(out))
    assert loaded.model_dump() == sample_profile.model_dump()


def test_build_profile_from_file_not_found():
    import pytest
    with pytest.raises(FileNotFoundError):
        build_profile_from_file("/nonexistent/path.json")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_profile_io.py -v`
Expected: FAIL — `cannot import name 'save_profile'`

- [ ] **Step 3: Implement `save_profile` and `build_profile_from_file`**

In `src/corp_profile/profile.py`, add at top with existing imports:

```python
import json
from pathlib import Path
```

Then add at end of file:

```python
def save_profile(profile: CompanyProfile, path: str) -> None:
    """Save a CompanyProfile to a JSON file."""
    Path(path).write_text(
        json.dumps(profile.model_dump(), indent=2, default=str)
    )


def build_profile_from_file(path: str) -> CompanyProfile:
    """Load a CompanyProfile from a JSON file."""
    data = json.loads(Path(path).read_text())
    return build_profile_from_dict(data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_profile_io.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_profile_io.py src/corp_profile/profile.py
git commit -m "feat: add save_profile and build_profile_from_file for JSON I/O"
```

---

### Task 4: CLI `--from-file` flag

**Files:**
- Modify: `src/corp_profile/__main__.py`

- [ ] **Step 1: Update CLI to support `--from-file`**

Replace the contents of `src/corp_profile/__main__.py`:

```python
"""CLI entry point: python -m corp_profile build <ISIN>."""

from __future__ import annotations

import argparse
import sys

from .profile import (
    build_context_document,
    build_profile,
    build_profile_from_file,
    save_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="corp-profile",
        description="Build rich company context documents from corp-graph Postgres",
    )
    sub = parser.add_subparsers(dest="command")

    build_cmd = sub.add_parser("build", help="Build a company profile by ISIN")
    build_source = build_cmd.add_mutually_exclusive_group(required=True)
    build_source.add_argument("--isin", help="ISIN to look up in corp-graph DB")
    build_source.add_argument("--from-file", help="Load profile from JSON file")
    build_cmd.add_argument(
        "-o", "--output", help="Save profile JSON to file instead of printing"
    )

    args = parser.parse_args()

    if args.command == "build":
        try:
            if args.from_file:
                profile = build_profile_from_file(args.from_file)
            else:
                profile = build_profile(args.isin)
        except LookupError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if args.output:
            save_profile(profile, args.output)
            print(f"Profile saved to {args.output}", file=sys.stderr)
        else:
            print(build_context_document(profile))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help works**

Run: `uv run python -m corp_profile build --help`
Expected: Shows `--isin`, `--from-file`, and `-o`/`--output` options

- [ ] **Step 3: Commit**

```bash
git add src/corp_profile/__main__.py
git commit -m "feat: CLI --from-file and --output flags for DB-free usage"
```

---

## Chunk 2: LLM Provider Layer

### Task 5: Provider protocol and slug parsing (TDD)

**Files:**
- Create: `src/corp_profile/llm/__init__.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_llm.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL — `No module named 'corp_profile.llm'`

- [ ] **Step 3: Implement protocol and slug parsing**

Create `src/corp_profile/llm/__init__.py`:

```python
"""LLM provider abstraction with slug-based model selection."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers. Implementations must support complete()."""

    def complete(
        self,
        messages: list[dict],
        json_mode: bool = False,
        web_search: bool = False,
    ) -> str: ...


def parse_model_slug(slug: str) -> tuple[str, str]:
    """Parse 'provider/model' slug into (provider, model) tuple."""
    parts = slug.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid model slug '{slug}'. Expected 'provider/model' format."
        )
    return parts[0], parts[1]


def get_provider(slug: str) -> LLMProvider:
    """Instantiate an LLMProvider from a model slug like 'openai/gpt-5'."""
    provider_name, model_name = parse_model_slug(slug)
    if provider_name == "openai":
        from .openai import OpenAIProvider
        return OpenAIProvider(model=model_name)
    elif provider_name == "bedrock":
        from .bedrock import BedrockProvider
        return BedrockProvider(model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm.py -v -k "not get_provider_unknown"`
Expected: 4 passed (skip `get_provider_unknown` until adapters exist)

Then run: `uv run pytest tests/test_llm.py::test_get_provider_unknown -v`
Expected: PASS (raises ValueError for unknown provider)

- [ ] **Step 5: Commit**

```bash
git add src/corp_profile/llm/__init__.py tests/test_llm.py
git commit -m "feat: LLM provider protocol and slug parsing"
```

---

### Task 6: OpenAI adapter (TDD)

**Files:**
- Create: `src/corp_profile/llm/openai.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_llm.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm.py::TestOpenAIProvider -v`
Expected: FAIL — `No module named 'corp_profile.llm.openai'`

- [ ] **Step 3: Implement OpenAI adapter**

Create `src/corp_profile/llm/openai.py`:

```python
"""OpenAI LLM provider adapter."""

from __future__ import annotations

from openai import OpenAI


class OpenAIProvider:
    """LLM provider using OpenAI Chat Completions and Responses APIs."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.client = OpenAI()

    def complete(
        self,
        messages: list[dict],
        json_mode: bool = False,
        web_search: bool = False,
    ) -> str:
        if web_search:
            return self._complete_with_search(messages, json_mode)

        kwargs: dict = {"model": self.model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _complete_with_search(
        self, messages: list[dict], json_mode: bool
    ) -> str:
        """Use OpenAI Responses API with web search tool."""
        kwargs: dict = {
            "model": self.model,
            "input": messages,
            "tools": [{"type": "web_search_preview"}],
        }
        if json_mode:
            kwargs["text"] = {"format": {"type": "json_object"}}

        response = self.client.responses.create(**kwargs)
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        return content.text
        return ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/corp_profile/llm/openai.py tests/test_llm.py
git commit -m "feat: OpenAI adapter with web search support"
```

---

### Task 7: Bedrock adapter (TDD)

**Files:**
- Create: `src/corp_profile/llm/bedrock.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_llm.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm.py::TestBedrockProvider -v`
Expected: FAIL — `No module named 'corp_profile.llm.bedrock'`

- [ ] **Step 3: Implement Bedrock adapter**

Create `src/corp_profile/llm/bedrock.py`:

```python
"""Amazon Bedrock LLM provider adapter."""

from __future__ import annotations

import boto3


class BedrockProvider:
    """LLM provider using Amazon Bedrock Converse API."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.client = boto3.client("bedrock-runtime")

    def complete(
        self,
        messages: list[dict],
        json_mode: bool = False,
        web_search: bool = False,
    ) -> str:
        if web_search:
            raise NotImplementedError(
                "Bedrock does not support web search. "
                "Set CORPPROFILE_WEB_SEARCH_MODEL to an OpenAI model slug."
            )

        system_parts: list[dict] = []
        converse_messages: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append({"text": msg["content"]})
            else:
                converse_messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}],
                })

        kwargs: dict = {
            "modelId": self.model,
            "messages": converse_messages,
        }
        if system_parts:
            kwargs["system"] = system_parts

        response = self.client.converse(**kwargs)
        return response["output"]["message"]["content"][0]["text"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/corp_profile/llm/bedrock.py tests/test_llm.py
git commit -m "feat: Bedrock adapter with Converse API"
```

---

### Task 8: Update pyproject.toml with optional dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add optional dependency groups**

In `pyproject.toml`, add after `dependencies`:

```toml
[project.optional-dependencies]
openai = ["openai>=1.0"]
bedrock = ["boto3>=1.35"]
```

- [ ] **Step 2: Verify install**

Run: `uv sync`
Expected: Succeeds without installing openai or boto3

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add openai and bedrock as optional dependencies"
```

---

## Chunk 3: Enrichment Pipeline

### Task 9: Prompts module

**Files:**
- Create: `src/corp_profile/prompts.py`

- [ ] **Step 1: Create prompts module**

Create `src/corp_profile/prompts.py`:

```python
"""System prompts for LLM enrichment stages."""

from __future__ import annotations

from .profile import CompanyProfile

# Generate JSON schema once for use in prompts
_SCHEMA = CompanyProfile.model_json_schema()

CLEAN_SYSTEM_PROMPT = f"""\
You are a data-cleaning specialist for corporate entity profiles.

You will receive a company profile as JSON. Your job:
1. Fix garbled or corrupted company names (e.g. all-caps artifacts, encoding issues)
2. Normalize jurisdiction codes to ISO 3166-1 alpha-2
3. Deduplicate subsidiaries that appear under slightly different names
4. Flag any obviously inconsistent data

Return a JSON object with exactly two keys:
- "profile": the cleaned profile matching this schema: {_SCHEMA}
- "changes": a list of strings describing each change you made (empty list if none)

Do not invent data. Only fix what is clearly wrong or inconsistent.
"""

ENRICH_SYSTEM_PROMPT = f"""\
You are a corporate research analyst enriching company profiles.

You will receive a company profile as JSON. Your job:
1. Improve the company description if it is missing or very sparse
2. Fill in missing industry classification if determinable from context
3. Expand operating countries or business segments if clearly incomplete
4. Add relevant context about the company's operations

Return a JSON object with exactly two keys:
- "profile": the enriched profile matching this schema: {_SCHEMA}
- "changes": a list of strings describing each enrichment you made (empty list if none)

Preserve all existing data. Only add or improve — never remove information.
"""

WEB_SEARCH_ENRICH_SYSTEM_PROMPT = f"""\
You are a corporate research analyst with web search capability.

You will receive a company profile as JSON. Search the web to:
1. Find missing subsidiaries and add them to the subsidiaries list
2. Identify operating countries not already listed
3. Discover business segments or divisions
4. Improve the company description with current, factual information
5. Verify and correct existing data where web sources contradict it

Return a JSON object with exactly two keys:
- "profile": the enriched profile matching this schema: {_SCHEMA}
- "changes": a list of strings describing each discovery or correction (empty list if none)

Only include facts you can verify from search results. Cite sources in change descriptions.
"""
```

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from corp_profile.prompts import CLEAN_SYSTEM_PROMPT; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/corp_profile/prompts.py
git commit -m "feat: system prompts for clean, enrich, and web search stages"
```

---

### Task 10: EnrichConfig (TDD)

**Files:**
- Create: `tests/test_enrich.py`
- Create: `src/corp_profile/enrich.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_enrich.py`:

```python
"""Tests for enrichment pipeline."""

import json
from unittest.mock import MagicMock, patch

from corp_profile.enrich import EnrichConfig, enrich_profile
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_enrich.py::TestEnrichConfig -v`
Expected: FAIL — `No module named 'corp_profile.enrich'`

- [ ] **Step 3: Implement EnrichConfig**

Create `src/corp_profile/enrich.py`:

```python
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
```

- [ ] **Step 4: Run config tests to verify they pass**

Run: `uv run pytest tests/test_enrich.py::TestEnrichConfig -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/corp_profile/enrich.py tests/test_enrich.py
git commit -m "feat: EnrichConfig with slug-based model selection and env loading"
```

---

### Task 11: `enrich_profile` orchestrator (TDD)

**Files:**
- Modify: `tests/test_enrich.py`
- Modify: `src/corp_profile/enrich.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_enrich.py`:

```python
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
        mock_provider = self._make_mock_provider([clean_response, enrich_response])
        mock_get_provider.return_value = mock_provider

        config = EnrichConfig(model="openai/gpt-5")
        result, changes = enrich_profile(sample_profile, config)

        assert result.legal_name == "Acme Corporation"
        assert "Fixed company name" in changes[0]
        assert mock_provider.complete.call_count == 2

    @patch("corp_profile.enrich.get_provider")
    def test_with_web_search(self, mock_get_provider, sample_profile):
        profile_data = sample_profile.model_dump()
        clean_response = json.dumps({"profile": profile_data, "changes": []})
        search_response = json.dumps({"profile": profile_data, "changes": []})
        enrich_response = json.dumps({"profile": profile_data, "changes": []})

        mock_main = self._make_mock_provider([clean_response, enrich_response])
        mock_search = self._make_mock_provider([search_response])

        def pick_provider(slug):
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

        # Main provider called for clean + enrich
        assert mock_main.complete.call_count == 2
        # Search provider called once with web_search=True
        mock_search.complete.assert_called_once()
        search_call = mock_search.complete.call_args
        assert search_call.kwargs.get("web_search") is True or search_call[1].get("web_search") is True

    @patch("corp_profile.enrich.get_provider")
    def test_web_search_defaults_to_main_model(self, mock_get_provider, sample_profile):
        profile_data = sample_profile.model_dump()
        response = json.dumps({"profile": profile_data, "changes": []})
        mock_provider = self._make_mock_provider([response, response, response])
        mock_get_provider.return_value = mock_provider

        config = EnrichConfig(model="openai/gpt-5", web_search=True)
        enrich_profile(sample_profile, config)

        # Should call get_provider with main model for search too
        calls = [c.args[0] for c in mock_get_provider.call_args_list]
        assert all(s == "openai/gpt-5" for s in calls)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_enrich.py::TestEnrichProfile -v`
Expected: FAIL — `cannot import name 'enrich_profile'`

- [ ] **Step 3: Implement `enrich_profile`**

Append to `src/corp_profile/enrich.py`:

```python
from . import prompts


def enrich_profile(
    profile: CompanyProfile, config: EnrichConfig
) -> tuple[CompanyProfile, list[str]]:
    """Run the two-stage enrichment pipeline on a profile.

    Returns (enriched_profile, list_of_changes).
    """
    provider = get_provider(config.model)
    all_changes: list[str] = []

    # Stage 1: Clean & validate
    clean_messages = [
        {"role": "system", "content": prompts.CLEAN_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
    ]
    clean_raw = provider.complete(clean_messages, json_mode=True)
    clean_data = json.loads(clean_raw)
    if "changes" in clean_data:
        all_changes.extend(clean_data["changes"])
    if "profile" in clean_data:
        profile = CompanyProfile.model_validate(clean_data["profile"])

    # Stage 2a (optional): Web search for structured discovery
    if config.web_search:
        search_slug = config.web_search_model or config.model
        search_provider = get_provider(search_slug)
        search_messages = [
            {"role": "system", "content": prompts.WEB_SEARCH_ENRICH_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
        ]
        search_raw = search_provider.complete(
            search_messages, json_mode=True, web_search=True
        )
        search_data = json.loads(search_raw)
        if "changes" in search_data:
            all_changes.extend(search_data["changes"])
        if "profile" in search_data:
            profile = CompanyProfile.model_validate(search_data["profile"])

    # Stage 2b: Enrich descriptions and context
    enrich_messages = [
        {"role": "system", "content": prompts.ENRICH_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
    ]
    enrich_raw = provider.complete(enrich_messages, json_mode=True)
    enrich_data = json.loads(enrich_raw)
    if "changes" in enrich_data:
        all_changes.extend(enrich_data["changes"])
    if "profile" in enrich_data:
        profile = CompanyProfile.model_validate(enrich_data["profile"])

    return profile, all_changes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_enrich.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/corp_profile/enrich.py tests/test_enrich.py
git commit -m "feat: enrich_profile two-stage pipeline with optional web search"
```

---

### Task 12: CLI `--enrich` and `enrich` command

**Files:**
- Modify: `src/corp_profile/__main__.py`

- [ ] **Step 1: Add `--enrich` flag and `enrich` subcommand**

Replace `src/corp_profile/__main__.py`:

```python
"""CLI entry point: python -m corp_profile."""

from __future__ import annotations

import argparse
import sys

from .profile import (
    build_context_document,
    build_profile,
    build_profile_from_file,
    save_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="corp-profile",
        description="Build rich company context documents from corp-graph Postgres",
    )
    sub = parser.add_subparsers(dest="command")

    # build command
    build_cmd = sub.add_parser("build", help="Build a company profile")
    build_source = build_cmd.add_mutually_exclusive_group(required=True)
    build_source.add_argument("--isin", help="ISIN to look up in corp-graph DB")
    build_source.add_argument("--from-file", help="Load profile from JSON file")
    build_cmd.add_argument(
        "-o", "--output", help="Save profile JSON to file instead of printing"
    )
    build_cmd.add_argument(
        "--enrich", action="store_true", help="Run LLM enrichment on the profile"
    )

    # enrich command
    enrich_cmd = sub.add_parser("enrich", help="Enrich an existing profile JSON")
    enrich_cmd.add_argument("file", help="Path to profile JSON file")
    enrich_cmd.add_argument(
        "-o", "--output", help="Save enriched profile to file (default: print)"
    )

    args = parser.parse_args()

    if args.command == "build":
        try:
            if args.from_file:
                profile = build_profile_from_file(args.from_file)
            else:
                profile = build_profile(args.isin)
        except LookupError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if args.enrich:
            profile = _run_enrich(profile)

        if args.output:
            save_profile(profile, args.output)
            print(f"Profile saved to {args.output}", file=sys.stderr)
        else:
            print(build_context_document(profile))

    elif args.command == "enrich":
        profile = build_profile_from_file(args.file)
        profile = _run_enrich(profile)

        if args.output:
            save_profile(profile, args.output)
            print(f"Enriched profile saved to {args.output}", file=sys.stderr)
        else:
            print(build_context_document(profile))

    else:
        parser.print_help()
        sys.exit(1)


def _run_enrich(profile):
    """Run enrichment and print changes to stderr."""
    from .enrich import EnrichConfig, enrich_profile

    try:
        config = EnrichConfig.from_env()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    profile, changes = enrich_profile(profile, config)
    if changes:
        print("Enrichment changes:", file=sys.stderr)
        for c in changes:
            print(f"  - {c}", file=sys.stderr)
    return profile


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help**

Run: `uv run python -m corp_profile build --help`
Expected: Shows `--enrich` flag

Run: `uv run python -m corp_profile enrich --help`
Expected: Shows `file` argument and `-o` option

- [ ] **Step 3: Commit**

```bash
git add src/corp_profile/__main__.py
git commit -m "feat: CLI --enrich flag and enrich subcommand"
```

---

### Task 13: Update exports, .env.example, and CLAUDE.md

**Files:**
- Modify: `src/corp_profile/__init__.py`
- Modify: `.env.example`

- [ ] **Step 1: Update `__init__.py` exports**

Replace `src/corp_profile/__init__.py`:

```python
"""corp-profile: Build rich company context documents from corp-graph Postgres."""

from .enrich import EnrichConfig, enrich_profile
from .profile import (
    CompanyProfile,
    build_context_document,
    build_profile,
    build_profile_from_dict,
    build_profile_from_file,
    save_profile,
)

__all__ = [
    "CompanyProfile",
    "EnrichConfig",
    "build_context_document",
    "build_profile",
    "build_profile_from_dict",
    "build_profile_from_file",
    "enrich_profile",
    "save_profile",
]
```

- [ ] **Step 2: Update `.env.example`**

Replace `.env.example`:

```
CORPGRAPH_DB_URL=postgresql://corpgraph:corpgraph@localhost:5432/corpgraph

# LLM enrichment (optional)
CORPPROFILE_LLM_MODEL=openai/gpt-5
CORPPROFILE_WEB_SEARCH=false
CORPPROFILE_WEB_SEARCH_MODEL=openai/gpt-4o
```

- [ ] **Step 3: Verify full test suite passes**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/corp_profile/__init__.py .env.example
git commit -m "chore: update exports and .env.example with enrichment config"
```
