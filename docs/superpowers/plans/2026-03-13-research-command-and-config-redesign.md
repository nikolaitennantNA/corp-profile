# Research Command, Config Redesign & company_universe Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `research` subcommand that builds CompanyProfile from scratch via LLM + web search, decouple search backends (Exa/OpenAI), restructure config into per-stage sections, and fold `material_assets_types` into company_universe.

**Architecture:** Config is split into `[pipeline]`, `[research]`, `[enrich]`, `[web]` sections. A new `SearchBackend` protocol abstracts Exa and OpenAI web search. Both LLM providers get `complete_with_tools()` for search tool-use loops. The `research` command uses this to build profiles from scratch. Pipeline ordering ensures web-sourced facts are never overwritten by LLM hallucinations.

**Tech Stack:** Python 3.13+, Pydantic v2, exa-py, openai SDK, boto3 (Bedrock), psycopg v3, pycountry, hatchling

**Spec:** `docs/superpowers/specs/2026-03-13-research-command-and-config-redesign.md`

---

## File Structure

### New files (corp-profile)
| File | Responsibility |
|------|---------------|
| `src/corp_profile/config.py` | All config classes: `PipelineConfig`, `ResearchConfig`, `EnrichConfig`, `WebConfig`, `load_config()` |
| `src/corp_profile/search.py` | Search abstraction: `SearchResult`, `SearchBackend` protocol, `ExaSearch`, `get_search_backend()` |
| `src/corp_profile/research.py` | Research command logic: `research_profile()` |
| `tests/test_config.py` | Config class tests |
| `tests/test_search.py` | Search backend tests |
| `tests/test_research.py` | Research command tests |
| `tests/test_tool_loop.py` | Provider tool-use loop tests |

### Modified files (corp-profile)
| File | What changes |
|------|-------------|
| `config.toml` | Full restructure: `[pipeline]`, `[research]`, `[enrich]`, `[web]`, `[aws]` |
| `pyproject.toml:14` | Add `exa-py` to `[llm]` extra |
| `src/corp_profile/__init__.py:24-32` | Update lazy-load to import from `config.py` |
| `src/corp_profile/__main__.py` | Add `research` subcommand, rename `--llm` → `--enrich` with alias |
| `src/corp_profile/enrich.py` | Use new config, skip enrich when web enabled, remove old `EnrichConfig` and `load_config()`, remove dead async function |
| `src/corp_profile/prompts.py` | Add `RESEARCH_SYSTEM_PROMPT` |
| `src/corp_profile/llm/__init__.py` | No protocol change, keep as-is |
| `src/corp_profile/llm/bedrock.py` | Add `complete_with_tools()`, refactor `web_search=True` |
| `src/corp_profile/llm/openai.py` | Add `complete_with_tools()`, keep `web_search_preview` fallback |
| `src/corp_profile/profile.py:287-303` | Remove `asset_estimates` query, read from `company_universe` row |
| `tests/test_enrich.py` | Update for new config classes and pipeline ordering |
| `tests/conftest.py` | No changes needed |

### Modified files (corp-graph)
| File | What changes |
|------|-------------|
| `src/materialize.py` | Add LEFT JOIN to `asset_estimates`, add `material_assets_types` column |

### Modified files (asset-search-v2)
| File | What changes |
|------|-------------|
| `config.toml` | Restructure `[profile]` to nested sections |
| `src/asset_search/config.py` | New builder methods, rename fields |

---

## Chunk 1: Config Redesign

### Task 1: Create config.py with new config classes

**Files:**
- Create: `src/corp_profile/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for config classes**

```python
# tests/test_config.py
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

    def test_deprecated_env_var(self, monkeypatch):
        monkeypatch.setenv("CORPPROFILE_LLM_MODEL", "openai/gpt-5")
        monkeypatch.delenv("CORPPROFILE_ENRICH_MODEL", raising=False)
        cfg = EnrichConfig.load()
        assert cfg.model == "openai/gpt-5"

    def test_direct_construction(self):
        cfg = EnrichConfig(model="openai/gpt-5")
        assert cfg.aws_region is None


class TestResearchConfig:
    def test_load_from_toml(self):
        cfg = ResearchConfig.load()
        assert cfg.model == "openai/gpt-5-mini"
        assert cfg.provider == "auto"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("CORPPROFILE_RESEARCH_MODEL", "bedrock/claude-opus")
        cfg = ResearchConfig.load()
        assert cfg.model == "bedrock/claude-opus"


class TestWebConfig:
    def test_load_from_toml(self):
        cfg = WebConfig.load()
        assert cfg.model == "openai/gpt-5-mini"
        assert cfg.provider == "auto"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("CORPPROFILE_WEB_MODEL", "openai/gpt-5")
        cfg = WebConfig.load()
        assert cfg.model == "openai/gpt-5"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corp_profile.config'`

- [ ] **Step 3: Write config.py**

```python
# src/corp_profile/config.py
"""Pipeline configuration — per-stage config classes with config.toml + env var resolution."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from pydantic import BaseModel


def load_config() -> dict:
    """Load config.toml from CWD or package root."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    for candidate in [Path("config.toml"), Path(__file__).resolve().parents[3] / "config.toml"]:
        if candidate.exists():
            with open(candidate, "rb") as f:
                return tomllib.load(f)
    return {}


def _resolve_str(env_key: str, section: dict, toml_key: str, default: str) -> str:
    return os.environ.get(env_key) or section.get(toml_key, default)


def _resolve_bool(env_key: str, section: dict, toml_key: str, default: bool) -> bool:
    env_val = os.environ.get(env_key)
    if env_val is not None:
        return env_val.lower() in ("true", "1", "yes")
    toml_val = section.get(toml_key)
    if toml_val is not None:
        return bool(toml_val)
    return default


class PipelineConfig(BaseModel):
    """Top-level pipeline toggles (maps to [pipeline] in config.toml)."""

    enrich: bool = False
    web: bool = False

    @classmethod
    def load(cls) -> PipelineConfig:
        cfg = load_config().get("pipeline", {})
        return cls(
            enrich=_resolve_bool("CORPPROFILE_ENRICH", cfg, "enrich", False),
            web=_resolve_bool("CORPPROFILE_WEB", cfg, "web", False),
        )


class EnrichConfig(BaseModel):
    """Config for clean/enrich/refine stages (maps to [enrich] in config.toml)."""

    model: str
    aws_region: str | None = None
    aws_profile: str | None = None

    @classmethod
    def load(cls) -> EnrichConfig:
        cfg = load_config()
        enrich = cfg.get("enrich", {})
        aws = cfg.get("aws", {})

        # Support deprecated CORPPROFILE_LLM_MODEL env var
        model = os.environ.get("CORPPROFILE_ENRICH_MODEL")
        if not model:
            deprecated = os.environ.get("CORPPROFILE_LLM_MODEL")
            if deprecated:
                warnings.warn(
                    "CORPPROFILE_LLM_MODEL is deprecated, use CORPPROFILE_ENRICH_MODEL",
                    DeprecationWarning,
                    stacklevel=2,
                )
                model = deprecated
        if not model:
            model = enrich.get("model")
        if not model:
            raise RuntimeError(
                "Enrich model not configured. Set model in [enrich] in config.toml, "
                "or set CORPPROFILE_ENRICH_MODEL env var."
            )

        return cls(
            model=model,
            aws_region=_resolve_str("AWS_DEFAULT_REGION", aws, "region", "") or None,
            aws_profile=_resolve_str("AWS_PROFILE", aws, "profile", "") or None,
        )


class ResearchConfig(BaseModel):
    """Config for the research command (maps to [research] in config.toml)."""

    model: str
    provider: str = "auto"
    aws_region: str | None = None
    aws_profile: str | None = None

    @classmethod
    def load(cls) -> ResearchConfig:
        cfg = load_config()
        research = cfg.get("research", {})
        aws = cfg.get("aws", {})

        model = _resolve_str("CORPPROFILE_RESEARCH_MODEL", research, "model", "")
        if not model:
            raise RuntimeError(
                "Research model not configured. Set model in [research] in config.toml, "
                "or set CORPPROFILE_RESEARCH_MODEL env var."
            )

        return cls(
            model=model,
            provider=_resolve_str("CORPPROFILE_RESEARCH_PROVIDER", research, "provider", "auto"),
            aws_region=_resolve_str("AWS_DEFAULT_REGION", aws, "region", "") or None,
            aws_profile=_resolve_str("AWS_PROFILE", aws, "profile", "") or None,
        )


class WebConfig(BaseModel):
    """Config for web search enrichment stage (maps to [web] in config.toml)."""

    model: str
    provider: str = "auto"
    aws_region: str | None = None
    aws_profile: str | None = None

    @classmethod
    def load(cls) -> WebConfig:
        cfg = load_config()
        web = cfg.get("web", {})
        aws = cfg.get("aws", {})

        model = _resolve_str("CORPPROFILE_WEB_MODEL", web, "model", "")
        if not model:
            raise RuntimeError(
                "Web search model not configured. Set model in [web] in config.toml, "
                "or set CORPPROFILE_WEB_MODEL env var."
            )

        return cls(
            model=model,
            provider=_resolve_str("CORPPROFILE_WEB_PROVIDER", web, "provider", "auto"),
            aws_region=_resolve_str("AWS_DEFAULT_REGION", aws, "region", "") or None,
            aws_profile=_resolve_str("AWS_PROFILE", aws, "profile", "") or None,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all tests, once config.toml is updated in Task 2)

- [ ] **Step 5: Commit**

```bash
git add src/corp_profile/config.py tests/test_config.py
git commit -m "feat: add per-stage config classes (PipelineConfig, EnrichConfig, ResearchConfig, WebConfig)"
```

---

### Task 2: Update config.toml to new structure

**Files:**
- Modify: `config.toml`

- [ ] **Step 1: Rewrite config.toml**

```toml
[pipeline]
enrich = false                # default for --enrich CLI flag
web = false                   # default for --web CLI flag

[research]
model = "openai/gpt-5-mini"  # LLM for research command
provider = "auto"             # search backend: "exa", "openai", or "auto"

[enrich]
model = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"

[web]
model = "openai/gpt-5-mini"  # LLM for web enrichment stage
provider = "auto"             # search backend: "exa", "openai", or "auto"

[aws]
region = "us-east-2"
```

- [ ] **Step 2: Run config tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 3: Run all existing tests to check for breakage**

Run: `uv run pytest tests/ -v`
Expected: Some tests in `test_enrich.py` may fail (they import old `EnrichConfig` with `web_search` field). This is expected — we fix them in Task 4.

- [ ] **Step 4: Commit**

```bash
git add config.toml
git commit -m "refactor: restructure config.toml into per-stage sections ([pipeline], [research], [enrich], [web])"
```

---

### Task 3: Update __init__.py lazy-load and add backward compat

**Files:**
- Modify: `src/corp_profile/__init__.py`

- [ ] **Step 1: Update lazy-load to import from config.py**

Replace the `__getattr__` function:

```python
def __getattr__(name: str):
    """Lazy-load enrichment exports to avoid importing LLM deps eagerly."""
    if name == "EnrichConfig":
        from .config import EnrichConfig
        return EnrichConfig
    if name == "enrich_profile":
        from .enrich import enrich_profile
        return enrich_profile
    raise AttributeError(f"module 'corp_profile' has no attribute {name!r}")
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from corp_profile import EnrichConfig; print(EnrichConfig)"`
Expected: `<class 'corp_profile.config.EnrichConfig'>`

- [ ] **Step 3: Commit**

```bash
git add src/corp_profile/__init__.py
git commit -m "refactor: update lazy-load to import EnrichConfig from config.py"
```

---

### Task 4: Refactor enrich.py to use new config classes

**Files:**
- Modify: `src/corp_profile/enrich.py`
- Modify: `tests/test_enrich.py`

- [ ] **Step 1: Rewrite enrich.py**

Key changes:
- Remove `load_config()` (now in `config.py`)
- Remove `EnrichConfig` class (now in `config.py`)
- Import `EnrichConfig`, `WebConfig`, `PipelineConfig` from `config`
- Accept both `EnrichConfig` and `WebConfig` in `enrich_profile()`
- Skip enrich stage when `web_config` is provided
- Remove dead `_refine_estimates_stage` async function

```python
"""LLM-powered profile enrichment pipeline."""

from __future__ import annotations

import json

from .config import EnrichConfig, WebConfig, load_config
from .llm import get_provider
from .profile import CompanyProfile

from . import prompts


def enrich_profile(
    profile: CompanyProfile,
    config: EnrichConfig,
    web_config: WebConfig | None = None,
) -> tuple[CompanyProfile, list[str]]:
    """Run the enrichment pipeline on a profile.

    Pipeline ordering:
      --enrich only:  clean → enrich → refine
      --web:          clean → web search → refine (enrich skipped)

    Returns (enriched_profile, list_of_changes).
    """
    provider = get_provider(config.model, aws_region=config.aws_region, aws_profile=config.aws_profile)
    all_changes: list[str] = []

    def _strip_markdown_fences(text: str) -> str:
        """Strip ```json ... ``` wrappers that LLMs often add."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _apply_stage(
        raw: str, stage_name: str
    ) -> CompanyProfile | None:
        """Parse LLM response, extract changes, return updated profile or None."""
        try:
            data = json.loads(_strip_markdown_fences(raw))
        except json.JSONDecodeError:
            all_changes.append(f"[{stage_name}] WARNING: LLM returned invalid JSON, skipping")
            return None
        if "changes" in data:
            all_changes.extend(data["changes"])
        if "profile" in data:
            try:
                return CompanyProfile.model_validate(data["profile"])
            except Exception:
                all_changes.append(
                    f"[{stage_name}] WARNING: LLM returned invalid profile data, skipping"
                )
        return None

    # Stage 1: Clean & validate (always runs)
    clean_messages = [
        {"role": "system", "content": prompts.CLEAN_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
    ]
    clean_raw = provider.complete(clean_messages, json_mode=True)
    profile = _apply_stage(clean_raw, "clean") or profile

    # Stage 2: Web search OR LLM-only enrich (never both)
    if web_config:
        # Web search replaces enrich — real data with citations
        search_slug = web_config.model
        search_provider = get_provider(
            search_slug,
            aws_region=web_config.aws_region or config.aws_region,
            aws_profile=web_config.aws_profile or config.aws_profile,
        )
        search_messages = [
            {"role": "system", "content": prompts.WEB_SEARCH_ENRICH_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
        ]
        search_raw = search_provider.complete(
            search_messages, json_mode=True, web_search=True
        )
        profile = _apply_stage(search_raw, "web_search") or profile
    else:
        # LLM-only enrich (best effort, no web)
        enrich_messages = [
            {"role": "system", "content": prompts.ENRICH_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
        ]
        enrich_raw = provider.complete(enrich_messages, json_mode=True)
        profile = _apply_stage(enrich_raw, "enrich") or profile

    # Stage 3: Refine guestimator estimates
    if profile.material_asset_types:
        refine_messages = [
            {"role": "system", "content": prompts.REFINE_ESTIMATES_SYSTEM},
            {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
        ]
        refine_raw = provider.complete(refine_messages, json_mode=True)
        try:
            refine_data = json.loads(_strip_markdown_fences(refine_raw))
        except json.JSONDecodeError:
            all_changes.append("[refine_estimates] WARNING: LLM returned invalid JSON, skipping")
            refine_data = {}
        if refine_data:
            from .profile import refine_estimates
            profile = refine_estimates(profile, refine_data)
            all_changes.append("Refined guestimator asset count estimates via LLM")

    return profile, all_changes
```

- [ ] **Step 2: Update test_enrich.py**

```python
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
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_enrich.py -v`
Expected: PASS

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: PASS (33 tests)

- [ ] **Step 5: Commit**

```bash
git add src/corp_profile/enrich.py tests/test_enrich.py
git commit -m "refactor: enrich.py uses new config classes, skips enrich when web enabled"
```

---

### Task 5: Update __main__.py CLI

**Files:**
- Modify: `src/corp_profile/__main__.py`

- [ ] **Step 1: Update CLI — rename --llm to --enrich with alias**

Replace the build command argument setup and `_run_enrich`:

```python
"""CLI entry point: python -m corp_profile."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from .profile import (
    build_profile,
    build_profile_from_file,
    save_profile,
    save_profile_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="corp-profile",
        description="Build rich company context documents from corp-graph or JSON files",
    )
    sub = parser.add_subparsers(dest="command")

    # build command
    build_cmd = sub.add_parser("build", help="Build a company profile")
    build_source = build_cmd.add_mutually_exclusive_group(required=True)
    build_source.add_argument(
        "identifier", nargs="?", default=None,
        help="ISIN, LEI, issuer_id, or company name to look up in corp-graph DB",
    )
    build_source.add_argument("--from-file", help="Load profile from JSON file")
    build_cmd.add_argument(
        "-o", "--output", help="Also save profile JSON to this path"
    )
    build_cmd.add_argument(
        "--enrich", "--llm", action="store_true", dest="enrich",
        help="Run LLM enrichment on the profile",
    )
    build_cmd.add_argument(
        "--web", action="store_true",
        help="Enable web search during enrichment (implies --enrich)",
    )

    args = parser.parse_args()

    if args.command == "build":
        from .config import PipelineConfig

        pipeline_cfg = PipelineConfig.load()
        if pipeline_cfg.enrich:
            args.enrich = True
        if pipeline_cfg.web:
            args.web = True

        # --web implies --enrich
        if args.web:
            args.enrich = True

        try:
            if args.from_file:
                profile = build_profile_from_file(args.from_file)
            else:
                profile = build_profile(args.identifier)
        except LookupError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if args.enrich:
            profile = _run_enrich(profile, web=args.web)

        if args.output:
            save_profile(profile, args.output)
            print(f"Profile JSON saved to {args.output}", file=sys.stderr)

        out_path = save_profile_markdown(profile)
        print(f"Saved to {out_path}", file=sys.stderr)

    else:
        parser.print_help()
        sys.exit(1)


def _run_enrich(profile, *, web: bool = False):
    """Run enrichment and print changes to stderr."""
    from .config import EnrichConfig, WebConfig
    from .enrich import enrich_profile

    try:
        enrich_config = EnrichConfig.load()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    web_config = None
    if web:
        try:
            web_config = WebConfig.load()
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        profile, changes = enrich_profile(profile, enrich_config, web_config=web_config)
    except Exception as e:
        print(f"Enrichment failed: {e}", file=sys.stderr)
        sys.exit(1)
    if changes:
        print("Enrichment changes:", file=sys.stderr)
        for c in changes:
            print(f"  - {c}", file=sys.stderr)
    return profile


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 3: Verify CLI help shows --enrich**

Run: `uv run python -m corp_profile build --help`
Expected: Shows `--enrich/--llm` option

- [ ] **Step 4: Commit**

```bash
git add src/corp_profile/__main__.py
git commit -m "refactor: rename --llm to --enrich with --llm as deprecated alias, use new config classes"
```

---

## Chunk 2: Search Backend & Dependencies

### Task 6: Add exa-py dependency

**Files:**
- Modify: `pyproject.toml:14`

- [ ] **Step 1: Add exa-py to llm extra**

Change line 14 from:
```
llm = ["openai>=1.0", "anthropic[bedrock]>=0.84.0"]
```
to:
```
llm = ["openai>=1.0", "anthropic[bedrock]>=0.84.0", "exa-py>=1.0"]
```

- [ ] **Step 2: Sync dependencies**

Run: `uv sync --extra llm`
Expected: exa-py installed

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add exa-py to [llm] extra for search backend"
```

---

### Task 7: Create search backend abstraction

**Files:**
- Create: `src/corp_profile/search.py`
- Test: `tests/test_search.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_search.py
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
    @patch("corp_profile.search.Exa")
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_search.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write search.py**

```python
# src/corp_profile/search.py
"""Search backend abstraction — Exa and OpenAI web search."""

from __future__ import annotations

import os
from typing import Protocol

from pydantic import BaseModel


class SearchResult(BaseModel):
    """A single web search result."""

    title: str
    url: str
    content: str


class SearchBackend(Protocol):
    """Protocol for web search backends."""

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]: ...


WEB_SEARCH_TOOL_SCHEMA: dict = {
    "name": "web_search",
    "description": (
        "Search the web for current information about a company, "
        "its operations, subsidiaries, or assets."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "num_results": {
                "type": "integer",
                "default": 5,
                "description": "Number of results to return (1-10)",
            },
        },
        "required": ["query"],
    },
}


class ExaSearch:
    """Web search using the Exa API."""

    def __init__(self, api_key: str) -> None:
        from exa_py import Exa

        self.client = Exa(api_key=api_key)

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        response = self.client.search_and_contents(
            query,
            num_results=min(num_results, 10),
            type="auto",
            text=True,
        )
        return [
            SearchResult(
                title=r.title or "",
                url=r.url or "",
                content=r.text or "",
            )
            for r in response.results
        ]


def get_search_backend(provider: str, model: str) -> SearchBackend | None:
    """Resolve provider string to a SearchBackend, or None for OpenAI built-in.

    "auto" uses the model slug to decide: openai/ models → None, else → Exa.
    """
    if provider == "openai":
        return None

    if provider == "auto":
        model_provider = model.split("/", 1)[0] if "/" in model else model
        if model_provider == "openai":
            return None
        # Fall through to Exa for non-OpenAI models

    # provider == "exa" or auto resolved to Exa
    api_key = os.environ.get("EXA_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "EXA_API_KEY environment variable is required when search provider is "
            f"'{provider}'. Set it in .env or your environment."
        )
    return ExaSearch(api_key=api_key)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_search.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/corp_profile/search.py tests/test_search.py
git commit -m "feat: add search backend abstraction with Exa and auto-routing"
```

---

## Chunk 3: Provider Tool-Use Loop

### Task 8: Add complete_with_tools() to BedrockProvider

**Files:**
- Modify: `src/corp_profile/llm/bedrock.py`
- Test: `tests/test_tool_loop.py`

- [ ] **Step 1: Write failing test for Bedrock tool-use loop**

```python
# tests/test_tool_loop.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tool_loop.py::TestBedrockToolLoop -v`
Expected: FAIL — `AttributeError: 'BedrockProvider' has no attribute 'complete_with_tools'`

- [ ] **Step 3: Implement complete_with_tools() in BedrockProvider**

Add to `src/corp_profile/llm/bedrock.py`:

```python
def complete_with_tools(
    self,
    messages: list[dict],
    search_backend,
    json_mode: bool = False,
    max_iterations: int = 10,
) -> str:
    """Run a tool-use loop: LLM calls search → we execute → return results → repeat."""
    from ..search import WEB_SEARCH_TOOL_SCHEMA, SearchResult

    # Build Bedrock tool config
    tools = [{
        "toolSpec": {
            "name": WEB_SEARCH_TOOL_SCHEMA["name"],
            "description": WEB_SEARCH_TOOL_SCHEMA["description"],
            "inputSchema": {"json": WEB_SEARCH_TOOL_SCHEMA["parameters"]},
        }
    }]

    # Build initial messages
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

    if json_mode:
        schema = EnrichmentResponse.model_json_schema()
        tools.append({
            "toolSpec": {
                "name": "enrichment_response",
                "description": "Return the enriched company profile and list of changes made.",
                "inputSchema": {"json": schema},
            }
        })

    kwargs: dict = {
        "modelId": self.model,
        "messages": converse_messages,
        "inferenceConfig": {"maxTokens": 16384},
        "toolConfig": {"tools": tools},
    }
    if system_parts:
        kwargs["system"] = system_parts

    for _ in range(max_iterations):
        response = self.client.converse(**kwargs)
        content_blocks = response["output"]["message"]["content"]

        # Check for tool calls
        tool_calls = [b for b in content_blocks if "toolUse" in b]
        if not tool_calls:
            # No tool calls — return text
            for block in content_blocks:
                if "text" in block:
                    return block["text"]
            return ""

        # Append assistant message
        converse_messages.append({
            "role": "assistant",
            "content": content_blocks,
        })

        # Process each tool call
        tool_results = []
        for tc in tool_calls:
            tool_use = tc["toolUse"]
            if tool_use["name"] == "web_search":
                query = tool_use["input"].get("query", "")
                num = tool_use["input"].get("num_results", 5)
                results = search_backend.search(query, num)
                result_text = "\n\n".join(
                    f"### {r.title}\nURL: {r.url}\n{r.content}"
                    for r in results
                )
                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"text": result_text or "No results found."}],
                    }
                })
            elif tool_use["name"] == "enrichment_response":
                return json.dumps(tool_use["input"])

        converse_messages.append({
            "role": "user",
            "content": tool_results,
        })
        kwargs["messages"] = converse_messages

    # Hit max iterations — return whatever we have
    return ""
```

Also update the import at the top of bedrock.py to include `EnrichmentResponse`:

```python
from . import EnrichmentResponse
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_tool_loop.py::TestBedrockToolLoop -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/corp_profile/llm/bedrock.py tests/test_tool_loop.py
git commit -m "feat: add complete_with_tools() to BedrockProvider for Exa search loop"
```

---

### Task 9: Add complete_with_tools() to OpenAIProvider

**Files:**
- Modify: `src/corp_profile/llm/openai.py`
- Modify: `tests/test_tool_loop.py`

- [ ] **Step 1: Write failing test for OpenAI tool-use loop**

Add to `tests/test_tool_loop.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tool_loop.py::TestOpenAIToolLoop -v`
Expected: FAIL — `AttributeError: 'OpenAIProvider' has no attribute 'complete_with_tools'`

- [ ] **Step 3: Implement complete_with_tools() in OpenAIProvider**

Add to `src/corp_profile/llm/openai.py`:

```python
def complete_with_tools(
    self,
    messages: list[dict],
    search_backend,
    json_mode: bool = False,
    max_iterations: int = 10,
) -> str:
    """Run a tool-use loop using OpenAI function calling with Exa search."""
    import json as _json
    from ..search import WEB_SEARCH_TOOL_SCHEMA

    tools = [{
        "type": "function",
        "function": {
            "name": WEB_SEARCH_TOOL_SCHEMA["name"],
            "description": WEB_SEARCH_TOOL_SCHEMA["description"],
            "parameters": WEB_SEARCH_TOOL_SCHEMA["parameters"],
        },
    }]

    conv_messages = list(messages)

    kwargs: dict = {
        "model": self.model,
        "messages": conv_messages,
        "tools": tools,
    }
    if json_mode:
        kwargs["response_format"] = EnrichmentResponse

    for _ in range(max_iterations):
        if json_mode:
            response = self.client.beta.chat.completions.parse(**kwargs)
        else:
            response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            return msg.content or ""

        # Append assistant message with tool calls
        conv_messages.append(msg)

        # Execute each tool call
        for tc in msg.tool_calls:
            if tc.function.name == "web_search":
                args = _json.loads(tc.function.arguments)
                query = args.get("query", "")
                num = args.get("num_results", 5)
                results = search_backend.search(query, num)
                result_text = "\n\n".join(
                    f"### {r.title}\nURL: {r.url}\n{r.content}"
                    for r in results
                )
                conv_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text or "No results found.",
                })

        kwargs["messages"] = conv_messages

    return ""
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_tool_loop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/corp_profile/llm/openai.py tests/test_tool_loop.py
git commit -m "feat: add complete_with_tools() to OpenAIProvider for Exa search loop"
```

---

## Chunk 4: Research Command

### Task 10: Add RESEARCH_SYSTEM_PROMPT

**Files:**
- Modify: `src/corp_profile/prompts.py`

- [ ] **Step 1: Read existing prompts.py to understand the pattern**

Read `src/corp_profile/prompts.py` to see how other prompts reference the schema.

- [ ] **Step 2: Add RESEARCH_SYSTEM_PROMPT**

Append to `prompts.py`:

```python
RESEARCH_SYSTEM_PROMPT = f"""You are a corporate research analyst. Your task is to build a comprehensive
company profile from scratch using web search.

You have access to a `web_search` tool. Use it to find information about the company.
Search strategically — start with the company overview, then dig into corporate structure,
operations, and assets.

## Target Output Schema

Return your findings as JSON matching this schema:
```json
{json.dumps(_SCHEMA, indent=2)}
```

## What to Research

Search for and populate:
1. **Company identity**: legal_name, description, primary_industry, jurisdiction, lei
2. **Identifiers**: ISINs (isin_list), LEI, aliases/alternative names
3. **Corporate structure**: Key subsidiaries with jurisdiction, ownership percentage, LEI where available
4. **Geographic footprint**: operating_countries (ISO alpha-2 codes), business_segments
5. **Asset profile**: Types of physical assets the company operates (for material_asset_types with estimated counts), estimated total asset count
6. **Discovered context**: Any notable assets you find (for discovered_assets with name, type, address, coordinates)

## Guidelines

- Use multiple searches to build a complete picture — don't rely on a single search
- For subsidiaries, focus on operationally significant entities (those that own physical assets), not holding companies or SPVs
- For material_asset_types, estimate realistic counts based on what you learn about the company's scale
- Use ISO 3166-1 alpha-2 codes for countries and jurisdictions
- Include sources in your changes list (e.g. "Found 12 subsidiaries via TotalEnergies 2024 annual report")
- If you have a seed profile, fill in gaps rather than overwriting existing data

Return your response as JSON with two keys:
- "profile": the complete CompanyProfile
- "changes": list of strings describing what you found and where
"""
```

- [ ] **Step 3: Commit**

```bash
git add src/corp_profile/prompts.py
git commit -m "feat: add RESEARCH_SYSTEM_PROMPT for research command"
```

---

### Task 11: Create research.py

**Files:**
- Create: `src/corp_profile/research.py`
- Test: `tests/test_research.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_research.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_research.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write research.py**

```python
# src/corp_profile/research.py
"""Research command — build a CompanyProfile from scratch via LLM + web search."""

from __future__ import annotations

import json

from .config import EnrichConfig, ResearchConfig, load_config
from .llm import get_provider
from .profile import CompanyProfile
from .search import get_search_backend


def research_profile(
    identifier: str | None = None,
    name: str | None = None,
    seed: CompanyProfile | None = None,
    config: ResearchConfig | None = None,
) -> tuple[CompanyProfile, list[str]]:
    """Build a CompanyProfile from scratch via LLM + web search.

    Pipeline:
      1. Research: LLM + web search tool-use loop → initial CompanyProfile
      2. Clean: fix names, normalize jurisdictions, dedup (existing stage)

    Returns (profile, list_of_changes).
    """
    if config is None:
        config = ResearchConfig.load()

    # Validate we have at least one identifier
    if seed:
        has_id = bool(
            seed.issuer_id
            or seed.isin_list
            or seed.lei
        )
        if not has_id and not identifier:
            raise ValueError(
                "Seed profile must contain at least one of: "
                "issuer_id, isin_list, or lei (or pass identifier argument)."
            )

    all_changes: list[str] = []

    # --- Stage 1: Research via LLM + web search ---
    from . import prompts

    research_provider = get_provider(
        config.model,
        aws_region=config.aws_region,
        aws_profile=config.aws_profile,
    )
    backend = get_search_backend(config.provider, model=config.model)

    # Build user message with identifier + optional context
    user_parts = []
    if identifier:
        user_parts.append(f"Company identifier: {identifier}")
    if name:
        user_parts.append(f"Company name: {name}")
    if seed:
        user_parts.append(f"Partial profile (fill gaps):\n{json.dumps(seed.model_dump(), default=str)}")
    if not user_parts:
        raise ValueError("Must provide identifier, name, or seed.")

    messages = [
        {"role": "system", "content": prompts.RESEARCH_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]

    if backend:
        raw = research_provider.complete_with_tools(
            messages, search_backend=backend, json_mode=True,
        )
    else:
        # OpenAI built-in web search
        raw = research_provider.complete(messages, json_mode=True, web_search=True)

    # Parse research result
    def _strip_markdown_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    try:
        data = json.loads(_strip_markdown_fences(raw))
    except json.JSONDecodeError:
        raise RuntimeError("Research stage returned invalid JSON from LLM.")

    if "changes" in data:
        all_changes.extend(data["changes"])
    if "profile" in data:
        profile = CompanyProfile.model_validate(data["profile"])
    else:
        raise RuntimeError("Research stage did not return a profile.")

    # --- Stage 2: Clean ---
    try:
        enrich_config = EnrichConfig.load()
    except RuntimeError:
        # If no enrich config, use research model for clean
        enrich_config = EnrichConfig(
            model=config.model,
            aws_region=config.aws_region,
            aws_profile=config.aws_profile,
        )

    clean_provider = get_provider(
        enrich_config.model,
        aws_region=enrich_config.aws_region,
        aws_profile=enrich_config.aws_profile,
    )
    clean_messages = [
        {"role": "system", "content": prompts.CLEAN_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(profile.model_dump(), default=str)},
    ]
    clean_raw = clean_provider.complete(clean_messages, json_mode=True)
    try:
        clean_data = json.loads(_strip_markdown_fences(clean_raw))
        if "changes" in clean_data:
            all_changes.extend(clean_data["changes"])
        if "profile" in clean_data:
            profile = CompanyProfile.model_validate(clean_data["profile"])
    except (json.JSONDecodeError, Exception):
        all_changes.append("[clean] WARNING: LLM returned invalid JSON, skipping")

    return profile, all_changes
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_research.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/corp_profile/research.py tests/test_research.py
git commit -m "feat: add research_profile() for building profiles from scratch via web search"
```

---

### Task 12: Add `research` subcommand to CLI

**Files:**
- Modify: `src/corp_profile/__main__.py`

- [ ] **Step 1: Add research subcommand to the argument parser**

Add after the `build_cmd` setup:

```python
# research command
research_cmd = sub.add_parser("research", help="Research a company via web search (no DB needed)")
research_source = research_cmd.add_mutually_exclusive_group(required=True)
research_source.add_argument(
    "identifier", nargs="?", default=None,
    help="ISIN, LEI, issuer_id, or company name",
)
research_source.add_argument("--seed", help="Partial JSON file to seed research")
research_cmd.add_argument("--name", help="Company name to help search accuracy")
research_cmd.add_argument(
    "-o", "--output", help="Also save profile JSON to this path"
)
```

Add the `research` command handler in `main()`:

```python
elif args.command == "research":
    from .config import ResearchConfig
    from .research import research_profile
    from .profile import build_profile_from_file, save_profile, save_profile_markdown

    try:
        config = ResearchConfig.load()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    seed = None
    identifier = args.identifier
    if args.seed:
        seed = build_profile_from_file(args.seed)
        if not identifier:
            identifier = seed.issuer_id or (seed.isin_list[0] if seed.isin_list else None)

    try:
        profile, changes = research_profile(
            identifier=identifier,
            name=args.name,
            seed=seed,
            config=config,
        )
    except Exception as e:
        print(f"Research failed: {e}", file=sys.stderr)
        sys.exit(1)

    if changes:
        print("Research findings:", file=sys.stderr)
        for c in changes:
            print(f"  - {c}", file=sys.stderr)

    if args.output:
        save_profile(profile, args.output)
        print(f"Profile JSON saved to {args.output}", file=sys.stderr)

    out_path = save_profile_markdown(profile)
    print(f"Saved to {out_path}", file=sys.stderr)
```

- [ ] **Step 2: Verify CLI help**

Run: `uv run python -m corp_profile research --help`
Expected: Shows identifier, --seed, --name, -o options

- [ ] **Step 3: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/corp_profile/__main__.py
git commit -m "feat: add 'research' subcommand to CLI for DB-free profile building"
```

---

## Chunk 5: company_universe & Downstream Updates

### Task 13: Add material_assets_types to company_universe materialization

**Files:**
- Modify: `/Users/nikolai.tennant/Documents/GitHub/corp-graph/src/materialize.py`

- [ ] **Step 1: Read materialize.py to find the Step 7 LEFT JOIN block**

Find the exact location where the big SELECT builds `company_universe`. Look for the existing `estimated_assets_count` column to see where asset data is joined.

- [ ] **Step 2: Add LEFT JOIN to asset_estimates and include material_assets_types**

Add `ae.material_assets_types` to the SELECT column list and a corresponding `LEFT JOIN asset_estimates ae ON ae.issuer_id = eff.issuer_id` if not already joined.

- [ ] **Step 3: Rebuild company_universe**

Run: `cd /Users/nikolai.tennant/Documents/GitHub/corp-graph && python -m src.pipeline build-universe`
Expected: Completes without error

- [ ] **Step 4: Verify column exists**

Run SQL via corp-graph MCP: `SELECT material_assets_types FROM company_universe WHERE material_assets_types IS NOT NULL LIMIT 1`
Expected: Returns a JSONB array

- [ ] **Step 5: Commit in corp-graph repo**

```bash
cd /Users/nikolai.tennant/Documents/GitHub/corp-graph
git add src/materialize.py
git commit -m "feat: add material_assets_types to company_universe from asset_estimates"
```

---

### Task 14: Remove asset_estimates query from profile.py

**Files:**
- Modify: `src/corp_profile/profile.py:287-303`

- [ ] **Step 1: Remove the asset_estimates query block**

Remove lines 287-303 (the `SELECT ... FROM asset_estimates` query and its parsing). Replace with reading from the `company_universe` row:

After the `company_profiles` section (~line 275), add:

```python
# 6. Asset estimates (from company_universe row)
profile.estimated_asset_count = row.get("estimated_assets_count")
raw_types = row.get("material_assets_types")
if isinstance(raw_types, list):
    parsed: list[MaterialAssetType] = []
    for t in raw_types:
        if isinstance(t, dict):
            if "type" in t:
                parsed.append(MaterialAssetType.model_validate(t))
            else:
                for name, count in t.items():
                    parsed.append(MaterialAssetType(
                        type=name,
                        count=count if isinstance(count, int) else None,
                    ))
        else:
            parsed.append(MaterialAssetType(type=str(t)))
    profile.material_asset_types = parsed
```

- [ ] **Step 2: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/corp_profile/profile.py
git commit -m "refactor: read material_assets_types from company_universe, remove asset_estimates query"
```

---

### Task 15: Update asset-search-v2 config

**Files:**
- Modify: `/Users/nikolai.tennant/Documents/GitHub/asset-search-v2/config.toml`
- Modify: `/Users/nikolai.tennant/Documents/GitHub/asset-search-v2/src/asset_search/config.py`

- [ ] **Step 1: Update asset-search-v2 config.toml**

Replace the `[profile]` section with:

```toml
# ── corp-profile (mirrors corp-profile's own config) ──────────────────────────
[profile]
enrich = false
web = false

[profile.research]
model = "openai/gpt-5-mini"
provider = "exa"

[profile.enrich]
model = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"

[profile.web]
model = "openai/gpt-5-mini"
provider = "auto"
```

- [ ] **Step 2: Update asset-search-v2 Config class**

In `src/asset_search/config.py`, update:
- Rename `profile_llm` → `profile_enrich`
- Rename `profile_web_search` → `profile_web`
- Remove `profile_web_search_model`
- Add `profile_research_model`, `profile_research_provider`
- Add `profile_web_model`, `profile_web_provider`
- Update `profile_enrich_config()` builder method
- Add `profile_pipeline_config()`, `profile_web_config()`, `profile_research_config()` builder methods
- Update all the `_resolve_*` calls to read from new TOML sections

- [ ] **Step 3: Run asset-search-v2 tests if available**

Run: `cd /Users/nikolai.tennant/Documents/GitHub/asset-search-v2 && uv run pytest tests/ -v --timeout=30 2>&1 | head -50`
Expected: Config-related tests pass (some tests may require infra)

- [ ] **Step 4: Commit in asset-search-v2 repo**

```bash
cd /Users/nikolai.tennant/Documents/GitHub/asset-search-v2
git add config.toml src/asset_search/config.py
git commit -m "refactor: update corp-profile config mapping for new per-stage config structure"
```

---

## Chunk 6: Final Verification

### Task 16: End-to-end verification

- [ ] **Step 1: Run all corp-profile tests**

Run: `cd /Users/nikolai.tennant/Documents/GitHub/corp-profile && uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Verify CLI --from-file still works**

Run: `uv run python -m corp_profile build --from-file examples/totalenergies.json`
Expected: Markdown output saved to `outputs/`

- [ ] **Step 3: Verify CLI research --help**

Run: `uv run python -m corp_profile research --help`
Expected: Shows all options

- [ ] **Step 4: Update CLAUDE.md with new config structure and research command**

Add `research` command to the CLI docs. Update config section to reflect `[pipeline]`, `[research]`, `[enrich]`, `[web]`. Mention `--enrich` replaces `--llm`.

- [ ] **Step 5: Commit CLAUDE.md**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for config redesign and research command"
```
