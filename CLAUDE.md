# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

corp-profile builds rich company context documents from a corp-graph Postgres database, or from JSON files when the DB isn't available. Output is structured markdown optimized for downstream LLM asset search pipelines. `CompanyProfile` is the local Pydantic view defined in this repo — not the `company_profiles` table in corp-graph, which is just one of the upstream data sources.

## Setup & Commands

```bash
uv sync                                          # Install base dependencies
uv sync --extra llm                               # Add LLM support (OpenAI + Bedrock)
cp .env.example .env                              # Then edit with your config
```

```bash
# CLI — all commands save markdown to outputs/
python -m corp_profile build FR0000120271                              # Build from DB (ISIN, LEI, issuer_id, or name)
python -m corp_profile build --from-file examples/totalenergies.json  # Build from JSON (no DB needed)
python -m corp_profile build FR0000120271 -o out.json                 # Also save profile JSON
python -m corp_profile build --from-file examples/totalenergies.json --llm        # + LLM enrichment
python -m corp_profile build --from-file examples/totalenergies.json --llm --web  # + LLM + web search
# --web implies --llm; both can be defaulted on in config.toml [profile]
```

```bash
# Tests
uv run pytest tests/ -v                           # Run all tests (30 total)
uv run pytest tests/test_profile_io.py -v         # Profile I/O round-trip tests
uv run pytest tests/test_llm.py -v                # LLM provider tests
uv run pytest tests/test_enrich.py -v             # Enrichment pipeline tests
uv run pytest tests/test_demo.py -v -s            # Demo integration test (prints rendered output)
uv run pytest tests/test_llm.py::TestOpenAIProvider -v  # Single test class
```

## Architecture

**Data flow:** Source (DB/JSON) → `CompanyProfile` → optional `enrich_profile()` → `build_context_document()` → markdown in `outputs/`

**Data models** (`profile.py`): `CompanyProfile` contains typed sub-models — `Subsidiary`, `Asset`, `DiscoveredAsset`, `MaterialAssetType`. These are proper Pydantic models (not `list[dict]`) so they work with OpenAI's structured outputs.

**Entity resolution** (`profile.py:_resolve_entity`): Flexible 5-step lookup — ISIN on parent entity → ISIN via securities table → LEI → issuer_id → name/alias match → name prefix match.

**LLM provider routing** (`llm/__init__.py`): Provider-agnostic slug format `provider/model` (e.g. `openai/gpt-5`, `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`). `parse_model_slug()` splits, `get_provider()` returns the right adapter.

**Enrichment pipeline** (`enrich.py`): Three stages — clean (fix names, normalize jurisdictions, dedup) → optional web search (OpenAI Responses API only) → enrich (improve descriptions, add context). Each stage returns `{"profile": {...}, "changes": [...]}`.

**Web search** uses `responses.parse()` with Pydantic `EnrichmentResponse` model for structured output — this works with web search tools unlike `json_object` mode which OpenAI blocks. Bedrock does not support web search.

**Config structure** (`config.toml`): `[profile]` section has `llm` and `web` toggles (matching CLI flags). `[llm]` section has `model` and `web_search_model`. `[aws]` section has `region`/`profile`. CLI flags override config defaults.

**Config precedence** (`EnrichConfig.load()`): env vars → `config.toml` → defaults. Secrets (API keys) stay in `.env`, app config in `config.toml`.

**Lazy imports** (`__init__.py`): `EnrichConfig` and `enrich_profile` are lazy-loaded via `__getattr__` to avoid pulling in LLM dependencies when not needed.

**Markdown renderer** (`build_context_document()`): Outputs sections optimized for asset search: Company Identity, Geographic Footprint, Corporate Structure, Asset Inventory (capped at 10 samples), Discovery Gaps, Search Guidance. Country codes resolved to full names via `pycountry`.

## Key Dependencies

- Python >=3.13, managed with `uv`, built with `hatchling`
- `psycopg[binary]` (Postgres driver, v3 API with dict row factory)
- `pydantic` v2 (data models, structured output schemas)
- `pycountry` (ISO country code → name resolution)
- `python-dotenv` (loaded at CLI startup in `__main__.py`)
- Optional `[llm]` extra: `openai` (web search), `anthropic[bedrock]` (Bedrock LLM + boto3)

## Environment Variables

- `CORPGRAPH_DB_URL` — Postgres connection string (required for DB mode)
- `CORPPROFILE_LLM_MODEL` — LLM slug (overrides `config.toml` `[llm].model`)
- `OPENAI_API_KEY` — required for `--web` (loaded from `.env` via dotenv)
