# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

corp-profile builds rich company context documents from a corp-graph Postgres database, or from JSON files when the DB isn't available. Output is structured markdown optimized for downstream LLM asset search pipelines. `CompanyProfile` is the local Pydantic view defined in this repo — not the `company_profiles` table in corp-graph, which is just one of the upstream data sources.

## Setup & Commands

```bash
uv sync                                          # Install base dependencies
uv sync --extra llm                               # Add LLM + search support (OpenAI, Bedrock, Exa)
cp .env.example .env                              # Then edit with your config
```

```bash
# CLI — all commands save markdown to outputs/
python -m corp_profile build FR0000120271                              # Build from DB (ISIN, LEI, issuer_id, or name)
python -m corp_profile build --from-file examples/totalenergies.json  # Build from JSON (no DB needed)
python -m corp_profile build FR0000120271 -o out.json                 # Also save profile JSON
python -m corp_profile build --from-file examples/totalenergies.json --enrich       # + LLM enrichment
python -m corp_profile build --from-file examples/totalenergies.json --enrich --web # + LLM + web search
# --web implies --enrich; both can be defaulted on in config.toml [pipeline]

python -m corp_profile research "TotalEnergies"                        # Research from scratch via LLM + web search
python -m corp_profile research FR0000120271 --name "TotalEnergies"     # With ISIN + name hint
python -m corp_profile research --seed examples/totalenergies.json     # Seed from partial JSON
python -m corp_profile research "TotalEnergies" -o out.json            # Also save profile JSON
```

```bash
# Tests
uv run pytest tests/ -v                                 # Run all tests
uv run pytest tests/test_config.py -v                   # Config class tests
uv run pytest tests/test_profile_io.py -v               # Profile I/O round-trip tests
uv run pytest tests/test_llm.py -v                      # LLM provider tests
uv run pytest tests/test_enrich.py -v                   # Enrichment pipeline tests
uv run pytest tests/test_search.py -v                   # Search backend tests
uv run pytest tests/test_research.py -v                 # Research command tests
uv run pytest tests/test_tool_loop.py -v                # Provider tool-use loop tests
uv run pytest tests/test_estimate_refinement.py -v      # Asset estimate refinement tests
uv run pytest tests/test_demo.py -v -s                  # Demo integration test (prints rendered output)
uv run pytest tests/test_llm.py::TestOpenAIProvider -v  # Single test class
```

## Architecture

Source code is under `src/corp_profile/` (src layout).

**Data flow:** Source (DB/JSON) → `CompanyProfile` → optional `enrich_profile()` → `build_context_document()` → markdown in `outputs/`

**Research flow:** Identifier/name → `research_profile()` (LLM + web search tool loop → clean) → `CompanyProfile`

**Data models** (`profile.py`): `CompanyProfile` contains typed sub-models — `Subsidiary`, `Asset`, `DiscoveredAsset`, `MaterialAssetType`. All inherit from `_NullSafeModel` which coerces DB NULLs to empty strings for `str` fields. These are proper Pydantic models (not `list[dict]`) so they work with OpenAI's structured outputs.

**Entity resolution** (`profile.py:_resolve_entity`): Flexible 5-step lookup — ISIN on parent entity → ISIN via securities table → LEI → issuer_id → name/alias match → name prefix match.

**DB layer** (`db.py`): `get_connection()` returns a psycopg connection with dict row factory, reading `CORPGRAPH_DB_URL` from env.

**LLM provider routing** (`llm/__init__.py`): Provider-agnostic slug format `provider/model` (e.g. `openai/gpt-5`, `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`). `parse_model_slug()` splits, `get_provider()` returns the right adapter. Both providers support `complete()` and `complete_with_tools()` for search tool-use loops. JSON mode is enforced via OpenAI structured outputs (`beta.chat.completions.parse()`) or Bedrock tool-use with `toolChoice` force — both guarantee `EnrichmentResponse` schema.

**Enrichment pipeline** (`enrich.py`): Pipeline ordering depends on flags:
- `--enrich` only: clean → LLM enrich → refine
- `--web`: clean → web search → refine (enrich skipped — web provides real data)
- Web search works with any provider: OpenAI (built-in search) or Bedrock + Exa tool loop.

Each stage returns `{"profile": {...}, "changes": [...]}`.

**Research command** (`research.py`): Builds profiles from scratch via LLM + web search tool-use loop, then runs the clean stage. Uses `SearchBackend` protocol for Exa search (Bedrock models) or OpenAI built-in search (OpenAI models).

**Search backend** (`search.py`): `SearchBackend` protocol with `ExaSearch` implementation. `get_search_backend()` routes: `"auto"` uses Exa for non-OpenAI models, `None` for OpenAI models (built-in search). `WEB_SEARCH_TOOL_SCHEMA` defines the tool for LLM function calling.

**Prompts** (`prompts.py`): All LLM system prompts live here — `CLEAN_SYSTEM_PROMPT`, `ENRICH_SYSTEM_PROMPT`, `WEB_SEARCH_ENRICH_SYSTEM_PROMPT`, `REFINE_ESTIMATES_SYSTEM`, `RESEARCH_SYSTEM_PROMPT`. Each includes the `CompanyProfile` JSON schema and expects the LLM to return `{"profile": {...}, "changes": [...]}`.

**Config structure** (`config.py` + `config.toml`): Per-stage config classes:
- `[pipeline]` → `PipelineConfig`: `enrich` and `web` toggles (matching CLI flags)
- `[research]` → `ResearchConfig`: `model`, `provider` (search backend: "exa", "openai", "auto")
- `[enrich]` → `EnrichConfig`: `model`
- `[web]` → `WebConfig`: `model`, `provider`
- `[aws]` → shared `region`/`profile`

**Config precedence**: env vars → `config.toml` → defaults. Secrets (API keys) stay in `.env`, app config in `config.toml`.

**Public API** (`__init__.py`): Two high-level functions — `corp_profile.run()` (build + optional enrich/web) and `corp_profile.research()` (from-scratch via LLM + web). All config classes and lower-level functions are lazy-exported so `import corp_profile` doesn't pull in LLM deps until needed.

**Markdown renderer** (`build_context_document()`): Outputs sections optimized for asset search: Company Identity, Geographic Footprint, Corporate Structure, Asset Inventory (capped at 10 samples), Discovery Gaps, Search Guidance. Country codes resolved to full names via `pycountry`. Subsidiaries capped at 20, ranked by data completeness.

**Tests**: `conftest.py` provides `sample_profile_data` (dict) and `sample_profile` (`CompanyProfile` instance) fixtures used across all test modules.

## Key Dependencies

- Python >=3.13, managed with `uv`, built with `hatchling`
- `psycopg[binary]` (Postgres driver, v3 API with dict row factory)
- `pydantic` v2 (data models, structured output schemas)
- `pycountry` (ISO country code → name resolution)
- `python-dotenv` (loaded at CLI startup in `__main__.py`)
- Optional `[llm]` extra: `openai` (web search), `anthropic[bedrock]` (Bedrock LLM + boto3), `exa-py` (Exa search backend)

## Environment Variables

- `CORPGRAPH_DB_URL` — Postgres connection string (required for DB mode)
- `CORPPROFILE_ENRICH_MODEL` — LLM slug (overrides `config.toml` `[enrich].model`)
- `CORPPROFILE_RESEARCH_MODEL` — LLM slug (overrides `config.toml` `[research].model`)
- `CORPPROFILE_WEB_MODEL` — LLM slug (overrides `config.toml` `[web].model`)
- `OPENAI_API_KEY` — required for OpenAI models (loaded from `.env` via dotenv)
- `EXA_API_KEY` — required when search provider is "exa" (loaded from `.env` via dotenv)

## TODO

Upstream issues to fix in **corp-graph** (not this repo):

- [ ] **Fold `asset_estimates` into `company_universe`**: `estimated_assets_count` and `material_assets_types` are NULL on canonical entities when the data lives on a resolved source entity. The `company_universe` build should join through `canonical_links` to pick up estimate data. Currently corp-profile has a fallback query to `asset_estimates` directly.
- [ ] **Materialize GLEIF addresses into `assets` table**: `issuers.legal_address` and `hq_address` (JSONB) from GLEIF/FCA sources should be inserted into `assets` with `source = 'GLEIF'` etc. so corp-profile picks them up without source-specific extraction. ~3.2M entities have address data.
- [ ] **Full asset type breakdown in guestimator**: `material_assets_types` only covers environmentally material assets. For Sprouts (380 assets, 3 material), 377 retail stores are unclassified. Either add an `estimated_asset_types` column with the full breakdown, or expand the guestimator to classify all types.

Improvements for **this repo** and **asset-discovery**:

- [ ] **Rename config sections for clarity**: In asset-discovery, `[search]` should be `[discover]` (it's the discover stage search provider). Corp-profile's `[web]` provider should be called `search_provider` or similar. General naming pass to make the config self-documenting.
- [ ] **Parallel multi-agent discovery**: Current discover stage is a single sequential agent. Redesign as Opus supervisor + Sonnet sub-agents: supervisor plans strategy (primary site, subsidiaries, regulatory, external), fans out to parallel Sonnet workers that each discover URLs independently, results merge. Eliminates Bedrock timeouts and provides ~3-4x speedup.
