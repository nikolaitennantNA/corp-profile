# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

corp-profile builds rich company context documents from a corp-graph Postgres database, or from JSON files when the DB isn't available. It can optionally enrich profiles using LLMs. `CompanyProfile` is the local Pydantic view defined in this repo — not the `company_profiles` table in corp-graph, which is just one of the upstream data sources.

## Setup & Commands

```bash
uv sync                                          # Install base dependencies
uv sync --extra openai                            # Add OpenAI support
uv sync --extra bedrock                           # Add Bedrock support
cp .env.example .env                              # Then edit with your config
```

```bash
# CLI usage — all commands save markdown to outputs/
python -m corp_profile build FR0000120271                              # Build from DB by ISIN/LEI/name
python -m corp_profile build --from-file examples/totalenergies.json  # Build from JSON
python -m corp_profile build FR0000120271 -o out.json                 # Also save profile JSON
python -m corp_profile build --from-file examples/totalenergies.json --llm        # + LLM enrichment
python -m corp_profile build --from-file examples/totalenergies.json --llm --web  # + LLM + web search
```

```bash
# Tests
uv run pytest tests/ -v                           # Run all tests
uv run pytest tests/test_profile_io.py -v         # Profile I/O tests only
uv run pytest tests/test_llm.py -v                # LLM provider tests only
uv run pytest tests/test_enrich.py -v             # Enrichment tests only
uv run pytest tests/test_llm.py::TestOpenAIProvider -v  # Single test class
```

## Architecture

```
src/corp_profile/
├── db.py            # get_connection() — psycopg dict-row connection via CORPGRAPH_DB_URL
├── profile.py       # CompanyProfile model, build_profile(isin), build_profile_from_dict/file,
│                    #   save_profile, build_context_document
├── enrich.py        # EnrichConfig, enrich_profile() — two-stage LLM pipeline (clean → enrich)
├── prompts.py       # System prompts for clean, enrich, and web search stages
├── llm/
│   ├── __init__.py  # LLMProvider protocol, parse_model_slug(), get_provider()
│   ├── openai.py    # OpenAIProvider (Chat Completions + Responses API for web search)
│   └── bedrock.py   # BedrockProvider (Converse API)
└── __main__.py      # CLI: build, enrich subcommands
```

**Data flow:** Data source (DB/JSON/dict) → `CompanyProfile` → optional `enrich_profile()` → `build_context_document()` → structured text

**LLM model slugs:** Provider-agnostic via `provider/model` format (e.g. `openai/gpt-5`, `bedrock/anthropic.claude-3-sonnet`). Parsed by `parse_model_slug()`, routed by `get_provider()`.

**Enrichment pipeline:** Two stages — clean & validate (fix names, normalize jurisdictions, dedup) then enrich (improve descriptions, add context). Optional web search stage between them (off by default, uses OpenAI Responses API).

## Key Dependencies

- Python >=3.13, managed with `uv`, built with `hatchling`
- `psycopg[binary]` (Postgres driver, v3 API with dict row factory)
- `pydantic` v2 (data model + config)
- `python-dotenv` (env config)
- Optional: `openai` (LLM enrichment), `boto3` (Bedrock enrichment)

## Environment Variables

- `CORPGRAPH_DB_URL` — Postgres connection string (required for DB mode)
- `CORPPROFILE_LLM_MODEL` — LLM slug like `openai/gpt-5` (overrides config.toml for enrichment)
