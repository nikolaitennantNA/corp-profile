# Data Independence & LLM Enrichment Design

## Overview

Two features for corp-profile:
1. **Data independence** — run without a live corp-graph DB (JSON files, dicts)
2. **LLM enrichment** — optional pipeline to clean, validate, and enrich profiles

## Architecture

Layered pipeline: `DataSource → CompanyProfile → LLMEnricher → enriched CompanyProfile`

Each layer is independent and optional.

## 1. Data Independence

`CompanyProfile` is already a Pydantic model, so it natively supports dict/JSON construction.

New functions in `profile.py`:
- `build_profile_from_dict(data: dict) -> CompanyProfile` — validates and constructs from raw data
- `build_profile_from_file(path: str) -> CompanyProfile` — reads JSON file, delegates to above
- `save_profile(profile: CompanyProfile, path: str)` — exports profile as JSON

File format is `CompanyProfile.model_dump()` JSON. No custom schema.

CLI additions:
- `python -m corp_profile build --from-file path.json`
- `python -m corp_profile save <ISIN> -o output.json` (build from DB and save)

## 2. LLM Enrichment

### Config

`EnrichConfig` Pydantic model with slug-based model selection:
- `model`: slug like `openai/gpt-5` or `bedrock/anthropic.claude-3-sonnet` — parsed on `/`, left side picks adapter, right side is model name
- `web_search`: `bool` (default `False`) — enables structured discovery (find missing subsidiaries, countries, etc.)
- `web_search_model`: optional override slug, defaults to main model. Useful since OpenAI has better web search.

Config reads from:
1. Keyword args (programmatic use)
2. `.env` fallback: `CORPPROFILE_LLM_MODEL`, `CORPPROFILE_WEB_SEARCH=false`, `CORPPROFILE_WEB_SEARCH_MODEL`

### Two-stage pipeline

1. **Clean & validate** — LLM fixes garbled names, normalizes jurisdictions, deduplicates subsidiaries, flags inconsistencies. Returns cleaned `CompanyProfile` + list of changes.
2. **Enrich** — LLM improves descriptions, summarizes segments, adds industry context. If `web_search=True`, also extracts structured facts from search results into profile fields.

LLM always returns JSON matching `CompanyProfile` schema, validated back into the model.

### Provider adapter

```python
class LLMProvider(Protocol):
    def complete(self, messages: list[dict], json_mode: bool = False) -> str: ...
```

Implementations for OpenAI and Bedrock. Provider selected by slug prefix.

### CLI

```bash
python -m corp_profile build <ISIN> --enrich
python -m corp_profile enrich profile.json
```

## File Structure

```
src/corp_profile/
├── __init__.py          # add new exports
├── __main__.py          # extended CLI
├── db.py                # unchanged
├── profile.py           # add from_dict, from_file, save_profile
├── enrich.py            # EnrichConfig, enrich_profile() orchestrator
├── llm/
│   ├── __init__.py      # LLMProvider protocol, get_provider()
│   ├── openai.py        # OpenAI adapter
│   └── bedrock.py       # Bedrock adapter
└── prompts.py           # system prompts for clean & enrich stages
```

## Dependencies

Optional extras in `pyproject.toml`:
```toml
[project.optional-dependencies]
openai = ["openai>=1.0"]
bedrock = ["boto3>=1.35"]
```

`uv sync` works without them. `uv sync --extra openai` to enable enrichment.

## Key Decisions

- Slug-based model selection (`provider/model`) instead of separate fields
- Web search with structured discovery included but off by default
- `.env` for config (no YAML/TOML — already established pattern)
- Pydantic JSON round-trip for file format (no custom schema)
- `CompanyProfile` is the local view, not the `company_profiles` DB table
