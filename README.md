# corp-profile

Build rich company context documents from a corp-graph Postgres database or JSON files. Output is structured markdown optimized for downstream LLM asset search pipelines.

## Setup

```bash
uv sync                        # Base dependencies
uv sync --extra llm            # Add LLM support (OpenAI + Bedrock)
cp .env.example .env           # Edit with your DB URL and API keys
```

LLM configuration lives in `config.toml` (model, region, web search settings). Secrets stay in `.env`.

## Usage

```bash
# Build from corp-graph DB — accepts ISIN, LEI, issuer_id, or company name
python -m corp_profile build FR0000120271
python -m corp_profile build "TotalEnergies SE"

# Build from JSON file (no DB needed)
python -m corp_profile build --from-file examples/totalenergies.json

# Add LLM enrichment (cleans data, expands descriptions, fills gaps)
python -m corp_profile build --from-file examples/totalenergies.json --llm

# Add web search (discovers subsidiaries, countries, verifies data)
python -m corp_profile build --from-file examples/totalenergies.json --web

# Also save the raw profile JSON
python -m corp_profile build FR0000120271 -o profile.json
```

All commands save markdown to `outputs/`. The `--web` flag implies `--llm`.

## Output

The generated markdown is structured for LLM consumption with sections for:

- **Company Identity** — name, LEI, ISINs, aliases, description
- **Geographic & Operational Footprint** — countries (full names), business segments
- **Corporate Structure** — subsidiaries with ownership percentages
- **Asset Inventory** — type breakdown, sample of known assets (capped at 10), previously discovered assets
- **Discovery Gaps** — where known asset counts fall short of estimates
- **Search Guidance** — company/subsidiary names, countries, and asset types to search for

## Enrichment Pipeline

When `--llm` is used, the profile goes through up to three stages:

1. **Clean** — fix garbled names, normalize jurisdictions, dedup subsidiaries
2. **Web search** (optional, `--web`) — discover missing subsidiaries and countries via OpenAI Responses API
3. **Enrich** — improve descriptions, expand operating countries and business segments

The main LLM provider is configured in `config.toml` (default: Bedrock). Web search always uses OpenAI.

## Tests

```bash
uv run pytest tests/ -v
```
