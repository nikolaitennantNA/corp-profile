# corp-profile

Build rich company context documents from a corp-graph Postgres database or JSON files. Output is structured markdown optimized for downstream LLM asset search pipelines.

## Setup

```bash
uv sync                        # Base dependencies
uv sync --extra llm            # Add LLM + search support (OpenAI, Bedrock, Exa)
cp .env.example .env           # Edit with your DB URL and API keys
```

App configuration (models, regions, search backends) lives in `config.toml`. Secrets (API keys) stay in `.env`.

## Usage

All commands save markdown to `outputs/`.

### Build a profile

```bash
# Build from corp-graph DB — accepts ISIN, LEI, issuer_id, or company name
python -m corp_profile build FR0000120271
python -m corp_profile build "TotalEnergies SE"

# Build from JSON file (no DB needed)
python -m corp_profile build --from-file examples/totalenergies.json

# Add LLM enrichment (cleans data, expands descriptions, fills gaps)
python -m corp_profile build --from-file examples/totalenergies.json --enrich

# Add web search (discovers subsidiaries, countries, verifies data)
python -m corp_profile build --from-file examples/totalenergies.json --enrich --web

# Also save the raw profile JSON
python -m corp_profile build FR0000120271 -o profile.json
```

`--web` implies `--enrich`. Both flags can be defaulted on in `config.toml` under `[pipeline]`.

### Research from scratch

Build a profile entirely via LLM + web search, no database needed:

```bash
python -m corp_profile research "TotalEnergies"
python -m corp_profile research FR0000120271 --name "TotalEnergies"   # With ISIN + name hint
python -m corp_profile research --seed examples/totalenergies.json    # Seed from partial JSON
python -m corp_profile research "TotalEnergies" -o out.json           # Also save profile JSON
```

## Output

The generated markdown is structured for LLM consumption with sections for:

- **Company Identity** — name, LEI, ISINs, aliases, description
- **Geographic & Operational Footprint** — countries (full names), business segments
- **Corporate Structure** — subsidiaries with ownership percentages
- **Asset Inventory** — type breakdown, sample of known assets (capped at 10), previously discovered assets
- **Discovery Gaps** — where known asset counts fall short of estimates
- **Search Guidance** — company/subsidiary names, countries, and asset types to search for

## Enrichment Pipeline

When `--enrich` is used, the profile goes through pipeline stages that depend on the flags:

- **`--enrich` only:** clean → enrich → refine
- **`--web`:** clean → web search → refine (enrich skipped — web provides real data)

Each stage returns the updated profile along with a list of changes made.

## Configuration

`config.toml` provides per-stage settings:

| Section      | Key fields              | Purpose                                  |
|-------------|------------------------|------------------------------------------|
| `[pipeline]` | `enrich`, `web`        | Default CLI flag values                  |
| `[research]` | `model`, `provider`    | LLM and search backend for research      |
| `[enrich]`   | `model`                | LLM for enrichment stage                 |
| `[web]`      | `model`, `provider`    | LLM and search backend for web stage     |
| `[aws]`      | `region`, `profile`    | Shared AWS config for Bedrock            |

**Config precedence:** environment variables → `config.toml` → defaults.

### Environment Variables

| Variable                    | Purpose                                              |
|----------------------------|------------------------------------------------------|
| `CORPGRAPH_DB_URL`         | Postgres connection string (required for DB mode)    |
| `CORPPROFILE_ENRICH_MODEL` | LLM slug, overrides `[enrich].model`                 |
| `CORPPROFILE_RESEARCH_MODEL`| LLM slug, overrides `[research].model`              |
| `CORPPROFILE_WEB_MODEL`    | LLM slug, overrides `[web].model`                    |
| `OPENAI_API_KEY`           | Required for OpenAI models                           |
| `EXA_API_KEY`              | Required when search provider is `"exa"`             |

LLM models use a provider-agnostic slug format: `provider/model` (e.g. `openai/gpt-5-mini`, `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`).

## Tests

```bash
uv run pytest tests/ -v
```
