# Research Command, Config Redesign & company_universe Improvements

**Date:** 2026-03-13
**Status:** Approved
**Scope:** corp-profile (primary), corp-graph (materialization), asset-search-v2 (config mapping)

## Problem

1. **No DB-free profile building.** corp-profile requires either a full JSON file or a live corp-graph Postgres connection. There's no way to build a profile from just a company name/identifier using web search.

2. **Web search is OpenAI-only.** The enrichment pipeline's web search stage uses OpenAI's `web_search_preview` exclusively. Bedrock users must configure a separate OpenAI model just for web search, and there's no provider-agnostic search abstraction.

3. **Config is muddled.** The `[llm]` section mixes LLM model config with web search config (`web_search_model`, `search_backend`). The `[profile]` section name is confusing given `CompanyProfile` is the main data model.

4. **`material_assets_types` missing from `company_universe`.** corp-profile must query `asset_estimates` separately to get this data, adding an unnecessary DB round-trip.

## Solution Overview

- New `research` subcommand: build a CompanyProfile from scratch using LLM + web search
- Decoupled search backends: Exa (provider-agnostic) and OpenAI built-in, configurable per stage
- Tool-use loop in both providers: LLM calls search tools, we execute, return results
- Config restructured into pipeline-stage sections: `[research]`, `[enrich]`, `[web]`
- `material_assets_types` added to `company_universe` materialization

---

## 1. Config Redesign

### corp-profile `config.toml`

```toml
[pipeline]
enrich = false                # default for --enrich CLI flag (was --llm)
web = false                   # default for --web CLI flag

[research]
model = "openai/gpt-5-mini"  # LLM for research command
provider = "auto"             # search backend: "exa", "openai", or "auto"

[enrich]
model = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"  # LLM for clean/enrich/refine

[web]
model = "openai/gpt-5-mini"  # LLM for web enrichment stage
provider = "auto"             # search backend: "exa", "openai", or "auto"

[aws]
region = "us-east-2"
```

**Three pipeline stages, three config sections:**

| Section | Purpose | Has web search? |
|---------|---------|-----------------|
| `[research]` | Build profile from scratch via web | Yes — `model` + `provider` |
| `[enrich]` | Clean, enrich, refine existing profile | No — LLM only |
| `[web]` | Web-enhanced enrichment on `build --web` | Yes — `model` + `provider` |

**`provider` values:**
- `"auto"` (default) — if `model` is an OpenAI slug, use OpenAI's built-in `web_search_preview`. Otherwise use Exa.
- `"exa"` — always use Exa as a tool (works with any LLM).
- `"openai"` — always use OpenAI's built-in `web_search_preview` (only works with OpenAI models).

**`[pipeline]` toggles:**
- `enrich` — enables clean/enrich/refine stages on `build` (was `llm`).
- `web` — enables web search enrichment on `build` (implies `enrich`).
- `research` command doesn't need a toggle — running it is opting in.

**CLI flag rename:** `--llm` becomes `--enrich`. `--llm` kept as a deprecated alias for backward compatibility. `--web` stays as-is.

**API keys stay in `.env`:** `EXA_API_KEY`, `OPENAI_API_KEY`. AWS credentials are managed externally (e.g. `aws-mfa`), not in `.env`.

**Env var overrides per section:**
- `CORPPROFILE_ENRICH_MODEL` → `[enrich].model`
- `CORPPROFILE_RESEARCH_MODEL` → `[research].model`
- `CORPPROFILE_WEB_MODEL` → `[web].model`
- `CORPPROFILE_LLM_MODEL` → deprecated alias for `CORPPROFILE_ENRICH_MODEL`

### Config classes in corp-profile

```python
# src/corp_profile/config.py (new file, replaces EnrichConfig in enrich.py)

class PipelineConfig(BaseModel):
    enrich: bool = False
    web: bool = False

class ResearchConfig(BaseModel):
    model: str
    provider: str = "auto"       # "exa", "openai", "auto"
    aws_region: str | None = None
    aws_profile: str | None = None

class EnrichConfig(BaseModel):
    model: str
    aws_region: str | None = None
    aws_profile: str | None = None

class WebConfig(BaseModel):
    model: str
    provider: str = "auto"
    aws_region: str | None = None
    aws_profile: str | None = None
```

Each has a `load()` classmethod that reads from its corresponding config.toml section with env var overrides (same resolution pattern as today).

**Import path migration:** The old `from corp_profile.enrich import EnrichConfig` import path will be maintained via the existing lazy-load mechanism in `__init__.py`. It will re-export `EnrichConfig` from `config.py` so downstream consumers (asset-search-v2) don't break during migration. A deprecation warning will be emitted.

### asset-search-v2 integration

asset-search-v2's master config mirrors corp-profile's structure under `[profile]`:

```toml
# asset-search-v2 config.toml

[models]
discover = "bedrock/us.anthropic.claude-opus-4-6-v1"   # powerful, for discovery
# ...

# ── corp-profile (mirrors corp-profile's own config) ──
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

# ── Discovery (completely separate from corp-profile) ──
[search]
provider = "exa"
```

Master `Config` class gets builder methods:

```python
def profile_pipeline_config(self):
    from corp_profile.config import PipelineConfig
    return PipelineConfig(enrich=..., web=...)

def profile_enrich_config(self):
    from corp_profile.config import EnrichConfig
    return EnrichConfig(model=..., aws_region=..., aws_profile=...)

def profile_web_config(self):
    from corp_profile.config import WebConfig
    return WebConfig(model=..., provider=..., aws_region=...)

def profile_research_config(self):
    from corp_profile.config import ResearchConfig
    return ResearchConfig(model=..., provider=..., aws_region=...)
```

The `[search]` section in asset-search-v2 remains entirely separate — it configures the discovery stage's search provider, not corp-profile's.

---

## 2. Search Backend Abstraction

### New module: `src/corp_profile/search.py`

```python
class SearchResult(BaseModel):
    title: str
    url: str
    content: str

class SearchBackend(Protocol):
    def search(self, query: str, num_results: int = 5) -> list[SearchResult]: ...

class ExaSearch(SearchBackend):
    """Wraps exa-py SDK."""
    def __init__(self, api_key: str): ...
    def search(self, query: str, num_results: int = 5) -> list[SearchResult]: ...

def get_search_backend(provider: str, model: str) -> SearchBackend | None:
    """Resolve provider string to a SearchBackend instance, or None for OpenAI built-in.

    When provider is "auto", uses the model slug to decide:
    OpenAI models → None (use built-in web_search_preview),
    everything else → ExaSearch.
    """
    # "exa" → ExaSearch (validates EXA_API_KEY, raises if missing)
    # "openai" → None (signal to use OpenAI built-in)
    # "auto" → parse model slug prefix; "openai/" → None, else → ExaSearch
```

**Tool schema** exposed by the search backend for LLM tool-use:

```json
{
  "name": "web_search",
  "description": "Search the web for current information about a company, its operations, subsidiaries, or assets.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search query"},
      "num_results": {"type": "integer", "default": 5, "description": "Number of results to return"}
    },
    "required": ["query"]
  }
}
```

### New dependency

- `exa-py` added as an optional dependency in the `[llm]` extra group.

---

## 3. Provider Tool-Use Loop

Both providers get a new `complete_with_tools()` method that handles the search tool-use loop. This method is **not** added to the `LLMProvider` Protocol — it's a concrete method on each provider class. The Protocol stays minimal (`complete()` only). Call sites that need tool-use import the concrete provider or use duck typing.

### BedrockProvider

```python
def complete_with_tools(
    self,
    messages: list[dict],
    search_backend: SearchBackend,
    json_mode: bool = False,
    max_iterations: int = 10,
) -> str:
    """Run a tool-use loop: LLM calls search → we execute → return results → repeat."""
    # 1. Build tool config with web_search tool schema
    # 2. If json_mode, also add the enrichment_response tool for structured output
    # 3. Loop:
    #    a. Call Bedrock Converse API
    #    b. If response has toolUse block for "web_search":
    #       - Execute search_backend.search(query, num_results)
    #       - Append tool result to messages
    #       - Continue loop
    #    c. If response has toolUse block for "enrichment_response" (structured output):
    #       - Return the JSON
    #    d. If response has no tool calls:
    #       - Return the text
    # 4. Safety cap at max_iterations
```

The existing `complete(web_search=True)` is refactored: instead of raising `NotImplementedError`, it resolves the search backend and delegates to `complete_with_tools()`.

### OpenAIProvider

```python
def complete_with_tools(
    self,
    messages: list[dict],
    search_backend: SearchBackend,
    json_mode: bool = False,
    max_iterations: int = 10,
) -> str:
    """Run a tool-use loop using OpenAI function calling."""
    # Same pattern: loop on tool calls, execute search, return results
```

When `web_search=True` and provider is `"openai"`: keep current `web_search_preview` behavior (no loop, OpenAI handles internally).
When `web_search=True` and provider is `"exa"`: use `complete_with_tools` with Exa backend.

### How search_backend reaches the providers

The search backend is resolved at the call site (in `enrich.py` or `research.py`), not baked into the provider. The provider is LLM-only; search is injected per-call:

```python
# In enrich.py or research.py:
backend = get_search_backend(config.provider, model=config.model)
provider = get_provider(config.model, ...)

if backend:
    result = provider.complete_with_tools(messages, search_backend=backend, json_mode=True)
else:
    # backend is None → OpenAI built-in web_search_preview
    result = provider.complete(messages, json_mode=True, web_search=True)
```

---

## 4. `research` Command

### CLI

```bash
python -m corp_profile research FR0000120271                          # identifier required
python -m corp_profile research FR0000120271 --name "TotalEnergies"   # name helps search
python -m corp_profile research --seed examples/partial.json          # JSON must contain identifier
python -m corp_profile research FR0000120271 -o profile.json          # save JSON output
```

- `identifier` (positional, required unless `--seed`): ISIN, LEI, issuer_id, or company name.
- `--name`: optional display name to help the LLM search effectively.
- `--seed`: path to a partial JSON file. Must contain at least `issuer_id` or one identifier in `isin_list`. The LLM fills gaps.
- `-o / --output`: also save the CompanyProfile as JSON.
- Output: markdown saved to `outputs/` (same as `build`).

### New module: `src/corp_profile/research.py`

```python
def research_profile(
    identifier: str,
    name: str | None = None,
    seed: CompanyProfile | None = None,
    config: ResearchConfig,
) -> tuple[CompanyProfile, list[str]]:
    """Build a CompanyProfile from scratch via LLM + web search.

    Pipeline:
      1. Research: LLM + web search tool-use loop → initial CompanyProfile
      2. Clean: fix names, normalize jurisdictions, dedup (existing stage)

    Returns (profile, list_of_changes).
    """
```

### Research prompt

New prompt in `prompts.py`: `RESEARCH_SYSTEM_PROMPT`

Gives the LLM:
- The identifier and optional name
- The full `CompanyProfile` JSON schema
- The `web_search` tool
- Instructions to search for:
  - Company overview, description, headquarters, industry classification
  - Corporate structure: key subsidiaries with jurisdictions and ownership
  - Operating countries and business segments
  - Types of physical assets the company operates (for `material_asset_types`)
  - Estimated asset counts where possible
  - Aliases and alternative names
  - All ISINs, LEIs associated with the entity
- If `--seed` provided: the partial profile as starting context, with instructions to fill gaps

The LLM drives the search — it decides what queries to run, follows leads, and assembles the profile. The tool-use loop caps at ~10 iterations as a safety net.

**Structured output:** The research stage uses the same `EnrichmentResponse` schema (profile + changes) as other stages. The LLM returns a full `CompanyProfile` plus a changes list documenting what it found (e.g. "Found 12 subsidiaries via TotalEnergies annual report", "Identified 5 operating countries from corporate website").

**`--seed` validation:** If `--seed` is provided, `research_profile()` validates that the JSON contains at least one of: `issuer_id`, a non-empty `isin_list`, or `lei`. Raises `ValueError` if none are present.

### Pipeline flow

```
research <identifier>:
    │
    ├─ 1. Research stage (new)
    │     LLM + web search tool → builds initial CompanyProfile
    │
    └─ 2. Clean stage (existing)
          Fix garbled names, normalize jurisdictions, dedup subsidiaries
```

No enrich or refine — research already populated the profile with web-sourced data including asset type estimates.

---

## 5. Pipeline Ordering

Three distinct pipelines depending on the command and flags:

```
build <identifier> --enrich (no web):
  1. Clean         ← LLM fixes garbled names, normalizes jurisdictions, dedups
  2. Enrich        ← LLM fills gaps using training knowledge (best effort, may hallucinate)
  3. Refine        ← LLM adjusts guestimator asset counts (if estimates exist)

build <identifier> --web (implies --enrich):
  1. Clean         ← LLM fixes data quality
  2. Web search    ← LLM + web finds real data with citations (replaces enrich)
  3. Refine        ← LLM adjusts guestimator asset counts (if estimates exist)
  (enrich stage SKIPPED — web search already covers everything enrich does, with real data)

research <identifier>:
  1. Research       ← LLM + web builds CompanyProfile from scratch
  2. Clean          ← LLM fixes any issues in research output
  (no enrich — research used web search)
  (no refine — research estimated asset types directly)
```

**Key design decision:** When `--web` is enabled, the LLM-only enrich stage is skipped. The web search prompt already discovers subsidiaries, finds operating countries, identifies business segments, and improves descriptions — the same things enrich does — but with real, cited sources. Running enrich after web search risks overwriting verified facts with hallucinated data.

Clean always runs first because web search needs a clean company name to search effectively.

---

## 6. `company_universe` Materialization Change

**Repo:** corp-graph (`/Users/nikolai.tennant/Documents/GitHub/corp-graph`)
**File:** `src/materialize.py`

### Change

In Step 7 (the big LEFT JOIN that builds `company_universe`), add a LEFT JOIN to `asset_estimates`:

```sql
LEFT JOIN asset_estimates ae ON ae.issuer_id = eff.issuer_id
```

Pull in:
- `ae.material_assets_types` (jsonb) → new column `material_assets_types`

The `estimated_assets_count` and `estimated_material_assets_count` columns are already on `company_universe` from a previous materialization step.

### corp-profile change

In `build_profile()` (`profile.py`):
- Remove the separate `asset_estimates` query (current step 6).
- Read `material_assets_types` from the `company_universe` row directly (already returned by `SELECT *`).
- The parsing logic (handling both `{"type": "...", "count": ...}` and `{"Name": count}` formats) is already fixed (earlier in this session).

**Net result:** One fewer DB round-trip per profile build. `build_profile()` goes from 6 queries to 5.

---

## 7. New Dependencies

| Package | Extra group | Purpose |
|---------|-------------|---------|
| `exa-py` | `[llm]` | Exa search API client |

---

## 8. Files Changed

### corp-profile (primary)

| File | Change |
|------|--------|
| `src/corp_profile/config.py` | **New.** `PipelineConfig`, `ResearchConfig`, `EnrichConfig`, `WebConfig` classes. |
| `src/corp_profile/search.py` | **New.** `SearchBackend` protocol, `ExaSearch`, `get_search_backend()`. |
| `src/corp_profile/research.py` | **New.** `research_profile()` function. |
| `src/corp_profile/prompts.py` | Add `RESEARCH_SYSTEM_PROMPT`. |
| `src/corp_profile/__main__.py` | Add `research` subcommand. Rename `--llm` to `--enrich`. |
| `src/corp_profile/enrich.py` | Refactor to use new config classes. Use `complete_with_tools()` for web search stage. Skip enrich stage when web search is enabled. Remove old `EnrichConfig`. Remove dead `_refine_estimates_stage` async function. |
| `src/corp_profile/llm/__init__.py` | Update provider protocol with `complete_with_tools()`. |
| `src/corp_profile/llm/bedrock.py` | Add `complete_with_tools()`. Remove `NotImplementedError` on `web_search`. |
| `src/corp_profile/llm/openai.py` | Add `complete_with_tools()`. Keep `web_search_preview` as fallback. |
| `src/corp_profile/profile.py` | Remove `asset_estimates` query. Read `material_assets_types` from `company_universe` row. |
| `config.toml` | Restructure: `[pipeline]`, `[research]`, `[enrich]`, `[web]`, `[aws]`. |
| `pyproject.toml` | Add `exa-py` to `[llm]` extra. |

### corp-graph

| File | Change |
|------|--------|
| `src/materialize.py` | Add LEFT JOIN to `asset_estimates` in Step 7. Add `material_assets_types` column. |

### asset-search-v2

| File | Change |
|------|--------|
| `config.toml` | Restructure `[profile]` section to match corp-profile's new config. |
| `src/asset_search/config.py` | Update `Config` class: new builder methods for `ResearchConfig`, `EnrichConfig`, `WebConfig`. Rename `profile_llm` → `profile_enrich`, `profile_web_search` → `profile_web`. |

---

## 9. Testing

| Test | What it covers |
|------|----------------|
| `test_config.py` | New config classes load from toml, env overrides work, defaults are correct. |
| `test_search.py` | ExaSearch returns SearchResults. Mock Exa API. `get_search_backend()` routing. |
| `test_research.py` | `research_profile()` with mocked search backend + mocked LLM. Verifies tool-use loop executes, profile is populated. |
| `test_providers.py` | `complete_with_tools()` for both Bedrock and OpenAI with mocked search. Tool-use loop terminates correctly. |
| `test_enrich.py` | Existing tests updated for new config classes. Web search stage works with Exa backend. |
| `test_demo.py` | Existing demo test still passes with new config structure. |
