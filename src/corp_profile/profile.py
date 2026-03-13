"""Build rich company context documents from corp-graph or JSON files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from .db import get_connection


class _NullSafeModel(BaseModel):
    """Base model that coerces None to empty string for str fields from DB NULLs."""

    @model_validator(mode="before")
    @classmethod
    def _coerce_nulls(cls, data):
        if not isinstance(data, dict):
            return data
        for field_name, field_info in cls.model_fields.items():
            if field_name in data and data[field_name] is None:
                if field_info.annotation is str or field_info.annotation == str:
                    data[field_name] = ""
        return data


class Subsidiary(_NullSafeModel):
    """A subsidiary or child entity in the corporate structure."""

    issuer_id: str | None = None
    legal_name: str = ""
    jurisdiction: str = ""
    lei: str | None = None
    ownership_percentage: float | None = None
    rel_type: str = ""


class Asset(_NullSafeModel):
    """A known physical asset from the ALD database."""

    asset_name: str = ""
    address: str = ""
    latitude: float | None = None
    longitude: float | None = None
    naturesense_asset_type: str = Field(
        default="",
        description="Standardized asset type, e.g. Petroleum Refinery, LNG Terminal, "
        "Solar Farm, Wind Farm, Oil Production Platform, Coal Mine, Power Plant, "
        "Chemical Plant, Steel Mill, Cement Plant, Gas Processing Plant",
    )
    capacity: float | None = None
    capacity_units: str = ""
    status: str = Field(
        default="",
        description="Short operational status: Operating, Under Construction, "
        "Planned, Decommissioned, or Idle",
    )


class DiscoveredAsset(_NullSafeModel):
    """An asset found during a previous search/discovery run."""

    asset_name: str = ""
    asset_type_raw: str = ""
    address: str = ""
    latitude: float | None = None
    longitude: float | None = None


class MaterialAssetType(_NullSafeModel):
    """An expected asset type with optional estimated count."""

    type: str = Field(
        default="",
        description="Short standardized asset type name (2-3 words max), e.g. "
        "Petroleum Refinery, LNG Terminal, Solar Farm, Wind Farm, "
        "Oil Production Platform, Coal Mine, Power Plant, Service Station. "
        "Do NOT use long compound names or slashes.",
    )
    count: int | None = None


class CompanyProfile(BaseModel):
    """Structured company profile assembled from corp-graph data."""

    issuer_id: str
    legal_name: str
    lei: str | None = None
    jurisdiction: str | None = None
    isin_list: list[str] = []          # parent entity ISINs
    all_isins: list[str] = []          # parent + all subsidiary ISINs
    aliases: list[str] = []
    description: str = ""
    primary_industry: str = Field(
        default="",
        description="Short industry classification, e.g. Oil, Gas & Consumable Fuels; "
        "Electric Utilities; Metals & Mining; Chemicals",
    )
    operating_countries: list[str] = []
    business_segments: list[str] = []
    subsidiaries: list[Subsidiary] = []
    existing_assets: list[Asset] = []
    discovered_assets: list[DiscoveredAsset] = []
    estimated_asset_count: int | None = None
    material_asset_types: list[MaterialAssetType] = []


def _resolve_entity(conn, identifier: str) -> dict:
    """Try to find an entity in company_universe by any identifier.

    Attempts in order: ISIN, LEI, issuer_id, then name (case-insensitive).
    """
    # 1a. ISIN — match against parent entity's isin_list
    row = conn.execute(
        "SELECT * FROM company_universe WHERE isin_list @> ARRAY[%s]::varchar[]",
        [identifier],
    ).fetchone()
    if row:
        return dict(row)

    # 1b. ISIN — match against securities table (covers any ISIN in the system)
    sec = conn.execute(
        "SELECT issuer_id FROM securities WHERE isin = %s LIMIT 1",
        [identifier],
    ).fetchone()
    if sec:
        row = conn.execute(
            "SELECT * FROM company_universe WHERE issuer_id = %s",
            [sec["issuer_id"]],
        ).fetchone()
        if row:
            return dict(row)

    # 2. LEI (exact match)
    row = conn.execute(
        "SELECT * FROM company_universe WHERE lei = %s",
        [identifier],
    ).fetchone()
    if row:
        return dict(row)

    # 3. issuer_id (exact match)
    row = conn.execute(
        "SELECT * FROM company_universe WHERE issuer_id = %s",
        [identifier],
    ).fetchone()
    if row:
        return dict(row)

    # 4. Name — exact match on legal_name or any alias (case-insensitive)
    row = conn.execute(
        "SELECT * FROM company_universe WHERE UPPER(legal_name) = UPPER(%s)",
        [identifier],
    ).fetchone()
    if row:
        return dict(row)

    row = conn.execute(
        "SELECT * FROM company_universe WHERE alias_list @> ARRAY[%s]::varchar[]",
        [identifier],
    ).fetchone()
    if row:
        return dict(row)

    # 5. Name — prefix match as last resort
    row = conn.execute(
        "SELECT * FROM company_universe WHERE legal_name ILIKE %s LIMIT 1",
        [f"{identifier}%"],
    ).fetchone()
    if row:
        return dict(row)

    raise LookupError(
        f"No entity found for '{identifier}'. "
        "Tried matching as ISIN, LEI, issuer_id, and company name/alias."
    )


def build_profile(identifier: str) -> CompanyProfile:
    """Query corp-graph Postgres and build a CompanyProfile.

    Accepts any identifier: ISIN, LEI, issuer_id, or company name.
    """
    with get_connection() as conn:
        # 1. Resolve entity from company_universe
        row = _resolve_entity(conn, identifier)

        issuer_id = row["issuer_id"]

        profile = CompanyProfile(
            issuer_id=issuer_id,
            legal_name=row.get("legal_name", ""),
            lei=row.get("lei"),
            jurisdiction=row.get("jurisdiction"),
            isin_list=row.get("isin_list") or [],
            aliases=row.get("alias_list") or [],
            description=row.get("profile_description") or "",
            primary_industry=row.get("primary_industry") or "",
            operating_countries=row.get("operating_countries") or [],
            business_segments=row.get("business_segments") or [],
        )

        # Prefer company_profiles name over issuers.legal_name (which may be
        # corrupted by GLEIF alias-match name adoption)
        cp_row = conn.execute(
            "SELECT company_name, official_name FROM company_profiles WHERE issuer_id = %s",
            [issuer_id],
        ).fetchone()
        if cp_row:
            better_name = cp_row.get("official_name") or cp_row.get("company_name")
            if better_name:
                profile.legal_name = better_name

        # 2. Subsidiary tree
        subs = conn.execute(
            """
            SELECT child.issuer_id, child.legal_name, child.jurisdiction, child.lei,
                   r.ownership_percentage, r.rel_type
            FROM relationships r
            JOIN issuers child ON child.issuer_id = r.child_issuer_id
            WHERE r.parent_issuer_id = %s AND r.rel_status = 'ACTIVE'
            """,
            [issuer_id],
        ).fetchall()
        # Dedup subsidiaries by name (case-insensitive), prefer entry with LEI
        seen: dict[str, dict] = {}
        for s in subs:
            d = dict(s)
            key = (d.get("legal_name") or "").strip().upper()
            if not key:
                continue
            existing = seen.get(key)
            if existing is None:
                seen[key] = d
            elif d.get("lei") and not existing.get("lei"):
                seen[key] = d
        profile.subsidiaries = [Subsidiary.model_validate(v) for v in seen.values()]

        # 2b. Aggregate ISINs from parent + all subsidiaries
        sub_ids = [s.issuer_id for s in profile.subsidiaries if s.issuer_id]
        all_ids = [issuer_id] + sub_ids
        if all_ids:
            placeholders = ",".join(["%s"] * len(all_ids))
            sub_isins = conn.execute(
                f"""
                SELECT DISTINCT isin FROM securities
                WHERE issuer_id IN ({placeholders}) AND isin IS NOT NULL
                ORDER BY isin
                """,
                all_ids,
            ).fetchall()
            profile.all_isins = sorted(
                set(profile.isin_list) | {r["isin"] for r in sub_isins}
            )

        # 3. Existing assets from ALD
        assets = conn.execute(
            """
            SELECT asset_name, address, latitude, longitude, naturesense_asset_type,
                   capacity, capacity_units, status
            FROM assets WHERE issuer_id = %s
            """,
            [issuer_id],
        ).fetchall()
        # Filter out unnamed assets that duplicate a named one at the same address
        all_assets = [dict(a) for a in assets]
        named = {
            a["address"][:40]
            for a in all_assets
            if a.get("asset_name") and a.get("address")
        }
        profile.existing_assets = [
            Asset.model_validate(a) for a in all_assets
            if a.get("asset_name") or (a.get("address") or "")[:40] not in named
        ]

        # 4. Company profile (enriched)
        cp = conn.execute(
            "SELECT * FROM company_profiles WHERE issuer_id = %s",
            [issuer_id],
        ).fetchone()
        if cp:
            profile.description = cp.get("description") or profile.description
            profile.primary_industry = (
                cp.get("primary_industry") or profile.primary_industry
            )
            profile.operating_countries = (
                cp.get("operating_countries") or profile.operating_countries
            )
            profile.business_segments = (
                cp.get("business_segments") or profile.business_segments
            )

        # 5. Previously discovered assets
        discovered = conn.execute(
            """
            SELECT asset_name, asset_type_raw, address, latitude, longitude
            FROM discovered_assets WHERE issuer_id = %s
            """,
            [issuer_id],
        ).fetchall()
        profile.discovered_assets = [DiscoveredAsset.model_validate(dict(d)) for d in discovered]

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

    return profile


def build_profile_from_dict(data: dict) -> CompanyProfile:
    """Build a CompanyProfile from a plain dict (no DB required)."""
    return CompanyProfile.model_validate(data)


def profile_filename(profile: CompanyProfile) -> str:
    """Generate a filename from the profile: 'Company_Name_(IDENTIFIER)'."""
    import re
    name = profile.legal_name
    # Pick best identifier: first ISIN, then LEI, then issuer_id
    identifier = (
        (profile.isin_list[0] if profile.isin_list else None)
        or profile.lei
        or profile.issuer_id
    )
    base = f"{name}_({identifier})" if identifier else name
    # Sanitize for filesystem: replace spaces and illegal chars with underscores
    return re.sub(r'[<>:"/\\|?*\s]+', '_', base).strip("_")


def save_profile(profile: CompanyProfile, path: str) -> None:
    """Save a CompanyProfile to a JSON file."""
    Path(path).write_text(
        json.dumps(profile.model_dump(), indent=2, default=str)
    )


def save_profile_markdown(profile: CompanyProfile, path: str | None = None) -> str:
    """Save the rendered context document as markdown to outputs/. Returns the path used."""
    if path is None:
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        path = str(out_dir / f"{profile_filename(profile)}.md")
    Path(path).write_text(build_context_document(profile))
    return path


def build_profile_from_file(path: str) -> CompanyProfile:
    """Load a CompanyProfile from a JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")
    data = json.loads(p.read_text())
    return build_profile_from_dict(data)


_ASSET_SAMPLE_SIZE = 10
_SUB_SAMPLE_SIZE = 20

def _country_name(code: str) -> str:
    """Resolve an ISO country code to its common name via pycountry, falling back to the code."""
    import pycountry

    country = pycountry.countries.get(alpha_2=code.upper())
    if country is None:
        return code
    return getattr(country, "common_name", None) or country.name


def refine_estimates(
    profile: CompanyProfile,
    llm_response: dict,
) -> CompanyProfile:
    """Apply LLM-refined estimates to a profile."""
    if "material_asset_types" in llm_response:
        profile.material_asset_types = [
            MaterialAssetType(**t) for t in llm_response["material_asset_types"]
        ]
    # Always recompute total from per-type counts for consistency
    type_sum = sum(t.count for t in profile.material_asset_types if t.count is not None)
    if type_sum > 0:
        profile.estimated_asset_count = type_sum
    elif "estimated_asset_count" in llm_response:
        profile.estimated_asset_count = llm_response["estimated_asset_count"]
    return profile


def build_context_document(profile: CompanyProfile) -> str:
    """Render a CompanyProfile as a structured markdown document for LLM consumption.

    Optimized for asset search pipelines: emphasizes company identity, geographic
    footprint, asset type patterns, and naming conventions to guide discovery.
    """
    sections: list[str] = []

    # --- Company Identity ---
    id_lines = [f"# {profile.legal_name}"]
    id_lines.append(f"*Profile generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}*")
    meta: list[str] = []
    if profile.lei:
        meta.append(f"**LEI:** {profile.lei}")
    if profile.jurisdiction:
        meta.append(f"**Jurisdiction:** {profile.jurisdiction}")
    if profile.primary_industry:
        meta.append(f"**Industry:** {profile.primary_industry}")
    if meta:
        id_lines.append(" | ".join(meta))
    if profile.all_isins:
        id_lines.append(f"**ISINs:** {', '.join(profile.all_isins)}")
    elif profile.isin_list:
        id_lines.append(f"**ISINs:** {', '.join(profile.isin_list)}")
    if profile.aliases:
        id_lines.append(f"**Also known as:** {', '.join(profile.aliases)}")
    if profile.description:
        id_lines.append(f"\n{profile.description}")
    sections.append("\n".join(id_lines))

    # --- Geographic Footprint ---
    if profile.operating_countries or profile.business_segments:
        geo_lines = ["\n## Geographic & Operational Footprint"]
        if profile.operating_countries:
            country_names = [_country_name(c) for c in profile.operating_countries]
            geo_lines.append(
                f"**Operating countries ({len(profile.operating_countries)}):** "
                + ", ".join(country_names)
            )
        if profile.business_segments:
            geo_lines.append(
                f"**Business segments:** {', '.join(profile.business_segments)}"
            )
        sections.append("\n".join(geo_lines))

    # --- Corporate Structure ---
    if profile.subsidiaries:
        total_subs = len(profile.subsidiaries)
        parent_jur = (profile.jurisdiction or "").upper()

        # Rank subsidiaries by data completeness and search relevance
        def _sub_sort_key(s: Subsidiary) -> tuple:
            has_ownership = s.ownership_percentage is not None
            has_lei = bool(s.lei)
            has_real_name = bool(s.legal_name) and not s.legal_name.startswith("UK Company ")
            diff_jurisdiction = bool(s.jurisdiction) and s.jurisdiction.upper() != parent_jur
            return (has_ownership, has_real_name, diff_jurisdiction, has_lei)

        ranked = sorted(profile.subsidiaries, key=_sub_sort_key, reverse=True)
        show_subs = ranked[:_SUB_SAMPLE_SIZE] if total_subs > _SUB_SAMPLE_SIZE else ranked

        # Jurisdiction breakdown for large subsidiary lists
        jur_counts: dict[str, int] = {}
        for s in profile.subsidiaries:
            jur = _country_name(s.jurisdiction) if s.jurisdiction else "Unknown"
            jur_counts[jur] = jur_counts.get(jur, 0) + 1

        if total_subs > _SUB_SAMPLE_SIZE:
            sub_lines = [
                f"\n## Corporate Structure ({total_subs} subsidiaries "
                f"across {len(jur_counts)} jurisdictions)"
            ]
            sub_lines.append(f"\n### Key Subsidiaries ({len(show_subs)} of {total_subs})")
        else:
            sub_lines = [f"\n## Corporate Structure ({total_subs} subsidiaries)"]

        for s in show_subs:
            parts = [f"- **{s.legal_name or 'Unknown'}**"]
            if s.jurisdiction:
                parts.append(f"({_country_name(s.jurisdiction)})")
            if s.ownership_percentage is not None:
                parts.append(f"— {s.ownership_percentage}% owned")
            if s.lei:
                parts.append(f"[LEI: {s.lei}]")
            sub_lines.append(" ".join(parts))

        # Show jurisdiction breakdown when capped
        if total_subs > _SUB_SAMPLE_SIZE:
            sub_lines.append("\n### Subsidiary Jurisdictions")
            for jur, count in sorted(jur_counts.items(), key=lambda x: -x[1]):
                sub_lines.append(f"- {jur}: {count}")

        sections.append("\n".join(sub_lines))

    # --- Asset Inventory ---
    # Combines known assets, discovered assets, and estimates into one section
    # focused on what an asset search pipeline needs to know.
    has_assets = (
        profile.existing_assets
        or profile.discovered_assets
        or profile.estimated_asset_count is not None
        or profile.material_asset_types
    )
    if has_assets:
        asset_lines = ["\n## Asset Inventory"]

        # Summary stats
        known_count = len(profile.existing_assets)
        discovered_count = len(profile.discovered_assets)
        if profile.estimated_asset_count is not None:
            asset_lines.append(
                f"**Estimated total:** {profile.estimated_asset_count} assets"
            )
        if known_count:
            asset_lines.append(f"**Known (verified):** {known_count}")
        if discovered_count:
            asset_lines.append(f"**Previously discovered:** {discovered_count}")

        # Material asset types (what to search for)
        if profile.material_asset_types:
            asset_lines.append("\n### Expected Asset Types")
            for t in profile.material_asset_types:
                if t.count is not None:
                    asset_lines.append(f"- {t.type or 'Unknown'}: ~{t.count}")
                else:
                    asset_lines.append(f"- {t.type or 'Unknown'}")

        # Known asset type breakdown
        if profile.existing_assets:
            by_type: dict[str, list[Asset]] = {}
            for a in profile.existing_assets:
                atype = a.naturesense_asset_type or "Unknown"
                by_type.setdefault(atype, []).append(a)

            asset_lines.append("\n### Known Asset Type Breakdown")
            for atype, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
                asset_lines.append(f"- {atype}: {len(items)}")

            # Sample of named assets to show naming/location patterns
            named = [a for a in profile.existing_assets if a.asset_name]
            if named:
                sample = named[:_ASSET_SAMPLE_SIZE]
                asset_lines.append(
                    f"\n### Sample Known Assets ({len(sample)} of {len(named)} named)"
                )
                for a in sample:
                    parts = [f"- **{a.asset_name}**"]
                    if a.naturesense_asset_type:
                        parts.append(f"[{a.naturesense_asset_type}]")
                    if a.address:
                        parts.append(f"— {a.address}")
                    if a.status:
                        parts.append(f"({a.status})")
                    asset_lines.append(" ".join(parts))

        # Previously discovered assets (so the pipeline doesn't re-discover them)
        if profile.discovered_assets:
            sample_disc = profile.discovered_assets[:_ASSET_SAMPLE_SIZE]
            asset_lines.append(
                f"\n### Previously Discovered ({discovered_count} total, "
                f"showing {len(sample_disc)})"
            )
            asset_lines.append(
                "These assets were found in prior searches — skip during new discovery."
            )
            for d in sample_disc:
                parts = [f"- **{d.asset_name or 'Unknown'}**"]
                if d.asset_type_raw:
                    parts.append(f"[{d.asset_type_raw}]")
                if d.address:
                    parts.append(f"— {d.address}")
                asset_lines.append(" ".join(parts))

        sections.append("\n".join(asset_lines))

    # --- Discovery Gaps ---
    # Show where estimated counts exceed known counts by asset type
    if profile.material_asset_types and profile.existing_assets:
        by_type_count: dict[str, int] = {}
        for a in profile.existing_assets:
            atype = a.naturesense_asset_type or "Unknown"
            by_type_count[atype] = by_type_count.get(atype, 0) + 1

        gaps: list[str] = []
        for t in profile.material_asset_types:
            if t.type and t.count is not None:
                known = by_type_count.get(t.type, 0)
                if t.count > known:
                    gaps.append(f"- {t.type}: {known} of ~{t.count} known")

        if gaps:
            gap_lines = ["\n## Discovery Gaps"]
            gap_lines.append("Asset types where known count is below estimated:")
            gap_lines.extend(gaps)
            sections.append("\n".join(gap_lines))

    # --- Search Guidance ---
    # Hints for the asset search pipeline
    guidance_lines = ["\n## Search Guidance"]
    guidance_lines.append(
        "When searching for assets owned by this company, consider:"
    )
    search_names = [profile.legal_name] + profile.aliases
    guidance_lines.append(
        f"- **Company names to search:** {', '.join(search_names)}"
    )
    if profile.subsidiaries:
        # Reuse the ranked list from corporate structure section
        parent_jur_g = (profile.jurisdiction or "").upper()

        def _sub_sort_key_g(s: Subsidiary) -> tuple:
            has_ownership = s.ownership_percentage is not None
            has_lei = bool(s.lei)
            has_real_name = bool(s.legal_name) and not s.legal_name.startswith("UK Company ")
            diff_jurisdiction = bool(s.jurisdiction) and s.jurisdiction.upper() != parent_jur_g
            return (has_ownership, has_real_name, diff_jurisdiction, has_lei)

        ranked_for_search = sorted(profile.subsidiaries, key=_sub_sort_key_g, reverse=True)
        sub_names = [s.legal_name for s in ranked_for_search if s.legal_name][:_SUB_SAMPLE_SIZE]
        if sub_names:
            guidance_lines.append(
                f"- **Subsidiary names to search:** {', '.join(sub_names)}"
            )
    if profile.operating_countries:
        guidance_lines.append(
            f"- **Countries to focus on:** "
            + ", ".join(_country_name(c) for c in profile.operating_countries)
        )
    if profile.material_asset_types:
        type_names = [t.type for t in profile.material_asset_types if t.type]
        if type_names:
            guidance_lines.append(
                f"- **Asset types to look for:** {', '.join(type_names)}"
            )
    sections.append("\n".join(guidance_lines))

    return "\n".join(sections) + "\n"
