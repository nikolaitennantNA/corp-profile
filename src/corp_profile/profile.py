"""Build rich company context documents from corp-graph Postgres."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from .db import get_connection


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
    primary_industry: str = ""
    operating_countries: list[str] = []
    business_segments: list[str] = []
    subsidiaries: list[dict] = []
    existing_assets: list[dict] = []
    discovered_assets: list[dict] = []
    estimated_asset_count: int | None = None
    material_asset_types: list[dict] = []


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
            description=row.get("description") or "",
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
        profile.subsidiaries = list(seen.values())

        # 2b. Aggregate ISINs from parent + all subsidiaries
        sub_ids = [s["issuer_id"] for s in profile.subsidiaries if s.get("issuer_id")]
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
            a for a in all_assets
            if a.get("asset_name") or a.get("address", "")[:40] not in named
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
            SELECT asset_name, asset_type, address, latitude, longitude
            FROM discovered_assets WHERE issuer_id = %s
            """,
            [issuer_id],
        ).fetchall()
        profile.discovered_assets = [dict(d) for d in discovered]

        # 6. Asset estimates
        est = conn.execute(
            """
            SELECT estimated_assets_count, estimated_material_assets_count,
                   material_assets_types
            FROM asset_estimates WHERE issuer_id = %s
            """,
            [issuer_id],
        ).fetchone()
        if est:
            profile.estimated_asset_count = est.get("estimated_assets_count")
            raw_types = est.get("material_assets_types")
            if isinstance(raw_types, list):
                profile.material_asset_types = raw_types

    return profile


def build_profile_from_dict(data: dict) -> CompanyProfile:
    """Build a CompanyProfile from a plain dict (no DB required)."""
    return CompanyProfile.model_validate(data)


def save_profile(profile: CompanyProfile, path: str) -> None:
    """Save a CompanyProfile to a JSON file."""
    Path(path).write_text(
        json.dumps(profile.model_dump(), indent=2, default=str)
    )


def build_profile_from_file(path: str) -> CompanyProfile:
    """Load a CompanyProfile from a JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")
    data = json.loads(p.read_text())
    return build_profile_from_dict(data)


def build_context_document(profile: CompanyProfile) -> str:
    """Render a CompanyProfile as a structured text document for LLM consumption."""
    sections: list[str] = []

    # Company Overview
    overview_lines = [f"# {profile.legal_name}"]
    if profile.lei:
        overview_lines.append(f"LEI: {profile.lei}")
    if profile.jurisdiction:
        overview_lines.append(f"Jurisdiction: {profile.jurisdiction}")
    if profile.all_isins:
        overview_lines.append(f"ISINs: {', '.join(profile.all_isins)}")
    elif profile.isin_list:
        overview_lines.append(f"ISINs: {', '.join(profile.isin_list)}")
    if profile.aliases:
        overview_lines.append(f"Also known as: {', '.join(profile.aliases)}")
    if profile.primary_industry:
        overview_lines.append(f"Industry: {profile.primary_industry}")
    if profile.description:
        overview_lines.append(f"\n{profile.description}")
    if profile.operating_countries:
        overview_lines.append(
            f"Operating countries: {', '.join(profile.operating_countries)}"
        )
    if profile.business_segments:
        overview_lines.append(
            f"Business segments: {', '.join(profile.business_segments)}"
        )
    sections.append("\n".join(overview_lines))

    # Subsidiaries
    if profile.subsidiaries:
        sub_lines = [f"\n## Subsidiaries ({len(profile.subsidiaries)})"]
        for s in profile.subsidiaries:
            name = s.get("legal_name", "Unknown")
            jur = s.get("jurisdiction", "")
            lei = s.get("lei", "")
            pct = s.get("ownership_percentage")
            parts = [f"- {name}"]
            if jur:
                parts.append(f"({jur})")
            if lei:
                parts.append(f"[LEI: {lei}]")
            if pct is not None:
                parts.append(f"- {pct}% owned")
            sub_lines.append(" ".join(parts))
        sections.append("\n".join(sub_lines))

    # Existing Known Assets — aggregate by type, show sample
    if profile.existing_assets:
        # Group by asset type
        by_type: dict[str, list[dict]] = {}
        for a in profile.existing_assets:
            atype = a.get("naturesense_asset_type") or "Unknown"
            by_type.setdefault(atype, []).append(a)

        total = len(profile.existing_assets)
        named = [a for a in profile.existing_assets if a.get("asset_name")]
        unnamed = total - len(named)

        asset_lines = [f"\n## Existing Known Assets ({total} total, {len(named)} named)"]

        # Type breakdown
        asset_lines.append("Type breakdown:")
        for atype, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
            asset_lines.append(f"  - {atype}: {len(items)}")

        # Sample of named assets (max 30)
        if named:
            sample = named[:30]
            asset_lines.append(f"\nSample assets ({len(sample)} of {len(named)} named):")
            for a in sample:
                name = a.get("asset_name", "")
                atype = a.get("naturesense_asset_type", "")
                addr = a.get("address", "")
                status = a.get("status", "")
                parts = [f"- {name}"]
                if atype:
                    parts.append(f"[{atype}]")
                if addr:
                    parts.append(f"at {addr}")
                if status:
                    parts.append(f"({status})")
                asset_lines.append(" ".join(parts))
            if len(named) > 30:
                asset_lines.append(f"  ... and {len(named) - 30} more")

        sections.append("\n".join(asset_lines))

    # Previously Discovered Assets
    if profile.discovered_assets:
        disc_lines = [
            f"\n## Previously Discovered Assets ({len(profile.discovered_assets)})"
        ]
        for d in profile.discovered_assets:
            name = d.get("asset_name", "Unknown")
            atype = d.get("asset_type", "")
            addr = d.get("address", "")
            parts = [f"- {name}"]
            if atype:
                parts.append(f"[{atype}]")
            if addr:
                parts.append(f"at {addr}")
            disc_lines.append(" ".join(parts))
        sections.append("\n".join(disc_lines))

    # Asset Estimates
    if profile.estimated_asset_count is not None or profile.material_asset_types:
        est_lines = ["\n## Asset Estimates"]
        if profile.estimated_asset_count is not None:
            est_lines.append(
                f"Estimated total assets: {profile.estimated_asset_count}"
            )
        if profile.material_asset_types:
            est_lines.append("Material asset types:")
            for t in profile.material_asset_types:
                if isinstance(t, dict):
                    est_lines.append(f"  - {t}")
                else:
                    est_lines.append(f"  - {t}")
        sections.append("\n".join(est_lines))

    return "\n".join(sections) + "\n"
