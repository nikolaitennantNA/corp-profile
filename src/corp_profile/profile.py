"""Build rich company context documents from corp-graph Postgres."""

from __future__ import annotations

from pydantic import BaseModel

from .db import get_connection


class CompanyProfile(BaseModel):
    """Structured company profile assembled from corp-graph data."""

    issuer_id: str
    legal_name: str
    lei: str | None = None
    jurisdiction: str | None = None
    isin_list: list[str] = []
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


def build_profile(isin: str) -> CompanyProfile:
    """Query corp-graph Postgres and build a CompanyProfile for the given ISIN."""
    with get_connection() as conn:
        # 1. Main entity from company_universe
        row = conn.execute(
            "SELECT * FROM company_universe WHERE isin_list @> ARRAY[%s]::text[]",
            [isin],
        ).fetchone()
        if not row:
            raise LookupError(f"No entity found for ISIN {isin}")

        issuer_id = row["issuer_id"]

        profile = CompanyProfile(
            issuer_id=issuer_id,
            legal_name=row.get("legal_name", ""),
            lei=row.get("lei"),
            jurisdiction=row.get("jurisdiction"),
            isin_list=row.get("isin_list") or [],
            aliases=row.get("aliases") or [],
            description=row.get("description") or "",
            primary_industry=row.get("primary_industry") or "",
            operating_countries=row.get("operating_countries") or [],
            business_segments=row.get("business_segments") or [],
        )

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
        profile.subsidiaries = [dict(s) for s in subs]

        # 3. Existing assets from ALD
        assets = conn.execute(
            """
            SELECT asset_name, address, latitude, longitude, naturesense_asset_type,
                   capacity, capacity_units, status
            FROM assets WHERE issuer_id = %s
            """,
            [issuer_id],
        ).fetchall()
        profile.existing_assets = [dict(a) for a in assets]

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


def build_context_document(profile: CompanyProfile) -> str:
    """Render a CompanyProfile as a structured text document for LLM consumption."""
    sections: list[str] = []

    # Company Overview
    overview_lines = [f"# {profile.legal_name}"]
    if profile.lei:
        overview_lines.append(f"LEI: {profile.lei}")
    if profile.jurisdiction:
        overview_lines.append(f"Jurisdiction: {profile.jurisdiction}")
    if profile.isin_list:
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

    # Existing Known Assets
    if profile.existing_assets:
        asset_lines = [f"\n## Existing Known Assets ({len(profile.existing_assets)})"]
        for a in profile.existing_assets:
            name = a.get("asset_name", "Unknown")
            atype = a.get("naturesense_asset_type", "")
            addr = a.get("address", "")
            status = a.get("status", "")
            cap = a.get("capacity")
            units = a.get("capacity_units", "")
            parts = [f"- {name}"]
            if atype:
                parts.append(f"[{atype}]")
            if addr:
                parts.append(f"at {addr}")
            if status:
                parts.append(f"({status})")
            if cap is not None:
                parts.append(f"- capacity: {cap} {units}".strip())
            asset_lines.append(" ".join(parts))
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
