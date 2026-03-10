"""Demo test — run with `uv run pytest tests/test_demo.py -v -s` to see output."""

from corp_profile import (
    CompanyProfile,
    build_context_document,
    build_profile_from_dict,
    save_profile,
)


DEMO_DATA = {
    "issuer_id": "ISS-7742",
    "legal_name": "TotalEnergies SE",
    "lei": "529900S21EQ1BO4ESM68",
    "jurisdiction": "FR",
    "isin_list": ["FR0000120271"],
    "all_isins": ["FR0000120271", "US89151E1091", "XS2334359818"],
    "aliases": ["Total SA", "TotalEnergies", "Total Fina Elf"],
    "description": (
        "TotalEnergies SE is a global integrated energy company operating across "
        "oil & gas exploration and production, refining, petrochemicals, and "
        "renewable energy including solar, wind, and battery storage."
    ),
    "primary_industry": "Oil, Gas & Consumable Fuels",
    "operating_countries": ["FR", "US", "NG", "AO", "QA", "AU", "BR", "GB"],
    "business_segments": [
        "Exploration & Production",
        "Integrated LNG",
        "Refining & Chemicals",
        "Marketing & Services",
        "Integrated Power",
    ],
    "subsidiaries": [
        {
            "issuer_id": "ISS-7743",
            "legal_name": "TotalEnergies EP Nigeria Ltd",
            "jurisdiction": "NG",
            "lei": None,
            "ownership_percentage": 100,
            "rel_type": "DIRECT",
        },
        {
            "issuer_id": "ISS-7744",
            "legal_name": "SunPower Corporation",
            "jurisdiction": "US",
            "lei": "54930054HSEHHCESP175",
            "ownership_percentage": 51,
            "rel_type": "DIRECT",
        },
        {
            "issuer_id": "ISS-7745",
            "legal_name": "TotalEnergies Renewables SAS",
            "jurisdiction": "FR",
            "lei": None,
            "ownership_percentage": 100,
            "rel_type": "DIRECT",
        },
    ],
    "existing_assets": [
        {
            "asset_name": "Donges Refinery",
            "address": "Donges, Loire-Atlantique, France",
            "latitude": 47.315,
            "longitude": -2.075,
            "naturesense_asset_type": "Petroleum Refinery",
            "capacity": 219000,
            "capacity_units": "barrels/day",
            "status": "Operating",
        },
        {
            "asset_name": "Feyzin Refinery",
            "address": "Feyzin, Rhône, France",
            "latitude": 45.672,
            "longitude": 4.860,
            "naturesense_asset_type": "Petroleum Refinery",
            "capacity": 109000,
            "capacity_units": "barrels/day",
            "status": "Operating",
        },
        {
            "asset_name": "Dunkirk LNG Terminal",
            "address": "Dunkirk, Nord, France",
            "latitude": 51.035,
            "longitude": 2.200,
            "naturesense_asset_type": "LNG Terminal",
            "capacity": 13000000,
            "capacity_units": "tonnes/year",
            "status": "Operating",
        },
        {
            "asset_name": "Speyerbach Solar Farm",
            "address": "Speyer, Rhineland-Palatinate, Germany",
            "latitude": 49.317,
            "longitude": 8.431,
            "naturesense_asset_type": "Solar Farm",
            "capacity": 75,
            "capacity_units": "MW",
            "status": "Operating",
        },
    ],
    "discovered_assets": [
        {
            "asset_name": "Seagreen Offshore Wind Farm",
            "asset_type": "Offshore Wind",
            "address": "Firth of Forth, Scotland, UK",
            "latitude": 56.55,
            "longitude": -2.10,
        },
    ],
    "estimated_asset_count": 3200,
    "material_asset_types": [
        {"type": "Petroleum Refinery", "count": 12},
        {"type": "LNG Terminal", "count": 6},
        {"type": "Solar Farm", "count": 45},
        {"type": "Wind Farm", "count": 22},
        {"type": "Oil Production Platform", "count": 85},
    ],
}


def test_demo_build_and_render():
    """Build a realistic profile and render it — run with -s to see output."""
    profile = build_profile_from_dict(DEMO_DATA)

    assert isinstance(profile, CompanyProfile)
    assert profile.legal_name == "TotalEnergies SE"
    assert len(profile.subsidiaries) == 3
    assert len(profile.existing_assets) == 4

    doc = build_context_document(profile)
    print("\n" + "=" * 70)
    print("RENDERED CONTEXT DOCUMENT")
    print("=" * 70)
    print(doc)
    print("=" * 70)

    assert "# TotalEnergies SE" in doc
    assert "LEI: 529900S21EQ1BO4ESM68" in doc
    assert "Subsidiaries (3)" in doc
    assert "SunPower Corporation" in doc
    assert "Donges Refinery" in doc
    assert "Seagreen Offshore Wind Farm" in doc
    assert "Estimated total assets: 3200" in doc


def test_demo_save_and_reload(tmp_path):
    """Show the JSON round-trip."""
    profile = build_profile_from_dict(DEMO_DATA)
    out = tmp_path / "totalenergies.json"
    save_profile(profile, str(out))

    print(f"\nSaved to {out} ({out.stat().st_size} bytes)")

    from corp_profile import build_profile_from_file

    reloaded = build_profile_from_file(str(out))
    assert reloaded.legal_name == profile.legal_name
    assert reloaded.model_dump() == profile.model_dump()
    print("Round-trip: OK — saved and reloaded identically")
