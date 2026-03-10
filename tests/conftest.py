"""Shared test fixtures for corp-profile."""

import pytest

from corp_profile import CompanyProfile


SAMPLE_PROFILE_DATA = {
    "issuer_id": "ISS-001",
    "legal_name": "Acme Corp",
    "lei": "529900T8BM49AURSDO55",
    "jurisdiction": "US",
    "isin_list": ["US0378331005"],
    "all_isins": ["US0378331005", "US0378331013"],
    "aliases": ["Acme Corporation", "ACME"],
    "description": "A diversified industrial company.",
    "primary_industry": "Industrials",
    "operating_countries": ["US", "GB", "DE"],
    "business_segments": ["Manufacturing", "Services"],
    "subsidiaries": [
        {
            "issuer_id": "ISS-002",
            "legal_name": "Acme UK Ltd",
            "jurisdiction": "GB",
            "lei": "213800ABCDEF123456",
            "ownership_percentage": 100,
            "rel_type": "DIRECT",
        }
    ],
    "existing_assets": [
        {
            "asset_name": "Acme Plant Alpha",
            "address": "123 Industrial Rd, Chicago, IL",
            "latitude": 41.8781,
            "longitude": -87.6298,
            "naturesense_asset_type": "Manufacturing Facility",
            "capacity": 50000,
            "capacity_units": "tonnes/year",
            "status": "Operating",
        }
    ],
    "discovered_assets": [],
    "estimated_asset_count": 150,
    "material_asset_types": [{"type": "Manufacturing Facility", "count": 45}],
}


@pytest.fixture
def sample_profile_data() -> dict:
    """Return a complete sample profile dict."""
    return SAMPLE_PROFILE_DATA.copy()


@pytest.fixture
def sample_profile() -> CompanyProfile:
    """Return a CompanyProfile instance from sample data."""
    return CompanyProfile.model_validate(SAMPLE_PROFILE_DATA)
