"""Tests for LLM estimate refinement."""

import json
from unittest.mock import AsyncMock

import pytest

from corp_profile.profile import CompanyProfile, MaterialAssetType, refine_estimates


def _sample_profile() -> CompanyProfile:
    return CompanyProfile(
        issuer_id="test-001",
        legal_name="TestCo Mining Ltd",
        jurisdiction="AU",
        primary_industry="Metals & Mining",
        operating_countries=["AU", "CL"],
        material_asset_types=[
            MaterialAssetType(type="Mining Operations", count=50),
            MaterialAssetType(type="Office/Housing", count=10),
        ],
        estimated_asset_count=60,
    )


def test_refine_estimates_adjusts_counts():
    profile = _sample_profile()
    result = refine_estimates(profile, {
        "material_asset_types": [
            {"type": "Mining Operations", "count": 35},
            {"type": "Office/Housing", "count": 8},
        ],
        "estimated_asset_count": 43,
    })
    assert result.estimated_asset_count == 43
    assert result.material_asset_types[0].count == 35


def test_refine_estimates_preserves_unmodified():
    profile = _sample_profile()
    result = refine_estimates(profile, {})
    assert result.estimated_asset_count == 60
    assert len(result.material_asset_types) == 2


@pytest.mark.anyio
async def test_refine_estimates_stage_mock():
    from corp_profile.enrich import _refine_estimates_stage
    profile = _sample_profile()
    mock_provider = AsyncMock()
    mock_provider.complete.return_value = json.dumps({
        "material_asset_types": [
            {"type": "Mining Operations", "count": 35},
            {"type": "Office/Housing", "count": 8},
        ],
        "estimated_asset_count": 43,
    })
    result = await _refine_estimates_stage(profile, mock_provider)
    assert result.estimated_asset_count == 43
    mock_provider.complete.assert_called_once()
