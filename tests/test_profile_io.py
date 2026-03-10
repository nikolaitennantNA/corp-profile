"""Tests for profile I/O: from_dict, from_file, save."""

from corp_profile.profile import build_profile_from_dict, CompanyProfile


def test_build_profile_from_dict(sample_profile_data):
    profile = build_profile_from_dict(sample_profile_data)
    assert isinstance(profile, CompanyProfile)
    assert profile.issuer_id == "ISS-001"
    assert profile.legal_name == "Acme Corp"
    assert profile.lei == "529900T8BM49AURSDO55"
    assert len(profile.subsidiaries) == 1
    assert len(profile.existing_assets) == 1


def test_build_profile_from_dict_minimal():
    data = {"issuer_id": "ISS-999", "legal_name": "Minimal Co"}
    profile = build_profile_from_dict(data)
    assert profile.issuer_id == "ISS-999"
    assert profile.subsidiaries == []
    assert profile.description == ""


def test_build_profile_from_dict_invalid():
    import pytest
    with pytest.raises(Exception):
        build_profile_from_dict({"not_a_field": "bad"})
