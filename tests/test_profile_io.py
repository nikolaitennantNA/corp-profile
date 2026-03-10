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


import json
from pathlib import Path

from corp_profile.profile import save_profile, build_profile_from_file


def test_save_profile(sample_profile, tmp_path):
    out = tmp_path / "profile.json"
    save_profile(sample_profile, str(out))
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["issuer_id"] == "ISS-001"
    assert data["legal_name"] == "Acme Corp"


def test_build_profile_from_file(sample_profile, tmp_path):
    out = tmp_path / "profile.json"
    save_profile(sample_profile, str(out))
    loaded = build_profile_from_file(str(out))
    assert loaded.issuer_id == sample_profile.issuer_id
    assert loaded.legal_name == sample_profile.legal_name
    assert loaded.subsidiaries == sample_profile.subsidiaries


def test_round_trip(sample_profile, tmp_path):
    out = tmp_path / "profile.json"
    save_profile(sample_profile, str(out))
    loaded = build_profile_from_file(str(out))
    assert loaded.model_dump() == sample_profile.model_dump()


def test_build_profile_from_file_not_found():
    import pytest
    with pytest.raises(FileNotFoundError):
        build_profile_from_file("/nonexistent/path.json")
