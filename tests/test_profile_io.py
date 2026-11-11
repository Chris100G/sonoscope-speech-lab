"""Tests for safe portable profile archives."""

import io
import json
import zipfile

import pytest

from sonoscope.ui.profile_io import export_profile, import_profile


def test_profile_round_trip() -> None:
    profile = {"hello": [[[0.1, 0.2], [0.3, 0.4]]]}
    assert import_profile(export_profile(profile)) == profile


def test_rejects_unexpected_archive_members() -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("profile.json", json.dumps({"version": 1, "commands": {}}))
        archive.writestr("extra.txt", "unexpected")
    with pytest.raises(ValueError, match="only profile.json"):
        import_profile(stream.getvalue())


def test_rejects_non_finite_features() -> None:
    profile = {"hello": [[[float("nan")]]]}
    with pytest.raises(ValueError):
        export_profile(profile)
