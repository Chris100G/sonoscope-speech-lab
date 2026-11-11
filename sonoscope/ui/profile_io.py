"""Portable, bounded ZIP import/export for user enrollment profiles."""

from __future__ import annotations

import io
import json
import math
import zipfile
from typing import Any

_PROFILE_MEMBER = "profile.json"
_MAX_ARCHIVE_BYTES = 12 * 1024 * 1024
_MAX_PROFILE_BYTES = 10 * 1024 * 1024


def export_profile(commands: dict[str, list[list[list[float]]]]) -> bytes:
    """Serialize a feature-based enrollment profile without executable payloads."""
    payload = json.dumps(
        {"version": 1, "commands": commands},
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > _MAX_PROFILE_BYTES:
        raise ValueError("Profile is too large to export (10 MB maximum).")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_PROFILE_MEMBER, payload)
    result = output.getvalue()
    if len(result) > _MAX_ARCHIVE_BYTES:
        raise ValueError("Compressed profile is too large to export (12 MB maximum).")
    return result


def import_profile(archive_data: bytes) -> dict[str, list[list[list[float]]]]:
    """Validate and load a profile ZIP; never extract files to the filesystem."""
    if len(archive_data) > _MAX_ARCHIVE_BYTES:
        raise ValueError("Profile archive is too large (5 MB maximum).")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != _PROFILE_MEMBER:
                raise ValueError("Profile ZIP must contain only profile.json.")
            if members[0].file_size > _MAX_PROFILE_BYTES:
                raise ValueError("Profile data is too large (10 MB maximum).")
            raw: Any = json.loads(archive.read(_PROFILE_MEMBER))
    except (zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Invalid SONOSCOPE profile archive.") from error
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("Unsupported profile format.")
    commands = raw.get("commands")
    if not isinstance(commands, dict) or len(commands) > 200:
        raise ValueError("Profile must contain at most 200 named commands.")
    validated: dict[str, list[list[list[float]]]] = {}
    total_values = 0
    for label, samples in commands.items():
        if not isinstance(label, str) or not label.strip() or len(label) > 120:
            raise ValueError("Profile command labels must be non-empty strings under 120 characters.")
        if not isinstance(samples, list) or len(samples) > 100:
            raise ValueError(f"Invalid enrollment samples for {label!r}.")
        validated_samples: list[list[list[float]]] = []
        for sample in samples:
            if not isinstance(sample, list) or not sample or len(sample) > 10_000:
                raise ValueError(f"Invalid feature sequence for {label!r}.")
            frames: list[list[float]] = []
            dimension: int | None = None
            for frame in sample:
                if not isinstance(frame, list) or not frame or len(frame) > 512:
                    raise ValueError(f"Invalid feature frame for {label!r}.")
                if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in frame):
                    raise ValueError("Profile features must be numeric.")
                values = [float(value) for value in frame]
                if dimension is not None and len(values) != dimension:
                    raise ValueError("All frames in a feature sequence must have equal dimensions.")
                dimension = len(values)
                if any(not math.isfinite(value) for value in values):
                    raise ValueError("Profile features must be finite numbers.")
                frames.append(values)
                total_values += len(values)
            validated_samples.append(frames)
            if total_values > 1_000_000:
                raise ValueError("Profile contains too many feature values.")
        validated[label] = validated_samples
    return validated
