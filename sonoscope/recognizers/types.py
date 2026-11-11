"""Shared result type for recognizer implementations."""

from dataclasses import dataclass


@dataclass
class RecognitionResult:
    """Recognition decision and raw, higher-is-better candidate scores."""

    label: str | None
    score: float
    scores: dict[str, float]
    rejected: bool = False
