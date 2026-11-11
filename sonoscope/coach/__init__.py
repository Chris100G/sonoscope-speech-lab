"""Practical, non-clinical speaking practice feedback."""

from .compressor import CompressionResult, compress_audio
from .metrics import PracticeReport, analyze_speech

__all__ = ["CompressionResult", "PracticeReport", "analyze_speech", "compress_audio"]
