"""Speech-recognition strategies."""

from .dtw import DTWTemplateRecognizer
from .types import RecognitionResult
from .vq_hmm import VQHMMRecognizer

__all__ = ["DTWTemplateRecognizer", "RecognitionResult", "VQHMMRecognizer"]
