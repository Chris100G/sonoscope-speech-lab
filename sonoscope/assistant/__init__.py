"""Command assistant, fusion helpers, and built-in multilingual demo."""

from .demo import (
    DemoCommand,
    build_demo_recognizer,
    demo_commands,
    recognize_demo_command,
    synthesize_command,
)
from .recognition import fuse_scores, recognize_with_fusion, reject

__all__ = [
    "DemoCommand",
    "build_demo_recognizer",
    "demo_commands",
    "fuse_scores",
    "recognize_demo_command",
    "recognize_with_fusion",
    "reject",
    "synthesize_command",
]
