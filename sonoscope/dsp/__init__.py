"""Signal-processing primitives used by SONOSCOPE."""

from .audio import load_audio, resample_audio, to_mono
from .distances import (
    cepstral_distance,
    cosine_distance,
    dtw,
    euclidean_distance,
    itakura_distance,
)
from .features import (
    delta,
    extract_features,
    frame_signal,
    mel_filterbank,
    mfcc,
    pre_emphasis,
    window_frames,
)
from .hmm import DiscreteHMM, backward, baum_welch, forward, viterbi
from .quantization import lbg, quantize
from .speech import estimate_pitch, formants, lpc, lpc_resynthesis, vad

__all__ = [
    "DiscreteHMM",
    "backward",
    "baum_welch",
    "cepstral_distance",
    "cosine_distance",
    "delta",
    "dtw",
    "estimate_pitch",
    "euclidean_distance",
    "extract_features",
    "formants",
    "forward",
    "frame_signal",
    "itakura_distance",
    "lbg",
    "load_audio",
    "lpc",
    "lpc_resynthesis",
    "mel_filterbank",
    "mfcc",
    "pre_emphasis",
    "quantize",
    "resample_audio",
    "to_mono",
    "vad",
    "viterbi",
    "window_frames",
]
