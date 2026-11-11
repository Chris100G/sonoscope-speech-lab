"""Backend smoke tests for the self-contained SONOSCOPE modules."""

import io
import wave

import numpy as np

from sonoscope.assistant import (
    build_demo_recognizer,
    demo_commands,
    fuse_scores,
    recognize_demo_command,
    reject,
    synthesize_command,
)
from sonoscope.dsp.audio import load_audio, resample_audio, to_mono
from sonoscope.dsp.distances import dtw, dtw_distance
from sonoscope.dsp.features import (
    delta,
    extract_features,
    frame_signal,
    mel_filterbank,
    mfcc,
    window_frames,
)
from sonoscope.dsp.hmm import DiscreteHMM, backward, baum_welch, forward, viterbi
from sonoscope.dsp.quantization import lbg, quantize
from sonoscope.dsp.vocal import estimate_pitch, formant, lpc, lpc_resynthesis, vad
from sonoscope.recognizers import DTWTemplateRecognizer, VQHMMRecognizer


def test_audio_helpers_and_wav_bytes():
    assert np.allclose(to_mono(np.array([[1.0, -1.0], [0.5, 0.5]])), [0, 0.5])
    assert len(resample_audio(np.ones(100), 8000, 16000)) == 200
    data = io.BytesIO()
    with wave.open(data, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
    audio, rate = load_audio(data.getvalue(), target_sr=16000)
    assert rate == 16000
    assert len(audio) == 6
    assert np.max(np.abs(audio)) <= 1


def test_audio_loader_supports_flac_bytes():
    import soundfile as sf

    original = np.sin(2 * np.pi * 330 * np.arange(2205) / 22050)
    encoded = io.BytesIO()
    sf.write(encoded, original, 22050, format="FLAC")
    audio, rate = load_audio(encoded.getvalue(), target_sr=16000)
    assert rate == 16000
    assert len(audio) == 1600
    assert np.isfinite(audio).all()


def test_features_are_finite_and_have_expected_shapes():
    signal = np.sin(2 * np.pi * 220 * np.arange(1600) / 16000)
    assert frame_signal(signal, 400, 160).shape[1] == 400
    assert window_frames(np.ones((2, 8))).shape == (2, 8)
    assert mel_filterbank(16000, 512, 20).shape == (20, 257)
    features = mfcc(signal, 16000, n_mfcc=12, n_mels=24)
    assert features.shape[1] == 12
    assert np.isfinite(features).all()
    assert extract_features(signal, 16000, include_deltas=True).shape[1] == 39
    assert delta(np.ones((4, 3))).shape == (4, 3)


def test_vocal_tools_and_resynthesis():
    sample_rate = 8000
    signal = np.sin(2 * np.pi * 200 * np.arange(800) / sample_rate)
    assert abs(estimate_pitch(signal, sample_rate) - 200) < 15
    coeffs = lpc(signal, 12)
    assert coeffs.shape == (13,)
    assert formant(signal, sample_rate).ndim == 1
    assert lpc_resynthesis(np.ones(50), coeffs).shape == (50,)
    assert vad(np.r_[np.zeros(400), signal], 200, 100).any()


def test_distances_and_quantization():
    seq = np.array([[0.0], [1.0], [2.0]])
    assert dtw_distance(seq, seq) == 0
    cost, path = dtw(seq, seq)
    assert cost == 0 and len(path) == 3
    vectors = np.array([[0.0], [0.1], [4.9], [5.0]])
    codebook = lbg(vectors, 2)
    symbols = quantize(vectors, codebook)
    assert set(symbols) == {0, 1}
    from sonoscope.dsp.distances import cepstral_distance, itakura_distance

    assert cepstral_distance(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0
    assert itakura_distance(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0
    assert dtw_distance(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]), "itakura") == 0


def test_hmm_inference_and_training():
    pi = np.array([0.6, 0.4])
    transition = np.array([[0.7, 0.3], [0.2, 0.8]])
    emission = np.array([[0.9, 0.1], [0.1, 0.9]])
    observations = np.array([0, 0, 1])
    model = DiscreteHMM(pi, transition, emission)
    fwd_score, alpha = forward(observations, pi, transition, emission)
    back_score, beta = backward(observations, pi, transition, emission)
    path_score, path = viterbi(observations, pi, transition, emission)
    assert np.isclose(fwd_score, back_score)
    assert alpha.shape == beta.shape == (3, 2)
    assert path_score <= fwd_score and path.shape == (3,)
    trained = baum_welch([observations, observations], 2, 2, iterations=3)
    assert np.isfinite(model.score(observations))
    assert trained.transition.shape == (2, 2)
    assert np.allclose(trained.emission.sum(axis=1), 1)


def test_recognizers_and_fusion():
    a = np.array([[0.0], [0.1], [0.0]])
    b = np.array([[2.0], [2.1], [2.0]])
    query = np.array([[0.0], [0.05], [0.0]])
    template = DTWTemplateRecognizer().fit({"a": a, "b": b})
    assert template.predict(query) == "a"
    hmm = VQHMMRecognizer(codebook_size=2, n_states=2, iterations=3, seed=1)
    hmm.fit({"a": [a, query], "b": [b, b + 0.1]})
    assert hmm.predict(query) in {"a", "b"}
    fused = fuse_scores({"dtw": {"a": -0.1, "b": -2}, "hmm": {"a": -1, "b": -3}})
    assert max(fused, key=fused.get) == "a"
    assert reject({"unknown": -2}, threshold=0).rejected


def test_synthetic_multilingual_demo():
    assert {command.language for command in demo_commands()} == {"en", "hi", "kn"}
    recognizer = build_demo_recognizer()
    for command in demo_commands():
        signal = synthesize_command(command)
        result = recognize_demo_command(signal, recognizer=recognizer)
        assert result.label == command.key
