# Privacy, fairness, and responsible use

SONOSCOPE is an educational speech laboratory, not an identity, medical, employment, or safety decision system.

## Data handling

In local mode, decoding, feature extraction, and recognition are performed by the local Python process. The application does not intentionally upload audio or save recordings to disk. Browser microphone permissions are controlled by the browser. The optional Whisper baseline runs locally after model weights are available; first use may download those weights. Profile archives may contain biometric-like voice features and must be protected accordingly.

With cloud hosting, uploaded or microphone audio is transmitted to the selected host for processing. Cloud filesystems may be temporary and are governed by the host's own policies; do not upload sensitive audio to a public demo.

## Limitations and fairness

Small command datasets and synthetic signals cannot represent real-world speech diversity. Recognition may vary with language, accent, age, voice characteristics, microphone, speaking rate, and environment. Report per-speaker and per-language results, test noisy and quiet conditions, and avoid treating scores as calibrated probabilities unless calibration has been measured.

Voice similarity is not secure authentication. Require explicit human confirmation before actions, especially actions with external effects. Never use these demonstrations to infer protected traits or to make consequential decisions about people.

## Reproducibility

Record the sample rate, feature configuration, vocabulary, train/test speaker split, and random seed when reporting results. Keep evaluation speakers separate from enrollment speakers where possible. Do not tune thresholds on the test set.
