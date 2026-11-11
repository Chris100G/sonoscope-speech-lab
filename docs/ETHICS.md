# Privacy, fairness, and responsible use

SpeakReady is an educational speaking-practice tool, not a medical, therapeutic, employment, identity, or safety decision system.

## Data handling

In local mode, decoding, feature extraction, analysis and compression are performed by the local Python process. The app does not intentionally upload audio or save recordings to disk. Browser microphone permissions are controlled by the browser. Practice history stores summary measurements only and is session-scoped.

With cloud hosting, uploaded or microphone audio is transmitted to the selected host for processing. Cloud filesystems may be temporary and are governed by the host's own policies; do not upload sensitive audio to a public demo.

## Limitations and fairness

Energy-based speech detection is sensitive to background noise, microphone gain, distance, and quiet speech. Pitch estimates are periodicity heuristics, not voice-quality or emotion measurements. Transcript-based word pace depends on transcript accuracy and whitespace tokenization; the reference ranges are adjustable examples, not population norms.

Do not use these measurements to rank, hire, diagnose, police, authenticate or make consequential decisions about anyone. The compressor cannot restore clipped audio or remove background noise; compare the processed copy to the original.

## Reproducibility

Record the sample rate, frame/hop settings, detection threshold, microphone and transcript when comparing takes. Recordings made with different devices or rooms are not directly comparable.
