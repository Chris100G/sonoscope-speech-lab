---
title: SONOSCOPE
emoji: 🔊
colorFrom: teal
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
---

# SONOSCOPE

**A local-first speech laboratory and voice-command assistant** built with Streamlit, NumPy, and SciPy. SONOSCOPE demonstrates interpretable speech features and classical recognition methods without relying on librosa or a cloud transcription service.

> **Academic/demo software:** Results depend on recording quality, speaker, and command vocabulary. This is not a production-grade or safety-critical assistant.

## What is included

- Signal X-Ray: framing, windowing, mel filterbanks, MFCCs, and cepstral dynamics.
- Vocal Tract Studio: LPC/formants, pitch, and source-filter resynthesis demonstrations.
- Distance Lab: framewise metrics and dynamic time warping (DTW).
- HMM Lab: discrete-HMM inference and parameter estimation.
- Voice Assistant: synthetic multilingual command examples, user recordings, template matching, and confirmation-aware actions.
- Benchmark & Ethics: evaluation tools and clear local-processing/privacy guidance.
- Pronunciation Coach: an experimental DTW comparison, not a phonetic assessment.

The included demo vocabulary covers English, Hindi, and Kannada. It can be exercised without a microphone or optional cloud models.

## Run locally (Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Upload WAV/FLAC/MP3 audio or record through the browser microphone. In local mode, decoding and processing stay in the local app process; audio is not sent to an external service. Optional Whisper inference is local, though its model weights may be downloaded if they are not cached.

For development:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/smoke_test.py
```

## Optional integrations

Whisper, hmmlearn, and pyttsx3 are optional extras and are not installed by the lean deployment requirements. The optional local Whisper transcription and pyttsx3 playback controls report clearly when their packages are absent. SONOSCOPE's discrete HMM is implemented from scratch and does not rely on hmmlearn. For optional tools, install only what you intend to use (for example `python -m pip install openai-whisper pyttsx3 hmmlearn`); Whisper may download its model on first use after explicit opt-in.

## Project layout

```text
app.py
sonoscope/
  dsp/
  recognizers/
  assistant/
  ui/
tests/
scripts/smoke_test.py
docs/
```

## Deployment

- [Streamlit Community Cloud](DEPLOY.md#streamlit-community-cloud-primary)
- [Hugging Face Spaces](DEPLOY.md#hugging-face-spaces-docker)
- [Local Docker](DEPLOY.md#local-docker)

Deployments must use the repository's lean `requirements.txt` and `packages.txt`. Streamlit Community Cloud has limited CPU/memory, so keep audio clips short and avoid loading large neural models. Community Cloud builds the app from a GitHub repository; deployment is not performed automatically by this project.

## Privacy and limitations

- Local mode processes audio in memory and does not persist recordings by default.
- Avoid uploading sensitive or third-party recordings without consent.
- Session-state recordings are ephemeral and can be cleared by restarting the app/session.
- This project is not a secure identity-verification system; voice similarity is not authentication.
- Speech commands are suggestions, not automatically executed high-impact operations.
- Accent, language, microphone, and noise differences can cause errors. See the Ethics tab and `docs/ETHICS.md`.

## License

No license is declared yet. Add a license before redistributing this repository.
