---
title: SpeakReady
emoji: 🎙️
colorFrom: teal
colorTo: cyan
sdk: streamlit
app_file: app.py
pinned: false
---

# SpeakReady

**[Open the live practice coach →](https://sonoscope-speech-lab-gaz6gbduw9ge5xjiwac5tc.streamlit.app/) · [Source on GitHub](https://github.com/Chris100G/sonoscope-speech-lab)**

**Make your next answer easier to deliver.** SpeakReady is a practical speaking-practice coach for interviews, meetings, class presentations and everyday conversations. Pick a realistic prompt, record a short take, see your speaking rhythm, pitch and pauses, and try one small change.

> SpeakReady measures audio features. It does not measure confidence, diagnose a speech condition, or decide how anyone should sound.

## Try it

1. Open [SpeakReady](https://sonoscope-speech-lab-gaz6gbduw9ge5xjiwac5tc.streamlit.app/).
2. In **Practice**, choose an interview, meeting, presentation or everyday prompt.
3. Record your answer in the browser or upload WAV, FLAC or MP3.
4. Optionally trim the start and end of your take.
5. Optionally paste your own transcript to estimate words per minute. There is no required speech-to-text service.
6. Choose a pace reference (or your own range), select **Review this take**, and explore the speaking/pause and pitch-contour charts plus a pace gauge.
7. Add a journal note and save the take to your private, session-only history to compare takes. Audio and transcripts are not added to history.
8. In **Audio compressor**, adjust threshold/ratio/attack/release, compare before and after with a mix slider, then download a WAV copy.
9. In **My progress**, view your pace and pause trends across saved takes, or enter comparison mode to compare two takes side by side.
10. Use the sidebar to switch dark mode, export your session to JSON, import a previous session, or start a new session.

**Illustrative sample:** the **My progress** tab can show a clearly marked example chart without uploading a recording. It is not speech or measured user data.

## What it measures

- **Speaking time and pauses:** energy-based voice-activity detection on 25 ms frames at 10 ms hops. Short gaps are joined; pauses are estimates, not word or breath boundaries.
- **Speaking pace:** whitespace-delimited transcript words divided by detected speaking time. A user-provided transcript is optional and its accuracy affects the estimate.
- **Pitch contour:** autocorrelation-based pitch estimates on voiced frames, with a median line and optional pitch-range band.
- **Recording level/clipping:** digital signal level, which changes with microphone, distance and device gain.
- **Compressor:** RMS-envelope, feed-forward, soft-knee dynamics processing implemented in the project. It is not a brick-wall limiter or denoiser.
- **Session history:** compares your saved take summaries; the optional example is explicitly illustrative. Export to JSON to keep a copy.

The pace ranges are editable practice references, not universal targets. Speech conventions vary across people, languages and contexts. Feedback is a suggestion, not a score.

## Features

- **Practice tab:** eight rehearsal prompts across interviews, meetings, presentations and everyday conversations; optional audio trimming; custom pace range; pace gauge; annotated rhythm timeline; pitch contour chart; journal notes; session-only history.
- **My progress tab:** pace and pause trend lines across saved takes; comparison mode to overlay two takes; exportable session JSON; clear history.
- **Audio compressor tab:** soft-knee compressor with threshold, ratio, knee, attack, release and make-up gain; before/after waveforms; mix slider; gain-over-time chart; WAV download.
- **Appearance:** light and dark mode, responsive layout, metric cards, section badges.
- **Privacy:** audio is processed in memory. Session history stores summary numbers only and lasts for the current browser session.

## Languages and accessibility

The interface supports English, Hindi and Kannada transcripts for rough word counting. Word counting is whitespace-based; it is not language parsing or transcription. Use keyboard-accessible browser controls, upload audio if microphone access is unavailable, and adjust browser zoom for larger controls.

## Privacy

- **Run locally for private practice:** the app analyzes audio in the local Python process, in memory.
- **Hosted app:** audio is sent to Streamlit Community Cloud to run the app. Do not upload sensitive recordings to the public demo.
- Recording audio is never put in practice history. Session history contains summary metrics and can be exported as JSON.
- Compression runs in memory; a WAV is created only when you explicitly download it.
- The hosted app does not use a cloud speech-transcription API. Add a transcript yourself if you want a pace estimate.

## Run locally (Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Upload a short take or use the browser microphone.

### Tests and smoke check

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/smoke_test.py
python -m ruff check app.py sonoscope tests scripts
```

## Deployment

- **Live app:** [sonoscope-speech-lab-gaz6gbduw9ge5xjiwac5tc.streamlit.app](https://sonoscope-speech-lab-gaz6gbduw9ge5xjiwac5tc.streamlit.app/)
- **Source:** [github.com/Chris100G/sonoscope-speech-lab](https://github.com/Chris100G/sonoscope-speech-lab)
- **Instructions:** [DEPLOY.md](DEPLOY.md)
- Streamlit Community Cloud deploys the current `main` branch from `app.py`; subsequent pushes redeploy automatically.

Local Docker is also supported:

```powershell
docker build -t speakready .
docker run --rm -p 7860:7860 speakready
```

## Project layout

```text
app.py                    Streamlit practice, progress, and compressor workflows
sonoscope/coach/           Speech-practice metrics and compressor DSP
sonoscope/dsp/             MFCC, LPC, DTW, VQ and discrete-HMM implementations
sonoscope/ui/              Charts, session management, and styling
sonoscope/recognizers/     DTW template and VQ-HMM recognizers
tests/                     DSP, recognition, coach, compressor, and session tests
scripts/smoke_test.py      Streamlit load and sample workflow check
docs/                      Algorithms, ethics, and deployment notes
```

## Responsible use

SpeakReady is not a medical, therapeutic, hiring, assessment, identity, or emergency system. It does not judge accents or prescribe a "correct" voice. Recording conditions strongly affect its estimates. See [docs/ETHICS.md](docs/ETHICS.md).

## License

No license is declared yet. Add a license before redistributing this repository.

## Project title and brief description

**SpeakReady - Speaking Practice Coach:** A privacy-minded web app for practicing interviews, presentations, meetings, and everyday conversations. It visualizes speech timing, pauses, and pitch, tracks session-only progress, and includes an audio compressor for listening copies.
