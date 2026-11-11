# Deployment guide

## Streamlit Community Cloud (primary)

The live app is [SpeakReady on Streamlit Community Cloud](https://sonoscope-speech-lab-gaz6gbduw9ge5xjiwac5tc.streamlit.app/), deployed from the public [GitHub repository](https://github.com/Chris100G/sonoscope-speech-lab), branch `main`, entrypoint `app.py`.

To create another deployment or redeploy manually:

1. In [Streamlit Community Cloud](https://share.streamlit.io/), select **Create app** and the repository, branch, and `app.py`.
2. Keep the Python version at 3.11 where configurable. The repository includes a lean `requirements.txt`, Linux system package list (`packages.txt`), and `.streamlit/config.toml`.
3. Deploy and inspect the app logs. Verify the practice, progress, compressor and guide flows.
4. No cloud API credentials or hosted transcription service are required.

Cloud instances are resource-constrained and ephemeral. Practice analysis is in-memory; session history stores summary numbers only and resets with the session. Hosted audio is sent to Streamlit Community Cloud for processing. Do not upload sensitive recordings to the public demo.

## Hugging Face Spaces (Docker)

1. Create a Space and select **Docker** as the SDK.
2. Upload/push this repository. The root `Dockerfile` uses Python 3.11 and binds Streamlit to the required port `7860`.
3. Wait for the image build, then verify the Space's health and all app sections.
4. If the Space is public, remember that code and app outputs may be visible to others. Do not add recordings to the image or repository.

## Local Docker

Build and run:

```powershell
docker build -t speakready .
docker run --rm -p 7860:7860 speakready
```

Open <http://localhost:7860>. No audio is mounted into the container unless you explicitly add a volume.

## Configuration and troubleshooting

- Audio processing runs on CPU. The app accepts short audio clips up to 90 seconds.
- `libsndfile1` is listed in `packages.txt` and installed in the Docker image for SoundFile.
- WAV and FLAC are preferred for predictable decoding. MP3 support depends on the installed SoundFile/libsndfile codec build.
- The app has no mandatory credentials or hosted model dependency.
- Cloud filesystems are ephemeral; export the summary JSON if you want to keep it. Audio is not stored in that export.

## Deployment status

The public Community Cloud app is live at [sonoscope-speech-lab-gaz6gbduw9ge5xjiwac5tc.streamlit.app](https://sonoscope-speech-lab-gaz6gbduw9ge5xjiwac5tc.streamlit.app/). Its health endpoint has returned HTTP 200. The app source is in [Chris100G/sonoscope-speech-lab](https://github.com/Chris100G/sonoscope-speech-lab).
