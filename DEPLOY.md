# Deployment guide

## Streamlit Community Cloud (primary)

1. Push this repository to a GitHub repository visible to your Streamlit account.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app and select the repository, branch, and `app.py`.
3. Keep the Python version at 3.11 where configurable. The repository includes a lean `requirements.txt`, Linux system package list (`packages.txt`), and `.streamlit/config.toml`.
4. Deploy and inspect the app logs. Verify all eight navigation sections, upload handling, demo recognition, and profile import/export.
5. No cloud API credentials are required. If enabling optional local Whisper, account for its first-use model download.

Cloud instances are resource-constrained and ephemeral. The application must not depend on writable persistent storage: use in-memory state and temporary files that are cleaned up. Do not place uploads, recordings, or profile data in the repository or a shared persistent directory. The profile archive is an explicit user download/upload operation and can contain sensitive voice data.

## Hugging Face Spaces (Docker)

1. Create a Space and select **Docker** as the SDK.
2. Upload/push this repository. The root `Dockerfile` uses Python 3.11 and binds Streamlit to the required port `7860`.
3. Wait for the image build, then verify the Space's health and all app sections.
4. If the Space is public, remember that code and app outputs may be visible to others. Do not add recordings to the image or repository.

## Local Docker

Build and run:

```powershell
docker build -t sonoscope .
docker run --rm -p 7860:7860 sonoscope
```

Open <http://localhost:7860>. No audio is mounted into the container unless you explicitly add a volume.

## Configuration and troubleshooting

- Audio processing runs on CPU. Keep clips to a few seconds for the smoothest cloud experience.
- `libsndfile1` is listed in `packages.txt` and installed in the Docker image for SoundFile.
- WAV and FLAC are preferred for predictable decoding. MP3 support depends on the installed SoundFile/libsndfile codec build.
- The app has no mandatory credentials or hosted model dependency.
- Cloud filesystems are ephemeral; export profiles before restarting if you want to keep them.

## Deployment status

This repository contains deployment configuration and instructions; it does not claim that a public deployment has been created. A live deployment requires a GitHub repository and an authorized hosting account.
