FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps — match pipecat-ai version from main server
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy test audio files (real speech WAVs for accurate VAD/STT testing)
COPY test_audio/ /app/test_audio/

# Copy test scripts
COPY test_rtvi.py .

ENTRYPOINT ["python", "test_rtvi.py"]
CMD ["--test", "all"]
