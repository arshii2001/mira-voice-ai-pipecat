FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# TTS Configuration
ENV TTS_PROVIDER="elevenlabs"
ENV TTS_VOICE_GENDER="female"
ENV TTS_WS_URL="ws://localhost:8080/v1/audio/text-to-speech/stream"
ENV DEFAULT_VOICE="hi_male"

# LLM Configuration
ENV LLM_BASE_URL="http://vllm-gpt-oss-120b/v1"
ENV LLM_MODEL="openai/gpt-oss-120b"

# Server Configuration
ENV HOST="0.0.0.0"
ENV PORT="7860"
ENV DEFAULT_LANGUAGE="auto"

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run server
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
