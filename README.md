# MiraVoiceAI Pipecat - Voice AI Pipeline

A Pipecat-based voice AI bot that orchestrates:
- **IndicASR-Streaming** for Speech-to-Text (WebSocket)
- **vLLM** for LLM inference (OpenAI-compatible API)
- **Svara-TTS-FastAPI** for Text-to-Speech (WebSocket streaming)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MiraVoiceAI Pipecat Server                  │
│                      (FastAPI WebSocket)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    Client Audio ──► STT ──► LLM ──► TTS ──► Audio to Client     │
│         ▲                                         │             │
│         │         ┌──────────┐                    │             │
│         └─────────│ Pipeline │────────────────────┘             │
│                   └──────────┘                                  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  External Services:                                             │
│  • IndicASR (ws://localhost:8082) - Streaming ASR               │
│  • vLLM (http://gpt-oss-120b/v1) - LLM inference                │
│  • Svara TTS (http://localhost:8080) - TTS synthesis            │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

Ensure the following services are running:

| Service | URL | Status Check |
|---------|-----|--------------|
| IndicASR-Streaming | `ws://localhost:8082/v1/audio/speech-to-text/stream` | `curl http://localhost:8082/health` |
| vLLM (GPT-OSS) | `http://gpt-oss-120b/v1` | `curl http://gpt-oss-120b/v1/models` |
| Svara-TTS | `ws://localhost:8080/v1/audio/text-to-speech/stream` | `curl http://localhost:8080/health` |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Components

Before running the full pipeline, verify all services are accessible:

```bash
# Test all components
python test_components.py

# Test specific component
python test_components.py --stt
python test_components.py --llm
python test_components.py --tts

# With custom URLs
python test_components.py --stt-url ws://localhost:8082/v1/audio/stream
```

### 3. Start the Server

```bash
# Start with defaults
python server.py

# Or with custom configuration
ASR_WS_URL=ws://localhost:8082/v1/audio/speech-to-text/stream \
TTS_WS_URL=ws://localhost:8080/v1/audio/text-to-speech/stream \
LLM_BASE_URL=http://gpt-oss-120b/v1 \
python server.py
```

### 4. Test the Pipeline

```bash
# Test with audio file
python test_client.py --audio sample.wav

# Test TTS directly
python test_client.py --text "नमस्ते, आप कैसे हैं?"

# Check server health
python test_client.py --health

# Check server config
python test_client.py --config
```

## API Reference

### WebSocket Endpoint

```
WebSocket /ws
```

**Query Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `voice` | `hi_male` | TTS voice ID |
| `language` | `auto` | STT language code |
| `sample_rate` | `16000` | Audio sample rate |

**Protocol:**
1. Connect to WebSocket with query parameters
2. Send PCM16 audio as binary frames
3. Receive transcription and audio responses
4. Connection stays open for continuous conversation

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/config` | GET | Get current configuration |
| `/voices` | GET | List available TTS voices |
| `/languages` | GET | List supported STT languages |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `ASR_WS_URL` | `ws://localhost:8082/v1/audio/speech-to-text/stream` | ASR WebSocket streaming URL |
| `TTS_WS_URL` | `ws://localhost:8080/v1/audio/text-to-speech/stream` | TTS WebSocket streaming URL |
| `LLM_BASE_URL` | `http://gpt-oss-120b/v1` | vLLM API base URL |
| `LLM_MODEL` | `openai/gpt-oss-120b` | LLM model name |
| `LLM_API_KEY` | `not-needed` | LLM API key |
| `DEFAULT_VOICE` | `hi_male` | Default TTS voice |
| `DEFAULT_LANGUAGE` | `auto` | Default STT language |
| `SYSTEM_PROMPT` | See code | LLM system prompt |

## Available Voices

| Voice ID | Name | Language |
|----------|------|----------|
| `hi_male` | Hindi (Male) | Hindi |
| `hi_female` | Hindi (Female) | Hindi |
| `en_male` | English (Male) | English |
| `en_female` | English (Female) | English |
| `ta_male` | Tamil (Male) | Tamil |
| `ta_female` | Tamil (Female) | Tamil |
| `te_male` | Telugu (Male) | Telugu |
| `te_female` | Telugu (Female) | Telugu |
| `bn_male` | Bengali (Male) | Bengali |
| `bn_female` | Bengali (Female) | Bengali |
| `mr_male` | Marathi (Male) | Marathi |
| `mr_female` | Marathi (Female) | Marathi |
| `kn_male` | Kannada (Male) | Kannada |
| `kn_female` | Kannada (Female) | Kannada |

## Supported Languages (STT)

| Code | Language |
|------|----------|
| `auto` | Auto-detect |
| `hi` | Hindi |
| `en` | English |
| `ta` | Tamil |
| `te` | Telugu |
| `bn` | Bengali |
| `mr` | Marathi |
| `gu` | Gujarati |
| `kn` | Kannada |
| `ml` | Malayalam |
| `pa` | Punjabi |
| `ur` | Urdu |

## Docker

### Build

```bash
docker build -t mira-voice-ai-pipecat .
```

### Run

```bash
docker run -d \
  --name mira-voice-ai-pipecat \
  -p 8000:8000 \
  -e ASR_WS_URL=ws://host.docker.internal:8082/v1/audio/speech-to-text/stream \
  -e TTS_WS_URL=ws://host.docker.internal:8080/v1/audio/text-to-speech/stream \
  -e LLM_BASE_URL=http://gpt-oss-120b/v1 \
  mira-voice-ai-pipecat
```

## Troubleshooting

### STT Not Connecting

```bash
# Check IndicASR health
curl http://localhost:8082/health

# Test WebSocket connection
python test_components.py --stt --stt-url ws://localhost:8082/v1/audio/stream
```

### LLM Errors

```bash
# Test vLLM endpoint
curl -X POST http://gpt-oss-120b/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "Hello"}]}'
```

### TTS Not Working

```bash
# Check Svara TTS health
curl http://localhost:8080/health

# List voices
curl http://localhost:8080/v1/speech/text-to-speech/voices

# Test synthesis
curl -X POST http://localhost:8080/v1/speech/text-to-speech \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "voice": "hi_male"}' \
  --output test.wav
```

## Client Integration

The server is compatible with Pipecat client SDKs:
- **JavaScript/Web**: `@pipecat/web-client`
- **React**: `@pipecat/react`
- **React Native**: `@pipecat/react-native`
- **iOS (Swift)**: Pipecat iOS SDK
- **Android (Kotlin)**: Pipecat Android SDK

Example JavaScript client:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws?voice=hi_male&language=auto');

ws.onmessage = (event) => {
  if (event.data instanceof Blob) {
    // Audio data - play it
    playAudio(event.data);
  } else {
    // JSON message - handle it
    const data = JSON.parse(event.data);
    console.log('Received:', data);
  }
};

// Send audio from microphone
mediaRecorder.ondataavailable = (event) => {
  ws.send(event.data);
};
```

## License

MIT License
