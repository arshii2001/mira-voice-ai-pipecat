# MiraVoiceAI Pipecat

Voice AI pipeline using Pipecat with Soniox STT and ElevenLabs/Svara TTS.

Supports English, Hindi, Tamil, and Kannada.

## Quick Start

```bash
# Build
docker build -t mira-voice .

# Run with .env file
docker run -d --name mira-voice -p 7860:7860 --env-file .env mira-voice

# Or with explicit env vars
docker run -d --name mira-voice -p 7860:7860 \
  -e SONIOX_API_KEY=$SONIOX_API_KEY \
  -e ELEVENLABS_API_KEY=$ELEVENLABS_API_KEY \
  -e LLM_BASE_URL=http://your-llm-server/v1 \
  -e LLM_MODEL=your-model \
  -e LLM_API_KEY=$LLM_API_KEY \
  mira-voice
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SONIOX_API_KEY` | Yes | - | Soniox STT API key |
| `ELEVENLABS_API_KEY` | Yes* | - | ElevenLabs API key (*if using ElevenLabs) |
| `TTS_PROVIDER` | No | `elevenlabs` | `elevenlabs` or `svara` |
| `TTS_VOICE_GENDER` | No | `female` | `female` or `male` (ElevenLabs) |
| `LLM_BASE_URL` | No | `http://vllm-gpt-oss-120b/v1` | OpenAI-compatible LLM endpoint |
| `LLM_MODEL` | No | `openai/gpt-oss-120b` | Model name |
| `LLM_API_KEY` | No | `DUMMY_KEY` | LLM API key |
| `HOST` | No | `0.0.0.0` | Server bind address |
| `PORT` | No | `7860` | Server port |

See `.env.example` for Svara TTS and other optional settings.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /config` | Current configuration |
| `WS /ws` | WebSocket for voice streaming |

## WebSocket Protocol

Connect to `/ws` for bidirectional audio streaming. Optionally send a config message within 5 seconds to customize the assistant:

```json
{
  "type": "config",
  "system_prompt": "You are a helpful assistant...",
  "context": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"}
  ]
}
```

| Field | Description |
|-------|-------------|
| `system_prompt` | Custom system prompt (default: `prompts/v0.md`) |
| `context` | Prior conversation history to pre-load |

If no config message is sent, defaults are used after 5 seconds.

### Example

```javascript
const ws = new WebSocket('wss://localhost:7860/ws');

ws.onopen = () => {
  // Optional: send config for custom behavior
  ws.send(JSON.stringify({
    type: "config",
    system_prompt: "You are a pirate. Keep responses brief."
  }));

  // Then stream audio...
};
```

## Testing

Use the [Pipecat WebSocket client](https://github.com/pipecat-ai/pipecat-examples/tree/main/websocket/client):

```bash
git clone https://github.com/pipecat-ai/pipecat-examples.git
cd pipecat-examples/websocket/client
npm install && npm run dev
```
