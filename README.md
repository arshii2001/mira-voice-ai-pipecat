# MiraVoiceAI Pipecat

Voice AI pipeline using Pipecat with Soniox STT and ElevenLabs TTS.

## Quick Start

1. Install dependencies:
```bash
uv sync
```

2. Set environment variables:
```bash
export SONIOX_API_KEY=your_soniox_key
export ELEVENLABS_API_KEY=your_elevenlabs_key
export LLM_BASE_URL=http://vllm-gpt-oss-120b/v1
# Note: Change this to your together.ai or groq endpoint
```

3. Run the server:
```bash
uv run python server.py
```

## Docker

### Build

```bash
docker build -t mira-voice-ai-pipecat .
```

### Run (Linux Production)

```bash
docker run -d \
  --name mira-voice-ai-pipecat \
  --network=host \
  -e SONIOX_API_KEY=your_soniox_key \
  -e ELEVENLABS_API_KEY=your_elevenlabs_key \
  -e LLM_BASE_URL=http://vllm-gpt-oss-120b/v1 \
  mira-voice-ai-pipecat
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SONIOX_API_KEY` | Yes | - | Soniox API key |
| `ELEVENLABS_API_KEY` | Yes | - | ElevenLabs API key |
| `LLM_BASE_URL` | No | `http://vllm-gpt-oss-120b/v1` | LLM API endpoint |
| `LLM_MODEL` | No | `openai/gpt-oss-120b` | LLM model name |
| `TTS_PROVIDER` | No | `elevenlabs` | TTS provider: `elevenlabs` or `svara` |
| `TTS_VOICE_GENDER` | No | `female` | Voice gender: `female` or `male` |

## Testing with the WebSocket Client

To quickly test the voice agent, use the Pipecat examples WebSocket client:

1. Clone the pipecat-examples repository:
```bash
git clone https://github.com/pipecat-ai/pipecat-examples.git
```

2. Navigate to the WebSocket client directory:
```bash
cd pipecat-examples/websocket/client
```

3. Install dependencies:
```bash
npm install
```

4. Start the client:
```bash
npm run dev
```

5. Open the URL shown in the terminal (typically `http://localhost:xxxx`)

6. Click connect and start talking to the voice agent

## API

| Endpoint | Description |
|----------|-------------|
| `ws://localhost:7860/ws` | WebSocket for audio streaming (PCM16, 16kHz, mono) |
| `GET /health` | Health check |
| `GET /config` | Current configuration |

## WebSocket Protocol

### Connection with Initial Context

Clients can provide initial conversation context when connecting. This allows pre-loading conversation history or user-specific information into the LLM context.

**Protocol:**
1. Client connects to `/ws` with optional query params (`voice`, `language`, `sample_rate`)
2. Client sends a JSON config message: `{"type": "config", "context": [...]}`
3. Server initializes the pipeline with the provided context
4. Client starts sending audio frames

**Important:** Always send a config message before streaming audio. If no context is needed, send an empty config:
```json
{"type": "config", "context": []}
```

**Example (JavaScript):**
```javascript
const ws = new WebSocket('ws://localhost:7860/ws?language=auto&sample_rate=16000');

ws.onopen = () => {
  // Send config message (required before sending audio)
  // Use empty context array if no conversation history is needed
  ws.send(JSON.stringify({
    type: "config",
    context: [
      {"role": "user", "content": "My name is John"},
      {"role": "assistant", "content": "Nice to meet you, John!"}
    ]
  }));

  // Then start sending audio frames...
};
```

**Context Message Format:**
- `type`: Must be `"config"`
- `context`: Array of messages in OpenAI format with `role` and `content` fields
  - `role`: `"user"` or `"assistant"`
  - `content`: The message text

**Note:** The server waits up to 5 seconds for a config message. Sending audio before the config message may result in lost frames. Always send the config message first.
