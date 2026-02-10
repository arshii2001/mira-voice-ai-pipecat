# MiraVoiceAI — Pipeline Architecture & Implementation Notes

> **Version**: 2.0 · **Last updated**: February 2026
> **Stack**: Pipecat 0.0.98 · FastAPI 0.115.6 · Python 3.12

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Three Pipeline Modes](#2-three-pipeline-modes)
3. [Voice Pipeline (Pipecat)](#3-voice-pipeline-pipecat)
4. [Classroom Pipeline](#4-classroom-pipeline)
5. [Text Chat Pipeline](#5-text-chat-pipeline)
6. [Prompt Architecture (v4)](#6-prompt-architecture-v4)
7. [Key Components & Implementation Notes](#7-key-components--implementation-notes)
8. [Infrastructure](#8-infrastructure)
9. [Evaluation Framework (mira-eval)](#9-evaluation-framework-mira-eval)
10. [Improvement Roadmap](#10-improvement-roadmap)

---

## 1. System Overview

MiraVoiceAI is a multilingual AI tutor for Indian students (Grades 5–8) that operates in three modes: **1:1 Voice Tutor**, **Classroom Co-Teacher**, and **Text Chat**. The system is built on the [Pipecat](https://github.com/pipecat-ai/pipecat) real-time voice AI framework and uses external services for STT, LLM, and TTS.

### High-Level Data Flow

```
┌─────────────┐     WebSocket (RTVI/Protobuf)     ┌──────────────────────────────┐
│  OpenWebUI   │ ◄──────────────────────────────► │  FastAPI Server (server.py)  │
│  Frontend    │     HTTP (REST / SSE)             │  Port 7860                   │
└─────────────┘                                    └──────────┬───────────────────┘
                                                              │
                                            ┌─────────────────┼─────────────────┐
                                            ▼                 ▼                 ▼
                                     ┌────────────┐   ┌────────────┐   ┌────────────┐
                                     │ Voice Mode │   │ Classroom  │   │ Text Chat  │
                                     │ (bot.py)   │   │(classroom. │   │ (server.py │
                                     │            │   │   py)       │   │  /chat)    │
                                     └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
                                           │                │                │
                              ┌─────┬──────┼────────┐       │                │
                              ▼     ▼      ▼        ▼       ▼                ▼
                           Soniox  LLM  ElevenLabs  VAD   Translator      OpenAI
                           (STT)  (GPT) (TTS)     (Silero) (GPT-4o-mini) (GPT-4o-mini)
```

### External Services

| Service | Provider | Protocol | Purpose |
|---------|----------|----------|---------|
| **STT** | Soniox | WebSocket (`wss://stt-rt.soniox.com`) | Real-time multilingual ASR with speaker diarization, 50+ languages |
| **LLM** | OpenAI | HTTPS REST | `gpt-4o-mini` for tutoring responses and translation |
| **TTS** | ElevenLabs | WebSocket | `eleven_multilingual_v2` model, male/female voice presets |
| **VAD** | Silero (local) | In-process | Voice Activity Detection for barge-in support |

---

## 2. Three Pipeline Modes

| Mode | Entry Point | Transport | STT | LLM | TTS | Special Features |
|------|------------|-----------|-----|-----|-----|-----------------|
| **Voice Tutor** | `WS /ws` | Pipecat WebSocket (Protobuf) | Soniox | GPT-4o-mini | ElevenLabs | Barge-in, voice switching, text injection |
| **Classroom** | `WS /ws` + `room_id` | Pipecat WS (speaker) + classroom WS (listeners) | Soniox (speaker only) | GPT-4o-mini | ElevenLabs (per-listener) | Speaker token, real-time translation, broadcast, SQLite persistence |
| **Text Chat** | `POST /chat` | HTTP SSE (streaming) / JSON (sync) | — | GPT-4o-mini | — | Student context injection (name, topic) |

---

## 3. Voice Pipeline (Pipecat)

### Pipeline Stages (text_and_audio mode)

```
┌──────────────┐    ┌─────────┐    ┌──────────────────────┐    ┌─────────────────┐
│ 1. Transport │    │ 2. STT  │    │ 2b. TextInputInjector│    │ 3. UserTranscript│
│    Input     │───►│ Soniox  │───►│  (typed text inject) │───►│   Forwarder      │
│ (WebSocket)  │    │         │    │                      │    │ (→ client JSON)  │
└──────────────┘    └─────────┘    └──────────────────────┘    └────────┬─────────┘
                                                                        │
┌──────────────┐    ┌──────────┐    ┌──────────────────┐    ┌──────────▼─────────┐
│ 12. Assistant│    │ 11. Xport│    │ 10. TTS          │    │ 4. User Aggregator │
│  Aggregator  │◄───│  Output  │◄───│  ElevenLabs      │◄───│  (LLM context)     │
│              │    │          │    │                   │    └──────────┬─────────┘
└──────────────┘    └──────────┘    └──────────────────┘               │
                                           ▲                           ▼
                                           │               ┌──────────────────────┐
                    ┌──────────────────┐    │               │ 5. LLM (OpenAI)      │
                    │ 9. TextStream    │    │               │    GPT-4o-mini        │
                    │    Forwarder     │────┘               └──────────┬────────────┘
                    │ (→ client JSON)  │                               │
                    └──────────┬───────┘               ┌──────────────▼────────────┐
                               │                       │ 5b. ActionTagFilter       │
                    ┌──────────▼───────┐               │ (strip [TEACHER_ACTION:]) │
                    │ 8. Greeting      │               └──────────────┬────────────┘
                    │    Processor     │                               │
                    └──────────┬───────┘               ┌──────────────▼────────────┐
                               │                       │ 6. PipelineInstrumentor   │
                    ┌──────────▼───────┐               │ (metrics + logging)       │
                    │ 7. Extra         │               └──────────────┬────────────┘
                    │  Processors      │◄──────────────────────────────┘
                    │ (ClassroomBcast) │
                    └──────────────────┘
```

### Key Implementation Details

- **Barge-in**: Enabled via `allow_interruptions=True`. When VAD detects user speech during bot playback, `StartInterruptionFrame` cancels TTS and LLM generation. STT keeps its persistent WebSocket open.
- **Voice Switching**: LLM has a `select_voice` function tool. When triggered, it hot-swaps the ElevenLabs `voice_id` and reconnects the TTS WebSocket mid-session.
- **Text Injection**: The `TextInputInjector` processor allows typed text to enter the voice pipeline via `POST /inject_text`. It creates a synthetic `TranscriptionFrame` so the LLM processes it identically to speech.
- **TextStreamForwarder**: Intercepts `TextFrame` events from the LLM and sends them as JSON `{"type":"bot_text","text":"..."}` over the same WebSocket, enabling the frontend to display text while audio plays.
- **UserTranscriptForwarder**: Sends final STT transcriptions back to the client as JSON `{"type":"user_transcript","text":"..."}` for display.
- **Greeting**: On first connection, `GreetingProcessor` injects a `TTSSpeakFrame` with a welcome message. Skipped on reconnects (when conversation history exists).
- **WebSocket Binary Safety**: `make_websocket_binary_safe()` monkey-patches `receive_bytes()` to silently skip text frames, preventing `KeyError('bytes')` crashes when late config JSON arrives after the binary transport starts.

### Soniox STT Service (`services/soniox_stt.py`)

Custom Pipecat `STTService` implementation:
- **Persistent WebSocket** to `wss://stt-rt.soniox.com/transcribe-websocket`
- Streams PCM16 audio (16kHz, mono, 16-bit signed LE)
- Language identification across 50+ languages
- Speaker diarization support
- `detect_language_from_script()` fallback: inspects Unicode blocks (Devanagari → Hindi, Tamil script → Tamil, Kannada script → Kannada) when STT doesn't report language
- Language tag injection: prepends `[User is speaking Hindi]` to transcriptions for the LLM

### ElevenLabs TTS Service (`services/elevenlabs_tts.py`)

Factory wrapper around Pipecat's built-in `ElevenLabsTTSService`:
- **Model**: `eleven_multilingual_v2` (supports Hindi, Tamil, Kannada)
- **Voice presets**: Female (Monika Sogam) / Male (Ruhaan)
- **Tuned params**: `stability=0.75`, `similarity_boost=0.85`, `style=0.0` — prevents high-pitch first-syllable artifacts
- WebSocket streaming for low latency

---

## 4. Classroom Pipeline

### Architecture

```
                     ┌──────────────────────────────────────────────────────┐
                     │                    Room (in-memory)                  │
                     │  room_id, topic, conversation_history, users{}      │
                     └─────────────────────────┬────────────────────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
           ┌────────▼────────┐      ┌──────────▼──────────┐    ┌────────▼────────┐
           │ Speaker (Teacher)│      │ Listener A (Hindi)  │    │ Listener B (Tamil)│
           │ Full Pipecat     │      │ WS: /classroom/     │    │ WS: /classroom/   │
           │ pipeline via /ws │      │   rooms/{id}/ws     │    │   rooms/{id}/ws   │
           │ + ClassroomBcast │      │                     │    │                   │
           └────────┬─────────┘      └──────────┬──────────┘    └────────┬──────────┘
                    │                           │                         │
                    │  LLM response text        │ Translated text + TTS   │
                    │  (via broadcaster)        │ audio (PCM16 binary)    │
                    └───────────────────────────┴─────────────────────────┘
```

### Flow: Speaker Asks a Question

1. Speaker's audio → Soniox STT → transcription with language tag
2. `ClassroomBroadcaster` (Pipecat processor) intercepts the transcription:
   - Prepends `[Ravi asks]` speaker attribution
   - Broadcasts the question to all listeners (translated per language)
3. LLM generates response using co-teaching prompt
4. `ClassroomBroadcaster` intercepts LLM text output:
   - Streams sentence-by-sentence to listeners
   - Each sentence is translated in parallel (one `Translator.translate()` call per listener language)
   - Translated text is synthesized via per-listener `ClassroomTTS` (ElevenLabs REST API)
   - PCM16 audio chunks are streamed to each listener's WebSocket
5. Conversation history is appended to `room.conversation_history` for LLM context continuity
6. Messages are persisted to SQLite via `ClassroomDB`

### Classroom TTS (`services/classroom_tts.py`)

Standalone ElevenLabs TTS (not Pipecat's WebSocket service):
- Uses REST streaming API (`/v1/text-to-speech/{voice_id}/stream`)
- Returns raw PCM16 chunks via async generator
- No Pipecat pipeline or `StartFrame` required
- Used for listener audio delivery outside the main pipeline

### Translation (`translator.py`)

- Uses `AsyncOpenAI` client with `gpt-4o-mini`
- Low temperature (0.3) for faithful translation
- System prompt enforces: Devanagari for Hindi, Tamil script for Tamil, Kannada script for Kannada
- Skips translation when source == target language
- Session-level metrics: call count, average latency, skip count
- Graceful fallback: returns original text on error

### Database (`database.py`)

SQLite via `aiosqlite` with WAL mode for concurrent reads.

**Schema**:
- `sessions` — one row per classroom session (room activation period)
- `messages` — every transcription + bot response, with translations JSON
- `hand_raises` — hand-raise events within a session
- `reactions` — emoji reactions on messages (unique per user+message+emoji)
- `recordings` — audio recording metadata

**Indexes**: on `session_id`, `room_id`, `timestamp`, `status`

### Speaker Token Protocol

Only one user can speak at a time. The speaker token is managed via:
- `POST /classroom/rooms/{room_id}/token` — request or pass token
- WebSocket messages: `request_token`, `pass_token`, `release_token`
- `token_changed` events broadcast to all room users

### Teacher Commands

Toolbar buttons in the frontend send special commands:
- `[TEACHER_ACTION: SET_TOPIC <topic>]` — introduce a topic
- `[TEACHER_ACTION: QUIZ]` — generate 3 MCQ questions
- `[TEACHER_ACTION: SUMMARIZE]` — checkpoint summary
- `[TEACHER_ACTION: SIMPLIFY]` — re-explain more simply
- `[TEACHER_ACTION: NEXT]` — advance to next subtopic

The `ActionTagFilter` processor strips these tags from LLM output to prevent echo.

---

## 5. Text Chat Pipeline

### Flow

```
Client (POST /chat)
  │
  ▼
server.py: chat_completion()
  │
  ├─ Compose system prompt: load_system_prompt(version, mode="text")
  ├─ Inject student context: "--- STUDENT CONTEXT ---\nName: Ravi\nTopic: Fractions\n"
  ├─ Build messages array: [system_msg] + user conversation history
  │
  ▼
OpenAI API (streaming or sync)
  │
  ├─ stream=True  → SSE response (text/event-stream), tokens streamed as they arrive
  └─ stream=False → JSON response (full completion)
```

### Implementation Notes

- **Streaming**: Uses `httpx.AsyncClient` to proxy streaming responses from OpenAI. Each chunk is forwarded as an SSE event with `data: {json}` format.
- **Metrics**: Records TTFT (time to first token), total latency, token count, and tokens/sec.
- **Student Context**: `user_name` and `topic` fields in `ChatRequest` are appended to the system prompt as a `--- STUDENT CONTEXT ---` block.

---

## 6. Prompt Architecture (v4)

### Modular Composition

```
┌──────────────────────────────────────────────────────────┐
│                    v4-base.md                             │
│  Identity, Language Rules, Personality, Emotional         │
│  Awareness, Student Context, Anti-Repetition, Safety     │
└──────────────────────┬───────────────────────────────────┘
                       │
          ┌────────────┼────────────┬────────────────┐
          ▼            ▼            ▼                ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐   ┌──────────────┐
    │v4-voice  │ │v4-class  │ │v4-text   │   │ Dynamic      │
    │  .md     │ │  room.md │ │  .md     │   │ Context      │
    │          │ │          │ │          │   │ (appended)   │
    │Max 2-3   │ │Speaker   │ │Max 2-5   │   │Name: Ravi    │
    │sentences │ │attrib,   │ │sentences │   │Topic: Water  │
    │No format │ │broadcast │ │Prose,    │   │Cycle         │
    │Spoken    │ │Teacher   │ │minimal   │   │              │
    │tone      │ │commands  │ │emoji     │   │              │
    └──────────┘ └──────────┘ └──────────┘   └──────────────┘
```

### Prompt Loading (`bot.py: load_system_prompt()`)

```python
def load_system_prompt(version="v4", mode="voice"):
    if version.startswith("v4"):
        base = _load_prompt_file(f"{version}-base.md")      # prompts/v4-base.md
        mode_prompt = _load_prompt_file(f"{version}-{mode}.md")  # prompts/v4-voice.md
        return base + "\n\n" + mode_prompt
    else:
        return _load_prompt_file(f"{version}.md")  # Legacy single-file
```

### Dynamic Context Injection

Context is appended **after** the base+mode prompt:

```
--- STUDENT CONTEXT ---
Name: Ravi
Topic: The Water Cycle
```

- **Voice mode**: Context injected via initial config message → system prompt
- **Text mode**: `ChatRequest.user_name` and `ChatRequest.topic` → appended to system prompt
- **Classroom mode**: `room.current_lesson_topic` and `speaker.name` → appended to co-teaching prompt

### Base Prompt Key Rules (`v4-base.md`)

1. **Language Rules (highest priority)**: Respond in detected language only. Language tag `[User is speaking Hindi]` is single source of truth. No tag → English default.
2. **Script enforcement**: Hindi → Devanagari only (never Roman Hindi). Tamil → Tamil script. Kannada → Kannada script.
3. **Personality**: Warm older sister, curious, playful. Uses Indian references (monsoon, chai, cricket, IPL, Diwali).
4. **Emotional Awareness**: 3-tier system — Casual (1-2 sentences), Confused (2-3 sentences), Shutdown (3-4 sentences, zero teaching).
5. **Student Context**: Use name naturally (max once per response). Stay focused on topic if set.

### Mode-Specific Prompts

| Mode | File | Sentence Limit | Key Rules |
|------|------|---------------|-----------|
| Voice | `v4-voice.md` | 2-3 (hard cap) | No formatting, spoken tone, one idea per turn, Socratic teaching |
| Classroom | `v4-classroom.md` | 2-5 (default), 8-10 (intros/quizzes) | Speaker attribution, broadcast awareness, teacher commands |
| Text | `v4-text.md` | 2-5 (default), 7 (complex), 8 (hard cap) | Prose default, minimal emoji, toolbar commands |

---

## 7. Key Components & Implementation Notes

### `server.py` — FastAPI Application

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/ws` | WebSocket | Main voice pipeline (tutor + classroom) |
| `/chat` | POST | Text chat with SSE streaming |
| `/connect` | POST | Returns WebSocket URL for client |
| `/inject_text` | POST | Inject typed text into active voice session |
| `/health` | GET | Health check |
| `/metrics` | GET | Aggregated performance metrics |
| `/config` | GET | Current server configuration |
| `/voices` | GET | Available TTS voices |
| `/languages` | GET | Supported STT languages |
| `/classroom/*` | Various | Classroom room management (via router) |

**WebSocket Connection Flow**:
1. Client connects to `/ws`
2. Server waits 5s for optional config JSON (`type: "config"`)
3. Config may include: `system_prompt`, `context`, `mode`, `room_id`, `speaker_id`, `speaker_name`, `speaker_language`
4. Server sends `session_id` back to client
5. `make_websocket_binary_safe()` patches the WebSocket
6. If `room_id` present → classroom mode (attaches `ClassroomBroadcaster`)
7. `run_bot()` starts the Pipecat pipeline

### `bot.py` — Pipeline Orchestration

**Key Classes**:

| Class | Purpose |
|-------|---------|
| `PipelineInstrumentor` | Comprehensive timing diagnostics — tracks STT latency, LLM TTFT, LLM total, TTS start, turn latency |
| `ActionTagFilter` | Strips `[TEACHER_ACTION: ...]` and `[TUTOR_ACTION: ...]` tags from LLM output |
| `GreetingProcessor` | Injects welcome `TTSSpeakFrame` on `StartFrame`, skippable for reconnects |
| `TextStreamForwarder` | Sends LLM text as JSON to client WebSocket (parallel to TTS audio) |
| `UserTranscriptForwarder` | Sends STT transcriptions back to client as JSON |
| `TextInputInjector` | Converts typed text into `TranscriptionFrame` for the voice pipeline |
| `VoiceState` | Manages male/female voice toggle state |

**Pipeline Params**:
- `allow_interruptions=True` — enables barge-in
- `enable_metrics=True` — Pipecat internal metrics
- `enable_usage_metrics=True` — token usage tracking

### `classroom.py` — Classroom Mode

**Key Classes**:

| Class | Purpose |
|-------|---------|
| `RoomManager` | Creates/manages rooms, speaker tokens, LLM client, co-teaching prompt |
| `Room` | In-memory room state: users, speaker, topic, conversation history |
| `RoomUser` | User state: id, name, language, WebSocket, role |
| `ClassroomBroadcaster` | Pipecat processor — intercepts transcriptions and LLM output for broadcast |
| `MetricsCollector` | Singleton for aggregated metrics (LLM, translation, TTS, sessions) |

**Sentence-Level Streaming**:
The classroom broadcasts LLM output sentence-by-sentence (split on `.!?।\n`) rather than waiting for the full response. Each sentence is translated and TTS'd in parallel for all listeners.

### `translator.py` — Translation Service

- Wraps `AsyncOpenAI` for lightweight LLM-based translation
- System prompt enforces correct script (Devanagari, Tamil, Kannada)
- Strips `[User is speaking ...]` tags before translating
- Session metrics: call count, total latency, average latency

### `database.py` — SQLite Persistence

- Singleton `ClassroomDB` instance
- WAL mode + foreign keys enabled
- Auto-creates schema on `init()`
- Dashboard query support: `get_dashboard_stats()` aggregates sessions, messages, participants, hand raises, reactions

---

## 8. Infrastructure

### Docker (`Dockerfile`)

```dockerfile
FROM python:3.12-slim
# System deps: build-essential, libsndfile1, curl
# Python deps: requirements.txt (pipecat-ai, fastapi, uvicorn, websockets, aiohttp, etc.)
# Entrypoint: docker-entrypoint.sh (loads .env) → python server.py
# Port: 7860
# Health check: curl https://localhost:7860/health
```

### Docker Compose (`docker-compose.yml`)

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `mira-voice` | Local build | 7860 | Pipecat backend |
| `open-webui` | `../mira_openwebui` build | 3000 → 8080 | Frontend (Pipecat-enabled OpenWebUI fork) |
| `test-voice` | `tests/Dockerfile` | — | Automated voice pipeline tests (profile: test) |
| `test-ui` | `../mira_openwebui/cypress/Dockerfile` | — | Cypress E2E UI tests (profile: test) |

**Volumes**: `open-webui-data` (persistent), `classroom-data` (SQLite DB)
**Network**: `mira-network` (bridge)

### Kubernetes (`k8s/v2-deploy.yaml`)

**Namespace**: `openwebui`

| Resource | Name | Details |
|----------|------|---------|
| ConfigMap | `pipecat-v2-config` | STT_PROVIDER, TTS_PROVIDER, LLM_MODEL, PROMPT_VERSION, etc. |
| ConfigMap | `openwebui-v2-config` | PIPECAT_ENABLED, PIPECAT_API_URL, WEBUI_AUTH |
| PVC | `openwebui-v2-data-pvc` | 10Gi for OpenWebUI data |
| Deployment | `pipecat-v2` | 1 replica, 500m–1 CPU, 1–2Gi RAM, liveness/readiness probes |
| Deployment | `openwebui-v2` | 1 replica, 500m–1 CPU, 1–2Gi RAM, Recreate strategy |
| Service | `pipecat-v2-service` | ClusterIP → 7860 |
| Service | `openwebui-v2-service` | ClusterIP → 8080 |
| Ingress | `pipecat-v2-tailscale-ingress` | Tailscale HTTPS with auto TLS |
| Ingress | `openwebui-v2-tailscale-ingress` | Tailscale HTTPS with auto TLS |

**ACR Image**: `iaienterpriseacr.azurecr.io/audio:mira-voice-v2.0`
**Image Pull Secret**: `acr-iaienterpriseacr`

### Build & Deploy Flow

```
1. git push
2. az acr build -t audio:mira-voice-v2.0 -r iaienterpriseacr .
3. kubectl apply -f k8s/v2-deploy.yaml
4. kubectl rollout restart deployment/pipecat-v2 -n openwebui
5. kubectl rollout restart deployment/openwebui-v2 -n openwebui
```

---

## 9. Evaluation Framework (mira-eval)

A standalone Dockerized evaluation framework in `/Users/prasadvellanki/work/mira-eval/`.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      mira-eval (Docker)                      │
│                                                              │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ Personas     │   │ Scenarios    │   │ Judges           │  │
│  │ 6 students   │   │ 6 tutor     │   │ 6 score (1-5)    │  │
│  │ Hindi/English│   │ 2 classroom │   │ 3 yes/no         │  │
│  └──────┬──────┘   └──────┬───────┘   └──────┬───────────┘  │
│         │                 │                   │              │
│         ▼                 ▼                   ▼              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  eval_tutor.py / eval_classroom.py                    │   │
│  │  Simulates multi-turn conversations                   │   │
│  │  Student (GPT-4o-mini) ↔ Mira (/chat endpoint)       │   │
│  │  Then judges each conversation with GPT-4o            │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│              results/*.json (scores + transcripts)           │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
  Mira backend (http://host.docker.internal:7860/chat)
```

### Student Personas (6)

| Name | Age | Language | Archetype | Key Trait |
|------|-----|----------|-----------|-----------|
| Ravi | 11 | Hindi | Shy, struggling | First-gen learner, auto-rickshaw driver's son |
| Ananya | 12 | English | Confident, curious | IT parents, science enthusiast |
| Arjun | 10 | Hinglish | Bored, energetic | Cricket fan, class clown |
| Priya | 13 | English | Anxious, perfectionist | Board exam stress, apologizes a lot |
| Sneha | 11 | Hindi | Quiet, thoughtful | Army family, changed schools 3 times |
| Vikram | 12 | English | Practical, street-smart | Kirana shop helper, wants real-world use |

### Tutor Scenarios (6)

Water Cycle, Fractions, Photosynthesis, Percentages, Forces, Exam Prep

### Classroom Scenarios (2)

Science Grade 6 (Water Cycle, multilingual), Math Grade 7 (Fractions, mixed ability)

### Judges (9)

**Score-based (1–5 Likert)**:
1. `judge_friendly` — Warmth and friendliness
2. `judge_educational` — Learning encouragement
3. `judge_indian_cultural` — Indian cultural references
4. `judge_eq` — Emotional intelligence
5. `judge_socratic` — Socratic teaching quality
6. `judge_brevity` — Brevity and voice-readiness

**Yes/No**:
7. `judge_hindi_correct` — Devanagari script correctness
8. `judge_name_usage` — Natural name usage
9. `judge_no_repetition` — Avoids repetition

### Latest Eval Scores (Tutor, 6 scenarios)

| Dimension | Ravi | Ananya | Arjun | Priya | Sneha | Vikram | **Avg** |
|-----------|------|--------|-------|-------|-------|--------|---------|
| Friendly/Warm | 5 | 5 | 5 | 5 | 5 | 5 | **5.0** |
| Educational | 5 | 5 | 5 | 5 | 5 | 5 | **5.0** |
| Indian Cultural | 4 | 4 | 5 | 4 | 4 | 5 | **4.3** |
| Emotional Intelligence | 5 | 5 | 5 | 5 | 5 | 5 | **5.0** |
| Socratic Teaching | 5 | 5 | 5 | 4 | 4 | 5 | **4.7** |
| Brevity/Voice-Ready | 5 | 4 | 4 | 4 | 5 | 4 | **4.3** |

### Latest Eval Scores (Classroom, 2 scenarios)

| Dimension | Science (Multilingual) | Math (Mixed) | **Avg** |
|-----------|----------------------|--------------|---------|
| Speaker Attribution | 5 | 5 | **5.0** |
| Broadcast Awareness | 5 | 4 | **4.5** |
| Language Handling | 5 | 5 | **5.0** |
| Topic Continuity | 5 | 5 | **5.0** |
| Emotional Differentiation | 4 | 4 | **4.0** |
| Indian Cultural | 5 | 3 | **4.0** |
| Socratic Teaching | 5 | 4 | **4.5** |
| Classroom Energy | 5 | 4 | **4.5** |

---

## 10. Improvement Roadmap

### A. Prompt Optimization

| Area | Current State | Improvement | Impact |
|------|--------------|-------------|--------|
| **Indian Cultural References** | Avg 4.3/5 in tutor eval | Add a "Cultural Reference Bank" section to v4-base.md with 20+ ready-to-use examples organized by subject (science→monsoon, math→kirana shop, etc.) | Higher cultural score, more authentic feel |
| **Brevity in Voice Mode** | Avg 4.3/5, some responses too long | Strengthen v4-voice.md: add "If your response has >3 sentences, delete the least important one" rule. Add few-shot examples of ideal 2-sentence responses. | Faster TTS, better conversational rhythm |
| **Classroom Emotional Differentiation** | 4.0/5, treats all students similarly | Add to v4-classroom.md: "Adapt tone based on student's emotional cues — celebrate Ananya's curiosity differently than comforting Ravi's confusion" | More personalized classroom experience |
| **Socratic Depth** | 4.7/5 tutor, 4.5/5 classroom | Add explicit "Question Before Answer" examples in each mode prompt. Include a "Socratic Toolkit" with 10 question stems. | More discovery-based learning |
| **Anti-Repetition** | Good but can improve | Add a "Last 3 Response Memory" instruction: "Before responding, mentally review your last 3 responses and ensure this one differs in structure, opening, and examples." | More varied conversations |
| **Language Mixing Control** | Some inconsistency with Hinglish input | Add rule: "If input is Hinglish (Roman Hindi + English), respond in Hindi (Devanagari) with natural English technical terms." | Cleaner language handling |
| **EQ Prompt Extraction** | EQ rules embedded in base prompt | Extract EQ-specific rules into a standalone `v4-eq.md` module (inspired by Pi.ai's emotional intelligence prompt). Compose as: `base + eq + mode`. This allows independent tuning of EQ behavior. | Modular, tunable emotional intelligence |

### B. Model Fine-Tuning

| Stage | Approach | Data Source | Expected Outcome |
|-------|----------|-------------|------------------|
| **Stage 1: Supervised Fine-Tuning (SFT)** | Fine-tune GPT-4o-mini on curated Mira conversations | 500+ high-scoring eval transcripts (score ≥4 on all dimensions) + manually crafted gold-standard examples | Model learns Mira's personality, Indian cultural references, and Socratic style natively — reducing prompt length by ~60% |
| **Stage 2: DPO/RLHF** | Direct Preference Optimization using eval judge scores | Pairs of (good response, bad response) from eval runs. Good = score 5, Bad = score ≤3 on any dimension. | Model internalizes quality preferences: brevity, EQ, cultural grounding |
| **Stage 3: Distillation** | Distill from GPT-4o teacher to smaller model | GPT-4o generates ideal responses for 2000+ diverse scenarios, used to train a smaller model (GPT-4o-mini or open-source 7B) | Lower latency, lower cost, same quality |
| **Stage 4: Language-Specific Adapters** | LoRA adapters for Hindi, Tamil, Kannada | Monolingual conversation datasets in each language, sourced from eval framework + real classroom transcripts | Better script correctness, more natural code-mixing |

**Fine-Tuning Data Pipeline**:
```
mira-eval (generate conversations)
    → filter by judge scores (≥4 on all dimensions)
    → format as OpenAI fine-tuning JSONL
    → upload to OpenAI fine-tuning API
    → evaluate fine-tuned model with same eval framework
    → iterate
```

### C. Latency Optimization

| Bottleneck | Current | Target | Approach |
|------------|---------|--------|----------|
| **LLM TTFT** | ~400-800ms | <300ms | Shorter prompts (via fine-tuning), prompt caching (OpenAI), speculative decoding |
| **LLM Total** | ~1.5-3s | <1s | Shorter responses (better brevity prompt), smaller model after fine-tuning |
| **TTS Latency** | ~200-500ms | <150ms | ElevenLabs Turbo v2.5 model, pre-warm WebSocket, chunk-level streaming |
| **STT Latency** | ~100-300ms | <100ms | Already good with Soniox; consider endpoint detection tuning |
| **Translation** | ~300-600ms per language | <200ms | Batch translations, cache common phrases, consider local model |
| **End-to-End Turn** | ~2-4s (voice) | <1.5s | Parallel STT+VAD, streaming LLM→TTS (already implemented), reduce prompt tokens |
| **Classroom Broadcast** | ~3-4s per listener | <2s | Pre-translate common phrases, parallel TTS generation (already implemented), consider TTS caching for repeated content |

### D. Feature Improvements

| Feature | Description | Priority |
|---------|-------------|----------|
| **Prompt Caching** | Use OpenAI's prompt caching for the static base+mode prompt portion (saves ~50% of prompt tokens on repeated calls) | High |
| **Conversation Summarization** | After N turns, summarize conversation history to reduce context window size | Medium |
| **Adaptive Difficulty** | Track student's correct/incorrect answers and adjust explanation depth automatically | Medium |
| **Multi-Modal Support** | Add image input for math problems (photo of textbook page) | Low |
| **Offline Mode** | Local STT (Whisper) + local LLM (Llama 3) + local TTS for areas with poor connectivity | Low |
| **Analytics Dashboard** | Real-time teacher dashboard showing student engagement, question patterns, topic coverage | Medium |
| **Voice Cloning** | Custom teacher voice for classroom TTS (familiar voice for students) | Low |

### E. Eval Framework Improvements

| Improvement | Description |
|-------------|-------------|
| **More Personas** | Add 10+ personas covering more Indian states, languages (Telugu, Bengali, Marathi), and socioeconomic backgrounds |
| **Regression Testing** | Run eval suite on every prompt change; fail CI if any dimension drops below threshold |
| **Human-in-the-Loop** | Add human rating alongside LLM judge scores for calibration |
| **Latency Benchmarks** | Add response time assertions to eval: voice responses must complete in <2s |
| **A/B Testing** | Framework to compare two prompt versions side-by-side with same scenarios |
| **Adversarial Scenarios** | Add edge cases: student asking off-topic questions, testing safety boundaries, rapid language switching |

---

## Appendix: File Map

```
mira-voice-ai-pipecat/
├── server.py              # FastAPI app, endpoints, WebSocket handler
├── bot.py                 # Pipecat pipeline, processors, prompt loading
├── classroom.py           # Classroom mode: rooms, broadcast, metrics
├── translator.py          # LLM-based translation service
├── database.py            # SQLite persistence for classroom
├── services/
│   ├── soniox_stt.py      # Custom Soniox STT service
│   ├── elevenlabs_tts.py  # ElevenLabs TTS factory
│   ├── classroom_tts.py   # Standalone TTS for classroom listeners
│   └── svara_tts.py       # Alternative TTS (Svara)
├── prompts/
│   ├── v4-base.md         # Base prompt: identity, language, personality, EQ
│   ├── v4-voice.md        # Voice mode: brevity, spoken tone, Socratic
│   ├── v4-classroom.md    # Classroom: speaker attribution, broadcast, commands
│   ├── v4-text.md         # Text mode: prose, formatting, toolbar commands
│   ├── v3.md              # Legacy single-file prompt
│   └── v2.md              # Legacy single-file prompt
├── tests/
│   └── ...                # pytest tests + Dockerfile for test container
├── k8s/
│   └── v2-deploy.yaml     # Kubernetes deployment manifest
├── Dockerfile             # Application container
├── docker-compose.yml     # Local dev stack (mira-voice + open-webui + tests)
├── docker-entrypoint.sh   # .env loader for Docker
├── requirements.txt       # Python dependencies
└── ARCHITECTURE.md        # This document
```
