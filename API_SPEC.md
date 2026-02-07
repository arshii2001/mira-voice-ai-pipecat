# MIRA Voice AI — API Specification

> **Base URL:** `http://<host>:7860`  
> **Protocols:** REST (JSON) + WebSocket (Protobuf binary + JSON text)  
> **Audio Format:** 16-bit PCM, mono  
> **Audio In:** 16 kHz (client → server)  
> **Audio Out:** 24 kHz (server → client)

---

## Table of Contents

1. [REST Endpoints (Shared)](#1-rest-endpoints-shared)
2. [Tutor Mode (Single-User Voice)](#2-tutor-mode-single-user-voice)
3. [Classroom Mode (Multi-User)](#3-classroom-mode-multi-user)
4. [Data Types & Enums](#4-data-types--enums)
5. [Error Handling](#5-error-handling)
6. [Architecture Diagrams](#6-architecture-diagrams)

---

## 1. REST Endpoints (Shared)

### `GET /health`

Health check.

**Response:**
```json
{ "status": "healthy", "service": "mira-voice-ai-pipecat" }
```

---

### `GET /config`

Returns current server configuration — useful for UI to discover capabilities.

**Response:**
```json
{
  "stt_provider": "soniox",
  "stt_api_key_set": true,
  "supported_stt_providers": ["soniox", "deepgram", "whisper"],

  "tts_provider": "elevenlabs",
  "tts_ws_url": "ws://svara-tts/v1/audio/text-to-speech/stream",
  "elevenlabs_api_key_set": true,
  "tts_voice_gender": "female",
  "supported_tts_providers": ["elevenlabs", "svara", "openai"],

  "llm_provider": "openai",
  "llm_base_url": "https://api.openai.com/v1",
  "llm_model": "gpt-4o-mini",
  "supported_llm_providers": ["openai"],

  "default_voice": "en_female",
  "default_language": "auto",
  "supported_languages": ["en", "hi", "ta", "kn"],
  "supported_modes": ["text_and_audio", "text_only"]
}
```

---

### `POST /connect`

Returns the WebSocket URL for direct connection.

**Response:**
```json
{ "ws_url": "ws://hostname:7860/ws" }
```

---

### `GET /voices`

List available TTS voices.

**Response:**
```json
{
  "voices": [
    { "id": "en_male", "name": "English (Male)", "language": "English", "provider": "svara" },
    { "id": "JBFqnCBsd6RMkjVDRZzb", "name": "Rachel", "language": "Multilingual", "provider": "elevenlabs", "gender": "female" }
  ],
  "current_provider": "elevenlabs",
  "current_voice_gender": "female"
}
```

---

### `GET /languages`

List supported STT languages.

**Response:**
```json
{
  "languages": [
    { "code": "auto", "name": "Auto-detect" },
    { "code": "en", "name": "English" },
    { "code": "hi", "name": "Hindi" },
    { "code": "ta", "name": "Tamil" },
    { "code": "kn", "name": "Kannada" }
  ]
}
```

---

## 2. Tutor Mode (Single-User Voice)

One user talks to Mira. Full duplex voice with barge-in support.

### 2.1. Connection

**Endpoint:** `ws://<host>:7860/ws`  
**Protocol:** Pipecat RTVI over WebSocket  
**Serialization:** Protobuf (binary frames) + JSON (text messages)

#### Connection Flow

```
Client                                      Server
  |                                           |
  |--- WebSocket Connect (/ws) ------------->|
  |<-- 101 Switching Protocols --------------|
  |                                           |
  |--- JSON: config message (optional) ----->|  ← must be sent within 1 second
  |                                           |
  |  (Pipecat RTVI handshake begins)          |
  |<-- Protobuf: BOT_READY -----------------|
  |                                           |
  |<-- JSON: bot_text_complete (greeting) ---|  ← "Namaste! I'm Mira..."
  |<-- Protobuf: audio frames (greeting) ----|  ← TTS audio (if text_and_audio mode)
  |                                           |
  |--- Protobuf: audio frames (user mic) --->|  ← 16kHz PCM, chunked
  |<-- Protobuf: transcription (interim) ----|  ← partial STT results
  |<-- Protobuf: transcription (final) ------|  ← final STT result
  |                                           |
  |<-- JSON: bot_text (streaming chunks) ----|  ← per-token LLM output
  |<-- JSON: bot_text_complete --------------|  ← full LLM response
  |<-- Protobuf: audio frames (bot voice) ---|  ← TTS audio (if text_and_audio mode)
  |                                           |
  |  (repeat for each turn)                   |
```

### 2.2. Config Message (Client → Server)

**Must be sent as the first message** after WebSocket opens (within 1 second timeout). If not sent, defaults are used.

```json
{
  "type": "config",
  "mode": "text_and_audio",
  "system_prompt": "You are a helpful tutor...",
  "context": [
    { "role": "user", "content": "Previous question" },
    { "role": "assistant", "content": "Previous answer" }
  ]
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | string | Yes | — | Must be `"config"` |
| `mode` | string | No | `"text_and_audio"` | `"text_and_audio"` or `"text_only"` |
| `system_prompt` | string | No | Built-in v3 prompt | Custom system prompt for the LLM |
| `context` | array | No | `[]` | Prior conversation messages |

### 2.3. Audio Frames (Bidirectional)

Audio is sent as **Pipecat Protobuf frames** (binary WebSocket messages).

**Client → Server (mic audio):**
- Format: 16-bit signed PCM, mono, 16 kHz
- Wrapped in Pipecat `Frame.audio` protobuf
- Chunk size: typically 512 samples (~32ms)
- Send continuously while mic is active

**Server → Client (bot audio):**
- Format: 16-bit signed PCM, mono, 24 kHz
- Wrapped in Pipecat `Frame.audio` protobuf
- Only sent when `mode = "text_and_audio"`

> **Note for UI developers:** Use `@pipecat-ai/client-js` SDK with `WebSocketTransport` and `ProtobufFrameSerializer` for automatic handling. See the existing `pipecat.ts` for reference implementation.

### 2.4. JSON Text Messages (Server → Client)

These are sent as **text WebSocket messages** (not binary) alongside protobuf audio frames.

#### `bot_text` — Streaming LLM Token

Sent for each LLM token as it's generated. Use for real-time text rendering.

```json
{
  "type": "bot_text",
  "text": "Hello",
  "streaming": true
}
```

#### `bot_text_complete` — Full Response

Sent when the LLM response is complete. Use to finalize the displayed text.

```json
{
  "type": "bot_text_complete",
  "text": "Hello! How can I help you today?"
}
```

> **Note:** `bot_text_complete` is also sent for the initial greeting immediately on connection.

### 2.5. RTVI SDK Events

If using the Pipecat SDK (`@pipecat-ai/client-js`), these callbacks fire automatically:

| SDK Callback | When | Data |
|---|---|---|
| `onConnected` | WebSocket connected | — |
| `onBotReady` | Pipeline ready | — |
| `onUserStartedSpeaking` | VAD detects speech | — |
| `onUserStoppedSpeaking` | VAD detects silence | — |
| `onUserTranscript` | STT result | `{ text: string, final: boolean }` |
| `onBotTranscript` | Bot text token | `{ text: string }` |
| `onTrackStarted` | Audio track available | `MediaStreamTrack` |
| `onTrackStopped` | Audio track ended | `MediaStreamTrack` |
| `onDisconnected` | Connection closed | — |
| `onError` | Error occurred | `{ message: string }` |

### 2.6. Modes

| Mode | Audio Out | Text JSON | Use Case |
|------|-----------|-----------|----------|
| `text_and_audio` | ✅ Protobuf audio frames | ✅ `bot_text` + `bot_text_complete` | Full voice experience with live text |
| `text_only` | ❌ No audio | ✅ `bot_text` + `bot_text_complete` | Low bandwidth, text-only display |

Both modes accept audio input (mic) for STT. The difference is only in output.

---

## 3. Classroom Mode (Multi-User)

Multiple users in a room. One holds the **speaker token** and talks to Mira. All others are **listeners** who receive translated text + audio in their preferred language.

### 3.1. Room Management (REST)

#### `POST /classroom/rooms` — Create Room

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `"Classroom"` | Room display name |

**Response:**
```json
{
  "room_id": "a1b2c3d4",
  "name": "Physics Class",
  "created_at": 1707300000.0,
  "user_count": 0,
  "users": [],
  "speaker_id": null,
  "speaker_name": null,
  "token_queue": []
}
```

---

#### `GET /classroom/rooms` — List Rooms

**Response:**
```json
{
  "rooms": [
    {
      "room_id": "a1b2c3d4",
      "name": "Physics Class",
      "created_at": 1707300000.0,
      "user_count": 3,
      "users": [
        { "user_id": "ravi-01", "name": "Ravi", "language": "hi", "mode": "text_and_audio", "is_speaker": true },
        { "user_id": "priya-02", "name": "Priya", "language": "ta", "mode": "text_and_audio", "is_speaker": false },
        { "user_id": "anil-03", "name": "Anil", "language": "kn", "mode": "text_only", "is_speaker": false }
      ],
      "speaker_id": "ravi-01",
      "speaker_name": "Ravi",
      "token_queue": []
    }
  ]
}
```

---

#### `GET /classroom/rooms/{room_id}` — Get Room

**Response:** Same as single room object above.

**Errors:** `404` if room not found.

---

#### `DELETE /classroom/rooms/{room_id}` — Delete Room

**Response:**
```json
{ "status": "deleted" }
```

**Errors:** `404` if room not found.

---

#### `POST /classroom/rooms/{room_id}/token` — Manage Speaker Token

**Query Parameters:**

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `action` | string | Yes | `"request"`, `"pass"`, or `"release"` |
| `user_id` | string | Yes | The user performing the action |
| `to_user_id` | string | No | Target user for `"pass"` action |

**Actions:**

| Action | Behavior |
|--------|----------|
| `request` | Request token. Granted immediately if free, queued if taken. |
| `pass` | Pass to `to_user_id` (or next in queue if omitted). |
| `release` | Release token. Auto-assigns to next in queue, or leaves free. |

**Response (request):**
```json
{ "granted": true, "speaker_id": "ravi-01" }
```

**Response (pass / release):**
```json
{ "speaker_id": "priya-02" }
```

---

### 3.2. Listener WebSocket

**Endpoint:** `ws://<host>:7860/classroom/rooms/{room_id}/ws`  
**Protocol:** Plain JSON (text messages) + raw binary PCM audio

This WebSocket is for **all users** in the room (both potential speakers and listeners). It handles room events, token control, and receiving translated broadcasts.

#### Connection Flow

```
Client                                      Server
  |                                           |
  |--- WebSocket Connect ------------------->|
  |<-- 101 Switching Protocols --------------|
  |                                           |
  |--- JSON: join message ------------------>|  ← must be sent within 10 seconds
  |<-- JSON: joined (room state) ------------|
  |<-- JSON: token_changed (if auto) --------|  ← first user gets token automatically
  |                                           |
  |  (listen for events)                      |
  |<-- JSON: transcription (translated) -----|  ← speaker's words in your language
  |<-- JSON: bot_audio_start ----------------|
  |<-- Binary: PCM audio chunks -------------|  ← TTS audio of translation
  |<-- JSON: bot_audio_end ------------------|
  |<-- JSON: bot_response (translated) ------|  ← Mira's answer in your language
  |<-- JSON: bot_audio_start ----------------|
  |<-- Binary: PCM audio chunks -------------|
  |<-- JSON: bot_audio_end ------------------|
  |                                           |
  |--- JSON: request_token ----------------->|  ← request to speak
  |<-- JSON: token_response -----------------|
  |<-- JSON: token_changed ------------------|  ← broadcast to all
```

### 3.3. Client → Server Messages

#### `join` — Join Room (Required First Message)

```json
{
  "type": "join",
  "user_id": "ravi-01",
  "name": "Ravi",
  "language": "hi",
  "mode": "text_and_audio"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | string | Yes | — | Must be `"join"` |
| `user_id` | string | No | Auto-generated | Unique user identifier |
| `name` | string | No | `"User-{id}"` | Display name |
| `language` | string | No | `"en"` | Preferred language: `en`, `hi`, `ta`, `kn` |
| `mode` | string | No | `"text_and_audio"` | `"text_and_audio"` or `"text_only"` |

---

#### `request_token` — Request Speaker Token

```json
{ "type": "request_token" }
```

**Server responds with:**
```json
{ "type": "token_response", "granted": true }
```
- `granted: true` — you are now the speaker
- `granted: false` — added to queue, you'll get a `token_changed` event when it's your turn

---

#### `pass_token` — Pass Token to Another User

```json
{ "type": "pass_token", "to": "priya-02" }
```
- If `to` is omitted, passes to next user in the queue (round-robin).

---

#### `release_token` — Release Token

```json
{ "type": "release_token" }
```
- Token goes to the next user in queue, or becomes free.

---

#### `set_mode` — Switch Output Mode at Runtime

```json
{ "type": "set_mode", "mode": "text_only" }
```

**Server confirms with:**
```json
{ "type": "mode_changed", "mode": "text_only" }
```

---

### 3.4. Server → Client Messages

#### `joined` — Room State on Join

```json
{
  "type": "joined",
  "room": {
    "room_id": "a1b2c3d4",
    "name": "Physics Class",
    "user_count": 2,
    "users": [ ... ],
    "speaker_id": "ravi-01",
    "speaker_name": "Ravi",
    "token_queue": []
  },
  "you": {
    "user_id": "priya-02",
    "name": "Priya",
    "language": "ta",
    "mode": "text_and_audio",
    "is_speaker": false
  }
}
```

---

#### `user_joined` — Another User Joined

```json
{
  "type": "user_joined",
  "user": {
    "user_id": "anil-03",
    "name": "Anil",
    "language": "kn",
    "mode": "text_and_audio",
    "is_speaker": false
  }
}
```

---

#### `user_left` — User Left

```json
{ "type": "user_left", "user_id": "anil-03" }
```

---

#### `token_changed` — Speaker Changed

Broadcast to ALL users whenever the speaker token changes.

```json
{
  "type": "token_changed",
  "speaker_id": "priya-02",
  "speaker_name": "Priya"
}
```

When no one is speaking:
```json
{
  "type": "token_changed",
  "speaker_id": null,
  "speaker_name": null
}
```

---

#### `transcription` — Speaker's Words (Translated)

Sent to listeners when the speaker says something.

```json
{
  "type": "transcription",
  "text": "[User is speaking Hindi] भारत के बारे में बताओ",
  "language": "hi",
  "translated_text": "இந்தியா பற்றி சொல்லுங்கள்",
  "tts_text": "Ravi asks: இந்தியா பற்றி சொல்லுங்கள்",
  "target_language": "ta",
  "user_id": "ravi-01",
  "speaker_name": "Ravi"
}
```

| Field | Description |
|-------|-------------|
| `text` | Original transcription (in speaker's language) |
| `language` | Detected source language |
| `translated_text` | Text translated to listener's language |
| `tts_text` | Text with speaker attribution (what TTS speaks) |
| `target_language` | Listener's language code |
| `user_id` | Speaker's user_id |
| `speaker_name` | Speaker's display name |

---

#### `bot_response` — Mira's Answer (Translated)

```json
{
  "type": "bot_response",
  "text": "भारत एक बहुत बड़ा देश है...",
  "language": "hi",
  "translated_text": "இந்தியா மிகவும் பெரிய நாடு...",
  "tts_text": "Mira says: இந்தியா மிகவும் பெரிய நாடு...",
  "target_language": "ta"
}
```

---

#### `bot_audio_start` / `bot_audio_end` — Audio Boundary Markers

Sent before and after binary audio frames. Only sent if the user's mode is `text_and_audio`.

```json
{ "type": "bot_audio_start" }
```

Between `bot_audio_start` and `bot_audio_end`, the server sends **binary WebSocket messages** containing raw 16-bit PCM audio at 24 kHz mono. Buffer and play these sequentially.

```json
{ "type": "bot_audio_end" }
```

---

#### `token_response` — Response to Token Request

```json
{ "type": "token_response", "granted": true }
```

---

#### `mode_changed` — Mode Switch Confirmation

```json
{ "type": "mode_changed", "mode": "text_only" }
```

---

#### `error` — Error Message

```json
{ "type": "error", "message": "Room not found" }
```

---

### 3.5. Speaker Pipeline (Active Speaker Only)

When a user holds the speaker token and wants to talk to Mira, they connect to the **main `/ws` endpoint** (same as tutor mode) with additional classroom fields in the config:

```json
{
  "type": "config",
  "mode": "text_and_audio",
  "room_id": "a1b2c3d4",
  "speaker_id": "ravi-01"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `room_id` | string | Yes | Room to broadcast to |
| `speaker_id` | string | Yes | Must match current token holder |

This opens a **full Pipecat voice pipeline** (STT → LLM → TTS) for the speaker. The pipeline automatically broadcasts:
- Speaker's transcription → translated to all listeners
- Mira's response → translated to all listeners

The speaker hears Mira directly. Listeners hear translated versions.

**Important:** The speaker must have already joined the room via the classroom WebSocket (`/classroom/rooms/{room_id}/ws`) and hold the speaker token before connecting to `/ws`.

**Validation errors:**
```json
{ "type": "error", "message": "Room not found" }
{ "type": "error", "message": "Speaker token required for classroom session" }
{ "type": "error", "message": "Speaker must join classroom room first" }
```

---

## 4. Data Types & Enums

### Language Codes

| Code | Language |
|------|----------|
| `en` | English |
| `hi` | Hindi |
| `ta` | Tamil |
| `kn` | Kannada |
| `auto` | Auto-detect (STT only) |

### Modes

| Mode | Description |
|------|-------------|
| `text_and_audio` | Receive both text JSON messages and audio (TTS) |
| `text_only` | Receive only text JSON messages, no audio |

### Token Actions

| Action | Description |
|--------|-------------|
| `request` | Request the speaker token (queued if busy) |
| `pass` | Pass token to specific user or next in queue |
| `release` | Release token (auto-assigns or leaves free) |

---

## 5. Error Handling

### REST Errors

| Status | When |
|--------|------|
| `404` | Room not found |
| `400` | Invalid action or parameters |

### WebSocket Errors

Sent as JSON `{"type": "error", "message": "..."}` before closing.

| Error | When |
|-------|------|
| `"Room not found"` | Room ID doesn't exist |
| `"Expected join message"` | First message was not a `join` |
| `"Join timeout"` | No join message within 10 seconds |
| `"Speaker token required for classroom session"` | `/ws` with `room_id` but user doesn't hold token |
| `"Speaker must join classroom room first"` | `/ws` with `room_id` but user hasn't joined via classroom WS |
| `"Invalid mode: ..."` | Invalid mode value in `set_mode` |

---

## 6. Architecture Diagrams

### Tutor Mode

```
┌─────────────┐         WebSocket /ws           ┌──────────────────┐
│   Browser    │ ◄─────────────────────────────► │   Pipecat Server │
│  (Open WebUI │   Protobuf audio (bidirectional)│                  │
│   or custom) │   JSON text (server → client)   │  STT → LLM → TTS│
└─────────────┘                                  └──────────────────┘
```

### Classroom Mode

```
                      ┌──────────────────────────────────────┐
                      │           Pipecat Server              │
                      │                                      │
  Speaker             │  ┌──────────────────────────┐        │
  ┌────────┐  /ws     │  │  Full Pipeline            │        │
  │ Ravi   │─────────►│  │  STT → LLM → TTS         │        │
  │ (Hindi)│◄─────────│  │      │                    │        │
  └────────┘ audio+   │  │      ▼                    │        │
              text    │  │  ClassroomBroadcaster     │        │
                      │  │      │                    │        │
                      │  └──────┼────────────────────┘        │
                      │         │                             │
                      │         ▼                             │
                      │  ┌──────────────────┐                 │
                      │  │  Room Manager     │                 │
                      │  │  + Translator     │                 │
                      │  │  + TTS (per user) │                 │
                      │  └──────┬───────────┘                 │
                      │         │                             │
                      │    ┌────┼─────────┐                   │
                      │    ▼    ▼         ▼                   │
  Listener 1         │  /classroom/rooms/{id}/ws             │
  ┌────────┐         │                                      │
  │ Priya  │◄────────│  JSON: translated text (Tamil)        │
  │ (Tamil)│◄────────│  Binary: TTS audio (Tamil)            │
  └────────┘         │                                      │
                      │                                      │
  Listener 2         │                                      │
  ┌────────┐         │                                      │
  │ Anil   │◄────────│  JSON: translated text (Kannada)      │
  │(Kannada│◄────────│  Binary: TTS audio (Kannada)          │
  │text_only)        │  (no audio — text_only mode)          │
  └────────┘         │                                      │
                      └──────────────────────────────────────┘
```

### Classroom Typical Flow (End-to-End)

```
1. POST /classroom/rooms?name=Physics     → creates room "a1b2c3d4"

2. Ravi connects:   ws://host/classroom/rooms/a1b2c3d4/ws
   → sends:  {"type":"join","user_id":"ravi","name":"Ravi","language":"hi"}
   ← receives: {"type":"joined", "room":{...}, "you":{...}}
   ← receives: {"type":"token_changed", "speaker_id":"ravi", "speaker_name":"Ravi"}

3. Priya connects:  ws://host/classroom/rooms/a1b2c3d4/ws
   → sends:  {"type":"join","user_id":"priya","name":"Priya","language":"ta"}
   ← receives: {"type":"joined", "room":{...}, "you":{...}}

4. Ravi (speaker) opens voice pipeline:
   ws://host/ws
   → sends: {"type":"config","room_id":"a1b2c3d4","speaker_id":"ravi"}
   ← Pipecat handshake + greeting audio

5. Ravi speaks Hindi → Mira responds in Hindi
   ← Priya receives on classroom WS:
     {"type":"transcription", "translated_text":"...(Tamil)...", "speaker_name":"Ravi"}
     {"type":"bot_audio_start"}
     [binary PCM audio in Tamil]
     {"type":"bot_audio_end"}
     {"type":"bot_response", "translated_text":"...(Tamil)..."}
     {"type":"bot_audio_start"}
     [binary PCM audio in Tamil]
     {"type":"bot_audio_end"}

6. Ravi passes token:
   → sends: {"type":"pass_token","to":"priya"}
   ← all receive: {"type":"token_changed","speaker_id":"priya","speaker_name":"Priya"}

7. Priya opens voice pipeline on /ws with her speaker_id
```

---

## Quick Reference: WebSocket URLs

| Endpoint | Purpose | Protocol |
|----------|---------|----------|
| `ws://host:7860/ws` | Tutor mode (or classroom speaker) | Protobuf + JSON |
| `ws://host:7860/classroom/rooms/{id}/ws` | Classroom join + listen | JSON + binary PCM |
