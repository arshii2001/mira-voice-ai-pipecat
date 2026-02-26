"""
MiraVoiceAI Pipecat Server - FastAPI WebSocket Server

This server provides a WebSocket endpoint for voice interaction with the
MiraVoiceAI bot that orchestrates STT -> LLM -> TTS pipeline.

Endpoints:
- WebSocket /ws - Main bot interaction endpoint
- GET /health - Health check
- GET /config - Get current configuration
"""

import asyncio
import json
import logging
import os
import signal
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from auth import AUTH_ENABLED, decode_openwebui_jwt, extract_bearer_token, get_verified_user_id

from bot import (
    run_bot,
    get_text_injector,
    load_system_prompt,
    PROMPT_VERSION,
    DEFAULT_LANGUAGE,
    DEFAULT_VOICE,
    TTS_WS_URL,
    LLM_PROVIDER,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_API_KEY,
    STT_PROVIDER,
    SONIOX_API_KEY,
    DEEPGRAM_API_KEY,
    STT_LANGUAGE_HINTS,
    TTS_PROVIDER,
    ELEVENLABS_API_KEY,
    TTS_VOICE_GENDER,
)
from services.elevenlabs_tts import VOICE_PRESETS
from classroom import router as classroom_router, room_manager, ClassroomBroadcaster, _metrics_collector
from maths_manager import get_maths_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))


# ── JWT auth dependency for all REST endpoints ──────────────────────
async def _require_jwt(request: Request):
    """FastAPI dependency: reject requests without a valid JWT when auth is enabled."""
    if not AUTH_ENABLED:
        return  # Dev/test mode — no JWT required

    auth_header = request.headers.get("authorization")
    user_id = get_verified_user_id(auth_header)
    if not user_id:
        raise HTTPException(status_code=401, detail="Valid JWT required")
    return user_id


def _normalize_user_language(lang: Optional[str]) -> str:
    """Normalize user language to one of en/hi/ta; unknowns default to English."""
    if not lang:
        return "en"
    raw = str(lang).strip().lower()
    base = raw.split("-", 1)[0].split("_", 1)[0]
    alias_map = {
        "english": "en",
        "eng": "en",
        "hindi": "hi",
        "hin": "hi",
        "tamil": "ta",
        "tam": "ta",
    }
    normalized = alias_map.get(base, base)
    return normalized if normalized in {"en", "hi", "ta"} else "en"


def _stt_hints_for_registered_language(lang: Optional[str]) -> list[str]:
    """Allowed STT languages by registered user language."""
    normalized = _normalize_user_language(lang)
    if normalized == "hi":
        return ["en", "hi"]
    if normalized == "ta":
        return ["en", "ta"]
    return ["en"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting MiraVoiceAI Pipecat server...")

    # ── Startup config check ──────────────────────────────────────────
    # Logs the status of every critical env var at boot.
    # On Railway: check the deployment logs for any MISSING entries.
    # Set all vars in Railway's 'Variables' panel, NOT in .env
    # (.env is not copied into the Docker image on Railway).
    from provider_config import LLM, TTS, STT
    logger.info("[CONFIG CHECK] ── Environment variables at startup ──")
    logger.info(f"[CONFIG CHECK] LLM_PROVIDER    = {LLM.provider}")
    logger.info(f"[CONFIG CHECK] LLM_BASE_URL    = {LLM.base_url}")
    logger.info(f"[CONFIG CHECK] LLM_MODEL       = {LLM.model}")
    logger.info(f"[CONFIG CHECK] LLM_API_KEY     = {'SET ✓' if LLM.api_key else 'MISSING ✗ — chat/voice will fail!'}")
    logger.info(f"[CONFIG CHECK] STT_PROVIDER    = {STT.provider}")
    logger.info(f"[CONFIG CHECK] STT_API_KEY     = {'SET ✓' if (STT.soniox_api_key or STT.deepgram_api_key) else 'MISSING ✗'}")
    logger.info(f"[CONFIG CHECK] TTS_PROVIDER    = {TTS.provider}")
    logger.info(f"[CONFIG CHECK] TTS_API_KEY     = {'SET ✓' if (TTS.elevenlabs_api_key or TTS.svara_api_key) else '(none required for svara ws-only)'}")
    logger.info(f"[CONFIG CHECK] AUTH_ENABLED    = {AUTH_ENABLED}")
    from auth import get_secret_fingerprint
    logger.info(f"[CONFIG CHECK] WEBUI_SECRET    = {'SET ✓ fingerprint=' + get_secret_fingerprint() if get_secret_fingerprint() else 'MISSING ✗ — JWT auth will fail!'}")
    logger.info(f"[CONFIG CHECK] PORT            = {PORT}")
    logger.info("[CONFIG CHECK] ─────────────────────────────────────")

    if not LLM.api_key:
        logger.error(
            "[CONFIG CHECK] FATAL: LLM_API_KEY is empty! "
            "On Railway, set LLM_API_KEY (or LLM_GROQ_API_KEY) in the Variables panel. "
            "The .env file is NOT loaded in Railway deployments."
        )

    # Initialize classroom database
    await room_manager.init_db()
    yield
    logger.info("Shutting down MiraVoiceAI Pipecat server...")
    from database import db as classroom_db
    await classroom_db.close()


app = FastAPI(
    title="MiraVoiceAI Pipecat Server",
    description="Voice AI bot with STT -> LLM -> TTS pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register classroom mode routes (additive — does not touch /ws)
app.include_router(classroom_router)


@app.get("/health")
async def health_check():
    """Health check endpoint.

    No JWT required — used by K8s probes and client pre-flight checks.
    """
    return {"status": "healthy", "service": "mira-voice-ai-pipecat"}


@app.get("/auth/health")
async def auth_health_check():
    """Auth diagnostics endpoint — no JWT required.

    Returns the secret key fingerprint (first 8 hex chars of SHA-256) so
    operators can compare it with the OpenWebUI pod's fingerprint to verify
    both services loaded the SAME WEBUI_SECRET_KEY.

    Usage:
        curl https://pipecat-host/auth/health
        curl https://openwebui-host/api/health  # compare fingerprints
    """
    from auth import AUTH_ENABLED, get_secret_fingerprint
    return {
        "auth_enabled": AUTH_ENABLED,
        "secret_fingerprint": get_secret_fingerprint(),
        "hint": "Compare this fingerprint with OpenWebUI's to verify JWT trust",
    }


@app.get("/metrics", dependencies=[Depends(_require_jwt)])
async def metrics():
    """Server-side performance metrics.

    Returns aggregated stats for LLM (TTFT, total latency, token counts),
    translation, TTS, error counts, active/total sessions, and recent
    session summaries.
    """
    return _metrics_collector.snapshot()


@app.post("/connect", dependencies=[Depends(_require_jwt)])
async def connect():
    """Return WebSocket URL for direct connection to backend."""
    public_host = os.getenv("PUBLIC_HOST", "hp-fury")

    # Check if SSL is enabled (same logic as run_server)
    ssl_keyfile = os.getenv("SSL_KEYFILE", "hp-fury+1-key.pem")
    ssl_certfile = os.getenv("SSL_CERTFILE", "hp-fury+1.pem")
    use_ssl = os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile)

    protocol = "wss" if use_ssl else "ws"
    return {"ws_url": f"{protocol}://{public_host}:{PORT}/ws"}


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    stream: Optional[bool] = True
    user_name: Optional[str] = None
    topic: Optional[str] = None

@app.post("/chat", dependencies=[Depends(_require_jwt)])
async def chat_completion(req: ChatRequest):
    """
    Text chat endpoint — proxies to the same LLM that Pipecat uses for voice.
    Supports streaming (SSE) so the frontend can show tokens as they arrive.
    """
    api_key = LLM_API_KEY
    base_url = LLM_BASE_URL
    model = req.model or LLM_MODEL

    # Guard: empty api_key produces httpx.LocalProtocolError ("Illegal header value b'Bearer '")
    # This happens on Railway when LLM_API_KEY is not set in the Variables panel.
    # The .env file is NOT loaded on Railway — all vars must be set in the dashboard.
    if not api_key:
        logger.error(
            "[CHAT] LLM_API_KEY is empty — cannot call LLM. "
            "On Railway: set LLM_API_KEY in the Variables panel."
        )
        raise HTTPException(
            status_code=500,
            detail="LLM API key not configured. Set LLM_API_KEY in Railway Variables."
        )

    # System prompt for Mira text tutor mode (composed from versioned files)
    prompt_mode = "text"
    topic_name = req.topic  # default: use topic as-is for STUDENT CONTEXT
    if req.topic:
        mm = get_maths_manager()
        _topic_obj = mm.get_topic(str(req.topic))
        if _topic_obj:
            prompt_mode = "math-word-problems"
            # Resolve human-readable name instead of raw index (e.g. "15")
            topic_name = _topic_obj.get("improved_topic_name") or _topic_obj.get("original_topic") or req.topic

    prompt_content = load_system_prompt(version=PROMPT_VERSION, mode=prompt_mode)

    # Inject dynamic student context if available
    context_lines = []
    if req.user_name:
        context_lines.append(f"Name: {req.user_name}")
    if topic_name:
        context_lines.append(f"Topic: {topic_name}")
    if context_lines:
        prompt_content += "\n\n--- STUDENT CONTEXT ---\n" + "\n".join(context_lines) + "\n"

    # Inject curriculum context
    # For math mode: inject Maths context directly — do NOT check Science first
    # (Science and Maths share the same numeric topic IDs 1-35, causing collisions)
    if req.topic:
        if prompt_mode == "math-word-problems":
            mm = get_maths_manager()
            maths_ctx = mm.get_context_for_topic(str(req.topic))
            if maths_ctx:
                prompt_content += "\n\n" + maths_ctx + "\n"
        else:
            # Non-math topic: try Science curriculum
            from curriculum_manager import get_curriculum_manager
            cm = get_curriculum_manager()
            curriculum_ctx = None
            if cm.available:
                curriculum_ctx = cm.get_context_for_topic(req.topic)
            if curriculum_ctx:
                prompt_content += "\n\n--- CURRICULUM CONTEXT (SCIENCE) ---\n" + curriculum_ctx + "\n"

    system_msg = {
        "role": "system",
        "content": prompt_content,
    }

    # Truncate history to last 20 messages to avoid LLM context limit (Groq ~32K tokens)
    # System prompt is always prepended; only the conversation turns are trimmed.
    MAX_HISTORY_MESSAGES = 20
    trimmed_messages = req.messages[-MAX_HISTORY_MESSAGES:] if len(req.messages) > MAX_HISTORY_MESSAGES else req.messages
    messages = [system_msg] + [{"role": m.role, "content": m.content} for m in trimmed_messages]

    if req.stream:
        async def generate():
            t0 = time.time()
            t_first_token = 0.0
            token_count = 0
            async with httpx.AsyncClient(timeout=60.0) as client:
                from provider_config import LLM as _llm_cfg
                _json_body = {
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.7,     # locked — prevents output variation from Groq default changes
                    "max_tokens": 1024,     # prevent runaway responses
                }
                if _llm_cfg.is_vllm:
                    _json_body["chat_template_kwargs"] = {"enable_thinking": False}
                async with client.stream(
                    "POST",
                    f"{base_url}/chat/completions",
                    json=_json_body,
                    headers={"Authorization": f"Bearer {api_key}"},
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            token_count += 1
                            if token_count == 1:
                                t_first_token = time.time()
                            yield line + "\n\n"
                        elif line == "":
                            continue
            # Record tutor text metrics
            total_ms = round((time.time() - t0) * 1000, 1)
            ttft_ms = round((t_first_token - t0) * 1000, 1) if t_first_token else 0.0
            tokens_per_sec = round(token_count / ((time.time() - t0) or 1), 1)
            _metrics_collector.record_tutor_text_query(total_ms, token_count, ttft_ms)
            _metrics_collector.record_trace({
                "mode": "tutor_text",
                "query": messages[-1]["content"][:80] if messages else "",
                "ts": t0,
                "total_ms": total_ms,
                "stages": [
                    {"name": "llm_ttft", "ms": ttft_ms},
                    {"name": "llm_stream", "ms": total_ms, "tokens": token_count,
                     "tok_per_sec": tokens_per_sec},
                ],
            })
            logger.info(
                f"[METRICS][TUTOR] chat_stream | total={total_ms}ms | "
                f"ttft={ttft_ms}ms | tokens={token_count}"
            )

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        t0 = time.time()
        async with httpx.AsyncClient(timeout=60.0) as client:
            from provider_config import LLM as _llm_cfg
            _json_body_ns = {
                "model": model,
                "messages": messages,
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 1024,
            }
            if _llm_cfg.is_vllm:
                _json_body_ns["chat_template_kwargs"] = {"enable_thinking": False}
            resp = await client.post(
                f"{base_url}/chat/completions",
                json=_json_body_ns,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            total_ms = round((time.time() - t0) * 1000, 1)
            _metrics_collector.record_tutor_text_query(total_ms, 0, 0.0)
            _metrics_collector.record_trace({
                "mode": "tutor_text",
                "query": messages[-1]["content"][:80] if messages else "",
                "ts": t0,
                "total_ms": total_ms,
                "stages": [
                    {"name": "llm_sync", "ms": total_ms},
                ],
            })
            logger.info(f"[METRICS][TUTOR] chat_sync | total={total_ms}ms")
            return resp.json()


@app.get("/config", dependencies=[Depends(_require_jwt)])
async def get_config():
    """Get current server configuration — all values come from env vars."""
    lang_hints = [h.strip() for h in STT_LANGUAGE_HINTS.split(",") if h.strip()]
    return {
        # STT
        "stt_provider": STT_PROVIDER,
        "stt_api_key_set": bool(SONIOX_API_KEY or DEEPGRAM_API_KEY),
        "supported_stt_providers": ["soniox", "deepgram", "whisper"],
        # TTS
        "tts_provider": TTS_PROVIDER,
        "tts_ws_url": TTS_WS_URL,
        "elevenlabs_api_key_set": bool(ELEVENLABS_API_KEY),
        "tts_voice_gender": TTS_VOICE_GENDER,
        "supported_tts_providers": ["elevenlabs", "svara", "openai"],
        # LLM
        "llm_provider": LLM_PROVIDER,
        "llm_base_url": LLM_BASE_URL,
        "llm_model": LLM_MODEL,
        "supported_llm_providers": ["openai"],
        # General
        "default_voice": DEFAULT_VOICE,
        "default_language": DEFAULT_LANGUAGE,
        "supported_languages": lang_hints,
        "supported_modes": ["text_and_audio", "text_only"],
    }


async def receive_client_config(websocket: WebSocket, timeout: float = 5.0) -> dict:
    """
    Wait for an optional config message from the client.

    Expected format:
    {
        "type": "config",
        "system_prompt": "...",  // optional
        "context": [...]        // optional
        "mode": "text_and_audio", // optional: "text_and_audio" (default) or "text_only"
        "enable_greeting": false, // optional: default false (greeting is opt-in)
        "room_id": "...",         // optional: classroom room to broadcast to
        "speaker_id": "...",      // optional: classroom speaker user_id
        "speaker_name": "...",    // optional: classroom speaker name
        "speaker_language": "en", // optional: classroom speaker language
        "language": "en"          // optional: tutor user registered language
    }

    Returns dict with system_prompt, context, mode, and optional classroom fields.
    """
    try:
        # Wait for a message with timeout
        message = await asyncio.wait_for(websocket.receive_text(), timeout=timeout)
        data = json.loads(message)

        # Check if it's a config message
        if data.get("type") == "config":
            logger.info("Received client config message")

            # Extract and validate system_prompt
            system_prompt = data.get("system_prompt")
            if system_prompt and not isinstance(system_prompt, str):
                logger.warning("Invalid system_prompt type, ignoring")
                system_prompt = None

            # Extract and validate context
            context = data.get("context")
            if context:
                if not isinstance(context, list):
                    logger.warning("Invalid context type, ignoring")
                    context = None
                else:
                    # Filter valid context entries
                    valid_context = []
                    for entry in context:
                        if (isinstance(entry, dict) and
                            entry.get("role") in ("user", "assistant") and
                            isinstance(entry.get("content"), str)):
                            valid_context.append(entry)
                        else:
                            logger.warning(f"Filtering out invalid context entry: {entry}")
                    context = valid_context if valid_context else None

            # Extract and validate mode
            mode = data.get("mode", "text_and_audio")
            if mode not in ("text_and_audio", "text_only"):
                logger.warning(f"Invalid mode '{mode}', defaulting to text_and_audio")
                mode = "text_and_audio"

            # Greeting behavior (opt-in): default is disabled to avoid replaying on reconnects
            enable_greeting = data.get("enable_greeting", False)
            if not isinstance(enable_greeting, bool):
                logger.warning("Invalid enable_greeting type, defaulting to false")
                enable_greeting = False

            # Optional classroom fields
            room_id = data.get("room_id")
            speaker_id = data.get("speaker_id")
            speaker_name = data.get("speaker_name")
            speaker_language = data.get("speaker_language")
            language = data.get("language")
            token = data.get("token")  # JWT for auth

            return {
                "system_prompt": system_prompt,
                "context": context,
                "mode": mode,
                "enable_greeting": enable_greeting,
                "room_id": room_id,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "speaker_language": speaker_language,
                "language": language,
                "token": token,
            }
        else:
            logger.info("First message was not a config message, using defaults")
            return {
                "system_prompt": None,
                "context": None,
                "mode": "text_and_audio",
                "enable_greeting": False,
                "room_id": None,
                "speaker_id": None,
                "speaker_name": None,
                "speaker_language": None,
                "language": None,
                "token": None,
            }

    except asyncio.TimeoutError:
        logger.info("No config message received within timeout, using defaults")
        return {
            "system_prompt": None,
            "context": None,
            "mode": "text_and_audio",
            "enable_greeting": False,
            "room_id": None,
            "speaker_id": None,
            "speaker_name": None,
            "speaker_language": None,
            "language": None,
            "token": None,
        }
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in config message: {e}, using defaults")
        return {
            "system_prompt": None,
            "context": None,
            "mode": "text_and_audio",
            "enable_greeting": False,
            "room_id": None,
            "speaker_id": None,
            "speaker_name": None,
            "speaker_language": None,
            "language": None,
            "token": None,
        }
    except Exception as e:
        logger.warning(f"Error receiving config: {e}, using defaults")
        return {
            "system_prompt": None,
            "context": None,
            "mode": "text_and_audio",
            "enable_greeting": False,
            "room_id": None,
            "speaker_id": None,
            "speaker_name": None,
            "speaker_language": None,
            "language": None,
            "token": None,
        }


def make_websocket_binary_safe(websocket: WebSocket) -> WebSocket:
    """
    Monkey-patch a FastAPI WebSocket so that receive_bytes() silently skips
    any text-only messages instead of crashing with KeyError('bytes').

    Pipecat's FastAPIWebsocketInputTransport uses iter_bytes() → receive_bytes()
    which internally calls self.receive() and then accesses message["bytes"].
    If a text message arrives (e.g. a late config JSON), the missing "bytes" key
    causes a KeyError that kills the transport.

    This patch overrides receive_bytes() to loop until a binary frame arrives,
    silently discarding any interleaved text frames.
    """
    _original_receive = websocket.receive

    async def _binary_safe_receive_bytes() -> bytes:
        while True:
            message = await _original_receive()
            msg_type = message.get("type", "")

            # Disconnect → raise so iter_bytes() terminates cleanly
            if msg_type == "websocket.disconnect":
                from starlette.websockets import WebSocketDisconnect
                raise WebSocketDisconnect(
                    code=message.get("code", 1000),
                    reason=message.get("reason"),
                )

            # Binary frame — return it
            if "bytes" in message and message["bytes"] is not None:
                return message["bytes"]

            # Text-only frame — log and skip
            text = message.get("text", "")
            logger.info(
                f"[BinarySafeWS] Skipping text message ({len(text)} chars): "
                f"{text[:120]}..."
            )

    websocket.receive_bytes = _binary_safe_receive_bytes  # type: ignore[assignment]
    return websocket


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for voice interaction."""
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    voice_session_start = time.time()
    ws_session_mode = "tutor"  # Will be updated to "classroom" if room_id is present

    # Generate a unique session ID for text injection support
    session_id = str(uuid.uuid4())

    # Wait for optional config message (5s timeout for high-latency connections like Tailscale Ingress)
    config = await receive_client_config(websocket, timeout=5.0)

    # ── JWT auth on WebSocket ──────────────────────────────────────
    if AUTH_ENABLED:
        ws_token = config.get("token")
        if not ws_token:
            await websocket.send_json({"type": "error", "message": "Authentication required"})
            await websocket.close(code=4401)
            return
        payload = decode_openwebui_jwt(ws_token)
        if not payload or "id" not in payload:
            await websocket.send_json({"type": "error", "message": "Invalid or expired token"})
            await websocket.close(code=4401)
            return
        logger.info(f"[AUTH] WS authenticated: user_id={payload['id']}")

    # Send session_id to the client so it can use /inject_text
    try:
        await websocket.send_json({"type": "session_id", "session_id": session_id})
        logger.info(f"Sent session_id to client: {session_id}")
    except Exception as e:
        logger.warning(f"Failed to send session_id: {e}")

    # Patch the WebSocket so Pipecat's binary transport ignores any stray text messages
    make_websocket_binary_safe(websocket)

    try:
        room_id = config.get("room_id")
        speaker_id = config.get("speaker_id")
        ws_session_mode = "classroom" if room_id else "tutor"
        _metrics_collector.session_start(mode=ws_session_mode)

        extra_processors = None
        classroom_system_prompt = config.get("system_prompt")
        skip_greeting = not bool(config.get("enable_greeting", False))
        context_messages = config.get("context")
        stt_language_hints = None

        # Tutor mode: enforce STT language allowlist from registered language.
        # Priority:
        #   1) config.language
        #   2) config.speaker_language (legacy clients)
        #   3) query param language
        # Unknown/missing language defaults to English-only.
        if not room_id:
            requested_lang = (
                config.get("language")
                or config.get("speaker_language")
                or websocket.query_params.get("language")
            )
            stt_language_hints = _stt_hints_for_registered_language(requested_lang)
            logger.info(
                f"[TUTOR] STT language hints enforced: {stt_language_hints} "
                f"(registered_lang={requested_lang or 'en(default)'})"
            )

        if room_id:
            room = room_manager.get_room(room_id)
            if not room:
                await websocket.send_json({"type": "error", "message": "Room not found"})
                await websocket.close()
                return
            if not speaker_id or room.speaker_id != speaker_id:
                await websocket.send_json({"type": "error", "message": "Speaker token required for classroom session"})
                await websocket.close()
                return
            if speaker_id not in room.users:
                await websocket.send_json({"type": "error", "message": "Speaker must join classroom room first"})
                await websocket.close()
                return
            extra_processors = [ClassroomBroadcaster(room=room, room_mgr=room_manager)]
            # In classroom mode, constrain Soniox by speaker registered language.
            speaker_lang = room.users[speaker_id].language
            stt_language_hints = _stt_hints_for_registered_language(speaker_lang)
            # Use co-teaching prompt for classroom voice sessions
            classroom_system_prompt = room_manager._get_co_teaching_prompt(room)
            logger.info(f"[CLASSROOM] Attached broadcaster for room {room_id} (speaker={speaker_id}) with co-teaching prompt")
            logger.info(f"[CLASSROOM] STT language hints override: {stt_language_hints}")

            # If room has conversation history, seed the pipeline with it and skip greeting
            if room.conversation_history:
                context_messages = list(room.conversation_history)  # Copy to avoid mutation
                skip_greeting = True
                logger.info(
                    f"[CLASSROOM] Seeding pipeline with {len(context_messages)} prior messages "
                    f"from room {room_id} (skipping greeting)"
                )

        await run_bot(
            websocket=websocket,
            system_prompt=classroom_system_prompt,
            context_messages=context_messages,
            mode=config.get("mode", "text_and_audio"),
            extra_processors=extra_processors,
            session_id=session_id,
            skip_greeting=skip_greeting,
            metrics_collector=_metrics_collector,
            is_classroom=bool(room_id),
            stt_language_hints=stt_language_hints,
        )
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        duration = round(time.time() - voice_session_start, 1)
        room_id = config.get("room_id") if config else None
        _metrics_collector.session_end({
            "type": ws_session_mode,
            "subtype": "voice",
            "session_id": session_id,
            "room_id": room_id,
            "duration_s": duration,
            "disconnected_at": time.time(),
        })
        logger.info(
            f"[METRICS] voice_session_end | mode={ws_session_mode} "
            f"| session={session_id[:8]}... "
            f"| room={room_id or 'tutor'} | duration={duration}s"
        )


class InjectTextRequest(BaseModel):
    session_id: str
    text: str


@app.post("/inject_text", dependencies=[Depends(_require_jwt)])
async def inject_text(req: InjectTextRequest):
    """
    Inject typed text into an active voice pipeline session.

    When a user types text while voice mode is active, the frontend sends
    the text here instead of the /chat endpoint. The text is injected into
    the running Pipecat pipeline as a TranscriptionFrame, so the LLM
    processes it and TTS speaks the response back.
    """
    injector = get_text_injector(req.session_id)
    if not injector:
        raise HTTPException(
            status_code=404,
            detail=f"No active voice session found for session_id: {req.session_id}"
        )

    await injector.inject_text(req.text)
    logger.info(f"[INJECT_TEXT] Text injected for session {req.session_id}: '{req.text[:80]}'")
    return {"status": "ok", "session_id": req.session_id}


@app.get("/voices", dependencies=[Depends(_require_jwt)])
async def list_voices():
    """List available TTS voices."""
    # Svara TTS voices (English, Hindi, Tamil only)
    svara_voices = [
        {"id": "en_male", "name": "English (Male)", "language": "English", "provider": "svara"},
        {"id": "en_female", "name": "English (Female)", "language": "English", "provider": "svara"},
        {"id": "hi_male", "name": "Hindi (Male)", "language": "Hindi", "provider": "svara"},
        {"id": "hi_female", "name": "Hindi (Female)", "language": "Hindi", "provider": "svara"},
        {"id": "ta_male", "name": "Tamil (Male)", "language": "Tamil", "provider": "svara"},
        {"id": "ta_female", "name": "Tamil (Female)", "language": "Tamil", "provider": "svara"},
    ]

    # ElevenLabs voices
    elevenlabs_voices = [
        {
            "id": VOICE_PRESETS["female"]["id"],
            "name": VOICE_PRESETS["female"]["name"],
            "language": "Multilingual",
            "provider": "elevenlabs",
            "gender": "female",
        },
        {
            "id": VOICE_PRESETS["male"]["id"],
            "name": VOICE_PRESETS["male"]["name"],
            "language": "Multilingual",
            "provider": "elevenlabs",
            "gender": "male",
        },
    ]

    return {
        "voices": svara_voices + elevenlabs_voices,
        "current_provider": TTS_PROVIDER,
        "current_voice_gender": TTS_VOICE_GENDER,
    }


@app.get("/languages", dependencies=[Depends(_require_jwt)])
async def list_languages():
    """List supported STT languages."""
    languages = [
        {"code": "auto", "name": "Auto-detect"},
        {"code": "en", "name": "English"},
        {"code": "hi", "name": "Hindi"},
        {"code": "ta", "name": "Tamil"},
    ]
    return {"languages": languages}


# ── Maths Curriculum Endpoints ─────────────────────────────────────

@app.get("/maths/grades", dependencies=[Depends(_require_jwt)])
async def list_maths_grades():
    """List available grades for Maths curriculum."""
    mm = get_maths_manager()
    return {"grades": mm.get_grades()}


@app.get("/maths/topics/{grade}", dependencies=[Depends(_require_jwt)])
async def list_maths_topics(grade: int):
    """List topics for a specific grade in Maths curriculum."""
    mm = get_maths_manager()
    topics = mm.get_topics_for_grade(grade)
    return {"topics": topics}


@app.get("/maths/context/{topic_id}", dependencies=[Depends(_require_jwt)])
async def get_maths_context(topic_id: str):
    """Get full context for a specific Maths topic."""
    mm = get_maths_manager()
    context = mm.get_context_for_topic(topic_id)
    if not context:
        raise HTTPException(status_code=404, detail="Topic not found")
    return {"context": context}


def run_server():
    """Run the server with proper signal handling."""
    # Handle shutdown signals - use os._exit for immediate termination
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, forcing shutdown...")
        os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Check for SSL certificates
    ssl_keyfile = os.getenv("SSL_KEYFILE", "hp-fury+1-key.pem")
    ssl_certfile = os.getenv("SSL_CERTFILE", "hp-fury+1.pem")
    use_ssl = os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile)

    config_kwargs = {
        "app": app,
        "host": HOST,
        "port": PORT,
        "access_log": True,
        "log_level": "info",
    }

    if use_ssl:
        config_kwargs["ssl_keyfile"] = ssl_keyfile
        config_kwargs["ssl_certfile"] = ssl_certfile
        logger.info(f"Starting server with HTTPS on port {PORT}")
    else:
        logger.info(f"Starting server with HTTP on port {PORT}")

    config = uvicorn.Config(**config_kwargs)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    run_server()
