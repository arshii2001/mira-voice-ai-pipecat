"""
Classroom Mode — multilingual real-time classroom with speaker token.

Allows multiple users to join a room. One user holds the "speaker token" and
interacts with Mira normally (full voice pipeline) over the standard `/ws`
endpoint, with `room_id` and `speaker_id` provided in the config message.
All other users are "listeners" who receive the speaker's question and
Mira's answer translated into their preferred language, streamed as TTS audio.

Architecture:
    - Room manager: creates/joins/leaves rooms, manages speaker token
    - Speaker: runs the normal Pipecat pipeline via `/ws` with room_id
    - Listeners: receive translated text + TTS audio via WebSocket
    - Translation: uses translator.py (same LLM, lightweight call)
    - Listener TTS: per-listener ElevenLabs TTS for translated audio

Endpoints (all registered under /classroom):
    POST   /classroom/rooms                     - Create a room
    GET    /classroom/rooms                     - List active rooms
    GET    /classroom/rooms/{room_id}           - Get room details
    DELETE /classroom/rooms/{room_id}           - Delete a room
    POST   /classroom/rooms/{room_id}/token     - Request / pass speaker token
    WS     /classroom/rooms/{room_id}/ws        - Join room (speaker or listener)

WebSocket protocol:
    Client → Server:
        {"type": "join", "user_id": "...", "language": "hi", "name": "Ravi"}
        {"type": "request_token"}           — request speaker token
        {"type": "pass_token", "to": "..."}  — pass token to another user (or omit "to" for round-robin)
        {"type": "release_token"}           — release token (no one speaking)

    Server → Client:
        {"type": "joined", "room": {...}, "you": {...}}
        {"type": "user_joined", "user": {...}}
        {"type": "user_left", "user_id": "..."}
        {"type": "token_changed", "speaker_id": "...", "speaker_name": "..."}
        {"type": "transcription", "user_id": "...", "text": "...", "language": "...", "translated_text": "..."}
        {"type": "bot_response", "text": "...", "language": "...", "translated_text": "..."}
        {"type": "bot_audio_start"}
        {"type": "bot_audio_end"}
        {"type": "error", "message": "..."}
        Binary audio frames                 — translated TTS audio for listeners
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from translator import Translator, LANG_NAMES

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────


@dataclass
class RoomUser:
    """A user connected to a classroom room."""
    user_id: str
    name: str
    language: str  # preferred language code: en, hi, ta, kn
    websocket: WebSocket
    mode: str = "text_and_audio"  # "text_and_audio" or "text_only"
    is_speaker: bool = False
    joined_at: float = field(default_factory=time.time)


@dataclass
class Room:
    """A classroom room with multiple users and a speaker token."""
    room_id: str
    name: str
    created_at: float = field(default_factory=time.time)
    users: Dict[str, RoomUser] = field(default_factory=dict)
    speaker_id: Optional[str] = None
    token_queue: List[str] = field(default_factory=list)  # user_ids waiting for token


    def to_dict(self) -> dict:
        """Serialize room state for API responses."""
        return {
            "room_id": self.room_id,
            "name": self.name,
            "created_at": self.created_at,
            "user_count": len(self.users),
            "users": [
                {
                    "user_id": u.user_id,
                    "name": u.name,
                    "language": u.language,
                    "mode": u.mode,
                    "is_speaker": u.is_speaker,
                }
                for u in self.users.values()
            ],
            "speaker_id": self.speaker_id,
            "speaker_name": self.users[self.speaker_id].name if self.speaker_id and self.speaker_id in self.users else None,
            "token_queue": self.token_queue,
        }


# ─────────────────────────────────────────────────────────────────────
# Room Manager (singleton)
# ─────────────────────────────────────────────────────────────────────


class RoomManager:
    """Manages classroom rooms, users, and speaker tokens."""

    def __init__(self):
        self._rooms: Dict[str, Room] = {}
        self._translator: Optional[Translator] = None
        self._tts = None  # Shared TTS service for listener audio
        self._init_translator()
        self._init_tts()

    def _init_translator(self):
        """Initialize the translator with the same LLM config as bot.py."""
        api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        if api_key:
            self._translator = Translator(
                api_key=api_key,
                base_url=base_url,
                model=model,
            )
            logger.info("Classroom translator initialized")
        else:
            logger.warning("No LLM API key found — classroom translation disabled")

    def _init_tts(self):
        """Initialize shared TTS service for listener audio synthesis."""
        try:
            from bot import create_tts_service
            self._tts = create_tts_service(sample_rate=24000)
            logger.info("Classroom TTS service initialized (provider-agnostic)")
        except Exception as e:
            logger.warning(f"Could not init classroom TTS: {e} — listeners will get text only")

    # ── Room CRUD ──

    def create_room(self, name: str = "Classroom") -> Room:
        """Create a new room."""
        room_id = str(uuid.uuid4())[:8]
        room = Room(room_id=room_id, name=name)
        self._rooms[room_id] = room
        logger.info(f"[CLASSROOM] Room created: {room_id} ({name})")
        return room

    def get_room(self, room_id: str) -> Optional[Room]:
        """Get a room by ID."""
        return self._rooms.get(room_id)

    def list_rooms(self) -> List[dict]:
        """List all active rooms."""
        return [room.to_dict() for room in self._rooms.values()]

    def delete_room(self, room_id: str) -> bool:
        """Delete a room."""
        if room_id in self._rooms:
            del self._rooms[room_id]
            logger.info(f"[CLASSROOM] Room deleted: {room_id}")
            return True
        return False

    # ── User management ──

    async def add_user(self, room_id: str, user: RoomUser) -> bool:
        """
        Add a user to a room.

        Returns True on success. The caller is responsible for sending the
        'joined' response BEFORE calling finalize_join() so that the user
        receives 'joined' before any 'token_changed' events.
        """
        room = self.get_room(room_id)
        if not room:
            return False

        room.users[user.user_id] = user
        logger.info(
            f"[CLASSROOM] User {user.name} ({user.user_id}) joined room {room_id} "
            f"[lang={user.language}]"
        )

        # Notify other users
        await self._broadcast_json(room, {
            "type": "user_joined",
            "user": {
                "user_id": user.user_id,
                "name": user.name,
                "language": user.language,
                "mode": user.mode,
                "is_speaker": user.is_speaker,
            },
        }, exclude=user.user_id)

        return True

    async def finalize_join(self, room_id: str, user_id: str):
        """
        Called after the 'joined' response is sent to the user.
        Handles auto-assignment of speaker token for the first user.
        """
        room = self.get_room(room_id)
        if not room:
            return

        # If first user, auto-assign speaker token
        if len(room.users) == 1 and room.speaker_id is None:
            await self._assign_token(room, user_id)

    async def remove_user(self, room_id: str, user_id: str):
        """Remove a user from a room."""
        room = self.get_room(room_id)
        if not room or user_id not in room.users:
            return

        user = room.users.pop(user_id)
        logger.info(f"[CLASSROOM] User {user.name} ({user_id}) left room {room_id}")

        # Remove from token queue
        if user_id in room.token_queue:
            room.token_queue.remove(user_id)

        # If speaker left, pass token
        if room.speaker_id == user_id:
            room.speaker_id = None
            # Auto-assign to next in queue or first remaining user
            if room.users:
                next_speaker = room.token_queue.pop(0) if room.token_queue else next(iter(room.users))
                await self._assign_token(room, next_speaker)

        # Notify others
        await self._broadcast_json(room, {
            "type": "user_left",
            "user_id": user_id,
        })

        # Clean up empty rooms
        if not room.users:
            self.delete_room(room_id)

    # ── Speaker token ──

    async def request_token(self, room_id: str, user_id: str) -> bool:
        """Request the speaker token."""
        room = self.get_room(room_id)
        if not room or user_id not in room.users:
            return False

        if room.speaker_id is None:
            # Token is free, assign directly
            await self._assign_token(room, user_id)
            return True
        elif room.speaker_id == user_id:
            # Already the speaker
            return True
        else:
            # Add to queue
            if user_id not in room.token_queue:
                room.token_queue.append(user_id)
                logger.info(f"[CLASSROOM] {user_id} queued for token in {room_id} (position {len(room.token_queue)})")
            return False

    async def pass_token(self, room_id: str, from_user_id: str, to_user_id: Optional[str] = None):
        """Pass speaker token to another user."""
        room = self.get_room(room_id)
        if not room or room.speaker_id != from_user_id:
            return

        # Clear current speaker
        if from_user_id in room.users:
            room.users[from_user_id].is_speaker = False
        room.speaker_id = None

        # Determine next speaker
        if to_user_id and to_user_id in room.users:
            next_speaker = to_user_id
            if to_user_id in room.token_queue:
                room.token_queue.remove(to_user_id)
        elif room.token_queue:
            next_speaker = room.token_queue.pop(0)
        else:
            # No one waiting — token is free
            await self._broadcast_json(room, {
                "type": "token_changed",
                "speaker_id": None,
                "speaker_name": None,
            })
            return

        await self._assign_token(room, next_speaker)

    async def release_token(self, room_id: str, user_id: str):
        """Release speaker token without passing to anyone specific."""
        await self.pass_token(room_id, user_id, to_user_id=None)

    async def _assign_token(self, room: Room, user_id: str):
        """Assign speaker token to a user."""
        # Clear previous speaker
        if room.speaker_id and room.speaker_id in room.users:
            room.users[room.speaker_id].is_speaker = False

        room.speaker_id = user_id
        if user_id in room.users:
            room.users[user_id].is_speaker = True

        logger.info(f"[CLASSROOM] Speaker token → {room.users[user_id].name} ({user_id}) in room {room.room_id}")

        await self._broadcast_json(room, {
            "type": "token_changed",
            "speaker_id": user_id,
            "speaker_name": room.users[user_id].name if user_id in room.users else None,
        })

    # ── Broadcasting ──

    async def broadcast_transcription(
        self,
        room: Room,
        speaker_id: str,
        text: str,
        language: str,
    ):
        """Broadcast a speaker's transcription to all listeners, translated."""
        if not self._translator:
            logger.warning("No translator available for classroom broadcast")
            return

        speaker = room.users.get(speaker_id)
        if not speaker:
            return

        t0 = time.time()
        listener_count = 0

        # Translate + send to each listener in their language (parallel)
        tasks = []
        for user in room.users.values():
            if user.user_id == speaker_id:
                continue
            listener_count += 1
            tasks.append(
                self._send_translated_event(
                    user=user,
                    event_type="transcription",
                    original_text=text,
                    source_lang=language,
                    extra_fields={"user_id": speaker_id, "speaker_name": speaker.name},
                )
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        fanout_ms = round((time.time() - t0) * 1000, 1)
        logger.info(
            f"[METRICS][CLASSROOM] broadcast_transcription | "
            f"room={room.room_id} | listeners={listener_count} | "
            f"fanout_latency={fanout_ms}ms | lang={language} | "
            f"text='{text[:50]}'"
        )

    async def broadcast_bot_response(
        self,
        room: Room,
        text: str,
        language: str,
    ):
        """Broadcast Mira's response to all listeners, translated."""
        if not self._translator:
            return

        t0 = time.time()
        listener_count = 0

        tasks = []
        for user in room.users.values():
            if user.user_id == room.speaker_id:
                continue
            listener_count += 1
            tasks.append(
                self._send_translated_event(
                    user=user,
                    event_type="bot_response",
                    original_text=text,
                    source_lang=language,
                )
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        fanout_ms = round((time.time() - t0) * 1000, 1)
        logger.info(
            f"[METRICS][CLASSROOM] broadcast_bot_response | "
            f"room={room.room_id} | listeners={listener_count} | "
            f"fanout_latency={fanout_ms}ms | lang={language} | "
            f"text='{text[:50]}'"
        )

    async def broadcast_bot_audio(
        self,
        room: Room,
        audio_data: bytes,
        source_lang: str,
    ):
        """
        Broadcast Mira's TTS audio to listeners.

        For listeners who speak the same language as the speaker, forward audio directly.
        For others, translate + re-synthesize (handled by broadcast_bot_response text path).

        NOTE: For the MVP, we use the text path (translate + TTS per listener).
        Direct audio forwarding for same-language listeners is a future optimization.
        """
        pass  # Text-based translation is used instead (see broadcast_bot_response)

    async def _send_translated_event(
        self,
        user: RoomUser,
        event_type: str,
        original_text: str,
        source_lang: str,
        extra_fields: dict = None,
    ):
        """
        Translate text, add speaker attribution, send JSON event,
        and synthesize + stream TTS audio to a listener.

        Flow:
          1. Translate original_text to listener's language
          2. Prepend speaker attribution ("Ravi asks:" / "Mira says:")
          3. Send JSON event with text
          4. Send bot_audio_start JSON
          5. Synthesize speech via run_tts() (provider-agnostic)
          6. Stream audio chunks as binary WebSocket frames
          7. Send bot_audio_end JSON
        """
        t0 = time.time()
        translate_ms = 0.0
        tts_ms = 0.0
        audio_bytes_sent = 0

        try:
            # ── Step 1: Translate ──
            if user.language != source_lang and self._translator:
                translate_t0 = time.time()
                translated = await self._translator.translate(
                    text=original_text,
                    target_lang=user.language,
                    source_lang=source_lang,
                )
                translate_ms = round((time.time() - translate_t0) * 1000, 1)
            else:
                translated = original_text

            # ── Step 2: Speaker attribution ──
            speaker_name = (extra_fields or {}).get("speaker_name")
            if event_type == "transcription" and speaker_name:
                tts_text = f"{speaker_name} asks: {translated}"
            elif event_type == "bot_response":
                tts_text = f"Mira says: {translated}"
            else:
                tts_text = translated

            # ── Step 3: Send JSON event ──
            event = {
                "type": event_type,
                "text": original_text,
                "language": source_lang,
                "translated_text": translated,
                "tts_text": tts_text,
                "target_language": user.language,
            }
            if extra_fields:
                event.update(extra_fields)

            await self._send_json(user.websocket, event)

            # ── Step 4-7: Synthesize + stream audio (only if user wants audio) ──
            if self._tts and user.mode == "text_and_audio":
                await self._send_json(user.websocket, {"type": "bot_audio_start"})
                tts_t0 = time.time()

                try:
                    async for frame in self._tts.run_tts(tts_text):
                        # TTSAudioRawFrame has .audio (bytes)
                        if hasattr(frame, "audio") and frame.audio:
                            await self._send_bytes(user.websocket, frame.audio)
                            audio_bytes_sent += len(frame.audio)
                except Exception as tts_err:
                    logger.error(f"[METRICS][CLASSROOM] tts_error | user={user.name} | error={tts_err}")

                tts_ms = round((time.time() - tts_t0) * 1000, 1)
                await self._send_json(user.websocket, {"type": "bot_audio_end"})

            total_ms = round((time.time() - t0) * 1000, 1)
            logger.info(
                f"[METRICS][CLASSROOM] deliver_{event_type} | "
                f"user={user.name}({user.language}) | "
                f"{source_lang}→{user.language} | "
                f"translate={translate_ms}ms | tts={tts_ms}ms | "
                f"audio={audio_bytes_sent}B | total={total_ms}ms"
            )

        except Exception as e:
            total_ms = round((time.time() - t0) * 1000, 1)
            logger.error(
                f"[METRICS][CLASSROOM] deliver_{event_type} FAILED | "
                f"user={user.name}({user.language}) | total={total_ms}ms | error={e}"
            )

    async def _broadcast_json(self, room: Room, data: dict, exclude: str = None):
        """Send JSON to all users in a room."""
        tasks = []
        for user in room.users.values():
            if user.user_id != exclude:
                tasks.append(self._send_json(user.websocket, data))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_json(self, ws: WebSocket, data: dict):
        """Send JSON to a WebSocket, handling closed connections."""
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_json(data)
        except Exception as e:
            logger.debug(f"Failed to send JSON to WebSocket: {e}")

    async def _send_bytes(self, ws: WebSocket, data: bytes):
        """Send binary audio data to a WebSocket, handling closed connections."""
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_bytes(data)
        except Exception as e:
            logger.debug(f"Failed to send bytes to WebSocket: {e}")


# ─────────────────────────────────────────────────────────────────────
# Singleton room manager
# ─────────────────────────────────────────────────────────────────────

room_manager = RoomManager()


# ─────────────────────────────────────────────────────────────────────
# Pipeline event tap — captures STT + LLM events from speaker's pipeline
# ─────────────────────────────────────────────────────────────────────

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ClassroomBroadcaster(FrameProcessor):
    """
    Pipeline processor that taps into the speaker's pipeline and broadcasts
    transcription + bot response events to all listeners in the room.

    Inserted into the pipeline alongside PipelineInstrumentor.
    Does NOT modify any frames — purely observational.

    Metrics logged with [METRICS][CLASSROOM] prefix:
      - STT→broadcast latency (time from transcription frame to broadcast start)
      - LLM→broadcast latency (time from LLM end to broadcast start)
      - LLM accumulation time (first token to last token)
    """

    def __init__(self, room: Room, room_mgr: RoomManager, name: str = "ClassroomBroadcaster", **kwargs):
        super().__init__(name=name, **kwargs)
        self._room = room
        self._room_mgr = room_mgr
        self._llm_buffer = ""
        self._last_user_text = ""
        self._last_user_lang = "en"

        # Timing anchors
        self._stt_received_at: float = 0.0
        self._llm_first_token_at: float = 0.0
        self._llm_end_at: float = 0.0
        self._turn_count: int = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Tap STT final transcription — broadcast speaker's question
        if isinstance(frame, TranscriptionFrame):
            self._stt_received_at = time.time()
            self._last_user_text = frame.text
            self._last_user_lang = getattr(frame, "language", "en") or "en"
            self._turn_count += 1

            logger.info(
                f"[METRICS][CLASSROOM] stt_received | turn={self._turn_count} | "
                f"lang={self._last_user_lang} | text='{frame.text[:50]}'"
            )

            # Broadcast in background so we don't slow down the pipeline
            asyncio.create_task(
                self._room_mgr.broadcast_transcription(
                    room=self._room,
                    speaker_id=self._room.speaker_id or "",
                    text=frame.text,
                    language=self._last_user_lang,
                )
            )

        # Accumulate LLM response text
        elif isinstance(frame, TextFrame):
            if not self._llm_buffer:
                self._llm_first_token_at = time.time()
            self._llm_buffer += frame.text

        # LLM response complete — broadcast bot answer
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._llm_end_at = time.time()

            if self._llm_buffer:
                response_text = self._llm_buffer
                self._llm_buffer = ""

                # Log LLM accumulation time
                llm_accum_ms = round((self._llm_end_at - self._llm_first_token_at) * 1000, 1) if self._llm_first_token_at else 0
                stt_to_llm_end_ms = round((self._llm_end_at - self._stt_received_at) * 1000, 1) if self._stt_received_at else 0

                logger.info(
                    f"[METRICS][CLASSROOM] llm_complete | turn={self._turn_count} | "
                    f"llm_accum={llm_accum_ms}ms | stt_to_llm_end={stt_to_llm_end_ms}ms | "
                    f"response_len={len(response_text)} chars"
                )

                asyncio.create_task(
                    self._room_mgr.broadcast_bot_response(
                        room=self._room,
                        text=response_text,
                        language=self._last_user_lang,
                    )
                )

        # Forward all frames unchanged
        await self.push_frame(frame, direction)


# ─────────────────────────────────────────────────────────────────────
# FastAPI Router
# ─────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/classroom", tags=["classroom"])


@router.post("/rooms")
async def create_room(name: str = "Classroom"):
    """Create a new classroom room."""
    room = room_manager.create_room(name=name)
    return room.to_dict()


@router.get("/rooms")
async def list_rooms():
    """List all active classroom rooms."""
    return {"rooms": room_manager.list_rooms()}


@router.get("/rooms/{room_id}")
async def get_room(room_id: str):
    """Get room details."""
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room.to_dict()


@router.delete("/rooms/{room_id}")
async def delete_room(room_id: str):
    """Delete a room."""
    if room_manager.delete_room(room_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Room not found")


@router.post("/rooms/{room_id}/token")
async def manage_token(room_id: str, action: str = "request", user_id: str = "", to_user_id: str = ""):
    """
    Manage speaker token.

    Actions:
        request — request the token (queues if unavailable)
        pass    — pass token to to_user_id (or next in queue)
        release — release token
    """
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    if action == "request":
        granted = await room_manager.request_token(room_id, user_id)
        return {"granted": granted, "speaker_id": room.speaker_id}
    elif action == "pass":
        await room_manager.pass_token(room_id, user_id, to_user_id or None)
        return {"speaker_id": room.speaker_id}
    elif action == "release":
        await room_manager.release_token(room_id, user_id)
        return {"speaker_id": room.speaker_id}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")


@router.websocket("/rooms/{room_id}/ws")
async def classroom_websocket(websocket: WebSocket, room_id: str):
    """
    WebSocket endpoint for classroom participation.

    Flow:
    1. Client connects and sends {"type": "join", "user_id": "...", "language": "hi", "name": "Ravi"}
    2. Server responds with room state
    3. Speaker connects to /ws separately with room_id + speaker_id
    4. Listener receives translated text events + TTS audio
    5. User can request/pass/release token via JSON messages
    """
    await websocket.accept()
    logger.info(f"[CLASSROOM] WebSocket connected for room {room_id}")

    room = room_manager.get_room(room_id)
    if not room:
        await websocket.send_json({"type": "error", "message": "Room not found"})
        await websocket.close()
        return

    # Wait for join message
    user = None
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        data = json.loads(raw)

        if data.get("type") != "join":
            await websocket.send_json({"type": "error", "message": "Expected join message"})
            await websocket.close()
            return

        user_id = data.get("user_id", str(uuid.uuid4())[:8])
        language = data.get("language", "en")
        name = data.get("name", f"User-{user_id[:4]}")
        mode = data.get("mode", "text_and_audio")  # "text_and_audio" or "text_only"
        if mode not in ("text_and_audio", "text_only"):
            mode = "text_and_audio"

        user = RoomUser(
            user_id=user_id,
            name=name,
            language=language,
            websocket=websocket,
            mode=mode,
        )

        # Add to room (does NOT auto-assign token yet)
        if not await room_manager.add_user(room_id, user):
            await websocket.send_json({"type": "error", "message": "Failed to join room"})
            await websocket.close()
            return

        # Send room state to new user FIRST
        await websocket.send_json({
            "type": "joined",
            "room": room.to_dict(),
            "you": {
                "user_id": user_id,
                "name": name,
                "language": language,
                "mode": mode,
                "is_speaker": user.is_speaker,
            },
        })

        # NOW auto-assign speaker token if first user (after 'joined' is sent)
        await room_manager.finalize_join(room_id, user_id)

    except asyncio.TimeoutError:
        await websocket.send_json({"type": "error", "message": "Join timeout"})
        await websocket.close()
        return
    except Exception as e:
        logger.error(f"[CLASSROOM] Join error: {e}")
        await websocket.close()
        return

    # Main message loop
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "request_token":
                granted = await room_manager.request_token(room_id, user.user_id)
                await websocket.send_json({"type": "token_response", "granted": granted})

            elif msg_type == "pass_token":
                to = data.get("to")
                await room_manager.pass_token(room_id, user.user_id, to)

            elif msg_type == "release_token":
                await room_manager.release_token(room_id, user.user_id)

            elif msg_type == "set_mode":
                # Switch between text_only and text_and_audio at runtime
                new_mode = data.get("mode", "text_and_audio")
                if new_mode in ("text_and_audio", "text_only"):
                    user.mode = new_mode
                    await websocket.send_json({"type": "mode_changed", "mode": new_mode})
                    logger.info(f"[CLASSROOM] {user.name} switched to mode={new_mode}")
                else:
                    await websocket.send_json({"type": "error", "message": f"Invalid mode: {new_mode}"})

            else:
                logger.debug(f"[CLASSROOM] Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        logger.info(f"[CLASSROOM] User {user.user_id} disconnected from room {room_id}")
    except Exception as e:
        logger.error(f"[CLASSROOM] WebSocket error for {user.user_id}: {e}")
    finally:
        if user:
            await room_manager.remove_user(room_id, user.user_id)


