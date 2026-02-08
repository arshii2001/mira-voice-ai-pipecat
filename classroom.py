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

import openai
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from translator import Translator, LANG_NAMES
from database import db as classroom_db

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
    hand_raises: List[dict] = field(default_factory=list)  # live hand-raise queue
    active_session_id: Optional[str] = None  # DB session ID for persistence
    topic: Optional[str] = None  # Current topic (for topic suggestions)
    teacher_id: Optional[str] = None  # Room creator = teacher
    current_lesson_topic: Optional[str] = None  # Current lesson topic set by teacher
    conversation_history: List[dict] = field(default_factory=list)  # [{role, content}] for LLM context


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
                    "is_teacher": u.user_id == self.teacher_id,
                }
                for u in self.users.values()
            ],
            "speaker_id": self.speaker_id,
            "speaker_name": self.users[self.speaker_id].name if self.speaker_id and self.speaker_id in self.users else None,
            "token_queue": self.token_queue,
            "hand_raises": self.hand_raises,
            "active_session_id": self.active_session_id,
            "topic": self.topic,
            "teacher_id": self.teacher_id,
            "current_lesson_topic": self.current_lesson_topic,
        }


# ─────────────────────────────────────────────────────────────────────
# Room Manager (singleton)
# ─────────────────────────────────────────────────────────────────────


DEFAULT_ROOM_ID = os.getenv("CLASSROOM_DEFAULT_ROOM_ID", "default")
DEFAULT_ROOM_NAME = os.getenv("CLASSROOM_DEFAULT_ROOM_NAME", "Mira Classroom")


class RoomManager:
    """Manages classroom rooms, users, and speaker tokens."""

    def __init__(self):
        self._rooms: Dict[str, Room] = {}
        self._permanent_rooms: set = set()  # room IDs that survive empty state
        self._translator: Optional[Translator] = None
        self._tts = None  # Shared TTS service for listener audio
        self._llm_client: Optional[openai.AsyncOpenAI] = None
        self._llm_model: str = "gpt-4o-mini"
        self._init_translator()
        self._init_tts()
        self._init_llm()
        self._create_default_room()

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
        """Initialize standalone TTS service for listener audio synthesis."""
        try:
            from services.classroom_tts import create_classroom_tts
            self._tts = create_classroom_tts(sample_rate=24000)
            if self._tts:
                logger.info("Classroom TTS initialized (ElevenLabs REST API)")
            else:
                logger.warning("Classroom TTS not available — listeners will get text only")
        except Exception as e:
            logger.warning(f"Could not init classroom TTS: {e} — listeners will get text only")

    def _init_llm(self):
        """Initialize LLM client for text-mode queries."""
        api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self._llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        if api_key:
            self._llm_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"Classroom LLM client initialized (model={self._llm_model})")
        else:
            logger.warning("No LLM API key — classroom text mode disabled")

    def _get_co_teaching_prompt(self, room: Optional["Room"] = None) -> str:
        """Return the co-teaching system prompt for AI-assisted teaching mode."""
        topic_line = ""
        if room and room.current_lesson_topic:
            topic_line = f"\nCurrent lesson topic: {room.current_lesson_topic}\n"

        return (
            "You are Mira, an AI co-teacher in a live classroom. "
            "A human teacher is guiding the lesson. Students are listening.\n\n"
            "## YOUR ROLE\n"
            "- You are the teacher's teaching assistant. Support them.\n"
            "- Give structured explanations — not just answers. Use step-by-step breakdowns.\n"
            "- Use analogies and real-world examples to make concepts stick.\n"
            "- When explaining, break complex topics into digestible steps.\n"
            "- Occasionally check understanding: 'Does that make sense?' or "
            "'Let me check — can someone tell me...'\n"
            "- Maintain full lesson context — remember what was covered.\n\n"
            f"{topic_line}"
            "## TEACHER COMMANDS (respond appropriately)\n"
            "- [TEACHER_ACTION: SET_TOPIC <topic>] — Introduce this topic with a structured overview. "
            "Give a clear 3-4 sentence introduction, mention what students will learn.\n"
            "- [TEACHER_ACTION: QUIZ] — Generate exactly 3 quick-check questions about what was just discussed. "
            "Format: number each question, give 4 options (A-D), mark the correct answer.\n"
            "- [TEACHER_ACTION: SUMMARIZE] — Produce a checkpoint summary of the lesson so far. "
            "List the key points covered, what students should remember.\n"
            "- [TEACHER_ACTION: SIMPLIFY] — Re-explain the last point more simply. "
            "Use a different analogy, simpler words, or a concrete example.\n"
            "- [TEACHER_ACTION: NEXT] — Move to the next logical subtopic. "
            "Bridge from what was just covered to the next concept naturally.\n\n"
            "## RESPONSE RULES\n"
            "- Do NOT greet or introduce yourself — just respond to what's asked.\n"
            "- Keep responses concise but educational (aim for 2-5 sentences normally, "
            "longer for topic introductions and quizzes).\n"
            "- Use simple language suitable for students.\n"
            "- When a student asks a question, guide them to understanding rather than "
            "just giving the answer.\n"
        )

    async def ask_llm(self, question: str, speaker_ws: WebSocket, room: Optional["Room"] = None) -> Optional[str]:
        """Send a text question to the LLM, stream tokens to speaker, return full response.
        Maintains conversation history per room for context continuity."""
        if not self._llm_client:
            try:
                await speaker_ws.send_json({"type": "error", "message": "LLM not configured"})
            except Exception:
                pass
            return None

        # Use co-teaching prompt
        system_prompt = self._get_co_teaching_prompt(room)

        t0 = time.time()
        full_response = ""

        # Build messages with conversation history for context
        messages = [{"role": "system", "content": system_prompt}]
        if room and room.conversation_history:
            # Include last 20 messages for context
            messages.extend(room.conversation_history[-20:])
        messages.append({"role": "user", "content": question})

        try:
            stream = await self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=messages,
                max_tokens=800,
                temperature=0.7,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    token = delta.content
                    full_response += token
                    # Stream each token to the speaker
                    try:
                        await speaker_ws.send_json({
                            "type": "bot_text",
                            "text": token,
                            "streaming": True,
                        })
                    except Exception:
                        break

            # Send complete response
            try:
                await speaker_ws.send_json({
                    "type": "bot_text_complete",
                    "text": full_response,
                })
            except Exception:
                pass

            # Update conversation history for context continuity
            if room:
                room.conversation_history.append({"role": "user", "content": question})
                room.conversation_history.append({"role": "assistant", "content": full_response})
                # Keep history manageable
                if len(room.conversation_history) > 40:
                    room.conversation_history = room.conversation_history[-30:]

            latency_ms = round((time.time() - t0) * 1000, 1)
            logger.info(
                f"[CLASSROOM] LLM text query | latency={latency_ms}ms | "
                f"q='{question[:50]}' | a='{full_response[:50]}'"
            )
            return full_response

        except Exception as e:
            logger.error(f"[CLASSROOM] LLM query failed: {e}")
            try:
                await speaker_ws.send_json({"type": "error", "message": f"LLM error: {e}"})
            except Exception:
                pass
            return None

    async def init_db(self):
        """Initialize the database (called from server lifespan)."""
        await classroom_db.init()
        logger.info("[CLASSROOM] Database initialized")

    def _create_default_room(self):
        """Create a permanent default room that persists even when empty."""
        room = Room(room_id=DEFAULT_ROOM_ID, name=DEFAULT_ROOM_NAME)
        self._rooms[DEFAULT_ROOM_ID] = room
        self._permanent_rooms.add(DEFAULT_ROOM_ID)
        logger.info(f"[CLASSROOM] Default permanent room created: {DEFAULT_ROOM_ID} ({DEFAULT_ROOM_NAME})")

    async def _ensure_session(self, room: Room):
        """Ensure the room has an active DB session. Create one if not."""
        if room.active_session_id:
            return room.active_session_id
        try:
            session = await classroom_db.create_session(room.room_id, room.name)
            room.active_session_id = session.id
            logger.info(f"[CLASSROOM] DB session started: {session.id} for room {room.room_id}")
            return session.id
        except Exception as e:
            logger.error(f"[CLASSROOM] Failed to create DB session: {e}")
            return None

    async def _end_session(self, room: Room):
        """End the DB session for a room."""
        if room.active_session_id:
            try:
                await classroom_db.end_session(room.active_session_id)
                logger.info(f"[CLASSROOM] DB session ended: {room.active_session_id}")
            except Exception as e:
                logger.error(f"[CLASSROOM] Failed to end DB session: {e}")
            room.active_session_id = None

    async def save_message_to_db(
        self, room: Room, role: str, content: str,
        speaker_id: str = None, speaker_name: str = None,
        original_language: str = "en", translations: dict = None,
    ) -> Optional[str]:
        """Save a message to the database. Returns message ID."""
        session_id = await self._ensure_session(room)
        if not session_id:
            return None
        try:
            msg = await classroom_db.save_message(
                session_id=session_id,
                room_id=room.room_id,
                role=role,
                content=content,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                original_language=original_language,
                translations=translations,
            )
            return msg.id
        except Exception as e:
            logger.error(f"[CLASSROOM] Failed to save message: {e}")
            return None

    # ── Hand Raises ──

    async def raise_hand(self, room: Room, user_id: str, question_preview: str = None) -> Optional[dict]:
        """User raises their hand to ask a question."""
        user = room.users.get(user_id)
        if not user:
            return None

        # Check if already raised
        for hr in room.hand_raises:
            if hr["user_id"] == user_id and hr["status"] == "pending":
                return None  # Already raised

        raise_id = str(uuid.uuid4())[:8]
        hr_entry = {
            "id": raise_id,
            "user_id": user_id,
            "user_name": user.name,
            "question_preview": question_preview,
            "status": "pending",
            "raised_at": time.time(),
        }
        room.hand_raises.append(hr_entry)

        # Persist to DB
        session_id = await self._ensure_session(room)
        if session_id:
            try:
                await classroom_db.create_hand_raise(
                    session_id=session_id,
                    room_id=room.room_id,
                    user_id=user_id,
                    user_name=user.name,
                    question_preview=question_preview,
                )
            except Exception as e:
                logger.error(f"[CLASSROOM] Failed to save hand raise to DB: {e}")

        logger.info(f"[CLASSROOM] Hand raised: {user.name} in room {room.room_id}")
        return hr_entry

    async def lower_hand(self, room: Room, user_id: str):
        """User lowers their hand."""
        room.hand_raises = [hr for hr in room.hand_raises
                            if not (hr["user_id"] == user_id and hr["status"] == "pending")]
        logger.info(f"[CLASSROOM] Hand lowered: {user_id} in room {room.room_id}")

    async def acknowledge_hand(self, room: Room, raise_id: str, acknowledged_by: str) -> Optional[dict]:
        """Speaker/teacher acknowledges a hand raise — passes token to that user."""
        for hr in room.hand_raises:
            if hr["id"] == raise_id and hr["status"] == "pending":
                hr["status"] = "acknowledged"
                hr["resolved_at"] = time.time()
                # Pass token to the hand-raiser
                await self.pass_token(room.room_id, acknowledged_by, hr["user_id"])
                logger.info(f"[CLASSROOM] Hand acknowledged: {hr['user_name']} → gets token")
                return hr
        return None

    async def dismiss_hand(self, room: Room, raise_id: str):
        """Dismiss a hand raise without passing token."""
        for hr in room.hand_raises:
            if hr["id"] == raise_id and hr["status"] == "pending":
                hr["status"] = "dismissed"
                hr["resolved_at"] = time.time()
                return hr
        return None

    # ── Reactions ──

    async def add_reaction(self, room: Room, message_id: str, user_id: str, emoji: str) -> bool:
        """Add a reaction to a message."""
        session_id = await self._ensure_session(room)
        if not session_id:
            return False
        try:
            return await classroom_db.add_reaction(
                message_id=message_id,
                session_id=session_id,
                user_id=user_id,
                emoji=emoji,
            )
        except Exception as e:
            logger.error(f"[CLASSROOM] Failed to add reaction: {e}")
            return False

    async def remove_reaction(self, room: Room, message_id: str, user_id: str, emoji: str) -> bool:
        """Remove a reaction from a message."""
        try:
            return await classroom_db.remove_reaction(
                message_id=message_id,
                user_id=user_id,
                emoji=emoji,
            )
        except Exception as e:
            logger.error(f"[CLASSROOM] Failed to remove reaction: {e}")
            return False

    # ── Topic Suggestions ──

    async def suggest_topics(self, room: Room) -> List[str]:
        """Use LLM to suggest follow-up topics based on conversation history."""
        if not self._llm_client:
            return []

        session_id = room.active_session_id
        if not session_id:
            return [
                "Introduction to the subject",
                "Ask a question about today's topic",
                "Review previous material",
            ]

        try:
            messages = await classroom_db.get_messages(session_id, limit=20)
            if not messages:
                return [
                    "Start with a question about your subject",
                    "Ask Mira to explain a concept",
                    "Request a practice problem",
                ]

            # Build conversation context
            convo = "\n".join([
                f"{'Student' if m.role == 'user' else 'Mira'}: {m.content[:100]}"
                for m in messages[-10:]
            ])

            response = await self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content":
                     "Based on this classroom conversation, suggest exactly 3 short follow-up "
                     "questions or topics the student could explore next. "
                     "Return them as a JSON array of strings. No explanation."},
                    {"role": "user", "content": convo},
                ],
                max_tokens=200,
                temperature=0.7,
            )

            raw = response.choices[0].message.content.strip()
            topics = json.loads(raw)
            return topics[:3] if isinstance(topics, list) else []
        except Exception as e:
            logger.error(f"[CLASSROOM] Topic suggestion failed: {e}")
            return []

    # ── Session Summary + Quiz ──

    async def generate_session_summary(self, session_id: str) -> Optional[dict]:
        """Generate an AI summary and quiz for a completed session."""
        if not self._llm_client:
            return None

        try:
            messages = await classroom_db.get_messages(session_id, limit=100)
            if len(messages) < 3:
                return None

            convo = "\n".join([
                f"{'Student' if m.role == 'user' else 'Mira'} ({m.speaker_name or ''}): {m.content}"
                for m in messages
            ])

            response = await self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content":
                     "You are a teacher reviewing a classroom session. "
                     "Generate a JSON object with exactly these fields:\n"
                     '1. "summary": A 2-3 sentence summary of the key topics discussed.\n'
                     '2. "key_concepts": An array of 3-5 key concepts covered.\n'
                     '3. "quiz": An array of 3 quiz questions, each with:\n'
                     '   - "question": The question text\n'
                     '   - "options": Array of 4 answer choices\n'
                     '   - "correct": Index (0-3) of the correct answer\n'
                     '   - "explanation": Brief explanation of the answer\n'
                     "Return ONLY the JSON object, no markdown or explanation."},
                    {"role": "user", "content": f"Here is the classroom conversation:\n\n{convo}"},
                ],
                max_tokens=800,
                temperature=0.5,
            )

            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)

            # Save to DB
            await classroom_db.set_session_summary(
                session_id=session_id,
                summary=result.get("summary", ""),
                quiz_json=json.dumps(result.get("quiz", [])),
            )

            logger.info(f"[CLASSROOM] Session summary generated for {session_id}")
            return result
        except Exception as e:
            logger.error(f"[CLASSROOM] Summary generation failed: {e}")
            return None

    # ── Room CRUD ──

    def create_room(self, name: str = "Classroom", teacher_id: Optional[str] = None) -> Room:
        """Create a new room. The creator becomes the teacher."""
        room_id = str(uuid.uuid4())[:8]
        room = Room(room_id=room_id, name=name, teacher_id=teacher_id)
        self._rooms[room_id] = room
        logger.info(f"[CLASSROOM] Room created: {room_id} ({name}) teacher={teacher_id}")
        return room

    def get_room(self, room_id: str) -> Optional[Room]:
        """Get a room by ID."""
        return self._rooms.get(room_id)

    def list_rooms(self) -> List[dict]:
        """List all active rooms."""
        return [room.to_dict() for room in self._rooms.values()]

    def delete_room(self, room_id: str) -> bool:
        """Delete a room. Permanent rooms cannot be deleted."""
        if room_id in self._permanent_rooms:
            logger.info(f"[CLASSROOM] Skipping delete of permanent room: {room_id}")
            return False
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

        # Ensure DB session exists
        asyncio.create_task(self._ensure_session(room))

        # Update participant count in DB
        async def _update_stats():
            if room.active_session_id:
                try:
                    await classroom_db.update_session_stats(
                        room.active_session_id,
                        participant_count=len(room.users),
                    )
                except Exception:
                    pass
        asyncio.create_task(_update_stats())

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
        Handles auto-assignment of speaker token for the first user or teacher.
        """
        room = self.get_room(room_id)
        if not room:
            return

        # If this is the teacher joining and no one is speaking, give them the token
        if user_id == room.teacher_id and room.speaker_id is None:
            await self._assign_token(room, user_id)
        # If first user, auto-assign speaker token
        elif len(room.users) == 1 and room.speaker_id is None:
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

        # Remove pending hand raises
        room.hand_raises = [hr for hr in room.hand_raises
                            if hr["user_id"] != user_id]

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

        # Clean up empty rooms — end DB session
        if not room.users:
            await self._end_session(room)
            self.delete_room(room_id)

    # ── Speaker token ──

    async def request_token(self, room_id: str, user_id: str) -> bool:
        """Request the speaker token. Teacher gets priority (can reclaim anytime)."""
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
        elif user_id == room.teacher_id:
            # Teacher always gets priority — reclaim token immediately
            logger.info(f"[CLASSROOM] Teacher {user_id} reclaiming token in {room_id}")
            await self._assign_token(room, user_id)
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

        # Persist message to DB
        asyncio.create_task(
            self.save_message_to_db(
                room=room, role="user", content=text,
                speaker_id=speaker_id, speaker_name=speaker.name,
                original_language=language,
            )
        )

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

        # Persist bot response to DB
        asyncio.create_task(
            self.save_message_to_db(
                room=room, role="assistant", content=text,
                speaker_name="Mira", original_language="en",
            )
        )

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
                        # AudioChunk has .audio (raw PCM bytes)
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
async def create_room(name: str = "Classroom", teacher_id: str = None):
    """Create a new classroom room. The creator becomes the teacher."""
    room = room_manager.create_room(name=name, teacher_id=teacher_id)
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


# ── Session History Endpoints ──

@router.get("/sessions")
async def list_sessions(room_id: str = None, limit: int = 50, offset: int = 0):
    """List past classroom sessions."""
    sessions = await classroom_db.list_sessions(room_id=room_id, limit=limit, offset=offset)
    return {"sessions": [s.to_dict() for s in sessions]}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific session with its messages."""
    session = await classroom_db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = await classroom_db.get_messages(session_id)
    return {
        "session": session.to_dict(),
        "messages": [m.to_dict() for m in messages],
    }


@router.get("/sessions/{session_id}/summary")
async def get_session_summary(session_id: str):
    """Get or generate session summary + quiz."""
    session = await classroom_db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # If summary already exists, return it
    if session.summary:
        return {
            "session_id": session_id,
            "summary": session.summary,
            "quiz": json.loads(session.quiz_json) if session.quiz_json else None,
        }

    # Generate summary
    result = await room_manager.generate_session_summary(session_id)
    if result:
        return {
            "session_id": session_id,
            "summary": result.get("summary"),
            "key_concepts": result.get("key_concepts"),
            "quiz": result.get("quiz"),
        }

    raise HTTPException(status_code=400, detail="Not enough messages to generate summary")


@router.get("/dashboard")
async def get_dashboard(room_id: str = None):
    """Teacher dashboard stats."""
    stats = await classroom_db.get_dashboard_stats(room_id=room_id)
    return stats


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

        # If no teacher assigned yet, the first user to join becomes teacher
        if room.teacher_id is None:
            room.teacher_id = user_id
            logger.info(f"[CLASSROOM] {name} ({user_id}) is now the teacher of room {room_id}")

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
                "is_teacher": user_id == room.teacher_id,
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

            elif msg_type == "text_message":
                # Speaker typed a text question — query LLM, stream to speaker, broadcast to listeners
                text = data.get("text", "").strip()
                room = room_manager.get_room(room_id)
                if text and room and room.speaker_id == user.user_id:
                    logger.info(f"[CLASSROOM] Text message from {user.name}: {text[:60]}")

                    # 1. Broadcast the speaker's question to listeners (translated)
                    await room_manager.broadcast_transcription(
                        room=room,
                        speaker_id=user.user_id,
                        text=text,
                        language=user.language,
                    )

                    # 2. Query LLM — streams tokens to speaker via bot_text / bot_text_complete
                    llm_response = await room_manager.ask_llm(text, websocket, room=room)

                    # 3. Broadcast Mira's response to listeners (translated)
                    if llm_response:
                        await room_manager.broadcast_bot_response(
                            room=room,
                            text=llm_response,
                            language=user.language,
                        )
                else:
                    await websocket.send_json({"type": "error", "message": "Only the speaker can send messages"})

            elif msg_type == "teacher_action":
                # Teacher toolbar action — converts to a special LLM prompt
                action = data.get("action", "").strip().upper()
                payload = data.get("payload", "").strip()
                room = room_manager.get_room(room_id)

                if not room:
                    continue
                if user.user_id != room.teacher_id:
                    await websocket.send_json({"type": "error", "message": "Only the teacher can use lesson controls"})
                    continue

                # Build the teacher action command
                if action == "SET_TOPIC":
                    teacher_cmd = f"[TEACHER_ACTION: SET_TOPIC {payload}]"
                    room.current_lesson_topic = payload
                    # Notify all users of the topic change
                    await room_manager._broadcast_json(room, {
                        "type": "lesson_topic_changed",
                        "topic": payload,
                    })
                elif action == "QUIZ":
                    teacher_cmd = "[TEACHER_ACTION: QUIZ]"
                elif action == "SUMMARIZE":
                    teacher_cmd = "[TEACHER_ACTION: SUMMARIZE]"
                elif action == "SIMPLIFY":
                    teacher_cmd = "[TEACHER_ACTION: SIMPLIFY]"
                elif action == "NEXT":
                    teacher_cmd = "[TEACHER_ACTION: NEXT]"
                else:
                    await websocket.send_json({"type": "error", "message": f"Unknown teacher action: {action}"})
                    continue

                logger.info(f"[CLASSROOM] Teacher action: {action} (payload='{payload[:40] if payload else ''}')")

                # Don't broadcast the raw command to listeners — just send a label
                display_text = {
                    "SET_TOPIC": f"📝 Topic: {payload}",
                    "QUIZ": "❓ Quiz time!",
                    "SUMMARIZE": "📋 Let's summarize what we've covered",
                    "SIMPLIFY": "🔄 Let me simplify that",
                    "NEXT": "⏭️ Moving to the next topic",
                }.get(action, action)

                # Broadcast a neutral message to listeners
                await room_manager.broadcast_transcription(
                    room=room,
                    speaker_id=user.user_id,
                    text=display_text,
                    language="en",
                )

                # Query LLM with the teacher command (don't show command to students)
                llm_response = await room_manager.ask_llm(teacher_cmd, websocket, room=room)

                if llm_response:
                    await room_manager.broadcast_bot_response(
                        room=room,
                        text=llm_response,
                        language=user.language,
                    )

            elif msg_type == "set_mode":
                # Switch between text_only and text_and_audio at runtime
                new_mode = data.get("mode", "text_and_audio")
                if new_mode in ("text_and_audio", "text_only"):
                    user.mode = new_mode
                    await websocket.send_json({"type": "mode_changed", "mode": new_mode})
                    logger.info(f"[CLASSROOM] {user.name} switched to mode={new_mode}")
                else:
                    await websocket.send_json({"type": "error", "message": f"Invalid mode: {new_mode}"})

            elif msg_type == "hand_raise":
                # User raises hand to ask a question
                question = data.get("question_preview", "")
                room = room_manager.get_room(room_id)
                if room:
                    hr = await room_manager.raise_hand(room, user.user_id, question)
                    if hr:
                        await room_manager._broadcast_json(room, {
                            "type": "hand_raised",
                            "raise": hr,
                        })

            elif msg_type == "hand_lower":
                # User lowers their hand
                room = room_manager.get_room(room_id)
                if room:
                    await room_manager.lower_hand(room, user.user_id)
                    await room_manager._broadcast_json(room, {
                        "type": "hand_lowered",
                        "user_id": user.user_id,
                    })

            elif msg_type == "hand_acknowledge":
                # Speaker acknowledges a hand raise — passes token
                raise_id = data.get("raise_id", "")
                room = room_manager.get_room(room_id)
                if room and room.speaker_id == user.user_id and raise_id:
                    hr = await room_manager.acknowledge_hand(room, raise_id, user.user_id)
                    if hr:
                        await room_manager._broadcast_json(room, {
                            "type": "hand_acknowledged",
                            "raise": hr,
                        })

            elif msg_type == "hand_dismiss":
                # Speaker dismisses a hand raise
                raise_id = data.get("raise_id", "")
                room = room_manager.get_room(room_id)
                if room and room.speaker_id == user.user_id and raise_id:
                    hr = await room_manager.dismiss_hand(room, raise_id)
                    if hr:
                        await room_manager._broadcast_json(room, {
                            "type": "hand_dismissed",
                            "raise": hr,
                        })

            elif msg_type == "reaction":
                # User reacts to a message
                msg_id = data.get("message_id", "")
                emoji = data.get("emoji", "")
                action = data.get("action", "add")  # "add" or "remove"
                room = room_manager.get_room(room_id)
                if room and msg_id and emoji:
                    if action == "remove":
                        success = await room_manager.remove_reaction(room, msg_id, user.user_id, emoji)
                    else:
                        success = await room_manager.add_reaction(room, msg_id, user.user_id, emoji)
                    if success:
                        await room_manager._broadcast_json(room, {
                            "type": "reaction_update",
                            "message_id": msg_id,
                            "user_id": user.user_id,
                            "user_name": user.name,
                            "emoji": emoji,
                            "action": action,
                        })

            elif msg_type == "request_topics":
                # Request topic suggestions
                room = room_manager.get_room(room_id)
                if room:
                    topics = await room_manager.suggest_topics(room)
                    await websocket.send_json({
                        "type": "topic_suggestions",
                        "topics": topics,
                    })

            else:
                logger.debug(f"[CLASSROOM] Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        logger.info(f"[CLASSROOM] User {user.user_id} disconnected from room {room_id}")
    except Exception as e:
        logger.error(f"[CLASSROOM] WebSocket error for {user.user_id}: {e}")
    finally:
        if user:
            await room_manager.remove_user(room_id, user.user_id)


