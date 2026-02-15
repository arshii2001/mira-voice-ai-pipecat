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
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import openai
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from translator import Translator, LANG_NAMES
from database import db as classroom_db
from bot import load_system_prompt, PROMPT_VERSION
from auth import AUTH_ENABLED, get_verified_user_id, extract_bearer_token, decode_openwebui_jwt

logger = logging.getLogger(__name__)

# Regex to strip [TEACHER_ACTION: ...] / [TUTOR_ACTION: ...] tags from LLM output
_ACTION_TAG_RE = re.compile(r'\[(?:TEACHER_ACTION|TUTOR_ACTION):\s*[^\]]*\]\s*', re.IGNORECASE)
_SUPPORTED_CLASSROOM_LANGUAGES = {"en", "hi", "ta"}


def _normalize_classroom_language(lang: str) -> Optional[str]:
    """Normalize user-provided language to one of en/hi/ta."""
    if not lang:
        return None
    normalized = lang.strip().lower()
    alias_map = {
        "english": "en",
        "hindi": "hi",
        "tamil": "ta",
    }
    code = alias_map.get(normalized, normalized)
    return code if code in _SUPPORTED_CLASSROOM_LANGUAGES else None


def _response_language_instruction(lang_code: str) -> str:
    """Hard instruction for assistant response language by speaker profile."""
    code = _normalize_classroom_language(lang_code or "en") or "en"
    if code == "hi":
        return (
            "CRITICAL RESPONSE LANGUAGE RULE: Reply only in Hindi (Devanagari script). "
            "Do not use English except unavoidable technical terms."
        )
    if code == "ta":
        return (
            "CRITICAL RESPONSE LANGUAGE RULE: Reply only in Tamil script. "
            "Do not use English except unavoidable technical terms."
        )
    return (
        "CRITICAL RESPONSE LANGUAGE RULE: Reply only in English. "
        "Do not use Hindi, Tamil, Urdu, Chinese, or any other language."
    )


# ─────────────────────────────────────────────────────────────────────
# Metrics Collector (singleton, thread-safe via asyncio single-thread model)
# ─────────────────────────────────────────────────────────────────────


class _ComponentBucket:
    """Rolling stats buffer for a single component dimension."""

    def __init__(self, max_samples: int = 500):
        self._max = max_samples
        self.llm_ttft_ms: list[float] = []
        self.llm_total_ms: list[float] = []
        self.llm_tokens: list[int] = []
        self.translation_ms: list[float] = []
        self.tts_ms: list[float] = []
        self.tts_bytes: list[int] = []
        self.listener_delivery_ms: list[float] = []  # end-to-end per-listener sentence delivery
        self.turn_latency_ms: list[float] = []  # voice pipeline: user-stop → first-audio
        self.stt_latency_ms: list[float] = []   # voice pipeline: VAD→STT
        self.llm_to_tts_ms: list[float] = []    # voice pipeline: LLM first token → TTS start
        self.errors: Dict[str, int] = {}
        self.query_count: int = 0
        self.turn_count: int = 0

    def _append(self, buf: list, value):
        buf.append(value)
        if len(buf) > self._max:
            buf[:] = buf[-self._max:]

    @staticmethod
    def _stats(samples: list) -> dict:
        if not samples:
            return {"count": 0, "min": 0, "avg": 0, "p95": 0, "max": 0}
        s = sorted(samples)
        n = len(s)
        return {
            "count": n,
            "min": round(s[0], 1),
            "avg": round(sum(s) / n, 1),
            "p95": round(s[int(n * 0.95)], 1) if n > 1 else round(s[0], 1),
            "max": round(s[-1], 1),
        }

    def to_dict(self) -> dict:
        d: dict = {}
        if self.llm_ttft_ms:
            d["llm_ttft_ms"] = self._stats(self.llm_ttft_ms)
        if self.llm_total_ms:
            d["llm_total_ms"] = self._stats(self.llm_total_ms)
        if self.llm_tokens:
            d["llm_tokens"] = self._stats(self.llm_tokens)
        if self.translation_ms:
            d["translation_ms"] = self._stats(self.translation_ms)
        if self.tts_ms:
            d["tts_ms"] = self._stats(self.tts_ms)
        if self.tts_bytes:
            d["tts_audio_bytes"] = self._stats(self.tts_bytes)
        if self.listener_delivery_ms:
            d["listener_delivery_ms"] = self._stats(self.listener_delivery_ms)
        if self.turn_latency_ms:
            d["turn_latency_ms"] = self._stats(self.turn_latency_ms)
        if self.stt_latency_ms:
            d["stt_latency_ms"] = self._stats(self.stt_latency_ms)
        if self.llm_to_tts_ms:
            d["llm_to_tts_ms"] = self._stats(self.llm_to_tts_ms)
        d["query_count"] = self.query_count
        d["turn_count"] = self.turn_count
        if self.errors:
            d["errors"] = dict(self.errors)
        return d


class MetricsCollector:
    """Server-side performance metrics, separated by mode and role.

    Structure:
        tutor/          – voice pipeline (Pipecat) metrics for 1:1 tutor mode
          voice/        – full voice turns (VAD→STT→LLM→TTS→audio)
          text/         – typed text queries via /chat or /inject_text
        classroom/
          speaker/      – speaker's LLM queries (text or voice)
          listener/     – per-listener delivery: translation + TTS
        sessions/       – active/total counts, recent summaries
        errors/         – global error counts
    """

    def __init__(self, max_samples: int = 500, max_traces: int = 50):
        self._max = max_samples
        self._max_traces = max_traces
        self._start_time = time.time()

        # ── Tutor mode ──
        self.tutor_voice = _ComponentBucket(max_samples)
        self.tutor_text = _ComponentBucket(max_samples)

        # ── Classroom mode ──
        self.classroom_speaker = _ComponentBucket(max_samples)
        self.classroom_listener = _ComponentBucket(max_samples)

        # ── Sessions ──
        self.active_sessions: int = 0
        self.total_sessions: int = 0
        self.active_tutor_sessions: int = 0
        self.active_classroom_sessions: int = 0
        self.total_tutor_sessions: int = 0
        self.total_classroom_sessions: int = 0
        self.session_summaries: list[dict] = []

        # ── Per-call trace log (rolling buffer of last N calls) ──
        self.traces: list[dict] = []

        # ── Global errors ──
        self.errors: Dict[str, int] = {}

    def _append(self, buf: list, value):
        buf.append(value)
        if len(buf) > self._max:
            buf[:] = buf[-self._max:]

    # ── Recording helpers ──

    def record_llm_ttft(self, ttft_ms: float, mode: str = "classroom"):
        """mode: 'classroom' or 'tutor'"""
        bucket = self.classroom_speaker if mode == "classroom" else self.tutor_voice
        bucket._append(bucket.llm_ttft_ms, ttft_ms)

    def record_llm_query(self, total_ms: float, tokens: int, mode: str = "classroom"):
        bucket = self.classroom_speaker if mode == "classroom" else self.tutor_voice
        bucket._append(bucket.llm_total_ms, total_ms)
        bucket._append(bucket.llm_tokens, tokens)
        bucket.query_count += 1

    def record_translation(self, ms: float):
        self.classroom_listener._append(self.classroom_listener.translation_ms, ms)

    def record_tts(self, ms: float, audio_bytes: int, mode: str = "classroom"):
        bucket = self.classroom_listener if mode == "classroom" else self.tutor_voice
        bucket._append(bucket.tts_ms, ms)
        bucket._append(bucket.tts_bytes, audio_bytes)

    def record_listener_delivery(self, total_ms: float):
        self.classroom_listener._append(self.classroom_listener.listener_delivery_ms, total_ms)

    def record_voice_turn(self, turn_latency_ms: float, stt_ms: float,
                          llm_ttft_ms: float, llm_total_ms: float,
                          llm_to_tts_ms: float, tts_ms: float,
                          tokens: int, tts_bytes: int,
                          is_classroom: bool = False):
        """Record a full voice pipeline turn from PipelineInstrumentor."""
        bucket = self.classroom_speaker if is_classroom else self.tutor_voice
        if turn_latency_ms > 0:
            bucket._append(bucket.turn_latency_ms, turn_latency_ms)
        if stt_ms > 0:
            bucket._append(bucket.stt_latency_ms, stt_ms)
        if llm_ttft_ms > 0:
            bucket._append(bucket.llm_ttft_ms, llm_ttft_ms)
        if llm_total_ms > 0:
            bucket._append(bucket.llm_total_ms, llm_total_ms)
            bucket._append(bucket.llm_tokens, tokens)
        if llm_to_tts_ms > 0:
            bucket._append(bucket.llm_to_tts_ms, llm_to_tts_ms)
        if tts_ms > 0:
            bucket._append(bucket.tts_ms, tts_ms)
            bucket._append(bucket.tts_bytes, tts_bytes)
        bucket.turn_count += 1

    def record_tutor_text_query(self, total_ms: float, tokens: int, ttft_ms: float):
        """Record a tutor text-mode query (via /chat endpoint)."""
        self.tutor_text._append(self.tutor_text.llm_total_ms, total_ms)
        self.tutor_text._append(self.tutor_text.llm_tokens, tokens)
        if ttft_ms > 0:
            self.tutor_text._append(self.tutor_text.llm_ttft_ms, ttft_ms)
        self.tutor_text.query_count += 1

    def record_trace(self, trace: dict):
        """Record a per-call trace with full pipeline breakdown.

        trace should contain:
            mode: "tutor_text" | "tutor_voice" | "classroom"
            query: str (truncated)
            ts: float (epoch)
            stages: list of {name, ms, detail?}
            total_ms: float
            listeners?: list of {user, language, stages: [{name, ms}], total_ms}
        """
        trace.setdefault("ts", time.time())
        self.traces.append(trace)
        if len(self.traces) > self._max_traces:
            self.traces[:] = self.traces[-self._max_traces:]

    def record_error(self, category: str):
        self.errors[category] = self.errors.get(category, 0) + 1

    def _stt_clarity_snapshot(self) -> dict:
        """Return STT clarity metrics for the /metrics endpoint."""
        scores = getattr(self, "_stt_clarity_scores", [])
        gated = getattr(self, "_stt_clarity_gated", 0)
        passed = getattr(self, "_stt_clarity_passed", 0)
        by_lang = getattr(self, "_stt_clarity_by_lang", {})
        total = gated + passed
        result = {
            "total_utterances": total,
            "gated_count": gated,
            "passed_count": passed,
            "gate_rate_pct": round(gated / total * 100, 1) if total > 0 else 0.0,
        }
        if scores:
            sorted_scores = sorted(scores)
            result["avg_score"] = round(sum(scores) / len(scores), 3)
            result["p50_score"] = round(sorted_scores[len(sorted_scores) // 2], 3)
            result["p10_score"] = round(sorted_scores[max(0, len(sorted_scores) // 10)], 3)
            result["min_score"] = round(sorted_scores[0], 3)
        # Per-language breakdown
        lang_summary = {}
        for lang, lang_scores in by_lang.items():
            if lang_scores:
                lang_summary[lang] = {
                    "count": len(lang_scores),
                    "avg_score": round(sum(lang_scores) / len(lang_scores), 3),
                }
        if lang_summary:
            result["by_language"] = lang_summary
        return result

    # ── STT Clarity metrics ──

    def record_stt_clarity(self, score: float, gated: bool, language: str = "en"):
        """Record an STT clarity score and whether the utterance was gated."""
        if not hasattr(self, "_stt_clarity_scores"):
            self._stt_clarity_scores: list[float] = []
            self._stt_clarity_gated: int = 0
            self._stt_clarity_passed: int = 0
            self._stt_clarity_force_passed: int = 0
            self._stt_clarity_by_lang: Dict[str, list[float]] = {}
        self._stt_clarity_scores.append(score)
        if len(self._stt_clarity_scores) > self._max:
            self._stt_clarity_scores[:] = self._stt_clarity_scores[-self._max:]
        if gated:
            self._stt_clarity_gated += 1
        else:
            self._stt_clarity_passed += 1
        # Per-language tracking
        lang_scores = self._stt_clarity_by_lang.setdefault(language, [])
        lang_scores.append(score)
        if len(lang_scores) > self._max:
            lang_scores[:] = lang_scores[-self._max:]

    def session_start(self, mode: str = "classroom"):
        self.active_sessions += 1
        self.total_sessions += 1
        if mode == "classroom":
            self.active_classroom_sessions += 1
            self.total_classroom_sessions += 1
        else:
            self.active_tutor_sessions += 1
            self.total_tutor_sessions += 1

    def session_end(self, summary: dict):
        self.active_sessions = max(0, self.active_sessions - 1)
        mode = summary.get("type", summary.get("mode", "classroom"))
        if mode in ("voice", "tutor"):
            self.active_tutor_sessions = max(0, self.active_tutor_sessions - 1)
        else:
            self.active_classroom_sessions = max(0, self.active_classroom_sessions - 1)
        self._append(self.session_summaries, summary)

    def snapshot(self) -> dict:
        """Return a JSON-serializable metrics snapshot."""
        uptime = time.time() - self._start_time
        return {
            "uptime_seconds": round(uptime, 1),
            "sessions": {
                "active": self.active_sessions,
                "total": self.total_sessions,
                "tutor": {
                    "active": self.active_tutor_sessions,
                    "total": self.total_tutor_sessions,
                },
                "classroom": {
                    "active": self.active_classroom_sessions,
                    "total": self.total_classroom_sessions,
                },
            },
            "tutor": {
                "voice": self.tutor_voice.to_dict(),
                "text": self.tutor_text.to_dict(),
            },
            "classroom": {
                "speaker": self.classroom_speaker.to_dict(),
                "listener": self.classroom_listener.to_dict(),
            },
            "errors": dict(self.errors),
            "stt_clarity": self._stt_clarity_snapshot(),
            "recent_sessions": self.session_summaries[-10:],
            "traces": self.traces[-20:],  # Last 20 per-call traces
        }


_metrics_collector = MetricsCollector()

# ─────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────


@dataclass
class RoomUser:
    """A user connected to a classroom room."""
    user_id: str
    name: str
    language: str  # preferred language code: en, hi, ta
    websocket: WebSocket
    mode: str = "text_and_audio"  # "text_only" or "text_and_audio"
    is_speaker: bool = False
    joined_at: float = field(default_factory=time.time)
    _audio_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


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
    room_type: str = "teacher_driven"  # "teacher_driven" or "discussion"
    # ── Curriculum topic context (chapter:section) ──
    curriculum_chapter_id: Optional[str] = None
    curriculum_section_id: Optional[str] = None
    # ── Reconnect support ──
    _disconnected_users: Dict[str, dict] = field(default_factory=dict)  # user_id -> {name, language, mode, is_speaker, disconnected_at}
    _speaker_grace_task: Optional[asyncio.Task] = None  # Pending speaker reassignment
    _grace_speaker_id: Optional[str] = None  # Speaker ID held during grace period
    _intro_announced: bool = False  # One-time room intro prompt has been sent
    _discussion_auto_release_task: Optional[asyncio.Task] = None  # Inactivity timer for discussion speaker


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
            "room_type": self.room_type,
            "curriculum_chapter_id": self.curriculum_chapter_id,
            "curriculum_section_id": self.curriculum_section_id,
        }


# ─────────────────────────────────────────────────────────────────────
# Room Manager (singleton)
# ─────────────────────────────────────────────────────────────────────


DEFAULT_ROOM_ID = os.getenv("CLASSROOM_DEFAULT_ROOM_ID", "default")
DEFAULT_ROOM_NAME = os.getenv("CLASSROOM_DEFAULT_ROOM_NAME", "Mira Classroom")
CLASSROOM_AUTO_DELETE_EMPTY_ROOMS = os.getenv("CLASSROOM_AUTO_DELETE_EMPTY_ROOMS", "false").strip().lower() in ("1", "true", "yes", "on")
DISCUSSION_AUTO_RELEASE_SECS = int(os.getenv("DISCUSSION_AUTO_RELEASE_SECS", "20"))


class RoomManager:
    """Manages classroom rooms, users, and speaker tokens."""
    _SPEAKER_HANDBACK_LINES = [
        "Yes {name}, how can I help you?",
        "Go ahead, {name}. What would you like to ask?",
        "{name}, I'm listening. What can I help you with?",
        "Ready, {name}. What should we work on?",
    ]

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
                logger.info("Classroom TTS initialized (provider from TTS_PROVIDER)")
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
        """Return the co-teaching system prompt for AI-assisted teaching mode.

        Composes base + mode-specific prompt from versioned files.
        Routes to 'discussion' prompt when room_type is 'discussion',
        otherwise uses 'classroom' prompt. Language rules are inherited
        from v4-base.md in both cases.
        Appends dynamic student context (topic, speaker name).
        Appends curriculum context if a topic matches loaded curriculum.
        """
        # Route prompt based on room type — teacher_driven uses 'classroom', discussion uses 'discussion'
        mode = "discussion" if (room and room.room_type == "discussion") else "classroom"
        prompt = load_system_prompt(version=PROMPT_VERSION, mode=mode)

        # Build dynamic context block
        context_lines = []
        if room and room.current_lesson_topic:
            context_lines.append(f"Topic: {room.current_lesson_topic}")
        if room and room.speaker_id and room.speaker_id in room.users:
            speaker = room.users[room.speaker_id]
            context_lines.append(f"Current speaker: {speaker.name}")
        # List all attendees so the LLM knows who is in the room
        if room and room.users:
            attendee_parts = []
            for u in room.users.values():
                role = "teacher" if u.user_id == room.teacher_id else "student"
                status = "speaking" if u.is_speaker else "listening"
                attendee_parts.append(f"{u.name} ({u.language}, {role}, {status})")
            if attendee_parts:
                context_lines.append(f"Attendees ({len(attendee_parts)}): {', '.join(attendee_parts)}")

        if context_lines:
            prompt += "\n\n--- STUDENT CONTEXT ---\n" + "\n".join(context_lines) + "\n"

        # Append curriculum context — prefer section-based lookup, fall back to concept-based
        if room:
            from curriculum_manager import get_curriculum_manager
            cm = get_curriculum_manager()
            if cm.available:
                speaker_lang = (
                    room.users[room.speaker_id].language
                    if room.speaker_id and room.speaker_id in room.users
                    else "english"
                )
                curriculum_ctx = None

                # 1. Section-based context (from chapter:section dropdowns)
                if room.curriculum_section_id:
                    curriculum_ctx = cm.get_section_context(
                        room.curriculum_section_id, language=speaker_lang,
                    )

                # 2. Fall back to concept-based context (from topic string)
                if not curriculum_ctx and room.current_lesson_topic:
                    curriculum_ctx = cm.get_context_for_topic(
                        room.current_lesson_topic, language=speaker_lang,
                    )

                if curriculum_ctx:
                    prompt += "\n\n--- CURRICULUM CONTEXT ---\n" + curriculum_ctx + "\n"
                    logger.debug(f"[CURRICULUM] Injected {len(curriculum_ctx)} chars for room {room.room_id}")

        return prompt

    # Clause boundary pattern for chunked streaming to listeners.
    # Splitting on clauses (commas, semicolons, dashes, colons) in addition to
    # sentence-enders (. ! ? ।) delivers smaller chunks to listeners sooner,
    # reducing perceived latency by ~40% compared to full-sentence buffering.
    _CLAUSE_RE = re.compile(r'(?<=[.!?।,;:\-–—\n])\s+')
    _MIN_CLAUSE_LEN = 20  # Don't dispatch tiny fragments (< 20 chars)

    async def _stream_sentence_to_listeners(
        self,
        room: Room,
        sentence: str,
        source_lang: str,
        is_final: bool = False,
    ) -> list[dict]:
        """Translate + send a sentence chunk to all listeners in parallel.

        Returns a list of per-listener delivery timing dicts.
        """
        if not sentence.strip():
            return []

        # Per-sentence language detection: the LLM may drift mid-response,
        # so re-detect the actual language of THIS sentence rather than
        # trusting the source_lang set at the start of the response.
        detected_lang = self._detect_text_language(sentence)
        if detected_lang != source_lang:
            logger.info(
                f"[CLASSROOM] Per-sentence drift: expected={source_lang}, "
                f"detected={detected_lang}, sentence='{sentence[:60]}'"
            )
            source_lang = detected_lang

        tasks = []
        # Snapshot users to avoid "dictionary changed size during iteration"
        listeners = [u for u in list(room.users.values()) if u.user_id != room.speaker_id]
        for user in listeners:
            tasks.append(
                self._deliver_sentence_to_listener(user, sentence, source_lang, is_final)
            )
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in results if isinstance(r, dict)]
        return []

    async def _deliver_sentence_to_listener(
        self,
        user: RoomUser,
        sentence: str,
        source_lang: str,
        is_final: bool,
    ) -> dict:
        """Deliver a single sentence chunk to one listener (translate if needed, TTS if audio mode).

        Text (bot_text) is sent immediately — never blocked by TTS.
        Audio is serialized via per-user _audio_lock so binary frames from
        different sentences never interleave on the same WebSocket.

        Returns a timing dict: {user, language, mode, translate_ms, tts_ms, audio_bytes, total_ms}
        """
        t_start = time.time()
        try:
            # ── Translation (no lock needed) ──
            t_translate_start = time.time()
            if user.language != source_lang and self._translator:
                translated = await self._translator.translate(
                    text=sentence,
                    target_lang=user.language,
                    source_lang=source_lang,
                )
            else:
                translated = sentence
            t_translate_done = time.time()
            translate_ms = round((t_translate_done - t_translate_start) * 1000, 1)

            if user.language != source_lang:
                logger.info(
                    f"[SMOOTH][CLASSROOM] translate | user={user.name} "
                    f"| {source_lang}→{user.language} | {translate_ms}ms | "
                    f"in={len(sentence)}→out={len(translated)} chars"
                )
                _metrics_collector.record_translation(translate_ms)

            # Send streamed text chunk IMMEDIATELY — not gated by audio lock.
            # This ensures listeners see text even while greeting/previous TTS
            # is still playing.
            await self._send_json(user.websocket, {
                "type": "bot_text",
                "text": translated,
                "streaming": True,
                "translated": user.language != source_lang,
                "target_language": user.language,
            })

            # TTS for audio-mode listeners — acquire lock only for audio portion
            tts_ms = 0.0
            tts_first_byte_ms = 0.0
            audio_bytes = 0
            audio_chunks = 0
            if self._tts and user.mode == "text_and_audio":
                async with user._audio_lock:
                    t_tts_start = time.time()
                    await self._send_json(user.websocket, {"type": "bot_audio_start"})
                    try:
                        async for frame in self._tts.run_tts(translated):
                            if hasattr(frame, "audio") and frame.audio:
                                if audio_chunks == 0:
                                    tts_first_byte_ms = round((time.time() - t_tts_start) * 1000, 1)
                                await self._send_bytes(user.websocket, frame.audio)
                                audio_bytes += len(frame.audio)
                                audio_chunks += 1
                    except Exception as tts_err:
                        logger.warning(f"[CLASSROOM] Sentence TTS error for {user.name}: {tts_err}")
                        _metrics_collector.record_error("listener_tts")
                    await self._send_json(user.websocket, {"type": "bot_audio_end"})
                    tts_ms = round((time.time() - t_tts_start) * 1000, 1)
                    _metrics_collector.record_tts(tts_ms, audio_bytes)
            elif not self._tts and user.mode == "text_and_audio":
                logger.warning(f"[CLASSROOM] No TTS available for audio-mode listener {user.name}")

            total_ms = round((time.time() - t_start) * 1000, 1)
            logger.info(
                f"[SMOOTH][CLASSROOM] deliver | user={user.name}({user.language}) "
                f"| total={total_ms}ms | translate={translate_ms}ms "
                f"| tts={tts_ms}ms (first_byte={tts_first_byte_ms}ms, {audio_chunks}chunks) "
                f"| audio={audio_bytes}B | final={is_final} "
                f"| '{translated[:40]}'"
            )
            _metrics_collector.record_listener_delivery(total_ms)

            return {
                "user": user.name,
                "language": user.language,
                "mode": user.mode,
                "translate_ms": translate_ms,
                "tts_ms": tts_ms,
                "audio_bytes": audio_bytes,
                "total_ms": total_ms,
            }

        except Exception as e:
            total_ms = round((time.time() - t_start) * 1000, 1)
            logger.warning(
                f"[METRICS][CLASSROOM] deliver_sentence_error | user={user.name} "
                f"| elapsed={total_ms}ms | error={e}"
            )
            _metrics_collector.record_error("deliver_sentence")
            return {
                "user": user.name,
                "language": user.language,
                "mode": user.mode,
                "error": str(e),
                "total_ms": total_ms,
            }

    @staticmethod
    def _detect_text_language(text: str) -> str:
        """Detect the language of text by examining Unicode script.

        Used to determine the actual language of LLM output (which may differ
        from the speaker's registered language if the LLM drifts).
        Returns 'hi', 'ta', or 'en'.
        """
        # Strip ASCII/Latin chars, markdown, and whitespace
        import re as _re
        non_latin = _re.sub(r'[\x00-\x7F]', '', text)
        if not non_latin:
            return "en"

        devanagari = sum(1 for c in non_latin if '\u0900' <= c <= '\u097F')
        tamil = sum(1 for c in non_latin if '\u0B80' <= c <= '\u0BFF')

        counts = {"hi": devanagari, "ta": tamil}
        best = max(counts, key=counts.get)
        if counts[best] > 0:
            return best
        return "en"

    @staticmethod
    def _make_language_tag(lang_code: str) -> str:
        """Convert a language code to a [User is speaking X] tag for the LLM."""
        label = {"en": "English", "hi": "Hindi", "ta": "Tamil"}.get(lang_code, "English")
        return f"[User is speaking {label}]"

    async def ask_llm(
        self,
        question: str,
        speaker_ws: WebSocket,
        room: Optional["Room"] = None,
        stream_to_listeners: bool = True,
    ) -> Optional[str]:
        """Send a text question to the LLM, stream tokens to speaker AND listeners.

        Speaker gets every token as it arrives (bot_text).
        Listeners get sentence-by-sentence delivery:
          - Same-language text_only: translated text chunk per sentence
          - Different-language: translated chunk per sentence
          - Audio mode: translated chunk + TTS per sentence

        When stream_to_listeners=False, behaves like before (speaker only).
        Maintains conversation history per room for context continuity.
        """
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
        sentence_buffer = ""  # Accumulates tokens until a sentence boundary
        pending_listener_tasks: list[asyncio.Task] = []
        source_lang = "en"  # Will be overridden to speaker's registered language

        # ── Timing anchors ──
        t_first_token: float = 0.0      # Time-to-first-token
        token_count: int = 0
        sentence_count: int = 0

        # Detect speaker's language — prefer actual text language detection
        # over registered language, since users may switch languages mid-session.
        if room and room.speaker_id and room.speaker_id in room.users:
            registered_lang = _normalize_classroom_language(room.users[room.speaker_id].language or "en") or "en"
            # Detect the actual language of the question text
            detected_lang = self._detect_text_language(question) if len(question) >= 5 else registered_lang
            source_lang = detected_lang
            # Update the user's language if they switched
            if detected_lang != registered_lang:
                room.users[room.speaker_id].language = detected_lang
                logger.info(
                    f"[CLASSROOM] Speaker language updated in text mode: "
                    f"{registered_lang} → {detected_lang} (detected from question text)"
                )

        # ── CRITICAL: Ensure language tag is always present ──
        # The voice pipeline (SonioxSTT) adds [User is speaking X] tags,
        # but the text-message and teacher-action paths do NOT.
        # Without a tag, conversation history in other languages causes drift.
        # Always prepend a tag if the question doesn't already have one.
        if not question.startswith("[User is speaking"):
            lang_tag = self._make_language_tag(source_lang)
            question = f"{lang_tag} {question}"
            logger.info(f"[CLASSROOM] Prepended language tag: {lang_tag} to text question")

        # Build messages with conversation history for context
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "system", "content": _response_language_instruction(source_lang)})
        if room and room.conversation_history:
            # Include last 20 messages for context
            messages.extend(room.conversation_history[-20:])
        messages.append({"role": "user", "content": question})

        try:
            # Only send chat_template_kwargs to vLLM (not real OpenAI)
            _llm_base = os.getenv("LLM_BASE_URL", "")
            _extra_body = {} if "api.openai.com" in _llm_base else {"chat_template_kwargs": {"enable_thinking": False}}
            stream = await asyncio.wait_for(
                self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=messages,
                    max_tokens=400,  # Classroom brevity: 1-3 sentences ≈ 50-150 tokens; 400 allows quizzes/summaries
                    temperature=0.7,
                    stream=True,
                    extra_body=_extra_body if _extra_body else None,
                ),
                timeout=30.0,  # 30s to start the stream
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    token = delta.content
                    full_response += token
                    sentence_buffer += token
                    token_count += 1

                    # Track TTFT (Time-to-First-Token)
                    if token_count == 1:
                        t_first_token = time.time()
                        ttft_ms = round((t_first_token - t0) * 1000, 1)
                        logger.info(
                            f"[METRICS][CLASSROOM] ask_llm_ttft | "
                            f"ttft={ttft_ms}ms | first_token='{token[:20]}'"
                        )
                        # Record for aggregation
                        _metrics_collector.record_llm_ttft(ttft_ms)

                    # After accumulating ~30 chars, detect the actual output language.
                    # This catches LLM drift (e.g. responding in Hindi when asked in English).
                    # We update source_lang so listener translation uses the correct source.
                    if token_count == 10 or (token_count < 10 and len(full_response) >= 30):
                        detected_output_lang = self._detect_text_language(full_response)
                        if detected_output_lang != source_lang:
                            logger.warning(
                                f"[CLASSROOM] LLM language drift detected! "
                                f"Expected={source_lang}, actual={detected_output_lang}, "
                                f"text='{full_response[:50]}'"
                            )
                            source_lang = detected_output_lang

                    # Stream each token to the speaker immediately
                    try:
                        await speaker_ws.send_json({
                            "type": "bot_text",
                            "text": token,
                            "streaming": True,
                        })
                    except Exception:
                        break

                    # Check for clause boundary — dispatch to listeners.
                    # We split on clauses (commas, semicolons, dashes) not just
                    # sentence-enders, so listeners hear audio ~800ms sooner.
                    # A minimum length guard prevents dispatching tiny fragments
                    # like "Yes," or "So," which translate/TTS poorly.
                    if (stream_to_listeners and room
                            and self._CLAUSE_RE.search(sentence_buffer)
                            and len(sentence_buffer) >= self._MIN_CLAUSE_LEN):
                        # Split on the last clause boundary
                        parts = self._CLAUSE_RE.split(sentence_buffer)
                        # Send all complete clauses, keep the remainder
                        complete = " ".join(parts[:-1]).strip()
                        sentence_buffer = parts[-1] if len(parts) > 1 else ""

                        if complete and len(complete) >= self._MIN_CLAUSE_LEN:
                            sentence_count += 1
                            t_sentence_dispatch = time.time()
                            logger.info(
                                f"[METRICS][CLASSROOM] clause_dispatch | "
                                f"clause={sentence_count} | "
                                f"elapsed={round((t_sentence_dispatch - t0) * 1000, 1)}ms | "
                                f"len={len(complete)} | text='{complete[:50]}'"
                            )
                            task = asyncio.create_task(
                                self._stream_sentence_to_listeners(
                                    room, complete, source_lang, is_final=False,
                                )
                            )
                            pending_listener_tasks.append(task)

            t_llm_done = time.time()

            # Flush remaining sentence buffer to listeners
            if stream_to_listeners and room and sentence_buffer.strip():
                sentence_count += 1
                logger.info(
                    f"[METRICS][CLASSROOM] clause_dispatch | "
                    f"clause={sentence_count} (final flush) | "
                    f"elapsed={round((t_llm_done - t0) * 1000, 1)}ms | "
                    f"len={len(sentence_buffer.strip())} | text='{sentence_buffer.strip()[:50]}'"
                )
                task = asyncio.create_task(
                    self._stream_sentence_to_listeners(
                        room, sentence_buffer.strip(), source_lang, is_final=True,
                    )
                )
                pending_listener_tasks.append(task)

            # Strip any [TEACHER_ACTION: ...] tags the LLM may have echoed
            cleaned_response = _ACTION_TAG_RE.sub('', full_response).strip()
            if cleaned_response != full_response:
                logger.info(f"[CLASSROOM] Stripped action tags from LLM response: '{full_response[:80]}' -> '{cleaned_response[:80]}'")
                full_response = cleaned_response

            # Send complete response to speaker
            try:
                await speaker_ws.send_json({
                    "type": "bot_text_complete",
                    "text": full_response,
                })
            except Exception:
                pass

            # Update conversation history for context continuity
            # (must happen before returning so the next message has context)
            if room:
                room.conversation_history.append({"role": "user", "content": question})
                room.conversation_history.append({"role": "assistant", "content": full_response})
                # Keep history manageable
                if len(room.conversation_history) > 40:
                    room.conversation_history = room.conversation_history[-30:]

            # ── Listener fanout completion + metrics in background ──
            # We must NOT block the message loop waiting for listener
            # translation / TTS — that prevents the speaker from passing
            # the token or sending new messages while listeners are still
            # receiving audio.
            # Capture speaker_id now — it may change by the time the
            # background task runs (e.g. token was passed).
            _original_speaker_id = room.speaker_id if room else None

            async def _finish_listener_fanout():
                try:
                    t_listener_wait_start = time.time()
                    listener_delivery_results: list[dict] = []
                    if pending_listener_tasks:
                        raw_results = await asyncio.gather(
                            *pending_listener_tasks, return_exceptions=True,
                        )
                        for r in raw_results:
                            if isinstance(r, list):
                                listener_delivery_results.extend(r)
                            elif isinstance(r, Exception):
                                logger.warning(f"[CLASSROOM] Listener fanout error: {r}")
                    t_listener_done = time.time()

                    # Signal listeners that streaming is done
                    if stream_to_listeners and room:
                        await self._broadcast_json(room, {
                            "type": "bot_text_complete",
                            "text": "",
                        }, exclude=_original_speaker_id)

                        asyncio.create_task(
                            self.save_message_to_db(
                                room=room, role="assistant",
                                content=full_response,
                                speaker_name="Mira",
                                original_language=source_lang,
                            )
                        )

                    # ── Comprehensive timing summary ──
                    total_ms = round((time.time() - t0) * 1000, 1)
                    llm_stream_ms = round((t_llm_done - t0) * 1000, 1)
                    _ttft_ms = round((t_first_token - t0) * 1000, 1) if t_first_token else 0.0
                    listener_wait_ms = round(
                        (t_listener_done - t_listener_wait_start) * 1000, 1,
                    )
                    _tok_per_sec = round(
                        token_count / ((t_llm_done - t0) or 1), 1,
                    )

                    logger.info(
                        f"[METRICS][CLASSROOM] ask_llm_complete | "
                        f"total={total_ms}ms | ttft={_ttft_ms}ms | "
                        f"llm_stream={llm_stream_ms}ms | "
                        f"listener_fanout={listener_wait_ms}ms | "
                        f"tokens={token_count} ({_tok_per_sec} tok/s) | "
                        f"sentences={sentence_count} | "
                        f"response_len={len(full_response)} | "
                        f"q='{question[:40]}' | a='{full_response[:40]}'"
                    )
                    _metrics_collector.record_llm_query(total_ms, token_count)

                    # ── Per-call trace ──
                    listener_summaries: Dict[str, dict] = {}
                    for ld in listener_delivery_results:
                        key = ld.get("user", "?")
                        if key not in listener_summaries:
                            listener_summaries[key] = {
                                "user": key,
                                "language": ld.get("language", "?"),
                                "mode": ld.get("mode", "?"),
                                "sentences": 0,
                                "translate_ms": 0.0,
                                "tts_ms": 0.0,
                                "audio_bytes": 0,
                                "total_ms": 0.0,
                            }
                        s = listener_summaries[key]
                        s["sentences"] += 1
                        s["translate_ms"] += ld.get("translate_ms", 0.0)
                        s["tts_ms"] += ld.get("tts_ms", 0.0)
                        s["audio_bytes"] += ld.get("audio_bytes", 0)
                        s["total_ms"] = max(
                            s["total_ms"], ld.get("total_ms", 0.0),
                        )

                    for s in listener_summaries.values():
                        s["translate_ms"] = round(s["translate_ms"], 1)
                        s["tts_ms"] = round(s["tts_ms"], 1)
                        s["total_ms"] = round(s["total_ms"], 1)

                    trace = {
                        "mode": "classroom",
                        "query": question[:80],
                        "answer": full_response[:80],
                        "ts": t0,
                        "total_ms": total_ms,
                        "stages": [
                            {"name": "llm_ttft", "ms": _ttft_ms},
                            {"name": "llm_stream", "ms": llm_stream_ms,
                             "tokens": token_count,
                             "tok_per_sec": _tok_per_sec},
                            {"name": "listener_fanout",
                             "ms": listener_wait_ms,
                             "sentences": sentence_count},
                        ],
                        "listeners": list(listener_summaries.values()),
                    }
                    _metrics_collector.record_trace(trace)
                except Exception as exc:
                    logger.warning(
                        f"[CLASSROOM] Listener fanout completion error: {exc}"
                    )

            asyncio.create_task(_finish_listener_fanout())

            return full_response

        except Exception as e:
            total_ms = round((time.time() - t0) * 1000, 1)
            logger.error(
                f"[METRICS][CLASSROOM] ask_llm_error | "
                f"elapsed={total_ms}ms | tokens={token_count} | error={e}"
            )
            _metrics_collector.record_error("ask_llm")
            try:
                await speaker_ws.send_json({"type": "error", "message": f"LLM error: {e}"})
            except Exception:
                pass
            return None

    async def init_db(self):
        """Initialize the database and hydrate rooms from persisted state."""
        await classroom_db.init()
        logger.info("[CLASSROOM] Database initialized")

        # Hydrate rooms from DB
        await self._hydrate_rooms_from_db()

    async def _hydrate_rooms_from_db(self):
        """Load all persisted rooms from the database into memory."""
        try:
            room_records = await classroom_db.list_rooms()
            loaded_count = 0
            for rec in room_records:
                if rec.room_id in self._rooms:
                    # Already in memory (e.g. default room) — update metadata
                    room = self._rooms[rec.room_id]
                    room.name = rec.name
                    room.topic = rec.topic
                    room.current_lesson_topic = rec.topic
                    room.teacher_id = rec.created_by
                    if rec.is_permanent:
                        self._permanent_rooms.add(rec.room_id)
                else:
                    # Create in-memory room from DB record
                    room = Room(
                        room_id=rec.room_id,
                        name=rec.name,
                        created_at=rec.created_at,
                        topic=rec.topic,
                        current_lesson_topic=rec.topic,
                        teacher_id=rec.created_by,
                    )
                    self._rooms[rec.room_id] = room
                    if rec.is_permanent:
                        self._permanent_rooms.add(rec.room_id)

                # Load persisted members for this room (for reconnect lookup)
                members = await classroom_db.get_room_members(rec.room_id)
                for m in members:
                    # Store in _disconnected_users so reconnecting users get their profile back
                    room._disconnected_users[m.user_id] = {
                        "name": m.display_name,
                        "language": m.language,
                        "mode": m.mode,
                        "is_speaker": m.role == "teacher",
                        "disconnected_at": m.last_active,
                        "persisted_role": m.role,
                    }

                # Load recent conversation history from DB (last 20 messages)
                recent_msgs = await classroom_db.get_messages_by_room(rec.room_id, limit=20)
                if recent_msgs:
                    room.conversation_history = [
                        {"role": msg.role, "content": msg.content}
                        for msg in recent_msgs
                    ]
                    logger.info(
                        f"[CLASSROOM] Loaded {len(recent_msgs)} messages for room {rec.room_id}"
                    )

                loaded_count += 1

            logger.info(f"[CLASSROOM] Hydrated {loaded_count} rooms from database")

            # Ensure default room exists in DB too
            if DEFAULT_ROOM_ID not in [r.room_id for r in room_records]:
                await classroom_db.save_room(
                    room_id=DEFAULT_ROOM_ID,
                    name=DEFAULT_ROOM_NAME,
                    is_permanent=True,
                )
                logger.info(f"[CLASSROOM] Persisted default room to DB: {DEFAULT_ROOM_ID}")

        except Exception as e:
            logger.error(f"[CLASSROOM] Failed to hydrate rooms from DB: {e}")

    def _create_default_room(self):
        """Create a permanent default room in memory (DB persistence happens in init_db)."""
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

    async def get_recent_room_messages(
        self, room_id: str, user_language: str = "en", limit: int = 50
    ) -> List[dict]:
        """Fetch recent room messages for chat hydration when a user joins.

        If a cached translation is missing for the joining user's language,
        performs a live translation and caches it back to the DB so
        subsequent joins don't re-translate.
        """
        try:
            records = await classroom_db.get_messages_by_room(room_id, limit=limit)
        except Exception as e:
            logger.error(f"[CLASSROOM] Failed to load room history for {room_id}: {e}")
            return []

        messages: List[dict] = []
        # Collect messages that need live translation (batch for efficiency logging)
        live_translate_count = 0

        for rec in records:
            text = rec.content or ""
            translated = False
            try:
                translations = json.loads(rec.translations) if rec.translations else {}
            except Exception:
                translations = {}

            if user_language and rec.original_language and user_language != rec.original_language:
                translated_text = translations.get(user_language)
                if translated_text:
                    text = translated_text
                    translated = True
                elif self._translator and rec.content:
                    # Live-translate missing language and cache back to DB
                    try:
                        translated_text = await self._translator.translate(
                            rec.content,
                            target_lang=user_language,
                            source_lang=rec.original_language,
                        )
                        if translated_text and translated_text != rec.content:
                            text = translated_text
                            translated = True
                            live_translate_count += 1
                            # Cache the new translation back to DB
                            translations[user_language] = translated_text
                            try:
                                await classroom_db.update_message_translations(
                                    rec.id, translations
                                )
                            except Exception as db_err:
                                logger.warning(
                                    f"[CLASSROOM] Failed to cache translation for msg {rec.id}: {db_err}"
                                )
                    except Exception as tr_err:
                        logger.warning(
                            f"[CLASSROOM] Live translation failed for msg {rec.id}: {tr_err}"
                        )

            messages.append({
                "id": rec.id,
                "role": rec.role,
                "speaker_name": rec.speaker_name,
                "content": text,
                "translated": translated,
                "timestamp": rec.timestamp,
                "reaction_counts": json.loads(rec.reaction_counts) if rec.reaction_counts else {},
            })

        if live_translate_count > 0:
            logger.info(
                f"[CLASSROOM] Live-translated {live_translate_count}/{len(records)} messages "
                f"to '{user_language}' for room {room_id} on join"
            )

        return messages

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

    async def create_room(self, name: str = "Classroom", teacher_id: Optional[str] = None,
                          teacher_name: Optional[str] = None, topic: Optional[str] = None,
                          is_permanent: bool = False, room_type: str = "teacher_driven",
                          curriculum_chapter_id: Optional[str] = None,
                          curriculum_section_id: Optional[str] = None) -> Room:
        """Create a new room and persist it to DB. The creator becomes the teacher."""
        room_id = str(uuid.uuid4())[:8]
        if room_type not in ("teacher_driven", "discussion"):
            room_type = "teacher_driven"

        # Auto-set topic from curriculum section if not explicitly provided
        if curriculum_section_id and not topic:
            try:
                from curriculum_manager import get_curriculum_manager
                cm = get_curriculum_manager()
                info = cm.get_section_info(curriculum_section_id)
                if info:
                    topic = f"{info['chapter_title']}: {info['section_title']}" if info.get('chapter_title') else info.get('section_title', '')
            except Exception:
                pass

        room = Room(room_id=room_id, name=name, teacher_id=teacher_id,
                     current_lesson_topic=topic, topic=topic, room_type=room_type,
                     curriculum_chapter_id=curriculum_chapter_id,
                     curriculum_section_id=curriculum_section_id)
        self._rooms[room_id] = room
        if is_permanent:
            self._permanent_rooms.add(room_id)

        # Persist to DB
        try:
            await classroom_db.save_room(
                room_id=room_id, name=name, topic=topic,
                created_by=teacher_id, created_by_name=teacher_name,
                is_permanent=is_permanent, room_type=room_type,
            )
        except Exception as e:
            logger.error(f"[CLASSROOM] Failed to persist room to DB: {e}")

        logger.info(f"[CLASSROOM] Room created: {room_id} ({name}) teacher={teacher_id}")
        return room

    def get_room(self, room_id: str) -> Optional[Room]:
        """Get a room by ID."""
        return self._rooms.get(room_id)

    def list_rooms(self) -> List[dict]:
        """List all active rooms."""
        return [room.to_dict() for room in self._rooms.values()]

    async def delete_room(self, room_id: str) -> bool:
        """Delete a room. Permanent rooms cannot be deleted."""
        if room_id in self._permanent_rooms:
            logger.info(f"[CLASSROOM] Skipping delete of permanent room: {room_id}")
            return False
        if room_id in self._rooms:
            del self._rooms[room_id]
            # Remove from DB
            try:
                await classroom_db.delete_room(room_id)
            except Exception as e:
                logger.error(f"[CLASSROOM] Failed to delete room from DB: {e}")
            logger.info(f"[CLASSROOM] Room deleted: {room_id}")
            return True
        return False

    def is_room_deletable(self, room_id: str) -> bool:
        """Whether the room exists and is not marked permanent."""
        return room_id in self._rooms and room_id not in self._permanent_rooms

    # ── User management ──

    async def add_user(self, room_id: str, user: RoomUser) -> bool:
        """
        Add a user to a room (or reconnect an existing user).

        Returns True on success. The caller is responsible for sending the
        'joined' response BEFORE calling finalize_join() so that the user
        receives 'joined' before any 'token_changed' events.
        """
        room = self.get_room(room_id)
        if not room:
            return False

        # Check if this is a reconnecting user (from memory or DB-hydrated)
        is_reconnect = user.user_id in room._disconnected_users
        prev_state = room._disconnected_users.pop(user.user_id, None)

        if is_reconnect and prev_state:
            was_speaker = prev_state.get("is_speaker", False)
            persisted_role = prev_state.get("persisted_role")
            logger.info(
                f"[CLASSROOM] User {user.name} ({user.user_id}) RECONNECTED to room {room_id} "
                f"[lang={user.language}] (was_speaker={was_speaker})"
            )

            # If they were the speaker and grace period is still active, restore speaker role
            if was_speaker and room._grace_speaker_id == user.user_id:
                # Cancel grace timeout
                if room._speaker_grace_task and not room._speaker_grace_task.done():
                    room._speaker_grace_task.cancel()
                room._grace_speaker_id = None
                room._speaker_grace_task = None
                user.is_speaker = True
                room.speaker_id = user.user_id
                logger.info(f"[CLASSROOM] Speaker role RESTORED for {user.name} ({user.user_id})")

            # If this is a DB-hydrated reconnect (server restart), restore teacher_id
            if persisted_role == "teacher" and room.teacher_id is None:
                room.teacher_id = user.user_id
                logger.info(f"[CLASSROOM] Teacher role RESTORED from DB for {user.name}")
        else:
            # Check if user profile exists in DB (first time in this room but known user)
            try:
                profile = await classroom_db.get_user_profile(user.user_id)
                if profile:
                    # Use DB name only if the user didn't provide a custom name
                    if user.name == f"User-{user.user_id[:4]}":
                        user.name = profile.display_name
                    # NOTE: We intentionally do NOT override user.language from DB.
                    # The user explicitly selects their language in the lobby UI,
                    # and overriding it caused Bug #3 where English users saw Hindi text.
                    logger.info(
                        f"[CLASSROOM] Loaded profile for {user.name} ({user.user_id}) "
                        f"from DB [lang={user.language}, db_lang={profile.preferred_language}]"
                    )
            except Exception as e:
                logger.debug(f"[CLASSROOM] Could not load user profile: {e}")

            logger.info(
                f"[CLASSROOM] User {user.name} ({user.user_id}) joined room {room_id} "
                f"[lang={user.language}]"
            )

        room.users[user.user_id] = user

        # Ensure DB session exists
        asyncio.create_task(self._ensure_session(room))

        # Persist user profile and room membership (fire-and-forget)
        async def _persist_user():
            try:
                role = "teacher" if user.user_id == room.teacher_id else "student"
                await classroom_db.upsert_user_profile(
                    user_id=user.user_id,
                    display_name=user.name,
                    preferred_language=user.language,
                    role=role,
                )
                await classroom_db.upsert_room_member(
                    room_id=room_id,
                    user_id=user.user_id,
                    display_name=user.name,
                    language=user.language,
                    role=role,
                    mode=user.mode,
                )
            except Exception as e:
                logger.error(f"[CLASSROOM] Failed to persist user: {e}")
        asyncio.create_task(_persist_user())

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
        Skips if the user already has the speaker role (reconnect case).
        """
        room = self.get_room(room_id)
        if not room:
            return

        # If speaker role was already restored during reconnect, just notify
        if room.speaker_id == user_id:
            # Notify all users about the (restored) speaker
            await self._broadcast_json(room, {
                "type": "token_changed",
                "speaker_id": user_id,
                "speaker_name": room.users[user_id].name if user_id in room.users else None,
            })
            return

        # Determine if there is an *active* speaker (connected user with the token).
        active_speaker = (
            room.speaker_id is not None and room.speaker_id in room.users
        )

        # If a speaker recently disconnected and is in the grace period,
        # do NOT auto-assign the token to anyone — the original speaker
        # may reconnect within 30s and reclaim it.
        grace_active = room._grace_speaker_id is not None

        # If this is the teacher joining and no one is actively speaking
        # (and no grace period is active), give them the token
        if user_id == room.teacher_id and not active_speaker and not grace_active:
            await self._assign_token(room, user_id)
        # If first user, auto-assign speaker token
        elif len(room.users) == 1 and not active_speaker and not grace_active:
            await self._assign_token(room, user_id)

    async def remove_user(self, room_id: str, user_id: str):
        """Remove a user from a room (with grace period for reconnects)."""
        room = self.get_room(room_id)
        if not room or user_id not in room.users:
            return

        user = room.users.pop(user_id)
        was_speaker = room.speaker_id == user_id
        logger.info(f"[CLASSROOM] User {user.name} ({user_id}) disconnected from room {room_id} (was_speaker={was_speaker})")

        # Save disconnected user state for potential reconnect
        room._disconnected_users[user_id] = {
            "name": user.name,
            "language": user.language,
            "mode": user.mode,
            "is_speaker": was_speaker,
            "disconnected_at": time.time(),
        }

        # Update last_active in DB (fire-and-forget)
        async def _touch():
            try:
                await classroom_db.touch_room_member(room_id, user_id)
                await classroom_db.touch_user(user_id)
            except Exception:
                pass
        asyncio.create_task(_touch())

        # Remove from token queue
        if user_id in room.token_queue:
            room.token_queue.remove(user_id)

        # Remove pending hand raises
        room.hand_raises = [hr for hr in room.hand_raises
                            if hr["user_id"] != user_id]

        # If speaker left, start grace period before reassigning
        if was_speaker:
            self._cancel_discussion_auto_release(room)
            room._grace_speaker_id = user_id
            # Cancel any existing grace task
            if room._speaker_grace_task and not room._speaker_grace_task.done():
                room._speaker_grace_task.cancel()
            room._speaker_grace_task = asyncio.create_task(
                self._speaker_grace_timeout(room, user_id)
            )
            logger.info(f"[CLASSROOM] Speaker {user.name} disconnected — 30s grace period started")

        # Notify others (user_disconnected, not user_left — they may come back)
        await self._broadcast_json(room, {
            "type": "user_left",
            "user_id": user_id,
        })

        # Don't delete room during grace period — check after grace expires
        if not room.users and not room._grace_speaker_id:
            await self._end_session(room)
            if CLASSROOM_AUTO_DELETE_EMPTY_ROOMS:
                await self.delete_room(room_id)
            else:
                logger.info(f"[CLASSROOM] Room {room_id} is empty; keeping room (auto-delete disabled)")

    async def _speaker_grace_timeout(self, room: Room, user_id: str):
        """After grace period, if speaker hasn't reconnected, reassign token."""
        try:
            await asyncio.sleep(30)  # 30-second grace period
        except asyncio.CancelledError:
            logger.info(f"[CLASSROOM] Grace period cancelled for {user_id} (reconnected)")
            return

        # Grace period expired — check if user reconnected
        if room._grace_speaker_id == user_id:
            room._grace_speaker_id = None
            room.speaker_id = None
            logger.info(f"[CLASSROOM] Grace period expired for {user_id} — reassigning speaker token")

            # Clean up disconnected user record
            room._disconnected_users.pop(user_id, None)

            # Assign to next available user
            if room.users:
                next_speaker = room.token_queue.pop(0) if room.token_queue else next(iter(room.users))
                await self._assign_token(room, next_speaker)
            elif not room.users:
                # Room is empty now
                await self._end_session(room)
                if CLASSROOM_AUTO_DELETE_EMPTY_ROOMS:
                    await self.delete_room(room.room_id)
                else:
                    logger.info(f"[CLASSROOM] Room {room.room_id} is empty after grace timeout; keeping room (auto-delete disabled)")

    # ── Speaker token ──

    def _cancel_discussion_auto_release(self, room: Room):
        """Cancel pending discussion auto-release timer, if any."""
        task = room._discussion_auto_release_task
        if task and not task.done():
            task.cancel()
        room._discussion_auto_release_task = None

    async def _schedule_discussion_auto_release(self, room: Room):
        """Start/restart inactivity timer for discussion rooms."""
        self._cancel_discussion_auto_release(room)

        if room.room_type != "discussion" or not room.speaker_id:
            return

        speaker_id = room.speaker_id
        timeout_s = max(1, DISCUSSION_AUTO_RELEASE_SECS)

        async def _timeout():
            try:
                await asyncio.sleep(timeout_s)
            except asyncio.CancelledError:
                return

            if room.room_type != "discussion":
                return
            if room.speaker_id != speaker_id:
                return

            logger.info(
                f"[CLASSROOM] Discussion inactivity timeout ({timeout_s}s): "
                f"auto-releasing token from {speaker_id} in {room.room_id}"
            )
            await self.pass_token(room.room_id, speaker_id)

        room._discussion_auto_release_task = asyncio.create_task(_timeout())

    async def request_token(self, room_id: str, user_id: str) -> bool:
        """Request the speaker token.

        In teacher_driven rooms: teacher gets priority and can reclaim anytime.
        In discussion rooms: first-come-first-served, any student can grab the token.
        """
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
        elif user_id == room.teacher_id and room.room_type == "teacher_driven":
            # Teacher always gets priority in teacher-driven rooms
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

        self._cancel_discussion_auto_release(room)

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
        self._cancel_discussion_auto_release(room)

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

        # Prompt the newly assigned speaker:
        # Short handback line when speaker token changes.
        # Translate to the speaker's language if not English.
        speaker = room.users.get(user_id)
        if speaker:
            template = random.choice(self._SPEAKER_HANDBACK_LINES)
            prompt_text = template.format(name=speaker.name)

            if speaker.language and speaker.language != "en" and self._translator:
                try:
                    translated = await self._translator.translate(
                        text=prompt_text,
                        target_lang=speaker.language,
                        source_lang="en",
                    )
                    if translated:
                        prompt_text = translated
                except Exception as e:
                    logger.warning(f"[CLASSROOM] Handback translation failed for {speaker.name}: {e}")

            await self._send_json(speaker.websocket, {
                "type": "bot_text_complete",
                "text": prompt_text,
            })

    async def send_first_join_greeting(self, room: Room, user: RoomUser):
        """Send the *text* part of the join greeting synchronously.

        This must be called (and awaited) during the join flow so that the
        ``bot_text_complete`` for the greeting arrives *before* the message
        loop starts.  That way it cannot be confused with an LLM response.

        The TTS audio part (slow) is returned as an optional coroutine that
        the caller should fire-and-forget via ``asyncio.create_task``.

        Returns:
            An awaitable for the TTS streaming, or ``None`` if no audio is
            needed.  The caller should wrap it in ``asyncio.create_task``.
        """
        base_text = f"Welcome to {room.name}, {user.name}. I am Mira. Let's start learning together."
        text = base_text

        # Translate greeting to user's preferred language when possible.
        try:
            if user.language != "en" and self._translator:
                translated = await self._translator.translate(
                    text=base_text,
                    target_lang=user.language,
                    source_lang="en",
                )
                if translated:
                    text = translated
        except Exception as e:
            logger.warning(f"[CLASSROOM] Greeting translation failed for {user.name}: {e}")

        await self._send_json(user.websocket, {
            "type": "bot_text_complete",
            "text": text,
        })

        # Return a coroutine for TTS streaming (caller runs as background task).
        if self._tts and user.mode == "text_and_audio":
            return self._stream_greeting_audio(user, text)
        return None

    async def _stream_greeting_audio(self, user: RoomUser, text: str):
        """Stream greeting TTS audio under the per-user lock.

        Safe to call via ``asyncio.create_task`` — all exceptions are caught.
        """
        try:
            async with user._audio_lock:
                await self._send_json(user.websocket, {"type": "bot_audio_start"})
                try:
                    async for frame in self._tts.run_tts(text):
                        if hasattr(frame, "audio") and frame.audio:
                            await self._send_bytes(user.websocket, frame.audio)
                except Exception as e:
                    logger.warning(f"[CLASSROOM] Greeting TTS failed for {user.name}: {e}")
                await self._send_json(user.websocket, {"type": "bot_audio_end"})
        except Exception as e:
            logger.warning(f"[CLASSROOM] Greeting audio failed for {user.name}: {e}")

    # ── Broadcasting ──

    async def broadcast_transcription(
        self,
        room: Room,
        speaker_id: str,
        text: str,
        language: str,
    ):
        """Broadcast a speaker's transcription to all listeners, translated.

        Even without a translator, sends the original text so listeners always
        see the question (untranslated is better than invisible).

        Safe to call via ``asyncio.create_task`` — all exceptions are caught.
        """
        try:
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
            for user in list(room.users.values()):
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
        except Exception as e:
            logger.warning(f"[CLASSROOM] broadcast_transcription error: {e}")

    async def broadcast_bot_response(
        self,
        room: Room,
        text: str,
        language: str,
    ):
        """Broadcast Mira's response to all listeners, translated.

        The `language` param is the *expected* language (speaker's registered lang),
        but the LLM may have drifted. We detect the actual output language to ensure
        correct translation for listeners.

        Even without a translator, sends the original text so listeners always
        see the response (untranslated is better than invisible).
        """
        # Detect actual output language — the LLM may have drifted
        actual_lang = self._detect_text_language(text)
        if actual_lang != language:
            logger.warning(
                f"[CLASSROOM] broadcast_bot_response: LLM drift detected! "
                f"Expected={language}, actual={actual_lang}, text='{text[:50]}'"
            )
            language = actual_lang

        # Persist bot response to DB
        asyncio.create_task(
            self.save_message_to_db(
                room=room, role="assistant", content=text,
                speaker_name="Mira", original_language=language,
            )
        )

        t0 = time.time()
        listener_count = 0

        tasks = []
        for user in list(room.users.values()):
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

        Uses per-user lock to serialize audio delivery so frames from
        different events never interleave on the same WebSocket.

        Flow:
          1. Detect actual language of text (guards against LLM drift)
          2. Translate original_text to listener's language
          3. Prepend speaker attribution ("Ravi asks:" / "Mira says:")
          4. Send JSON event with text
          5. Send bot_audio_start JSON
          6. Synthesize speech via run_tts() (provider-agnostic)
          7. Stream audio chunks as binary WebSocket frames
          8. Send bot_audio_end JSON
        """
        async with user._audio_lock:
            await self._send_translated_event_inner(
                user, event_type, original_text, source_lang, extra_fields
            )

    async def _send_translated_event_inner(
        self,
        user: RoomUser,
        event_type: str,
        original_text: str,
        source_lang: str,
        extra_fields: dict = None,
    ):
        """Inner logic — always called under user._audio_lock."""
        t0 = time.time()
        translate_ms = 0.0
        tts_ms = 0.0
        audio_bytes_sent = 0

        try:
            # ── Step 0: Per-text language detection ──
            # The caller's source_lang may be stale if the LLM drifted mid-response.
            # For bot_response events, re-detect to ensure correct translation direction.
            # For transcription events, trust the STT-reported language (user speech).
            if event_type == "bot_response":
                detected_lang = self._detect_text_language(original_text)
                if detected_lang != source_lang:
                    logger.info(
                        f"[CLASSROOM] _send_translated_event drift: "
                        f"expected={source_lang}, detected={detected_lang}, "
                        f"user={user.name}, text='{original_text[:60]}'"
                    )
                    source_lang = detected_lang

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
        for user in list(room.users.values()):
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
    BotStoppedSpeakingFrame,
    Frame,
    StartInterruptionFrame,
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
        self._clause_buffer = ""  # Accumulates tokens until a clause boundary
        self._pending_listener_tasks: list[asyncio.Task] = []
        self._clause_count: int = 0
        self._last_user_text = ""
        self._last_user_lang = "en"
        self._awaiting_bot_audio_end: bool = False

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
            self._awaiting_bot_audio_end = False

            # ── Dynamic language update ──
            # If the STT detects a different language than the speaker's
            # registered language, update it so translations and listener
            # delivery use the correct source language.
            if self._room.speaker_id and self._room.speaker_id in self._room.users:
                speaker_user = self._room.users[self._room.speaker_id]
                if speaker_user.language != self._last_user_lang:
                    old_lang = speaker_user.language
                    speaker_user.language = self._last_user_lang
                    logger.info(
                        f"[CLASSROOM] Speaker {speaker_user.name} language updated: "
                        f"{old_lang} → {self._last_user_lang} (detected by STT)"
                    )

            logger.info(
                f"[METRICS][CLASSROOM] stt_received | turn={self._turn_count} | "
                f"lang={self._last_user_lang} | text='{frame.text[:50]}'"
            )

            # Prepend speaker name so the LLM knows who asked
            speaker_name = None
            if self._room.speaker_id and self._room.speaker_id in self._room.users:
                speaker_name = self._room.users[self._room.speaker_id].name
            if speaker_name:
                frame.text = f"[{speaker_name} asks] {frame.text}"
                logger.info(f"[CLASSROOM] Attributed transcription to {speaker_name}")

            # Broadcast in background so we don't slow down the pipeline
            asyncio.create_task(
                self._room_mgr.broadcast_transcription(
                    room=self._room,
                    speaker_id=self._room.speaker_id or "",
                    text=self._last_user_text,  # Original text without prefix for listeners
                    language=self._last_user_lang,
                )
            )

        # Accumulate LLM response text — dispatch clauses to listeners as they form
        elif isinstance(frame, TextFrame):
            if not self._llm_buffer:
                self._llm_first_token_at = time.time()
                self._clause_buffer = ""
                self._pending_listener_tasks = []
                self._clause_count = 0
            self._llm_buffer += frame.text
            self._clause_buffer += frame.text

            # Clause-level streaming: dispatch to listeners as soon as a clause
            # boundary is detected, rather than waiting for the full response.
            # Uses the same _CLAUSE_RE / _MIN_CLAUSE_LEN as the text-mode path.
            if (self._room_mgr._CLAUSE_RE.search(self._clause_buffer)
                    and len(self._clause_buffer) >= self._room_mgr._MIN_CLAUSE_LEN):
                parts = self._room_mgr._CLAUSE_RE.split(self._clause_buffer)
                complete = " ".join(parts[:-1]).strip()
                self._clause_buffer = parts[-1] if len(parts) > 1 else ""

                if complete and len(complete) >= self._room_mgr._MIN_CLAUSE_LEN:
                    self._clause_count += 1
                    # Detect actual output language from the accumulated buffer
                    source_lang = self._room_mgr._detect_text_language(self._llm_buffer) if len(self._llm_buffer) >= 30 else self._last_user_lang
                    logger.info(
                        f"[METRICS][CLASSROOM] voice_clause_dispatch | "
                        f"turn={self._turn_count} | clause={self._clause_count} | "
                        f"len={len(complete)} | text='{complete[:50]}'"
                    )
                    task = asyncio.create_task(
                        self._room_mgr._stream_sentence_to_listeners(
                            self._room, complete, source_lang, is_final=False,
                        )
                    )
                    self._pending_listener_tasks.append(task)

        # LLM response complete — flush remaining clause buffer + update history
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
                    f"response_len={len(response_text)} chars | "
                    f"clauses_dispatched={self._clause_count}"
                )

                # Flush remaining clause buffer to listeners
                remaining = self._clause_buffer.strip()
                if remaining:
                    self._clause_count += 1
                    source_lang = self._room_mgr._detect_text_language(response_text) if len(response_text) >= 30 else self._last_user_lang
                    logger.info(
                        f"[METRICS][CLASSROOM] voice_clause_dispatch | "
                        f"turn={self._turn_count} | clause={self._clause_count} (final flush) | "
                        f"len={len(remaining)} | text='{remaining[:50]}'"
                    )
                    task = asyncio.create_task(
                        self._room_mgr._stream_sentence_to_listeners(
                            self._room, remaining, source_lang, is_final=True,
                        )
                    )
                    self._pending_listener_tasks.append(task)
                self._clause_buffer = ""

                # ── Voice-mode conversation history ──
                # The text-mode path (ask_llm) handles its own history.
                # For voice-mode, the Pipecat pipeline drives the LLM directly,
                # so we must record the exchange here for context continuity.
                if self._last_user_text:
                    self._room.conversation_history.append(
                        {"role": "user", "content": self._last_user_text}
                    )
                self._room.conversation_history.append(
                    {"role": "assistant", "content": response_text}
                )
                # Keep history manageable (same limit as ask_llm)
                if len(self._room.conversation_history) > 40:
                    self._room.conversation_history = self._room.conversation_history[-30:]

                # Also broadcast the full response for text-mode listeners
                # who may have joined late or need the complete message
                asyncio.create_task(
                    self._room_mgr.broadcast_bot_response(
                        room=self._room,
                        text=response_text,
                        language=self._last_user_lang,
                    )
                )
                if self._room.room_type == "discussion" and self._room.speaker_id:
                    self._awaiting_bot_audio_end = True

        # ── Barge-in: user interrupted the bot mid-response ──
        # Clear all accumulated buffers so the next response starts fresh.
        # Cancel pending listener delivery tasks to avoid sending stale
        # partial translations from the interrupted response.
        elif isinstance(frame, StartInterruptionFrame):
            interrupted_len = len(self._llm_buffer)
            interrupted_clauses = self._clause_count
            cancelled = 0
            for task in self._pending_listener_tasks:
                if not task.done():
                    task.cancel()
                    cancelled += 1

            # Save partial response to conversation history so context isn't lost
            if self._llm_buffer.strip():
                partial = self._llm_buffer.strip()
                if self._last_user_text:
                    self._room.conversation_history.append(
                        {"role": "user", "content": self._last_user_text}
                    )
                self._room.conversation_history.append(
                    {"role": "assistant", "content": f"{partial} [interrupted]"}
                )
                if len(self._room.conversation_history) > 40:
                    self._room.conversation_history = self._room.conversation_history[-30:]

            self._llm_buffer = ""
            self._clause_buffer = ""
            self._pending_listener_tasks = []
            self._clause_count = 0
            self._awaiting_bot_audio_end = False

            logger.warning(
                f"[CLASSROOM] ⚡ BARGE-IN in voice mode | turn={self._turn_count} | "
                f"interrupted_chars={interrupted_len} | clauses_sent={interrupted_clauses} | "
                f"listener_tasks_cancelled={cancelled}"
            )

        # Voice mode: arm discussion inactivity timer only after bot audio is fully done.
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._awaiting_bot_audio_end and self._room.room_type == "discussion" and self._room.speaker_id:
                logger.info(
                    f"[CLASSROOM] Discussion auto-release armed: "
                    f"{DISCUSSION_AUTO_RELEASE_SECS}s inactivity for speaker {self._room.speaker_id} in {self._room.room_id}"
                )
                await self._room_mgr._schedule_discussion_auto_release(self._room)
            self._awaiting_bot_audio_end = False

        # Forward all frames unchanged
        await self.push_frame(frame, direction)


# ─────────────────────────────────────────────────────────────────────
# FastAPI Router
# ─────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/classroom", tags=["classroom"])


def _get_request_actor(request: Request) -> dict:
    """Extract caller identity from JWT (if AUTH_ENABLED) or headers.

    When WEBUI_SECRET_KEY is set:
      - Authorization: Bearer <jwt> is REQUIRED
      - user_id is extracted from the verified JWT payload ("id" claim)
      - name / email / role still come from headers (supplementary display info)

    When WEBUI_SECRET_KEY is NOT set (dev/test):
      - Falls back to x-user-id / x-user-name / x-user-role headers
    """
    jwt_user_id = None

    if AUTH_ENABLED:
        auth_header = request.headers.get("authorization")
        jwt_user_id = get_verified_user_id(auth_header)
        if not jwt_user_id:
            raise HTTPException(status_code=401, detail="Valid JWT required (WEBUI_SECRET_KEY is set)")

    # user_id: prefer JWT-verified ID, fall back to headers in dev mode
    user_id = jwt_user_id or (
        request.headers.get("x-user-id")
        or request.headers.get("x-openwebui-user-id")
        or request.query_params.get("user_id")
        or ""
    ).strip()

    # Supplementary identity info (from headers — not verified, display only)
    user_name = (
        request.headers.get("x-user-name")
        or request.headers.get("x-openwebui-user-name")
        or request.query_params.get("user_name")
        or user_id
    ).strip()
    user_email = (
        request.headers.get("x-user-email")
        or request.headers.get("x-openwebui-user-email")
        or request.query_params.get("user_email")
        or ""
    ).strip()
    role = (
        request.headers.get("x-user-role")
        or request.headers.get("x-openwebui-user-role")
        or request.query_params.get("user_role")
        or "user"
    ).strip().lower()

    return {
        "user_id": user_id,
        "user_name": user_name,
        "user_email": user_email,
        "role": role,
        "is_admin": role == "admin",
    }


async def _require_authenticated(request: Request) -> dict:
    """Require any authenticated user (JWT when AUTH_ENABLED, headers in dev)."""
    actor = _get_request_actor(request)
    if not actor["user_id"]:
        raise HTTPException(status_code=401, detail="Missing user identity")
    return actor


async def _require_admin(request: Request) -> dict:
    actor = _get_request_actor(request)
    if not actor["user_id"]:
        raise HTTPException(status_code=401, detail="Missing user identity")
    if not actor["is_admin"]:
        raise HTTPException(status_code=403, detail="Admin role required")
    return actor


async def _require_teacher_or_admin(request: Request) -> dict:
    actor = _get_request_actor(request)
    if not actor["user_id"]:
        raise HTTPException(status_code=401, detail="Missing user identity")
    is_teacher = await classroom_db.is_teacher(actor["user_id"])
    actor["is_teacher"] = is_teacher
    if not (actor["is_admin"] or is_teacher):
        raise HTTPException(
            status_code=403,
            detail="Teacher role required. Submit a teacher-role request first.",
        )
    return actor


@router.get("/teacher-status")
async def get_teacher_status(request: Request):
    """Get caller's teacher-role status and latest request state."""
    actor = _get_request_actor(request)
    if not actor["user_id"]:
        raise HTTPException(status_code=401, detail="Missing user identity")

    is_teacher = await classroom_db.is_teacher(actor["user_id"])
    latest = await classroom_db.get_latest_teacher_role_request(actor["user_id"])
    return {
        "user_id": actor["user_id"],
        "is_admin": actor["is_admin"],
        "is_teacher": is_teacher,
        "latest_request": latest.to_dict() if latest else None,
    }


@router.post("/teacher-requests")
async def request_teacher_role(request: Request, purpose: str = ""):
    """Create a teacher-role request for the caller."""
    actor = _get_request_actor(request)
    if not actor["user_id"]:
        raise HTTPException(status_code=401, detail="Missing user identity")
    if actor["is_admin"]:
        raise HTTPException(status_code=400, detail="Admins already have teacher privileges")
    if await classroom_db.is_teacher(actor["user_id"]):
        raise HTTPException(status_code=400, detail="User is already an approved teacher")

    req = await classroom_db.create_teacher_role_request(
        user_id=actor["user_id"],
        user_name=actor["user_name"] or actor["user_id"],
        user_email=actor["user_email"],
        purpose=purpose,
    )
    return req.to_dict()


@router.get("/teacher-requests")
async def list_teacher_role_requests(
    request: Request,
    status: str = None,
    limit: int = 100,
    offset: int = 0,
):
    """Admin panel: list teacher-role requests."""
    await _require_admin(request)
    requests = await classroom_db.list_teacher_role_requests(
        status=status,
        limit=limit,
        offset=offset,
    )
    return {"requests": [r.to_dict() for r in requests]}


@router.post("/teacher-requests/{request_id}/approve")
async def approve_teacher_role_request(request_id: str, request: Request, note: str = ""):
    """Admin action: approve a teacher-role request."""
    actor = await _require_admin(request)
    reviewed = await classroom_db.review_teacher_role_request(
        request_id=request_id,
        approved=True,
        reviewed_by=actor["user_id"],
        note=note,
    )
    if not reviewed:
        raise HTTPException(status_code=404, detail="Teacher-role request not found")
    return reviewed.to_dict()


@router.post("/teacher-requests/{request_id}/reject")
async def reject_teacher_role_request(request_id: str, request: Request, note: str = ""):
    """Admin action: reject a teacher-role request."""
    actor = await _require_admin(request)
    reviewed = await classroom_db.review_teacher_role_request(
        request_id=request_id,
        approved=False,
        reviewed_by=actor["user_id"],
        note=note,
    )
    if not reviewed:
        raise HTTPException(status_code=404, detail="Teacher-role request not found")
    return reviewed.to_dict()


@router.post("/rooms")
async def create_room(
    request: Request,
    name: str = "Classroom",
    topic: str = None,
    is_permanent: bool = False,
    room_type: str = "teacher_driven",
    curriculum_chapter_id: str = None,
    curriculum_section_id: str = None,
):
    """Create a new classroom room. Only approved teachers/admins can create."""
    actor = await _require_teacher_or_admin(request)
    room = await room_manager.create_room(
        name=name,
        teacher_id=actor["user_id"],
        teacher_name=actor.get("user_name") or actor["user_id"],
        topic=topic,
        is_permanent=is_permanent,
        room_type=room_type,
        curriculum_chapter_id=curriculum_chapter_id,
        curriculum_section_id=curriculum_section_id,
    )
    return room.to_dict()


@router.get("/rooms")
async def list_rooms(request: Request):
    """List all active classroom rooms."""
    await _require_authenticated(request)
    return {"rooms": room_manager.list_rooms()}


@router.get("/rooms/{room_id}")
async def get_room(room_id: str, request: Request):
    """Get room details."""
    await _require_authenticated(request)
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room.to_dict()


@router.delete("/rooms/{room_id}")
async def delete_room(room_id: str, request: Request):
    """Delete a room.

    Rules:
      - Admins can delete any non-permanent room.
      - Teachers can delete only rooms they created.
    """
    actor = await _require_teacher_or_admin(request)
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    if not actor["is_admin"] and room.teacher_id != actor["user_id"]:
        raise HTTPException(status_code=403, detail="Only the room creator can delete this room")

    if not room_manager.is_room_deletable(room_id):
        raise HTTPException(status_code=403, detail="This room cannot be deleted")

    if await room_manager.delete_room(room_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=500, detail="Failed to delete room")


@router.post("/rooms/{room_id}/token")
async def manage_token(room_id: str, request: Request, action: str = "request", user_id: str = "", to_user_id: str = ""):
    """
    Manage speaker token.

    Actions:
        request — request the token (queues if unavailable)
        pass    — pass token to to_user_id (or next in queue)
        release — release token
    """
    await _require_authenticated(request)
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
async def list_sessions(request: Request, room_id: str = None, limit: int = 50, offset: int = 0):
    """List past classroom sessions."""
    await _require_authenticated(request)
    sessions = await classroom_db.list_sessions(room_id=room_id, limit=limit, offset=offset)
    return {"sessions": [s.to_dict() for s in sessions]}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    """Get a specific session with its messages."""
    await _require_authenticated(request)
    session = await classroom_db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = await classroom_db.get_messages(session_id)
    return {
        "session": session.to_dict(),
        "messages": [m.to_dict() for m in messages],
    }


@router.get("/sessions/{session_id}/summary")
async def get_session_summary(session_id: str, request: Request):
    """Get or generate session summary + quiz."""
    await _require_authenticated(request)
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
async def get_dashboard(request: Request, room_id: str = None):
    """Teacher dashboard stats."""
    await _require_authenticated(request)
    stats = await classroom_db.get_dashboard_stats(room_id=room_id)
    return stats


# ── User Profile Endpoints ──

@router.get("/users")
async def list_users(request: Request):
    """List all known user profiles."""
    await _require_authenticated(request)
    profiles = await classroom_db.list_user_profiles()
    return {"users": [p.to_dict() for p in profiles]}


@router.get("/users/{user_id}")
async def get_user(user_id: str, request: Request):
    """Get a user profile."""
    await _require_authenticated(request)
    profile = await classroom_db.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    # Also return which rooms they belong to
    memberships = await classroom_db.get_user_rooms(user_id)
    return {
        "profile": profile.to_dict(),
        "rooms": [m.to_dict() for m in memberships],
    }


@router.put("/users/{user_id}")
async def update_user(user_id: str, request: Request, display_name: str = None,
                      preferred_language: str = None, role: str = None):
    """Update a user profile."""
    await _require_authenticated(request)
    profile = await classroom_db.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    normalized_language = None
    if preferred_language is not None:
        normalized_language = _normalize_classroom_language(preferred_language)
        if not normalized_language:
            raise HTTPException(status_code=400, detail="Unsupported preferred_language. Allowed: en, hi, ta")
    await classroom_db.upsert_user_profile(
        user_id=user_id,
        display_name=display_name or profile.display_name,
        preferred_language=normalized_language or profile.preferred_language,
        role=role or profile.role,
    )
    updated = await classroom_db.get_user_profile(user_id)
    return updated.to_dict()


@router.post("/users")
async def create_user(user_id: str, request: Request, display_name: str = "",
                      preferred_language: str = "en", role: str = "student"):
    """Create or update a user profile."""
    await _require_authenticated(request)
    normalized_language = _normalize_classroom_language(preferred_language)
    if not normalized_language:
        raise HTTPException(status_code=400, detail="Unsupported preferred_language. Allowed: en, hi, ta")
    profile = await classroom_db.upsert_user_profile(
        user_id=user_id,
        display_name=display_name,
        preferred_language=normalized_language,
        role=role,
    )
    return profile.to_dict()


# ── Room Update Endpoint ──

@router.put("/rooms/{room_id}")
async def update_room(room_id: str, request: Request, name: str = None, topic: str = None,
                      teacher_id: str = None, is_permanent: bool = None,
                      room_type: str = None,
                      curriculum_chapter_id: str = None,
                      curriculum_section_id: str = None):
    """Update room configuration."""
    await _require_teacher_or_admin(request)
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    # Update in-memory
    if name is not None:
        room.name = name
    if topic is not None:
        room.topic = topic
        room.current_lesson_topic = topic
    if teacher_id is not None:
        room.teacher_id = teacher_id
    if is_permanent is not None:
        if is_permanent:
            room_manager._permanent_rooms.add(room_id)
        else:
            room_manager._permanent_rooms.discard(room_id)
    if room_type is not None and room_type in ("teacher_driven", "discussion"):
        room.room_type = room_type
    if curriculum_chapter_id is not None:
        room.curriculum_chapter_id = curriculum_chapter_id
    if curriculum_section_id is not None:
        room.curriculum_section_id = curriculum_section_id
        # Auto-set topic from section title for display
        from curriculum_manager import get_curriculum_manager
        cm = get_curriculum_manager()
        info = cm.get_section_info(curriculum_section_id)
        if info:
            topic_label = f"{info['chapter_title']}: {info['section_title']}" if info['chapter_title'] else info['section_title']
            room.topic = topic_label
            room.current_lesson_topic = topic_label

    # Persist to DB
    kwargs = {}
    if name is not None:
        kwargs["name"] = name
    if topic is not None or curriculum_section_id is not None:
        kwargs["topic"] = room.topic  # use the (possibly auto-set) topic
    if teacher_id is not None:
        kwargs["created_by"] = teacher_id
    if is_permanent is not None:
        kwargs["is_permanent"] = is_permanent
    if room_type is not None:
        kwargs["room_type"] = room.room_type
    if kwargs:
        await classroom_db.update_room(room_id, **kwargs)

    return room.to_dict()


# ── Curriculum Endpoints ──

@router.get("/curriculum/files")
async def list_curriculum_files(request: Request):
    """List available curriculum files with metadata."""
    await _require_authenticated(request)
    from curriculum_manager import get_curriculum_manager
    cm = get_curriculum_manager()
    return {"files": cm.list_files(), "available": cm.available}


@router.get("/curriculum/topics")
async def get_curriculum_topics(request: Request, filename: str = None):
    """Get browseable chapter -> section -> concepts tree for the topic picker.

    Optional query param `filename` to filter to a single curriculum file.
    """
    await _require_authenticated(request)
    from curriculum_manager import get_curriculum_manager
    cm = get_curriculum_manager()
    tree = cm.get_browseable_tree(filename)
    return {"tree": tree}


@router.get("/curriculum/search")
async def search_curriculum_concepts(request: Request, q: str, limit: int = 20):
    """Search concepts by name (case-insensitive substring match)."""
    await _require_authenticated(request)
    from curriculum_manager import get_curriculum_manager
    cm = get_curriculum_manager()
    results = cm.search_concepts(q, limit=limit)
    return {"results": results, "query": q}


@router.get("/curriculum/concept/{concept_name}")
async def get_curriculum_concept(concept_name: str, request: Request):
    """Get full concept detail from the concept registry."""
    await _require_authenticated(request)
    from curriculum_manager import get_curriculum_manager
    cm = get_curriculum_manager()
    detail = cm.get_concept_detail(concept_name)
    if not detail:
        raise HTTPException(status_code=404, detail="Concept not found")
    return detail


@router.get("/curriculum/context/{topic}")
async def get_curriculum_context(topic: str, request: Request, max_tokens: int = None, language: str = "english"):
    """Preview the curriculum context block that would be injected for a topic."""
    await _require_authenticated(request)
    from curriculum_manager import get_curriculum_manager
    cm = get_curriculum_manager()
    ctx = cm.get_context_for_topic(topic, max_tokens=max_tokens, language=language)
    if not ctx:
        raise HTTPException(status_code=404, detail="No curriculum found for this topic")
    return {"topic": topic, "context": ctx, "char_count": len(ctx)}


@router.get("/curriculum/section/{section_id}")
async def get_curriculum_section_info(section_id: str, request: Request):
    """Get section info (title, chapter title, concepts) for display."""
    await _require_authenticated(request)
    from curriculum_manager import get_curriculum_manager
    cm = get_curriculum_manager()
    info = cm.get_section_info(section_id)
    if not info:
        raise HTTPException(status_code=404, detail="Section not found")
    return info


# ── Room Members Endpoint ──

@router.get("/rooms/{room_id}/members")
async def get_room_members(room_id: str, request: Request):
    """Get persisted members of a room (includes offline users)."""
    await _require_authenticated(request)
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    # Get persisted members from DB
    db_members = await classroom_db.get_room_members(room_id)

    # Merge with live status
    result = []
    for m in db_members:
        d = m.to_dict()
        d["is_online"] = m.user_id in room.users
        d["is_speaker"] = room.speaker_id == m.user_id
        result.append(d)

    return {"members": result, "online_count": len(room.users)}


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
    _metrics_collector.session_start()
    session_start_time = time.time()

    room = room_manager.get_room(room_id)
    if not room:
        await websocket.send_json({"type": "error", "message": "Room not found"})
        await websocket.close()
        _metrics_collector.session_end({"room_id": room_id, "error": "room_not_found", "duration_s": 0})
        return

    # Wait for join message
    user = None
    is_first_join_for_user = False
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        data = json.loads(raw)

        if data.get("type") != "join":
            await websocket.send_json({"type": "error", "message": "Expected join message"})
            await websocket.close()
            return

        # ── JWT verification for WebSocket join ──
        # When AUTH_ENABLED, the join message must include a "token" field
        # containing a valid OpenWebUI JWT.  The user_id is extracted from
        # the verified JWT; any user_id in the message body is ignored.
        if AUTH_ENABLED:
            ws_token = data.get("token")
            if not ws_token:
                await websocket.send_json({"type": "error", "message": "JWT token required in join message"})
                await websocket.close()
                return
            jwt_payload = decode_openwebui_jwt(ws_token)
            if not jwt_payload or "id" not in jwt_payload:
                await websocket.send_json({"type": "error", "message": "Invalid or expired JWT token"})
                await websocket.close()
                return
            user_id = jwt_payload["id"]
            logger.info(f"[CLASSROOM] WS join authenticated via JWT: user_id={user_id}")
        else:
            user_id = data.get("user_id", str(uuid.uuid4())[:8])
        language = _normalize_classroom_language(data.get("language", "en"))
        if not language:
            await websocket.send_json({
                "type": "error",
                "message": "Unsupported language. Allowed: en, hi, ta",
            })
            await websocket.close()
            return
        name = data.get("name", f"User-{user_id[:4]}")
        mode = data.get("mode", "text_and_audio")  # "text_only" or "text_and_audio"
        if mode not in ("text_and_audio", "text_only"):
            mode = "text_and_audio"

        user = RoomUser(
            user_id=user_id,
            name=name,
            language=language,
            websocket=websocket,
            mode=mode,
        )

        # Determine first-ever join in this room before upsert_room_member runs.
        try:
            existing_member = await classroom_db.get_room_member(room_id, user_id)
            is_first_join_for_user = existing_member is None
        except Exception as e:
            logger.warning(f"[CLASSROOM] Could not check existing membership for greeting: {e}")
            is_first_join_for_user = False

        # Legacy fallback: if no teacher is set, only the first user in an
        # empty room becomes teacher. Never rotate teacher on later joins.
        if room.teacher_id is None and len(room.users) == 0:
            room.teacher_id = user_id
            logger.info(f"[CLASSROOM] {name} ({user_id}) is now the teacher of room {room_id}")

        # Add to room (does NOT auto-assign token yet)
        # Note: add_user may update user.name and user.language from DB profile
        if not await room_manager.add_user(room_id, user):
            await websocket.send_json({"type": "error", "message": "Failed to join room"})
            await websocket.close()
            return

        # Send room state + recent chat history to the new user FIRST
        recent_messages = await room_manager.get_recent_room_messages(
            room_id=room_id,
            user_language=user.language,
            limit=60,
        )
        await websocket.send_json({
            "type": "joined",
            "room": room.to_dict(),
            "you": {
                "user_id": user_id,
                "name": user.name,
                "language": user.language,
                "mode": user.mode,
                "is_speaker": user.is_speaker,
                "is_teacher": user_id == room.teacher_id,
            },
            "recent_messages": recent_messages,
        })

        # Auto-assign speaker token BEFORE greeting so the client knows its
        # role immediately.  The greeting can take 5-20 s (translation + TTS)
        # and must not delay token delivery.
        #
        # Capture the current user count *now* so that concurrent joins that
        # happen while the greeting is running don't invalidate the "first
        # user" check inside finalize_join.
        await room_manager.finalize_join(room_id, user_id)

        # One-time per-user-per-room greeting.
        # The TEXT part (translate + bot_text_complete) runs synchronously so
        # it arrives before the message loop — clients can drain it during
        # join and won't confuse it with an LLM response.
        # The AUDIO part (TTS streaming) is slow and runs as a background
        # task.  The per-user _audio_lock inside _stream_greeting_audio
        # ensures greeting audio never interleaves with response audio.
        if is_first_join_for_user:
            try:
                audio_coro = await room_manager.send_first_join_greeting(room, user)
                if audio_coro is not None:
                    asyncio.create_task(audio_coro)
            except Exception as e:
                logger.warning(f"[CLASSROOM] Greeting text failed for {user.name}: {e}")

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
                # Speaker typed a text question — query LLM, stream to speaker AND listeners
                text = data.get("text", "").strip()
                room = room_manager.get_room(room_id)
                if text and room and room.speaker_id == user.user_id:
                    if room.room_type == "discussion":
                        room_manager._cancel_discussion_auto_release(room)
                    logger.info(f"[CLASSROOM] Text message from {user.name}: {text[:60]}")

                    # 1. Broadcast the speaker's question to listeners (translated).
                    #    Fire-and-forget so the LLM call starts immediately
                    #    and the message loop isn't blocked by listener TTS.
                    asyncio.create_task(
                        room_manager.broadcast_transcription(
                            room=room,
                            speaker_id=user.user_id,
                            text=text,
                            language=user.language,
                        )
                    )

                    # 2. Query LLM — streams tokens to speaker AND listeners sentence-by-sentence
                    #    (ask_llm handles listener delivery inline, no separate broadcast needed)
                    await room_manager.ask_llm(
                        text, websocket, room=room, stream_to_listeners=True,
                    )
                else:
                    await websocket.send_json({"type": "error", "message": "Only the speaker can send messages"})

            elif msg_type == "speaker_transcript":
                # Speaker voice transcript (final) forwarded by frontend.
                # This is broadcast-only; LLM response already comes from voice pipeline.
                text = data.get("text", "").strip()
                room = room_manager.get_room(room_id)
                if text and room and room.speaker_id == user.user_id:
                    if room.room_type == "discussion":
                        room_manager._cancel_discussion_auto_release(room)
                    logger.info(f"[CLASSROOM] Speaker transcript from {user.name}: {text[:60]}")
                    asyncio.create_task(
                        room_manager.broadcast_transcription(
                            room=room,
                            speaker_id=user.user_id,
                            text=text,
                            language=user.language,
                        )
                    )

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

                # ── Non-LLM teacher actions (KICK, PROMOTE_TEACHER) ──
                if action == "KICK":
                    target_user_id = payload
                    if not target_user_id or target_user_id == user.user_id:
                        await websocket.send_json({"type": "error", "message": "Invalid kick target"})
                        continue
                    target = room.users.get(target_user_id)
                    if not target:
                        await websocket.send_json({"type": "error", "message": "User not in room"})
                        continue
                    target_name = target.name
                    # Close the kicked user's WebSocket (triggers remove_user in finally block)
                    try:
                        await room_manager._send_json(target.websocket, {
                            "type": "kicked",
                            "message": f"You were removed from the room by the teacher"
                        })
                        await target.websocket.close(code=4001, reason="Kicked by teacher")
                    except Exception as kick_err:
                        logger.debug(f"[CLASSROOM] Error closing kicked user WS: {kick_err}")
                    logger.info(f"[CLASSROOM] Teacher {user.name} kicked {target_name} from {room_id}")
                    await websocket.send_json({"type": "teacher_action_result", "action": "KICK", "success": True, "target": target_name})
                    continue

                elif action == "PROMOTE_TEACHER":
                    target_user_id = payload
                    if not target_user_id:
                        await websocket.send_json({"type": "error", "message": "No user specified for promotion"})
                        continue
                    target = room.users.get(target_user_id)
                    if not target:
                        await websocket.send_json({"type": "error", "message": "User not in room"})
                        continue
                    old_teacher_id = room.teacher_id
                    room.teacher_id = target_user_id
                    # Persist to DB
                    try:
                        await classroom_db.update_room(room_id, created_by=target_user_id)
                        # Update roles in room_members table
                        await classroom_db.upsert_room_member(room_id, target_user_id, target.name, target.language, role="teacher", mode=target.mode)
                        if old_teacher_id and old_teacher_id in room.users:
                            old_teacher = room.users[old_teacher_id]
                            await classroom_db.upsert_room_member(room_id, old_teacher_id, old_teacher.name, old_teacher.language, role="student", mode=old_teacher.mode)
                    except Exception as e:
                        logger.error(f"[CLASSROOM] Failed to persist teacher change: {e}")
                    # Broadcast teacher change to all users
                    await room_manager._broadcast_json(room, {
                        "type": "teacher_changed",
                        "teacher_id": target_user_id,
                        "teacher_name": target.name,
                    })
                    logger.info(f"[CLASSROOM] Teacher changed: {user.name} -> {target.name} in room {room_id}")
                    continue

                # ── LLM-based teacher actions ──
                # Build the teacher action command
                if action == "SET_TOPIC":
                    teacher_cmd = f"[TEACHER_ACTION: SET_TOPIC {payload}]"
                    room.current_lesson_topic = payload
                    room.topic = payload
                    # Persist topic to DB
                    try:
                        await classroom_db.update_room(room_id, topic=payload)
                    except Exception as e:
                        logger.error(f"[CLASSROOM] Failed to persist topic: {e}")
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

                # Broadcast a neutral message to listeners (fire-and-forget)
                asyncio.create_task(
                    room_manager.broadcast_transcription(
                        room=room,
                        speaker_id=user.user_id,
                        text=display_text,
                        language="en",
                    )
                )

                # Query LLM with the teacher command — streams to speaker AND listeners
                await room_manager.ask_llm(
                    teacher_cmd, websocket, room=room, stream_to_listeners=True,
                )

            elif msg_type == "set_mode":
                # Switch between text_only and text_and_audio at runtime
                new_mode = data.get("mode", "text_only")
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

            elif msg_type in {"leave", "exit_room"}:
                # Explicit leave from client: close socket now so user is removed
                # immediately in finally block (no wait for network disconnect).
                logger.info(f"[CLASSROOM] User {user.user_id} requested leave from room {room_id}")
                await websocket.close(code=1000, reason="left_room")
                break

            else:
                logger.debug(f"[CLASSROOM] Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        logger.info(f"[CLASSROOM] User {user.user_id} disconnected from room {room_id}")
    except Exception as e:
        logger.error(f"[CLASSROOM] WebSocket error for {user.user_id}: {e}")
    finally:
        session_duration = round(time.time() - session_start_time, 1)
        session_summary = {
            "room_id": room_id,
            "user_id": user.user_id if user else "unknown",
            "user_name": user.name if user else "unknown",
            "language": user.language if user else "unknown",
            "mode": user.mode if user else "unknown",
            "was_speaker": user.is_speaker if user else False,
            "duration_s": session_duration,
            "disconnected_at": time.time(),
        }
        _metrics_collector.session_end(session_summary)
        logger.info(
            f"[METRICS][CLASSROOM] session_end | user={session_summary['user_name']} "
            f"| room={room_id} | duration={session_duration}s | "
            f"speaker={session_summary['was_speaker']} | mode={session_summary['mode']}"
        )
        if user:
            await room_manager.remove_user(room_id, user.user_id)
