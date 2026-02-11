"""
Classroom Database — SQLite persistence for session history, hand raises, reactions.

Provides async-compatible SQLite storage so that:
- Classroom conversations persist across container restarts
- Session summaries and quizzes can reference past conversations
- Teacher dashboard can query historical data
- Recordings metadata can be stored

Schema:
    sessions        — one row per classroom session (room activation period)
    messages        — every transcription + bot response, with translations
    hand_raises     — hand-raise events within a session
    reactions       — emoji reactions on messages
    session_summaries — AI-generated summaries for completed sessions
    recordings      — audio recording metadata

Thread safety:
    Uses aiosqlite for async I/O. A single database file is shared across
    the process. The module exposes a singleton `db` instance.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("CLASSROOM_DB_PATH", "/app/data/classroom.db")

# ─────────────────────────────────────────────────────────────────────
# Data classes for query results
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SessionRecord:
    id: str
    room_id: str
    room_name: str
    started_at: float
    ended_at: Optional[float] = None
    speaker_ids: str = "[]"  # JSON list
    participant_count: int = 0
    message_count: int = 0
    summary: Optional[str] = None
    quiz_json: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "room_id": self.room_id,
            "room_name": self.room_name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "speaker_ids": json.loads(self.speaker_ids) if self.speaker_ids else [],
            "participant_count": self.participant_count,
            "message_count": self.message_count,
            "summary": self.summary,
            "quiz": json.loads(self.quiz_json) if self.quiz_json else None,
        }


@dataclass
class MessageRecord:
    id: str
    session_id: str
    room_id: str
    role: str  # "user" | "assistant"
    speaker_id: Optional[str] = None
    speaker_name: Optional[str] = None
    content: str = ""
    original_language: str = "en"
    translations: str = "{}"  # JSON dict {lang_code: translated_text}
    timestamp: float = 0.0
    reaction_counts: str = "{}"  # JSON dict {emoji: count}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "room_id": self.room_id,
            "role": self.role,
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "content": self.content,
            "original_language": self.original_language,
            "translations": json.loads(self.translations) if self.translations else {},
            "timestamp": self.timestamp,
            "reaction_counts": json.loads(self.reaction_counts) if self.reaction_counts else {},
        }


@dataclass
class HandRaiseRecord:
    id: str
    session_id: str
    room_id: str
    user_id: str
    user_name: str
    status: str  # "pending" | "acknowledged" | "dismissed"
    question_preview: Optional[str] = None
    raised_at: float = 0.0
    resolved_at: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReactionRecord:
    id: str
    message_id: str
    session_id: str
    user_id: str
    emoji: str
    timestamp: float = 0.0


@dataclass
class RecordingRecord:
    id: str
    session_id: str
    room_id: str
    filename: str
    duration_secs: float = 0.0
    size_bytes: int = 0
    started_at: float = 0.0
    ended_at: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── Persistence data classes ──────────────────────────────────────

@dataclass
class RoomRecord:
    """Persisted room configuration."""
    room_id: str
    name: str
    topic: Optional[str] = None
    created_by: Optional[str] = None  # teacher user_id
    created_by_name: Optional[str] = None  # teacher display name
    settings_json: str = "{}"
    is_permanent: bool = False
    room_type: str = "teacher_driven"  # "teacher_driven" or "discussion"
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["settings"] = json.loads(self.settings_json) if self.settings_json else {}
        return d


@dataclass
class UserProfileRecord:
    """Persisted user profile — remembers name, language, preferences."""
    user_id: str
    display_name: str
    preferred_language: str = "en"
    role: str = "student"  # "teacher" | "student"
    settings_json: str = "{}"
    created_at: float = 0.0
    last_seen: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["settings"] = json.loads(self.settings_json) if self.settings_json else {}
        return d


@dataclass
class RoomMemberRecord:
    """Persisted room membership — who belongs to which room."""
    room_id: str
    user_id: str
    display_name: str
    language: str = "en"
    role: str = "student"  # "teacher" | "student" | "observer"
    mode: str = "text_only"  # "text_only" | "text_and_audio"
    joined_at: float = 0.0
    last_active: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────
# Database Manager
# ─────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    room_id TEXT NOT NULL,
    room_name TEXT NOT NULL DEFAULT '',
    started_at REAL NOT NULL,
    ended_at REAL,
    speaker_ids TEXT DEFAULT '[]',
    participant_count INTEGER DEFAULT 0,
    message_count INTEGER DEFAULT 0,
    summary TEXT,
    quiz_json TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    room_id TEXT NOT NULL,
    role TEXT NOT NULL,
    speaker_id TEXT,
    speaker_name TEXT,
    content TEXT NOT NULL DEFAULT '',
    original_language TEXT DEFAULT 'en',
    translations TEXT DEFAULT '{}',
    timestamp REAL NOT NULL,
    reaction_counts TEXT DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS hand_raises (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    room_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    user_name TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    question_preview TEXT,
    raised_at REAL NOT NULL,
    resolved_at REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS reactions (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    emoji TEXT NOT NULL,
    timestamp REAL NOT NULL,
    FOREIGN KEY (message_id) REFERENCES messages(id),
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    UNIQUE(message_id, user_id, emoji)
);

CREATE TABLE IF NOT EXISTS recordings (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    room_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    duration_secs REAL DEFAULT 0.0,
    size_bytes INTEGER DEFAULT 0,
    started_at REAL NOT NULL,
    ended_at REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_room ON messages(room_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_hand_raises_session ON hand_raises(session_id);
CREATE INDEX IF NOT EXISTS idx_hand_raises_status ON hand_raises(status);
CREATE INDEX IF NOT EXISTS idx_reactions_message ON reactions(message_id);
CREATE INDEX IF NOT EXISTS idx_sessions_room ON sessions(room_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_recordings_session ON recordings(session_id);

-- ── Persistence tables (rooms, users, memberships) ──

CREATE TABLE IF NOT EXISTS rooms (
    room_id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT 'Classroom',
    topic TEXT,
    created_by TEXT,
    created_by_name TEXT,
    settings_json TEXT DEFAULT '{}',
    is_permanent INTEGER DEFAULT 0,
    room_type TEXT DEFAULT 'teacher_driven',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    preferred_language TEXT DEFAULT 'en',
    role TEXT DEFAULT 'student',
    settings_json TEXT DEFAULT '{}',
    created_at REAL NOT NULL,
    last_seen REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS room_members (
    room_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    language TEXT DEFAULT 'en',
    role TEXT DEFAULT 'student',
    mode TEXT DEFAULT 'text_only',
    joined_at REAL NOT NULL,
    last_active REAL NOT NULL,
    PRIMARY KEY (room_id, user_id),
    FOREIGN KEY (room_id) REFERENCES rooms(room_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_room_members_room ON room_members(room_id);
CREATE INDEX IF NOT EXISTS idx_room_members_user ON room_members(user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_name ON user_profiles(display_name);
"""


class ClassroomDB:
    """Async SQLite database for classroom persistence."""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._initialized = False

    async def init(self):
        """Initialize the database connection and create schema."""
        if self._initialized:
            return

        # Ensure directory exists
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row

        # Enable WAL mode for concurrent reads
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        # Create schema
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()

        # ── Migrations for existing databases ──
        await self._run_migrations()

        self._initialized = True
        logger.info(f"[DB] Classroom database initialized at {self._db_path}")

    async def _run_migrations(self):
        """Run safe ALTER TABLE migrations for columns added after initial schema."""
        migrations = [
            ("rooms", "room_type", "ALTER TABLE rooms ADD COLUMN room_type TEXT DEFAULT 'teacher_driven'"),
        ]
        for table, column, sql in migrations:
            try:
                # Check if column exists
                async with self._db.execute(f"PRAGMA table_info({table})") as cursor:
                    cols = [row[1] for row in await cursor.fetchall()]
                if column not in cols:
                    await self._db.execute(sql)
                    await self._db.commit()
                    logger.info(f"[DB] Migration: added {table}.{column}")
            except Exception as e:
                logger.warning(f"[DB] Migration skipped ({table}.{column}): {e}")

    async def close(self):
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
            self._initialized = False
            logger.info("[DB] Database connection closed")

    # ── Sessions ──────────────────────────────────────────────────

    async def create_session(self, room_id: str, room_name: str) -> SessionRecord:
        """Start a new session for a room."""
        session = SessionRecord(
            id=str(uuid.uuid4())[:12],
            room_id=room_id,
            room_name=room_name,
            started_at=time.time(),
        )
        await self._db.execute(
            """INSERT INTO sessions (id, room_id, room_name, started_at, speaker_ids, participant_count, message_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session.id, session.room_id, session.room_name, session.started_at,
             session.speaker_ids, session.participant_count, session.message_count),
        )
        await self._db.commit()
        logger.info(f"[DB] Session created: {session.id} for room {room_id}")
        return session

    async def end_session(self, session_id: str):
        """End a session (set ended_at timestamp)."""
        await self._db.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (time.time(), session_id),
        )
        await self._db.commit()
        logger.info(f"[DB] Session ended: {session_id}")

    async def update_session_stats(self, session_id: str, participant_count: int = None,
                                     message_count: int = None, speaker_ids: List[str] = None):
        """Update session statistics."""
        updates = []
        params = []
        if participant_count is not None:
            updates.append("participant_count = ?")
            params.append(participant_count)
        if message_count is not None:
            updates.append("message_count = ?")
            params.append(message_count)
        if speaker_ids is not None:
            updates.append("speaker_ids = ?")
            params.append(json.dumps(speaker_ids))

        if updates:
            params.append(session_id)
            await self._db.execute(
                f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            await self._db.commit()

    async def set_session_summary(self, session_id: str, summary: str, quiz_json: str = None):
        """Save an AI-generated summary (and optional quiz) for a session."""
        await self._db.execute(
            "UPDATE sessions SET summary = ?, quiz_json = ? WHERE id = ?",
            (summary, quiz_json, session_id),
        )
        await self._db.commit()
        logger.info(f"[DB] Summary saved for session {session_id}")

    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Get a session by ID."""
        async with self._db.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return SessionRecord(**dict(row))
        return None

    async def list_sessions(self, room_id: str = None, limit: int = 50,
                             offset: int = 0) -> List[SessionRecord]:
        """List sessions, optionally filtered by room."""
        if room_id:
            query = "SELECT * FROM sessions WHERE room_id = ? ORDER BY started_at DESC LIMIT ? OFFSET ?"
            params = (room_id, limit, offset)
        else:
            query = "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ? OFFSET ?"
            params = (limit, offset)

        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [SessionRecord(**dict(row)) for row in rows]

    async def get_active_session(self, room_id: str) -> Optional[SessionRecord]:
        """Get the currently active (non-ended) session for a room."""
        async with self._db.execute(
            "SELECT * FROM sessions WHERE room_id = ? AND ended_at IS NULL ORDER BY started_at DESC LIMIT 1",
            (room_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return SessionRecord(**dict(row))
        return None

    # ── Messages ──────────────────────────────────────────────────

    async def save_message(
        self,
        session_id: str,
        room_id: str,
        role: str,
        content: str,
        speaker_id: str = None,
        speaker_name: str = None,
        original_language: str = "en",
        translations: dict = None,
    ) -> MessageRecord:
        """Save a message to the database."""
        msg = MessageRecord(
            id=str(uuid.uuid4())[:12],
            session_id=session_id,
            room_id=room_id,
            role=role,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            content=content,
            original_language=original_language,
            translations=json.dumps(translations or {}),
            timestamp=time.time(),
        )
        await self._db.execute(
            """INSERT INTO messages (id, session_id, room_id, role, speaker_id, speaker_name,
               content, original_language, translations, timestamp, reaction_counts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (msg.id, msg.session_id, msg.room_id, msg.role, msg.speaker_id,
             msg.speaker_name, msg.content, msg.original_language, msg.translations,
             msg.timestamp, msg.reaction_counts),
        )
        await self._db.commit()

        # Update session message count
        await self._db.execute(
            "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
            (session_id,),
        )
        await self._db.commit()

        return msg

    async def get_messages(self, session_id: str, limit: int = 200,
                            offset: int = 0) -> List[MessageRecord]:
        """Get messages for a session."""
        async with self._db.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC LIMIT ? OFFSET ?",
            (session_id, limit, offset),
        ) as cursor:
            rows = await cursor.fetchall()
            return [MessageRecord(**dict(row)) for row in rows]

    async def get_messages_by_room(self, room_id: str, limit: int = 200) -> List[MessageRecord]:
        """Get recent messages for a room across all sessions."""
        async with self._db.execute(
            "SELECT * FROM messages WHERE room_id = ? ORDER BY timestamp DESC LIMIT ?",
            (room_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
            return [MessageRecord(**dict(row)) for row in reversed(rows)]

    # ── Hand Raises ───────────────────────────────────────────────

    async def create_hand_raise(
        self, session_id: str, room_id: str, user_id: str,
        user_name: str, question_preview: str = None,
    ) -> HandRaiseRecord:
        """Record a hand raise."""
        hr = HandRaiseRecord(
            id=str(uuid.uuid4())[:12],
            session_id=session_id,
            room_id=room_id,
            user_id=user_id,
            user_name=user_name,
            status="pending",
            question_preview=question_preview,
            raised_at=time.time(),
        )
        await self._db.execute(
            """INSERT INTO hand_raises (id, session_id, room_id, user_id, user_name,
               status, question_preview, raised_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (hr.id, hr.session_id, hr.room_id, hr.user_id, hr.user_name,
             hr.status, hr.question_preview, hr.raised_at),
        )
        await self._db.commit()
        return hr

    async def resolve_hand_raise(self, raise_id: str, status: str = "acknowledged"):
        """Resolve a hand raise (acknowledged / dismissed)."""
        await self._db.execute(
            "UPDATE hand_raises SET status = ?, resolved_at = ? WHERE id = ?",
            (status, time.time(), raise_id),
        )
        await self._db.commit()

    async def get_pending_hand_raises(self, room_id: str) -> List[HandRaiseRecord]:
        """Get all pending hand raises for a room."""
        async with self._db.execute(
            "SELECT * FROM hand_raises WHERE room_id = ? AND status = 'pending' ORDER BY raised_at ASC",
            (room_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [HandRaiseRecord(**dict(row)) for row in rows]

    async def get_session_hand_raises(self, session_id: str) -> List[HandRaiseRecord]:
        """Get all hand raises for a session."""
        async with self._db.execute(
            "SELECT * FROM hand_raises WHERE session_id = ? ORDER BY raised_at ASC",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [HandRaiseRecord(**dict(row)) for row in rows]

    # ── Reactions ─────────────────────────────────────────────────

    async def add_reaction(
        self, message_id: str, session_id: str, user_id: str, emoji: str
    ) -> bool:
        """Add a reaction to a message. Returns True if new, False if duplicate."""
        try:
            reaction_id = str(uuid.uuid4())[:12]
            await self._db.execute(
                """INSERT OR IGNORE INTO reactions (id, message_id, session_id, user_id, emoji, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (reaction_id, message_id, session_id, user_id, emoji, time.time()),
            )
            # Update reaction counts on the message
            await self._update_reaction_counts(message_id)
            await self._db.commit()
            return True
        except Exception as e:
            logger.error(f"[DB] Failed to add reaction: {e}")
            return False

    async def remove_reaction(self, message_id: str, user_id: str, emoji: str) -> bool:
        """Remove a reaction from a message."""
        try:
            await self._db.execute(
                "DELETE FROM reactions WHERE message_id = ? AND user_id = ? AND emoji = ?",
                (message_id, user_id, emoji),
            )
            await self._update_reaction_counts(message_id)
            await self._db.commit()
            return True
        except Exception as e:
            logger.error(f"[DB] Failed to remove reaction: {e}")
            return False

    async def _update_reaction_counts(self, message_id: str):
        """Recalculate and update the reaction_counts JSON on a message."""
        async with self._db.execute(
            "SELECT emoji, COUNT(*) as cnt FROM reactions WHERE message_id = ? GROUP BY emoji",
            (message_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            counts = {row["emoji"]: row["cnt"] for row in rows}
            await self._db.execute(
                "UPDATE messages SET reaction_counts = ? WHERE id = ?",
                (json.dumps(counts), message_id),
            )

    async def get_message_reactions(self, message_id: str) -> List[ReactionRecord]:
        """Get all reactions for a message."""
        async with self._db.execute(
            "SELECT * FROM reactions WHERE message_id = ? ORDER BY timestamp ASC",
            (message_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [ReactionRecord(**dict(row)) for row in rows]

    # ── Recordings ────────────────────────────────────────────────

    async def create_recording(
        self, session_id: str, room_id: str, filename: str
    ) -> RecordingRecord:
        """Start a recording."""
        rec = RecordingRecord(
            id=str(uuid.uuid4())[:12],
            session_id=session_id,
            room_id=room_id,
            filename=filename,
            started_at=time.time(),
        )
        await self._db.execute(
            """INSERT INTO recordings (id, session_id, room_id, filename, started_at)
               VALUES (?, ?, ?, ?, ?)""",
            (rec.id, rec.session_id, rec.room_id, rec.filename, rec.started_at),
        )
        await self._db.commit()
        return rec

    async def end_recording(self, recording_id: str, duration_secs: float, size_bytes: int):
        """End a recording with final stats."""
        await self._db.execute(
            "UPDATE recordings SET ended_at = ?, duration_secs = ?, size_bytes = ? WHERE id = ?",
            (time.time(), duration_secs, size_bytes, recording_id),
        )
        await self._db.commit()

    async def get_session_recordings(self, session_id: str) -> List[RecordingRecord]:
        """Get recordings for a session."""
        async with self._db.execute(
            "SELECT * FROM recordings WHERE session_id = ? ORDER BY started_at ASC",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [RecordingRecord(**dict(row)) for row in rows]

    # ── Rooms (Persistence) ────────────────────────────────────────

    async def save_room(self, room_id: str, name: str, topic: str = None,
                        created_by: str = None, created_by_name: str = None,
                        settings: dict = None, is_permanent: bool = False,
                        room_type: str = "teacher_driven") -> RoomRecord:
        """Upsert a room configuration."""
        now = time.time()
        rec = RoomRecord(
            room_id=room_id, name=name, topic=topic,
            created_by=created_by, created_by_name=created_by_name,
            settings_json=json.dumps(settings or {}),
            is_permanent=is_permanent, room_type=room_type,
            created_at=now, updated_at=now,
        )
        await self._db.execute(
            """INSERT INTO rooms (room_id, name, topic, created_by, created_by_name,
               settings_json, is_permanent, room_type, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(room_id) DO UPDATE SET
                   name=excluded.name, topic=excluded.topic,
                   created_by=excluded.created_by, created_by_name=excluded.created_by_name,
                   settings_json=excluded.settings_json, is_permanent=excluded.is_permanent,
                   room_type=excluded.room_type, updated_at=excluded.updated_at""",
            (rec.room_id, rec.name, rec.topic, rec.created_by, rec.created_by_name,
             rec.settings_json, int(rec.is_permanent), rec.room_type, rec.created_at, rec.updated_at),
        )
        await self._db.commit()
        logger.info(f"[DB] Room saved: {room_id} ({name}) type={room_type}")
        return rec

    async def update_room(self, room_id: str, **kwargs):
        """Update specific room fields."""
        allowed = {"name", "topic", "created_by", "created_by_name", "settings_json", "is_permanent", "room_type"}
        updates = []
        params = []
        for k, v in kwargs.items():
            if k in allowed:
                if k == "is_permanent":
                    v = int(v)
                updates.append(f"{k} = ?")
                params.append(v)
        if updates:
            updates.append("updated_at = ?")
            params.append(time.time())
            params.append(room_id)
            await self._db.execute(
                f"UPDATE rooms SET {', '.join(updates)} WHERE room_id = ?", params
            )
            await self._db.commit()

    async def get_room(self, room_id: str) -> Optional[RoomRecord]:
        """Get a persisted room by ID."""
        async with self._db.execute(
            "SELECT * FROM rooms WHERE room_id = ?", (room_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                d = dict(row)
                d["is_permanent"] = bool(d.get("is_permanent", 0))
                return RoomRecord(**d)
        return None

    async def list_rooms(self) -> List[RoomRecord]:
        """List all persisted rooms."""
        async with self._db.execute(
            "SELECT * FROM rooms ORDER BY created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["is_permanent"] = bool(d.get("is_permanent", 0))
                result.append(RoomRecord(**d))
            return result

    async def delete_room(self, room_id: str):
        """Delete a room and its memberships from the DB."""
        await self._db.execute("DELETE FROM room_members WHERE room_id = ?", (room_id,))
        await self._db.execute("DELETE FROM rooms WHERE room_id = ?", (room_id,))
        await self._db.commit()
        logger.info(f"[DB] Room deleted: {room_id}")

    # ── User Profiles (Persistence) ──────────────────────────────

    async def upsert_user_profile(self, user_id: str, display_name: str,
                                   preferred_language: str = "en",
                                   role: str = "student",
                                   settings: dict = None) -> UserProfileRecord:
        """Create or update a user profile."""
        now = time.time()
        rec = UserProfileRecord(
            user_id=user_id, display_name=display_name,
            preferred_language=preferred_language, role=role,
            settings_json=json.dumps(settings or {}),
            created_at=now, last_seen=now,
        )
        await self._db.execute(
            """INSERT INTO user_profiles (user_id, display_name, preferred_language, role,
               settings_json, created_at, last_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   display_name=excluded.display_name,
                   preferred_language=excluded.preferred_language,
                   role=excluded.role,
                   settings_json=excluded.settings_json,
                   last_seen=excluded.last_seen""",
            (rec.user_id, rec.display_name, rec.preferred_language, rec.role,
             rec.settings_json, rec.created_at, rec.last_seen),
        )
        await self._db.commit()
        return rec

    async def get_user_profile(self, user_id: str) -> Optional[UserProfileRecord]:
        """Get a user profile by ID."""
        async with self._db.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return UserProfileRecord(**dict(row))
        return None

    async def list_user_profiles(self) -> List[UserProfileRecord]:
        """List all user profiles."""
        async with self._db.execute(
            "SELECT * FROM user_profiles ORDER BY last_seen DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [UserProfileRecord(**dict(row)) for row in rows]

    async def touch_user(self, user_id: str):
        """Update last_seen timestamp."""
        await self._db.execute(
            "UPDATE user_profiles SET last_seen = ? WHERE user_id = ?",
            (time.time(), user_id),
        )
        await self._db.commit()

    # ── Room Members (Persistence) ───────────────────────────────

    async def upsert_room_member(self, room_id: str, user_id: str,
                                  display_name: str, language: str = "en",
                                  role: str = "student",
                                  mode: str = "text_only") -> RoomMemberRecord:
        """Add or update a room member."""
        now = time.time()
        rec = RoomMemberRecord(
            room_id=room_id, user_id=user_id, display_name=display_name,
            language=language, role=role, mode=mode,
            joined_at=now, last_active=now,
        )
        await self._db.execute(
            """INSERT INTO room_members (room_id, user_id, display_name, language, role, mode,
               joined_at, last_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(room_id, user_id) DO UPDATE SET
                   display_name=excluded.display_name,
                   language=excluded.language,
                   role=excluded.role,
                   mode=excluded.mode,
                   last_active=excluded.last_active""",
            (rec.room_id, rec.user_id, rec.display_name, rec.language,
             rec.role, rec.mode, rec.joined_at, rec.last_active),
        )
        await self._db.commit()
        return rec

    async def get_room_members(self, room_id: str) -> List[RoomMemberRecord]:
        """Get all members of a room."""
        async with self._db.execute(
            "SELECT * FROM room_members WHERE room_id = ? ORDER BY joined_at ASC",
            (room_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [RoomMemberRecord(**dict(row)) for row in rows]

    async def get_user_rooms(self, user_id: str) -> List[RoomMemberRecord]:
        """Get all rooms a user belongs to."""
        async with self._db.execute(
            "SELECT * FROM room_members WHERE user_id = ? ORDER BY last_active DESC",
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [RoomMemberRecord(**dict(row)) for row in rows]

    async def remove_room_member(self, room_id: str, user_id: str):
        """Remove a member from a room."""
        await self._db.execute(
            "DELETE FROM room_members WHERE room_id = ? AND user_id = ?",
            (room_id, user_id),
        )
        await self._db.commit()

    async def touch_room_member(self, room_id: str, user_id: str):
        """Update last_active timestamp for a room member."""
        await self._db.execute(
            "UPDATE room_members SET last_active = ? WHERE room_id = ? AND user_id = ?",
            (time.time(), room_id, user_id),
        )
        await self._db.commit()

    # ── Dashboard Queries ─────────────────────────────────────────

    async def get_dashboard_stats(self, room_id: str = None) -> dict:
        """Get aggregate stats for the teacher dashboard."""
        where = "WHERE room_id = ?" if room_id else ""
        params = (room_id,) if room_id else ()

        stats = {}

        # Total sessions
        async with self._db.execute(
            f"SELECT COUNT(*) as cnt FROM sessions {where}", params
        ) as cursor:
            row = await cursor.fetchone()
            stats["total_sessions"] = row["cnt"]

        # Total messages
        async with self._db.execute(
            f"SELECT COUNT(*) as cnt FROM messages {where}", params
        ) as cursor:
            row = await cursor.fetchone()
            stats["total_messages"] = row["cnt"]

        # Total unique participants
        async with self._db.execute(
            f"SELECT COUNT(DISTINCT speaker_id) as cnt FROM messages {where} AND speaker_id IS NOT NULL" if room_id
            else "SELECT COUNT(DISTINCT speaker_id) as cnt FROM messages WHERE speaker_id IS NOT NULL",
            params if room_id else (),
        ) as cursor:
            row = await cursor.fetchone()
            stats["unique_participants"] = row["cnt"]

        # Avg session duration (for completed sessions)
        async with self._db.execute(
            f"SELECT AVG(ended_at - started_at) as avg_dur FROM sessions {where} AND ended_at IS NOT NULL" if room_id
            else "SELECT AVG(ended_at - started_at) as avg_dur FROM sessions WHERE ended_at IS NOT NULL",
            params if room_id else (),
        ) as cursor:
            row = await cursor.fetchone()
            stats["avg_session_duration_secs"] = round(row["avg_dur"] or 0, 1)

        # Total hand raises
        async with self._db.execute(
            f"SELECT COUNT(*) as cnt FROM hand_raises {where}", params
        ) as cursor:
            row = await cursor.fetchone()
            stats["total_hand_raises"] = row["cnt"]

        # Total reactions
        async with self._db.execute(
            "SELECT COUNT(*) as cnt FROM reactions", ()
        ) as cursor:
            row = await cursor.fetchone()
            stats["total_reactions"] = row["cnt"]

        # Recent sessions (last 5)
        recent = await self.list_sessions(room_id=room_id, limit=5)
        stats["recent_sessions"] = [s.to_dict() for s in recent]

        return stats


# ─────────────────────────────────────────────────────────────────────
# Singleton instance
# ─────────────────────────────────────────────────────────────────────

db = ClassroomDB()
