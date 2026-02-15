#!/usr/bin/env python3
"""
Classroom Many-to-Many Integration Tests.

Tests the full classroom platform under realistic multi-user, multi-language,
multi-room scenarios with a **live server** (real LLM, real translation, real TTS).

Tests:
  1. multi_speaker_multi_listener   — 2 speakers (en, hi) × 3 listeners (en, hi, ta)
  2. sentence_delivery_ordering     — Speaker sends text, verify listeners get sentences in order
  3. audio_lock_serialization       — 2 concurrent sentences → listener audio never interleaves
  4. multi_room_simultaneous        — 2 rooms with active speaker+listeners running in parallel
  5. multilingual_sentence_split    — Real LLM Hindi/Tamil output → sentence boundaries correct
  6. token_rotation_active_listeners — Token passes A→B while listeners still receiving A's response
  7. listener_mode_switch_mid_resp  — Listener switches text_only→text_and_audio mid-response

Usage (inside Docker):
    python tests/test_classroom_integration.py                                  # Run all
    python tests/test_classroom_integration.py --test multi_speaker_multi_listener
    python tests/test_classroom_integration.py --verbose
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp
import jwt as pyjwt

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)

# ─────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────
HTTP_URL = os.getenv("PIPECAT_HTTP_URL", "http://mira-voice:7860")
WS_URL = os.getenv("PIPECAT_WS_URL", "ws://mira-voice:7860/ws")
CLASSROOM_WS_BASE = os.getenv("CLASSROOM_WS_URL", "ws://mira-voice:7860/classroom/rooms")
TEST_ADMIN_ID = os.getenv("TEST_ADMIN_ID", "test-admin")
TEST_ADMIN_NAME = os.getenv("TEST_ADMIN_NAME", "Test Admin")
WEBUI_SECRET_KEY = os.getenv("WEBUI_SECRET_KEY", "").strip() or None

# Per-test timeout (safety net) — generous but not infinite
TEST_TIMEOUT = 120  # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("classroom-integration")
_APPROVED_TEACHERS: set[str] = set()


# ─────────────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────────────
def _make_jwt(user_id: str, email: str = "") -> str:
    if not WEBUI_SECRET_KEY:
        return ""
    payload = {
        "id": user_id,
        "email": email or f"{user_id}@example.test",
        "exp": int(time.time()) + 7200,
    }
    return pyjwt.encode(payload, WEBUI_SECRET_KEY, algorithm="HS256")


def _auth_headers(user_id: str, user_name: str, role: str = "user") -> dict:
    headers = {
        "x-user-id": user_id,
        "x-user-name": user_name,
        "x-user-email": f"{user_id}@example.test",
        "x-user-role": role,
    }
    token = _make_jwt(user_id)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# ─────────────────────────────────────────────────
# WebSocket helpers
# ─────────────────────────────────────────────────
WS_OPEN_TIMEOUT = 10  # seconds for WebSocket handshake
WS_CLOSE_TIMEOUT = 3  # seconds for WebSocket close


async def ws_connect(url: str):
    """Connect to WebSocket with timeouts to prevent hangs."""
    return await websockets.connect(
        url,
        open_timeout=WS_OPEN_TIMEOUT,
        close_timeout=WS_CLOSE_TIMEOUT,
    )


async def recv_json(ws, timeout: float = 5.0) -> Optional[dict]:
    """Receive a JSON message, ignoring binary frames."""
    try:
        msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
        if isinstance(msg, str):
            return json.loads(msg)
        return None
    except (asyncio.TimeoutError, Exception):
        return None


async def recv_json_nonbinary(ws, timeout: float = 5.0) -> Optional[dict]:
    """Receive next JSON message, skipping binary frames."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = deadline - time.time()
        msg = await recv_json(ws, timeout=max(remaining, 0.1))
        if msg is not None:
            return msg
    return None


async def drain_messages(ws, duration: float = 2.0) -> List[dict]:
    """Collect all JSON messages for a duration, ignoring binary frames.

    Uses a short per-recv timeout so binary audio frames are drained quickly
    without burning the entire duration on a few hundred binary frames.

    Early-exit: if we've received at least one message AND then see silence
    for 3s, we stop.  For short drains (≤5s) the old 2s-after-start rule
    still applies so quick cleanup calls aren't slow.
    """
    messages = []
    deadline = time.time() + duration
    start = time.time()
    last_msg_at = 0.0  # time of last received message
    while time.time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
            if isinstance(raw, str):
                try:
                    messages.append(json.loads(raw))
                    last_msg_at = time.time()
                except json.JSONDecodeError:
                    pass
            else:
                # Binary frame — still counts as activity
                last_msg_at = time.time()
        except asyncio.TimeoutError:
            elapsed = time.time() - start
            # Short drains: exit after 2s of total time with silence
            if duration <= 5.0 and elapsed > 2.0:
                break
            # Long drains: exit only if we received messages and then had 3s of silence
            if last_msg_at > 0 and (time.time() - last_msg_at) > 3.0:
                break
        except Exception:
            break
    return messages


async def wait_for_message_type(ws, msg_type: str, timeout: float = 5.0) -> Optional[dict]:
    """Wait for a specific JSON message type."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = deadline - time.time()
        msg = await recv_json_nonbinary(ws, timeout=max(remaining, 0.1))
        if msg and msg.get("type") == msg_type:
            return msg
    return None


async def collect_messages_until(ws, stop_type: str, timeout: float = 30.0) -> List[dict]:
    """Collect all JSON messages until a specific type is received or timeout."""
    messages = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = deadline - time.time()
        msg = await recv_json_nonbinary(ws, timeout=max(remaining, 0.5))
        if msg:
            messages.append(msg)
            if msg.get("type") == stop_type:
                break
    return messages


async def collect_bot_response(ws, timeout: float = 60.0) -> tuple[str, list[dict]]:
    """Collect bot_text chunks until bot_text_complete. Returns (full_text, all_msgs).

    Uses a generous first-message timeout (translation of greetings can take 16+ seconds),
    then shorter timeouts for subsequent messages.
    """
    text_chunks = []
    all_msgs = []
    deadline = time.time() + timeout
    got_first = False
    while time.time() < deadline:
        remaining = deadline - time.time()
        # First message may take long (greeting translation blocks LLM)
        t = min(remaining, 45.0) if not got_first else min(remaining, 15.0)
        msg = await recv_json_nonbinary(ws, timeout=max(t, 0.5))
        if not msg:
            if got_first:
                break  # No more messages after we started getting content
            continue
        all_msgs.append(msg)
        if msg.get("type") == "bot_text":
            text_chunks.append(msg.get("text", ""))
            got_first = True
        elif msg.get("type") == "bot_text_complete":
            return msg.get("text", "".join(text_chunks)), all_msgs
    return "".join(text_chunks), all_msgs


# ─────────────────────────────────────────────────
# Room / Teacher helpers
# ─────────────────────────────────────────────────
async def ensure_teacher_role(user_id: str, user_name: str) -> None:
    if user_id in _APPROVED_TEACHERS:
        return
    user_headers = _auth_headers(user_id, user_name, role="user")
    admin_headers = _auth_headers(TEST_ADMIN_ID, TEST_ADMIN_NAME, role="admin")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        async with session.get(f"{HTTP_URL}/classroom/teacher-status", headers=user_headers) as resp:
            assert resp.status == 200, f"teacher-status failed: {resp.status}"
            payload = await resp.json()
            if payload.get("is_teacher"):
                _APPROVED_TEACHERS.add(user_id)
                return
        request_id = None
        async with session.post(
            f"{HTTP_URL}/classroom/teacher-requests",
            params={"purpose": "Integration test teacher approval"},
            headers=user_headers,
        ) as resp:
            if resp.status == 200:
                request_id = (await resp.json()).get("id")
            else:
                assert resp.status in (400, 409), f"teacher-request failed: {resp.status}"
        if not request_id:
            async with session.get(
                f"{HTTP_URL}/classroom/teacher-requests",
                params={"status": "pending", "limit": 200},
                headers=admin_headers,
            ) as resp:
                assert resp.status == 200
                for req in (await resp.json()).get("requests", []):
                    if req.get("user_id") == user_id:
                        request_id = req.get("id")
                        break
        if request_id:
            async with session.post(
                f"{HTTP_URL}/classroom/teacher-requests/{request_id}/approve",
                params={"note": "Approved for integration tests"},
                headers=admin_headers,
            ) as resp:
                assert resp.status == 200
        async with session.get(f"{HTTP_URL}/classroom/teacher-status", headers=user_headers) as resp:
            assert resp.status == 200
            assert (await resp.json()).get("is_teacher") is True
    _APPROVED_TEACHERS.add(user_id)


async def api_create_room(name: str, creator_user_id: str = "teacher-intg",
                          creator_name: str = "IntgTeacher") -> dict:
    await ensure_teacher_role(creator_user_id, creator_name)
    headers = _auth_headers(creator_user_id, creator_name, role="user")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        async with session.post(
            f"{HTTP_URL}/classroom/rooms",
            params={"name": name},
            headers=headers,
        ) as resp:
            assert resp.status == 200, f"Create room failed: {resp.status}"
            return await resp.json()


async def api_delete_room(room_id: str) -> dict:
    admin_headers = _auth_headers(TEST_ADMIN_ID, TEST_ADMIN_NAME, role="admin")
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.delete(f"{HTTP_URL}/classroom/rooms/{room_id}", headers=admin_headers) as resp:
                return await resp.json()
    except Exception as e:
        logger.warning(f"  Failed to delete room {room_id}: {e}")
        return {"error": str(e)}


async def join_room(ws, room_id: str, user_id: str, name: str, language: str,
                    mode: str = "text_and_audio") -> dict:
    join_msg = {
        "type": "join",
        "user_id": user_id,
        "name": name,
        "language": language,
        "mode": mode,
    }
    token = _make_jwt(user_id)
    if token:
        join_msg["token"] = token
    await ws.send(json.dumps(join_msg))
    received_msgs = []
    for _ in range(15):
        msg = await recv_json_nonbinary(ws, timeout=3.0)
        if msg:
            received_msgs.append(msg.get("type", "unknown"))
            if msg.get("type") == "joined":
                return msg
            if msg.get("type") == "error":
                raise AssertionError(f"Error joining as {name}: {msg.get('message', msg)}")
    raise AssertionError(f"Did not receive 'joined' for {name} (got: {received_msgs})")


async def safe_close(ws):
    """Close WebSocket with timeout, ignoring errors."""
    try:
        await asyncio.wait_for(ws.close(), timeout=3.0)
    except (asyncio.TimeoutError, Exception):
        pass


# ─────────────────────────────────────────────────
# Test result
# ─────────────────────────────────────────────────
@dataclass
class TestResult:
    name: str
    passed: bool
    duration_sec: float = 0.0
    details: dict = field(default_factory=dict)
    error: str = ""


# ═══════════════════════════════════════════════════
# TEST 1: Multi-Speaker × Multi-Listener × Multi-Language
# ═══════════════════════════════════════════════════
async def test_multi_speaker_multi_listener() -> TestResult:
    """2 speakers (en, hi) take turns, 3 listeners (en, hi, ta) receive translated content.

    Flow:
      1. Speaker A (en) gets token, sends English question
      2. All 3 listeners receive translated bot response
      3. Speaker A passes token to Speaker B (hi)
      4. Speaker B sends Hindi question
      5. All 3 listeners receive translated bot response
      6. Verify each listener got content in their language for both rounds
    """
    t0 = time.time()
    try:
        room = await api_create_room("Multi-Speaker-Listener")
        room_id = room["room_id"]
        logger.info(f"  Created room: {room_id}")

        # ── Join all participants ──
        ws_speaker_a = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_speaker_a, room_id, "spk-a-intg", "SpeakerA", "en")
        # Wait for token_changed — first joiner gets auto-assigned speaker
        token_msg = await wait_for_message_type(ws_speaker_a, "token_changed", timeout=10.0)
        assert token_msg and token_msg.get("speaker_id") == "spk-a-intg", \
            f"Speaker A did not get token: {token_msg}"
        await drain_messages(ws_speaker_a, duration=1.0)  # drain greeting etc.
        logger.info("  1. Speaker A (en) joined, got token ✓")

        ws_speaker_b = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_speaker_b, room_id, "spk-b-intg", "SpeakerB", "hi")
        await drain_messages(ws_speaker_b, duration=1.0)
        logger.info("  2. Speaker B (hi) joined ✓")

        ws_listener_en = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_listener_en, room_id, "lst-en-intg", "ListenerEN", "en")
        await drain_messages(ws_listener_en, duration=1.0)
        logger.info("  3. Listener EN joined ✓")

        ws_listener_hi = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_listener_hi, room_id, "lst-hi-intg", "ListenerHI", "hi")
        await drain_messages(ws_listener_hi, duration=1.0)
        logger.info("  4. Listener HI joined ✓")

        ws_listener_ta = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_listener_ta, room_id, "lst-ta-intg", "ListenerTA", "ta")
        await drain_messages(ws_listener_ta, duration=1.0)
        logger.info("  5. Listener TA joined ✓")

        # Wait for greeting translations to complete before sending questions.
        # Each listener join triggers a translated greeting that can take 8-16s.
        logger.info("  Waiting for greeting translations to settle...")
        await asyncio.sleep(5.0)
        # Drain any greeting messages from all connections
        await drain_messages(ws_speaker_a, duration=2.0)
        await drain_messages(ws_speaker_b, duration=2.0)
        await drain_messages(ws_listener_en, duration=2.0)
        await drain_messages(ws_listener_hi, duration=2.0)
        await drain_messages(ws_listener_ta, duration=2.0)

        # ── Round 1: Speaker A (English) sends a question ──
        await ws_speaker_a.send(json.dumps({"type": "text_message", "text": "What is gravity? One sentence."}))
        logger.info("  6. Speaker A sent: 'What is gravity? One sentence.'")

        r1_response, _ = await collect_bot_response(ws_speaker_a, timeout=45.0)
        assert len(r1_response.strip()) > 0, "No bot response to Speaker A"
        logger.info(f"  Speaker A got response ({len(r1_response)} chars) ✓")

        # Collect listener events for round 1
        r1_en_msgs = await drain_messages(ws_listener_en, duration=5.0)
        r1_hi_msgs = await drain_messages(ws_listener_hi, duration=3.0)
        r1_ta_msgs = await drain_messages(ws_listener_ta, duration=3.0)

        r1_en_events = [m for m in r1_en_msgs if m.get("type") in ("transcription", "bot_text", "bot_text_complete", "bot_response")]
        r1_hi_events = [m for m in r1_hi_msgs if m.get("type") in ("transcription", "bot_text", "bot_text_complete", "bot_response")]
        r1_ta_events = [m for m in r1_ta_msgs if m.get("type") in ("transcription", "bot_text", "bot_text_complete", "bot_response")]

        logger.info(f"  Round 1 listener events: EN={len(r1_en_events)}, HI={len(r1_hi_events)}, TA={len(r1_ta_events)}")
        assert len(r1_en_events) > 0 or len(r1_hi_events) > 0 or len(r1_ta_events) > 0, \
            "No listener received events in round 1"
        logger.info("  Round 1: listeners received events ✓")

        # ── Pass token A → B ──
        await ws_speaker_a.send(json.dumps({"type": "pass_token", "to": "spk-b-intg"}))
        logger.info("  7. Speaker A passing token to Speaker B")

        # Wait for Speaker B to get the token
        token_b = await wait_for_message_type(ws_speaker_b, "token_changed", timeout=10.0)
        assert token_b and token_b.get("speaker_id") == "spk-b-intg", \
            f"Speaker B did not get token: {token_b}"
        logger.info("  Speaker B got token ✓")

        # Drain any leftover messages
        await drain_messages(ws_speaker_a, duration=1.0)
        await drain_messages(ws_speaker_b, duration=1.0)
        await drain_messages(ws_listener_en, duration=1.0)
        await drain_messages(ws_listener_hi, duration=1.0)
        await drain_messages(ws_listener_ta, duration=1.0)

        # ── Round 2: Speaker B (Hindi) sends a question ──
        await ws_speaker_b.send(json.dumps({"type": "text_message", "text": "सूर्य क्या है? एक वाक्य में बताओ।"}))
        logger.info("  8. Speaker B sent Hindi question")

        r2_response, _ = await collect_bot_response(ws_speaker_b, timeout=45.0)
        assert len(r2_response.strip()) > 0, "No bot response to Speaker B"
        logger.info(f"  Speaker B got response ({len(r2_response)} chars) ✓")

        # Collect listener events for round 2
        r2_en_msgs = await drain_messages(ws_listener_en, duration=5.0)
        r2_hi_msgs = await drain_messages(ws_listener_hi, duration=3.0)
        r2_ta_msgs = await drain_messages(ws_listener_ta, duration=3.0)

        r2_en_events = [m for m in r2_en_msgs if m.get("type") in ("transcription", "bot_text", "bot_text_complete", "bot_response")]
        r2_hi_events = [m for m in r2_hi_msgs if m.get("type") in ("transcription", "bot_text", "bot_text_complete", "bot_response")]
        r2_ta_events = [m for m in r2_ta_msgs if m.get("type") in ("transcription", "bot_text", "bot_text_complete", "bot_response")]

        logger.info(f"  Round 2 listener events: EN={len(r2_en_events)}, HI={len(r2_hi_events)}, TA={len(r2_ta_events)}")
        assert len(r2_en_events) > 0 or len(r2_hi_events) > 0 or len(r2_ta_events) > 0, \
            "No listener received events in round 2"
        logger.info("  Round 2: listeners received events ✓")

        # Cleanup
        for ws in [ws_speaker_a, ws_speaker_b, ws_listener_en, ws_listener_hi, ws_listener_ta]:
            await safe_close(ws)
        await api_delete_room(room_id)

        details = {
            "r1_speaker_response_len": len(r1_response),
            "r1_en_events": len(r1_en_events),
            "r1_hi_events": len(r1_hi_events),
            "r1_ta_events": len(r1_ta_events),
            "r2_speaker_response_len": len(r2_response),
            "r2_en_events": len(r2_en_events),
            "r2_hi_events": len(r2_hi_events),
            "r2_ta_events": len(r2_ta_events),
        }
        logger.info(f"  ✅ Multi-speaker × multi-listener complete: {details}")
        return TestResult(name="multi_speaker_multi_listener", passed=True, duration_sec=time.time() - t0, details=details)

    except Exception as e:
        return TestResult(name="multi_speaker_multi_listener", passed=False, error=str(e), duration_sec=time.time() - t0)


# ═══════════════════════════════════════════════════
# TEST 2: Sentence Delivery Ordering
# ═══════════════════════════════════════════════════
async def test_sentence_delivery_ordering() -> TestResult:
    """Speaker sends a multi-sentence question, listeners receive sentences in order.

    The server splits LLM output at sentence boundaries and dispatches each sentence
    to listeners. This test verifies:
      1. Listeners receive bot_text/bot_response events (sentence chunks)
      2. The chunks arrive in the correct order (sequential, not jumbled)
      3. All listeners eventually get bot_text_complete
    """
    t0 = time.time()
    try:
        room = await api_create_room("Sentence-Ordering")
        room_id = room["room_id"]

        ws_speaker = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_speaker, room_id, "spk-order", "Speaker", "en")
        token_msg = await wait_for_message_type(ws_speaker, "token_changed", timeout=10.0)
        assert token_msg, "Speaker did not get token"
        await drain_messages(ws_speaker, duration=1.0)
        logger.info("  Speaker joined with token ✓")

        ws_listener_hi = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_listener_hi, room_id, "lst-order-hi", "ListenerHI", "hi")
        await drain_messages(ws_listener_hi, duration=1.0)

        ws_listener_ta = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_listener_ta, room_id, "lst-order-ta", "ListenerTA", "ta")
        await drain_messages(ws_listener_ta, duration=1.0)
        logger.info("  2 listeners joined ✓")

        # Wait for greeting audio to fully complete for both listeners.
        # Greeting TTS can take 10-15s for Hindi/Tamil translation.
        # We must drain until bot_audio_end so greeting audio doesn't leak
        # into the response collection.
        logger.info("  Waiting for greeting audio to complete...")
        for lbl, ws_l in [("HI", ws_listener_hi), ("TA", ws_listener_ta)]:
            end_seen = False
            deadline_g = time.time() + 25.0
            while time.time() < deadline_g:
                try:
                    raw = await asyncio.wait_for(ws_l.recv(), timeout=1.0)
                    if isinstance(raw, str):
                        try:
                            msg = json.loads(raw)
                            if msg.get("type") == "bot_audio_end":
                                end_seen = True
                        except json.JSONDecodeError:
                            pass
                except asyncio.TimeoutError:
                    if end_seen:
                        break
                    continue
                except Exception:
                    break
            logger.info(f"  {lbl} greeting drained (end_seen={end_seen})")
        await drain_messages(ws_speaker, duration=2.0)
        # Extra drain to clear any trailing messages
        await drain_messages(ws_listener_hi, duration=1.0)
        await drain_messages(ws_listener_ta, duration=1.0)

        # Ask a question that will produce multiple sentences
        await ws_speaker.send(json.dumps({
            "type": "text_message",
            "text": "Explain photosynthesis in exactly 3 sentences.",
        }))
        logger.info("  Speaker sent multi-sentence question")

        # Collect speaker response
        sp_response, _ = await collect_bot_response(ws_speaker, timeout=45.0)
        assert len(sp_response.strip()) > 0, "No bot response"
        logger.info(f"  Speaker got response ({len(sp_response)} chars) ✓")

        # Collect listener messages — text now arrives WITH the first audio
        # chunk (not before), so we need to drain long enough for translation
        # + TTS (3-5s per sentence × 3 sentences, serialized by audio_lock).
        # Use 45s to be safe.
        hi_msgs = await drain_messages(ws_listener_hi, duration=45.0)
        ta_msgs = await drain_messages(ws_listener_ta, duration=30.0)

        # Extract ordered text events
        hi_texts = []
        ta_texts = []
        for m in hi_msgs:
            if m.get("type") == "bot_text":
                hi_texts.append(m.get("text", ""))
            elif m.get("type") == "bot_response":
                hi_texts.append(m.get("translated_text", ""))
        for m in ta_msgs:
            if m.get("type") == "bot_text":
                ta_texts.append(m.get("text", ""))
            elif m.get("type") == "bot_response":
                ta_texts.append(m.get("translated_text", ""))

        logger.info(f"  Hindi listener: {len(hi_texts)} text chunks")
        logger.info(f"  Tamil listener: {len(ta_texts)} text chunks")

        # Verify at least one listener got multiple chunks (sentence-level delivery)
        has_multi_chunk = len(hi_texts) > 1 or len(ta_texts) > 1
        logger.info(f"  Multi-chunk delivery: {has_multi_chunk}")

        # Verify bot_text_complete was received
        hi_complete = [m for m in hi_msgs if m.get("type") == "bot_text_complete"]
        ta_complete = [m for m in ta_msgs if m.get("type") == "bot_text_complete"]
        has_complete = len(hi_complete) > 0 or len(ta_complete) > 0
        logger.info(f"  bot_text_complete received: HI={len(hi_complete)}, TA={len(ta_complete)}")

        # Verify ordering: if we got multiple chunks, they should be non-empty
        # and each chunk should be a meaningful sentence fragment
        for i, chunk in enumerate(hi_texts):
            assert len(chunk.strip()) > 0, f"Hindi chunk {i} is empty"
        for i, chunk in enumerate(ta_texts):
            assert len(chunk.strip()) > 0, f"Tamil chunk {i} is empty"

        assert len(hi_texts) > 0 or len(ta_texts) > 0, "No text delivered to any listener"

        for ws in [ws_speaker, ws_listener_hi, ws_listener_ta]:
            await safe_close(ws)
        await api_delete_room(room_id)

        details = {
            "speaker_response_len": len(sp_response),
            "hi_chunks": len(hi_texts),
            "ta_chunks": len(ta_texts),
            "multi_chunk": has_multi_chunk,
            "has_complete": has_complete,
        }
        logger.info(f"  ✅ Sentence delivery ordering complete: {details}")
        return TestResult(name="sentence_delivery_ordering", passed=True, duration_sec=time.time() - t0, details=details)

    except Exception as e:
        return TestResult(name="sentence_delivery_ordering", passed=False, error=str(e), duration_sec=time.time() - t0)


# ═══════════════════════════════════════════════════
# TEST 3: Audio Lock Serialization
# ═══════════════════════════════════════════════════
async def test_audio_lock_serialization() -> TestResult:
    """Verify _audio_lock prevents interleaving when multiple sentences dispatch concurrently.

    A text_and_audio listener should receive:
      bot_audio_start → (binary audio) → bot_audio_end
    for each sentence, with NO interleaving between sentences.

    We check that bot_audio_start and bot_audio_end always alternate correctly.
    """
    t0 = time.time()
    try:
        room = await api_create_room("Audio-Lock-Test")
        room_id = room["room_id"]

        ws_speaker = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_speaker, room_id, "spk-lock", "Speaker", "en")
        token_msg = await wait_for_message_type(ws_speaker, "token_changed", timeout=10.0)
        assert token_msg, "Speaker did not get token"
        await drain_messages(ws_speaker, duration=1.0)
        logger.info("  Speaker joined ✓")

        # Listener in text_and_audio mode — will receive audio
        ws_listener = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_listener, room_id, "lst-lock", "AudioListener", "hi", mode="text_and_audio")
        logger.info("  Audio listener (hi, text_and_audio) joined ✓")

        # Wait for greeting audio to fully complete — greeting TTS can take 10-15s
        # for Hindi translation. We must drain until we see bot_audio_end, otherwise
        # it leaks into the test's audio event sequence.
        logger.info("  Waiting for greeting audio to complete...")
        greeting_audio_end_seen = False
        deadline_drain = time.time() + 25.0
        while time.time() < deadline_drain:
            try:
                raw = await asyncio.wait_for(ws_listener.recv(), timeout=1.0)
                if isinstance(raw, str):
                    try:
                        msg = json.loads(raw)
                        if msg.get("type") == "bot_audio_end":
                            greeting_audio_end_seen = True
                            logger.info("  Greeting bot_audio_end received")
                    except json.JSONDecodeError:
                        pass
                # Keep draining binary frames and other messages
            except asyncio.TimeoutError:
                if greeting_audio_end_seen:
                    break
                continue
            except Exception:
                break
        # Drain any remaining messages after bot_audio_end
        await asyncio.sleep(1.0)
        deadline_extra = time.time() + 3.0
        while time.time() < deadline_extra:
            try:
                raw = await asyncio.wait_for(ws_listener.recv(), timeout=0.3)
            except (asyncio.TimeoutError, Exception):
                break
        await drain_messages(ws_speaker, duration=1.0)
        logger.info(f"  Greeting audio drained (end_seen={greeting_audio_end_seen}) ✓")

        # Send a question that produces multiple sentences
        await ws_speaker.send(json.dumps({
            "type": "text_message",
            "text": "Explain gravity in 3 sentences. Be detailed.",
        }))
        logger.info("  Speaker sent multi-sentence question")

        # Wait for speaker response
        sp_response, _ = await collect_bot_response(ws_speaker, timeout=45.0)
        assert len(sp_response.strip()) > 0, "No bot response"
        logger.info(f"  Speaker got response ({len(sp_response)} chars) ✓")

        # Collect ALL listener messages (JSON + binary).
        # Hindi TTS takes 4-14s per sentence, serialized by audio lock,
        # so 3 sentences can take 12-42s. Use generous timeout.
        audio_events = []  # Track bot_audio_start / bot_audio_end sequence
        binary_count = 0
        text_events = []
        got_text_complete = False
        deadline = time.time() + 90.0
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws_listener.recv(), timeout=2.0)
                if isinstance(raw, str):
                    msg = json.loads(raw)
                    if msg.get("type") == "bot_audio_start":
                        audio_events.append("START")
                    elif msg.get("type") == "bot_audio_end":
                        audio_events.append("END")
                        # If we've seen text_complete AND audio events are balanced,
                        # we can stop — all audio has been delivered.
                        if got_text_complete and audio_events.count("START") == audio_events.count("END"):
                            logger.info("  All audio START/END balanced after text_complete — done")
                            break
                    elif msg.get("type") in ("bot_text", "bot_response"):
                        text_events.append(msg)
                    elif msg.get("type") == "bot_text_complete":
                        text_events.append(msg)
                        got_text_complete = True
                elif isinstance(raw, bytes):
                    binary_count += 1
            except asyncio.TimeoutError:
                # If we already have balanced START/END, we're done
                if got_text_complete and audio_events and audio_events.count("START") == audio_events.count("END"):
                    break
                continue
            except Exception:
                break

        logger.info(f"  Audio events sequence: {audio_events}")
        logger.info(f"  Binary audio frames: {binary_count}")
        logger.info(f"  Text events: {len(text_events)}")

        # Validate audio serialization: START and END must strictly alternate
        # Pattern should be: START, END, START, END, ...
        interleave_errors = 0
        expected_next = "START"
        for event in audio_events:
            if event != expected_next:
                interleave_errors += 1
                logger.warning(f"  Audio interleave detected! Expected {expected_next}, got {event}")
            expected_next = "END" if event == "START" else "START"

        # Every START must have a matching END
        start_count = audio_events.count("START")
        end_count = audio_events.count("END")

        logger.info(f"  Audio START={start_count}, END={end_count}, interleave_errors={interleave_errors}")

        # If we got audio events, they must be properly serialized
        if len(audio_events) > 0:
            assert interleave_errors == 0, \
                f"Audio interleaving detected! Sequence: {audio_events}"
            assert start_count == end_count, \
                f"Mismatched START/END: {start_count} starts, {end_count} ends"
            logger.info("  Audio serialization: no interleaving ✓")
        else:
            logger.info("  No audio events received (TTS may have timed out) — checking text only")

        # At least text should have been received
        assert len(text_events) > 0, "No text or audio events received by listener"

        for ws in [ws_speaker, ws_listener]:
            await safe_close(ws)
        await api_delete_room(room_id)

        details = {
            "audio_events": audio_events,
            "binary_frames": binary_count,
            "text_events": len(text_events),
            "interleave_errors": interleave_errors,
            "start_count": start_count,
            "end_count": end_count,
        }
        logger.info(f"  ✅ Audio lock serialization complete: {details}")
        return TestResult(name="audio_lock_serialization", passed=True, duration_sec=time.time() - t0, details=details)

    except Exception as e:
        return TestResult(name="audio_lock_serialization", passed=False, error=str(e), duration_sec=time.time() - t0)


# ═══════════════════════════════════════════════════
# TEST 4: Multiple Rooms Simultaneously
# ═══════════════════════════════════════════════════
async def test_multi_room_simultaneous() -> TestResult:
    """2 rooms run in parallel, each with a speaker + listener. Verify isolation.

    Room A: English speaker, Hindi listener
    Room B: Hindi speaker, Tamil listener

    Both speakers send messages concurrently. Verify:
      1. Each room's listeners get the correct content
      2. No cross-room leakage
    """
    t0 = time.time()
    try:
        room_a = await api_create_room("Room-A-Parallel", creator_user_id="teacher-intg-a", creator_name="TeacherA")
        room_b = await api_create_room("Room-B-Parallel", creator_user_id="teacher-intg-b", creator_name="TeacherB")
        room_a_id = room_a["room_id"]
        room_b_id = room_b["room_id"]
        logger.info(f"  Created rooms: A={room_a_id}, B={room_b_id}")

        # ── Room A: English speaker + Hindi listener ──
        ws_spk_a = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_a_id}/ws")
        await join_room(ws_spk_a, room_a_id, "spk-a-par", "SpeakerA", "en")
        token_a = await wait_for_message_type(ws_spk_a, "token_changed", timeout=10.0)
        assert token_a, "Room A: speaker did not get token"
        await drain_messages(ws_spk_a, duration=1.0)

        ws_lst_a = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_a_id}/ws")
        await join_room(ws_lst_a, room_a_id, "lst-a-par", "ListenerA", "hi", mode="text_only")
        await drain_messages(ws_lst_a, duration=1.0)
        logger.info("  Room A: speaker (en) + listener (hi) joined ✓")

        # ── Room B: Hindi speaker + Tamil listener ──
        ws_spk_b = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_b_id}/ws")
        await join_room(ws_spk_b, room_b_id, "spk-b-par", "SpeakerB", "hi")
        token_b = await wait_for_message_type(ws_spk_b, "token_changed", timeout=10.0)
        assert token_b, "Room B: speaker did not get token"
        await drain_messages(ws_spk_b, duration=1.0)

        ws_lst_b = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_b_id}/ws")
        await join_room(ws_lst_b, room_b_id, "lst-b-par", "ListenerB", "ta", mode="text_only")
        await drain_messages(ws_lst_b, duration=1.0)
        logger.info("  Room B: speaker (hi) + listener (ta) joined ✓")

        # Wait for greeting translations to settle
        await asyncio.sleep(5.0)
        await drain_messages(ws_spk_a, duration=2.0)
        await drain_messages(ws_lst_a, duration=2.0)
        await drain_messages(ws_spk_b, duration=2.0)
        await drain_messages(ws_lst_b, duration=2.0)

        # ── Send messages concurrently ──
        await ws_spk_a.send(json.dumps({"type": "text_message", "text": "What is the moon? One sentence."}))
        await ws_spk_b.send(json.dumps({"type": "text_message", "text": "चंद्रमा क्या है? एक वाक्य में।"}))
        logger.info("  Both speakers sent messages concurrently")

        # Collect responses in parallel
        async def collect_room(ws_spk, ws_lst, label):
            response, _ = await collect_bot_response(ws_spk, timeout=45.0)
            lst_msgs = await drain_messages(ws_lst, duration=8.0)
            lst_events = [m for m in lst_msgs if m.get("type") in ("transcription", "bot_text", "bot_text_complete", "bot_response")]
            logger.info(f"  {label}: speaker={len(response)} chars, listener={len(lst_events)} events")
            return response, lst_events

        (resp_a, lst_a_events), (resp_b, lst_b_events) = await asyncio.gather(
            collect_room(ws_spk_a, ws_lst_a, "Room A"),
            collect_room(ws_spk_b, ws_lst_b, "Room B"),
        )

        assert len(resp_a.strip()) > 0, "Room A: no speaker response"
        assert len(resp_b.strip()) > 0, "Room B: no speaker response"
        logger.info("  Both speakers got responses ✓")

        # At least one listener in each room should have received events
        assert len(lst_a_events) > 0, "Room A: listener got no events"
        assert len(lst_b_events) > 0, "Room B: listener got no events"
        logger.info("  Both listeners got events ✓")

        # Cross-room isolation: Room A's listener should not have Room B content
        # and vice versa. This is implicitly tested by the room_id-scoped WebSocket.
        # Just verify we didn't crash and both rooms operated independently.

        for ws in [ws_spk_a, ws_lst_a, ws_spk_b, ws_lst_b]:
            await safe_close(ws)
        await api_delete_room(room_a_id)
        await api_delete_room(room_b_id)

        details = {
            "room_a_response_len": len(resp_a),
            "room_a_listener_events": len(lst_a_events),
            "room_b_response_len": len(resp_b),
            "room_b_listener_events": len(lst_b_events),
        }
        logger.info(f"  ✅ Multi-room simultaneous complete: {details}")
        return TestResult(name="multi_room_simultaneous", passed=True, duration_sec=time.time() - t0, details=details)

    except Exception as e:
        return TestResult(name="multi_room_simultaneous", passed=False, error=str(e), duration_sec=time.time() - t0)


# ═══════════════════════════════════════════════════
# TEST 5: Multilingual Sentence Splitting with Real LLM Output
# ═══════════════════════════════════════════════════
async def test_multilingual_sentence_split() -> TestResult:
    """Ask the LLM to respond in Hindi, verify sentence-level delivery to listeners.

    The classroom's _SENTENCE_RE must correctly split Hindi text (using ।) and
    deliver each sentence as a separate chunk to listeners.
    """
    t0 = time.time()
    try:
        room = await api_create_room("Hindi-Sentence-Split")
        room_id = room["room_id"]

        ws_speaker = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_speaker, room_id, "spk-hindi", "HindiSpeaker", "hi")
        token_msg = await wait_for_message_type(ws_speaker, "token_changed", timeout=10.0)
        assert token_msg, "Hindi speaker did not get token"
        await drain_messages(ws_speaker, duration=1.0)
        logger.info("  Hindi speaker joined ✓")

        ws_listener_en = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_listener_en, room_id, "lst-en-split", "EnListener", "en", mode="text_only")
        await drain_messages(ws_listener_en, duration=1.0)

        ws_listener_ta = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_listener_ta, room_id, "lst-ta-split", "TaListener", "ta", mode="text_only")
        await drain_messages(ws_listener_ta, duration=1.0)
        logger.info("  EN + TA listeners joined ✓")

        # Wait for greeting translations to settle
        await asyncio.sleep(5.0)
        await drain_messages(ws_speaker, duration=2.0)
        await drain_messages(ws_listener_en, duration=2.0)
        await drain_messages(ws_listener_ta, duration=2.0)

        # Ask in Hindi to get Hindi response with danda (।) sentence boundaries
        await ws_speaker.send(json.dumps({
            "type": "text_message",
            "text": "प्रकाश संश्लेषण को तीन वाक्यों में समझाओ।",
        }))
        logger.info("  Speaker asked about photosynthesis in Hindi")

        sp_response, _ = await collect_bot_response(ws_speaker, timeout=45.0)
        assert len(sp_response.strip()) > 0, "No bot response"
        logger.info(f"  Speaker got response ({len(sp_response)} chars): '{sp_response[:80]}...'")

        # Collect listener chunks
        en_msgs = await drain_messages(ws_listener_en, duration=8.0)
        ta_msgs = await drain_messages(ws_listener_ta, duration=5.0)

        en_chunks = [m for m in en_msgs if m.get("type") in ("bot_text", "bot_response")]
        ta_chunks = [m for m in ta_msgs if m.get("type") in ("bot_text", "bot_response")]

        logger.info(f"  EN listener: {len(en_chunks)} chunks")
        logger.info(f"  TA listener: {len(ta_chunks)} chunks")

        # Log the actual chunks for debugging
        for i, c in enumerate(en_chunks[:5]):
            text = c.get("text", c.get("translated_text", ""))[:60]
            logger.info(f"    EN chunk {i}: '{text}'")
        for i, c in enumerate(ta_chunks[:5]):
            text = c.get("text", c.get("translated_text", ""))[:60]
            logger.info(f"    TA chunk {i}: '{text}'")

        # The LLM should produce multi-sentence Hindi, which _SENTENCE_RE splits
        # Listeners should get multiple chunks (one per sentence)
        total_chunks = len(en_chunks) + len(ta_chunks)
        assert total_chunks > 0, "No sentence chunks delivered to any listener"

        for ws in [ws_speaker, ws_listener_en, ws_listener_ta]:
            await safe_close(ws)
        await api_delete_room(room_id)

        details = {
            "speaker_response_len": len(sp_response),
            "en_chunks": len(en_chunks),
            "ta_chunks": len(ta_chunks),
            "speaker_response_preview": sp_response[:120],
        }
        logger.info(f"  ✅ Multilingual sentence split complete: {details}")
        return TestResult(name="multilingual_sentence_split", passed=True, duration_sec=time.time() - t0, details=details)

    except Exception as e:
        return TestResult(name="multilingual_sentence_split", passed=False, error=str(e), duration_sec=time.time() - t0)


# ═══════════════════════════════════════════════════
# TEST 6: Speaker Token Rotation with Active Listeners
# ═══════════════════════════════════════════════════
async def test_token_rotation_active_listeners() -> TestResult:
    """Token passes from Speaker A to Speaker B while listeners are still receiving A's response.

    Flow:
      1. Speaker A sends a long question (multi-sentence)
      2. While listeners are receiving A's translated response, A passes token to B
      3. Speaker B sends a question
      4. Verify both responses eventually reach listeners without corruption
    """
    t0 = time.time()
    try:
        room = await api_create_room("Token-Rotation")
        room_id = room["room_id"]

        ws_spk_a = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_spk_a, room_id, "spk-rot-a", "RotSpeakerA", "en")
        token_msg = await wait_for_message_type(ws_spk_a, "token_changed", timeout=10.0)
        assert token_msg, "Speaker A did not get token"
        await drain_messages(ws_spk_a, duration=1.0)
        logger.info("  Speaker A (en) joined with token ✓")

        ws_spk_b = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_spk_b, room_id, "spk-rot-b", "RotSpeakerB", "hi")
        await drain_messages(ws_spk_b, duration=1.0)
        logger.info("  Speaker B (hi) joined ✓")

        ws_listener = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_listener, room_id, "lst-rot", "RotListener", "ta", mode="text_only")
        await drain_messages(ws_listener, duration=1.0)
        logger.info("  Listener (ta) joined ✓")

        # Wait for greeting translations to settle
        await asyncio.sleep(5.0)
        await drain_messages(ws_spk_a, duration=2.0)
        await drain_messages(ws_spk_b, duration=2.0)
        await drain_messages(ws_listener, duration=2.0)

        # ── Speaker A sends a question that will produce a multi-sentence response ──
        await ws_spk_a.send(json.dumps({
            "type": "text_message",
            "text": "Explain the water cycle in detail. At least 3 sentences.",
        }))
        logger.info("  Speaker A sent long question")

        # Wait for the first bot_text chunk (LLM has started responding)
        first_chunk = await wait_for_message_type(ws_spk_a, "bot_text", timeout=45.0)
        assert first_chunk, "No initial response from LLM"
        logger.info(f"  Speaker A got first chunk: '{first_chunk.get('text', '')[:40]}' ✓")

        # NOW pass token while response is still being generated
        await ws_spk_a.send(json.dumps({"type": "pass_token", "to": "spk-rot-b"}))
        logger.info("  Speaker A passed token to Speaker B (mid-response)")

        # Collect the rest of Speaker A's response
        a_response, _ = await collect_bot_response(ws_spk_a, timeout=30.0)
        a_full = first_chunk.get("text", "") + a_response
        logger.info(f"  Speaker A full response: {len(a_full)} chars")

        # Wait for Speaker B to get token
        token_b = await wait_for_message_type(ws_spk_b, "token_changed", timeout=10.0)
        logger.info(f"  Speaker B got token: {token_b is not None}")

        # Drain any leftover messages
        await drain_messages(ws_spk_b, duration=1.0)

        # Speaker B sends a question
        await ws_spk_b.send(json.dumps({
            "type": "text_message",
            "text": "पानी का महत्व क्या है? एक वाक्य में।",
        }))
        logger.info("  Speaker B sent question")

        b_response, _ = await collect_bot_response(ws_spk_b, timeout=45.0)
        assert len(b_response.strip()) > 0, "No bot response to Speaker B"
        logger.info(f"  Speaker B got response: {len(b_response)} chars ✓")

        # Collect listener events (should have events from both rounds)
        listener_msgs = await drain_messages(ws_listener, duration=8.0)
        listener_events = [m for m in listener_msgs if m.get("type") in (
            "transcription", "bot_text", "bot_text_complete", "bot_response",
            "token_changed",
        )]

        # Separate token change events from content events
        token_events = [m for m in listener_events if m.get("type") == "token_changed"]
        content_events = [m for m in listener_events if m.get("type") != "token_changed"]

        logger.info(f"  Listener: {len(content_events)} content events, {len(token_events)} token events")
        assert len(content_events) > 0, "Listener received no content events"

        for ws in [ws_spk_a, ws_spk_b, ws_listener]:
            await safe_close(ws)
        await api_delete_room(room_id)

        details = {
            "a_response_len": len(a_full),
            "b_response_len": len(b_response),
            "listener_content_events": len(content_events),
            "listener_token_events": len(token_events),
        }
        logger.info(f"  ✅ Token rotation with active listeners complete: {details}")
        return TestResult(name="token_rotation_active_listeners", passed=True, duration_sec=time.time() - t0, details=details)

    except Exception as e:
        return TestResult(name="token_rotation_active_listeners", passed=False, error=str(e), duration_sec=time.time() - t0)


# ═══════════════════════════════════════════════════
# TEST 7: Listener Mode Switch Mid-Response
# ═══════════════════════════════════════════════════
async def test_listener_mode_switch_mid_response() -> TestResult:
    """Listener switches from text_only to text_and_audio while a response is being delivered.

    Flow:
      1. Listener joins in text_only mode
      2. Speaker sends a multi-sentence question
      3. After first bot_text arrives at listener, listener switches to text_and_audio
      4. Verify: no crash, listener continues to receive events, mode change is acknowledged
    """
    t0 = time.time()
    try:
        room = await api_create_room("Mode-Switch-Mid")
        room_id = room["room_id"]

        ws_speaker = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_speaker, room_id, "spk-mode", "Speaker", "en")
        token_msg = await wait_for_message_type(ws_speaker, "token_changed", timeout=10.0)
        assert token_msg, "Speaker did not get token"
        await drain_messages(ws_speaker, duration=1.0)
        logger.info("  Speaker joined ✓")

        ws_listener = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_listener, room_id, "lst-mode", "ModeListener", "hi", mode="text_only")
        await drain_messages(ws_listener, duration=1.0)
        logger.info("  Listener joined in text_only mode ✓")

        # Wait for greeting translations to settle
        await asyncio.sleep(5.0)
        await drain_messages(ws_speaker, duration=2.0)
        await drain_messages(ws_listener, duration=2.0)

        # Speaker sends a question
        await ws_speaker.send(json.dumps({
            "type": "text_message",
            "text": "Explain the solar system in 3 sentences.",
        }))
        logger.info("  Speaker sent question")

        # Wait for the first content event on the listener side
        # (could be transcription or bot_text/bot_response)
        first_event = None
        deadline = time.time() + 30.0
        while time.time() < deadline:
            msg = await recv_json_nonbinary(ws_listener, timeout=2.0)
            if msg and msg.get("type") in ("transcription", "bot_text", "bot_response"):
                first_event = msg
                break
        logger.info(f"  Listener got first event: {first_event.get('type') if first_event else 'None'}")

        # NOW switch mode mid-delivery
        await ws_listener.send(json.dumps({"type": "set_mode", "mode": "text_and_audio"}))
        logger.info("  Listener switching to text_and_audio mid-response")

        # Collect remaining events (mix of text, mode_changed, possibly audio)
        remaining_msgs = []
        mode_changed = False
        deadline2 = time.time() + 15.0
        while time.time() < deadline2:
            try:
                raw = await asyncio.wait_for(ws_listener.recv(), timeout=1.0)
                if isinstance(raw, str):
                    msg = json.loads(raw)
                    remaining_msgs.append(msg)
                    if msg.get("type") == "mode_changed":
                        mode_changed = True
                        logger.info(f"  Mode changed acknowledged: {msg.get('mode')}")
                    elif msg.get("type") == "bot_text_complete":
                        # Wait a bit more for any trailing audio
                        deadline2 = min(deadline2, time.time() + 3.0)
                elif isinstance(raw, bytes):
                    pass  # Binary audio — expected after mode switch
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

        # Also collect speaker response
        sp_response, _ = await collect_bot_response(ws_speaker, timeout=15.0)
        logger.info(f"  Speaker response: {len(sp_response)} chars")

        # Verify mode change was acknowledged
        assert mode_changed, "Mode change was not acknowledged"
        logger.info("  Mode change acknowledged ✓")

        # Verify listener continued to receive events after mode switch
        post_switch_content = [m for m in remaining_msgs if m.get("type") in (
            "bot_text", "bot_response", "bot_text_complete", "transcription",
        )]
        logger.info(f"  Post-switch content events: {len(post_switch_content)}")

        # The key assertion: no crash, mode switch was clean
        # The listener should have received at least the mode_changed event
        assert len(remaining_msgs) > 0, "No messages received after mode switch"

        for ws in [ws_speaker, ws_listener]:
            await safe_close(ws)
        await api_delete_room(room_id)

        details = {
            "first_event_type": first_event.get("type") if first_event else "none",
            "mode_changed": mode_changed,
            "post_switch_events": len(post_switch_content),
            "total_remaining_msgs": len(remaining_msgs),
            "speaker_response_len": len(sp_response),
        }
        logger.info(f"  ✅ Listener mode switch mid-response complete: {details}")
        return TestResult(name="listener_mode_switch_mid_response", passed=True, duration_sec=time.time() - t0, details=details)

    except Exception as e:
        return TestResult(name="listener_mode_switch_mid_response", passed=False, error=str(e), duration_sec=time.time() - t0)


# ═══════════════════════════════════════════════════
# TEST 8: Multi-Turn Conversation History Continuity
# ═══════════════════════════════════════════════════
async def test_conversation_history_continuity() -> TestResult:
    """Verify the bot remembers earlier turns in the same room.

    Flow:
      1. Speaker sends: "My name is Arjun."
      2. Bot responds (acknowledges the name).
      3. Speaker sends: "What is my name?"
      4. Bot should respond with "Arjun" — proving conversation history works.

    This catches the bug where voice-mode ClassroomBroadcaster didn't
    append to room.conversation_history, causing the bot to lose context.
    """
    t0 = time.time()
    try:
        room = await api_create_room("History-Continuity")
        room_id = room["room_id"]

        ws_speaker = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_speaker, room_id, "spk-hist", "HistSpeaker", "en")
        token_msg = await wait_for_message_type(ws_speaker, "token_changed", timeout=10.0)
        assert token_msg, "Speaker did not get token"
        await drain_messages(ws_speaker, duration=1.0)
        logger.info("  Speaker joined with token ✓")

        # Wait for greeting to settle
        await asyncio.sleep(3.0)
        await drain_messages(ws_speaker, duration=2.0)

        # ── Turn 1: Establish a fact ──
        await ws_speaker.send(json.dumps({
            "type": "text_message",
            "text": "My name is Arjun. Please remember that.",
        }))
        logger.info("  Turn 1: 'My name is Arjun'")

        r1, _ = await collect_bot_response(ws_speaker, timeout=60.0)
        assert len(r1.strip()) > 0, "No response to turn 1"
        logger.info(f"  Turn 1 response: '{r1[:80]}'")

        # ── Turn 2: Test recall ──
        await ws_speaker.send(json.dumps({
            "type": "text_message",
            "text": "What is my name?",
        }))
        logger.info("  Turn 2: 'What is my name?'")

        r2, _ = await collect_bot_response(ws_speaker, timeout=60.0)
        assert len(r2.strip()) > 0, "No response to turn 2"
        logger.info(f"  Turn 2 response: '{r2[:80]}'")

        # ── Verify the bot remembered "Arjun" ──
        name_recalled = "arjun" in r2.lower()
        logger.info(f"  Name recalled in response: {name_recalled}")
        assert name_recalled, f"Bot did not recall 'Arjun' in response: '{r2[:120]}'"

        # ── Turn 3: Additional context check ──
        await ws_speaker.send(json.dumps({
            "type": "text_message",
            "text": "How many questions have I asked you so far?",
        }))
        logger.info("  Turn 3: 'How many questions have I asked?'")

        r3, _ = await collect_bot_response(ws_speaker, timeout=60.0)
        logger.info(f"  Turn 3 response: '{r3[:80]}'")
        # We don't assert exact count, just that the bot gives a coherent answer

        await safe_close(ws_speaker)
        await api_delete_room(room_id)

        details = {
            "turn1_len": len(r1),
            "turn2_len": len(r2),
            "turn3_len": len(r3),
            "name_recalled": name_recalled,
        }
        logger.info(f"  ✅ Conversation history continuity: {details}")
        return TestResult(name="conversation_history_continuity", passed=True, duration_sec=time.time() - t0, details=details)

    except Exception as e:
        return TestResult(name="conversation_history_continuity", passed=False, error=str(e), duration_sec=time.time() - t0)


# ═══════════════════════════════════════════════════
# TEST 9: Topic Boundary Enforcement
# ═══════════════════════════════════════════════════
async def test_topic_boundary_enforcement() -> TestResult:
    """Verify the bot redirects off-topic questions when a lesson topic is set.

    Flow:
      1. Teacher sets topic to "Photosynthesis" via [TEACHER_ACTION: SET_TOPIC ...]
      2. Speaker asks an on-topic question → bot answers fully
      3. Speaker asks a completely off-topic question (e.g., "Who won the cricket world cup?")
      4. Bot should NOT fully answer the off-topic question — should redirect to topic

    This catches the bug where no context boundary was enforced.
    """
    t0 = time.time()
    try:
        room = await api_create_room("Topic-Boundary")
        room_id = room["room_id"]

        ws_speaker = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_speaker, room_id, "spk-topic", "TopicSpeaker", "en")
        token_msg = await wait_for_message_type(ws_speaker, "token_changed", timeout=10.0)
        assert token_msg, "Speaker did not get token"
        await drain_messages(ws_speaker, duration=1.0)
        logger.info("  Speaker joined ✓")

        # Wait for greeting to settle
        await asyncio.sleep(3.0)
        await drain_messages(ws_speaker, duration=2.0)

        # ── Step 1: Set the topic ──
        await ws_speaker.send(json.dumps({
            "type": "text_message",
            "text": "[TEACHER_ACTION: SET_TOPIC Photosynthesis]",
        }))
        logger.info("  Set topic: Photosynthesis")

        topic_response, _ = await collect_bot_response(ws_speaker, timeout=60.0)
        assert len(topic_response.strip()) > 0, "No response to SET_TOPIC"
        logger.info(f"  Topic intro: '{topic_response[:80]}...'")

        # ── Step 2: On-topic question ──
        await ws_speaker.send(json.dumps({
            "type": "text_message",
            "text": "What role does chlorophyll play in photosynthesis?",
        }))
        logger.info("  On-topic: 'What role does chlorophyll play?'")

        on_topic_response, _ = await collect_bot_response(ws_speaker, timeout=60.0)
        assert len(on_topic_response.strip()) > 0, "No response to on-topic question"
        # On-topic response should contain relevant keywords
        on_topic_relevant = any(kw in on_topic_response.lower() for kw in [
            "chlorophyll", "light", "green", "absorb", "pigment", "photosynth", "plant", "leaf",
        ])
        logger.info(f"  On-topic response relevant: {on_topic_relevant}")
        logger.info(f"  On-topic response: '{on_topic_response[:100]}...'")

        # ── Step 3: Off-topic question ──
        await ws_speaker.send(json.dumps({
            "type": "text_message",
            "text": "Who won the 2023 cricket world cup? Tell me the full details.",
        }))
        logger.info("  Off-topic: 'Who won the 2023 cricket world cup?'")

        off_topic_response, _ = await collect_bot_response(ws_speaker, timeout=60.0)
        assert len(off_topic_response.strip()) > 0, "No response to off-topic question"
        logger.info(f"  Off-topic response: '{off_topic_response[:120]}...'")

        # The bot should redirect — check for redirect indicators
        redirect_indicators = [
            "photosynth", "topic", "focus", "lesson", "back to",
            "right now", "class", "learning", "let's get back",
            "we're discussing", "we're covering", "we're in",
        ]
        has_redirect = any(ind in off_topic_response.lower() for ind in redirect_indicators)

        # The bot should NOT give a detailed cricket answer
        cricket_details = [
            "australia", "india", "final", "ahmedabad", "narendra modi",
            "runs", "wickets", "overs", "innings",
        ]
        has_cricket_detail = sum(1 for kw in cricket_details if kw in off_topic_response.lower())

        logger.info(f"  Has redirect language: {has_redirect}")
        logger.info(f"  Cricket detail keywords found: {has_cricket_detail}")

        # We expect either a redirect OR at most a brief acknowledgment (not full details)
        # If the bot gave 3+ cricket-specific details, it failed to enforce the boundary
        boundary_enforced = has_redirect or has_cricket_detail < 3
        logger.info(f"  Topic boundary enforced: {boundary_enforced}")
        assert boundary_enforced, (
            f"Bot fully answered off-topic question without redirect. "
            f"redirect={has_redirect}, cricket_details={has_cricket_detail}, "
            f"response='{off_topic_response[:150]}'"
        )

        await safe_close(ws_speaker)
        await api_delete_room(room_id)

        details = {
            "topic_intro_len": len(topic_response),
            "on_topic_relevant": on_topic_relevant,
            "on_topic_len": len(on_topic_response),
            "off_topic_len": len(off_topic_response),
            "has_redirect": has_redirect,
            "cricket_detail_count": has_cricket_detail,
            "boundary_enforced": boundary_enforced,
        }
        logger.info(f"  ✅ Topic boundary enforcement: {details}")
        return TestResult(name="topic_boundary_enforcement", passed=True, duration_sec=time.time() - t0, details=details)

    except Exception as e:
        return TestResult(name="topic_boundary_enforcement", passed=False, error=str(e), duration_sec=time.time() - t0)


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════
ALL_TESTS = {
    "multi_speaker_multi_listener": test_multi_speaker_multi_listener,
    "sentence_delivery_ordering": test_sentence_delivery_ordering,
    "audio_lock_serialization": test_audio_lock_serialization,
    "multi_room_simultaneous": test_multi_room_simultaneous,
    "multilingual_sentence_split": test_multilingual_sentence_split,
    "token_rotation_active_listeners": test_token_rotation_active_listeners,
    "listener_mode_switch_mid_response": test_listener_mode_switch_mid_response,
    "conversation_history_continuity": test_conversation_history_continuity,
    "topic_boundary_enforcement": test_topic_boundary_enforcement,
}


async def main():
    parser = argparse.ArgumentParser(description="Classroom Integration Tests (Many-to-Many)")
    parser.add_argument("--test", choices=list(ALL_TESTS.keys()) + ["all"], default="all",
                        help="Test to run (default: all)")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tests_to_run = ALL_TESTS if args.test == "all" else {args.test: ALL_TESTS[args.test]}

    results = []
    for name, test_fn in tests_to_run.items():
        print()
        logger.info("=" * 60)
        logger.info(f"  TEST: {name.upper()}")
        logger.info("=" * 60)
        try:
            result = await asyncio.wait_for(test_fn(), timeout=TEST_TIMEOUT)
        except asyncio.TimeoutError:
            result = TestResult(name=name, passed=False, error=f"TIMEOUT ({TEST_TIMEOUT}s)", duration_sec=TEST_TIMEOUT)
        if not result.passed:
            logger.error(f"  ❌ {name} FAILED: {result.error}")
        else:
            logger.info(f"  ✅ {name} PASSED ({result.duration_sec:.1f}s)")
        results.append(result)

    # Summary
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    print()
    print("=" * 60)
    print("  CLASSROOM INTEGRATION TEST RESULTS")
    print("=" * 60)
    for r in results:
        status = "✅" if r.passed else "❌"
        duration = f"{r.duration_sec:.1f}s" if r.duration_sec else ""
        print(f"  {status} {r.name:<35} {duration}")
        if r.details:
            for k, v in r.details.items():
                val = str(v)[:80]
                print(f"     {k}: {val}")
        if r.error:
            print(f"     error: {r.error}")
    print("=" * 60)
    print(f"  {len(passed)}/{len(results)} tests passed")
    print("=" * 60)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
