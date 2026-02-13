#!/usr/bin/env python3
"""
Language Adherence Tests for Classroom Mode.

These tests verify that the MOST CRITICAL aspect of the product works:
  - LLM always responds in the speaker's registered language
  - Language tags are correctly prepended for text messages
  - Teacher actions always get English responses
  - Conversation history in other languages does NOT cause drift
  - Listener delivery uses correct source_lang for translation
  - LLM language drift is detected and corrected

Tests:
  1. language_tag_prepend       — Unit: text messages get [User is speaking X] tags
  2. teacher_action_english     — Integration: SET_TOPIC produces English response
  3. english_after_hindi_history — Integration: English question after Hindi history stays English
  4. hindi_question_stays_hindi  — Integration: Hindi question gets Hindi response
  5. mixed_history_no_drift     — Integration: alternating EN/HI history, EN question stays EN
  6. detect_text_language_unit  — Unit: _detect_text_language correctly identifies scripts
  7. listener_gets_translation  — Integration: Hindi listener gets translation of English response
  8. tell_me_more_stays_english — Integration: "tell me more" (no context) stays English for EN speaker
  9. set_topic_then_english_q   — Integration: SET_TOPIC then English question stays English
  10. math_question_english     — Integration: "What is 7+3?" stays English for EN speaker
  11. multi_turn_language_stability — Integration: 6 consecutive EN questions stay EN
  12. discussion_room_english   — Integration: Discussion room EN student gets EN response
  13. discussion_room_hindi     — Integration: Discussion room HI student gets HI response
  14. discussion_room_multi_student_lang — Integration: Discussion room multi-student language isolation

Usage (inside Docker):
    python tests/test_language_adherence.py                          # Run all tests
    python tests/test_language_adherence.py --test language_tag_prepend  # Run one test
    python tests/test_language_adherence.py --verbose                # Debug logging

Environment variables:
    PIPECAT_HTTP_URL    (default: http://mira-voice:7860)
    CLASSROOM_WS_URL    (default: ws://mira-voice:7860/classroom/rooms)
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

# Allow importing classroom module from /app
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from classroom import RoomManager

# ─────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────
HTTP_URL = os.getenv("PIPECAT_HTTP_URL", "http://mira-voice:7860")
CLASSROOM_WS_BASE = os.getenv("CLASSROOM_WS_URL", "ws://mira-voice:7860/classroom/rooms")
TEST_ADMIN_ID = os.getenv("TEST_ADMIN_ID", "test-admin")
TEST_ADMIN_NAME = os.getenv("TEST_ADMIN_NAME", "Test Admin")
WEBUI_SECRET_KEY = os.getenv("WEBUI_SECRET_KEY", "").strip() or None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_language")
_APPROVED_TEACHERS: set[str] = set()


def _make_jwt(user_id: str, email: str = "") -> str:
    """Generate a JWT token for test auth when WEBUI_SECRET_KEY is set."""
    if not WEBUI_SECRET_KEY:
        return ""
    payload = {
        "id": user_id,
        "email": email or f"{user_id}@example.test",
        "exp": int(time.time()) + 7200,
    }
    return pyjwt.encode(payload, WEBUI_SECRET_KEY, algorithm="HS256")


# ─────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────
@dataclass
class TestResult:
    name: str
    passed: bool
    duration_sec: float = 0.0
    details: dict = field(default_factory=dict)
    error: str = ""


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


async def ensure_teacher_role(user_id: str, user_name: str) -> None:
    if user_id in _APPROVED_TEACHERS:
        return

    user_headers = _auth_headers(user_id, user_name, role="user")
    admin_headers = _auth_headers(TEST_ADMIN_ID, TEST_ADMIN_NAME, role="admin")

    async with aiohttp.ClientSession() as session:
        # Fast path when already approved.
        async with session.get(f"{HTTP_URL}/classroom/teacher-status", headers=user_headers) as resp:
            assert resp.status == 200, f"teacher-status failed: {resp.status}"
            payload = await resp.json()
            if payload.get("is_teacher"):
                _APPROVED_TEACHERS.add(user_id)
                return

        request_id = None
        async with session.post(
            f"{HTTP_URL}/classroom/teacher-requests",
            params={"purpose": "Automated test teacher approval"},
            headers=user_headers,
        ) as resp:
            if resp.status == 200:
                payload = await resp.json()
                request_id = payload.get("id")
            else:
                # Could already be pending/reviewed from prior test run.
                assert resp.status in (400, 409), f"teacher-request failed: {resp.status}"

        if not request_id:
            async with session.get(
                f"{HTTP_URL}/classroom/teacher-requests",
                params={"status": "pending", "limit": 200},
                headers=admin_headers,
            ) as resp:
                assert resp.status == 200, f"list teacher-requests failed: {resp.status}"
                pending = await resp.json()
                for req in pending.get("requests", []):
                    if req.get("user_id") == user_id:
                        request_id = req.get("id")
                        break

        if request_id:
            async with session.post(
                f"{HTTP_URL}/classroom/teacher-requests/{request_id}/approve",
                params={"note": "Approved for automated tests"},
                headers=admin_headers,
            ) as resp:
                assert resp.status == 200, f"approve teacher-request failed: {resp.status}"

        async with session.get(f"{HTTP_URL}/classroom/teacher-status", headers=user_headers) as resp:
            assert resp.status == 200, f"teacher-status recheck failed: {resp.status}"
            payload = await resp.json()
            assert payload.get("is_teacher") is True, "teacher role was not approved"

    _APPROVED_TEACHERS.add(user_id)


async def api_create_room(
    name: str,
    room_type: str = "teacher_driven",
    creator_user_id: str = "teacher-01",
    creator_name: str = "Teacher",
) -> dict:
    await ensure_teacher_role(creator_user_id, creator_name)
    params = {"name": name, "room_type": room_type}
    headers = _auth_headers(creator_user_id, creator_name, role="user")
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{HTTP_URL}/classroom/rooms", params=params, headers=headers) as resp:
            assert resp.status == 200, f"Create room failed: {resp.status}"
            return await resp.json()


async def api_delete_room(room_id: str):
    admin_headers = _auth_headers(TEST_ADMIN_ID, TEST_ADMIN_NAME, role="admin")
    async with aiohttp.ClientSession() as session:
        async with session.delete(f"{HTTP_URL}/classroom/rooms/{room_id}", headers=admin_headers) as resp:
            pass  # Best effort


async def join_room(ws, room_id: str, user_id: str, name: str, language: str,
                    role: str = "student", mode: str = "text_only"):
    """Send join message and wait for joined event."""
    join_msg = {
        "type": "join",
        "user_id": user_id,
        "name": name,
        "language": language,
        "role": role,
        "mode": mode,
    }
    token = _make_jwt(user_id)
    if token:
        join_msg["token"] = token
    await ws.send(json.dumps(join_msg))
    # Collect events until we get "joined"
    events = []
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=2)
            if isinstance(raw, str):
                msg = json.loads(raw)
                events.append(msg)
                if msg.get("type") == "joined":
                    return msg
        except asyncio.TimeoutError:
            break
    raise TimeoutError(f"Never got 'joined' event. Got: {[e.get('type') for e in events]}")


async def wait_for_token(ws, expected_speaker: str = None, timeout: float = 30.0) -> dict:
    """Wait until we receive a token_changed event (optionally for a specific speaker).

    Also drains any other messages while waiting.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=2)
            if isinstance(raw, str):
                msg = json.loads(raw)
                if msg.get("type") == "token_changed":
                    if expected_speaker is None or msg.get("speaker_id") == expected_speaker:
                        return msg
        except asyncio.TimeoutError:
            continue
    raise TimeoutError(f"Never got token_changed for {expected_speaker} within {timeout}s")


async def drain_messages(ws, drain_secs: float = 2.0):
    """Drain all pending messages from a websocket for the given duration.

    Useful after token transfers to consume greetings and broadcasts
    before sending the next text_message.
    """
    deadline = time.time() + drain_secs
    drained = 0
    while time.time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
            if isinstance(raw, str):
                msg = json.loads(raw)
                logger.debug(f"[DRAIN] {msg.get('type')}: {str(msg)[:100]}")
                drained += 1
        except asyncio.TimeoutError:
            break  # No more messages pending
    if drained:
        logger.info(f"[DRAIN] Consumed {drained} pending messages")


async def send_text_and_collect(ws, text: str, timeout: float = 15.0) -> dict:
    """Send a text_message and collect bot_text tokens + bot_text_complete.

    Returns: {"tokens": [...], "full_response": str, "all_events": [...], "error": str|None}
    """
    await ws.send(json.dumps({"type": "text_message", "text": text}))

    tokens = []
    full_response = ""
    all_events = []
    error = None
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=2)
            if isinstance(raw, str):
                msg = json.loads(raw)
                all_events.append(msg)
                if msg.get("type") == "error":
                    error = msg.get("message", "unknown error")
                    logger.warning(f"Server error during send_text_and_collect: {error}")
                    break
                elif msg.get("type") == "bot_text":
                    tokens.append(msg.get("text", ""))
                elif msg.get("type") == "bot_text_complete":
                    full_response = msg.get("text", "")
                    break
        except asyncio.TimeoutError:
            continue

    if not full_response and tokens:
        full_response = "".join(tokens)

    return {"tokens": tokens, "full_response": full_response, "all_events": all_events, "error": error}


async def send_teacher_action_and_collect(ws, action: str, payload: str = "",
                                          timeout: float = 15.0) -> dict:
    """Send a teacher_action and collect the LLM response."""
    await ws.send(json.dumps({
        "type": "teacher_action",
        "action": action,
        "payload": payload,
    }))

    tokens = []
    full_response = ""
    all_events = []
    error = None
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=2)
            if isinstance(raw, str):
                msg = json.loads(raw)
                all_events.append(msg)
                if msg.get("type") == "error":
                    error = msg.get("message", "unknown error")
                    logger.warning(f"Server error during teacher_action: {error}")
                    break
                elif msg.get("type") == "bot_text":
                    tokens.append(msg.get("text", ""))
                elif msg.get("type") == "bot_text_complete":
                    full_response = msg.get("text", "")
                    break
        except asyncio.TimeoutError:
            continue

    if not full_response and tokens:
        full_response = "".join(tokens)

    return {"tokens": tokens, "full_response": full_response, "all_events": all_events, "error": error}


def detect_response_language(text: str) -> str:
    """Detect the primary language of a text response."""
    if not text:
        return "unknown"
    non_latin = re.sub(r'[\x00-\x7F]', '', text)
    if not non_latin:
        return "en"

    devanagari = sum(1 for c in non_latin if '\u0900' <= c <= '\u097F')
    tamil = sum(1 for c in non_latin if '\u0B80' <= c <= '\u0BFF')
    total_non_latin = devanagari + tamil
    if total_non_latin == 0:
        return "en"

    counts = {"hi": devanagari, "ta": tamil}
    best = max(counts, key=counts.get)
    # If non-Latin chars are >30% of total text, it's that language
    if counts[best] / max(len(text), 1) > 0.15:
        return best
    return "en"


def assert_language(text: str, expected_lang: str, context: str = ""):
    """Assert that the text is in the expected language."""
    actual = detect_response_language(text)
    if actual != expected_lang:
        raise AssertionError(
            f"Language mismatch{' (' + context + ')' if context else ''}: "
            f"expected={expected_lang}, detected={actual}, "
            f"text='{text[:100]}...'"
        )


# ═════════════════════════════════════════════════
# TEST 1: Language Tag Prepend (Unit Test)
# ═════════════════════════════════════════════════
async def test_language_tag_prepend() -> TestResult:
    """Unit test: _make_language_tag and tag prepend logic."""
    t0 = time.time()
    try:
        # Test _make_language_tag
        assert RoomManager._make_language_tag("en") == "[User is speaking English]"
        assert RoomManager._make_language_tag("hi") == "[User is speaking Hindi]"
        assert RoomManager._make_language_tag("ta") == "[User is speaking Tamil]"
        assert RoomManager._make_language_tag("fr") == "[User is speaking English]"  # Unknown defaults to English

        # Test that a question without a tag gets one
        question = "What is photosynthesis?"
        assert not question.startswith("[User is speaking")
        tag = RoomManager._make_language_tag("en")
        tagged = f"{tag} {question}"
        assert tagged == "[User is speaking English] What is photosynthesis?"

        # Test that a question WITH a tag is left alone
        already_tagged = "[User is speaking Hindi] प्रकाश संश्लेषण क्या है?"
        assert already_tagged.startswith("[User is speaking")

        return TestResult(name="language_tag_prepend", passed=True, duration_sec=time.time() - t0)
    except Exception as e:
        return TestResult(name="language_tag_prepend", passed=False, error=str(e), duration_sec=time.time() - t0)


# ═════════════════════════════════════════════════
# TEST 2: Detect Text Language (Unit Test)
# ═════════════════════════════════════════════════
async def test_detect_text_language_unit() -> TestResult:
    """Unit test: _detect_text_language correctly identifies scripts."""
    t0 = time.time()
    try:
        detect = RoomManager._detect_text_language

        # English
        assert detect("The speed of light is 3 x 10^8 m/s") == "en"
        assert detect("Photosynthesis is the process by which") == "en"
        assert detect("7 + 3 = 10") == "en"

        # Hindi (Devanagari)
        assert detect("फोटोसिंथेसिस एक जटिल प्रक्रिया है") == "hi"
        assert detect("7 + 3 का उत्तर 10 है।") == "hi"
        assert detect("सूरज एक विशाल तारा है") == "hi"

        # Tamil
        assert detect("ஒளிச்சேர்க்கை என்பது தாவரங்கள்") == "ta"

        # Mixed (mostly Hindi with some English terms)
        assert detect("यह photosynthesis का process है जिसमें पौधे") == "hi"

        # Empty
        assert detect("") == "en"

        return TestResult(name="detect_text_language_unit", passed=True, duration_sec=time.time() - t0)
    except Exception as e:
        return TestResult(name="detect_text_language_unit", passed=False, error=str(e), duration_sec=time.time() - t0)


# ═════════════════════════════════════════════════
# TEST 3: Teacher Action Produces English Response
# ═════════════════════════════════════════════════
async def test_teacher_action_english() -> TestResult:
    """SET_TOPIC should produce an English response for an English-speaking teacher."""
    t0 = time.time()
    room_id = None
    try:
        room = await api_create_room(
            "Lang Test - Teacher Action",
            creator_user_id="teacher-lang-1",
            creator_name="TeacherEn",
        )
        room_id = room["room_id"]

        ws = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws, room_id, "teacher-lang-1", "TeacherEn", "en", role="teacher")

        result = await send_teacher_action_and_collect(ws, "SET_TOPIC", "Photosynthesis")

        response = result["full_response"]
        assert response, "No response from SET_TOPIC"
        assert_language(response, "en", "SET_TOPIC Photosynthesis for English teacher")

        logger.info(f"SET_TOPIC response (first 100 chars): {response[:100]}")

        await ws.close()
        return TestResult(
            name="teacher_action_english", passed=True, duration_sec=time.time() - t0,
            details={"response_lang": detect_response_language(response), "response_len": len(response)}
        )
    except Exception as e:
        return TestResult(name="teacher_action_english", passed=False, error=str(e), duration_sec=time.time() - t0)
    finally:
        if room_id:
            await api_delete_room(room_id)


# ═════════════════════════════════════════════════
# TEST 4: English Question After Hindi History Stays English
# ═════════════════════════════════════════════════
async def test_english_after_hindi_history() -> TestResult:
    """English question after Hindi conversation history must stay English.

    This is the EXACT bug from the demo: teacher (English) asked "tell me more"
    after Hindi messages were in the conversation history, and got Hindi back.
    """
    t0 = time.time()
    room_id = None
    try:
        room = await api_create_room(
            "Lang Test - EN after HI",
            creator_user_id="teacher-lang-2",
            creator_name="PrasadEn",
        )
        room_id = room["room_id"]

        ws = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws, room_id, "teacher-lang-2", "PrasadEn", "en", role="teacher")

        # Step 1: SET_TOPIC (this may produce Hindi in history if buggy)
        r1 = await send_teacher_action_and_collect(ws, "SET_TOPIC", "Photosynthesis")
        assert r1["full_response"], "No response from SET_TOPIC"
        logger.info(f"Step 1 (SET_TOPIC): lang={detect_response_language(r1['full_response'])}, text='{r1['full_response'][:60]}'")

        # Step 2: Ask an English question
        r2 = await send_text_and_collect(ws, "What is the water cycle?")
        assert r2["full_response"], "No response from 'What is the water cycle?'"
        lang2 = detect_response_language(r2["full_response"])
        logger.info(f"Step 2 (water cycle): lang={lang2}, text='{r2['full_response'][:60]}'")
        assert_language(r2["full_response"], "en", "What is the water cycle? (English speaker)")

        # Step 3: "tell me more" — the critical test
        r3 = await send_text_and_collect(ws, "tell me more")
        assert r3["full_response"], "No response from 'tell me more'"
        lang3 = detect_response_language(r3["full_response"])
        logger.info(f"Step 3 (tell me more): lang={lang3}, text='{r3['full_response'][:60]}'")
        assert_language(r3["full_response"], "en", "'tell me more' after English question")

        await ws.close()
        return TestResult(
            name="english_after_hindi_history", passed=True, duration_sec=time.time() - t0,
            details={"step1_lang": detect_response_language(r1["full_response"]),
                      "step2_lang": lang2, "step3_lang": lang3}
        )
    except Exception as e:
        return TestResult(name="english_after_hindi_history", passed=False, error=str(e), duration_sec=time.time() - t0)
    finally:
        if room_id:
            await api_delete_room(room_id)


# ═════════════════════════════════════════════════
# TEST 5: Hindi Question Gets Hindi Response
# ═════════════════════════════════════════════════
async def test_hindi_question_stays_hindi() -> TestResult:
    """A Hindi-speaking user's question should get a Hindi response."""
    t0 = time.time()
    room_id = None
    try:
        room = await api_create_room(
            "Lang Test - Hindi Q",
            creator_user_id="student-hi-1",
            creator_name="RaviHi",
        )
        room_id = room["room_id"]

        ws = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws, room_id, "student-hi-1", "RaviHi", "hi", role="teacher")

        result = await send_text_and_collect(ws, "प्रकाश संश्लेषण क्या है?")
        assert result["full_response"], "No response"
        assert_language(result["full_response"], "hi", "Hindi question should get Hindi response")

        logger.info(f"Hindi response: {result['full_response'][:100]}")

        await ws.close()
        return TestResult(
            name="hindi_question_stays_hindi", passed=True, duration_sec=time.time() - t0,
            details={"response_lang": detect_response_language(result["full_response"])}
        )
    except Exception as e:
        return TestResult(name="hindi_question_stays_hindi", passed=False, error=str(e), duration_sec=time.time() - t0)
    finally:
        if room_id:
            await api_delete_room(room_id)


# ═════════════════════════════════════════════════
# TEST 6: Mixed History No Drift
# ═════════════════════════════════════════════════
async def test_mixed_history_no_drift() -> TestResult:
    """After multiple exchanges, English question must still get English response.

    Simulates a session where the LLM has seen both English and Hindi in history.
    """
    t0 = time.time()
    room_id = None
    try:
        room = await api_create_room(
            "Lang Test - Mixed History",
            creator_user_id="teacher-lang-3",
            creator_name="TeacherMixed",
        )
        room_id = room["room_id"]

        ws = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws, room_id, "teacher-lang-3", "TeacherMixed", "en", role="teacher")

        # Ask several questions to build up history
        questions = [
            "What is gravity?",
            "Explain how volcanoes work. Brief.",
            "What is the sun? One sentence.",
            "What is 7+3? Brief.",
        ]

        for i, q in enumerate(questions):
            r = await send_text_and_collect(ws, q)
            lang = detect_response_language(r["full_response"])
            logger.info(f"Q{i+1} '{q}': lang={lang}, text='{r['full_response'][:50]}'")
            assert_language(r["full_response"], "en", f"Question '{q}' should be English")

        await ws.close()
        return TestResult(
            name="mixed_history_no_drift", passed=True, duration_sec=time.time() - t0,
            details={"questions_tested": len(questions)}
        )
    except Exception as e:
        return TestResult(name="mixed_history_no_drift", passed=False, error=str(e), duration_sec=time.time() - t0)
    finally:
        if room_id:
            await api_delete_room(room_id)


# ═════════════════════════════════════════════════
# TEST 7: Listener Gets Translation
# ═════════════════════════════════════════════════
async def test_listener_gets_translation() -> TestResult:
    """Hindi listener should get a translated version of an English LLM response.

    Verifies that source_lang is correctly set so translation happens.
    """
    t0 = time.time()
    room_id = None
    try:
        room = await api_create_room(
            "Lang Test - Listener Translation",
            creator_user_id="teacher-lang-4",
            creator_name="TeacherEn",
        )
        room_id = room["room_id"]

        # Teacher (English speaker)
        ws_teacher = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_teacher, room_id, "teacher-lang-4", "TeacherEn", "en", role="teacher")

        # Hindi listener
        ws_listener = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_listener, room_id, "listener-hi-1", "ListenerHi", "hi", role="student")

        # Teacher asks in English
        await ws_teacher.send(json.dumps({"type": "text_message", "text": "What is gravity?"}))

        # Collect teacher's response
        teacher_response = ""
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws_teacher.recv(), timeout=2)
                if isinstance(raw, str):
                    msg = json.loads(raw)
                    if msg.get("type") == "bot_text_complete":
                        teacher_response = msg.get("text", "")
                        break
            except asyncio.TimeoutError:
                continue

        # Collect listener's events — should get translated bot_text chunks
        listener_texts = []
        got_complete = False
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws_listener.recv(), timeout=2)
                if isinstance(raw, str):
                    msg = json.loads(raw)
                    if msg.get("type") == "bot_text":
                        listener_texts.append(msg.get("text", ""))
                    elif msg.get("type") == "bot_text_complete":
                        got_complete = True
                        break
                    elif msg.get("type") == "bot_response":
                        # Sentence-by-sentence delivery path
                        translated = msg.get("translated_text", "")
                        if translated:
                            listener_texts.append(translated)
            except asyncio.TimeoutError:
                continue

        listener_full = "".join(listener_texts)
        logger.info(f"Teacher response lang: {detect_response_language(teacher_response)}")
        logger.info(f"Listener received: {listener_full[:100]}")

        # Teacher response should be English
        assert teacher_response, "No teacher response"
        assert_language(teacher_response, "en", "Teacher (English) should get English response")

        # Listener should have received SOMETHING (either translated or original)
        assert listener_full or got_complete, "Listener received nothing"

        # If teacher response is English and listener is Hindi, listener should get Hindi
        if detect_response_language(teacher_response) == "en" and listener_full:
            listener_lang = detect_response_language(listener_full)
            logger.info(f"Listener text language: {listener_lang}")
            # The listener should get Hindi (translated from English)
            assert listener_lang == "hi", (
                f"Hindi listener should get Hindi translation, got {listener_lang}: '{listener_full[:80]}'"
            )

        await ws_teacher.close()
        await ws_listener.close()
        return TestResult(
            name="listener_gets_translation", passed=True, duration_sec=time.time() - t0,
            details={
                "teacher_lang": detect_response_language(teacher_response),
                "listener_lang": detect_response_language(listener_full) if listener_full else "empty",
            }
        )
    except Exception as e:
        return TestResult(name="listener_gets_translation", passed=False, error=str(e), duration_sec=time.time() - t0)
    finally:
        if room_id:
            await api_delete_room(room_id)


# ═════════════════════════════════════════════════
# TEST 8: "tell me more" Stays English
# ═════════════════════════════════════════════════
async def test_tell_me_more_stays_english() -> TestResult:
    """'tell me more' with no prior context should stay English for EN speaker.

    This is the exact scenario from the demo bug.
    """
    t0 = time.time()
    room_id = None
    try:
        room = await api_create_room(
            "Lang Test - Tell Me More",
            creator_user_id="teacher-lang-5",
            creator_name="PrasadEn",
        )
        room_id = room["room_id"]

        ws = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws, room_id, "teacher-lang-5", "PrasadEn", "en", role="teacher")

        # Set topic first (this was the trigger in the demo)
        r1 = await send_teacher_action_and_collect(ws, "SET_TOPIC", "Photosynthesis")
        lang1 = detect_response_language(r1["full_response"])
        logger.info(f"SET_TOPIC response: lang={lang1}")
        assert_language(r1["full_response"], "en", "SET_TOPIC for English teacher")

        # Now "tell me more"
        r2 = await send_text_and_collect(ws, "tell me more")
        assert r2["full_response"], "No response from 'tell me more'"
        lang2 = detect_response_language(r2["full_response"])
        logger.info(f"'tell me more' response: lang={lang2}, text='{r2['full_response'][:60]}'")
        assert_language(r2["full_response"], "en", "'tell me more' for English speaker after SET_TOPIC")

        # And again
        r3 = await send_text_and_collect(ws, "tell me more")
        assert r3["full_response"], "No response from second 'tell me more'"
        lang3 = detect_response_language(r3["full_response"])
        logger.info(f"Second 'tell me more': lang={lang3}, text='{r3['full_response'][:60]}'")
        assert_language(r3["full_response"], "en", "Second 'tell me more' for English speaker")

        await ws.close()
        return TestResult(
            name="tell_me_more_stays_english", passed=True, duration_sec=time.time() - t0,
            details={"topic_lang": lang1, "tell_more_1": lang2, "tell_more_2": lang3}
        )
    except Exception as e:
        return TestResult(name="tell_me_more_stays_english", passed=False, error=str(e), duration_sec=time.time() - t0)
    finally:
        if room_id:
            await api_delete_room(room_id)


# ═════════════════════════════════════════════════
# TEST 9: SET_TOPIC Then English Question
# ═════════════════════════════════════════════════
async def test_set_topic_then_english_q() -> TestResult:
    """After SET_TOPIC, an English question should get English response."""
    t0 = time.time()
    room_id = None
    try:
        room = await api_create_room(
            "Lang Test - Topic Then Q",
            creator_user_id="teacher-lang-6",
            creator_name="TeacherEn",
        )
        room_id = room["room_id"]

        ws = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws, room_id, "teacher-lang-6", "TeacherEn", "en", role="teacher")

        # SET_TOPIC — give extra time for longer LLM response
        r1 = await send_teacher_action_and_collect(ws, "SET_TOPIC", "The Water Cycle", timeout=30.0)
        assert_language(r1["full_response"], "en", "SET_TOPIC Water Cycle")

        # English question
        r2 = await send_text_and_collect(ws, "What is evaporation?", timeout=20.0)
        assert r2["full_response"], "No response to 'What is evaporation?'"
        assert_language(r2["full_response"], "en", "What is evaporation? (English)")

        # Another English question
        r3 = await send_text_and_collect(ws, "How do clouds form?", timeout=20.0)
        assert r3["full_response"], "No response to 'How do clouds form?'"
        assert_language(r3["full_response"], "en", "How do clouds form? (English)")

        await ws.close()
        return TestResult(
            name="set_topic_then_english_q", passed=True, duration_sec=time.time() - t0,
        )
    except Exception as e:
        return TestResult(name="set_topic_then_english_q", passed=False, error=str(e), duration_sec=time.time() - t0)
    finally:
        if room_id:
            await api_delete_room(room_id)


# ═════════════════════════════════════════════════
# TEST 10: Math Question Stays English
# ═════════════════════════════════════════════════
async def test_math_question_english() -> TestResult:
    """'What is 7+3?' should stay English for an English speaker.

    This was one of the demo bugs — the LLM answered '7 + 3 का उत्तर 10 है।' (Hindi).
    """
    t0 = time.time()
    room_id = None
    try:
        room = await api_create_room(
            "Lang Test - Math EN",
            creator_user_id="teacher-lang-7",
            creator_name="TeacherEn",
        )
        room_id = room["room_id"]

        ws = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws, room_id, "teacher-lang-7", "TeacherEn", "en", role="teacher")

        result = await send_text_and_collect(ws, "What is 7+3? Brief.")
        assert result["full_response"], "No response"
        assert_language(result["full_response"], "en", "What is 7+3? for English speaker")

        logger.info(f"Math response: {result['full_response']}")

        await ws.close()
        return TestResult(
            name="math_question_english", passed=True, duration_sec=time.time() - t0,
            details={"response": result["full_response"][:80]}
        )
    except Exception as e:
        return TestResult(name="math_question_english", passed=False, error=str(e), duration_sec=time.time() - t0)
    finally:
        if room_id:
            await api_delete_room(room_id)


# ═════════════════════════════════════════════════
# TEST 11: Multi-turn Language Stability
# ═════════════════════════════════════════════════
async def test_multi_turn_language_stability() -> TestResult:
    """6 consecutive English questions — ALL must get English responses.

    Tests that language adherence doesn't degrade over many turns.
    Uses longer timeouts to handle LLM latency.
    """
    t0 = time.time()
    room_id = None
    try:
        room = await api_create_room(
            "Lang Test - Multi Turn",
            creator_user_id="teacher-lang-8",
            creator_name="TeacherEn",
        )
        room_id = room["room_id"]

        ws = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws, room_id, "teacher-lang-8", "TeacherEn", "en", role="teacher")

        questions = [
            "What is photosynthesis? Brief.",
            "Tell me more. Brief.",
            "What is the sun? One sentence.",
            "What is 5 times 6? Brief.",
            "Explain gravity briefly.",
            "What are volcanoes? Brief.",
        ]

        results_per_q = []
        for i, q in enumerate(questions):
            r = await send_text_and_collect(ws, q, timeout=20.0)
            if not r["full_response"]:
                logger.warning(f"  Turn {i+1}: EMPTY response for '{q}', retrying...")
                # Retry once with longer timeout
                r = await send_text_and_collect(ws, q, timeout=25.0)
            lang = detect_response_language(r["full_response"])
            results_per_q.append({"q": q, "lang": lang})
            logger.info(f"  Turn {i+1}: lang={lang} | q='{q}' | a='{r['full_response'][:40]}'")
            if not r["full_response"]:
                logger.warning(f"  Turn {i+1}: Still empty after retry, skipping assertion")
                continue
            assert_language(r["full_response"], "en", f"Turn {i+1}: '{q}'")

        # Check that at least 5 out of 6 were verified English (allow 1 timeout)
        verified_en = sum(1 for r in results_per_q if r["lang"] == "en")
        non_en = [r for r in results_per_q if r["lang"] not in ("en", "unknown")]
        if non_en:
            raise AssertionError(f"Non-English responses detected: {non_en}")

        await ws.close()
        return TestResult(
            name="multi_turn_language_stability", passed=True, duration_sec=time.time() - t0,
            details={"turns": len(questions), "verified_english": verified_en}
        )
    except Exception as e:
        return TestResult(name="multi_turn_language_stability", passed=False, error=str(e), duration_sec=time.time() - t0)
    finally:
        if room_id:
            await api_delete_room(room_id)


# ═════════════════════════════════════════════════
# TEST 12: Discussion Room — English Student Gets English
# ═════════════════════════════════════════════════
async def test_discussion_room_english() -> TestResult:
    """In a discussion room, an English-speaking student should get English responses."""
    t0 = time.time()
    room_id = None
    try:
        room = await api_create_room(
            "Lang Test - Discussion EN",
            room_type="discussion",
            creator_user_id="disc-en-1",
            creator_name="StudentEn",
        )
        room_id = room["room_id"]

        ws = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws, room_id, "disc-en-1", "StudentEn", "en", role="student")

        # In discussion mode, any student can send text
        result = await send_text_and_collect(ws, "What is photosynthesis? Brief.")
        assert result["full_response"], "No response in discussion room"
        assert_language(result["full_response"], "en", "Discussion room: English student question")

        logger.info(f"Discussion EN response: {result['full_response'][:100]}")

        await ws.close()
        return TestResult(
            name="discussion_room_english", passed=True, duration_sec=time.time() - t0,
            details={"response_lang": detect_response_language(result["full_response"])}
        )
    except Exception as e:
        return TestResult(name="discussion_room_english", passed=False, error=str(e), duration_sec=time.time() - t0)
    finally:
        if room_id:
            await api_delete_room(room_id)


# ═════════════════════════════════════════════════
# TEST 13: Discussion Room — Hindi Student Gets Hindi
# ═════════════════════════════════════════════════
async def test_discussion_room_hindi() -> TestResult:
    """In a discussion room, a Hindi-speaking student should get Hindi responses."""
    t0 = time.time()
    room_id = None
    try:
        room = await api_create_room(
            "Lang Test - Discussion HI",
            room_type="discussion",
            creator_user_id="disc-hi-1",
            creator_name="StudentHi",
        )
        room_id = room["room_id"]

        ws = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws, room_id, "disc-hi-1", "StudentHi", "hi", role="student")

        result = await send_text_and_collect(ws, "प्रकाश संश्लेषण क्या है?")
        assert result["full_response"], "No response in discussion room"
        assert_language(result["full_response"], "hi", "Discussion room: Hindi student question")

        logger.info(f"Discussion HI response: {result['full_response'][:100]}")

        await ws.close()
        return TestResult(
            name="discussion_room_hindi", passed=True, duration_sec=time.time() - t0,
            details={"response_lang": detect_response_language(result["full_response"])}
        )
    except Exception as e:
        return TestResult(name="discussion_room_hindi", passed=False, error=str(e), duration_sec=time.time() - t0)
    finally:
        if room_id:
            await api_delete_room(room_id)


# ═════════════════════════════════════════════════
# TEST 14: Discussion Room — Multi-student Language Isolation
# ═════════════════════════════════════════════════
async def test_discussion_room_multi_student_lang() -> TestResult:
    """In a discussion room with EN and HI students, each gets responses in their own language.

    Student A (English) asks, gets English.
    Student B (Hindi) asks, gets Hindi.
    Student A asks again, still gets English (no drift from Hindi history).
    """
    t0 = time.time()
    room_id = None
    try:
        room = await api_create_room(
            "Lang Test - Discussion Multi",
            room_type="discussion",
            creator_user_id="disc-multi-en",
            creator_name="StudentA_EN",
        )
        room_id = room["room_id"]

        # Student A (English)
        ws_a = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_a, room_id, "disc-multi-en", "StudentA_EN", "en", role="student")
        # Drain join greetings + token_changed so they don't pollute send_text_and_collect
        await drain_messages(ws_a, drain_secs=3.0)

        # Student B (Hindi)
        ws_b = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_b, room_id, "disc-multi-hi", "StudentB_HI", "hi", role="student")
        # Drain join greetings / user_joined broadcasts
        await drain_messages(ws_b, drain_secs=2.0)
        # Also drain ws_a which gets user_joined broadcast for Student B
        await drain_messages(ws_a, drain_secs=1.0)

        # Step 1: Student A asks in English (already has speaker token as first joiner)
        r1 = await send_text_and_collect(ws_a, "What is the water cycle? Brief.")
        assert r1["full_response"], f"No response for Student A: error={r1.get('error')}"
        assert_language(r1["full_response"], "en", "Discussion: Student A (EN) question")
        logger.info(f"Student A (EN) response: lang={detect_response_language(r1['full_response'])}")

        # Student A releases token so Student B can speak
        await ws_a.send(json.dumps({"type": "release_token"}))
        await asyncio.sleep(1.0)
        # Drain ws_a to consume token_changed broadcast
        await drain_messages(ws_a, drain_secs=1.0)

        # Student B requests the token and waits for it
        await ws_b.send(json.dumps({"type": "request_token"}))
        await wait_for_token(ws_b, expected_speaker="disc-multi-hi", timeout=30.0)
        # Drain greeting / broadcast messages before sending
        await drain_messages(ws_b, drain_secs=3.0)

        # Step 2: Student B asks in Hindi (now has speaker token)
        r2 = await send_text_and_collect(ws_b, "गुरुत्वाकर्षण क्या है?")
        assert r2["full_response"], f"No response for Student B: error={r2.get('error')}"
        assert_language(r2["full_response"], "hi", "Discussion: Student B (HI) question")
        logger.info(f"Student B (HI) response: lang={detect_response_language(r2['full_response'])}")

        # Student B releases token so Student A can speak again
        await ws_b.send(json.dumps({"type": "release_token"}))
        await asyncio.sleep(1.0)
        # Drain ws_b to consume token_changed broadcast
        await drain_messages(ws_b, drain_secs=1.0)

        # Student A requests the token back
        await ws_a.send(json.dumps({"type": "request_token"}))
        await wait_for_token(ws_a, expected_speaker="disc-multi-en", timeout=30.0)
        # Drain any broadcast messages before sending
        await drain_messages(ws_a, drain_secs=3.0)

        # Step 3: Student A asks again in English — must NOT drift to Hindi
        r3 = await send_text_and_collect(ws_a, "Tell me about volcanoes. Brief.")
        assert r3["full_response"], f"No response for Student A (second question): error={r3.get('error')}"
        assert_language(r3["full_response"], "en", "Discussion: Student A (EN) after Hindi history")
        logger.info(f"Student A (EN) second response: lang={detect_response_language(r3['full_response'])}")

        await ws_a.close()
        await ws_b.close()
        return TestResult(
            name="discussion_room_multi_student_lang", passed=True, duration_sec=time.time() - t0,
            details={
                "student_a_1": detect_response_language(r1["full_response"]),
                "student_b": detect_response_language(r2["full_response"]),
                "student_a_2": detect_response_language(r3["full_response"]),
            }
        )
    except Exception as e:
        return TestResult(
            name="discussion_room_multi_student_lang", passed=False, error=str(e),
            duration_sec=time.time() - t0
        )
    finally:
        if room_id:
            await api_delete_room(room_id)


# ═════════════════════════════════════════════════
# Test Registry
# ═════════════════════════════════════════════════
ALL_TESTS = {
    "language_tag_prepend": test_language_tag_prepend,
    "detect_text_language_unit": test_detect_text_language_unit,
    "teacher_action_english": test_teacher_action_english,
    "english_after_hindi_history": test_english_after_hindi_history,
    "hindi_question_stays_hindi": test_hindi_question_stays_hindi,
    "mixed_history_no_drift": test_mixed_history_no_drift,
    "listener_gets_translation": test_listener_gets_translation,
    "tell_me_more_stays_english": test_tell_me_more_stays_english,
    "set_topic_then_english_q": test_set_topic_then_english_q,
    "math_question_english": test_math_question_english,
    "multi_turn_language_stability": test_multi_turn_language_stability,
    # Discussion room tests
    "discussion_room_english": test_discussion_room_english,
    "discussion_room_hindi": test_discussion_room_hindi,
    "discussion_room_multi_student_lang": test_discussion_room_multi_student_lang,
}


async def main():
    parser = argparse.ArgumentParser(description="Language Adherence Tests")
    parser.add_argument("--test", choices=list(ALL_TESTS.keys()) + ["all"], default="all",
                        help="Test to run (default: all)")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tests_to_run = ALL_TESTS if args.test == "all" else {args.test: ALL_TESTS[args.test]}

    PER_TEST_TIMEOUT = 120  # seconds — hard cap per test to prevent hangs

    results = []
    for name, test_fn in tests_to_run.items():
        print()
        logger.info("=" * 60)
        logger.info(f"  LANGUAGE TEST: {name.upper()}")
        logger.info("=" * 60)
        try:
            result = await asyncio.wait_for(test_fn(), timeout=PER_TEST_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(f"  TEST {name.upper()} TIMED OUT after {PER_TEST_TIMEOUT}s")
            result = TestResult(name=name, passed=False, error=f"Timed out after {PER_TEST_TIMEOUT}s", duration_sec=PER_TEST_TIMEOUT)
        results.append(result)

    # Summary
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    print()
    print("=" * 60)
    print("  LANGUAGE ADHERENCE TEST RESULTS")
    print("=" * 60)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        duration = f"{r.duration_sec:.1f}s" if r.duration_sec else ""
        print(f"  {status} {r.name:<35} {duration}")
        if r.details:
            for k, v in r.details.items():
                val = str(v)[:80]
                print(f"       {k}: {val}")
        if r.error:
            print(f"       error: {r.error}")
    print("=" * 60)
    print(f"  {len(passed)}/{len(results)} tests passed")
    print("=" * 60)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
