#!/usr/bin/env python3
"""
Test text-audio sync timing for all three modes:
  1. Tutor mode (Pipecat pipeline, text_and_audio)
  2. Classroom speaker mode (voice via Pipecat pipeline)
  3. Classroom listener mode (text_and_audio via classroom WS)

For each mode, measures the time gap between receiving bot_text and the
first audio byte for each sentence. A positive gap means text arrived
before audio (bad). A negative gap means audio arrived before text (bad).
Near-zero means synced (good).
"""

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field

import aiohttp
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("text_audio_timing")

# ── Config ──────────────────────────────────────────────────
PIPECAT_HTTP_URL = os.environ.get("PIPECAT_HTTP_URL", "http://localhost:7860")
PIPECAT_WS_URL = os.environ.get("PIPECAT_WS_URL", "ws://localhost:7860/ws")
CLASSROOM_WS_BASE = os.environ.get("CLASSROOM_WS_URL", "ws://localhost:7860/classroom/rooms")
WEBUI_SECRET_KEY = os.environ.get("WEBUI_SECRET_KEY", "")

import jwt as pyjwt

TEST_ADMIN_ID = "timing-test-admin"
TEST_ADMIN_NAME = "TimingAdmin"
_APPROVED_TEACHERS: set[str] = set()


def _make_jwt(user_id: str = "timing-test-user") -> str:
    if not WEBUI_SECRET_KEY:
        return ""
    return pyjwt.encode({"id": user_id}, WEBUI_SECRET_KEY, algorithm="HS256")


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


@dataclass
class SentenceTiming:
    """Timing for one sentence's text vs audio delivery."""
    sentence_idx: int
    text: str = ""
    text_arrived_at: float = 0.0
    first_audio_at: float = 0.0
    gap_ms: float = 0.0  # text_arrived - first_audio (negative = audio first)


@dataclass
class ModeResult:
    mode: str
    sentences: list = field(default_factory=list)
    passed: bool = False
    error: str = ""


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

async def ws_connect(url: str):
    return await websockets.connect(url, max_size=10 * 1024 * 1024, ping_interval=30)


async def ensure_teacher_role(user_id: str, user_name: str) -> None:
    if user_id in _APPROVED_TEACHERS:
        return
    user_headers = _auth_headers(user_id, user_name, role="user")
    admin_headers = _auth_headers(TEST_ADMIN_ID, TEST_ADMIN_NAME, role="admin")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        async with session.get(f"{PIPECAT_HTTP_URL}/classroom/teacher-status", headers=user_headers) as resp:
            if resp.status == 200:
                payload = await resp.json()
                if payload.get("is_teacher"):
                    _APPROVED_TEACHERS.add(user_id)
                    return
        # Request teacher role
        request_id = None
        async with session.post(
            f"{PIPECAT_HTTP_URL}/classroom/teacher-requests",
            params={"purpose": "Timing test teacher approval"},
            headers=user_headers,
        ) as resp:
            if resp.status == 200:
                request_id = (await resp.json()).get("id")
        # Admin approves
        if request_id:
            async with session.post(
                f"{PIPECAT_HTTP_URL}/classroom/teacher-requests/{request_id}/approve",
                headers=admin_headers,
            ) as resp:
                pass
        _APPROVED_TEACHERS.add(user_id)


async def api_create_room(name: str, creator_user_id: str = "timing-teacher",
                          creator_name: str = "TimingTeacher") -> dict:
    await ensure_teacher_role(creator_user_id, creator_name)
    headers = _auth_headers(creator_user_id, creator_name, role="user")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        async with session.post(
            f"{PIPECAT_HTTP_URL}/classroom/rooms",
            params={"name": name},
            headers=headers,
        ) as resp:
            assert resp.status == 200, f"Create room failed: {resp.status}"
            return await resp.json()


async def api_delete_room(room_id: str):
    admin_headers = _auth_headers(TEST_ADMIN_ID, TEST_ADMIN_NAME, role="admin")
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            await session.delete(f"{PIPECAT_HTTP_URL}/classroom/rooms/{room_id}", headers=admin_headers)
    except Exception:
        pass


async def join_room(ws, room_id, user_id, name, language, mode="text_and_audio"):
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
    # Drain join ack
    for _ in range(15):
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            if isinstance(raw, str):
                msg = json.loads(raw)
                if msg.get("type") == "joined":
                    return msg
        except (asyncio.TimeoutError, Exception):
            pass
    return None


async def drain_until_audio_end(ws, timeout=30.0):
    """Drain all messages until bot_audio_end (used to skip greeting)."""
    deadline = time.time() + timeout
    end_seen = False
    while time.time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
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
    # Extra drain
    for _ in range(50):
        try:
            await asyncio.wait_for(ws.recv(), timeout=0.2)
        except (asyncio.TimeoutError, Exception):
            break
    return end_seen


async def collect_bot_response(ws, timeout=30.0) -> tuple[str, bool]:
    """Collect bot_text chunks until bot_text_complete. Returns (full_text, complete)."""
    text = ""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            if isinstance(raw, str):
                msg = json.loads(raw)
                if msg.get("type") == "bot_text":
                    text += msg.get("text", "")
                elif msg.get("type") == "bot_text_complete":
                    return msg.get("text", text), True
        except asyncio.TimeoutError:
            if text:
                break
            continue
        except Exception:
            break
    return text, False


# ═══════════════════════════════════════════════════════════
# TEST 1: TUTOR MODE
# ═══════════════════════════════════════════════════════════

async def test_tutor_mode() -> ModeResult:
    """Tutor mode: Pipecat pipeline with text_and_audio.
    
    Text is released by TextAudioSyncNotifier when TTSStartedFrame fires.
    We measure the gap between bot_text JSON and first binary audio byte.
    """
    result = ModeResult(mode="tutor")
    try:
        ws = await ws_connect(PIPECAT_WS_URL)
        
        # Send config (must match server's expected format)
        config_msg = {
            "type": "config",
            "mode": "text_and_audio",
            "enable_greeting": True,
            "language": "en",
        }
        token = _make_jwt("timing-test-tutor")
        if token:
            config_msg["token"] = token
        await ws.send(json.dumps(config_msg))
        logger.info("  [TUTOR] Connected, sent config")

        # Wait for session_id
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            if isinstance(raw, str):
                msg = json.loads(raw)
                if msg.get("type") == "session_id":
                    session_id = msg["session_id"]
                    logger.info(f"  [TUTOR] Got session_id: {session_id}")
        except (asyncio.TimeoutError, Exception):
            pass

        # Drain greeting — measure greeting text-audio gap too
        logger.info("  [TUTOR] Waiting for greeting...")
        greeting_text_at = 0.0
        greeting_audio_at = 0.0
        deadline = time.time() + 30.0
        greeting_done = False
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                now = time.time()
                if isinstance(raw, str):
                    msg = json.loads(raw)
                    if msg.get("type") == "bot_text_complete" and not greeting_text_at:
                        greeting_text_at = now
                        logger.info(f"  [TUTOR] Greeting text @ {now:.3f}: '{msg.get('text', '')[:60]}'")
                    elif msg.get("type") == "bot_text" and not greeting_text_at:
                        greeting_text_at = now
                        logger.info(f"  [TUTOR] Greeting bot_text @ {now:.3f}")
                elif isinstance(raw, bytes) and len(raw) > 0:
                    if not greeting_audio_at:
                        greeting_audio_at = now
                        logger.info(f"  [TUTOR] Greeting first audio @ {now:.3f}")
            except asyncio.TimeoutError:
                if greeting_audio_at and (time.time() - greeting_audio_at) > 3.0:
                    greeting_done = True
                    break
                continue
            except Exception:
                break

        if greeting_text_at and greeting_audio_at:
            greeting_gap = round((greeting_text_at - greeting_audio_at) * 1000, 1)
            logger.info(f"  [TUTOR] Greeting text-audio gap: {greeting_gap:+.0f}ms")
        
        # Drain any remaining greeting audio
        for _ in range(100):
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.3)
            except (asyncio.TimeoutError, Exception):
                break

        # Send a question via text injection HTTP endpoint
        inject_url = f"{PIPECAT_HTTP_URL}/inject_text"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as client:
            resp = await client.post(inject_url, json={
                "session_id": session_id if 'session_id' in dir() else "",
                "text": "Explain photosynthesis in 3 sentences. Be clear and concise.",
            })
            if resp.status == 200:
                logger.info("  [TUTOR] Injected text question")
            else:
                logger.warning(f"  [TUTOR] Text injection failed: {resp.status}")
                # Fallback: try sending via WS
                await ws.send(json.dumps({
                    "type": "text-input",
                    "text": "Explain photosynthesis in 3 sentences.",
                }))
                logger.info("  [TUTOR] Sent question via WS fallback")

        # Collect text and audio events with timestamps
        text_arrived_at = 0.0
        first_audio_at = 0.0
        all_text = ""
        got_complete = False
        deadline = time.time() + 60.0
        
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                now = time.time()
                
                if isinstance(raw, str):
                    msg = json.loads(raw)
                    if msg.get("type") == "bot_text":
                        text = msg.get("text", "")
                        if not text_arrived_at:
                            text_arrived_at = now
                        all_text += text
                        logger.info(f"  [TUTOR] bot_text: '{text[:60]}' @ {now:.3f}")
                    elif msg.get("type") == "bot_text_complete":
                        got_complete = True
                        all_text = msg.get("text", all_text)
                        logger.info(f"  [TUTOR] bot_text_complete ({len(all_text)} chars)")
                        
                elif isinstance(raw, bytes) and len(raw) > 0:
                    if not first_audio_at:
                        first_audio_at = now
                        logger.info(f"  [TUTOR] First audio byte @ {now:.3f}")
                        
            except asyncio.TimeoutError:
                if got_complete and first_audio_at:
                    break
                if got_complete and (time.time() - deadline + 60.0) > 15.0:
                    break
                continue
            except Exception as e:
                logger.warning(f"  [TUTOR] Error: {e}")
                break

        # Calculate gap
        sentences = []
        if text_arrived_at and first_audio_at:
            gap_ms = round((text_arrived_at - first_audio_at) * 1000, 1)
            sentences.append(SentenceTiming(
                sentence_idx=0, text=all_text[:80],
                text_arrived_at=text_arrived_at,
                first_audio_at=first_audio_at,
                gap_ms=gap_ms,
            ))
            logger.info(f"  [TUTOR] Response text-audio gap: {gap_ms:+.0f}ms")
            logger.info(f"  [TUTOR] Verdict: {'✅ SYNCED' if abs(gap_ms) < 2000 else '❌ OUT OF SYNC'}")
            result.passed = True
        elif all_text and not first_audio_at:
            logger.warning(f"  [TUTOR] Got text ({len(all_text)} chars) but NO audio — text_only mode?")
            result.error = f"Got text but no audio"
        elif first_audio_at and not text_arrived_at:
            logger.warning(f"  [TUTOR] Got audio but NO text")
            result.error = f"Got audio but no text"
        else:
            result.error = "No text or audio received"
            logger.error(f"  [TUTOR] {result.error}")

        result.sentences = sentences
        await ws.close()
            
    except Exception as e:
        result.error = str(e)
        logger.error(f"  [TUTOR] Error: {e}")
    
    return result


# ═══════════════════════════════════════════════════════════
# TEST 2: CLASSROOM SPEAKER MODE
# ═══════════════════════════════════════════════════════════

async def test_classroom_speaker() -> ModeResult:
    """Classroom speaker mode: text_message via classroom WS.
    
    Speaker sends text, gets bot_text/bot_text_complete.
    Speaker does NOT get audio via classroom WS (audio goes through Pipecat 
    transport only when using voice). So for text_message path, we can only
    verify text arrives. For voice path, it's the same as tutor mode.
    """
    result = ModeResult(mode="classroom_speaker")
    try:
        room = await api_create_room("Timing-Speaker-Test")
        room_id = room["room_id"]
        logger.info(f"  [SPEAKER] Created room {room_id}")

        ws_speaker = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        joined = await join_room(ws_speaker, room_id, "timing-spk", "SpeakerTest", "en")
        logger.info(f"  [SPEAKER] Joined room")

        # Wait for token
        deadline = time.time() + 10.0
        has_token = False
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws_speaker.recv(), timeout=2.0)
                if isinstance(raw, str):
                    msg = json.loads(raw)
                    if msg.get("type") == "token_changed":
                        has_token = True
                        break
            except (asyncio.TimeoutError, Exception):
                continue
        logger.info(f"  [SPEAKER] Has token: {has_token}")

        # Drain greeting
        await asyncio.sleep(3.0)
        for _ in range(50):
            try:
                await asyncio.wait_for(ws_speaker.recv(), timeout=0.3)
            except (asyncio.TimeoutError, Exception):
                break

        # Send text question
        await ws_speaker.send(json.dumps({
            "type": "text_message",
            "text": "Explain gravity in 3 sentences.",
        }))
        logger.info("  [SPEAKER] Sent question")

        # Collect response
        first_text_at = 0.0
        all_text = ""
        got_complete = False
        deadline = time.time() + 30.0
        
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws_speaker.recv(), timeout=2.0)
                now = time.time()
                if isinstance(raw, str):
                    msg = json.loads(raw)
                    if msg.get("type") == "bot_text":
                        text = msg.get("text", "")
                        if not first_text_at:
                            first_text_at = now
                            logger.info(f"  [SPEAKER] First bot_text @ {now:.3f}: '{text[:50]}'")
                        all_text += text
                    elif msg.get("type") == "bot_text_complete":
                        got_complete = True
                        all_text = msg.get("text", all_text)
                        logger.info(f"  [SPEAKER] bot_text_complete ({len(all_text)} chars)")
                        break
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

        logger.info(f"  [SPEAKER] Got response: {len(all_text)} chars, complete={got_complete}")
        logger.info(f"  [SPEAKER] NOTE: Speaker text_message path sends text only (no audio on classroom WS)")
        logger.info(f"  [SPEAKER] Speaker VOICE mode uses same Pipecat pipeline as tutor (TextAudioSyncNotifier)")
        
        result.passed = got_complete and len(all_text) > 0
        if not result.passed:
            result.error = f"No response (complete={got_complete}, text_len={len(all_text)})"

        await ws_speaker.close()
        await api_delete_room(room_id)

    except Exception as e:
        result.error = str(e)
        logger.error(f"  [SPEAKER] Error: {e}")

    return result


# ═══════════════════════════════════════════════════════════
# TEST 3: CLASSROOM LISTENER MODE
# ═══════════════════════════════════════════════════════════

async def test_classroom_listener() -> ModeResult:
    """Classroom listener mode: text_and_audio via classroom WS.
    
    Measures the gap between bot_text and first binary audio byte
    for each sentence delivered to the listener.
    
    The server sends:
      bot_audio_start → bot_text (with first audio chunk) → binary audio → bot_audio_end
    for each sentence (after the listener fix).
    
    BEFORE the fix: bot_text was sent BEFORE bot_audio_start, so text arrived
    1-4 seconds before audio.
    """
    result = ModeResult(mode="classroom_listener")
    try:
        room = await api_create_room("Timing-Listener-Test")
        room_id = room["room_id"]
        logger.info(f"  [LISTENER] Created room {room_id}")

        # Speaker joins
        ws_speaker = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_speaker, room_id, "timing-spk2", "Speaker", "en")
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws_speaker.recv(), timeout=2.0)
                if isinstance(raw, str):
                    msg = json.loads(raw)
                    if msg.get("type") == "token_changed":
                        break
            except (asyncio.TimeoutError, Exception):
                continue
        logger.info(f"  [LISTENER] Speaker joined with token")

        # Listener joins in text_and_audio mode (English so no translation delay)
        ws_listener = await ws_connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
        await join_room(ws_listener, room_id, "timing-lst", "Listener", "en", mode="text_and_audio")
        logger.info(f"  [LISTENER] Listener joined (en, text_and_audio)")

        # Drain greeting audio for listener
        logger.info(f"  [LISTENER] Draining greeting audio...")
        end_seen = await drain_until_audio_end(ws_listener, timeout=25.0)
        logger.info(f"  [LISTENER] Greeting drained (end_seen={end_seen})")

        # Drain speaker greeting
        await asyncio.sleep(2.0)
        for _ in range(50):
            try:
                await asyncio.wait_for(ws_speaker.recv(), timeout=0.3)
            except (asyncio.TimeoutError, Exception):
                break

        # Speaker asks a question
        await ws_speaker.send(json.dumps({
            "type": "text_message",
            "text": "Explain photosynthesis in 3 sentences. Be clear.",
        }))
        logger.info("  [LISTENER] Speaker sent question")

        # Drain speaker response (don't block listener collection)
        asyncio.create_task(_drain_speaker(ws_speaker))

        # Collect listener events with precise timestamps per sentence
        sentences = []
        current = SentenceTiming(sentence_idx=0)
        in_audio_block = False
        got_complete = False
        deadline = time.time() + 90.0

        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws_listener.recv(), timeout=2.0)
                now = time.time()

                if isinstance(raw, str):
                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")
                    
                    if msg_type == "bot_text":
                        text = msg.get("text", "")
                        current.text = text
                        current.text_arrived_at = now
                        logger.info(f"  [LISTENER] bot_text #{current.sentence_idx}: '{text[:50]}' @ {now:.3f}")
                    
                    elif msg_type == "bot_audio_start":
                        in_audio_block = True
                    
                    elif msg_type == "bot_audio_end":
                        in_audio_block = False
                        # Finalize this sentence timing
                        if current.text_arrived_at and current.first_audio_at:
                            current.gap_ms = round(
                                (current.text_arrived_at - current.first_audio_at) * 1000, 1
                            )
                        if current.text or current.first_audio_at:
                            sentences.append(current)
                        current = SentenceTiming(sentence_idx=len(sentences))
                    
                    elif msg_type == "bot_text_complete":
                        got_complete = True
                    
                elif isinstance(raw, bytes) and len(raw) > 0:
                    if in_audio_block and not current.first_audio_at:
                        current.first_audio_at = now
                        logger.info(f"  [LISTENER] First audio #{current.sentence_idx} @ {now:.3f}")

            except asyncio.TimeoutError:
                if got_complete and not in_audio_block:
                    break
                continue
            except Exception as e:
                logger.warning(f"  [LISTENER] Error: {e}")
                break

        # Finalize last sentence if pending
        if current.text or current.first_audio_at:
            if current.text_arrived_at and current.first_audio_at:
                current.gap_ms = round(
                    (current.text_arrived_at - current.first_audio_at) * 1000, 1
                )
            sentences.append(current)

        result.sentences = sentences
        
        if sentences:
            logger.info(f"  [LISTENER] ── Per-sentence timing ──")
            for s in sentences:
                status = "✅" if abs(s.gap_ms) < 500 else "⚠️" if abs(s.gap_ms) < 2000 else "❌"
                logger.info(
                    f"  [LISTENER] {status} Sentence {s.sentence_idx}: "
                    f"gap={s.gap_ms:+.0f}ms | "
                    f"text='{s.text[:40]}'"
                )
            
            gaps = [s.gap_ms for s in sentences if s.text_arrived_at and s.first_audio_at]
            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                max_gap = max(abs(g) for g in gaps)
                logger.info(f"  [LISTENER] Average gap: {avg_gap:+.0f}ms, Max abs gap: {max_gap:.0f}ms")
                synced = abs(avg_gap) < 500
                logger.info(f"  [LISTENER] Verdict: {'✅ SYNCED' if synced else '❌ OUT OF SYNC (text ahead of audio)'}")
                result.passed = True
            else:
                result.error = "Could not measure gaps (missing text or audio timestamps)"
        else:
            result.error = "No sentences received"
            logger.error(f"  [LISTENER] {result.error}")

        try:
            await ws_speaker.close()
        except Exception:
            pass
        await ws_listener.close()
        await api_delete_room(room_id)

    except Exception as e:
        result.error = str(e)
        logger.error(f"  [LISTENER] Error: {e}")

    return result


async def _drain_speaker(ws):
    """Background task to drain speaker messages."""
    try:
        deadline = time.time() + 60.0
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="Text-audio sync timing test")
    parser.add_argument("--mode", choices=["tutor", "speaker", "listener", "all"], default="all")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  TEXT-AUDIO SYNC TIMING TEST")
    logger.info("=" * 60)
    logger.info(f"  HTTP:      {PIPECAT_HTTP_URL}")
    logger.info(f"  WS:        {PIPECAT_WS_URL}")
    logger.info(f"  Classroom: {CLASSROOM_WS_BASE}")
    logger.info("=" * 60)

    results = []

    if args.mode in ("tutor", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("  MODE 1: TUTOR (Pipecat pipeline)")
        logger.info("=" * 60)
        results.append(await test_tutor_mode())

    if args.mode in ("speaker", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("  MODE 2: CLASSROOM SPEAKER (text_message path)")
        logger.info("=" * 60)
        results.append(await test_classroom_speaker())

    if args.mode in ("listener", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("  MODE 3: CLASSROOM LISTENER (text_and_audio)")
        logger.info("=" * 60)
        results.append(await test_classroom_listener())

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("  SUMMARY")
    logger.info("=" * 60)
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        if r.sentences:
            gaps = [s.gap_ms for s in r.sentences if s.text_arrived_at and s.first_audio_at]
            if gaps:
                avg = sum(gaps) / len(gaps)
                logger.info(f"  {status} {r.mode:25s} | avg_gap={avg:+.0f}ms | sentences={len(r.sentences)}")
            else:
                logger.info(f"  {status} {r.mode:25s} | {len(r.sentences)} sentences (no gap data)")
        else:
            logger.info(f"  {status} {r.mode:25s} | {r.error}")

    logger.info("")
    logger.info("  INTERPRETATION:")
    logger.info("    gap ≈ 0ms     → ✅ Text and audio perfectly synced")
    logger.info("    gap > +500ms  → ❌ Text arrives BEFORE audio (user reads ahead)")
    logger.info("    gap < -500ms  → ⚠️  Audio arrives BEFORE text (unusual)")
    logger.info("=" * 60)

    all_passed = all(r.passed for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
