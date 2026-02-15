#!/usr/bin/env python3
"""
Standalone test: verify all LLM sentences are synthesized to audio by Svara TTS.

Connects via WebSocket (like the web app), injects text via /inject_text,
and verifies that:
  1. All sentences appear as bot_text JSON messages
  2. Audio is received for the full response (no gaps/drops)
  3. bot_text_complete is received with the full response
  4. No TTS timeouts occur

Usage:
    # Test against local Svara stack
    python tests/test_tts_sentences.py --url ws://localhost:7861/ws --http http://localhost:7861

    # Test against local ElevenLabs stack
    python tests/test_tts_sentences.py --url ws://localhost:7860/ws --http http://localhost:7860

    # Via Docker Compose
    docker compose run --rm -e PIPECAT_WS_URL=ws://mira-voice-svara:7860/ws test-voice \
        python tests/test_tts_sentences.py
"""

import asyncio
import json
import logging
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")

try:
    import pipecat.frames.protobufs.frames_pb2 as frame_protos
except ImportError:
    sys.exit("pip install pipecat-ai  (need protobuf frames)")

try:
    import jwt as pyjwt
except ImportError:
    pyjwt = None

try:
    import httpx
except ImportError:
    httpx = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tts-test")

SAMPLE_RATE = 16000
WEBUI_SECRET_KEY = os.getenv("WEBUI_SECRET_KEY", "").strip() or None
# Greeting text to ignore when collecting responses
GREETING_TEXT = "Namaste! I'm Mira, your study buddy. I speak English, Hindi, and Tamil. Ask me anything!"


def _make_jwt(user_id: str = "tts-test-user") -> str:
    if not WEBUI_SECRET_KEY or not pyjwt:
        return ""
    payload = {
        "id": user_id,
        "email": f"{user_id}@example.test",
        "exp": int(time.time()) + 7200,
    }
    return pyjwt.encode(payload, WEBUI_SECRET_KEY, algorithm="HS256")


@dataclass
class TestResult:
    prompt: str
    bot_text_chunks: List[str] = field(default_factory=list)
    bot_text_complete: str = ""
    audio_bytes: int = 0
    audio_chunks: int = 0
    audio_duration_sec: float = 0.0
    ttfab_ms: float = -1
    total_ms: float = 0
    tts_timeouts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    passed: bool = False
    # Timing diagnostics: detect bursts and gaps in audio delivery
    audio_arrival_wall_sec: float = 0.0  # wall clock from first to last audio chunk
    max_inter_chunk_gap_ms: float = 0.0  # largest gap between consecutive audio chunks
    pacing_ratio: float = 0.0  # wall_time / audio_duration (1.0 = perfect real-time)
    burst_detected: bool = False  # True if pacing_ratio < 0.5 for >0.5s of audio
    # Content validation
    content_warnings: List[str] = field(default_factory=list)


async def run_test(ws_url: str, http_url: str, prompt: str, timeout: float = 90.0) -> TestResult:
    """Connect, inject text, collect response, disconnect."""
    result = TestResult(prompt=prompt)
    t0 = time.time()

    try:
        # Connect
        ws = await asyncio.wait_for(
            websockets.connect(ws_url, max_size=10 * 1024 * 1024),
            timeout=15.0,
        )
        logger.info(f"  Connected to {ws_url}")

        # Send config
        config = {
            "type": "config",
            "mode": "text_and_audio",
            "language": "en",
            "enable_greeting": True,
        }
        token = _make_jwt()
        if token:
            config["token"] = token
        await ws.send(json.dumps(config))

        # Wait for greeting + session_id
        session_id = None
        greeting_done = False
        greeting_audio_chunks = 0
        bot_speaking = False
        greeting_start = time.time()

        while time.time() - greeting_start < 45.0:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=12.0)
            except asyncio.TimeoutError:
                if session_id:
                    logger.info("  Greeting wait timed out, proceeding anyway")
                    break
                continue

            if isinstance(msg, bytes):
                greeting_audio_chunks += 1
                continue

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type", "")
            if msg_type == "session_id":
                session_id = data.get("session_id", "")
                logger.info(f"  Session: {session_id[:16]}...")
            elif msg_type == "bot_text_complete":
                greeting_done = True
                logger.info(f"  Greeting complete: '{data.get('text', '')[:60]}...'")
                # Drain remaining greeting audio
                drain_start = time.time()
                while time.time() - drain_start < 10.0:
                    try:
                        extra = await asyncio.wait_for(ws.recv(), timeout=3.0)
                        if isinstance(extra, bytes):
                            greeting_audio_chunks += 1
                    except asyncio.TimeoutError:
                        break
                break

        if not session_id:
            result.errors.append("No session_id received")
            await ws.close()
            return result

        logger.info(f"  Greeting done ({greeting_audio_chunks} audio chunks)")

        # Inject text prompt
        inject_start = time.time()
        if httpx:
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{http_url}/inject_text",
                    json={"session_id": session_id, "text": prompt},
                    headers=headers,
                    timeout=10.0,
                )
                if resp.status_code != 200:
                    result.errors.append(f"inject_text failed: {resp.status_code} {resp.text}")
                    await ws.close()
                    return result
                logger.info(f"  Injected text: '{prompt[:60]}...'")
        else:
            result.errors.append("httpx not available for /inject_text")
            await ws.close()
            return result

        # Collect response
        audio_bytes = 0
        audio_chunks = 0
        bot_text_chunks = []
        bot_text_complete = ""
        first_audio_at = None
        last_audio_at = None
        max_gap_ms = 0.0
        last_activity = time.time()
        response_done = False
        consecutive_timeouts = 0

        while time.time() - inject_start < timeout:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                consecutive_timeouts = 0
            except asyncio.TimeoutError:
                consecutive_timeouts += 1
                # For long responses, bot_text_complete arrives after all audio
                # is drained.  Don't break early — only break if:
                #   a) We already have bot_text_complete, OR
                #   b) We've had 3+ consecutive timeouts (30s idle) after audio started
                if bot_text_complete:
                    break
                if audio_chunks > 0 and consecutive_timeouts >= 3:
                    logger.warning(f"  No bot_text_complete after {consecutive_timeouts * 10}s idle — giving up")
                    break
                continue

            last_activity = time.time()

            if isinstance(msg, bytes):
                # Audio frame (protobuf)
                now = time.time()
                audio_chunks += 1
                try:
                    pf = frame_protos.Frame()
                    pf.ParseFromString(msg)
                    if pf.HasField("audio"):
                        audio_bytes += len(pf.audio.audio)
                        if first_audio_at is None:
                            first_audio_at = now
                except Exception:
                    audio_bytes += len(msg)
                    if first_audio_at is None:
                        first_audio_at = now
                # Track inter-chunk gaps
                if last_audio_at is not None:
                    gap = (now - last_audio_at) * 1000
                    if gap > max_gap_ms:
                        max_gap_ms = gap
                last_audio_at = now
                continue

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type", "")
            if msg_type == "bot_text":
                text = data.get("text", "")
                bot_text_chunks.append(text)
            elif msg_type == "bot_text_complete":
                complete_text = data.get("text", "")
                # Skip the greeting's bot_text_complete — it can arrive late
                if complete_text.strip() == GREETING_TEXT.strip():
                    logger.info(f"  (Ignoring late greeting bot_text_complete)")
                    continue
                bot_text_complete = complete_text
                logger.info(f"  Response complete: '{bot_text_complete[:80]}...'")
                response_done = True
                # Drain remaining audio — Hindi multi-chunk TTS can take 15-20s
                drain_start = time.time()
                while time.time() - drain_start < 20.0:
                    try:
                        extra = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        if isinstance(extra, bytes):
                            now = time.time()
                            audio_chunks += 1
                            try:
                                pf = frame_protos.Frame()
                                pf.ParseFromString(extra)
                                if pf.HasField("audio"):
                                    audio_bytes += len(pf.audio.audio)
                            except Exception:
                                audio_bytes += len(extra)
                            if last_audio_at is not None:
                                gap = (now - last_audio_at) * 1000
                                if gap > max_gap_ms:
                                    max_gap_ms = gap
                            last_audio_at = now
                    except asyncio.TimeoutError:
                        break
                break

        await ws.close()

        # Calculate results
        result.bot_text_chunks = bot_text_chunks
        # If bot_text_complete was never received (long responses where the
        # message arrives after all audio is drained), reconstruct from chunks
        if not bot_text_complete and bot_text_chunks:
            bot_text_complete = " ".join(bot_text_chunks)
            logger.info(f"  Reconstructed text from {len(bot_text_chunks)} chunks: '{bot_text_complete[:80]}...'")
        result.bot_text_complete = bot_text_complete
        result.audio_bytes = audio_bytes
        result.audio_chunks = audio_chunks
        result.audio_duration_sec = audio_bytes / (SAMPLE_RATE * 2) if audio_bytes > 0 else 0
        result.ttfab_ms = (first_audio_at - inject_start) * 1000 if first_audio_at else -1
        result.total_ms = (time.time() - t0) * 1000

        # Timing diagnostics
        if first_audio_at and last_audio_at and last_audio_at > first_audio_at:
            result.audio_arrival_wall_sec = last_audio_at - first_audio_at
        result.max_inter_chunk_gap_ms = max_gap_ms
        if result.audio_duration_sec > 0 and result.audio_arrival_wall_sec > 0:
            result.pacing_ratio = result.audio_arrival_wall_sec / result.audio_duration_sec
        result.burst_detected = (
            result.audio_duration_sec > 0.5
            and result.pacing_ratio > 0
            and result.pacing_ratio < 0.5
        )

        # ── Content validation ──
        # Check that audio duration is proportional to text length.
        # Real speech ≈ 150 words/min ≈ 0.4s/word.  We use a very
        # conservative floor: 0.08s per word (5× faster than real speech).
        # If audio is shorter than this, sentences were likely skipped.
        word_count = len(bot_text_complete.split()) if bot_text_complete else 0
        min_audio_sec = word_count * 0.08  # conservative: ~750 wpm floor
        sentence_count = len(bot_text_chunks)
        audio_per_sentence = result.audio_duration_sec / sentence_count if sentence_count > 0 else 0

        content_ok = True
        content_warnings = []

        if word_count > 5 and result.audio_duration_sec < min_audio_sec:
            content_warnings.append(
                f"Audio too short: {result.audio_duration_sec:.1f}s for {word_count} words "
                f"(min expected {min_audio_sec:.1f}s)"
            )
            content_ok = False

        if sentence_count > 1 and audio_per_sentence < 0.5:
            content_warnings.append(
                f"Audio per sentence too low: {audio_per_sentence:.2f}s/sentence "
                f"({sentence_count} sentences, {result.audio_duration_sec:.1f}s total)"
            )
            content_ok = False

        # Expect at least ~1s of audio per bot_text sentence chunk
        if sentence_count >= 3 and result.audio_duration_sec < sentence_count * 0.8:
            content_warnings.append(
                f"Likely missing sentences in audio: {result.audio_duration_sec:.1f}s for "
                f"{sentence_count} sentences (expected ≥{sentence_count * 0.8:.1f}s)"
            )
            content_ok = False

        result.content_warnings = content_warnings

        # Check for TTS timeouts in the text chunks
        result.passed = (
            audio_chunks > 0
            and bot_text_complete != ""
            and len(result.tts_timeouts) == 0
            and len(result.errors) == 0
            and not result.burst_detected  # Burst = client likely dropped audio
            and content_ok  # Audio duration matches text content
        )

    except Exception as e:
        result.errors.append(f"Exception: {e}")
        import traceback
        traceback.print_exc()

    return result


async def run_multi_turn_test(ws_url: str, http_url: str, turns: List[str], timeout: float = 90.0) -> List[TestResult]:
    """Run multiple turns in a single session, collecting results per turn."""
    results = []
    t0 = time.time()

    try:
        ws = await asyncio.wait_for(
            websockets.connect(ws_url, max_size=10 * 1024 * 1024),
            timeout=15.0,
        )

        config = {
            "type": "config",
            "mode": "text_and_audio",
            "language": "en",
            "enable_greeting": True,
        }
        token = _make_jwt()
        if token:
            config["token"] = token
        await ws.send(json.dumps(config))

        # Wait for session_id and greeting
        session_id = None
        greeting_start = time.time()
        while time.time() - greeting_start < 45.0:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=12.0)
            except asyncio.TimeoutError:
                if session_id:
                    break
                continue

            if isinstance(msg, bytes):
                continue

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue

            if data.get("type") == "session_id":
                session_id = data.get("session_id", "")
                logger.info(f"  MT Session: {session_id[:16]}...")
            elif data.get("type") == "bot_text_complete":
                # Drain greeting audio
                drain_start = time.time()
                while time.time() - drain_start < 10.0:
                    try:
                        extra = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    except asyncio.TimeoutError:
                        break
                break

        if not session_id:
            return [TestResult(prompt=t, errors=["No session_id"]) for t in turns]

        # Run each turn
        for turn_idx, prompt in enumerate(turns):
            result = TestResult(prompt=prompt)
            turn_start = time.time()

            if not httpx:
                result.errors.append("httpx not available")
                results.append(result)
                continue

            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{http_url}/inject_text",
                    json={"session_id": session_id, "text": prompt},
                    headers=headers,
                    timeout=10.0,
                )
                if resp.status_code != 200:
                    result.errors.append(f"inject_text failed: {resp.status_code}")
                    results.append(result)
                    continue

            logger.info(f"  MT Turn {turn_idx+1}: injected '{prompt[:50]}...'")

            # Collect response for this turn
            audio_bytes = 0
            audio_chunks = 0
            bot_text_chunks = []
            bot_text_complete = ""
            first_audio_at = None
            mt_consecutive_timeouts = 0

            while time.time() - turn_start < timeout:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    mt_consecutive_timeouts = 0
                except asyncio.TimeoutError:
                    mt_consecutive_timeouts += 1
                    if bot_text_complete:
                        break
                    if audio_chunks > 0 and mt_consecutive_timeouts >= 3:
                        logger.warning(f"  MT Turn {turn_idx+1}: no bot_text_complete after {mt_consecutive_timeouts * 10}s idle")
                        break
                    continue

                if isinstance(msg, bytes):
                    audio_chunks += 1
                    try:
                        pf = frame_protos.Frame()
                        pf.ParseFromString(msg)
                        if pf.HasField("audio"):
                            audio_bytes += len(pf.audio.audio)
                            if first_audio_at is None:
                                first_audio_at = time.time()
                    except Exception:
                        audio_bytes += len(msg)
                        if first_audio_at is None:
                            first_audio_at = time.time()
                    continue

                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")
                if msg_type == "bot_text":
                    bot_text_chunks.append(data.get("text", ""))
                elif msg_type == "bot_text_complete":
                    complete_text = data.get("text", "")
                    if complete_text.strip() == GREETING_TEXT.strip():
                        continue
                    bot_text_complete = complete_text
                    # Drain remaining audio
                    drain_start = time.time()
                    while time.time() - drain_start < 15.0:
                        try:
                            extra = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            if isinstance(extra, bytes):
                                audio_chunks += 1
                                try:
                                    pf = frame_protos.Frame()
                                    pf.ParseFromString(extra)
                                    if pf.HasField("audio"):
                                        audio_bytes += len(pf.audio.audio)
                                except Exception:
                                    audio_bytes += len(extra)
                        except asyncio.TimeoutError:
                            break
                    break

            result.bot_text_chunks = bot_text_chunks
            # Reconstruct from chunks if bot_text_complete was never received
            if not bot_text_complete and bot_text_chunks:
                bot_text_complete = " ".join(bot_text_chunks)
                logger.info(f"  MT Turn {turn_idx+1}: reconstructed text from {len(bot_text_chunks)} chunks")
            result.bot_text_complete = bot_text_complete
            result.audio_bytes = audio_bytes
            result.audio_chunks = audio_chunks
            result.audio_duration_sec = audio_bytes / (SAMPLE_RATE * 2) if audio_bytes > 0 else 0
            result.ttfab_ms = (first_audio_at - turn_start) * 1000 if first_audio_at else -1
            result.total_ms = (time.time() - turn_start) * 1000

            # Content validation (same as single-turn)
            wc = len(bot_text_complete.split()) if bot_text_complete else 0
            min_audio = wc * 0.08
            sc = len(bot_text_chunks)
            aps = result.audio_duration_sec / sc if sc > 0 else 0
            content_ok = True
            cw = []
            if wc > 5 and result.audio_duration_sec < min_audio:
                cw.append(f"Audio too short: {result.audio_duration_sec:.1f}s for {wc} words (min {min_audio:.1f}s)")
                content_ok = False
            if sc > 1 and aps < 0.5:
                cw.append(f"Audio/sentence too low: {aps:.2f}s ({sc} sentences)")
                content_ok = False
            if sc >= 3 and result.audio_duration_sec < sc * 0.8:
                cw.append(f"Likely missing sentences: {result.audio_duration_sec:.1f}s for {sc} sentences")
                content_ok = False
            result.content_warnings = cw

            result.passed = (
                audio_chunks > 0
                and bot_text_complete != ""
                and len(result.errors) == 0
                and content_ok
            )
            results.append(result)

        await ws.close()

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Fill remaining results
        while len(results) < len(turns):
            results.append(TestResult(prompt=turns[len(results)], errors=[f"Exception: {e}"]))

    return results


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test TTS sentence completeness")
    parser.add_argument("--url", default=os.getenv("PIPECAT_WS_URL", "ws://localhost:7860/ws"),
                        help="WebSocket URL")
    parser.add_argument("--http", default=os.getenv("PIPECAT_HTTP_URL", ""),
                        help="HTTP URL for /inject_text")
    parser.add_argument("--multi-turn", action="store_true", default=False,
                        help="Also run multi-turn conversation tests")
    args = parser.parse_args()

    ws_url = args.url
    http_url = args.http
    if not http_url:
        # Derive from WS URL
        http_url = ws_url.replace("ws://", "http://").replace("wss://", "https://").replace("/ws", "")

    test_prompts = [
        # 1. Short (1-2 sentences expected)
        "Hello, how are you?",
        # 2. Medium (2-3 sentences expected)
        "What is photosynthesis? Explain briefly.",
        # 3. Long English (4-6 sentences expected — the bug trigger)
        "Explain photosynthesis in detail. How do plants use sunlight, water, and carbon dioxide to make food? Give me a complete explanation with examples.",
        # 4. Hindi (3+ sentences expected — NLTK sentence merge bug trigger)
        "Photosynthesis क्या है? मुझे हिंदी में विस्तार से बताओ। पौधे सूरज की रोशनी का उपयोग कैसे करते हैं?",
        # 5. Very long English (force 5+ sentences)
        "I want a detailed explanation of the water cycle. Start from evaporation, then condensation, precipitation, and collection. How does each stage work? Why is it important for life on Earth? Give me real world examples.",
        # 6. Multi-turn stress: history topic (force long response)
        "Tell me about the French Revolution. What caused it? Who were the key people involved? What happened during the Reign of Terror? How did it end and what was its impact on the world?",
    ]

    # Multi-turn conversation sequences (each turn is a prompt in order)
    multi_turn_conversations = [
        {
            "name": "Multi-turn Science (EN)",
            "turns": [
                "What is gravity? Explain it to me like I'm in 6th grade.",
                "That's cool! So why do things fall down and not sideways?",
                "What would happen if there was no gravity on Earth? Give me some fun examples.",
            ],
        },
        {
            "name": "Multi-turn Science (HI)",
            "turns": [
                "गुरुत्वाकर्षण क्या है? मुझे सरल भाषा में समझाओ।",
                "अच्छा, तो चंद्रमा पर गुरुत्वाकर्षण कम क्यों होता है?",
            ],
        },
    ]

    print("\n" + "=" * 70)
    print("  TTS Sentence Completeness Test")
    print(f"  Target: {ws_url}")
    print("=" * 70 + "\n")

    results = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'─' * 60}")
        print(f"  Test {i}/{len(test_prompts)}: \"{prompt[:50]}...\"")
        print(f"{'─' * 60}")

        result = await run_test(ws_url, http_url, prompt)
        results.append(result)

        status = "✅ PASS" if result.passed else "❌ FAIL"
        word_count = len(result.bot_text_complete.split()) if result.bot_text_complete else 0
        print(f"  {status}")
        print(f"  Response: '{result.bot_text_complete[:100]}{'...' if len(result.bot_text_complete) > 100 else ''}'")
        print(f"  Text: {len(result.bot_text_chunks)} sentences, {word_count} words | "
              f"Audio: {result.audio_duration_sec:.1f}s ({result.audio_chunks} chunks)")
        if len(result.bot_text_chunks) > 0:
            print(f"  Audio/sentence: {result.audio_duration_sec / len(result.bot_text_chunks):.1f}s | "
                  f"Audio/word: {result.audio_duration_sec / max(word_count, 1):.2f}s "
                  f"(expect ~0.3-0.5s/word)")
        print(f"  TTFAB: {result.ttfab_ms:.0f}ms | Total: {result.total_ms:.0f}ms")
        print(f"  Pacing: {result.pacing_ratio:.2f}x real-time | "
              f"Max gap: {result.max_inter_chunk_gap_ms:.0f}ms | "
              f"Wall: {result.audio_arrival_wall_sec:.1f}s")
        if result.burst_detected:
            print(f"  ⚠️  BURST DETECTED: audio arrived {result.pacing_ratio:.2f}x real-time "
                  f"(expected ~1.0x) — client likely dropped audio!")
        if result.content_warnings:
            for w in result.content_warnings:
                print(f"  ⚠️  CONTENT: {w}")
        if result.tts_timeouts:
            print(f"  ⚠️  TTS timeouts: {result.tts_timeouts}")
        if result.errors:
            print(f"  ❌ Errors: {result.errors}")

    # Multi-turn tests (optional)
    if args.multi_turn:
        print(f"\n{'=' * 70}")
        print("  MULTI-TURN CONVERSATION TESTS")
        print(f"{'=' * 70}")

        for conv in multi_turn_conversations:
            print(f"\n  📚 {conv['name']} ({len(conv['turns'])} turns)")
            mt_results = await run_multi_turn_test(ws_url, http_url, conv["turns"])
            for j, r in enumerate(mt_results, 1):
                status = "✅" if r.passed else "❌"
                wc = len(r.bot_text_complete.split()) if r.bot_text_complete else 0
                sc = len(r.bot_text_chunks)
                print(f"    Turn {j}: {status} | {sc} sentences, {wc} words | "
                      f"Audio={r.audio_duration_sec:.1f}s ({r.audio_duration_sec / max(wc, 1):.2f}s/word) | "
                      f"Text='{r.bot_text_complete[:60]}...'")
                if r.content_warnings:
                    for w in r.content_warnings:
                        print(f"      ⚠️  {w}")
                if r.errors:
                    print(f"      ❌ {r.errors}")
                results.append(r)

    # Summary
    passed = sum(1 for r in results if r.passed)
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {passed}/{len(results)} passed")
    print(f"{'=' * 70}\n")

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
