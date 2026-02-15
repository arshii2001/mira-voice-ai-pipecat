#!/usr/bin/env python3
"""
STT Verification Test: TTS audio → Soniox STT → compare with LLM text.

This test validates TTS audio completeness by:
1. Connecting to the Pipecat backend via WebSocket
2. Injecting a text prompt (bypassing STT)
3. Collecting the TTS audio output AND the LLM text (bot_text_complete)
4. Sending the collected audio to Soniox STT for transcription
5. Comparing the STT transcript against the LLM text
6. Reporting word-level accuracy, missing segments, and timing gaps

This catches:
- Missing audio chunks (sentences that were never synthesized)
- Audio truncation (sentences cut off mid-word)
- Audio quality issues (garbled audio → STT produces garbage)
- Silence gaps (visible in word timestamp gaps)

Usage:
    # Test against local Svara stack
    python tests/test_stt_verify.py --url ws://localhost:7861/ws

    # Test against local ElevenLabs stack
    python tests/test_stt_verify.py --url ws://localhost:7860/ws

    # Run just 1 test
    python tests/test_stt_verify.py --limit 1

    # Via Docker Compose
    docker compose run --rm \
        -e PIPECAT_WS_URL=ws://mira-voice-svara:7860/ws \
        -e SONIOX_API_KEY=... \
        test-voice python tests/test_stt_verify.py
"""

import asyncio
import io
import json
import logging
import os
import struct
import sys
import time
import wave
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

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

try:
    import numpy as np
except ImportError:
    np = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stt-verify")

# --- Config ---
TTS_SAMPLE_RATE = 24000  # Svara TTS outputs 24kHz
SONIOX_SAMPLE_RATE = 16000  # Soniox expects 16kHz
SONIOX_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
SONIOX_API_KEY = os.getenv("SONIOX_API_KEY", "").strip()
WEBUI_SECRET_KEY = os.getenv("WEBUI_SECRET_KEY", "").strip() or None
GREETING_TEXT = "Namaste! I'm Mira, your study buddy. I speak English, Hindi, and Tamil. Ask me anything!"

OUTPUT_DIR = os.getenv("STT_VERIFY_OUTPUT_DIR", "tests/stt_verify_output")


# --- Helpers ---

def _make_jwt(user_id: str = "stt-verify-user") -> str:
    if not WEBUI_SECRET_KEY or not pyjwt:
        return ""
    payload = {
        "id": user_id,
        "email": f"{user_id}@example.test",
        "exp": int(time.time()) + 7200,
    }
    return pyjwt.encode(payload, WEBUI_SECRET_KEY, algorithm="HS256")


def resample_pcm16(audio_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample PCM16 mono audio from one sample rate to another."""
    if from_rate == to_rate:
        return audio_bytes
    if np is None:
        # Fallback: simple linear interpolation without numpy
        samples_in = struct.unpack(f"<{len(audio_bytes)//2}h", audio_bytes)
        ratio = to_rate / from_rate
        out_len = int(len(samples_in) * ratio)
        samples_out = []
        for i in range(out_len):
            src_idx = i / ratio
            idx0 = int(src_idx)
            idx1 = min(idx0 + 1, len(samples_in) - 1)
            frac = src_idx - idx0
            val = int(samples_in[idx0] * (1 - frac) + samples_in[idx1] * frac)
            samples_out.append(max(-32768, min(32767, val)))
        return struct.pack(f"<{len(samples_out)}h", *samples_out)

    # Numpy path (much faster)
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    ratio = to_rate / from_rate
    out_len = int(len(samples) * ratio)
    indices = np.arange(out_len) / ratio
    idx0 = np.floor(indices).astype(int)
    idx1 = np.minimum(idx0 + 1, len(samples) - 1)
    frac = indices - idx0
    resampled = (samples[idx0] * (1 - frac) + samples[idx1] * frac).astype(np.int16)
    return resampled.tobytes()


def save_wav(audio_bytes: bytes, sample_rate: int, filepath: str):
    """Save raw PCM16 mono audio as WAV file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    # Remove common punctuation
    text = re.sub(r"[.,!?;:\"'()\[\]{}\-—–…]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_overlap_score(reference: str, hypothesis: str) -> Tuple[float, List[str], List[str]]:
    """
    Compute word-level overlap between reference (LLM text) and hypothesis (STT text).
    Returns (score, missing_words, extra_words).
    Score is 0.0-1.0 where 1.0 = perfect match.
    """
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    if not ref_words:
        return 1.0 if not hyp_words else 0.0, [], hyp_words

    # Use SequenceMatcher for fuzzy word-level alignment
    matcher = SequenceMatcher(None, ref_words, hyp_words)
    matching_blocks = matcher.get_matching_blocks()
    matched_count = sum(block.size for block in matching_blocks)

    score = matched_count / len(ref_words) if ref_words else 1.0

    # Find missing words (in reference but not in hypothesis)
    hyp_set = set(hyp_words)
    ref_set = set(ref_words)
    missing = [w for w in ref_words if w not in hyp_set]
    extra = [w for w in hyp_words if w not in ref_set]

    return score, missing[:20], extra[:20]  # Cap lists for readability


def detect_timing_gaps(tokens: List[dict], gap_threshold_ms: float = 1500) -> List[dict]:
    """
    Detect unusual gaps in word timing from Soniox tokens.
    Returns list of gaps > threshold with context.
    """
    gaps = []
    for i in range(1, len(tokens)):
        prev = tokens[i - 1]
        curr = tokens[i]
        prev_end = prev.get("start_ms", 0) + prev.get("duration_ms", 0)
        curr_start = curr.get("start_ms", 0)
        gap_ms = curr_start - prev_end
        if gap_ms > gap_threshold_ms:
            gaps.append({
                "after_word": prev.get("text", "?"),
                "before_word": curr.get("text", "?"),
                "gap_ms": gap_ms,
                "position_ms": prev_end,
            })
    return gaps


@dataclass
class STTVerifyResult:
    """Result of a single STT verification test."""
    prompt: str
    llm_text: str = ""
    stt_text: str = ""
    word_score: float = 0.0
    missing_words: List[str] = field(default_factory=list)
    extra_words: List[str] = field(default_factory=list)
    timing_gaps: List[dict] = field(default_factory=list)
    audio_duration_sec: float = 0.0
    audio_bytes: int = 0
    stt_word_count: int = 0
    stt_tokens: List[dict] = field(default_factory=list)
    ttfab_ms: float = -1
    total_ms: float = 0
    errors: List[str] = field(default_factory=list)
    passed: bool = False
    wav_path: str = ""


# --- Soniox STT Transcription ---

async def transcribe_with_soniox(audio_pcm16: bytes, sample_rate: int = 16000,
                                  language_hints: List[str] = None) -> Tuple[str, List[dict]]:
    """
    Send audio to Soniox realtime STT and get back transcription + token details.

    Returns (full_text, tokens) where tokens have {text, start_ms, duration_ms, is_final, language}.
    """
    if not SONIOX_API_KEY:
        raise ValueError("SONIOX_API_KEY not set")

    if language_hints is None:
        language_hints = ["en", "hi"]

    all_tokens = []
    final_text_parts = []
    # Track the longest interim transcript seen — Soniox may not finalize
    # the last utterance if endpoint detection triggers early
    best_interim_text = ""

    try:
        ws = await websockets.connect(
            SONIOX_WS_URL,
            max_size=10 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
        )
    except Exception as e:
        logger.error(f"  Soniox connect failed: {e}")
        return "", []

    try:
        # Send config
        config = {
            "api_key": SONIOX_API_KEY,
            "model": "stt-rt-v3",
            "language_hints": language_hints,
            "language_hints_strict": True,
            "enable_language_identification": True,
            "enable_endpoint_detection": False,  # Disable so all audio is one utterance
            "audio_format": "pcm_s16le",
            "sample_rate": sample_rate,
            "num_channels": 1,
        }
        await ws.send(json.dumps(config))
        logger.info(f"  Soniox config sent (model=stt-rt-v3, rate={sample_rate})")

        # Wait for ack
        try:
            ack = await asyncio.wait_for(ws.recv(), timeout=5.0)
            ack_str = ack if isinstance(ack, str) else ack.decode("utf-8", errors="replace")
            logger.info(f"  Soniox ack: {ack_str[:200]}")
            # Check if ack contains an error
            try:
                ack_data = json.loads(ack_str)
                if ack_data.get("error_code") == 429 or "error" in ack_data:
                    error_msg = ack_data.get("error_message", ack_data.get("error", "unknown"))
                    logger.warning(f"  Soniox rate limited: {error_msg}")
                    await ws.close()
                    raise ConnectionError(f"Soniox 429: {error_msg}")
            except json.JSONDecodeError:
                pass
        except asyncio.TimeoutError:
            logger.warning("  Soniox: no ack received, proceeding anyway")

        # Append 1 second of silence to the end of the audio to give Soniox
        # time to finalize the last utterance before we send end-of-audio
        silence_pad = b"\x00" * (sample_rate * 2 * 1)  # 1 second of silence
        padded_audio = audio_pcm16 + silence_pad

        # Stream audio in smaller chunks (320 bytes = 10ms at 16kHz mono 16-bit)
        # Soniox expects near-real-time streaming
        chunk_duration_ms = 100  # 100ms chunks
        chunk_size = int(sample_rate * 2 * chunk_duration_ms / 1000)  # bytes per chunk
        offset = 0
        chunks_sent = 0
        while offset < len(padded_audio):
            chunk = padded_audio[offset:offset + chunk_size]
            try:
                await ws.send(chunk)
                chunks_sent += 1
            except websockets.exceptions.ConnectionClosed as e:
                logger.error(f"  Soniox WS closed while sending audio at chunk {chunks_sent}: {e}")
                break
            offset += chunk_size
            # Pace at ~1.2x real-time — too fast and Soniox may drop the tail
            await asyncio.sleep(chunk_duration_ms / 1000 / 1.2)

        logger.info(f"  Sent {chunks_sent} audio chunks ({len(padded_audio)} bytes, incl 1s silence pad)")

        # Signal end of audio — send empty string then wait a moment for
        # Soniox to finalize the last utterance
        try:
            await ws.send("")
            logger.info("  Sent end-of-audio signal")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("  Soniox WS already closed when sending end signal")

        # Collect responses until we get final results
        collect_start = time.time()
        got_end = False
        while time.time() - collect_start < 30.0:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=8.0)
            except asyncio.TimeoutError:
                logger.info("  Soniox: recv timeout, done collecting")
                break
            except websockets.exceptions.ConnectionClosed:
                logger.info("  Soniox: connection closed, done collecting")
                break

            if isinstance(msg, bytes):
                continue

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue

            if "error" in data:
                logger.error(f"  Soniox error: {data['error']}")
                break

            tokens = data.get("tokens", [])
            # Build current message's full text to track best interim
            msg_final_parts = []
            msg_nonfinal_parts = []
            for token in tokens:
                token_text = token.get("text", "")
                if token_text == "<end>":
                    got_end = True
                    continue
                if not token_text:
                    continue

                all_tokens.append(token)

                if token.get("is_final", False):
                    final_text_parts.append(token_text)
                    msg_final_parts.append(token_text)
                else:
                    msg_nonfinal_parts.append(token_text)

            # The best interim is: all finalized text so far + current non-final tail
            current_full = "".join(final_text_parts) + "".join(msg_nonfinal_parts)
            if len(current_full) > len(best_interim_text):
                best_interim_text = current_full

            if got_end:
                logger.info("  Soniox: received <end> token")
                # Keep collecting for a moment — there may be final tokens
                # that arrive in the same batch as <end> or just after
                try:
                    extra_msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    if isinstance(extra_msg, str):
                        try:
                            extra_data = json.loads(extra_msg)
                            for token in extra_data.get("tokens", []):
                                token_text = token.get("text", "")
                                if token_text and token_text != "<end>":
                                    all_tokens.append(token)
                                    if token.get("is_final", False):
                                        final_text_parts.append(token_text)
                        except json.JSONDecodeError:
                            pass
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    pass
                break

    finally:
        try:
            await ws.close()
        except Exception:
            pass

    final_only = "".join(final_text_parts).strip()
    best_interim = best_interim_text.strip()

    # Use the longer of final-only vs best-interim transcript
    if len(best_interim) > len(final_only) * 1.05:
        logger.info(f"  Soniox: using best interim ({len(best_interim)} chars) over final-only ({len(final_only)} chars)")
        full_text = best_interim
    else:
        full_text = final_only

    logger.info(f"  Soniox transcript ({len(all_tokens)} tokens, {len(final_text_parts)} final): '{full_text[:100]}...'")
    return full_text, all_tokens


# --- Main Test Logic ---

async def collect_tts_audio(ws_url: str, http_url: str, prompt: str,
                             timeout: float = 60.0) -> Tuple[bytes, str, float]:
    """
    Connect to Pipecat, inject text, collect TTS audio and LLM text.
    Returns (audio_pcm16_bytes, bot_text_complete, ttfab_ms).
    """
    token = _make_jwt()
    audio_chunks = []
    bot_text_complete = ""
    first_audio_at = None
    inject_start = None

    ws = await asyncio.wait_for(
        websockets.connect(ws_url, max_size=10 * 1024 * 1024),
        timeout=15.0,
    )

    try:
        # Send config
        config = {
            "type": "config",
            "mode": "text_and_audio",
            "language": "en",
            "enable_greeting": True,
        }
        if token:
            config["token"] = token
        await ws.send(json.dumps(config))

        # Wait for greeting + session_id
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
                logger.info(f"  Session: {session_id[:16]}...")
            elif data.get("type") == "bot_text_complete":
                # Drain greeting audio
                drain_start = time.time()
                while time.time() - drain_start < 10.0:
                    try:
                        await asyncio.wait_for(ws.recv(), timeout=3.0)
                    except asyncio.TimeoutError:
                        break
                break

        if not session_id:
            raise RuntimeError("No session_id received")

        # Inject text
        inject_start = time.time()
        if not httpx:
            raise RuntimeError("httpx not available")

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
                raise RuntimeError(f"inject_text failed: {resp.status_code} {resp.text}")

        logger.info(f"  Injected: '{prompt[:60]}...'")

        # Collect response
        while time.time() - inject_start < timeout:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
            except asyncio.TimeoutError:
                if audio_chunks or bot_text_complete:
                    break
                continue

            if isinstance(msg, bytes):
                try:
                    pf = frame_protos.Frame()
                    pf.ParseFromString(msg)
                    if pf.HasField("audio"):
                        audio_chunks.append(pf.audio.audio)
                        if first_audio_at is None:
                            first_audio_at = time.time()
                except Exception:
                    audio_chunks.append(msg)
                    if first_audio_at is None:
                        first_audio_at = time.time()
                continue

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue

            if data.get("type") == "bot_text_complete":
                complete_text = data.get("text", "")
                if complete_text.strip() == GREETING_TEXT.strip():
                    continue
                bot_text_complete = complete_text
                # Drain remaining audio
                drain_start = time.time()
                while time.time() - drain_start < 20.0:
                    try:
                        extra = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        if isinstance(extra, bytes):
                            try:
                                pf = frame_protos.Frame()
                                pf.ParseFromString(extra)
                                if pf.HasField("audio"):
                                    audio_chunks.append(pf.audio.audio)
                            except Exception:
                                audio_chunks.append(extra)
                    except asyncio.TimeoutError:
                        break
                break

    finally:
        await ws.close()

    audio_bytes = b"".join(audio_chunks)
    ttfab_ms = (first_audio_at - inject_start) * 1000 if first_audio_at and inject_start else -1
    return audio_bytes, bot_text_complete, ttfab_ms


async def run_stt_verify_test(ws_url: str, http_url: str, prompt: str,
                               test_id: str, language_hints: List[str] = None) -> STTVerifyResult:
    """Run a single STT verification test."""
    result = STTVerifyResult(prompt=prompt)
    t0 = time.time()

    try:
        # Step 1: Collect TTS audio from Pipecat
        logger.info(f"  [1/3] Collecting TTS audio...")
        audio_pcm16, bot_text, ttfab_ms = await collect_tts_audio(ws_url, http_url, prompt)

        result.llm_text = bot_text
        result.audio_bytes = len(audio_pcm16)
        result.audio_duration_sec = len(audio_pcm16) / (TTS_SAMPLE_RATE * 2) if audio_pcm16 else 0
        result.ttfab_ms = ttfab_ms

        if not audio_pcm16:
            result.errors.append("No audio received from TTS")
            result.total_ms = (time.time() - t0) * 1000
            return result

        if not bot_text:
            result.errors.append("No bot_text_complete received")
            result.total_ms = (time.time() - t0) * 1000
            return result

        # Wait a moment for the backend's Soniox connection to release
        # (Soniox has a max concurrent connections limit)
        logger.info("  Waiting 3s for Soniox rate limit to clear...")
        await asyncio.sleep(3.0)

        # Save WAV file
        wav_path = os.path.join(OUTPUT_DIR, f"{test_id}_tts.wav")
        save_wav(audio_pcm16, TTS_SAMPLE_RATE, wav_path)
        result.wav_path = wav_path
        logger.info(f"  Saved TTS audio: {wav_path} ({result.audio_duration_sec:.1f}s)")

        # Step 2: Resample to 16kHz for Soniox
        logger.info(f"  [2/3] Transcribing with Soniox STT...")
        audio_16k = resample_pcm16(audio_pcm16, TTS_SAMPLE_RATE, SONIOX_SAMPLE_RATE)

        # Also save the 16kHz version for debugging
        wav_16k_path = os.path.join(OUTPUT_DIR, f"{test_id}_16k.wav")
        save_wav(audio_16k, SONIOX_SAMPLE_RATE, wav_16k_path)

        # Transcribe with retry for 429 rate limiting
        stt_text, stt_tokens = "", []
        for attempt in range(3):
            try:
                stt_text, stt_tokens = await transcribe_with_soniox(
                    audio_16k, SONIOX_SAMPLE_RATE, language_hints or ["en", "hi"]
                )
                break  # Success
            except ConnectionError as e:
                if attempt < 2:
                    wait = 5 * (attempt + 1)
                    logger.warning(f"  Retry {attempt+1}/3: waiting {wait}s ({e})")
                    await asyncio.sleep(wait)
                else:
                    result.errors.append(f"Soniox rate limited after 3 retries")
                    result.total_ms = (time.time() - t0) * 1000
                    return result

        result.stt_text = stt_text
        result.stt_tokens = stt_tokens
        result.stt_word_count = len(stt_text.split()) if stt_text else 0

        # Step 3: Compare
        logger.info(f"  [3/3] Comparing LLM text vs STT transcript...")
        score, missing, extra = word_overlap_score(bot_text, stt_text)
        result.word_score = score
        result.missing_words = missing
        result.extra_words = extra

        # Detect timing gaps
        final_tokens = [t for t in stt_tokens if t.get("is_final", False)]
        result.timing_gaps = detect_timing_gaps(final_tokens, gap_threshold_ms=1500)

        # Pass criteria: >70% word overlap and no critical errors
        result.passed = score >= 0.70 and len(result.errors) == 0

    except Exception as e:
        result.errors.append(f"Exception: {e}")
        import traceback
        traceback.print_exc()

    result.total_ms = (time.time() - t0) * 1000
    return result


async def collect_only(ws_url: str, http_url: str, test_cases: list):
    """Phase 1: Collect TTS audio and save WAV files + metadata JSON."""
    print("\n" + "=" * 80)
    print("  PHASE 1: Collect TTS Audio")
    print(f"  Target: {ws_url}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 80 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata = []

    for i, (test_id, prompt, lang_hints) in enumerate(test_cases, 1):
        print(f"\n  [{i}/{len(test_cases)}] {test_id}: '{prompt[:50]}...'")

        try:
            audio_pcm16, bot_text, ttfab_ms = await collect_tts_audio(ws_url, http_url, prompt)
            duration = len(audio_pcm16) / (TTS_SAMPLE_RATE * 2) if audio_pcm16 else 0

            wav_path = os.path.join(OUTPUT_DIR, f"{test_id}_tts.wav")
            if audio_pcm16:
                save_wav(audio_pcm16, TTS_SAMPLE_RATE, wav_path)

            entry = {
                "test_id": test_id,
                "prompt": prompt,
                "language_hints": lang_hints,
                "llm_text": bot_text,
                "audio_bytes": len(audio_pcm16),
                "audio_duration_sec": round(duration, 2),
                "ttfab_ms": round(ttfab_ms, 1),
                "wav_path": wav_path,
            }
            metadata.append(entry)
            print(f"    ✅ Audio: {duration:.1f}s | LLM: '{bot_text[:60]}...'")
            print(f"    WAV: {wav_path}")

        except Exception as e:
            print(f"    ❌ Error: {e}")
            metadata.append({
                "test_id": test_id, "prompt": prompt, "language_hints": lang_hints,
                "llm_text": "", "audio_bytes": 0, "error": str(e),
            })

    # Save metadata
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\n  Metadata saved: {meta_path}")
    print(f"  {len([m for m in metadata if m.get('audio_bytes', 0) > 0])}/{len(metadata)} tests collected audio")


async def verify_only(test_cases: list):
    """Phase 2: Transcribe saved WAV files with Soniox and compare."""
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        sys.exit(f"ERROR: {meta_path} not found. Run --collect first.")

    with open(meta_path) as f:
        metadata = json.load(f)

    if not SONIOX_API_KEY:
        sys.exit("ERROR: SONIOX_API_KEY environment variable not set")

    print("\n" + "=" * 80)
    print("  PHASE 2: STT Verification (Soniox)")
    print(f"  Soniox API Key: ...{SONIOX_API_KEY[-8:]}")
    print(f"  Input: {OUTPUT_DIR}")
    print("=" * 80 + "\n")

    results = []
    for i, entry in enumerate(metadata, 1):
        test_id = entry["test_id"]
        llm_text = entry.get("llm_text", "")
        wav_path = entry.get("wav_path", "")
        lang_hints = entry.get("language_hints", ["en", "hi"])

        print(f"\n  [{i}/{len(metadata)}] {test_id}")

        if not wav_path or not os.path.exists(wav_path):
            print(f"    ⏭️  No WAV file, skipping")
            results.append(STTVerifyResult(prompt=entry["prompt"], errors=["No WAV file"]))
            continue

        if not llm_text:
            print(f"    ⏭️  No LLM text, skipping")
            results.append(STTVerifyResult(prompt=entry["prompt"], errors=["No LLM text"]))
            continue

        result = STTVerifyResult(
            prompt=entry["prompt"],
            llm_text=llm_text,
            audio_bytes=entry.get("audio_bytes", 0),
            audio_duration_sec=entry.get("audio_duration_sec", 0),
            ttfab_ms=entry.get("ttfab_ms", -1),
            wav_path=wav_path,
        )

        try:
            # Read WAV and get raw PCM
            with wave.open(wav_path, "rb") as wf:
                src_rate = wf.getframerate()
                audio_pcm16 = wf.readframes(wf.getnframes())

            # Resample to 16kHz for Soniox
            audio_16k = resample_pcm16(audio_pcm16, src_rate, SONIOX_SAMPLE_RATE)

            # Transcribe with retry
            stt_text, stt_tokens = "", []
            for attempt in range(3):
                try:
                    stt_text, stt_tokens = await transcribe_with_soniox(
                        audio_16k, SONIOX_SAMPLE_RATE, lang_hints
                    )
                    break
                except ConnectionError as e:
                    if attempt < 2:
                        wait = 5 * (attempt + 1)
                        logger.warning(f"    Retry {attempt+1}/3: waiting {wait}s ({e})")
                        await asyncio.sleep(wait)
                    else:
                        result.errors.append("Soniox rate limited after 3 retries")

            result.stt_text = stt_text
            result.stt_tokens = stt_tokens
            result.stt_word_count = len(stt_text.split()) if stt_text else 0

            # Compare
            if stt_text:
                score, missing, extra = word_overlap_score(llm_text, stt_text)
                result.word_score = score
                result.missing_words = missing
                result.extra_words = extra

                # Timing gaps
                final_tokens = [t for t in stt_tokens if t.get("is_final", False)]
                result.timing_gaps = detect_timing_gaps(final_tokens, gap_threshold_ms=1500)

            result.passed = result.word_score >= 0.70 and len(result.errors) == 0

        except Exception as e:
            result.errors.append(f"Exception: {e}")
            import traceback
            traceback.print_exc()

        results.append(result)

        # Print
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"    {status}  Word Match: {result.word_score:.0%} ({result.stt_word_count} STT words)")
        if result.llm_text:
            print(f"    LLM: '{result.llm_text[:80]}...'")
        if result.stt_text:
            print(f"    STT: '{result.stt_text[:80]}...'")
        if result.missing_words:
            print(f"    ⚠️  Missing: {' '.join(result.missing_words[:15])}")
        if result.timing_gaps:
            for gap in result.timing_gaps[:3]:
                print(f"    ⏱️  {gap['gap_ms']:.0f}ms gap after '{gap['after_word']}' at {gap['position_ms']/1000:.1f}s")
        if result.errors:
            print(f"    ❌ {result.errors}")

    # Summary
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY")
    print(f"{'=' * 80}")

    passed = sum(1 for r in results if r.passed)
    avg_score = sum(r.word_score for r in results) / len(results) if results else 0

    print(f"\n  Results: {passed}/{len(results)} passed")
    print(f"  Average word match: {avg_score:.0%}\n")

    print(f"  {'ID':<16} {'Score':>6} {'Audio':>7} {'STT Words':>10} {'Gaps':>5} {'Status':<6}")
    print(f"  {'─'*16} {'─'*6} {'─'*7} {'─'*10} {'─'*5} {'─'*6}")
    for entry, r in zip(metadata, results):
        status = "PASS" if r.passed else "FAIL"
        gaps = len(r.timing_gaps)
        print(f"  {entry['test_id']:<16} {r.word_score:>5.0%} {r.audio_duration_sec:>6.1f}s {r.stt_word_count:>10} {gaps:>5} {status:<6}")

    print(f"\n{'=' * 80}\n")
    sys.exit(0 if passed == len(results) else 1)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="STT Verification Test")
    parser.add_argument("--url", default=os.getenv("PIPECAT_WS_URL", "ws://localhost:7860/ws"),
                        help="WebSocket URL")
    parser.add_argument("--http", default=os.getenv("PIPECAT_HTTP_URL", ""),
                        help="HTTP URL for /inject_text")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tests (0=all)")
    parser.add_argument("--collect", action="store_true",
                        help="Phase 1 only: collect TTS audio and save WAV files")
    parser.add_argument("--verify", action="store_true",
                        help="Phase 2 only: transcribe saved WAVs with Soniox and compare")
    args = parser.parse_args()

    ws_url = args.url
    http_url = args.http
    if not http_url:
        http_url = ws_url.replace("ws://", "http://").replace("wss://", "https://").replace("/ws", "")

    # Test cases: (id, prompt, language_hints)
    test_cases = [
        ("en_short", "Hello, how are you?", ["en"]),
        ("en_medium", "What is photosynthesis? Explain briefly.", ["en"]),
        ("en_long", "Explain photosynthesis in detail. How do plants use sunlight, water, and carbon dioxide to make food? Give me a complete explanation with examples.", ["en"]),
        ("hi_medium", "Photosynthesis क्या है? मुझे हिंदी में विस्तार से बताओ।", ["en", "hi"]),
        ("en_water_cycle", "Explain the water cycle in detail. Start from evaporation, then condensation, precipitation, and collection. How does each stage work?", ["en"]),
        ("en_history", "Tell me about the French Revolution. What caused it and who were the key people involved?", ["en"]),
    ]

    if args.limit > 0:
        test_cases = test_cases[:args.limit]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.collect:
        # Phase 1 only: collect audio
        await collect_only(ws_url, http_url, test_cases)
    elif args.verify:
        # Phase 2 only: transcribe and compare
        await verify_only(test_cases)
    else:
        # Full run: collect + verify (works when no Soniox concurrency issue)
        if not SONIOX_API_KEY:
            sys.exit("ERROR: SONIOX_API_KEY not set. Use --collect to save audio, then --verify separately.")

        print("\n" + "=" * 80)
        print("  STT Verification Test: TTS Audio → Soniox STT → Text Comparison")
        print(f"  Target: {ws_url}")
        print(f"  Soniox API Key: ...{SONIOX_API_KEY[-8:]}")
        print(f"  Output: {OUTPUT_DIR}")
        print("=" * 80 + "\n")

        results = []
        for i, (test_id, prompt, lang_hints) in enumerate(test_cases, 1):
            print(f"\n{'─' * 70}")
            print(f"  Test {i}/{len(test_cases)} [{test_id}]: \"{prompt[:60]}...\"")
            print(f"{'─' * 70}")

            result = await run_stt_verify_test(ws_url, http_url, prompt, test_id, lang_hints)
            results.append(result)

            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"\n  {status}  Word Match: {result.word_score:.0%}")
            print(f"  Audio: {result.audio_duration_sec:.1f}s | STT Words: {result.stt_word_count} | TTFAB: {result.ttfab_ms:.0f}ms")

            if result.llm_text:
                print(f"  LLM text: '{result.llm_text[:100]}{'...' if len(result.llm_text) > 100 else ''}'")
            if result.stt_text:
                print(f"  STT text: '{result.stt_text[:100]}{'...' if len(result.stt_text) > 100 else ''}'")
            if result.missing_words:
                print(f"  ⚠️  Missing from audio: {' '.join(result.missing_words[:15])}")
            if result.extra_words:
                print(f"  ℹ️  Extra in STT: {' '.join(result.extra_words[:10])}")
            if result.timing_gaps:
                print(f"  ⏱️  Timing gaps (>{1500}ms):")
                for gap in result.timing_gaps[:5]:
                    print(f"     {gap['gap_ms']:.0f}ms gap after '{gap['after_word']}' → '{gap['before_word']}' at {gap['position_ms']/1000:.1f}s")
            if result.errors:
                print(f"  ❌ Errors: {result.errors}")
            if result.wav_path:
                print(f"  📁 WAV: {result.wav_path}")

        # Summary
        passed = sum(1 for r in results if r.passed)
        avg_score = sum(r.word_score for r in results) / len(results) if results else 0
        print(f"\n{'=' * 80}")
        print(f"  RESULTS: {passed}/{len(results)} passed | Avg word match: {avg_score:.0%}")
        print(f"{'=' * 80}\n")
        sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
