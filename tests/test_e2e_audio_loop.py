#!/usr/bin/env python3
"""
End-to-End Audio Loop Test — the test that catches what manual testing catches.

Closes the loop that our other tests miss:
  1. Send speech audio → Pipecat pipeline
  2. Receive bot audio back (PCM via protobuf)
  3. Save bot audio as WAV
  4. Run Soniox STT on the WAV → verify transcript matches bot_text
  5. Analyze PCM waveform for pops/clicks (zero-crossing discontinuities)
  6. Measure silence gaps between audio chunks
  7. Multi-turn: verify consistent quality across turns

Usage:
    # Via Docker Compose (recommended)
    docker compose run --rm --no-deps \
        -e PIPECAT_WS_URL=ws://mira-voice-svara:7860/ws \
        -e WEBUI_SECRET_KEY=... \
        -e SONIOX_API_KEY=... \
        --entrypoint python test-voice tests/test_e2e_audio_loop.py

    # Direct
    python tests/test_e2e_audio_loop.py --url ws://localhost:7861/ws
    python tests/test_e2e_audio_loop.py --url ws://localhost:7861/ws --no-soniox
"""

import argparse
import asyncio
import json
import logging
import math
import os
import struct
import sys
import time
import wave
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import websockets
except ImportError:
    print("ERROR: pip install websockets")
    sys.exit(1)

try:
    import jwt as pyjwt
except ImportError:
    pyjwt = None

try:
    import pipecat.frames.protobufs.frames_pb2 as frame_protos
except ImportError:
    frame_protos = None
    print("ERROR: pipecat not installed")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("e2e_audio")

# ── Constants ──
INPUT_SAMPLE_RATE = 16000   # What we send (mic audio)
OUTPUT_SAMPLE_RATE = 24000  # What Svara TTS returns
SONIOX_SAMPLE_RATE = 16000  # Soniox expects 16kHz
CHUNK_MS = 20

# ── Soniox ──
SONIOX_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
SONIOX_API_KEY = os.getenv("SONIOX_API_KEY", "").strip()

# ── Thresholds ──
POP_THRESHOLD = 15000       # Max sample-to-sample jump considered "smooth" (24kHz PCM)
SILENCE_GAP_WARN_MS = 300   # Warn if gap between audio chunks > this
STT_MATCH_THRESHOLD = 0.50  # Min word overlap ratio to pass


# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AudioChunkInfo:
    """Metadata for a received audio chunk."""
    timestamp: float       # monotonic time when received
    pcm_bytes: bytes       # raw PCM16 data
    sample_rate: int = OUTPUT_SAMPLE_RATE


@dataclass
class TurnResult:
    """Full result of one conversation turn."""
    turn_num: int
    label: str
    idle_before_sec: float

    # Timing
    speech_start: float = 0.0
    first_transcript_ms: Optional[float] = None
    first_audio_ms: Optional[float] = None
    last_audio_ms: Optional[float] = None
    total_audio_duration_sec: float = 0.0

    # Content
    stt_text: str = ""
    bot_text: str = ""
    bot_text_streaming: str = ""

    # Audio analysis
    audio_chunks: List[AudioChunkInfo] = field(default_factory=list)
    total_audio_bytes: int = 0
    pop_count: int = 0
    max_discontinuity: float = 0.0
    inter_chunk_gaps_ms: List[float] = field(default_factory=list)
    silence_gaps_detected: int = 0

    # Soniox STT verification
    soniox_text: str = ""
    soniox_match_ratio: float = 0.0

    # Overall
    passed: bool = True
    issues: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def make_jwt(secret: str) -> str:
    if not pyjwt:
        return ""
    payload = {"id": "test-e2e-user", "role": "user", "name": "E2EAudioTest"}
    return pyjwt.encode(payload, secret, algorithm="HS256")


def make_audio_frame(pcm_bytes: bytes) -> bytes:
    frame = frame_protos.Frame()
    frame.audio.audio = pcm_bytes
    frame.audio.sample_rate = INPUT_SAMPLE_RATE
    frame.audio.num_channels = 1
    return frame.SerializeToString()


def parse_audio_frame(data: bytes) -> Optional[bytes]:
    """Extract raw PCM from a protobuf Frame. Returns None if not audio."""
    try:
        proto = frame_protos.Frame.FromString(data)
        if proto.WhichOneof("frame") == "audio":
            return proto.audio.audio
    except Exception:
        pass
    return None


def load_wav(path: str) -> bytes:
    with wave.open(path, 'rb') as wf:
        return wf.readframes(wf.getnframes())


def save_wav(path: str, pcm: bytes, sample_rate: int = OUTPUT_SAMPLE_RATE):
    """Save raw PCM16 mono bytes as a WAV file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def generate_silence(duration_sec: float, sr: int = INPUT_SAMPLE_RATE) -> bytes:
    return np.zeros(int(sr * duration_sec), dtype=np.int16).tobytes()


def resample_pcm16(audio_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample PCM16 mono audio."""
    if from_rate == to_rate:
        return audio_bytes
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    ratio = to_rate / from_rate
    out_len = int(len(samples) * ratio)
    indices = np.arange(out_len) / ratio
    idx0 = np.floor(indices).astype(int)
    idx1 = np.minimum(idx0 + 1, len(samples) - 1)
    frac = indices - idx0
    resampled = (samples[idx0] * (1 - frac) + samples[idx1] * frac).astype(np.int16)
    return resampled.tobytes()


# ═══════════════════════════════════════════════════════════════════
# Audio Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_pops(pcm_bytes: bytes) -> tuple:
    """Detect pops/clicks in PCM audio by finding large sample-to-sample jumps."""
    if len(pcm_bytes) < 4:
        return 0, 0.0, []
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float64)
    diffs = np.abs(np.diff(samples))
    pop_positions = np.where(diffs > POP_THRESHOLD)[0]
    pop_count = len(pop_positions)
    max_disc = float(np.max(diffs)) if len(diffs) > 0 else 0.0
    return pop_count, max_disc, pop_positions.tolist()


def analyze_chunk_gaps(chunks: List[AudioChunkInfo]) -> tuple:
    """Analyze timing gaps between consecutive audio chunks."""
    if len(chunks) < 2:
        return [], 0
    gaps = []
    large_gaps = 0
    for i in range(1, len(chunks)):
        gap_ms = (chunks[i].timestamp - chunks[i - 1].timestamp) * 1000
        prev_duration_ms = len(chunks[i - 1].pcm_bytes) / (2 * chunks[i - 1].sample_rate) * 1000
        actual_gap = gap_ms - prev_duration_ms
        gaps.append(actual_gap)
        if actual_gap > SILENCE_GAP_WARN_MS:
            large_gaps += 1
    return gaps, large_gaps


def word_overlap_ratio(text_a: str, text_b: str) -> float:
    """Compute word overlap ratio between two texts (order-independent)."""
    if not text_a or not text_b:
        return 0.0
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a:
        return 0.0
    overlap = words_a & words_b
    return len(overlap) / max(len(words_a), len(words_b))


# ═══════════════════════════════════════════════════════════════════
# Soniox STT Verification
# ═══════════════════════════════════════════════════════════════════

async def transcribe_with_soniox(audio_pcm16: bytes, sample_rate: int = 16000) -> str:
    """Send audio to Soniox realtime STT and return transcript text."""
    if not SONIOX_API_KEY:
        return ""

    final_text_parts = []
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
        return ""

    try:
        config = {
            "api_key": SONIOX_API_KEY,
            "model": "stt-rt-v3",
            "language_hints": ["en", "hi"],
            "language_hints_strict": True,
            "enable_language_identification": True,
            "enable_endpoint_detection": False,
            "audio_format": "pcm_s16le",
            "sample_rate": sample_rate,
            "num_channels": 1,
        }
        await ws.send(json.dumps(config))

        # Wait for ack
        try:
            ack = await asyncio.wait_for(ws.recv(), timeout=5.0)
            ack_str = ack if isinstance(ack, str) else ack.decode("utf-8", errors="replace")
            try:
                ack_data = json.loads(ack_str)
                if ack_data.get("error_code") == 429:
                    await ws.close()
                    raise ConnectionError(f"Soniox 429: {ack_data.get('error_message', '')}")
            except json.JSONDecodeError:
                pass
        except asyncio.TimeoutError:
            pass

        # Pad with 1s silence
        silence_pad = b"\x00" * (sample_rate * 2)
        padded_audio = audio_pcm16 + silence_pad

        # Stream at ~1.2x real-time
        chunk_duration_ms = 100
        chunk_size = int(sample_rate * 2 * chunk_duration_ms / 1000)
        offset = 0
        while offset < len(padded_audio):
            chunk = padded_audio[offset:offset + chunk_size]
            try:
                await ws.send(chunk)
            except websockets.exceptions.ConnectionClosed:
                break
            offset += chunk_size
            await asyncio.sleep(chunk_duration_ms / 1000 / 1.2)

        # Signal end of audio
        try:
            await ws.send("")
        except websockets.exceptions.ConnectionClosed:
            pass

        # Collect responses
        collect_start = time.time()
        while time.time() - collect_start < 30.0:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=8.0)
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                break

            if isinstance(msg, bytes):
                continue

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue

            if "error" in data:
                break

            tokens = data.get("tokens", [])
            msg_nonfinal_parts = []
            for token in tokens:
                token_text = token.get("text", "")
                if token_text == "<end>" or not token_text:
                    if token_text == "<end>":
                        # Done
                        final_text = "".join(final_text_parts).strip()
                        if len(final_text) >= len(best_interim_text):
                            return final_text
                        return best_interim_text.strip()
                    continue
                if token.get("is_final", False):
                    final_text_parts.append(token_text)
                else:
                    msg_nonfinal_parts.append(token_text)

            current_full = "".join(final_text_parts) + "".join(msg_nonfinal_parts)
            if len(current_full) > len(best_interim_text):
                best_interim_text = current_full

    except Exception as e:
        logger.error(f"  Soniox error: {e}")
    finally:
        try:
            await ws.close()
        except Exception:
            pass

    final_text = "".join(final_text_parts).strip()
    if len(final_text) >= len(best_interim_text):
        return final_text
    return best_interim_text.strip()


# ═══════════════════════════════════════════════════════════════════
# WebSocket Client
# ═══════════════════════════════════════════════════════════════════

async def send_audio_stream(ws, pcm_bytes: bytes):
    """Send PCM audio in realtime chunks via protobuf."""
    chunk_samples = int(INPUT_SAMPLE_RATE * CHUNK_MS / 1000)
    chunk_bytes = chunk_samples * 2
    offset = 0
    while offset < len(pcm_bytes):
        chunk = pcm_bytes[offset:offset + chunk_bytes]
        await ws.send(make_audio_frame(chunk))
        offset += chunk_bytes
        await asyncio.sleep(CHUNK_MS / 1000.0 * 0.9)


async def drain_greeting(ws, timeout: float = 15.0):
    """Drain the greeting audio and text until silence (no messages for 2s)."""
    greeting_text = ""
    greeting_audio_chunks = 0
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 2.0))
        except asyncio.TimeoutError:
            # 2s of silence after greeting = greeting is done
            break

        if isinstance(msg, str):
            try:
                data = json.loads(msg)
                if data.get("type") == "bot_text_complete":
                    greeting_text = data.get("text", "")
            except json.JSONDecodeError:
                pass
        elif isinstance(msg, bytes):
            pcm = parse_audio_frame(msg)
            if pcm:
                greeting_audio_chunks += 1

    logger.info(f"  Greeting drained: '{greeting_text[:60]}...' ({greeting_audio_chunks} audio chunks)")
    return greeting_text, greeting_audio_chunks


async def run_turn(
    ws,
    pcm_audio: bytes,
    turn_num: int,
    label: str,
    idle_sec: float,
    timeout: float = 30.0,
) -> TurnResult:
    """Execute one conversation turn and collect all results."""
    result = TurnResult(turn_num=turn_num, label=label, idle_before_sec=idle_sec)

    # ── Idle phase — drain any residual audio from previous turn ──
    logger.info(f"  [{label}] Waiting {idle_sec:.0f}s idle...")
    drain_end = time.monotonic() + idle_sec
    drained = 0
    while time.monotonic() < drain_end:
        remaining = drain_end - time.monotonic()
        if remaining <= 0:
            break
        try:
            _ = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 0.5))
            drained += 1
        except asyncio.TimeoutError:
            # No more messages — sleep the rest
            rest = drain_end - time.monotonic()
            if rest > 0:
                await asyncio.sleep(rest)
            break
    if drained > 0:
        logger.info(f"  [{label}] Drained {drained} residual messages during idle")

    # ── Send speech ──
    result.speech_start = time.monotonic()
    logger.info(f"  [{label}] Sending speech...")
    await send_audio_stream(ws, pcm_audio)
    # Trailing silence for VAD stop
    await send_audio_stream(ws, generate_silence(1.5))

    # ── Collect response ──
    bot_text_parts = []
    deadline = time.monotonic() + timeout
    got_text_complete = False
    audio_done_deadline = None  # Set when we decide to stop collecting

    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        # If we have text_complete + some audio, wait up to 3s more for trailing audio
        if audio_done_deadline and time.monotonic() > audio_done_deadline:
            break

        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 2.0))
        except asyncio.TimeoutError:
            if got_text_complete:
                break
            continue

        now = time.monotonic()

        if isinstance(msg, str):
            try:
                data = json.loads(msg)
                msg_type = data.get("type", "")

                if msg_type == "user_transcript":
                    since = (now - result.speech_start) * 1000
                    if result.first_transcript_ms is None:
                        result.first_transcript_ms = since
                    if data.get("final", False):
                        result.stt_text = data.get("text", "")

                elif msg_type == "bot_text":
                    bot_text_parts.append(data.get("text", ""))

                elif msg_type == "bot_text_complete":
                    result.bot_text = data.get("text", "")
                    got_text_complete = True
                    # Give 3 more seconds for trailing audio
                    audio_done_deadline = time.monotonic() + 3.0

            except json.JSONDecodeError:
                pass

        elif isinstance(msg, bytes):
            pcm = parse_audio_frame(msg)
            if pcm and len(pcm) > 0:
                chunk = AudioChunkInfo(timestamp=now, pcm_bytes=pcm)
                result.audio_chunks.append(chunk)
                result.total_audio_bytes += len(pcm)

                since = (now - result.speech_start) * 1000
                if result.first_audio_ms is None:
                    result.first_audio_ms = since
                result.last_audio_ms = since

                # Extend audio collection deadline if we're still getting audio
                if audio_done_deadline:
                    audio_done_deadline = time.monotonic() + 3.0

    if not result.bot_text and bot_text_parts:
        result.bot_text = "".join(bot_text_parts)
    result.bot_text_streaming = "".join(bot_text_parts)

    # Calculate audio duration
    if result.total_audio_bytes > 0:
        result.total_audio_duration_sec = result.total_audio_bytes / (2 * OUTPUT_SAMPLE_RATE)

    return result


def analyze_turn(result: TurnResult, output_dir: str, use_soniox: bool = True) -> TurnResult:
    """Post-process a turn: pop analysis, gap analysis, Soniox verification."""

    # ── Combine all audio chunks into one PCM buffer ──
    all_pcm = b"".join(c.pcm_bytes for c in result.audio_chunks)

    if not all_pcm:
        result.passed = False
        result.issues.append("No audio received")
        return result

    # ── Save WAV ──
    wav_path = os.path.join(output_dir, f"turn{result.turn_num:02d}_bot_audio.wav")
    save_wav(wav_path, all_pcm, OUTPUT_SAMPLE_RATE)

    # ── Pop/click analysis ──
    pop_count, max_disc, pop_positions = analyze_pops(all_pcm)
    result.pop_count = pop_count
    result.max_discontinuity = max_disc

    if pop_count > 5:
        result.issues.append(f"Pops detected: {pop_count} (max jump: {max_disc:.0f})")

    # ── Inter-chunk gap analysis ──
    gaps, large_gaps = analyze_chunk_gaps(result.audio_chunks)
    result.inter_chunk_gaps_ms = gaps
    result.silence_gaps_detected = large_gaps

    if large_gaps > 0:
        worst_gap = max(gaps) if gaps else 0
        result.issues.append(f"Large silence gaps: {large_gaps} (worst: {worst_gap:.0f}ms)")

    # ── Timing checks ──
    if result.first_transcript_ms is None:
        result.issues.append("No STT transcript received")
    if result.first_audio_ms is None:
        result.issues.append("No bot audio received")
    elif result.first_audio_ms > 10000:
        result.issues.append(f"Slow TTFAB: {result.first_audio_ms:.0f}ms (>10s)")

    # ── Audio duration sanity check ──
    if result.total_audio_duration_sec < 1.0 and result.bot_text:
        word_count = len(result.bot_text.split())
        if word_count > 5:
            result.issues.append(
                f"Very short audio ({result.total_audio_duration_sec:.1f}s) "
                f"for {word_count} words"
            )

    result.passed = len(result.issues) == 0
    return result


async def soniox_verify_turn(result: TurnResult, output_dir: str) -> TurnResult:
    """Run Soniox STT on the turn's audio and compare with bot_text."""
    all_pcm = b"".join(c.pcm_bytes for c in result.audio_chunks)
    if not all_pcm or not result.bot_text:
        return result

    # Resample 24kHz → 16kHz for Soniox
    audio_16k = resample_pcm16(all_pcm, OUTPUT_SAMPLE_RATE, SONIOX_SAMPLE_RATE)

    # Wait for Soniox rate limit
    logger.info(f"  [{result.label}] Transcribing with Soniox ({len(all_pcm)} bytes)...")
    await asyncio.sleep(2.0)

    for attempt in range(3):
        try:
            result.soniox_text = await transcribe_with_soniox(audio_16k, SONIOX_SAMPLE_RATE)
            break
        except ConnectionError as e:
            if attempt < 2:
                wait = 5 * (attempt + 1)
                logger.warning(f"  Soniox retry {attempt+1}/3: waiting {wait}s ({e})")
                await asyncio.sleep(wait)
            else:
                result.issues.append("Soniox rate limited after 3 retries")
                return result

    if result.soniox_text:
        result.soniox_match_ratio = word_overlap_ratio(result.bot_text, result.soniox_text)
        if result.soniox_match_ratio < STT_MATCH_THRESHOLD:
            result.issues.append(
                f"Soniox mismatch: {result.soniox_match_ratio:.0%} overlap "
                f"(expected ≥{STT_MATCH_THRESHOLD:.0%})"
            )
            result.passed = False
    else:
        result.issues.append("Soniox returned empty transcript")
        result.passed = False

    return result


# ═══════════════════════════════════════════════════════════════════
# Test Scenarios
# ═══════════════════════════════════════════════════════════════════

# Each scenario: (idle_before_sec, label, wav_file_or_None)
SCENARIOS = [
    (2.0,  "Cold start (2s idle)",     "user_english.wav"),
    (3.0,  "Quick follow-up (3s)",     "user_english.wav"),
    (8.0,  "After response (8s idle)", "user_english.wav"),
    (3.0,  "Hindi turn",              "user_greeting.wav"),
    (5.0,  "Follow-up (5s)",          "user_english.wav"),
]


async def run_all(
    ws_url: str,
    jwt_secret: str,
    output_dir: str,
    use_soniox: bool = True,
    wav_dir: str = "",
):
    """Run all test scenarios on a single WebSocket connection."""
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "type": "config",
        "mode": "text_and_audio",
        "language": "hi",
    }
    if jwt_secret:
        config["token"] = make_jwt(jwt_secret)

    # Load audio files
    audio_cache = {}
    for _, _, wav_name in SCENARIOS:
        if wav_name and wav_name not in audio_cache:
            wav_path = os.path.join(wav_dir, wav_name)
            if os.path.exists(wav_path):
                audio_cache[wav_name] = load_wav(wav_path)
                logger.info(f"Loaded {wav_name}: {len(audio_cache[wav_name])/(INPUT_SAMPLE_RATE*2):.1f}s")
            else:
                logger.warning(f"WAV not found: {wav_path}")

    results = []

    try:
        async with websockets.connect(
            ws_url,
            max_size=10 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            await ws.send(json.dumps(config))

            # Wait for session_id
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                if isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("type") == "session_id":
                        logger.info(f"Session: {data.get('session_id', '')[:12]}...")
            except Exception:
                pass

            # ── DRAIN GREETING before starting turns ──
            await drain_greeting(ws, timeout=15.0)

            for i, (idle_sec, label, wav_name) in enumerate(SCENARIOS):
                pcm = audio_cache.get(wav_name)
                if not pcm:
                    logger.warning(f"  Skipping — no audio for {wav_name}")
                    continue

                # Run the turn
                turn_result = await run_turn(
                    ws, pcm, turn_num=i + 1, label=label, idle_sec=idle_sec,
                    timeout=30.0,
                )

                # Analyze (pops, gaps, timing — no STT yet)
                turn_result = analyze_turn(turn_result, output_dir, use_soniox=False)
                results.append(turn_result)

                logger.info(
                    f"  [{label}] Audio: {turn_result.total_audio_duration_sec:.1f}s "
                    f"({len(turn_result.audio_chunks)} chunks) | "
                    f"Pops: {turn_result.pop_count} | "
                    f"Bot text: '{turn_result.bot_text[:50]}...'"
                )

    except Exception as e:
        logger.error(f"Connection error: {e}")
        import traceback
        traceback.print_exc()

    # ── Soniox STT verification (after WS is closed, freeing Soniox slot) ──
    if use_soniox and SONIOX_API_KEY:
        logger.info(f"\n  Running Soniox STT verification on {len(results)} turns...")
        for r in results:
            if r.audio_chunks and r.bot_text:
                r = await soniox_verify_turn(r, output_dir)
    elif use_soniox:
        logger.warning("  SONIOX_API_KEY not set — skipping STT verification")

    return results


def print_turn_summary(r: TurnResult):
    """Print a concise summary for one turn."""
    status = "✅ PASS" if r.passed else "❌ FAIL"
    transcript_ms = f"{r.first_transcript_ms:.0f}ms" if r.first_transcript_ms else "NONE"
    audio_ms = f"{r.first_audio_ms:.0f}ms" if r.first_audio_ms else "NONE"

    print(f"  Status:          {status}")
    print(f"  1st transcript:  {transcript_ms}")
    print(f"  1st bot audio:   {audio_ms}")
    print(f"  Audio duration:  {r.total_audio_duration_sec:.1f}s ({len(r.audio_chunks)} chunks)")
    print(f"  STT text:        '{r.stt_text[:50]}'")
    print(f"  Bot text:        '{r.bot_text[:60]}'")
    print(f"  Pops detected:   {r.pop_count} (max jump: {r.max_discontinuity:.0f})")

    if r.inter_chunk_gaps_ms:
        avg_gap = sum(r.inter_chunk_gaps_ms) / len(r.inter_chunk_gaps_ms)
        max_gap = max(r.inter_chunk_gaps_ms)
        print(f"  Chunk gaps:      avg={avg_gap:.0f}ms, max={max_gap:.0f}ms, large={r.silence_gaps_detected}")

    if r.soniox_text:
        print(f"  Soniox text:     '{r.soniox_text[:60]}'")
        print(f"  Soniox match:    {r.soniox_match_ratio:.0%}")

    if r.issues:
        for issue in r.issues:
            print(f"  ⚠️  {issue}")


def print_final_report(results: List[TurnResult], output_dir: str):
    """Print the final summary report."""
    print(f"\n{'═' * 70}")
    print(f"  END-TO-END AUDIO LOOP TEST — FINAL REPORT")
    print(f"{'═' * 70}")

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    # Summary table
    print(f"\n{'Turn':<6} {'Label':<30} {'TTFAB':<10} {'Audio':<8} {'Pops':<6} {'Gaps':<6} {'Soniox':<10} {'Status'}")
    print(f"{'─' * 90}")

    for r in results:
        ttfab = f"{r.first_audio_ms:.0f}ms" if r.first_audio_ms else "NONE"
        audio = f"{r.total_audio_duration_sec:.1f}s"
        pops = f"{r.pop_count}"
        gaps = f"{r.silence_gaps_detected}"
        soniox = f"{r.soniox_match_ratio:.0%}" if r.soniox_text else "n/a"
        status = "✅" if r.passed else "❌"
        print(f"{r.turn_num:<6} {r.label:<30} {ttfab:<10} {audio:<8} {pops:<6} {gaps:<6} {soniox:<10} {status}")

    # Aggregate metrics
    print(f"\n{'─' * 70}")

    ttfabs = [r.first_audio_ms for r in results if r.first_audio_ms]
    if ttfabs:
        print(f"  TTFAB:  avg={sum(ttfabs)/len(ttfabs):.0f}ms  "
              f"min={min(ttfabs):.0f}ms  max={max(ttfabs):.0f}ms  "
              f"spread={max(ttfabs)-min(ttfabs):.0f}ms")

    total_pops = sum(r.pop_count for r in results)
    total_gaps = sum(r.silence_gaps_detected for r in results)
    print(f"  Pops:   {total_pops} total across all turns")
    print(f"  Gaps:   {total_gaps} large silence gaps (>{SILENCE_GAP_WARN_MS}ms)")

    soniox_ratios = [r.soniox_match_ratio for r in results if r.soniox_text]
    if soniox_ratios:
        avg_match = sum(soniox_ratios) / len(soniox_ratios)
        print(f"  Soniox: avg match={avg_match:.0%}  "
              f"min={min(soniox_ratios):.0%}  max={max(soniox_ratios):.0%}")

    print(f"\n  Result: {passed}/{total} turns passed")
    print(f"  WAV files saved to: {output_dir}/")

    if failed == 0 and total > 0:
        print(f"\n  🎉 ALL TURNS PASSED — audio pipeline is smooth!")
    else:
        print(f"\n  ⚠️  {failed} turn(s) had issues — check WAV files and logs above.")

    # List all issues
    all_issues = []
    for r in results:
        for issue in r.issues:
            all_issues.append(f"  Turn {r.turn_num}: {issue}")
    if all_issues:
        print(f"\n  All issues:")
        for issue in all_issues:
            print(f"    {issue}")

    print(f"{'═' * 70}")


async def main():
    parser = argparse.ArgumentParser(description="End-to-End Audio Loop Test")
    parser.add_argument("--url", default=None,
                        help="Pipecat WebSocket URL (or PIPECAT_WS_URL env)")
    parser.add_argument("--secret", default=None,
                        help="WEBUI_SECRET_KEY for JWT auth")
    parser.add_argument("--wav-dir", default=None,
                        help="Directory containing test WAV files")
    parser.add_argument("--output-dir", default="/tmp/e2e_audio_test",
                        help="Directory to save output WAV files")
    parser.add_argument("--no-soniox", action="store_true",
                        help="Skip Soniox STT verification")
    args = parser.parse_args()

    # Resolve URL
    ws_url = args.url or os.getenv("PIPECAT_WS_URL", "ws://mira-voice-svara:7860/ws")

    # Resolve JWT secret
    secret = args.secret or os.getenv("WEBUI_SECRET_KEY", "")

    # Resolve WAV directory
    wav_dir = args.wav_dir
    if not wav_dir:
        wav_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_audio")

    use_soniox = not args.no_soniox

    print("═" * 70)
    print("  MIRA End-to-End Audio Loop Test")
    print(f"  Target:  {ws_url}")
    print(f"  Auth:    {'JWT' if secret else 'none'}")
    print(f"  Soniox:  {'yes' if use_soniox and SONIOX_API_KEY else 'no (--no-soniox or no key)'}")
    print(f"  Output:  {args.output_dir}")
    print(f"  WAVs:    {wav_dir}")
    print("═" * 70)

    results = await run_all(
        ws_url=ws_url,
        jwt_secret=secret,
        output_dir=args.output_dir,
        use_soniox=use_soniox,
        wav_dir=wav_dir,
    )

    # Print per-turn details
    for r in results:
        print(f"\n{'─' * 60}")
        print(f"  Turn {r.turn_num}: {r.label}")
        print(f"{'─' * 60}")
        print_turn_summary(r)

    print_final_report(results, args.output_dir)

    # Exit code
    all_passed = all(r.passed for r in results) if results else False
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
