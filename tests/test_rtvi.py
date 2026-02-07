#!/usr/bin/env python3
"""
Automated RTVI Protocol Test Client for MiraVoiceAI Pipecat.

Speaks the proper Protobuf-framed RTVI protocol that the Pipecat server expects.
Tests: connection, greeting, full round-trip STT→LLM→TTS, and barge-in interruption.

Usage (inside Docker):
    python test_rtvi.py                     # Run all tests
    python test_rtvi.py --test greeting     # Run specific test
    python test_rtvi.py --test roundtrip
    python test_rtvi.py --test bargein
    python test_rtvi.py --test latency
    python test_rtvi.py --verbose           # Extra debug logging
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
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)

# Pipecat protobuf frames
import pipecat.frames.protobufs.frames_pb2 as frame_protos

# ─────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────
WS_URL = os.getenv("PIPECAT_WS_URL", "ws://mira-voice:7860/ws")
AUDIO_DIR = os.getenv("AUDIO_DIR", "/app/test_audio")  # Pre-recorded WAV files
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
CHUNK_DURATION_MS = 100  # Send 100ms audio chunks (like a real client)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rtvi-test")


# ─────────────────────────────────────────────────
# Protobuf helpers
# ─────────────────────────────────────────────────
def make_audio_frame(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw PCM16 audio bytes in a Pipecat protobuf Frame."""
    frame = frame_protos.Frame()
    frame.audio.audio = pcm_bytes
    frame.audio.sample_rate = sample_rate
    frame.audio.num_channels = NUM_CHANNELS
    return frame.SerializeToString()


def make_text_frame(text: str) -> bytes:
    """Wrap text in a Pipecat protobuf Frame."""
    frame = frame_protos.Frame()
    frame.text.text = text
    return frame.SerializeToString()


def parse_frame(data: bytes) -> dict:
    """Parse a Pipecat protobuf Frame and return a dict with its contents."""
    try:
        proto = frame_protos.Frame.FromString(data)
        which = proto.WhichOneof("frame")
        if which == "audio":
            return {
                "type": "audio",
                "length": len(proto.audio.audio),
                "sample_rate": proto.audio.sample_rate,
                "num_channels": proto.audio.num_channels,
            }
        elif which == "text":
            return {"type": "text", "text": proto.text.text}
        elif which == "transcription":
            return {
                "type": "transcription",
                "text": proto.transcription.text,
                "user_id": proto.transcription.user_id,
                "timestamp": proto.transcription.timestamp,
            }
        elif which == "message":
            try:
                msg = json.loads(proto.message.data)
                return {"type": "message", "data": msg}
            except json.JSONDecodeError:
                return {"type": "message", "data": proto.message.data}
        else:
            return {"type": "unknown", "which": which}
    except Exception as e:
        return {"type": "parse_error", "error": str(e)}


# ─────────────────────────────────────────────────
# Synthetic audio generation
# ─────────────────────────────────────────────────
def generate_sine_tone(
    freq_hz: float = 440.0,
    duration_sec: float = 2.0,
    amplitude: float = 0.5,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Generate a pure sine wave as PCM16 bytes (triggers VAD with energy)."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    samples = (amplitude * 32767 * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)
    return samples.tobytes()


def generate_speech_like_audio(
    duration_sec: float = 3.0,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """
    Generate audio that resembles human speech patterns to trigger VAD.
    Uses multiple harmonics with amplitude modulation (syllable rhythm).
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)

    # Fundamental frequency (typical male voice ~120Hz)
    f0 = 120.0
    signal = np.zeros_like(t, dtype=np.float64)

    # Add harmonics (speech has many harmonics)
    for harmonic in range(1, 8):
        amp = 1.0 / harmonic  # Natural harmonic falloff
        signal += amp * np.sin(2 * np.pi * f0 * harmonic * t)

    # Amplitude modulation to simulate syllables (~4Hz = 4 syllables/sec)
    syllable_rate = 4.0
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t)
    signal *= envelope

    # Normalize and convert to PCM16
    signal = signal / np.max(np.abs(signal))
    samples = (0.7 * 32767 * signal).astype(np.int16)
    return samples.tobytes()


def generate_silence(duration_sec: float = 1.0, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Generate silence as PCM16 bytes."""
    n_samples = int(sample_rate * duration_sec)
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def load_wav_file(path: str) -> tuple[bytes, int]:
    """Load a WAV file and return (pcm_bytes, sample_rate)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        data = wf.readframes(n)
        if wf.getnchannels() == 2:
            arr = np.frombuffer(data, dtype=np.int16).reshape(-1, 2).mean(axis=1).astype(np.int16)
            data = arr.tobytes()
        return data, sr


# ─────────────────────────────────────────────────
# Test result tracking
# ─────────────────────────────────────────────────
@dataclass
class TestResult:
    name: str
    passed: bool
    duration_sec: float = 0.0
    details: dict = field(default_factory=dict)
    error: Optional[str] = None


# ─────────────────────────────────────────────────
# Core test runner
# ─────────────────────────────────────────────────
class RTVITestClient:
    """RTVI-protocol-aware test client for Pipecat."""

    def __init__(self, ws_url: str = WS_URL):
        self.ws_url = ws_url
        self.ws = None
        self.received_audio_bytes = 0
        self.received_audio_chunks = 0
        self.received_texts = []
        self.received_transcriptions = []
        self.received_messages = []
        self._receive_task = None

    async def connect(self, timeout: float = 10.0):
        """Connect to Pipecat server via WebSocket."""
        logger.info(f"Connecting to {self.ws_url}")
        self.ws = await asyncio.wait_for(
            websockets.connect(self.ws_url, max_size=10 * 1024 * 1024),
            timeout=timeout,
        )
        logger.info("✅ Connected")

    async def disconnect(self):
        if self.ws:
            await self.ws.close()
            self.ws = None

    def _reset_counters(self):
        self.received_audio_bytes = 0
        self.received_audio_chunks = 0
        self.received_texts = []
        self.received_transcriptions = []
        self.received_messages = []

    async def _receive_loop(self, duration: float = 30.0):
        """Receive frames from the server for up to `duration` seconds."""
        deadline = time.time() + duration
        try:
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(self.ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue

                if isinstance(msg, bytes):
                    parsed = parse_frame(msg)
                    if parsed["type"] == "audio":
                        self.received_audio_chunks += 1
                        self.received_audio_bytes += parsed["length"]
                        if self.received_audio_chunks % 20 == 1:
                            logger.debug(
                                f"  🔊 Audio chunk #{self.received_audio_chunks} "
                                f"({self.received_audio_bytes} bytes total)"
                            )
                    elif parsed["type"] == "text":
                        self.received_texts.append(parsed["text"])
                        logger.info(f"  📝 Text: '{parsed['text']}'")
                    elif parsed["type"] == "transcription":
                        self.received_transcriptions.append(parsed["text"])
                        logger.info(f"  🎤 Transcription: '{parsed['text']}'")
                    elif parsed["type"] == "message":
                        self.received_messages.append(parsed["data"])
                        logger.debug(f"  📨 Message: {parsed['data']}")
                    else:
                        logger.debug(f"  ❓ {parsed}")
                else:
                    # Text message (JSON)
                    try:
                        data = json.loads(msg)
                        self.received_messages.append(data)
                        logger.debug(f"  📨 JSON: {data}")
                    except json.JSONDecodeError:
                        logger.debug(f"  📨 Raw text: {msg[:100]}")
        except websockets.ConnectionClosed as e:
            logger.warning(f"Connection closed: {e}")

    async def send_audio_stream(
        self, pcm_bytes: bytes, chunk_ms: int = CHUNK_DURATION_MS, realtime: bool = True
    ):
        """Stream audio to the server in protobuf-framed chunks."""
        chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
        chunk_bytes = chunk_samples * 2  # 16-bit PCM
        total_chunks = math.ceil(len(pcm_bytes) / chunk_bytes)

        logger.info(
            f"Streaming {len(pcm_bytes)} bytes of audio "
            f"({len(pcm_bytes) / (SAMPLE_RATE * 2):.1f}s) in {total_chunks} chunks"
        )

        offset = 0
        sent = 0
        while offset < len(pcm_bytes):
            chunk = pcm_bytes[offset : offset + chunk_bytes]
            proto_data = make_audio_frame(chunk)
            await self.ws.send(proto_data)
            sent += 1
            offset += chunk_bytes

            if realtime:
                await asyncio.sleep(chunk_ms / 1000.0 * 0.8)  # Slightly faster than real-time

        logger.info(f"Sent {sent} audio chunks")


# ─────────────────────────────────────────────────
# Individual tests
# ─────────────────────────────────────────────────
async def test_greeting(client: RTVITestClient) -> TestResult:
    """Test 1: Connect and verify the bot sends a greeting (TTS audio)."""
    start = time.time()
    try:
        client._reset_counters()
        await client.connect()

        # The server should send a greeting TTS as audio frames immediately
        logger.info("Waiting for greeting audio...")
        await client._receive_loop(duration=15.0)

        elapsed = time.time() - start
        got_audio = client.received_audio_chunks > 0

        details = {
            "audio_chunks": client.received_audio_chunks,
            "audio_bytes": client.received_audio_bytes,
            "text_frames": client.received_texts,
            "messages": len(client.received_messages),
        }

        if got_audio:
            logger.info(
                f"✅ Greeting received: {client.received_audio_chunks} audio chunks "
                f"({client.received_audio_bytes} bytes)"
            )
        else:
            logger.warning("⚠️  No greeting audio received (may still be OK if greeting is text-only)")

        return TestResult(
            name="greeting",
            passed=got_audio,
            duration_sec=elapsed,
            details=details,
        )
    except Exception as e:
        return TestResult(name="greeting", passed=False, error=str(e), duration_sec=time.time() - start)
    finally:
        await client.disconnect()


async def test_roundtrip(client: RTVITestClient) -> TestResult:
    """Test 2: Send real speech audio → get STT transcription → LLM → TTS audio back."""
    start = time.time()
    try:
        client._reset_counters()
        await client.connect()

        # Wait for greeting to finish first
        logger.info("Waiting for greeting to finish...")
        await client._receive_loop(duration=12.0)
        greeting_audio = client.received_audio_chunks
        logger.info(f"Greeting done ({greeting_audio} audio chunks)")

        # Reset counters for our actual test
        client._reset_counters()

        # Load real speech WAV file
        wav_path = os.path.join(AUDIO_DIR, "user_greeting.wav")
        if not os.path.exists(wav_path):
            # Fallback to synthetic if no WAV available
            logger.warning(f"WAV file not found: {wav_path}, using synthetic audio")
            speech_audio = generate_speech_like_audio(duration_sec=3.0)
        else:
            speech_audio, sr = load_wav_file(wav_path)
            logger.info(f"Loaded {wav_path}: {len(speech_audio)} bytes, {sr}Hz, {len(speech_audio)/(sr*2):.1f}s")
            if sr != SAMPLE_RATE:
                logger.warning(f"WAV sample rate {sr} != {SAMPLE_RATE}, audio may not work correctly")

        silence_after = generate_silence(duration_sec=1.5)

        # Send speech + trailing silence (for VAD to detect end of speech)
        logger.info("Sending real speech audio...")
        send_start = time.time()
        await client.send_audio_stream(speech_audio, realtime=True)
        await client.send_audio_stream(silence_after, realtime=True)
        speech_done = time.time()

        # Now wait for the response pipeline
        logger.info("Waiting for STT → LLM → TTS response...")
        await client._receive_loop(duration=30.0)

        elapsed = time.time() - start

        details = {
            "greeting_chunks": greeting_audio,
            "response_audio_chunks": client.received_audio_chunks,
            "response_audio_bytes": client.received_audio_bytes,
            "transcriptions": client.received_transcriptions,
            "text_frames": client.received_texts,
            "round_trip_sec": round(elapsed, 2),
            "time_from_speech_done_sec": round(time.time() - speech_done, 2),
        }

        got_response = client.received_audio_chunks > 0 or len(client.received_texts) > 0
        if got_response:
            logger.info(
                f"✅ Round-trip complete: {client.received_audio_chunks} audio chunks, "
                f"{len(client.received_transcriptions)} transcriptions, "
                f"{len(client.received_texts)} text frames"
            )
        else:
            logger.warning("⚠️  No response received — STT may not have transcribed the audio.")

        return TestResult(name="roundtrip", passed=got_response, duration_sec=elapsed, details=details)

    except Exception as e:
        return TestResult(name="roundtrip", passed=False, error=str(e), duration_sec=time.time() - start)
    finally:
        await client.disconnect()


async def test_bargein(client: RTVITestClient) -> TestResult:
    """Test 3: Send speech, wait for bot to respond, then interrupt with more speech."""
    start = time.time()
    try:
        client._reset_counters()
        await client.connect()

        # Wait for greeting
        logger.info("Waiting for greeting...")
        await client._receive_loop(duration=12.0)
        greeting_chunks = client.received_audio_chunks
        logger.info(f"Greeting done ({greeting_chunks} chunks)")

        # Load real speech WAV files
        wav1_path = os.path.join(AUDIO_DIR, "user_english.wav")
        wav2_path = os.path.join(AUDIO_DIR, "user_interrupt.wav")

        if os.path.exists(wav1_path):
            speech1, _ = load_wav_file(wav1_path)
            logger.info(f"Loaded initial speech: {wav1_path}")
        else:
            speech1 = generate_speech_like_audio(duration_sec=3.0)
            logger.warning("Using synthetic audio for initial speech")

        if os.path.exists(wav2_path):
            interrupt_audio, _ = load_wav_file(wav2_path)
            logger.info(f"Loaded interrupt speech: {wav2_path}")
        else:
            interrupt_audio = generate_speech_like_audio(duration_sec=2.0)
            logger.warning("Using synthetic audio for interrupt")

        silence1 = generate_silence(duration_sec=1.5)

        # Send initial speech
        client._reset_counters()
        logger.info("Sending initial speech...")
        await client.send_audio_stream(speech1, realtime=True)
        await client.send_audio_stream(silence1, realtime=True)

        # Wait for bot to START responding (get some audio back)
        logger.info("Waiting for bot response to start...")
        await client._receive_loop(duration=15.0)

        first_response_chunks = client.received_audio_chunks
        logger.info(f"Bot responding: {first_response_chunks} audio chunks so far")

        if first_response_chunks == 0:
            logger.warning("⚠️  No response audio to interrupt.")
            return TestResult(
                name="bargein",
                passed=False,
                error="No response audio to interrupt",
                duration_sec=time.time() - start,
            )

        # NOW INTERRUPT: send more speech while bot is talking
        logger.info("🔴 BARGE-IN: Sending interrupt audio while bot is speaking...")
        interrupt_start = time.time()
        pre_interrupt_audio = client.received_audio_bytes
        client._reset_counters()

        await client.send_audio_stream(interrupt_audio, realtime=True)
        silence2 = generate_silence(duration_sec=1.5)
        await client.send_audio_stream(silence2, realtime=True)

        # Wait for new response
        logger.info("Waiting for post-barge-in response...")
        await client._receive_loop(duration=15.0)

        elapsed = time.time() - start
        interrupt_latency = time.time() - interrupt_start

        details = {
            "greeting_chunks": greeting_chunks,
            "first_response_chunks": first_response_chunks,
            "pre_interrupt_audio_bytes": pre_interrupt_audio,
            "post_interrupt_audio_chunks": client.received_audio_chunks,
            "post_interrupt_audio_bytes": client.received_audio_bytes,
            "interrupt_latency_sec": round(interrupt_latency, 2),
            "transcriptions": client.received_transcriptions,
        }

        passed = first_response_chunks > 0
        logger.info(
            f"{'✅' if passed else '⚠️ '} Barge-in test: "
            f"first response={first_response_chunks} chunks, "
            f"post-interrupt={client.received_audio_chunks} chunks"
        )

        return TestResult(name="bargein", passed=passed, duration_sec=elapsed, details=details)

    except Exception as e:
        return TestResult(name="bargein", passed=False, error=str(e), duration_sec=time.time() - start)
    finally:
        await client.disconnect()


async def test_latency(client: RTVITestClient) -> TestResult:
    """Test 4: Measure time-to-first-audio-byte after sending speech."""
    start = time.time()
    try:
        client._reset_counters()
        await client.connect()

        # Skip greeting
        logger.info("Waiting for greeting...")
        await client._receive_loop(duration=12.0)
        client._reset_counters()

        # Use real speech WAV for accurate latency measurement
        wav_path = os.path.join(AUDIO_DIR, "user_short.wav")
        if os.path.exists(wav_path):
            speech, _ = load_wav_file(wav_path)
            logger.info(f"Loaded {wav_path}: {len(speech)/(SAMPLE_RATE*2):.1f}s")
        else:
            speech = generate_speech_like_audio(duration_sec=2.5)
            logger.warning("Using synthetic audio for latency test")
        silence = generate_silence(duration_sec=1.5)

        logger.info("Sending speech for latency measurement...")
        send_start = time.time()
        await client.send_audio_stream(speech, realtime=True)
        await client.send_audio_stream(silence, realtime=True)
        speech_done_time = time.time()

        # Now listen for response, tracking exact time of first audio
        first_audio_at = None
        deadline = time.time() + 30.0

        try:
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(client.ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    if first_audio_at and client.received_audio_chunks > 5:
                        break
                    continue

                if isinstance(msg, bytes):
                    parsed = parse_frame(msg)
                    if parsed["type"] == "audio":
                        client.received_audio_chunks += 1
                        client.received_audio_bytes += parsed["length"]
                        if first_audio_at is None:
                            first_audio_at = time.time()
                    elif parsed["type"] == "text":
                        client.received_texts.append(parsed["text"])
                    elif parsed["type"] == "transcription":
                        client.received_transcriptions.append(parsed["text"])
        except websockets.ConnectionClosed:
            pass

        elapsed = time.time() - start

        if first_audio_at:
            ttfb_from_send = first_audio_at - send_start
            ttfb_from_silence = first_audio_at - speech_done_time
            logger.info(
                f"✅ TTFB (from speech start): {ttfb_from_send:.2f}s | "
                f"TTFB (from silence): {ttfb_from_silence:.2f}s"
            )
            details = {
                "ttfb_from_speech_start_sec": round(ttfb_from_send, 3),
                "ttfb_from_silence_sec": round(ttfb_from_silence, 3),
                "total_audio_chunks": client.received_audio_chunks,
                "total_audio_bytes": client.received_audio_bytes,
                "transcriptions": client.received_transcriptions,
            }
            return TestResult(name="latency", passed=True, duration_sec=elapsed, details=details)
        else:
            logger.warning("⚠️  No response audio received for latency measurement")
            return TestResult(
                name="latency",
                passed=False,
                error="No audio response received",
                duration_sec=elapsed,
                details={
                    "transcriptions": client.received_transcriptions,
                    "texts": client.received_texts,
                },
            )

    except Exception as e:
        return TestResult(name="latency", passed=False, error=str(e), duration_sec=time.time() - start)
    finally:
        await client.disconnect()


# ─────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────
async def test_streaming_quality(client: RTVITestClient) -> TestResult:
    """Test 5: Measure audio streaming cadence — is audio smooth or bursty?"""
    start = time.time()
    try:
        client._reset_counters()
        await client.connect()

        # Skip greeting — but measure its streaming quality too
        logger.info("Receiving greeting (measuring streaming pattern)...")
        greeting_timestamps = []
        greeting_sizes = []
        deadline = time.time() + 18.0
        try:
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(client.ws.recv(), timeout=2.5)
                except asyncio.TimeoutError:
                    if greeting_timestamps:
                        break
                    continue
                if isinstance(msg, bytes):
                    parsed = parse_frame(msg)
                    if parsed["type"] == "audio":
                        greeting_timestamps.append(time.time())
                        greeting_sizes.append(parsed["length"])
        except websockets.ConnectionClosed:
            pass

        greeting_analysis = _analyze_streaming(greeting_timestamps, greeting_sizes, "Greeting")
        logger.info(f"Greeting: {len(greeting_timestamps)} chunks received")

        # Now send speech and measure response streaming
        client._reset_counters()

        wav_path = os.path.join(AUDIO_DIR, "user_greeting.wav")
        if os.path.exists(wav_path):
            speech, sr = load_wav_file(wav_path)
            logger.info(f"Loaded {wav_path}: {len(speech)/(sr*2):.1f}s")
        else:
            speech = generate_speech_like_audio(duration_sec=3.0)
        silence = generate_silence(duration_sec=1.5)

        logger.info("Sending speech...")
        await client.send_audio_stream(speech, realtime=True)
        await client.send_audio_stream(silence, realtime=True)
        speech_done = time.time()

        # Receive response with per-chunk timestamps
        response_timestamps = []
        response_sizes = []
        first_audio_at = None
        deadline = time.time() + 30.0

        try:
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(client.ws.recv(), timeout=2.5)
                except asyncio.TimeoutError:
                    if response_timestamps:
                        break
                    continue
                if isinstance(msg, bytes):
                    parsed = parse_frame(msg)
                    if parsed["type"] == "audio":
                        now = time.time()
                        response_timestamps.append(now)
                        response_sizes.append(parsed["length"])
                        if first_audio_at is None:
                            first_audio_at = now
        except websockets.ConnectionClosed:
            pass

        response_analysis = _analyze_streaming(response_timestamps, response_sizes, "Response")

        elapsed = time.time() - start
        ttfb = (first_audio_at - speech_done) if first_audio_at else None

        # Determine pass/fail based on streaming quality
        passed = True
        issues = []

        if response_analysis:
            if response_analysis["burst_ratio"] > 0.4:
                issues.append(f"HIGH BURST RATIO: {response_analysis['burst_ratio']:.0%} of chunks arrived in bursts")
                passed = False
            if response_analysis["max_gap_ms"] > 500:
                issues.append(f"LARGE GAP: {response_analysis['max_gap_ms']:.0f}ms between chunks")
            if response_analysis["jitter_ms"] > 100:
                issues.append(f"HIGH JITTER: {response_analysis['jitter_ms']:.0f}ms std deviation")
            if response_analysis.get("streaming_ratio", 0) > 3.0:
                issues.append(f"AUDIO DUMPED: {response_analysis['streaming_ratio']:.1f}x faster than realtime")
                passed = False

        details = {
            "ttfb_sec": round(ttfb, 3) if ttfb else None,
            "greeting_analysis": greeting_analysis,
            "response_analysis": response_analysis,
            "response_chunks": len(response_timestamps),
            "issues": issues,
        }

        if issues:
            for issue in issues:
                logger.warning(f"  ⚠️  {issue}")
        else:
            logger.info("  ✅ Audio streaming is smooth — no burst/jitter issues detected")

        return TestResult(name="streaming", passed=passed, duration_sec=elapsed, details=details)

    except Exception as e:
        return TestResult(name="streaming", passed=False, error=str(e), duration_sec=time.time() - start)
    finally:
        await client.disconnect()


def _analyze_streaming(
    timestamps: list[float], sizes: list[int], label: str
) -> dict | None:
    """Analyze the streaming pattern of received audio chunks."""
    if len(timestamps) < 3:
        logger.warning(f"  {label}: Too few chunks ({len(timestamps)}) to analyze")
        return None

    # Inter-chunk intervals
    intervals_ms = [(timestamps[i+1] - timestamps[i]) * 1000 for i in range(len(timestamps) - 1)]

    mean_interval = sum(intervals_ms) / len(intervals_ms)
    jitter = (sum((x - mean_interval) ** 2 for x in intervals_ms) / len(intervals_ms)) ** 0.5
    min_interval = min(intervals_ms)
    max_interval = max(intervals_ms)

    # Total audio duration (assuming 24kHz 16-bit mono output)
    total_bytes = sum(sizes)
    audio_duration_sec = total_bytes / (24000 * 2)  # 24kHz, 16-bit

    # Delivery duration
    delivery_sec = timestamps[-1] - timestamps[0]
    streaming_ratio = (audio_duration_sec / delivery_sec) if delivery_sec > 0 else 0

    # Burst detection: how many chunks arrived within 5ms of each other
    burst_threshold_ms = 5.0
    burst_count = sum(1 for iv in intervals_ms if iv < burst_threshold_ms)
    burst_ratio = burst_count / len(intervals_ms) if intervals_ms else 0

    # Gap detection: intervals > 200ms
    gap_threshold_ms = 200.0
    gaps = [iv for iv in intervals_ms if iv > gap_threshold_ms]

    # Chunk size consistency
    mean_size = sum(sizes) / len(sizes)
    size_variance = (sum((s - mean_size) ** 2 for s in sizes) / len(sizes)) ** 0.5

    # Print detailed analysis
    logger.info(f"")
    logger.info(f"  ── {label} Streaming Analysis ──")
    logger.info(f"  Chunks:           {len(timestamps)}")
    logger.info(f"  Total audio:      {total_bytes} bytes ({audio_duration_sec:.2f}s @ 24kHz)")
    logger.info(f"  Delivery time:    {delivery_sec:.2f}s")
    logger.info(f"  Streaming ratio:  {streaming_ratio:.2f}x (1.0 = perfect realtime)")
    logger.info(f"  ──────────────────────────")
    logger.info(f"  Interval mean:    {mean_interval:.1f}ms")
    logger.info(f"  Interval min:     {min_interval:.1f}ms")
    logger.info(f"  Interval max:     {max_interval:.1f}ms")
    logger.info(f"  Jitter (σ):       {jitter:.1f}ms")
    logger.info(f"  ──────────────────────────")
    logger.info(f"  Burst chunks:     {burst_count}/{len(intervals_ms)} ({burst_ratio:.0%})")
    logger.info(f"  Gaps (>200ms):    {len(gaps)} (max: {max_interval:.0f}ms)")
    logger.info(f"  Chunk size mean:  {mean_size:.0f} bytes")
    logger.info(f"  Chunk size σ:     {size_variance:.0f} bytes")

    # Show first 20 intervals for visual inspection
    if len(intervals_ms) > 0:
        sample = intervals_ms[:30]
        bar_line = "  Timeline: "
        for iv in sample:
            if iv < 5:
                bar_line += "▏"   # burst (< 5ms)
            elif iv < 30:
                bar_line += "▎"   # fast
            elif iv < 80:
                bar_line += "▍"   # normal
            elif iv < 150:
                bar_line += "▌"   # slow
            elif iv < 300:
                bar_line += "▊"   # gap
            else:
                bar_line += "█"   # big gap
        bar_line += f"  ({len(sample)} of {len(intervals_ms)} intervals)"
        logger.info(bar_line)
        logger.info(f"  Legend: ▏<5ms ▎<30ms ▍<80ms ▌<150ms ▊<300ms █>300ms")

    return {
        "chunks": len(timestamps),
        "total_bytes": total_bytes,
        "audio_duration_sec": round(audio_duration_sec, 2),
        "delivery_sec": round(delivery_sec, 2),
        "streaming_ratio": round(streaming_ratio, 2),
        "mean_interval_ms": round(mean_interval, 1),
        "min_interval_ms": round(min_interval, 1),
        "max_gap_ms": round(max_interval, 1),
        "jitter_ms": round(jitter, 1),
        "burst_count": burst_count,
        "burst_ratio": round(burst_ratio, 2),
        "gap_count": len(gaps),
        "chunk_size_mean": round(mean_size, 0),
        "chunk_size_std": round(size_variance, 0),
    }


TESTS = {
    "greeting": test_greeting,
    "roundtrip": test_roundtrip,
    "bargein": test_bargein,
    "latency": test_latency,
    "streaming": test_streaming_quality,
}


async def run_tests(test_names: list[str], ws_url: str):
    """Run selected tests and print a summary."""
    results = []

    for name in test_names:
        if name not in TESTS:
            logger.error(f"Unknown test: {name}")
            continue

        logger.info("")
        logger.info(f"{'='*60}")
        logger.info(f"  TEST: {name.upper()}")
        logger.info(f"{'='*60}")

        client = RTVITestClient(ws_url=ws_url)
        result = await TESTS[name](client)
        results.append(result)

        # Brief pause between tests
        await asyncio.sleep(2.0)

    # ─── Summary ───
    print()
    print("=" * 60)
    print("  TEST RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        icon = "✅" if r.passed else "❌"
        err = f" — {r.error}" if r.error else ""
        print(f"  {icon} {r.name:<15} {r.duration_sec:6.1f}s{err}")
        if r.details:
            for k, v in r.details.items():
                if isinstance(v, list) and len(v) > 3:
                    v = f"[{len(v)} items]"
                print(f"     {k}: {v}")
    print("=" * 60)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"  {passed}/{total} tests passed")
    print("=" * 60)

    return all(r.passed for r in results)


def main():
    parser = argparse.ArgumentParser(description="RTVI Protocol Test Client for MiraVoiceAI")
    parser.add_argument(
        "--test",
        choices=list(TESTS.keys()) + ["all"],
        default="all",
        help="Which test to run (default: all)",
    )
    parser.add_argument(
        "--url",
        default=WS_URL,
        help=f"WebSocket URL (default: {WS_URL})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    test_names = list(TESTS.keys()) if args.test == "all" else [args.test]

    success = asyncio.run(run_tests(test_names, args.url))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
