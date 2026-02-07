#!/usr/bin/env python3
"""
Comprehensive tests for text_only / text_and_audio mode support.

Tests both the /config endpoint and the /ws pipeline behaviour in each mode.

Usage (inside Docker):
    python tests/test_text_mode.py                      # Run all tests
    python tests/test_text_mode.py --test config        # One test
    python tests/test_text_mode.py --verbose            # Debug logging

Tests:
  1. config         — /config advertises supported_modes
  2. default_mode   — No config message → defaults to text_and_audio (audio received)
  3. text_and_audio — Explicit text_and_audio → audio + bot_text JSON received
  4. text_only      — Explicit text_only → bot_text JSON only, zero audio
  5. text_only_greeting — text_only sends greeting as bot_text_complete JSON
  6. invalid_mode   — Invalid mode value falls back to text_and_audio
  7. roundtrip_text — text_only full round-trip: send speech → get LLM text back
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
import wave
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)

try:
    import aiohttp
except ImportError:
    print("pip install aiohttp")
    sys.exit(1)

import pipecat.frames.protobufs.frames_pb2 as frame_protos

# ─────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────
WS_URL = os.getenv("PIPECAT_WS_URL", "ws://mira-voice:7860/ws")
HTTP_URL = os.getenv("PIPECAT_HTTP_URL", "http://mira-voice:7860")
AUDIO_DIR = os.getenv("AUDIO_DIR", "/app/tests/test_audio")
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("text-mode-test")


# ─────────────────────────────────────────────────
# Protobuf helpers (same as test_rtvi.py)
# ─────────────────────────────────────────────────
def make_audio_frame(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    frame = frame_protos.Frame()
    frame.audio.audio = pcm_bytes
    frame.audio.sample_rate = sample_rate
    frame.audio.num_channels = 1
    return frame.SerializeToString()


def parse_frame(data: bytes) -> dict:
    try:
        proto = frame_protos.Frame.FromString(data)
        which = proto.WhichOneof("frame")
        if which == "audio":
            return {"type": "audio", "length": len(proto.audio.audio)}
        elif which == "text":
            return {"type": "text", "text": proto.text.text}
        elif which == "transcription":
            return {"type": "transcription", "text": proto.transcription.text}
        elif which == "message":
            try:
                return {"type": "message", "data": json.loads(proto.message.data)}
            except json.JSONDecodeError:
                return {"type": "message", "data": proto.message.data}
        return {"type": "unknown", "which": which}
    except Exception as e:
        return {"type": "parse_error", "error": str(e)}


# ─────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────
def generate_speech_like_audio(duration_sec: float = 3.0) -> bytes:
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
    f0 = 120.0
    signal = np.zeros_like(t, dtype=np.float64)
    for harmonic in range(1, 8):
        signal += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    signal *= envelope
    signal = signal / np.max(np.abs(signal))
    return (0.7 * 32767 * signal).astype(np.int16).tobytes()


def generate_silence(duration_sec: float = 1.5) -> bytes:
    return np.zeros(int(SAMPLE_RATE * duration_sec), dtype=np.int16).tobytes()


def load_wav_file(path: str) -> tuple:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        data = wf.readframes(wf.getnframes())
        if wf.getnchannels() == 2:
            arr = np.frombuffer(data, dtype=np.int16).reshape(-1, 2).mean(axis=1).astype(np.int16)
            data = arr.tobytes()
        return data, sr


# ─────────────────────────────────────────────────
# Result tracking
# ─────────────────────────────────────────────────
@dataclass
class TestResult:
    name: str
    passed: bool
    duration_sec: float = 0.0
    details: dict = field(default_factory=dict)
    error: Optional[str] = None


# ─────────────────────────────────────────────────
# Mode-aware test client
# ─────────────────────────────────────────────────
class ModeTestClient:
    """WebSocket client that can send a config message with mode."""

    def __init__(self, ws_url: str = WS_URL):
        self.ws_url = ws_url
        self.ws = None
        # Counters
        self.audio_chunks = 0
        self.audio_bytes = 0
        self.bot_text_chunks = []       # {"type": "bot_text", ...}
        self.bot_text_completes = []    # {"type": "bot_text_complete", ...}
        self.proto_texts = []           # Protobuf TextFrame
        self.proto_transcriptions = []  # Protobuf TranscriptionFrame
        self.json_messages = []         # All JSON messages received
        self.proto_messages = []        # All protobuf messages received

    async def connect(self, timeout: float = 10.0):
        logger.info(f"Connecting to {self.ws_url}")
        self.ws = await asyncio.wait_for(
            websockets.connect(self.ws_url, max_size=10 * 1024 * 1024),
            timeout=timeout,
        )
        logger.info("Connected")

    async def send_config(self, mode: str = None, system_prompt: str = None):
        """Send a config message to set mode."""
        config = {"type": "config"}
        if mode is not None:
            config["mode"] = mode
        if system_prompt is not None:
            config["system_prompt"] = system_prompt
        logger.info(f"Sending config: {json.dumps(config)}")
        await self.ws.send(json.dumps(config))

    async def disconnect(self):
        if self.ws:
            await self.ws.close()
            self.ws = None

    def reset(self):
        self.audio_chunks = 0
        self.audio_bytes = 0
        self.bot_text_chunks = []
        self.bot_text_completes = []
        self.proto_texts = []
        self.proto_transcriptions = []
        self.json_messages = []
        self.proto_messages = []

    async def receive(self, duration: float = 15.0):
        """Receive all frames for `duration` seconds."""
        deadline = time.time() + duration
        try:
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(self.ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue

                if isinstance(msg, bytes):
                    parsed = parse_frame(msg)
                    self.proto_messages.append(parsed)
                    if parsed["type"] == "audio":
                        self.audio_chunks += 1
                        self.audio_bytes += parsed["length"]
                    elif parsed["type"] == "text":
                        self.proto_texts.append(parsed["text"])
                    elif parsed["type"] == "transcription":
                        self.proto_transcriptions.append(parsed["text"])
                else:
                    # JSON text message
                    try:
                        data = json.loads(msg)
                        self.json_messages.append(data)
                        msg_type = data.get("type")
                        if msg_type == "bot_text":
                            self.bot_text_chunks.append(data)
                            logger.debug(f"  bot_text: '{data.get('text', '')}'")
                        elif msg_type == "bot_text_complete":
                            self.bot_text_completes.append(data)
                            logger.info(f"  bot_text_complete: '{data.get('text', '')[:100]}'")
                        else:
                            logger.debug(f"  JSON: {data}")
                    except json.JSONDecodeError:
                        pass
        except websockets.ConnectionClosed as e:
            logger.warning(f"Connection closed: {e}")

    async def send_audio_stream(self, pcm_bytes: bytes, realtime: bool = True):
        chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
        chunk_bytes = chunk_samples * 2
        offset = 0
        while offset < len(pcm_bytes):
            chunk = pcm_bytes[offset: offset + chunk_bytes]
            await self.ws.send(make_audio_frame(chunk))
            offset += chunk_bytes
            if realtime:
                await asyncio.sleep(CHUNK_DURATION_MS / 1000.0 * 0.8)


# ─────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────

async def test_config() -> TestResult:
    """Test 1: /config endpoint advertises supported_modes."""
    start = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{HTTP_URL}/config") as resp:
                assert resp.status == 200, f"Expected 200, got {resp.status}"
                data = await resp.json()

        modes = data.get("supported_modes")
        assert modes is not None, "supported_modes missing from /config"
        assert "text_and_audio" in modes, "text_and_audio not in supported_modes"
        assert "text_only" in modes, "text_only not in supported_modes"

        logger.info(f"✅ /config has supported_modes: {modes}")
        return TestResult(
            name="config",
            passed=True,
            duration_sec=time.time() - start,
            details={"supported_modes": modes, "full_config": data},
        )
    except Exception as e:
        return TestResult(name="config", passed=False, error=str(e), duration_sec=time.time() - start)


async def test_default_mode() -> TestResult:
    """Test 2: No config message → defaults to text_and_audio → receives audio."""
    start = time.time()
    client = ModeTestClient()
    try:
        await client.connect()
        # Do NOT send any config message — should default to text_and_audio
        logger.info("No config sent — waiting for greeting audio...")
        await client.receive(duration=15.0)

        got_audio = client.audio_chunks > 0
        logger.info(
            f"{'✅' if got_audio else '❌'} Default mode: "
            f"audio_chunks={client.audio_chunks}, audio_bytes={client.audio_bytes}"
        )

        return TestResult(
            name="default_mode",
            passed=got_audio,
            duration_sec=time.time() - start,
            details={
                "audio_chunks": client.audio_chunks,
                "audio_bytes": client.audio_bytes,
                "bot_text_chunks": len(client.bot_text_chunks),
                "bot_text_completes": len(client.bot_text_completes),
            },
            error=None if got_audio else "No audio received with default mode",
        )
    except Exception as e:
        return TestResult(name="default_mode", passed=False, error=str(e), duration_sec=time.time() - start)
    finally:
        await client.disconnect()


async def test_text_and_audio_mode() -> TestResult:
    """Test 3: Explicit text_and_audio → receives BOTH audio AND bot_text JSON."""
    start = time.time()
    client = ModeTestClient()
    try:
        await client.connect()
        await client.send_config(mode="text_and_audio")

        logger.info("Waiting for greeting (audio + text)...")
        await client.receive(duration=15.0)

        got_audio = client.audio_chunks > 0
        got_text = len(client.bot_text_completes) > 0

        logger.info(
            f"{'✅' if (got_audio and got_text) else '❌'} text_and_audio mode: "
            f"audio_chunks={client.audio_chunks}, bot_text_completes={len(client.bot_text_completes)}"
        )

        # In text_and_audio, we expect BOTH audio and text
        passed = got_audio and got_text

        return TestResult(
            name="text_and_audio",
            passed=passed,
            duration_sec=time.time() - start,
            details={
                "audio_chunks": client.audio_chunks,
                "audio_bytes": client.audio_bytes,
                "bot_text_completes": [m.get("text", "") for m in client.bot_text_completes],
                "bot_text_chunks_count": len(client.bot_text_chunks),
            },
            error=None if passed else f"Expected audio+text. Audio={got_audio}, Text={got_text}",
        )
    except Exception as e:
        return TestResult(name="text_and_audio", passed=False, error=str(e), duration_sec=time.time() - start)
    finally:
        await client.disconnect()


async def test_text_only_mode() -> TestResult:
    """Test 4: Explicit text_only → bot_text JSON only, ZERO audio frames."""
    start = time.time()
    client = ModeTestClient()
    try:
        await client.connect()
        await client.send_config(mode="text_only")

        logger.info("Waiting for greeting (text only, no audio expected)...")
        await client.receive(duration=12.0)

        got_audio = client.audio_chunks > 0
        got_text = len(client.bot_text_completes) > 0

        # In text_only, we expect text but NO audio
        passed = got_text and not got_audio

        logger.info(
            f"{'✅' if passed else '❌'} text_only mode: "
            f"audio_chunks={client.audio_chunks} (expect 0), "
            f"bot_text_completes={len(client.bot_text_completes)} (expect >0)"
        )

        return TestResult(
            name="text_only",
            passed=passed,
            duration_sec=time.time() - start,
            details={
                "audio_chunks": client.audio_chunks,
                "audio_bytes": client.audio_bytes,
                "bot_text_completes": [m.get("text", "") for m in client.bot_text_completes],
            },
            error=None if passed else f"Audio={got_audio} (want False), Text={got_text} (want True)",
        )
    except Exception as e:
        return TestResult(name="text_only", passed=False, error=str(e), duration_sec=time.time() - start)
    finally:
        await client.disconnect()


async def test_text_only_greeting() -> TestResult:
    """Test 5: In text_only mode, greeting arrives as bot_text_complete JSON."""
    start = time.time()
    client = ModeTestClient()
    try:
        await client.connect()
        await client.send_config(mode="text_only")

        logger.info("Waiting for greeting text...")
        await client.receive(duration=12.0)

        # Should have at least one bot_text_complete with the greeting
        greeting_texts = [m.get("text", "") for m in client.bot_text_completes]
        has_greeting = any("Mira" in t or "Namaste" in t or "study" in t for t in greeting_texts)

        passed = has_greeting and client.audio_chunks == 0

        logger.info(
            f"{'✅' if passed else '❌'} text_only greeting: "
            f"greeting_found={has_greeting}, audio={client.audio_chunks}"
        )
        if greeting_texts:
            logger.info(f"  Greeting text: '{greeting_texts[0][:120]}'")

        return TestResult(
            name="text_only_greeting",
            passed=passed,
            duration_sec=time.time() - start,
            details={
                "greeting_texts": greeting_texts,
                "audio_chunks": client.audio_chunks,
            },
            error=None if passed else f"Greeting found={has_greeting}, audio_chunks={client.audio_chunks}",
        )
    except Exception as e:
        return TestResult(name="text_only_greeting", passed=False, error=str(e), duration_sec=time.time() - start)
    finally:
        await client.disconnect()


async def test_invalid_mode() -> TestResult:
    """Test 6: Invalid mode value → server falls back to text_and_audio (gets audio)."""
    start = time.time()
    client = ModeTestClient()
    try:
        await client.connect()
        await client.send_config(mode="banana_mode")

        logger.info("Sent invalid mode 'banana_mode' — expecting fallback to text_and_audio...")
        await client.receive(duration=15.0)

        got_audio = client.audio_chunks > 0

        logger.info(
            f"{'✅' if got_audio else '❌'} Invalid mode fallback: "
            f"audio_chunks={client.audio_chunks} (expect >0 for text_and_audio fallback)"
        )

        return TestResult(
            name="invalid_mode",
            passed=got_audio,
            duration_sec=time.time() - start,
            details={
                "audio_chunks": client.audio_chunks,
                "audio_bytes": client.audio_bytes,
                "bot_text_completes": len(client.bot_text_completes),
            },
            error=None if got_audio else "No audio — invalid mode did not fallback to text_and_audio",
        )
    except Exception as e:
        return TestResult(name="invalid_mode", passed=False, error=str(e), duration_sec=time.time() - start)
    finally:
        await client.disconnect()


async def test_roundtrip_text_only() -> TestResult:
    """Test 7: text_only full round-trip — send speech audio → get LLM text response, zero audio."""
    start = time.time()
    client = ModeTestClient()
    try:
        await client.connect()
        await client.send_config(
            mode="text_only",
            system_prompt="You are a helpful assistant. Reply in one short sentence.",
        )

        # Wait for greeting text
        logger.info("Waiting for greeting...")
        await client.receive(duration=10.0)
        greeting_completes = len(client.bot_text_completes)
        greeting_audio = client.audio_chunks
        logger.info(f"Greeting: {greeting_completes} text completes, {greeting_audio} audio chunks")

        # Reset for round-trip
        client.reset()

        # Load or generate speech audio
        wav_path = os.path.join(AUDIO_DIR, "user_greeting.wav")
        if os.path.exists(wav_path):
            speech, sr = load_wav_file(wav_path)
            logger.info(f"Loaded {wav_path}: {len(speech) / (sr * 2):.1f}s")
        else:
            speech = generate_speech_like_audio(duration_sec=3.0)
            logger.warning("Using synthetic audio (real speech WAV not found)")

        silence = generate_silence(duration_sec=1.5)

        # Send speech
        logger.info("Sending speech audio...")
        await client.send_audio_stream(speech, realtime=True)
        await client.send_audio_stream(silence, realtime=True)

        # Wait for LLM text response
        logger.info("Waiting for LLM text response (no audio expected)...")
        await client.receive(duration=30.0)

        got_text_response = len(client.bot_text_completes) > 0
        got_text_chunks = len(client.bot_text_chunks) > 0
        got_audio = client.audio_chunks > 0

        passed = got_text_response and not got_audio

        response_texts = [m.get("text", "") for m in client.bot_text_completes]

        logger.info(
            f"{'✅' if passed else '❌'} text_only round-trip: "
            f"text_completes={len(client.bot_text_completes)}, "
            f"text_chunks={len(client.bot_text_chunks)}, "
            f"audio_chunks={client.audio_chunks} (expect 0)"
        )
        if response_texts:
            logger.info(f"  Response: '{response_texts[0][:200]}'")

        return TestResult(
            name="roundtrip_text",
            passed=passed,
            duration_sec=time.time() - start,
            details={
                "bot_text_completes": response_texts,
                "bot_text_chunk_count": len(client.bot_text_chunks),
                "audio_chunks": client.audio_chunks,
                "audio_bytes": client.audio_bytes,
                "streaming_chunks": [m.get("text", "") for m in client.bot_text_chunks[:20]],
            },
            error=None if passed else f"Text={got_text_response}, Audio={got_audio} (want False)",
        )
    except Exception as e:
        return TestResult(name="roundtrip_text", passed=False, error=str(e), duration_sec=time.time() - start)
    finally:
        await client.disconnect()


async def test_roundtrip_text_and_audio() -> TestResult:
    """Test 8: text_and_audio full round-trip — send speech → get LLM text + TTS audio."""
    start = time.time()
    client = ModeTestClient()
    try:
        await client.connect()
        await client.send_config(
            mode="text_and_audio",
            system_prompt="You are a helpful assistant. Reply in one short sentence.",
        )

        # Wait for greeting
        logger.info("Waiting for greeting...")
        await client.receive(duration=15.0)
        greeting_audio = client.audio_chunks
        greeting_text = len(client.bot_text_completes)
        logger.info(f"Greeting: {greeting_audio} audio chunks, {greeting_text} text completes")

        # Reset for round-trip
        client.reset()

        # Load or generate speech
        wav_path = os.path.join(AUDIO_DIR, "user_greeting.wav")
        if os.path.exists(wav_path):
            speech, sr = load_wav_file(wav_path)
            logger.info(f"Loaded {wav_path}: {len(speech) / (sr * 2):.1f}s")
        else:
            speech = generate_speech_like_audio(duration_sec=3.0)
            logger.warning("Using synthetic audio")

        silence = generate_silence(duration_sec=1.5)

        logger.info("Sending speech audio...")
        await client.send_audio_stream(speech, realtime=True)
        await client.send_audio_stream(silence, realtime=True)

        logger.info("Waiting for LLM text + TTS audio response...")
        await client.receive(duration=30.0)

        got_text = len(client.bot_text_completes) > 0
        got_audio = client.audio_chunks > 0

        # In text_and_audio mode, we expect BOTH
        passed = got_text and got_audio

        response_texts = [m.get("text", "") for m in client.bot_text_completes]

        logger.info(
            f"{'✅' if passed else '❌'} text_and_audio round-trip: "
            f"text_completes={len(client.bot_text_completes)}, "
            f"text_chunks={len(client.bot_text_chunks)}, "
            f"audio_chunks={client.audio_chunks}, audio_bytes={client.audio_bytes}"
        )
        if response_texts:
            logger.info(f"  Response: '{response_texts[0][:200]}'")

        return TestResult(
            name="roundtrip_audio",
            passed=passed,
            duration_sec=time.time() - start,
            details={
                "bot_text_completes": response_texts,
                "bot_text_chunk_count": len(client.bot_text_chunks),
                "audio_chunks": client.audio_chunks,
                "audio_bytes": client.audio_bytes,
            },
            error=None if passed else f"Text={got_text}, Audio={got_audio} (want both True)",
        )
    except Exception as e:
        return TestResult(name="roundtrip_audio", passed=False, error=str(e), duration_sec=time.time() - start)
    finally:
        await client.disconnect()


# ─────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────
TESTS = {
    "config": test_config,
    "default_mode": test_default_mode,
    "text_and_audio": test_text_and_audio_mode,
    "text_only": test_text_only_mode,
    "text_only_greeting": test_text_only_greeting,
    "invalid_mode": test_invalid_mode,
    "roundtrip_text": test_roundtrip_text_only,
    "roundtrip_audio": test_roundtrip_text_and_audio,
}


async def run_tests(test_names: list, ws_url: str, http_url: str):
    global WS_URL, HTTP_URL
    WS_URL = ws_url
    HTTP_URL = http_url

    results = []
    for name in test_names:
        if name not in TESTS:
            logger.error(f"Unknown test: {name}")
            continue

        logger.info("")
        logger.info(f"{'=' * 60}")
        logger.info(f"  TEST: {name.upper()}")
        logger.info(f"{'=' * 60}")

        result = await TESTS[name]()
        results.append(result)

        # Pause between tests to avoid server overload
        await asyncio.sleep(2.0)

    # ─── Summary ───
    print()
    print("=" * 60)
    print("  TEXT MODE TEST RESULTS")
    print("=" * 60)
    for r in results:
        icon = "✅" if r.passed else "❌"
        err = f" — {r.error}" if r.error else ""
        print(f"  {icon} {r.name:<20} {r.duration_sec:6.1f}s{err}")
        if r.details:
            for k, v in r.details.items():
                if k == "full_config":
                    continue  # Skip verbose config
                if isinstance(v, list) and len(v) > 3:
                    v = f"[{len(v)} items]"
                elif isinstance(v, str) and len(v) > 100:
                    v = v[:100] + "..."
                print(f"     {k}: {v}")
    print("=" * 60)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"  {passed}/{total} tests passed")
    print("=" * 60)

    return all(r.passed for r in results)


def main():
    parser = argparse.ArgumentParser(description="Text Mode Tests for MiraVoiceAI")
    parser.add_argument(
        "--test",
        choices=list(TESTS.keys()) + ["all"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument(
        "--ws-url", default=WS_URL, help=f"WebSocket URL (default: {WS_URL})"
    )
    parser.add_argument(
        "--http-url", default=HTTP_URL, help=f"HTTP URL (default: {HTTP_URL})"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    test_names = list(TESTS.keys()) if args.test == "all" else [args.test]
    success = asyncio.run(run_tests(test_names, args.ws_url, args.http_url))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
