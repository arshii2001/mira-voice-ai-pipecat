#!/usr/bin/env python3
"""
End-to-end speech detection timing test.

Connects to the Pipecat /ws endpoint and simulates the real user experience:
  1. Connect WebSocket → pipeline starts → Soniox connects eagerly
  2. Wait N seconds (simulating user reading the UI, getting ready)
  3. Send speech audio
  4. Measure time to first interim transcript and first audio response

This test requires a running Pipecat server (local or remote).

Usage:
    # Against local Docker:
    python tests/test_speech_timing.py --url ws://localhost:7860/ws

    # Against OSS deployment:
    python tests/test_speech_timing.py --url wss://mira-oss.inf7ks8.com/pipecat/ws

    # With different idle delays:
    python tests/test_speech_timing.py --url ws://localhost:7860/ws --delays 2,5,10,20,35
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

import numpy as np

# Add project root to path
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
    print("WARNING: pipecat not installed — protobuf framing unavailable, using raw PCM")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("speech_timing")

SAMPLE_RATE = 16000  # Input sample rate for STT
CHUNK_MS = 20        # 20ms chunks


def make_jwt(secret: str) -> str:
    """Create a JWT token for authentication."""
    if not pyjwt:
        return ""
    payload = {"id": "test-timing-user", "role": "user", "name": "TimingTest"}
    return pyjwt.encode(payload, secret, algorithm="HS256")


def generate_speech_pcm(text_hint: str = "hello", duration_sec: float = 2.0) -> bytes:
    """Generate speech-like audio (multi-frequency mix) as PCM16 @ 16kHz.

    This is synthetic — enough to trigger VAD but Soniox won't transcribe
    it as real words. For real transcription testing, use a WAV file.
    """
    n = int(SAMPLE_RATE * duration_sec)
    t = np.arange(n) / SAMPLE_RATE
    # Speech-like mix of fundamental + harmonics with amplitude modulation
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # ~3Hz modulation (syllable rate)
    signal = (
        0.5 * np.sin(2 * np.pi * 180 * t) +   # fundamental
        0.3 * np.sin(2 * np.pi * 360 * t) +   # 1st harmonic
        0.15 * np.sin(2 * np.pi * 720 * t) +  # 2nd harmonic
        0.05 * np.sin(2 * np.pi * 1200 * t)   # high freq
    ) * envelope * 16000
    return signal.astype(np.int16).tobytes()


def generate_silence(duration_sec: float = 1.5) -> bytes:
    """Generate silence as PCM16."""
    n = int(SAMPLE_RATE * duration_sec)
    return np.zeros(n, dtype=np.int16).tobytes()


def make_audio_frame(pcm_bytes: bytes) -> bytes:
    """Wrap raw PCM16 audio in a Pipecat protobuf Frame."""
    if frame_protos is None:
        return pcm_bytes  # Fallback: raw PCM
    frame = frame_protos.Frame()
    frame.audio.audio = pcm_bytes
    frame.audio.sample_rate = SAMPLE_RATE
    frame.audio.num_channels = 1
    return frame.SerializeToString()


def load_wav_file(path: str) -> tuple:
    """Load a WAV file and return (pcm_bytes, sample_rate)."""
    import wave
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return pcm, sr


async def run_timing_test(
    ws_url: str,
    idle_delay_sec: float,
    jwt_secret: str = "",
    use_wav: str = None,
) -> dict:
    """
    Run a single timing test:
      1. Connect to /ws
      2. Wait idle_delay_sec seconds
      3. Send speech audio
      4. Measure response times

    Returns dict with timing measurements.
    """
    result = {
        "idle_delay_sec": idle_delay_sec,
        "connected": False,
        "first_transcript_ms": None,
        "first_audio_ms": None,
        "bot_text": "",
        "stt_text": "",
        "total_audio_chunks": 0,
        "error": None,
    }

    # Build config message
    config = {
        "type": "config",
        "mode": "text_and_audio",
        "enable_greeting": False,
        "language": "en",
    }
    if jwt_secret:
        config["token"] = make_jwt(jwt_secret)

    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST: idle_delay={idle_delay_sec}s")
        logger.info(f"{'='*60}")

        async with websockets.connect(
            ws_url,
            max_size=10 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        ) as ws:
            result["connected"] = True

            # Send config
            await ws.send(json.dumps(config))
            logger.info(f"  t=0.0s: Connected, config sent")

            # Wait for session_id
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                if isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("type") == "session_id":
                        logger.info(f"  t=0.0s: Got session_id")
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"  No session_id: {e}")

            # === IDLE PHASE ===
            logger.info(f"  Waiting {idle_delay_sec}s (simulating user idle)...")
            await asyncio.sleep(idle_delay_sec)
            logger.info(f"  t={idle_delay_sec}s: Idle phase complete, sending speech")

            # === SPEECH PHASE ===
            # Load or generate audio
            if use_wav and os.path.exists(use_wav):
                pcm_audio, sr = load_wav_file(use_wav)
                logger.info(f"  Using WAV: {use_wav} ({len(pcm_audio)/(sr*2):.1f}s)")
            else:
                pcm_audio = generate_speech_pcm(duration_sec=2.5)
                logger.info(f"  Using synthetic speech ({len(pcm_audio)/(SAMPLE_RATE*2):.1f}s)")

            silence = generate_silence(duration_sec=1.5)

            # Send speech in realtime chunks
            chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
            chunk_bytes = chunk_samples * 2
            speech_start = time.monotonic()

            offset = 0
            while offset < len(pcm_audio):
                chunk = pcm_audio[offset:offset + chunk_bytes]
                proto = make_audio_frame(chunk)
                await ws.send(proto)
                offset += chunk_bytes
                await asyncio.sleep(CHUNK_MS / 1000.0 * 0.9)  # Slightly faster than realtime

            speech_sent_ms = (time.monotonic() - speech_start) * 1000
            logger.info(f"  Speech audio sent in {speech_sent_ms:.0f}ms")

            # Send trailing silence for VAD end-of-speech detection
            offset = 0
            while offset < len(silence):
                chunk = silence[offset:offset + chunk_bytes]
                proto = make_audio_frame(chunk)
                await ws.send(proto)
                offset += chunk_bytes
                await asyncio.sleep(CHUNK_MS / 1000.0 * 0.9)

            logger.info(f"  Trailing silence sent")

            # === RECEIVE PHASE ===
            # Collect responses for up to 15 seconds
            first_transcript_at = None
            first_audio_at = None
            bot_text_parts = []
            stt_text = ""
            audio_chunks = 0

            try:
                deadline = time.monotonic() + 15.0
                while time.monotonic() < deadline:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 2.0))
                    except asyncio.TimeoutError:
                        # If we have a response, we're done
                        if bot_text_parts or audio_chunks > 0:
                            break
                        continue

                    now = time.monotonic()

                    if isinstance(msg, str):
                        # JSON message
                        try:
                            data = json.loads(msg)
                            msg_type = data.get("type", "")

                            if msg_type == "user_transcript":
                                text = data.get("text", "")
                                is_final = data.get("final", False)
                                since = (now - speech_start) * 1000
                                if not first_transcript_at:
                                    first_transcript_at = now
                                    result["first_transcript_ms"] = since
                                    logger.info(
                                        f"  ✅ First STT transcript at +{since:.0f}ms: "
                                        f"'{text[:60]}' (final={is_final})"
                                    )
                                if is_final:
                                    stt_text = text

                            elif msg_type == "bot_text":
                                bot_text_parts.append(data.get("text", ""))

                            elif msg_type == "bot_text_complete":
                                result["bot_text"] = data.get("text", "")
                                logger.info(
                                    f"  Bot response complete at "
                                    f"+{(now - speech_start)*1000:.0f}ms: "
                                    f"'{result['bot_text'][:80]}...'"
                                )
                                # Wait a bit more for any trailing audio
                                await asyncio.sleep(1.0)
                                break

                        except json.JSONDecodeError:
                            pass

                    elif isinstance(msg, bytes):
                        # Binary: protobuf audio frame
                        audio_chunks += 1
                        if not first_audio_at:
                            first_audio_at = now
                            since = (now - speech_start) * 1000
                            result["first_audio_ms"] = since
                            logger.info(
                                f"  ✅ First bot audio at +{since:.0f}ms "
                                f"({len(msg)} bytes)"
                            )

            except websockets.ConnectionClosed:
                logger.warning("  WebSocket closed during receive")

            result["stt_text"] = stt_text
            result["total_audio_chunks"] = audio_chunks
            if not result["bot_text"] and bot_text_parts:
                result["bot_text"] = "".join(bot_text_parts)

            # Summary
            logger.info(f"\n  --- Results (idle={idle_delay_sec}s) ---")
            if result["first_transcript_ms"]:
                logger.info(f"  First STT transcript:  {result['first_transcript_ms']:.0f}ms after speech start")
            else:
                logger.info(f"  First STT transcript:  ❌ NONE RECEIVED")
            if result["first_audio_ms"]:
                logger.info(f"  First bot audio:       {result['first_audio_ms']:.0f}ms after speech start")
            else:
                logger.info(f"  First bot audio:       ❌ NONE RECEIVED")
            logger.info(f"  STT text:              '{stt_text[:60]}'")
            logger.info(f"  Bot text:              '{result['bot_text'][:60]}'")
            logger.info(f"  Audio chunks received: {audio_chunks}")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"  ERROR: {e}")

    return result


async def main():
    parser = argparse.ArgumentParser(description="Speech detection timing test")
    parser.add_argument("--url", default="wss://mira-oss.inf7ks8.com/pipecat/ws",
                        help="WebSocket URL")
    parser.add_argument("--delays", default="2,5,10,20",
                        help="Comma-separated idle delays in seconds")
    parser.add_argument("--wav", default=None,
                        help="Path to WAV file (16kHz mono) instead of synthetic audio")
    parser.add_argument("--secret", default=None,
                        help="WEBUI_SECRET_KEY for JWT auth")
    args = parser.parse_args()

    # Try to get secret from env or .env file
    secret = args.secret or os.getenv("WEBUI_SECRET_KEY", "")
    if not secret:
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("WEBUI_SECRET_KEY="):
                        secret = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

    delays = [float(d) for d in args.delays.split(",")]

    print("=" * 60)
    print("MIRA Speech Detection Timing Test")
    print(f"Target: {args.url}")
    print(f"Idle delays: {delays}")
    print(f"Auth: {'JWT' if secret else 'none'}")
    if args.wav:
        print(f"Audio: {args.wav}")
    else:
        print(f"Audio: synthetic speech (2.5s)")
    print("=" * 60)

    results = []
    for delay in delays:
        result = await run_timing_test(
            ws_url=args.url,
            idle_delay_sec=delay,
            jwt_secret=secret,
            use_wav=args.wav,
        )
        results.append(result)
        # Brief pause between tests
        await asyncio.sleep(2.0)

    # === SUMMARY TABLE ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Idle(s)':<10} {'1st Transcript':<18} {'1st Audio':<15} {'STT Text':<30} {'Status'}")
    print("-" * 90)

    all_smooth = True
    for r in results:
        transcript = f"{r['first_transcript_ms']:.0f}ms" if r['first_transcript_ms'] else "NONE"
        audio = f"{r['first_audio_ms']:.0f}ms" if r['first_audio_ms'] else "NONE"
        stt = r['stt_text'][:28] if r['stt_text'] else "(none)"
        error = r.get('error')

        if error:
            status = f"❌ {error[:20]}"
            all_smooth = False
        elif not r['first_transcript_ms']:
            status = "❌ No transcript"
            all_smooth = False
        elif r['first_transcript_ms'] > 5000:
            status = "⚠️  Slow (>5s)"
            all_smooth = False
        else:
            status = "✅ Smooth"

        print(f"{r['idle_delay_sec']:<10.0f} {transcript:<18} {audio:<15} {stt:<30} {status}")

    print()
    if all_smooth:
        print("✅ ALL TESTS SMOOTH — idle keepalive is working correctly!")
        print("   No word loss regardless of idle duration.")
    else:
        print("⚠️  Some tests had issues — check logs above for details.")
        print("   If transcript is NONE or slow for longer idle delays,")
        print("   the Soniox connection may be dropping despite keepalive.")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
