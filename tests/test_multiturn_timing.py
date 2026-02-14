#!/usr/bin/env python3
"""
Multi-turn conversation timing test.

Simulates a REAL multi-turn voice conversation with idle gaps between turns,
verifying that Soniox stays connected and speech detection remains fast
across all turns — not just the first one.

This test was created because single-turn tests missed a critical bug:
the Soniox keepalive only ran before the first speech, so subsequent turns
after idle gaps required a ~300ms reconnect and lost initial audio.

Usage:
    # Against OSS deployment:
    python tests/test_multiturn_timing.py --url wss://mira-oss.inf7ks8.com/pipecat/ws

    # Against local Docker:
    python tests/test_multiturn_timing.py --url ws://localhost:7860/ws
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import wave
from typing import Optional, List

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
    print("WARNING: pipecat not installed — protobuf framing unavailable")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("multiturn_timing")

SAMPLE_RATE = 16000
CHUNK_MS = 20


def make_jwt(secret: str) -> str:
    if not pyjwt:
        return ""
    payload = {"id": "test-multiturn-user", "role": "user", "name": "MultiTurnTest"}
    return pyjwt.encode(payload, secret, algorithm="HS256")


def make_audio_frame(pcm_bytes: bytes) -> bytes:
    if frame_protos is None:
        return pcm_bytes
    frame = frame_protos.Frame()
    frame.audio.audio = pcm_bytes
    frame.audio.sample_rate = SAMPLE_RATE
    frame.audio.num_channels = 1
    return frame.SerializeToString()


def load_wav_file(path: str) -> bytes:
    with wave.open(path, 'rb') as wf:
        return wf.readframes(wf.getnframes())


def generate_silence(duration_sec: float) -> bytes:
    n = int(SAMPLE_RATE * duration_sec)
    return np.zeros(n, dtype=np.int16).tobytes()


async def send_audio(ws, pcm_bytes: bytes):
    """Send PCM audio in realtime chunks."""
    chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
    chunk_bytes = chunk_samples * 2
    offset = 0
    while offset < len(pcm_bytes):
        chunk = pcm_bytes[offset:offset + chunk_bytes]
        proto = make_audio_frame(chunk)
        await ws.send(proto)
        offset += chunk_bytes
        await asyncio.sleep(CHUNK_MS / 1000.0 * 0.9)


async def collect_response(ws, speech_start: float, timeout: float = 15.0) -> dict:
    """Collect STT transcript, bot text, and bot audio from the server."""
    result = {
        "first_transcript_ms": None,
        "first_audio_ms": None,
        "stt_text": "",
        "bot_text": "",
        "audio_chunks": 0,
    }

    first_transcript_at = None
    first_audio_at = None
    bot_text_parts = []

    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 2.0))
            except asyncio.TimeoutError:
                if bot_text_parts or result["audio_chunks"] > 0:
                    break
                continue

            now = time.monotonic()

            if isinstance(msg, str):
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
                        if is_final:
                            result["stt_text"] = text

                    elif msg_type == "bot_text":
                        bot_text_parts.append(data.get("text", ""))

                    elif msg_type == "bot_text_complete":
                        result["bot_text"] = data.get("text", "")
                        # Wait a bit for trailing audio
                        await asyncio.sleep(1.0)
                        break

                except json.JSONDecodeError:
                    pass

            elif isinstance(msg, bytes):
                result["audio_chunks"] += 1
                if not first_audio_at:
                    first_audio_at = now
                    result["first_audio_ms"] = (now - speech_start) * 1000

    except websockets.ConnectionClosed:
        logger.warning("WebSocket closed during receive")

    if not result["bot_text"] and bot_text_parts:
        result["bot_text"] = "".join(bot_text_parts)

    return result


async def run_multiturn_test(
    ws_url: str,
    jwt_secret: str = "",
    wav_file: Optional[str] = None,
) -> List[dict]:
    """
    Run a multi-turn conversation test:
      Turn 1: Connect, wait 2s, speak
      Turn 2: Wait 8s (simulating reading response), speak again
      Turn 3: Wait 12s (longer pause), speak again
      Turn 4: Wait 5s, speak again

    Returns list of per-turn timing results.
    """
    config = {
        "type": "config",
        "mode": "text_and_audio",
        "enable_greeting": False,
        "language": "en",
    }
    if jwt_secret:
        config["token"] = make_jwt(jwt_secret)

    # Load audio
    if wav_file and os.path.exists(wav_file):
        pcm_audio = load_wav_file(wav_file)
        logger.info(f"Using WAV: {wav_file} ({len(pcm_audio)/(SAMPLE_RATE*2):.1f}s)")
    else:
        # Generate speech-like synthetic audio
        n = int(SAMPLE_RATE * 2.5)
        t = np.arange(n) / SAMPLE_RATE
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
        signal = (
            0.5 * np.sin(2 * np.pi * 180 * t) +
            0.3 * np.sin(2 * np.pi * 360 * t) +
            0.15 * np.sin(2 * np.pi * 720 * t) +
            0.05 * np.sin(2 * np.pi * 1200 * t)
        ) * envelope * 16000
        pcm_audio = signal.astype(np.int16).tobytes()
        logger.info(f"Using synthetic speech (2.5s)")

    silence = generate_silence(1.5)

    # Define turns: (idle_before_sec, label)
    turns = [
        (2.0,  "Turn 1 (cold start, 2s idle)"),
        (8.0,  "Turn 2 (after response, 8s idle)"),
        (12.0, "Turn 3 (long pause, 12s idle)"),
        (5.0,  "Turn 4 (quick follow-up, 5s idle)"),
    ]

    results = []
    ws = None

    try:
        ws = await websockets.connect(
            ws_url,
            max_size=10 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=3,
        )

        # Send config
        await ws.send(json.dumps(config))
        logger.info("Connected, config sent")

        # Wait for session_id
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            if isinstance(msg, str):
                data = json.loads(msg)
                if data.get("type") == "session_id":
                    logger.info(f"Got session_id: {data.get('session_id', '')[:12]}...")
        except Exception:
            pass

        for i, (idle_sec, label) in enumerate(turns):
            logger.info(f"\n{'='*60}")
            logger.info(f"{label}")
            logger.info(f"{'='*60}")

            # === IDLE PHASE ===
            logger.info(f"  Waiting {idle_sec}s (simulating user idle)...")
            await asyncio.sleep(idle_sec)

            # === SPEECH PHASE ===
            speech_start = time.monotonic()
            logger.info(f"  Sending speech audio...")
            await send_audio(ws, pcm_audio)
            speech_sent_ms = (time.monotonic() - speech_start) * 1000
            logger.info(f"  Speech sent in {speech_sent_ms:.0f}ms")

            # Send trailing silence for VAD end-of-speech
            await send_audio(ws, silence)
            logger.info(f"  Trailing silence sent")

            # === COLLECT RESPONSE ===
            result = await collect_response(ws, speech_start)
            result["turn"] = i + 1
            result["label"] = label
            result["idle_sec"] = idle_sec
            results.append(result)

            # Log per-turn summary
            transcript_ms = f"{result['first_transcript_ms']:.0f}ms" if result['first_transcript_ms'] else "NONE"
            audio_ms = f"{result['first_audio_ms']:.0f}ms" if result['first_audio_ms'] else "NONE"
            logger.info(f"\n  --- {label} Results ---")
            logger.info(f"  1st STT transcript: {transcript_ms}")
            logger.info(f"  1st bot audio:      {audio_ms}")
            logger.info(f"  STT text:           '{result['stt_text'][:50]}'")
            logger.info(f"  Bot text:           '{result['bot_text'][:60]}'")
            logger.info(f"  Audio chunks:       {result['audio_chunks']}")

    except Exception as e:
        logger.error(f"Connection error: {e}")
        results.append({"turn": 0, "error": str(e)})
    finally:
        # Force-close the WebSocket to avoid hanging on the close handshake
        if ws:
            try:
                await asyncio.wait_for(ws.close(), timeout=3.0)
            except Exception:
                pass
            # Abort the underlying transport if still open
            if ws.transport and not ws.transport.is_closing():
                ws.transport.abort()
            logger.info("WebSocket closed")

    return results


async def main():
    parser = argparse.ArgumentParser(description="Multi-turn conversation timing test")
    parser.add_argument("--url", default="wss://mira-oss.inf7ks8.com/pipecat/ws",
                        help="WebSocket URL")
    parser.add_argument("--wav", default=None,
                        help="Path to WAV file (16kHz mono)")
    parser.add_argument("--secret", default=None,
                        help="WEBUI_SECRET_KEY for JWT auth")
    args = parser.parse_args()

    secret = args.secret or os.getenv("WEBUI_SECRET_KEY", "")
    if not secret:
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("WEBUI_SECRET_KEY="):
                        secret = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

    print("=" * 70)
    print("MIRA Multi-Turn Conversation Timing Test")
    print(f"Target: {args.url}")
    print(f"Auth: {'JWT' if secret else 'none'}")
    print(f"Audio: {args.wav or 'synthetic (2.5s)'}")
    print("=" * 70)
    print()
    print("This test simulates a real multi-turn conversation with idle gaps")
    print("between turns to verify Soniox stays connected across the session.")
    print()

    results = await run_multiturn_test(
        ws_url=args.url,
        jwt_secret=secret,
        wav_file=args.wav,
    )

    # === SUMMARY TABLE ===
    print("\n" + "=" * 70)
    print("MULTI-TURN SUMMARY")
    print("=" * 70)
    print(f"{'Turn':<6} {'Idle':<8} {'1st Transcript':<18} {'1st Audio':<15} {'STT Text':<25} {'Status'}")
    print("-" * 95)

    all_smooth = True
    transcript_times = []
    for r in results:
        if r.get("error"):
            print(f"{'ERR':<6} {'':<8} {'':<18} {'':<15} {'':<25} ❌ {r['error'][:30]}")
            all_smooth = False
            continue

        turn = r.get("turn", "?")
        idle = r.get("idle_sec", 0)
        transcript = f"{r['first_transcript_ms']:.0f}ms" if r.get('first_transcript_ms') else "NONE"
        audio = f"{r['first_audio_ms']:.0f}ms" if r.get('first_audio_ms') else "NONE"
        stt = r.get('stt_text', '')[:23] or "(none)"

        if not r.get('first_transcript_ms'):
            status = "❌ No transcript"
            all_smooth = False
        elif r['first_transcript_ms'] > 3000:
            status = "⚠️  Slow (>3s)"
            all_smooth = False
        else:
            status = "✅ Smooth"
            transcript_times.append(r['first_transcript_ms'])

        print(f"{turn:<6} {idle:<8.0f} {transcript:<18} {audio:<15} {stt:<25} {status}")

    print()

    # Consistency check
    if len(transcript_times) >= 2:
        t_min = min(transcript_times)
        t_max = max(transcript_times)
        spread = t_max - t_min
        print(f"Transcript time spread: {spread:.0f}ms (min={t_min:.0f}ms, max={t_max:.0f}ms)")
        if spread > 500:
            print(f"⚠️  Spread > 500ms — connection may be dropping between turns")
            all_smooth = False
        else:
            print(f"✅ Consistent timing across turns (spread < 500ms)")

    print()
    if all_smooth:
        print("✅ ALL TURNS SMOOTH — Soniox keepalive works across the full conversation!")
        print("   No reconnection delays between turns.")
    else:
        print("⚠️  Some turns had issues — check logs above.")
        print("   If later turns are slower, Soniox may be dropping between turns.")

    print("=" * 70)

    # Force exit to prevent hanging on async cleanup / lingering WebSocket tasks
    sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit:
        pass  # Clean exit from sys.exit(0) in main()
    finally:
        os._exit(0)  # Hard exit — kill any lingering threads/tasks
