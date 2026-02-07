#!/usr/bin/env python3
"""
MIRA Voice AI — Tutor Mode Example Client

Demonstrates a single-user voice conversation with Mira using both
text_and_audio and text_only modes.

This example shows:
  1. Checking server health and config
  2. Connecting in text_and_audio mode (audio + text)
  3. Connecting in text_only mode (text only, no TTS audio)
  4. Sending mic audio (PCM via protobuf)
  5. Receiving bot text (streamed JSON) + audio (protobuf)
  6. Switching mode at runtime (text_and_audio ↔ text_only)

Prerequisites:
  pip install websockets aiohttp numpy

Usage:
  # Against a local server:
  python examples/tutor_example.py

  # Against a remote server:
  MIRA_HOST=192.168.1.100 python examples/tutor_example.py

  # Inside Docker (against the service name):
  MIRA_HOST=mira-voice python examples/tutor_example.py
"""

import asyncio
import json
import math
import os
import struct
import sys
import time

import numpy as np

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")

try:
    import aiohttp
except ImportError:
    sys.exit("pip install aiohttp")

# Try to import Pipecat protobuf — fall back to raw audio if unavailable
try:
    import pipecat.frames.protobufs.frames_pb2 as frame_protos
    HAS_PROTO = True
except ImportError:
    HAS_PROTO = False
    print("⚠  pipecat protobuf not available; audio send/receive will be raw bytes")

# ─────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────
HOST = os.getenv("MIRA_HOST", "localhost")
PORT = os.getenv("MIRA_PORT", "7860")
BASE_URL = f"http://{HOST}:{PORT}"
WS_URL = f"ws://{HOST}:{PORT}/ws"
SAMPLE_RATE = 16000


# ─────────────────────────────────────────────────
# Protobuf helpers
# ─────────────────────────────────────────────────
def make_audio_frame(pcm_bytes: bytes) -> bytes:
    """Wrap raw PCM16 audio in a Pipecat protobuf Frame."""
    if not HAS_PROTO:
        return pcm_bytes
    frame = frame_protos.Frame()
    frame.audio.audio = pcm_bytes
    frame.audio.sample_rate = SAMPLE_RATE
    frame.audio.num_channels = 1
    return frame.SerializeToString()


def parse_frame(data: bytes) -> dict:
    """Parse a binary protobuf message from the server."""
    if not HAS_PROTO:
        return {"type": "raw_audio", "length": len(data)}
    try:
        proto = frame_protos.Frame.FromString(data)
        which = proto.WhichOneof("frame")
        if which == "audio":
            return {"type": "audio", "length": len(proto.audio.audio)}
        elif which == "text":
            return {"type": "text", "text": proto.text.text}
        elif which == "transcription":
            return {
                "type": "transcription",
                "text": proto.transcription.text,
                "user_id": getattr(proto.transcription, "user_id", ""),
            }
        return {"type": "unknown", "which": which}
    except Exception as e:
        return {"type": "parse_error", "error": str(e)}


def generate_sine_audio(duration_sec: float = 2.0, freq: float = 440.0) -> bytes:
    """Generate a simple sine-wave tone as 16-bit PCM (for demo purposes)."""
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
    signal = (0.5 * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return signal.tobytes()


# ─────────────────────────────────────────────────
# 1. REST API helpers
# ─────────────────────────────────────────────────
async def check_health():
    """GET /health — verify server is running."""
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BASE_URL}/health") as r:
            data = await r.json()
            print(f"✅ Health: {data}")
            return data


async def get_config():
    """GET /config — discover server capabilities."""
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BASE_URL}/config") as r:
            data = await r.json()
            print(f"✅ Config:")
            print(f"   STT:  {data.get('stt_provider')}")
            print(f"   LLM:  {data.get('llm_provider')} / {data.get('llm_model')}")
            print(f"   TTS:  {data.get('tts_provider')}")
            print(f"   Modes: {data.get('supported_modes')}")
            print(f"   Languages: {data.get('supported_languages')}")
            return data


async def get_voices():
    """GET /voices — list available TTS voices."""
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BASE_URL}/voices") as r:
            data = await r.json()
            print(f"✅ Voices ({len(data['voices'])} available):")
            for v in data["voices"]:
                print(f"   - {v['name']} ({v['provider']})")
            return data


# ─────────────────────────────────────────────────
# 2. Tutor Mode: text_and_audio
# ─────────────────────────────────────────────────
async def tutor_text_and_audio():
    """
    Connect in text_and_audio mode.
    - Send a config message
    - Listen for greeting (text + audio)
    - Optionally send audio and get a response
    """
    print("\n" + "=" * 60)
    print("  TUTOR MODE — text_and_audio")
    print("=" * 60)

    async with websockets.connect(WS_URL, max_size=10 * 1024 * 1024) as ws:
        # Step 1: Send config
        config = {
            "type": "config",
            "mode": "text_and_audio",
            # Optional: custom system prompt
            # "system_prompt": "You are a friendly science tutor.",
            # Optional: prior conversation context
            # "context": [
            #     {"role": "user", "content": "What is photosynthesis?"},
            #     {"role": "assistant", "content": "Photosynthesis is..."}
            # ],
        }
        await ws.send(json.dumps(config))
        print("📤 Sent config: mode=text_and_audio")

        # Step 2: Listen for messages (greeting + pipeline setup)
        audio_chunks = 0
        text_chunks = []
        text_complete = None
        deadline = time.time() + 15  # wait up to 15s for greeting

        print("⏳ Waiting for greeting...")
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                if text_complete or audio_chunks > 0:
                    break  # Got something, move on
                continue

            if isinstance(msg, bytes):
                # Binary = protobuf frame (audio or transcription)
                parsed = parse_frame(msg)
                if parsed["type"] == "audio":
                    audio_chunks += 1
                elif parsed["type"] == "transcription":
                    print(f"🗣  STT: {parsed['text']}")
                # Don't print every audio chunk
            else:
                # Text = JSON message
                data = json.loads(msg)
                msg_type = data.get("type")

                if msg_type == "bot_text":
                    text_chunks.append(data["text"])
                    # Print streaming tokens inline
                    print(f"💬 [streaming] {data['text']}", end="", flush=True)

                elif msg_type == "bot_text_complete":
                    text_complete = data["text"]
                    print(f"\n📝 [complete] {text_complete}")

                else:
                    print(f"📨 {data}")

        print(f"\n📊 Received: {audio_chunks} audio chunks, "
              f"{len(text_chunks)} text tokens, "
              f"complete={'yes' if text_complete else 'no'}")

        # Step 3: (Optional) Send some audio to get a response
        # In a real app, you'd capture mic audio here.
        # For demo, we generate a short tone — Soniox won't transcribe it,
        # but it shows the protocol flow.
        print("\n📤 Sending 1s of demo audio...")
        audio = generate_sine_audio(1.0, 440)
        chunk_size = SAMPLE_RATE * 2  # 1 second of 16-bit mono
        for i in range(0, len(audio), chunk_size // 10):
            chunk = audio[i : i + chunk_size // 10]
            await ws.send(make_audio_frame(chunk))
            await asyncio.sleep(0.1)

        print("✅ Audio sent. In a real app, speak into your mic!")

    print("🔌 Disconnected\n")


# ─────────────────────────────────────────────────
# 3. Tutor Mode: text_only
# ─────────────────────────────────────────────────
async def tutor_text_only():
    """
    Connect in text_only mode.
    - No TTS audio is sent by the server
    - Only JSON text messages are received
    """
    print("\n" + "=" * 60)
    print("  TUTOR MODE — text_only")
    print("=" * 60)

    async with websockets.connect(WS_URL, max_size=10 * 1024 * 1024) as ws:
        # Send config with text_only mode
        config = {"type": "config", "mode": "text_only"}
        await ws.send(json.dumps(config))
        print("📤 Sent config: mode=text_only")

        # Listen for greeting (text only, no audio)
        audio_chunks = 0
        text_complete = None
        deadline = time.time() + 10

        print("⏳ Waiting for greeting (text only)...")
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                if text_complete:
                    break
                continue

            if isinstance(msg, bytes):
                audio_chunks += 1  # Should be 0 in text_only mode
            else:
                data = json.loads(msg)
                msg_type = data.get("type")
                if msg_type == "bot_text_complete":
                    text_complete = data["text"]
                    print(f"📝 Greeting: {text_complete}")
                elif msg_type == "bot_text":
                    print(f"💬 [token] {data['text']}", end="", flush=True)
                else:
                    print(f"📨 {data}")

        print(f"\n📊 Audio chunks received: {audio_chunks} (expected: 0 in text_only)")
        if audio_chunks == 0:
            print("✅ Confirmed: no audio in text_only mode")
        else:
            print("⚠️  Unexpected audio in text_only mode")

    print("🔌 Disconnected\n")


# ─────────────────────────────────────────────────
# 4. Runtime mode switching
# ─────────────────────────────────────────────────
async def tutor_mode_switch():
    """
    Start in text_and_audio, then switch to text_only at runtime
    without reconnecting.

    NOTE: Runtime mode switching sends a JSON `set_mode` message on the
    same WebSocket. Because Pipecat's transport reads binary frames for
    audio, the text `set_mode` message is handled by a separate listener
    task inside the server. The confirmation comes back as a JSON
    `mode_changed` message.

    If the Pipecat transport has already closed the connection by the
    time we send `set_mode`, it will fail — this is expected if no audio
    is being streamed. In production, you'd switch modes while actively
    sending mic audio.
    """
    print("\n" + "=" * 60)
    print("  TUTOR MODE — Runtime Mode Switch")
    print("  (Conceptual — shows the message format)")
    print("=" * 60)

    print("""
    To switch modes at runtime, send this JSON on the /ws WebSocket:

        {"type": "set_mode", "mode": "text_only"}

    The server responds with:

        {"type": "mode_changed", "mode": "text_only"}

    After this:
      - text_only:      no more audio frames, only bot_text JSON
      - text_and_audio:  audio frames resume alongside bot_text JSON

    This works on the same connection — no reconnect needed.
    The switch is instant and applies to the next LLM response.
    """)
    print("✅ Mode switch protocol documented\n")


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("  MIRA Voice AI — Tutor Mode Examples")
    print(f"  Server: {BASE_URL}")
    print("=" * 60)

    # 1. Check REST endpoints
    print("\n--- REST Endpoints ---")
    try:
        await check_health()
        await get_config()
        await get_voices()
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        print(f"   Make sure the server is running at {BASE_URL}")
        return

    # 2. Tutor mode: text_and_audio
    try:
        await tutor_text_and_audio()
    except Exception as e:
        print(f"❌ text_and_audio failed: {e}")

    # 3. Tutor mode: text_only
    try:
        await tutor_text_only()
    except Exception as e:
        print(f"❌ text_only failed: {e}")

    # 4. Runtime mode switch
    try:
        await tutor_mode_switch()
    except Exception as e:
        print(f"❌ mode_switch failed: {e}")

    print("\n" + "=" * 60)
    print("  All tutor examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
