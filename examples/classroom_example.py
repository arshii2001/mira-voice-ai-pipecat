#!/usr/bin/env python3
"""
MIRA Voice AI — Classroom Mode Example Client

Demonstrates a multi-user classroom where:
  - Ravi (English speaker) asks Mira a question
  - Priya (Hindi listener) receives translated text + audio
  - Anita (Tamil listener, text_only) receives translated text only

This example shows:
  1. Creating a room via REST API
  2. Three users joining via classroom WebSocket
  3. Speaker token auto-assignment and passing
  4. Runtime mode switching (text_only ↔ text_and_audio)
  5. Speaker connecting the voice pipeline on /ws
  6. Listeners receiving translated broadcasts
  7. Cleaning up (delete room)

Prerequisites:
  pip install websockets aiohttp numpy

Usage:
  # Against a local server:
  python examples/classroom_example.py

  # Against a remote server:
  MIRA_HOST=192.168.1.100 python examples/classroom_example.py

  # Inside Docker:
  MIRA_HOST=mira-voice python examples/classroom_example.py

  # Run full pipeline test (requires STT/LLM/TTS keys):
  RUN_FULL_PIPELINE=1 python examples/classroom_example.py
"""

import asyncio
import json
import os
import sys
import time
from typing import List, Optional

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")

try:
    import aiohttp
except ImportError:
    sys.exit("pip install aiohttp")

# Try to import Pipecat protobuf for speaker audio
try:
    import numpy as np
    import pipecat.frames.protobufs.frames_pb2 as frame_protos
    HAS_PROTO = True
except ImportError:
    HAS_PROTO = False

# ─────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────
HOST = os.getenv("MIRA_HOST", "localhost")
PORT = os.getenv("MIRA_PORT", "7860")
BASE_URL = f"http://{HOST}:{PORT}"
WS_URL = f"ws://{HOST}:{PORT}/ws"
CLASSROOM_WS = f"ws://{HOST}:{PORT}/classroom/rooms"
SAMPLE_RATE = 16000


# ─────────────────────────────────────────────────
# REST API helpers
# ─────────────────────────────────────────────────
async def create_room(name: str = "Demo Classroom") -> dict:
    """POST /classroom/rooms — Create a new room."""
    async with aiohttp.ClientSession() as s:
        async with s.post(f"{BASE_URL}/classroom/rooms", params={"name": name}) as r:
            data = await r.json()
            print(f"✅ Room created: {data['room_id']} ({data['name']})")
            return data


async def list_rooms() -> dict:
    """GET /classroom/rooms — List all active rooms."""
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BASE_URL}/classroom/rooms") as r:
            data = await r.json()
            print(f"📋 Active rooms: {len(data['rooms'])}")
            for room in data["rooms"]:
                print(f"   - {room['name']} ({room['room_id']}) "
                      f"— {room['user_count']} users, "
                      f"speaker: {room.get('speaker_name', 'none')}")
            return data


async def get_room(room_id: str) -> dict:
    """GET /classroom/rooms/{room_id} — Get room details."""
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BASE_URL}/classroom/rooms/{room_id}") as r:
            return await r.json()


async def delete_room(room_id: str) -> dict:
    """DELETE /classroom/rooms/{room_id} — Delete a room."""
    async with aiohttp.ClientSession() as s:
        async with s.delete(f"{BASE_URL}/classroom/rooms/{room_id}") as r:
            data = await r.json()
            print(f"🗑  Room deleted: {room_id}")
            return data


async def manage_token(room_id: str, action: str, user_id: str,
                       to_user_id: str = None) -> dict:
    """POST /classroom/rooms/{room_id}/token — Request/pass/release token."""
    params = {"action": action, "user_id": user_id}
    if to_user_id:
        params["to_user_id"] = to_user_id
    async with aiohttp.ClientSession() as s:
        async with s.post(f"{BASE_URL}/classroom/rooms/{room_id}/token",
                          params=params) as r:
            return await r.json()


# ─────────────────────────────────────────────────
# WebSocket helpers
# ─────────────────────────────────────────────────
async def recv_json(ws, timeout: float = 5.0) -> Optional[dict]:
    """Receive a JSON message, skip binary frames."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=max(deadline - time.time(), 0.1))
            if isinstance(msg, str):
                return json.loads(msg)
            # Binary frame (audio) — skip when looking for JSON
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None
    return None


async def drain_messages(ws, duration: float = 2.0) -> List[dict]:
    """Collect all JSON messages for a duration, skipping binary."""
    messages = []
    start = time.time()
    while time.time() - start < duration:
        msg = await recv_json(ws, timeout=0.3)
        if msg:
            messages.append(msg)
    return messages


async def join_room_ws(ws, room_id: str, user_id: str, name: str,
                       language: str, mode: str = "text_and_audio") -> dict:
    """Send join message and return the 'joined' response."""
    await ws.send(json.dumps({
        "type": "join",
        "user_id": user_id,
        "name": name,
        "language": language,
        "mode": mode,
    }))
    # Wait for 'joined' response
    for _ in range(10):
        msg = await recv_json(ws, timeout=3.0)
        if msg and msg.get("type") == "joined":
            return msg
    raise RuntimeError(f"Did not receive 'joined' for {name}")


# ─────────────────────────────────────────────────
# Example 1: Room CRUD
# ─────────────────────────────────────────────────
async def example_room_crud():
    """Demonstrate room create/list/get/delete via REST API."""
    print("\n" + "=" * 60)
    print("  EXAMPLE 1: Room CRUD (REST API)")
    print("=" * 60)

    # Create
    room = await create_room("Physics Class")
    room_id = room["room_id"]

    # List
    await list_rooms()

    # Get
    details = await get_room(room_id)
    print(f"📄 Room details: {details['name']}, "
          f"users: {details['user_count']}, "
          f"speaker: {details.get('speaker_name', 'none')}")

    # Delete
    await delete_room(room_id)

    # Verify
    rooms = await list_rooms()
    assert all(r["room_id"] != room_id for r in rooms["rooms"]), "Room should be gone"
    print("✅ CRUD complete\n")


# ─────────────────────────────────────────────────
# Example 2: Multi-user join + token management
# ─────────────────────────────────────────────────
async def example_join_and_tokens():
    """Three users join a room. Demonstrates token auto-assignment, passing, and queuing."""
    print("\n" + "=" * 60)
    print("  EXAMPLE 2: Join Room + Token Management")
    print("=" * 60)

    room = await create_room("Math Class")
    room_id = room["room_id"]

    # --- User 1: Ravi (English, text_and_audio) ---
    ws_ravi = await websockets.connect(f"{CLASSROOM_WS}/{room_id}/ws")
    joined_ravi = await join_room_ws(ws_ravi, room_id, "ravi-01", "Ravi", "en", "text_and_audio")
    print(f"👤 Ravi joined: language={joined_ravi['you']['language']}, "
          f"mode={joined_ravi['you']['mode']}")

    # First user gets auto-assigned speaker token
    ravi_events = await drain_messages(ws_ravi, duration=1.5)
    token_events = [e for e in ravi_events if e.get("type") == "token_changed"]
    if token_events:
        print(f"🎤 Ravi got speaker token: {token_events[0]['speaker_name']}")

    # --- User 2: Priya (Hindi, text_and_audio) ---
    ws_priya = await websockets.connect(f"{CLASSROOM_WS}/{room_id}/ws")
    joined_priya = await join_room_ws(ws_priya, room_id, "priya-01", "Priya", "hi", "text_and_audio")
    print(f"👤 Priya joined: language={joined_priya['you']['language']}")
    await drain_messages(ws_priya, duration=1.0)

    # --- User 3: Anita (Tamil, text_only) ---
    ws_anita = await websockets.connect(f"{CLASSROOM_WS}/{room_id}/ws")
    joined_anita = await join_room_ws(ws_anita, room_id, "anita-01", "Anita", "ta", "text_only")
    print(f"👤 Anita joined: language={joined_anita['you']['language']}, "
          f"mode={joined_anita['you']['mode']}")
    await drain_messages(ws_anita, duration=1.0)

    # Check room state
    state = await get_room(room_id)
    print(f"\n📊 Room state: {state['user_count']} users, "
          f"speaker: {state.get('speaker_name', 'none')}")
    for u in state["users"]:
        role = "🎤 speaker" if u.get("is_speaker") else "👂 listener"
        print(f"   {role}: {u['name']} ({u['language']}, {u['mode']})")

    # --- Pass token: Ravi → Priya ---
    print("\n🔄 Ravi passes token to Priya...")
    await ws_ravi.send(json.dumps({"type": "pass_token", "to": "priya-01"}))
    await asyncio.sleep(0.5)

    priya_events = await drain_messages(ws_priya, duration=2.0)
    token_to_priya = [e for e in priya_events
                      if e.get("type") == "token_changed" and e.get("speaker_id") == "priya-01"]
    if token_to_priya:
        print(f"✅ Token passed to Priya")
    else:
        print(f"⚠  Token pass event not received by Priya")

    # --- Request token (queue) ---
    print("\n📝 Ravi requests token (should be queued, Priya is speaking)...")
    await ws_ravi.send(json.dumps({"type": "request_token"}))
    ravi_resp = await drain_messages(ws_ravi, duration=2.0)
    req_resp = next((e for e in ravi_resp if e.get("type") == "token_response"), None)
    if req_resp:
        print(f"   Granted: {req_resp.get('granted')} "
              f"{'(queued)' if not req_resp.get('granted') else '(immediate)'}")

    # --- Release token (queue auto-assigns) ---
    print("\n🎤 Priya releases token...")
    await ws_priya.send(json.dumps({"type": "release_token"}))
    await asyncio.sleep(0.5)

    ravi_events2 = await drain_messages(ws_ravi, duration=2.0)
    token_back = [e for e in ravi_events2
                  if e.get("type") == "token_changed" and e.get("speaker_id") == "ravi-01"]
    if token_back:
        print(f"✅ Token auto-assigned to Ravi (from queue)")

    # Cleanup
    await ws_ravi.close()
    await ws_priya.close()
    await ws_anita.close()
    await delete_room(room_id)
    print("✅ Token management example complete\n")


# ─────────────────────────────────────────────────
# Example 3: Mode switching at runtime
# ─────────────────────────────────────────────────
async def example_mode_switch():
    """Demonstrate switching between text_only and text_and_audio at runtime."""
    print("\n" + "=" * 60)
    print("  EXAMPLE 3: Runtime Mode Switching")
    print("=" * 60)

    room = await create_room("Mode Switch Demo")
    room_id = room["room_id"]

    ws = await websockets.connect(f"{CLASSROOM_WS}/{room_id}/ws")
    joined = await join_room_ws(ws, room_id, "user-01", "TestUser", "en", "text_only")
    print(f"👤 Joined in mode: {joined['you']['mode']}")
    await drain_messages(ws, duration=1.0)

    # Switch to text_and_audio
    print("🔄 Switching to text_and_audio...")
    await ws.send(json.dumps({"type": "set_mode", "mode": "text_and_audio"}))
    resp = await recv_json(ws, timeout=3.0)
    if resp and resp.get("type") == "mode_changed":
        print(f"✅ Mode changed to: {resp['mode']}")

    # Switch back to text_only
    print("🔄 Switching to text_only...")
    await ws.send(json.dumps({"type": "set_mode", "mode": "text_only"}))
    resp = await recv_json(ws, timeout=3.0)
    if resp and resp.get("type") == "mode_changed":
        print(f"✅ Mode changed to: {resp['mode']}")

    # Invalid mode
    print("🔄 Trying invalid mode 'video_only'...")
    await ws.send(json.dumps({"type": "set_mode", "mode": "video_only"}))
    resp = await recv_json(ws, timeout=3.0)
    if resp and resp.get("type") == "error":
        print(f"✅ Rejected: {resp['message']}")

    await ws.close()
    await delete_room(room_id)
    print("✅ Mode switch example complete\n")


# ─────────────────────────────────────────────────
# Example 4: Speaker pipeline with listener broadcast
# ─────────────────────────────────────────────────
async def example_speaker_pipeline():
    """
    Full end-to-end classroom flow:
      1. Create room
      2. Ravi joins (English speaker)
      3. Priya joins (Hindi listener, text_and_audio)
      4. Anita joins (Tamil listener, text_only)
      5. Ravi connects voice pipeline on /ws with room_id
      6. Ravi speaks → Mira responds → listeners get translations
    """
    if os.getenv("RUN_FULL_PIPELINE", "0") != "1":
        print("\n" + "=" * 60)
        print("  EXAMPLE 4: Speaker Pipeline (SKIPPED)")
        print("  Set RUN_FULL_PIPELINE=1 to run this example")
        print("  (requires STT/LLM/TTS API keys)")
        print("=" * 60 + "\n")
        return

    if not HAS_PROTO:
        print("⚠  Pipecat protobuf not available; skipping pipeline example")
        return

    print("\n" + "=" * 60)
    print("  EXAMPLE 4: Full Speaker Pipeline + Listener Broadcast")
    print("=" * 60)

    # Step 1: Create room
    room = await create_room("Live Demo")
    room_id = room["room_id"]

    # Step 2: Ravi joins classroom WS (control channel)
    ws_ravi_ctrl = await websockets.connect(f"{CLASSROOM_WS}/{room_id}/ws")
    await join_room_ws(ws_ravi_ctrl, room_id, "ravi-01", "Ravi", "en", "text_and_audio")
    await drain_messages(ws_ravi_ctrl, duration=1.5)  # Get token_changed
    print("🎤 Ravi joined as speaker")

    # Step 3: Priya joins as listener (Hindi, text_and_audio)
    ws_priya = await websockets.connect(f"{CLASSROOM_WS}/{room_id}/ws")
    await join_room_ws(ws_priya, room_id, "priya-01", "Priya", "hi", "text_and_audio")
    await drain_messages(ws_priya, duration=1.0)
    print("👂 Priya joined (Hindi, text_and_audio)")

    # Step 4: Anita joins as listener (Tamil, text_only)
    ws_anita = await websockets.connect(f"{CLASSROOM_WS}/{room_id}/ws")
    await join_room_ws(ws_anita, room_id, "anita-01", "Anita", "ta", "text_only")
    await drain_messages(ws_anita, duration=1.0)
    print("👂 Anita joined (Tamil, text_only)")

    # Step 5: Ravi opens voice pipeline on /ws
    ws_speaker = await websockets.connect(WS_URL, max_size=10 * 1024 * 1024)
    await ws_speaker.send(json.dumps({
        "type": "config",
        "mode": "text_and_audio",
        "room_id": room_id,
        "speaker_id": "ravi-01",
    }))
    print("🔗 Speaker pipeline connected to /ws")

    # Wait for greeting
    print("⏳ Waiting for Mira's greeting...")
    await asyncio.sleep(5)
    speaker_msgs = await drain_messages(ws_speaker, duration=3.0)
    print(f"   Received {len(speaker_msgs)} greeting messages")

    # Step 6: Send audio from a WAV file (or generate synthetic)
    # In a real app, this would be microphone input
    audio_dir = os.getenv("AUDIO_DIR", "tests/test_audio")
    wav_path = os.path.join(audio_dir, "user_greeting.wav")

    if os.path.exists(wav_path):
        import wave
        with wave.open(wav_path, "rb") as wf:
            audio_data = wf.readframes(wf.getnframes())
            sr = wf.getframerate()
        print(f"📤 Sending speech audio ({len(audio_data)} bytes, {sr}Hz)...")
    else:
        print(f"⚠  WAV file not found at {wav_path}")
        print("   Generating synthetic audio (STT may not transcribe it)")
        t = np.linspace(0, 3.0, int(SAMPLE_RATE * 3.0), endpoint=False)
        signal = (0.5 * 32767 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)
        audio_data = signal.tobytes()
        sr = SAMPLE_RATE

    # Send audio in chunks
    chunk_ms = 100
    chunk_size = int(sr * (chunk_ms / 1000.0)) * 2  # 16-bit mono
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i : i + chunk_size]
        frame = frame_protos.Frame()
        frame.audio.audio = chunk
        frame.audio.sample_rate = sr
        frame.audio.num_channels = 1
        await ws_speaker.send(frame.SerializeToString())
        await asyncio.sleep(chunk_ms / 1000.0)

    # Send silence for VAD end-of-speech detection
    silence = np.zeros(int(sr * 2.0), dtype=np.int16).tobytes()
    for i in range(0, len(silence), chunk_size):
        chunk = silence[i : i + chunk_size]
        frame = frame_protos.Frame()
        frame.audio.audio = chunk
        frame.audio.sample_rate = sr
        frame.audio.num_channels = 1
        await ws_speaker.send(frame.SerializeToString())
        await asyncio.sleep(chunk_ms / 1000.0)
    print("📤 Audio sent, waiting for pipeline + broadcast...")

    # Step 7: Collect listener events
    priya_events = []
    anita_events = []
    start_wait = time.time()
    while time.time() - start_wait < 30:
        # Check Priya
        try:
            msg = await asyncio.wait_for(ws_priya.recv(), timeout=0.3)
            if isinstance(msg, str):
                data = json.loads(msg)
                priya_events.append(data)
                print(f"   👂 Priya: {data.get('type')} — "
                      f"{data.get('translated_text', data.get('text', ''))[:80]}")
            elif isinstance(msg, bytes):
                priya_events.append({"type": "binary_audio", "length": len(msg)})
        except asyncio.TimeoutError:
            pass

        # Check Anita
        try:
            msg = await asyncio.wait_for(ws_anita.recv(), timeout=0.3)
            if isinstance(msg, str):
                data = json.loads(msg)
                anita_events.append(data)
                print(f"   👂 Anita: {data.get('type')} — "
                      f"{data.get('translated_text', data.get('text', ''))[:80]}")
            elif isinstance(msg, bytes):
                anita_events.append({"type": "binary_audio", "length": len(msg)})
        except asyncio.TimeoutError:
            pass

        # Stop when we have bot_response from both
        priya_bot = any(e.get("type") == "bot_response" for e in priya_events)
        anita_bot = any(e.get("type") == "bot_response" for e in anita_events)
        if priya_bot and anita_bot:
            break

    # Summary
    print(f"\n📊 Priya received: {len(priya_events)} events")
    for e in priya_events:
        if e.get("type") not in ("binary_audio",):
            print(f"   {e.get('type')}: {str(e)[:100]}")

    print(f"\n📊 Anita received: {len(anita_events)} events")
    for e in anita_events:
        if e.get("type") not in ("binary_audio",):
            print(f"   {e.get('type')}: {str(e)[:100]}")

    # Verify Anita got no audio (text_only mode)
    anita_audio = [e for e in anita_events if e.get("type") == "binary_audio"]
    if not anita_audio:
        print("✅ Anita (text_only): no audio received — correct!")
    else:
        print(f"⚠  Anita (text_only): received {len(anita_audio)} audio chunks")

    # Cleanup
    await ws_speaker.close()
    await ws_ravi_ctrl.close()
    await ws_priya.close()
    await ws_anita.close()
    await delete_room(room_id)
    print("✅ Speaker pipeline example complete\n")


# ─────────────────────────────────────────────────
# Example 5: Token management via REST API
# ─────────────────────────────────────────────────
async def example_token_rest_api():
    """Demonstrate token management using the REST API instead of WebSocket messages."""
    print("\n" + "=" * 60)
    print("  EXAMPLE 5: Token Management via REST API")
    print("=" * 60)

    room = await create_room("REST Token Demo")
    room_id = room["room_id"]

    # Two users join via WebSocket
    ws1 = await websockets.connect(f"{CLASSROOM_WS}/{room_id}/ws")
    await join_room_ws(ws1, room_id, "user-a", "Alice", "en")
    await drain_messages(ws1, duration=1.5)
    print("👤 Alice joined")

    ws2 = await websockets.connect(f"{CLASSROOM_WS}/{room_id}/ws")
    await join_room_ws(ws2, room_id, "user-b", "Bob", "hi")
    await drain_messages(ws2, duration=1.0)
    print("👤 Bob joined")

    # Check current state
    state = await get_room(room_id)
    print(f"🎤 Current speaker: {state.get('speaker_name', 'none')}")

    # Bob requests token via REST
    print("\n📝 Bob requests token via REST...")
    resp = await manage_token(room_id, "request", "user-b")
    print(f"   Response: {resp}")

    # Alice passes token via REST
    print("🔄 Alice passes token to Bob via REST...")
    resp = await manage_token(room_id, "pass", "user-a", "user-b")
    print(f"   Response: {resp}")

    # Check state
    state = await get_room(room_id)
    print(f"🎤 Current speaker: {state.get('speaker_name', 'none')}")

    # Bob releases token via REST
    print("📤 Bob releases token via REST...")
    resp = await manage_token(room_id, "release", "user-b")
    print(f"   Response: {resp}")

    state = await get_room(room_id)
    print(f"🎤 Current speaker: {state.get('speaker_name', 'none')}")

    await ws1.close()
    await ws2.close()
    await delete_room(room_id)
    print("✅ REST token management complete\n")


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("  MIRA Voice AI — Classroom Mode Examples")
    print(f"  Server: {BASE_URL}")
    print("=" * 60)

    # Check server health first
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{BASE_URL}/health") as r:
                health = await r.json()
                print(f"✅ Server healthy: {health}")
    except Exception as e:
        print(f"❌ Server not reachable at {BASE_URL}: {e}")
        print("   Start the server first: docker compose up -d")
        return

    # Run examples
    try:
        await example_room_crud()
    except Exception as e:
        print(f"❌ Room CRUD failed: {e}")

    try:
        await example_join_and_tokens()
    except Exception as e:
        print(f"❌ Join & Tokens failed: {e}")

    try:
        await example_mode_switch()
    except Exception as e:
        print(f"❌ Mode Switch failed: {e}")

    try:
        await example_token_rest_api()
    except Exception as e:
        print(f"❌ REST Token API failed: {e}")

    try:
        await example_speaker_pipeline()
    except Exception as e:
        print(f"❌ Speaker Pipeline failed: {e}")

    print("\n" + "=" * 60)
    print("  All classroom examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
