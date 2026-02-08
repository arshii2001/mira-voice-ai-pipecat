#!/usr/bin/env python3
"""
Simulate a speaker in a classroom room who sends text questions to Mira.

Usage:
    python test_broadcast_sim.py <room_id>

The script will:
1. Connect to the classroom WebSocket as "Teacher Bot" (English)
2. Request the speaker token
3. Send a series of text questions with pauses between them
4. Print all received messages (including Mira's streamed responses)

Meanwhile, you should be in the same room as a Hindi listener in the browser
to verify that the translated conversation appears.
"""

import asyncio
import json
import sys
import websockets

BACKEND_URL = "localhost:7860"

QUESTIONS = [
    "What is photosynthesis and why is it important?",
    "Can you explain Newton's three laws of motion in simple terms?",
    "What are the planets in our solar system?",
]

DELAY_BETWEEN_QUESTIONS = 30  # seconds — enough for LLM response + TTS audio broadcast to finish


async def simulate_speaker(room_id: str):
    uri = f"ws://{BACKEND_URL}/classroom/rooms/{room_id}/ws"
    print(f"\n🎓 Connecting to classroom room {room_id}...")
    print(f"   URI: {uri}\n")

    async with websockets.connect(uri) as ws:
        # 1. Join the room
        join_msg = {
            "type": "join",
            "user_id": "sim-teacher-001",
            "name": "Teacher Bot",
            "language": "en",
            "mode": "text_and_audio",
        }
        await ws.send(json.dumps(join_msg))
        print("📤 Sent: join as 'Teacher Bot' (English)")

        # Read join response
        resp = json.loads(await ws.recv())
        if resp.get("type") == "joined":
            print(f"✅ Joined room: {resp['room'].get('name', room_id)}")
            print(f"   Users in room: {resp['room'].get('user_count', '?')}")
            is_speaker = resp.get("you", {}).get("is_speaker", False)
            print(f"   Am I speaker? {is_speaker}")
        else:
            print(f"⚠️  Unexpected response: {resp}")

        # 2. Request token if not already speaker
        if not is_speaker:
            await ws.send(json.dumps({"type": "request_token"}))
            print("\n📤 Sent: request_token")

            # Wait for token response
            while True:
                msg = json.loads(await ws.recv())
                print(f"📥 Received: {msg.get('type', '?')} - {json.dumps(msg)[:120]}")
                if msg.get("type") == "token_response" and msg.get("granted"):
                    print("✅ Got speaker token!")
                    break
                if msg.get("type") == "token_changed" and msg.get("speaker_id") == "sim-teacher-001":
                    print("✅ Got speaker token (via token_changed)!")
                    break
                if msg.get("type") == "error":
                    print(f"❌ Error: {msg.get('message')}")
                    return

        # Small pause to let things settle
        await asyncio.sleep(2)

        # 3. Send questions one by one
        for i, question in enumerate(QUESTIONS, 1):
            print(f"\n{'='*60}")
            print(f"📤 Question {i}/{len(QUESTIONS)}: {question}")
            print(f"{'='*60}")

            await ws.send(json.dumps({"type": "text_message", "text": question}))

            # Collect responses until we see bot_text_complete
            full_response = ""
            got_complete = False
            timeout = 30  # seconds max per question

            try:
                while not got_complete:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")

                    if msg_type == "bot_text":
                        token = msg.get("text", "")
                        full_response += token
                        print(token, end="", flush=True)

                    elif msg_type == "bot_text_complete":
                        got_complete = True
                        complete_text = msg.get("text", "")
                        if complete_text and not full_response:
                            full_response = complete_text
                        print()  # newline after streaming
                        print(f"\n✅ Mira's full response ({len(full_response)} chars):")
                        print(f"   {full_response[:200]}{'...' if len(full_response) > 200 else ''}")

                    elif msg_type == "error":
                        print(f"\n❌ Error: {msg.get('message')}")
                        got_complete = True

                    else:
                        # Other messages (token_changed, user_joined, etc.)
                        print(f"\n   [event] {msg_type}: {json.dumps(msg)[:100]}")

            except asyncio.TimeoutError:
                print(f"\n⏰ Timeout waiting for response to question {i}")

            # Pause before next question
            if i < len(QUESTIONS):
                print(f"\n⏳ Waiting {DELAY_BETWEEN_QUESTIONS}s before next question...")
                await asyncio.sleep(DELAY_BETWEEN_QUESTIONS)

        # 4. Done — release token and disconnect
        print(f"\n{'='*60}")
        print("🏁 All questions sent! Releasing token...")
        await ws.send(json.dumps({"type": "release_token"}))
        await asyncio.sleep(2)

        # Drain any remaining messages
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=2)
                msg = json.loads(raw)
                print(f"   [cleanup] {msg.get('type', '?')}: {json.dumps(msg)[:100]}")
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
            pass

        print("\n✅ Simulation complete! Check your browser for the translated conversation.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Try to get the first room automatically
        import urllib.request
        try:
            with urllib.request.urlopen(f"http://{BACKEND_URL}/classroom/rooms") as resp:
                data = json.loads(resp.read())
                rooms = data.get("rooms", [])
                if rooms:
                    room_id = rooms[0]["room_id"]
                    print(f"Auto-detected room: {room_id} ({rooms[0]['name']})")
                else:
                    print("No rooms found. Create one first.")
                    sys.exit(1)
        except Exception as e:
            print(f"Could not auto-detect room: {e}")
            print(f"Usage: python {sys.argv[0]} <room_id>")
            sys.exit(1)
    else:
        room_id = sys.argv[1]

    asyncio.run(simulate_speaker(room_id))
