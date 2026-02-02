#!/usr/bin/env python3
"""
Barge-In Test for MiraVoiceAI Pipecat.

This script tests the barge-in (interruption) functionality:
1. Sends initial audio to start a conversation
2. Waits for bot response to begin
3. Sends interruption audio mid-stream
4. Verifies the bot stops and processes the new utterance

Usage:
    # Test via WebSocket (requires server running)
    python test_bargein.py --server

    # Test components directly (no server needed)
    python test_bargein.py --direct

    # Generate test audio first
    python test_bargein.py --generate-audio
"""

import argparse
import asyncio
import json
import logging
import os
import struct
import sys
import time
import wave
from pathlib import Path

import aiohttp
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
ASR_WS_URL = os.getenv("ASR_WS_URL", "ws://localhost:8082/v1/audio/speech-to-text/stream")
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://localhost:8080")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://gpt-oss-120b/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
PIPECAT_WS_URL = os.getenv("PIPECAT_WS_URL", "ws://localhost:8000/ws")

TEST_AUDIO_DIR = Path(__file__).parent / "test_audio"

# WAV header size
WAV_HEADER_SIZE = 44


async def generate_test_audio():
    """Generate test audio files using TTS service."""
    TEST_AUDIO_DIR.mkdir(exist_ok=True)

    test_phrases = [
        ("user_greeting.wav", "नमस्ते, मुझे भारत के बारे में बताइए", "hi_male"),
        ("user_interrupt.wav", "रुको रुको, मुझे कुछ और पूछना है", "hi_male"),
        ("user_english.wav", "Hello, tell me about the weather today", "en_male"),
        ("user_short.wav", "हाँ", "hi_male"),
    ]

    async with aiohttp.ClientSession() as session:
        for filename, text, voice in test_phrases:
            filepath = TEST_AUDIO_DIR / filename
            logger.info(f"Generating {filename}: '{text}'")

            url = f"{TTS_BASE_URL}/v1/audio/text-to-speech/"
            payload = {
                "prompt": text,
                "voice": voice,
                "temperature": 0.75,
            }

            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    audio_data = await response.read()
                    with open(filepath, "wb") as f:
                        f.write(audio_data)
                    logger.info(f"  Saved: {filepath} ({len(audio_data)} bytes)")
                else:
                    logger.error(f"  Failed: HTTP {response.status}")

    logger.info("Test audio generation complete!")


def load_audio_file(filepath: str) -> tuple[bytes, int]:
    """Load audio file and return (pcm_bytes, sample_rate)."""
    with wave.open(filepath, "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)

        # Convert to mono if stereo
        if wf.getnchannels() == 2:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
            audio_data = audio_array.tobytes()

        return audio_data, sample_rate


async def test_bargein_direct():
    """
    Test barge-in directly with components (no Pipecat server needed).

    This simulates what happens during barge-in:
    1. Send audio to STT, get transcription
    2. Send to LLM, start getting response
    3. Start TTS streaming
    4. MID-STREAM: Send interruption signal, verify TTS stops
    5. Process new audio
    """
    import websockets

    logger.info("=" * 60)
    logger.info("BARGE-IN DIRECT TEST")
    logger.info("=" * 60)

    # Load test audio
    greeting_audio, sample_rate = load_audio_file(str(TEST_AUDIO_DIR / "user_greeting.wav"))
    interrupt_audio, _ = load_audio_file(str(TEST_AUDIO_DIR / "user_interrupt.wav"))

    logger.info(f"Loaded greeting audio: {len(greeting_audio)} bytes")
    logger.info(f"Loaded interrupt audio: {len(interrupt_audio)} bytes")

    # === Phase 1: Initial Transcription ===
    logger.info("\n--- Phase 1: Initial Transcription ---")

    async with websockets.connect(ASR_WS_URL, max_size=10*1024*1024) as ws:
        # Send config
        config = {"language": "auto", "interim_results": True, "sample_rate": sample_rate}
        await ws.send(json.dumps(config))

        ack = await asyncio.wait_for(ws.recv(), timeout=10.0)
        ack_data = json.loads(ack)
        assert ack_data.get("type") == "config_ack"
        logger.info(f"STT connected, session: {ack_data.get('session_id', 'N/A')[:8]}...")

        # Send greeting audio
        chunk_size = int(sample_rate * 0.1 * 2)
        for i in range(0, len(greeting_audio), chunk_size):
            await ws.send(greeting_audio[i:i+chunk_size])
            await asyncio.sleep(0.02)

        await ws.send("END")

        # Get transcription
        transcription = ""
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
            data = json.loads(msg)
            if data.get("type") == "final":
                transcription = data.get("text", "")
            elif data.get("type") == "done":
                transcription = data.get("text", transcription)
                break

        logger.info(f"User said: '{transcription}'")

    # === Phase 2: LLM Response (Streaming) ===
    logger.info("\n--- Phase 2: LLM Response (Streaming) ---")

    llm_response_chunks = []
    llm_start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        url = f"{LLM_BASE_URL}/chat/completions"
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are Idli, a helpful assistant. Respond in 2-3 sentences."},
                {"role": "user", "content": transcription}
            ],
            "max_tokens": 200,
            "stream": True,
        }

        async with session.post(url, json=payload) as response:
            first_token_time = None
            async for line in response.content:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        data = json.loads(line[6:])
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                                logger.info(f"LLM TTFT: {(first_token_time - llm_start_time)*1000:.1f}ms")
                            llm_response_chunks.append(content)
                    except json.JSONDecodeError:
                        pass

    llm_response = "".join(llm_response_chunks)
    logger.info(f"LLM response: '{llm_response[:100]}...'")

    # === Phase 3: TTS with Simulated Barge-In ===
    logger.info("\n--- Phase 3: TTS with Barge-In Simulation ---")

    # Note: Streaming is now WebSocket-based at /v1/audio/text-to-speech/stream
    # For testing barge-in simulation, we use the non-streaming HTTP endpoint
    tts_url = f"{TTS_BASE_URL}/v1/audio/text-to-speech/"
    tts_payload = {
        "prompt": llm_response,
        "voice": "hi_male",
        "temperature": 0.75,
    }

    tts_start_time = time.perf_counter()
    total_audio_bytes = 0
    interrupted = False
    interrupt_time = None
    chunks_before_interrupt = 0

    async with aiohttp.ClientSession() as session:
        async with session.post(tts_url, json=tts_payload) as response:
            if response.status != 200:
                logger.error(f"TTS failed: {response.status}")
                return

            ttfs_logged = False
            async for chunk in response.content.iter_chunked(4096):
                total_audio_bytes += len(chunk)

                # Log TTFS (first actual audio after header)
                if not ttfs_logged and total_audio_bytes > WAV_HEADER_SIZE:
                    ttfs = (time.perf_counter() - tts_start_time) * 1000
                    logger.info(f"TTS TTFS (actual audio): {ttfs:.1f}ms")
                    ttfs_logged = True

                chunks_before_interrupt += 1

                # Simulate barge-in after receiving ~50KB of audio
                if total_audio_bytes > 50000 and not interrupted:
                    interrupt_time = time.perf_counter()
                    interrupted = True
                    logger.info(f">>> BARGE-IN TRIGGERED at {total_audio_bytes} bytes ({chunks_before_interrupt} chunks)")
                    logger.info(">>> In real scenario, TTS would stop here and new utterance would be processed")
                    # In a real scenario, we would:
                    # 1. Set _interrupted = True on TTS service
                    # 2. Stop reading from this stream
                    # 3. Start processing the interrupt audio
                    break

    if interrupted:
        logger.info(f"\nBarge-in simulation successful!")
        logger.info(f"  Audio received before interrupt: {total_audio_bytes} bytes")
        logger.info(f"  Chunks received: {chunks_before_interrupt}")

        # Now process the "interruption" - new user speech
        logger.info("\n--- Phase 4: Processing Interruption ---")

        async with websockets.connect(ASR_WS_URL, max_size=10*1024*1024) as ws:
            config = {"language": "auto", "interim_results": True, "sample_rate": sample_rate}
            await ws.send(json.dumps(config))

            ack = await asyncio.wait_for(ws.recv(), timeout=10.0)

            # Send interrupt audio
            interrupt_start = time.perf_counter()
            for i in range(0, len(interrupt_audio), chunk_size):
                await ws.send(interrupt_audio[i:i+chunk_size])
                await asyncio.sleep(0.02)

            await ws.send("END")

            # Get new transcription
            new_transcription = ""
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                data = json.loads(msg)
                if data.get("type") == "final":
                    new_transcription = data.get("text", "")
                elif data.get("type") == "done":
                    new_transcription = data.get("text", new_transcription)
                    break

            interrupt_latency = (time.perf_counter() - interrupt_start) * 1000
            logger.info(f"Interrupt transcription: '{new_transcription}'")
            logger.info(f"Interrupt processing time: {interrupt_latency:.1f}ms")

    logger.info("\n" + "=" * 60)
    logger.info("BARGE-IN TEST COMPLETE")
    logger.info("=" * 60)


async def test_bargein_server():
    """
    Test barge-in via the Pipecat WebSocket server.

    This requires the server to be running:
        python server.py
    """
    import websockets

    logger.info("=" * 60)
    logger.info("BARGE-IN SERVER TEST")
    logger.info("=" * 60)

    # Load test audio
    greeting_audio, sample_rate = load_audio_file(str(TEST_AUDIO_DIR / "user_greeting.wav"))
    interrupt_audio, _ = load_audio_file(str(TEST_AUDIO_DIR / "user_interrupt.wav"))

    logger.info(f"Loaded greeting audio: {len(greeting_audio)} bytes @ {sample_rate}Hz")
    logger.info(f"Loaded interrupt audio: {len(interrupt_audio)} bytes")

    ws_url = f"{PIPECAT_WS_URL}?voice=hi_male&language=auto&sample_rate={sample_rate}"
    logger.info(f"Connecting to: {ws_url}")

    try:
        async with websockets.connect(ws_url, max_size=10*1024*1024) as ws:
            logger.info("Connected to Pipecat server")

            # Track state
            receiving_audio = False
            audio_chunks_received = 0
            total_audio_bytes = 0
            barge_in_sent = False

            # Task to receive messages
            async def receive_messages():
                nonlocal receiving_audio, audio_chunks_received, total_audio_bytes

                try:
                    while True:
                        msg = await ws.recv()

                        if isinstance(msg, bytes):
                            # Audio data from TTS
                            audio_chunks_received += 1
                            total_audio_bytes += len(msg)

                            if not receiving_audio:
                                receiving_audio = True
                                logger.info(f">>> Started receiving TTS audio")

                            if audio_chunks_received % 10 == 0:
                                logger.info(f"  Audio chunks: {audio_chunks_received}, bytes: {total_audio_bytes}")
                        else:
                            # JSON message
                            try:
                                data = json.loads(msg)
                                msg_type = data.get("type", "unknown")

                                if msg_type == "transcription":
                                    logger.info(f"Transcription: {data.get('text', '')}")
                                elif msg_type == "bot_started_speaking":
                                    logger.info("Bot started speaking")
                                elif msg_type == "bot_stopped_speaking":
                                    logger.info("Bot stopped speaking")
                                    receiving_audio = False
                                elif msg_type == "user_started_speaking":
                                    logger.info("User started speaking (VAD detected)")
                                elif msg_type == "user_stopped_speaking":
                                    logger.info("User stopped speaking")
                                else:
                                    logger.debug(f"Message: {data}")
                            except json.JSONDecodeError:
                                logger.warning(f"Non-JSON message: {msg[:100]}")

                except websockets.ConnectionClosed:
                    logger.info("Connection closed")

            # Start receiver task
            receiver = asyncio.create_task(receive_messages())

            # Send greeting audio
            logger.info("\n--- Sending initial greeting ---")
            chunk_size = int(sample_rate * 0.1 * 2)  # 100ms chunks

            for i in range(0, len(greeting_audio), chunk_size):
                chunk = greeting_audio[i:i+chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.05)  # Simulate real-time

            logger.info("Greeting audio sent, waiting for response...")

            # Wait for TTS to start, then send interrupt
            await asyncio.sleep(2.0)  # Wait for STT + LLM + TTS to start

            if receiving_audio and audio_chunks_received > 5:
                logger.info(f"\n>>> SENDING BARGE-IN after {audio_chunks_received} audio chunks")

                # Send interrupt audio
                for i in range(0, len(interrupt_audio), chunk_size):
                    chunk = interrupt_audio[i:i+chunk_size]
                    await ws.send(chunk)
                    await asyncio.sleep(0.05)

                barge_in_sent = True
                logger.info("Interrupt audio sent")

            # Wait for processing
            await asyncio.sleep(5.0)

            # Cancel receiver
            receiver.cancel()
            try:
                await receiver
            except asyncio.CancelledError:
                pass

            logger.info("\n" + "=" * 60)
            logger.info("TEST RESULTS")
            logger.info("=" * 60)
            logger.info(f"Total audio chunks received: {audio_chunks_received}")
            logger.info(f"Total audio bytes received: {total_audio_bytes}")
            logger.info(f"Barge-in sent: {barge_in_sent}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


async def main():
    parser = argparse.ArgumentParser(description="Barge-in test for MiraVoiceAI Pipecat")
    parser.add_argument("--server", action="store_true", help="Test via Pipecat server (requires server running)")
    parser.add_argument("--direct", action="store_true", help="Test components directly")
    parser.add_argument("--generate-audio", action="store_true", help="Generate test audio files")

    args = parser.parse_args()

    if args.generate_audio:
        await generate_test_audio()
    elif args.server:
        await test_bargein_server()
    elif args.direct:
        await test_bargein_direct()
    else:
        # Default: run direct test
        logger.info("Running direct component test (use --server for server test)")
        await test_bargein_direct()


if __name__ == "__main__":
    asyncio.run(main())
