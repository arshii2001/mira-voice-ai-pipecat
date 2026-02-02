#!/usr/bin/env python3
"""
Test client for IdliDemo Pipecat Server.

This client connects to the Pipecat server via WebSocket and tests the
STT -> LLM -> TTS pipeline with pre-recorded audio or real-time microphone input.

Usage:
    # Test with audio file
    python test_client.py --audio sample.wav

    # Test with microphone (requires pyaudio)
    python test_client.py --mic

    # Test with text (bypasses STT, useful for TTS testing)
    python test_client.py --text "Hello, how are you?"
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
from typing import Optional

import numpy as np

try:
    import websockets
except ImportError:
    print("Error: websockets package required. Install with: pip install websockets")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PipecatTestClient:
    """Test client for IdliDemo Pipecat server."""

    def __init__(
        self,
        ws_url: str = "ws://localhost:8000/ws",
        voice: str = "hi_male",
        language: str = "auto",
        sample_rate: int = 16000,
    ):
        """
        Initialize test client.

        Args:
            ws_url: WebSocket URL for Pipecat server
            voice: TTS voice ID
            language: STT language code
            sample_rate: Audio sample rate
        """
        self.ws_url = ws_url
        self.voice = voice
        self.language = language
        self.sample_rate = sample_rate
        self.websocket = None
        self.output_audio = []
        self.transcriptions = []

    async def connect(self):
        """Connect to Pipecat server."""
        url = f"{self.ws_url}?voice={self.voice}&language={self.language}&sample_rate={self.sample_rate}"
        logger.info(f"Connecting to {url}")

        self.websocket = await websockets.connect(
            url,
            max_size=10 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        )
        logger.info("Connected to Pipecat server")

    async def disconnect(self):
        """Disconnect from server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info("Disconnected from server")

    async def send_audio(self, audio_data: bytes, chunk_duration_ms: int = 100):
        """
        Send audio data to server in chunks.

        Args:
            audio_data: Raw PCM16 audio bytes
            chunk_duration_ms: Chunk duration in milliseconds
        """
        if not self.websocket:
            raise RuntimeError("Not connected")

        # Calculate chunk size
        chunk_samples = int(self.sample_rate * chunk_duration_ms / 1000)
        chunk_bytes = chunk_samples * 2  # 16-bit = 2 bytes per sample

        logger.info(f"Sending {len(audio_data)} bytes of audio in {chunk_duration_ms}ms chunks")

        offset = 0
        chunks_sent = 0

        while offset < len(audio_data):
            chunk = audio_data[offset:offset + chunk_bytes]
            await self.websocket.send(chunk)
            chunks_sent += 1
            offset += chunk_bytes

            # Simulate real-time streaming
            await asyncio.sleep(chunk_duration_ms / 1000.0 * 0.5)

        logger.info(f"Sent {chunks_sent} audio chunks")

    async def receive_responses(self, timeout: float = 30.0):
        """
        Receive responses from server.

        Args:
            timeout: Maximum time to wait for responses
        """
        if not self.websocket:
            raise RuntimeError("Not connected")

        logger.info("Listening for responses...")
        start_time = time.time()

        try:
            while time.time() - start_time < timeout:
                try:
                    msg = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=5.0
                    )

                    # Handle binary (audio) data
                    if isinstance(msg, bytes):
                        self.output_audio.append(msg)
                        logger.debug(f"Received {len(msg)} bytes of audio")

                    # Handle text (JSON) messages
                    else:
                        try:
                            data = json.loads(msg)
                            await self._handle_message(data)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON: {msg[:100]}")

                except asyncio.TimeoutError:
                    # Check if we've received any data
                    if self.output_audio or self.transcriptions:
                        logger.info("No more data, finishing")
                        break
                    continue

        except websockets.ConnectionClosed:
            logger.info("Connection closed by server")

    async def _handle_message(self, data: dict):
        """Handle a JSON message from server."""
        msg_type = data.get("type", "unknown")

        if msg_type == "transcription":
            text = data.get("text", "")
            is_final = data.get("is_final", False)
            status = "FINAL" if is_final else "INTERIM"
            logger.info(f"[{status}] {text}")

            if is_final:
                self.transcriptions.append(text)

        elif msg_type == "audio_start":
            logger.info("[AUDIO] TTS started")

        elif msg_type == "audio_stop":
            logger.info("[AUDIO] TTS stopped")

        elif msg_type == "bot_speaking":
            text = data.get("text", "")
            logger.info(f"[BOT] {text}")

        elif msg_type == "error":
            error = data.get("message", "Unknown error")
            logger.error(f"[ERROR] {error}")

        else:
            logger.debug(f"Unknown message type: {msg_type}")

    def save_output_audio(self, output_path: str):
        """Save received audio to WAV file."""
        if not self.output_audio:
            logger.warning("No audio received")
            return

        audio_data = b"".join(self.output_audio)
        logger.info(f"Saving {len(audio_data)} bytes of audio to {output_path}")

        # Create WAV file with proper header
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(24000)  # TTS output is 24kHz
            wf.writeframes(audio_data)

        logger.info(f"Saved output audio to {output_path}")


async def test_with_audio_file(
    client: PipecatTestClient,
    audio_path: str,
    output_path: str = "output.wav",
):
    """Test pipeline with audio file."""
    logger.info(f"Testing with audio file: {audio_path}")

    # Read audio file
    with wave.open(audio_path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)

    audio_duration = n_frames / sample_rate
    logger.info(f"Audio: {audio_duration:.2f}s, {sample_rate}Hz, {n_channels}ch, {sample_width*8}bit")

    # Convert to mono if stereo
    if n_channels == 2:
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
        audio_data = audio_array.tobytes()
        logger.info("Converted stereo to mono")

    # Resample if needed
    if sample_rate != client.sample_rate:
        logger.warning(f"Audio is {sample_rate}Hz, server expects {client.sample_rate}Hz")

    # Connect and test
    await client.connect()

    try:
        # Start receiving in background
        receive_task = asyncio.create_task(client.receive_responses(timeout=60.0))

        # Send audio
        await client.send_audio(audio_data)

        # Wait for responses
        await receive_task

        # Save output
        if client.output_audio:
            client.save_output_audio(output_path)

        # Print summary
        print("\n=== Test Summary ===")
        print(f"Input audio: {audio_duration:.2f}s")
        print(f"Transcriptions received: {len(client.transcriptions)}")
        for i, text in enumerate(client.transcriptions, 1):
            print(f"  {i}. {text}")
        print(f"Output audio: {len(client.output_audio)} chunks")
        if client.output_audio:
            print(f"Output saved to: {output_path}")

    finally:
        await client.disconnect()


async def test_with_text(
    ws_url: str,
    text: str,
    voice: str = "hi_male",
    output_path: str = "output.wav",
):
    """Test TTS directly with text input (bypasses STT)."""
    logger.info(f"Testing TTS with text: '{text}'")

    # For text-only test, we need a different endpoint or protocol
    # For now, we'll use the TTS server directly
    import aiohttp

    tts_url = os.getenv("SVARA_TTS_URL", "http://localhost:8080")

    async with aiohttp.ClientSession() as session:
        payload = {
            "prompt": text,
            "voice": voice,
            "temperature": 0.75,
            "top_p": 0.9,
            "max_tokens": 1500,
            "repetition_penalty": 1.1,
        }

        logger.info(f"Sending TTS request to {tts_url}")

        async with session.post(f"{tts_url}/v1/audio/text-to-speech", json=payload) as response:
            if response.status != 200:
                error = await response.text()
                logger.error(f"TTS failed: {response.status} - {error}")
                return

            audio_data = await response.read()
            logger.info(f"Received {len(audio_data)} bytes of audio")

            # Save to file
            with open(output_path, "wb") as f:
                f.write(audio_data)

            logger.info(f"Saved to {output_path}")


async def test_health(ws_url: str):
    """Test server health endpoint."""
    import aiohttp

    # Extract HTTP URL from WebSocket URL
    http_url = ws_url.replace("ws://", "http://").replace("wss://", "https://")
    http_url = http_url.split("/ws")[0]

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{http_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Server health: {data}")
                    return True
                else:
                    logger.error(f"Health check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False


async def test_config(ws_url: str):
    """Test server config endpoint."""
    import aiohttp

    http_url = ws_url.replace("ws://", "http://").replace("wss://", "https://")
    http_url = http_url.split("/ws")[0]

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{http_url}/config") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Server config:\n{json.dumps(data, indent=2)}")
                    return data
                else:
                    logger.error(f"Config check failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Config check error: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Test IdliDemo Pipecat Server")
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/ws",
        help="WebSocket URL (default: ws://localhost:8000/ws)"
    )
    parser.add_argument(
        "--audio",
        help="Audio file to stream (WAV format)"
    )
    parser.add_argument(
        "--text",
        help="Text to synthesize (bypasses STT)"
    )
    parser.add_argument(
        "--voice",
        default="hi_male",
        help="TTS voice ID (default: hi_male)"
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="STT language (default: auto)"
    )
    parser.add_argument(
        "--output",
        default="output.wav",
        help="Output audio file (default: output.wav)"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check server health"
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Get server configuration"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate (default: 16000)"
    )

    args = parser.parse_args()

    async def run():
        # Health check
        if args.health:
            await test_health(args.url)
            return

        # Config check
        if args.config:
            await test_config(args.url)
            return

        # Text-only TTS test
        if args.text:
            await test_with_text(
                ws_url=args.url,
                text=args.text,
                voice=args.voice,
                output_path=args.output,
            )
            return

        # Audio file test
        if args.audio:
            if not os.path.exists(args.audio):
                logger.error(f"Audio file not found: {args.audio}")
                sys.exit(1)

            client = PipecatTestClient(
                ws_url=args.url,
                voice=args.voice,
                language=args.language,
                sample_rate=args.sample_rate,
            )

            await test_with_audio_file(
                client=client,
                audio_path=args.audio,
                output_path=args.output,
            )
            return

        # No input specified
        logger.error("Please specify --audio, --text, --health, or --config")
        parser.print_help()
        sys.exit(1)

    asyncio.run(run())


if __name__ == "__main__":
    main()
