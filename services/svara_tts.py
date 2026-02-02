"""
Custom TTS Service for Svara-TTS-FastAPI with Barge-In Support.

This service integrates with the Svara TTS server which provides
high-performance multilingual TTS with vLLM backend.

KEY FEATURES FOR BARGE-IN:
- Immediate cancellation when user interrupts
- Proper cleanup of WebSocket/HTTP connections on interruption
- State tracking for interruption handling

Server endpoints:
- POST /v1/audio/text-to-speech/ - Non-streaming TTS (returns complete WAV)
- WEBSOCKET /v1/audio/text-to-speech/stream - Streaming TTS via WebSocket
- GET /v1/audio/text-to-speech/voices - List available voices
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import aiohttp
import websockets
from websockets.asyncio.client import connect as websocket_connect

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService

logger = logging.getLogger(__name__)

# WAV header size (44 bytes standard)
WAV_HEADER_SIZE = 44


@dataclass
class SvaraTTSConfig:
    """Configuration for Svara TTS service."""
    voice: str = "hi_male"
    temperature: float = 0.75
    top_p: float = 0.9
    max_tokens: int = 1500
    repetition_penalty: float = 1.1


class SvaraTTSService(TTSService):
    """
    Custom TTS service for Svara-TTS-FastAPI with barge-in support.

    Connects to Svara TTS server for multilingual text-to-speech synthesis.
    Supports streaming (WebSocket) and non-streaming (HTTP) modes.

    BARGE-IN SUPPORT:
    - Tracks interruption state to stop audio generation immediately
    - Properly cleans up WebSocket/HTTP connections on interruption
    - Handles StartInterruptionFrame to abort current TTS
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080",
        voice: str = "hi_male",
        temperature: float = 0.75,
        top_p: float = 0.9,
        max_tokens: int = 1500,
        repetition_penalty: float = 1.1,
        streaming: bool = True,
        sample_rate: int = 24000,
        **kwargs
    ):
        """
        Initialize Svara TTS service.

        Args:
            base_url: Base URL for Svara TTS server (http://host:port)
            voice: Voice ID (e.g., 'hi_male', 'hi_female', 'en_male')
            temperature: Sampling temperature (0.0-1.0)
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            repetition_penalty: Repetition penalty
            streaming: Whether to use WebSocket streaming endpoint
            sample_rate: Output audio sample rate
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._base_url = base_url.rstrip("/")
        # Convert http to ws for WebSocket URL
        self._ws_url = self._base_url.replace("http://", "ws://").replace("https://", "wss://")
        self._streaming = streaming
        self._config = SvaraTTSConfig(
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
        )

        self._session: Optional[aiohttp.ClientSession] = None
        self._current_websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._current_request_task: Optional[asyncio.Task] = None

        # Barge-in state
        self._interrupted = False
        self._generating = False

    @property
    def voice(self) -> str:
        """Get current voice."""
        return self._config.voice

    async def set_voice(self, voice: str):
        """Set the voice for TTS."""
        self._config.voice = voice
        logger.info(f"Svara TTS voice set to: {voice}")

    async def start(self, frame: StartFrame):
        """Start the TTS service."""
        await super().start(frame)
        self._session = aiohttp.ClientSession()
        logger.info(f"Svara TTS service started, base_url: {self._base_url}")

    async def stop(self, frame: EndFrame):
        """Stop the TTS service."""
        # Close any active WebSocket
        if self._current_websocket:
            try:
                await self._current_websocket.close()
            except Exception:
                pass
            self._current_websocket = None

        if self._session:
            await self._session.close()
            self._session = None
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel current TTS generation."""
        logger.info(">>> TTS CANCEL: Stopping audio generation")
        self._interrupted = True

        # Close WebSocket if active
        if self._current_websocket:
            try:
                await self._current_websocket.close()
            except Exception:
                pass
            self._current_websocket = None

        if self._current_request_task and not self._current_request_task.done():
            self._current_request_task.cancel()
            try:
                await self._current_request_task
            except asyncio.CancelledError:
                pass
        await super().cancel(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with barge-in support."""
        # Handle interruption frames for barge-in
        if isinstance(frame, StartInterruptionFrame):
            logger.info(">>> TTS BARGE-IN: User interrupted, stopping audio")
            self._interrupted = True

            # Close WebSocket immediately
            if self._current_websocket:
                try:
                    await self._current_websocket.close()
                except Exception:
                    pass
                self._current_websocket = None

            # Forward the frame but also handle it
            await self.push_frame(frame, direction)
            return

        # Let parent handle other frames (including text for TTS)
        await super().process_frame(frame, direction)

    def can_generate_metrics(self) -> bool:
        """Return whether this service supports metrics generation."""
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Synthesize text to speech with barge-in support.

        This is the core method required by TTSService base class.
        Yields audio frames as they are generated.
        Checks for interruption flag to abort early on barge-in.

        Args:
            text: Text to synthesize

        Yields:
            TTSStartedFrame, TTSAudioRawFrame chunks, TTSStoppedFrame
        """
        if not text.strip():
            return

        # Check if already interrupted before starting
        if self._interrupted:
            logger.info("TTS skipped - already interrupted")
            return

        if not self._session:
            self._session = aiohttp.ClientSession()

        logger.info(f"TTS INPUT TEXT: '{text}' [voice={self._config.voice}]")

        # Reset interrupted flag at start of new generation
        self._interrupted = False
        self._generating = True

        # Signal TTS started
        yield TTSStartedFrame()

        try:
            if self._streaming:
                async for frame in self._run_streaming_tts_websocket(text):
                    # Check for interruption between chunks
                    if self._interrupted:
                        logger.info("TTS interrupted during streaming - stopping")
                        break
                    yield frame
            else:
                async for frame in self._run_non_streaming_tts(text):
                    if self._interrupted:
                        logger.info("TTS interrupted during output - stopping")
                        break
                    yield frame

        except asyncio.CancelledError:
            logger.info("Svara TTS generation cancelled")
            raise
        except Exception as e:
            logger.error(f"Svara TTS error: {e}")
            yield ErrorFrame(f"Svara TTS error: {e}")
        finally:
            self._generating = False
            self._current_websocket = None
            # Signal TTS stopped
            yield TTSStoppedFrame()

    async def _run_streaming_tts_websocket(self, text: str) -> AsyncGenerator[Frame, None]:
        """Run streaming TTS via WebSocket with interruption support."""
        ws_url = f"{self._ws_url}/v1/audio/text-to-speech/stream"

        config = {
            "text": text,
            "voice": self._config.voice,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "max_tokens": self._config.max_tokens,
            "repetition_penalty": self._config.repetition_penalty,
        }

        try:
            self._current_websocket = await websocket_connect(
                ws_url,
                max_size=10 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=20,
            )

            # Send config
            await self._current_websocket.send(json.dumps(config))

            # Wait for acknowledgment
            ack_msg = await asyncio.wait_for(self._current_websocket.recv(), timeout=10.0)
            ack_data = json.loads(ack_msg)

            if ack_data.get("type") == "error":
                raise RuntimeError(ack_data.get("message", "Unknown error"))

            if ack_data.get("type") != "config_ack":
                raise RuntimeError(f"Unexpected response: {ack_data}")

            logger.debug(f"TTS WebSocket connected, session: {ack_data.get('session_id', 'N/A')}")

            # Receive audio chunks
            header_stripped = False
            while True:
                if self._interrupted:
                    logger.info("TTS WebSocket interrupted - closing")
                    break

                try:
                    msg = await asyncio.wait_for(self._current_websocket.recv(), timeout=30.0)

                    if isinstance(msg, bytes):
                        # Audio data
                        chunk = msg

                        # Strip WAV header from first chunk
                        if not header_stripped:
                            if len(chunk) > WAV_HEADER_SIZE:
                                chunk = chunk[WAV_HEADER_SIZE:]
                            header_stripped = True

                        if chunk:
                            yield TTSAudioRawFrame(
                                audio=chunk,
                                sample_rate=self._sample_rate,
                                num_channels=1,
                            )
                    else:
                        # JSON message
                        data = json.loads(msg)
                        msg_type = data.get("type")

                        if msg_type == "done":
                            logger.debug(f"TTS done: {data.get('audio_duration', 0):.2f}s")
                            break
                        elif msg_type == "error":
                            raise RuntimeError(data.get("message", "Unknown error"))

                except asyncio.TimeoutError:
                    logger.warning("TTS WebSocket timeout")
                    break

        except websockets.ConnectionClosed as e:
            if not self._interrupted:
                logger.warning(f"TTS WebSocket closed unexpectedly: {e}")
        except Exception as e:
            raise RuntimeError(f"TTS WebSocket error: {e}")
        finally:
            if self._current_websocket:
                try:
                    await self._current_websocket.close()
                except Exception:
                    pass
                self._current_websocket = None

    async def _run_non_streaming_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Run non-streaming TTS request with interruption support."""
        url = f"{self._base_url}/v1/audio/text-to-speech/"

        payload = {
            "prompt": text,
            "voice": self._config.voice,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "max_tokens": self._config.max_tokens,
            "repetition_penalty": self._config.repetition_penalty,
        }

        try:
            async with self._session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {error_text}")

                # Check for interruption before reading
                if self._interrupted:
                    logger.info("TTS interrupted before reading response")
                    return

                # Read complete response
                audio_data = await response.read()

                # Strip WAV header
                if len(audio_data) > WAV_HEADER_SIZE:
                    audio_data = audio_data[WAV_HEADER_SIZE:]

                # Yield as chunks, checking for interruption
                chunk_size = 4096
                for i in range(0, len(audio_data), chunk_size):
                    if self._interrupted:
                        logger.info("TTS interrupted during chunk output")
                        break
                    chunk = audio_data[i:i + chunk_size]
                    yield TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=self._sample_rate,
                        num_channels=1,
                    )

        except aiohttp.ClientError as e:
            raise RuntimeError(f"HTTP request failed: {e}")

    async def list_voices(self) -> list:
        """List available voices from the server."""
        if not self._session:
            self._session = aiohttp.ClientSession()

        url = f"{self._base_url}/v1/audio/text-to-speech/voices"

        try:
            async with self._session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to list voices: HTTP {response.status}")
                    return []

                data = await response.json()
                return data.get("voices", [])

        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            return []
