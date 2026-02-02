"""
Custom STT Service for IndicASR-Streaming WebSocket API with Barge-In Support.

This service integrates with the IndicASR-Streaming server which provides
real-time ASR for Indian languages via WebSocket streaming.

KEY FEATURES FOR BARGE-IN:
- Persistent WebSocket connection (always listening)
- Server-side VAD handles utterance segmentation
- No disconnect/reconnect between utterances
- Proper interruption handling

Server protocol:
1. Connect to WebSocket endpoint (once, at start)
2. Send config JSON: {"language": "auto", "interim_results": true, "sample_rate": 16000}
3. Receive config_ack with session_id
4. Stream PCM16 audio bytes continuously
5. Receive transcription results: interim, final, language
6. Connection stays open for continuous listening
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import websockets
from websockets.asyncio.client import connect as websocket_connect

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    StartFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)


@dataclass
class IndicASRConfig:
    """Configuration for IndicASR STT service."""
    language: str = "auto"
    interim_results: bool = True
    sample_rate: int = 16000


class IndicASRSTTService(FrameProcessor):
    """
    Custom STT service for IndicASR-Streaming WebSocket API with barge-in support.

    BARGE-IN ARCHITECTURE:
    - Maintains a PERSISTENT WebSocket connection to IndicASR
    - Audio is streamed continuously (no END signals between utterances)
    - Server-side VAD segments utterances and sends final transcriptions
    - When user interrupts during bot speech:
      1. StartInterruptionFrame cancels TTS/LLM
      2. STT continues listening (connection stays open)
      3. New user speech is processed immediately

    This processor:
    - Receives InputAudioRawFrame from the transport
    - Sends audio to IndicASR via persistent WebSocket
    - Emits TranscriptionFrame and InterimTranscriptionFrame
    - Handles interruptions without connection reset
    """

    def __init__(
        self,
        *,
        ws_url: str = "ws://localhost:8082/v1/audio/speech-to-text/stream",
        language: str = "auto",
        interim_results: bool = True,
        sample_rate: int = 16000,
        **kwargs
    ):
        """
        Initialize IndicASR STT service.

        Args:
            ws_url: WebSocket URL for IndicASR server
            language: Language code or "auto" for detection
            interim_results: Whether to emit interim transcriptions
            sample_rate: Audio sample rate (default 16000Hz)
        """
        super().__init__(**kwargs)

        self._ws_url = ws_url
        self._config = IndicASRConfig(
            language=language,
            interim_results=interim_results,
            sample_rate=sample_rate
        )

        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._session_id: Optional[str] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._connected = False
        self._connecting = False  # Lock to prevent concurrent connection attempts
        self._connect_lock = asyncio.Lock()
        self._detected_language: Optional[str] = None

        # Barge-in state
        self._user_speaking = False
        self._bot_speaking = False
        self._interrupted = False
        self._muted = False  # Mute STT output during bot speech (optional)
        self._keepalive_task: Optional[asyncio.Task] = None

    async def start(self, frame: StartFrame):
        """Start the STT service and establish persistent connection."""
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the STT service and close connection."""
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel - but keep connection open for barge-in."""
        # Don't disconnect on cancel - we want to keep listening
        self._interrupted = True

    async def _connect(self):
        """Establish persistent WebSocket connection to IndicASR server."""
        if self._connected:
            return

        # Prevent concurrent connection attempts
        if self._connecting:
            logger.debug("Connection already in progress, skipping")
            return

        self._connecting = True
        try:
            logger.info(f"Connecting to IndicASR at {self._ws_url} (persistent connection)")

            self._websocket = await websocket_connect(
                self._ws_url,
                max_size=10 * 1024 * 1024,
                ping_interval=30,  # Send ping every 30s
                ping_timeout=60,   # Wait 60s for pong response
                close_timeout=10,  # Wait 10s for close handshake
            )

            # Send configuration
            config = {
                "language": self._config.language,
                "interim_results": self._config.interim_results,
                "sample_rate": self._config.sample_rate,
            }
            await self._websocket.send(json.dumps(config))
            logger.debug(f"Sent config: {config}")

            # Wait for acknowledgment
            ack_msg = await asyncio.wait_for(self._websocket.recv(), timeout=10.0)
            ack_data = json.loads(ack_msg)

            if ack_data.get("type") != "config_ack":
                raise RuntimeError(f"Unexpected response: {ack_data}")

            self._session_id = ack_data.get("session_id")
            self._connected = True

            logger.info(f"Connected to IndicASR (persistent), session_id: {self._session_id}")

            # Start receiving messages in background
            self._receive_task = asyncio.create_task(self._receive_messages())

        except Exception as e:
            logger.error(f"Failed to connect to IndicASR: {e}")
            await self.push_frame(ErrorFrame(error=f"IndicASR connection failed: {e}"))
            raise
        finally:
            self._connecting = False

    async def _disconnect(self):
        """Disconnect from IndicASR server."""
        if not self._connected:
            return

        try:
            # Cancel receive task
            if self._receive_task and not self._receive_task.done():
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass

            # Close WebSocket
            if self._websocket:
                await self._websocket.close()

        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
        finally:
            self._websocket = None
            self._session_id = None
            self._connected = False
            self._receive_task = None
            logger.info("Disconnected from IndicASR")

    async def _reconnect(self):
        """Reconnect to IndicASR (used for error recovery, not normal operation)."""
        logger.info("Reconnecting to IndicASR...")
        await self._disconnect()
        await asyncio.sleep(0.1)
        await self._connect()

    async def _receive_messages(self):
        """Continuously receive and process messages from IndicASR."""
        try:
            while self._connected and self._websocket:
                try:
                    msg = await self._websocket.recv()
                    data = json.loads(msg)
                    await self._handle_message(data)
                except websockets.ConnectionClosed as e:
                    logger.warning(f"IndicASR WebSocket connection closed: {e}")
                    # Don't auto-reconnect here - let _process_audio handle it
                    # This avoids creating a new session that loses audio context
                    self._connected = False
                    self._websocket = None
                    break
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from IndicASR: {e}")
                except Exception as e:
                    logger.error(f"Error receiving from IndicASR: {e}")
                    self._connected = False
                    self._websocket = None
                    break
        except asyncio.CancelledError:
            pass
        finally:
            self._connected = False
            self._websocket = None

    async def _handle_message(self, data: dict):
        """Handle a message from IndicASR server."""
        msg_type = data.get("type", "unknown")

        if msg_type == "language":
            # Language detection result
            self._detected_language = data.get("language")
            probability = data.get("probability", 0)
            logger.info(f"IndicASR detected language: {self._detected_language} ({probability:.2f})")

        elif msg_type == "interim":
            # Interim transcription (partial result)
            text = data.get("text", "").strip()
            if text and self._config.interim_results and not self._muted:
                logger.debug(f"IndicASR interim: {text[:50]}...")
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text=text,
                        user_id="",
                        timestamp="",
                        language=self._detected_language or self._config.language,
                    )
                )

        elif msg_type == "final":
            # Final transcription for a segment
            text = data.get("text", "").strip()
            if text and not self._muted:
                start = data.get("start", 0.0)
                end = data.get("end", 0.0)
                logger.info(f"IndicASR final [{start:.1f}s-{end:.1f}s]: {text}")

                # Clear interrupted flag - we have a new valid transcription
                self._interrupted = False

                await self.push_frame(
                    TranscriptionFrame(
                        text=text,
                        user_id="",
                        timestamp="",
                        language=self._detected_language or self._config.language,
                    )
                )

        elif msg_type == "done":
            # Session complete (only happens if END was sent)
            # In persistent mode, we don't send END between utterances
            full_text = data.get("text", "").strip()
            language = data.get("language")
            logger.info(f"IndicASR session done: lang={language}, text='{full_text[:50]}...'")

            # Reconnect for next session
            await self._reconnect()

        elif msg_type == "error":
            # Error from server
            error_msg = data.get("message", "Unknown error")
            logger.error(f"IndicASR error: {error_msg}")
            await self.push_frame(ErrorFrame(error=f"IndicASR error: {error_msg}"))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames with barge-in support."""
        # Debug: Log all frame types to trace flow (except audio frames which are too frequent)
        if not isinstance(frame, InputAudioRawFrame):
            logger.debug(f"STT received frame: {type(frame).__name__}")

        await super().process_frame(frame, direction)

        # === Lifecycle Frames ===
        if isinstance(frame, StartFrame):
            await self.start(frame)
            await self.push_frame(frame, direction)

        elif isinstance(frame, EndFrame):
            await self.stop(frame)
            await self.push_frame(frame, direction)

        elif isinstance(frame, CancelFrame):
            await self.cancel(frame)
            await self.push_frame(frame, direction)

        # === Interruption Frames (KEY FOR BARGE-IN) ===
        elif isinstance(frame, StartInterruptionFrame):
            # User interrupted the bot - DON'T disconnect!
            # Just mark that we're in an interrupted state
            logger.info(">>> BARGE-IN: User interrupted bot speech")
            self._interrupted = True
            self._bot_speaking = False
            # Forward the interruption to cancel TTS/LLM
            await self.push_frame(frame, direction)

        # === Bot Speaking State (for optional STT muting) ===
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            # Start keepalive task to prevent WebSocket timeout during bot speech
            if self._keepalive_task is None or self._keepalive_task.done():
                self._keepalive_task = asyncio.create_task(self._keepalive_loop())
            await self.push_frame(frame, direction)

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._muted = False
            # Stop keepalive task
            if self._keepalive_task and not self._keepalive_task.done():
                self._keepalive_task.cancel()
                try:
                    await self._keepalive_task
                except asyncio.CancelledError:
                    pass
            self._keepalive_task = None
            await self.push_frame(frame, direction)

        # === User Speaking State ===
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
            # Ensure we're connected (should already be)
            if not self._connected:
                await self._connect()
            await self.push_frame(frame, direction)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            # User stopped speaking - send FINALIZE to force immediate transcription
            # This syncs client-side VAD with server transcription timing
            self._user_speaking = False
            logger.info("User stopped speaking - sending FINALIZE to IndicASR")

            # Send FINALIZE command to force immediate transcription
            if self._connected and self._websocket:
                try:
                    await self._websocket.send("FINALIZE")
                    logger.debug("FINALIZE sent successfully")
                except websockets.ConnectionClosed as e:
                    logger.warning(f"Connection closed when sending FINALIZE: {e}")
                    self._connected = False
                    self._websocket = None
                except Exception as e:
                    logger.warning(f"Failed to send FINALIZE: {e}")
            else:
                logger.warning("Cannot send FINALIZE - not connected to IndicASR")

            await self.push_frame(frame, direction)

        # === Audio Frames ===
        elif isinstance(frame, InputAudioRawFrame):
            # Always process audio - we're always listening
            await self._process_audio(frame)
            # Don't forward audio frames downstream

        # === Other Frames ===
        else:
            await self.push_frame(frame, direction)

    async def _process_audio(self, frame: InputAudioRawFrame):
        """Process audio frame and send to IndicASR."""
        if not self._connected:
            await self._connect()

        if not self._connected or not self._websocket:
            return

        try:
            # Get audio data
            audio = frame.audio

            # Convert to bytes if necessary
            if isinstance(audio, np.ndarray):
                if audio.dtype == np.float32 or audio.dtype == np.float64:
                    audio = (audio * 32767).astype(np.int16)
                elif audio.dtype != np.int16:
                    audio = audio.astype(np.int16)
                audio_bytes = audio.tobytes()
            elif isinstance(audio, bytes):
                audio_bytes = audio
            else:
                logger.warning(f"Unknown audio type: {type(audio)}")
                return

            # Send audio to IndicASR
            await self._websocket.send(audio_bytes)

        except websockets.ConnectionClosed:
            logger.warning("IndicASR connection closed during send, reconnecting...")
            self._connected = False
            await self._reconnect()
        except Exception as e:
            logger.error(f"Error sending audio to IndicASR: {e}")

    async def _keepalive_loop(self):
        """Send periodic silent audio to keep WebSocket connection alive during bot speech."""
        # Create a small silent audio buffer (100ms of silence at 16kHz)
        silence_samples = int(0.1 * self._config.sample_rate)  # 100ms
        silence = np.zeros(silence_samples, dtype=np.int16)
        silence_bytes = silence.tobytes()

        try:
            while self._bot_speaking and self._connected and self._websocket:
                try:
                    # Send silent audio every 5 seconds to keep connection alive
                    await asyncio.sleep(5.0)
                    if self._connected and self._websocket and self._bot_speaking:
                        await self._websocket.send(silence_bytes)
                        logger.debug("Sent keepalive silence to IndicASR")
                except websockets.ConnectionClosed:
                    logger.warning("Connection closed during keepalive")
                    self._connected = False
                    self._websocket = None
                    break
                except Exception as e:
                    logger.warning(f"Keepalive error: {e}")
                    break
        except asyncio.CancelledError:
            pass
