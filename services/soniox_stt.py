"""
Custom STT Service for Soniox Realtime WebSocket API with Speaker Diarization.

This service integrates with the Soniox realtime STT server which provides
multilingual ASR with speaker diarization via WebSocket streaming.

KEY FEATURES:
- Persistent WebSocket connection (always listening)
- Language identification across 50+ languages
- Speaker diarization (identify different speakers)
- Endpoint detection for utterance segmentation
- Barge-in support

Server protocol:
1. Connect to WebSocket endpoint: wss://stt-rt.soniox.com/transcribe-websocket
2. Send config JSON with API key immediately
3. Stream PCM16 audio bytes continuously (16kHz, mono, 16-bit signed LE)
4. Receive transcription responses with words array
5. On UserStoppedSpeakingFrame, send "" (empty string) to signal end of audio
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List

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

# Soniox WebSocket endpoint
SONIOX_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"


@dataclass
class SonioxConfig:
    """Configuration for Soniox STT service."""
    api_key: str
    model: str = "stt-rt-v3"
    sample_rate: int = 16000
    num_channels: int = 1
    include_nonfinal: bool = True
    enable_speaker_diarization: bool = True
    enable_language_identification: bool = True
    enable_endpoint_detection: bool = True
    language_hints: List[str] = field(default_factory=lambda: ["en", "hi"])


class SonioxSTTService(FrameProcessor):
    """
    Custom STT service for Soniox Realtime WebSocket API with barge-in support.

    BARGE-IN ARCHITECTURE:
    - Maintains a PERSISTENT WebSocket connection to Soniox
    - Audio is streamed continuously
    - Server-side VAD/endpoint detection segments utterances
    - When user interrupts during bot speech:
      1. StartInterruptionFrame cancels TTS/LLM
      2. STT continues listening (connection stays open)
      3. New user speech is processed immediately

    This processor:
    - Receives InputAudioRawFrame from the transport
    - Sends audio to Soniox via persistent WebSocket
    - Emits TranscriptionFrame and InterimTranscriptionFrame
    - Includes speaker labels when diarization is enabled
    - Handles interruptions without connection reset
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "stt-rt-v3",
        sample_rate: int = 16000,
        num_channels: int = 1,
        language_hints: Optional[List[str]] = None,
        include_nonfinal: bool = True,
        enable_speaker_diarization: bool = True,
        enable_language_identification: bool = True,
        enable_endpoint_detection: bool = True,
        **kwargs
    ):
        """
        Initialize Soniox STT service.

        Args:
            api_key: Soniox API key
            model: Soniox model name (default: stt-rt-v3)
            sample_rate: Audio sample rate (default 16000Hz)
            num_channels: Number of audio channels (default 1 for mono)
            language_hints: List of language codes to prioritize for detection
            include_nonfinal: Enable interim transcriptions
            enable_speaker_diarization: Enable speaker identification
            enable_language_identification: Enable language detection
            enable_endpoint_detection: Enable endpoint detection for utterance segmentation
        """
        super().__init__(**kwargs)

        if language_hints is None:
            language_hints = ["en", "hi"]

        self._config = SonioxConfig(
            api_key=api_key,
            model=model,
            sample_rate=sample_rate,
            num_channels=num_channels,
            include_nonfinal=include_nonfinal,
            enable_speaker_diarization=enable_speaker_diarization,
            enable_language_identification=enable_language_identification,
            enable_endpoint_detection=enable_endpoint_detection,
            language_hints=language_hints,
        )

        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._connected = False
        self._connecting = False
        self._connect_lock = asyncio.Lock()

        # Transcription state
        self._current_text = ""
        self._current_speaker: Optional[int] = None
        self._detected_language: Optional[str] = None

        # Barge-in state
        self._user_speaking = False
        self._bot_speaking = False
        self._interrupted = False
        self._muted = False
        self._keepalive_task: Optional[asyncio.Task] = None

    async def start(self, frame: StartFrame):
        """Start the STT service. Connection is lazy - established when user speaks."""
        # Don't connect here - wait for user to start speaking
        # This prevents Soniox from timing out idle connections
        logger.info("Soniox STT service started (lazy connection mode)")

    async def stop(self, frame: EndFrame):
        """Stop the STT service and close connection."""
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel - but keep connection open for barge-in."""
        self._interrupted = True

    async def _connect(self):
        """Establish persistent WebSocket connection to Soniox server."""
        if self._connected:
            return

        if self._connecting:
            logger.debug("Connection already in progress, skipping")
            return

        self._connecting = True
        try:
            logger.info(f"Connecting to Soniox at {SONIOX_WS_URL}")

            self._websocket = await websocket_connect(
                SONIOX_WS_URL,
                max_size=10 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
            )

            # Send configuration with API key - Soniox realtime format
            config = {
                "api_key": self._config.api_key,
                "model": self._config.model,
                "language_hints": self._config.language_hints,
                "enable_language_identification": self._config.enable_language_identification,
                "enable_speaker_diarization": self._config.enable_speaker_diarization,
                "enable_endpoint_detection": self._config.enable_endpoint_detection,
                "audio_format": "pcm_s16le",
                "sample_rate": self._config.sample_rate,
                "num_channels": self._config.num_channels,
            }

            # Log config being sent (without API key)
            config_log = {k: v for k, v in config.items() if k != "api_key"}
            logger.info(f"Soniox config: {json.dumps(config_log)}")

            await self._websocket.send(json.dumps(config))
            logger.info(f"Sent Soniox config: model={self._config.model}, include_nonfinal={self._config.include_nonfinal}")

            # Wait for initial response/acknowledgment from Soniox
            try:
                init_response = await asyncio.wait_for(self._websocket.recv(), timeout=5.0)
                logger.info(f"Soniox initial response: {init_response}")
            except asyncio.TimeoutError:
                logger.warning("No initial response from Soniox (continuing anyway)")
            except Exception as e:
                logger.warning(f"Error receiving initial response: {e}")

            self._connected = True
            logger.info("Connected to Soniox")

            # Start receiving messages in background
            self._receive_task = asyncio.create_task(self._receive_messages())

        except Exception as e:
            logger.error(f"Failed to connect to Soniox: {e}")
            await self.push_frame(ErrorFrame(error=f"Soniox connection failed: {e}"))
            raise
        finally:
            self._connecting = False

    async def _disconnect(self):
        """Disconnect from Soniox server."""
        if not self._connected:
            return

        try:
            if self._receive_task and not self._receive_task.done():
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass

            if self._keepalive_task and not self._keepalive_task.done():
                self._keepalive_task.cancel()
                try:
                    await self._keepalive_task
                except asyncio.CancelledError:
                    pass

            if self._websocket:
                await self._websocket.close()

        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
        finally:
            self._websocket = None
            self._connected = False
            self._receive_task = None
            self._keepalive_task = None
            logger.info("Disconnected from Soniox")

    async def _reconnect(self):
        """Reconnect to Soniox (used for error recovery)."""
        logger.info("Reconnecting to Soniox...")
        await self._disconnect()
        await asyncio.sleep(0.5)
        await self._connect()

    async def _receive_messages(self):
        """Continuously receive and process messages from Soniox."""
        try:
            while self._connected and self._websocket:
                try:
                    msg = await self._websocket.recv()
                    data = json.loads(msg)
                    await self._handle_message(data)
                except websockets.ConnectionClosed as e:
                    logger.warning(f"Soniox WebSocket connection closed: {e}")
                    self._connected = False
                    self._websocket = None
                    break
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from Soniox: {e}")
                except Exception as e:
                    logger.error(f"Error receiving from Soniox: {e}")
                    self._connected = False
                    self._websocket = None
                    break
        except asyncio.CancelledError:
            pass
        finally:
            self._connected = False
            self._websocket = None

    async def _handle_message(self, data: dict):
        """Handle a message from Soniox server."""
        # Check for error
        if "error" in data:
            error_msg = data.get("error", "Unknown error")
            logger.error(f"Soniox error: {error_msg}")
            await self.push_frame(ErrorFrame(error=f"Soniox error: {error_msg}"))
            return

        # Soniox response format:
        # {"tokens": [{"text": "hello", "start_ms": 0, "duration_ms": 500, "is_final": true, "speaker": 0}], ...}

        tokens = data.get("tokens", [])
        if not tokens:
            # Log raw response for debugging
            logger.debug(f"Soniox response (no tokens): {data}")
            return

        # Build text from tokens
        text_parts = []
        speaker = None
        is_final = False

        for token in tokens:
            token_text = token.get("text", "")
            if token_text:
                text_parts.append(token_text)

            # Track if any token is final
            if token.get("is_final", False):
                is_final = True

            # Track speaker if diarization is enabled
            if "speaker" in token and self._config.enable_speaker_diarization:
                speaker = token["speaker"]

        text = "".join(text_parts).strip()  # No space - tokens may include spaces
        # Remove Soniox end token
        text = text.replace("<end>", "").strip()
        if not text:
            return

        # Format text with speaker label if diarization is enabled
        formatted_text = text
        if speaker is not None and self._config.enable_speaker_diarization:
            formatted_text = f"Speaker {speaker}: {text}"
            self._current_speaker = speaker

        # Check fin_audio_proc (final audio processed) or is_final flag
        if data.get("fin_audio_proc", False) or is_final:
            if not self._muted:
                logger.info(f"Soniox final: {formatted_text}")
                self._interrupted = False
                self._current_text = ""

                await self.push_frame(
                    TranscriptionFrame(
                        text=formatted_text,
                        user_id="",
                        timestamp="",
                        language=self._detected_language or "en",
                    )
                )
        elif not self._muted and self._config.include_nonfinal:
            # Interim result
            logger.debug(f"Soniox interim: {formatted_text[:50]}...")
            self._current_text = formatted_text

            await self.push_frame(
                InterimTranscriptionFrame(
                    text=formatted_text,
                    user_id="",
                    timestamp="",
                    language=self._detected_language or "en",
                )
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames with barge-in support."""
        if not isinstance(frame, InputAudioRawFrame):
            logger.debug(f"Soniox STT received frame: {type(frame).__name__}")

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
            logger.info(">>> BARGE-IN: User interrupted bot speech (Soniox)")
            self._interrupted = True
            self._bot_speaking = False
            await self.push_frame(frame, direction)

        # === Bot Speaking State ===
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            if self._keepalive_task is None or self._keepalive_task.done():
                self._keepalive_task = asyncio.create_task(self._keepalive_loop())
            await self.push_frame(frame, direction)

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._muted = False
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
            # Connect to Soniox when user starts speaking (lazy connection)
            if not self._connected:
                logger.info("User started speaking - connecting to Soniox")
                await self._connect()
            await self.push_frame(frame, direction)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            logger.info("User stopped speaking - sending end signal to Soniox")

            # Send empty string to signal end of audio segment
            if self._connected and self._websocket:
                try:
                    # Send empty string to finalize current utterance
                    await self._websocket.send("")
                    logger.debug("End signal sent to Soniox")
                except websockets.ConnectionClosed as e:
                    logger.warning(f"Connection closed when sending end signal: {e}")
                    self._connected = False
                    self._websocket = None
                except Exception as e:
                    logger.warning(f"Failed to send end signal: {e}")
            else:
                logger.warning("Cannot send end signal - not connected to Soniox")

            await self.push_frame(frame, direction)

        # === Audio Frames ===
        elif isinstance(frame, InputAudioRawFrame):
            await self._process_audio(frame)
            # Don't forward audio frames downstream

        # === Other Frames ===
        else:
            await self.push_frame(frame, direction)

    async def _process_audio(self, frame: InputAudioRawFrame):
        """Process audio frame and send to Soniox."""
        # Only send audio if connected - connection is established on UserStartedSpeakingFrame
        if not self._connected or not self._websocket:
            return

        try:
            audio = frame.audio

            # Convert to bytes if necessary (Soniox expects 16-bit PCM, 16kHz, mono)
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

            # Send audio to Soniox
            await self._websocket.send(audio_bytes)
            logger.debug(f"Sent {len(audio_bytes)} bytes of audio to Soniox")

        except websockets.ConnectionClosed:
            # Don't reconnect immediately - will reconnect on next UserStartedSpeakingFrame
            logger.warning("Soniox connection closed during audio send")
            self._connected = False
            self._websocket = None
        except Exception as e:
            logger.error(f"Error sending audio to Soniox: {e}")

    async def _keepalive_loop(self):
        """Send periodic silent audio to keep WebSocket connection alive during bot speech."""
        silence_samples = int(0.1 * self._config.sample_rate)
        silence = np.zeros(silence_samples, dtype=np.int16)
        silence_bytes = silence.tobytes()

        try:
            while self._bot_speaking and self._connected and self._websocket:
                try:
                    await asyncio.sleep(1.0)  # Send keepalive every second
                    if self._connected and self._websocket and self._bot_speaking:
                        await self._websocket.send(silence_bytes)
                        logger.debug("Sent keepalive silence to Soniox")
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
