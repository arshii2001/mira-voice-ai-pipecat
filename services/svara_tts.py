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
import re
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional

import aiohttp
import numpy as np
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

# ---------------------------------------------------------------------------
# Text chunking for Svara long-form generation.
# Adapted from Sarvam's official svara-longform-generation notebook.
# Splits on clause/sentence boundaries with target/max/min size guards
# and merges tiny tail fragments.
# ---------------------------------------------------------------------------
_CLAUSE_SPLIT_RE = re.compile(r'(?<=[,.!?;:।؟\n])\s+')

# Chunk size parameters (characters).  Svara's max_tokens=350 generates
# at most ~4s of audio.  target=150 keeps most chunks in the 2-3s sweet
# spot; max=200 is the hard ceiling; min=50 prevents tiny fragments that
# produce pops or awkward prosody.
_CHUNK_TARGET = 120
_CHUNK_MAX = 200
_CHUNK_MIN = 50

# Crossfade duration in seconds between consecutive TTS chunks.
# Eliminates click/pop artifacts at chunk boundaries.
# Sarvam notebook uses 0.035-0.055s; we use 0.04s as a safe middle.
_CROSSFADE_SEC = 0.04


@dataclass
class SvaraTTSConfig:
    """Configuration for Svara TTS service."""
    voice: str = "en_female"
    temperature: float = 0.75
    top_p: float = 0.9
    max_tokens: int = 350  # Svara's per-request limit; long text is chunked by sentence
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
        api_key: str = None,
        voice: str = "en_female",
        temperature: float = 0.75,
        top_p: float = 0.9,
        max_tokens: int = 350,
        repetition_penalty: float = 1.1,
        streaming: bool = True,
        sample_rate: int = 24000,
        **kwargs
    ):
        """
        Initialize Svara TTS service.

        Args:
            base_url: Base URL for Svara TTS server (http://host:port)
            api_key: API key for Bearer authentication (optional)
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
        self._api_key = api_key
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

    def _auth_headers(self) -> dict:
        """Build auth headers for Svara gateways that expect different header names."""
        key = (self._api_key or "").strip()
        if not key:
            return {}

        bearer = key if key.lower().startswith("bearer ") else f"Bearer {key}"
        return {
            "Authorization": bearer,
            "x-api-key": key,
            "X-API-Key": key,
        }

    @staticmethod
    def _chunk_text(
        text: str,
        target_size: int = _CHUNK_TARGET,
        max_size: int = _CHUNK_MAX,
        min_size: int = _CHUNK_MIN,
    ) -> List[str]:
        """Split text into TTS-friendly chunks using Sarvam's proven algorithm.

        Adapted from the official svara-longform-generation notebook.
        Splits on clause/sentence boundaries (,.!?;:।) with:
          - target_size: soft stop — start a new chunk once buffer hits this
          - max_size: hard stop — never exceed this per chunk
          - min_size: merge tiny tail fragments back into the previous chunk

        Returns a list of text chunks ready for individual Svara TTS calls.
        """
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) <= target_size:
            return [text]

        parts = _CLAUSE_SPLIT_RE.split(text)

        chunks: List[str] = []
        buffer = ""

        for part in parts:
            if not part:
                continue

            if not buffer:
                buffer = part
                continue

            projected_len = len(buffer) + 1 + len(part)

            # Hard stop — never exceed max_size
            if projected_len > max_size:
                chunks.append(buffer)
                buffer = part
                continue

            # Soft stop — start new chunk once we hit target
            if len(buffer) >= target_size:
                chunks.append(buffer)
                buffer = part
                continue

            buffer += " " + part

        if buffer:
            chunks.append(buffer)

        # Merge tiny tail chunks back into the previous one
        final: List[str] = []
        for c in chunks:
            if final and len(c) < min_size and len(final[-1]) + 1 + len(c) <= max_size:
                final[-1] += " " + c
            else:
                final.append(c)

        if len(final) > 1:
            logger.info(
                f"TTS text chunked: {len(text)} chars → {len(final)} chunks "
                f"({[len(c) for c in final]})"
            )
        return final

    @staticmethod
    def _crossfade_pcm(
        a: bytes, b: bytes, fade_sec: float = _CROSSFADE_SEC, sample_rate: int = 24000
    ) -> bytes:
        """Crossfade two PCM16 audio buffers to eliminate click/pop artifacts.

        Uses a cosine crossfade (same as Sarvam's notebook) at the boundary
        between consecutive TTS chunks.

        Args:
            a: First PCM16 audio buffer (little-endian signed 16-bit)
            b: Second PCM16 audio buffer
            fade_sec: Duration of the crossfade in seconds
            sample_rate: Audio sample rate

        Returns:
            Combined PCM16 audio with smooth crossfade at the join point.
        """
        if not a or not b:
            return a + b

        fade_samples = int(fade_sec * sample_rate)

        # Convert PCM16 bytes → float arrays
        a_arr = np.frombuffer(a, dtype=np.int16).astype(np.float32)
        b_arr = np.frombuffer(b, dtype=np.int16).astype(np.float32)

        fade_samples = min(fade_samples, len(a_arr), len(b_arr))
        if fade_samples < 2:
            # Too short to crossfade — just concatenate
            return a + b

        # Cosine crossfade (same as Sarvam notebook)
        t = np.linspace(0, 1, fade_samples, endpoint=False)
        fade_out = np.cos(t * np.pi / 2)
        fade_in = np.sin(t * np.pi / 2)

        cross = a_arr[-fade_samples:] * fade_out + b_arr[:fade_samples] * fade_in

        result = np.concatenate([
            a_arr[:-fade_samples],
            cross,
            b_arr[fade_samples:],
        ])

        # Clip and convert back to PCM16
        result = np.clip(result, -32768, 32767).astype(np.int16)
        return result.tobytes()

    @staticmethod
    def _apply_fade_edges(
        pcm: bytes, fade_sec: float = _CROSSFADE_SEC, sample_rate: int = 24000
    ) -> bytes:
        """Apply fade-in at start and fade-out at end of PCM16 audio.

        This ensures each sentence's audio starts and ends at zero amplitude,
        eliminating click/pop artifacts when Pipecat concatenates consecutive
        sentences (each from a separate run_tts call).

        Uses cosine fade curves (same as Sarvam's notebook).
        """
        if not pcm or len(pcm) < 4:
            return pcm

        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        fade_samples = int(fade_sec * sample_rate)
        fade_samples = min(fade_samples, len(arr) // 4)  # Don't fade more than 25%

        if fade_samples < 2:
            return pcm

        # Fade-in: 0 → 1 (sine curve)
        t_in = np.linspace(0, 1, fade_samples, endpoint=False)
        arr[:fade_samples] *= np.sin(t_in * np.pi / 2)

        # Fade-out: 1 → 0 (cosine curve)
        t_out = np.linspace(0, 1, fade_samples, endpoint=False)
        arr[-fade_samples:] *= np.cos(t_out * np.pi / 2)

        arr = np.clip(arr, -32768, 32767).astype(np.int16)
        return arr.tobytes()

    @staticmethod
    def _apply_fade_in(
        pcm: bytes, fade_sec: float = _CROSSFADE_SEC, sample_rate: int = 24000
    ) -> bytes:
        """Apply fade-in only to the start of PCM16 audio (for streaming)."""
        if not pcm or len(pcm) < 4:
            return pcm
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        fade_samples = int(fade_sec * sample_rate)
        fade_samples = min(fade_samples, len(arr) // 2)
        if fade_samples < 2:
            return pcm
        t = np.linspace(0, 1, fade_samples, endpoint=False)
        arr[:fade_samples] *= np.sin(t * np.pi / 2)
        return np.clip(arr, -32768, 32767).astype(np.int16).tobytes()

    @staticmethod
    def _apply_fade_out(
        pcm: bytes, fade_sec: float = _CROSSFADE_SEC, sample_rate: int = 24000
    ) -> bytes:
        """Apply fade-out only to the end of PCM16 audio (for streaming)."""
        if not pcm or len(pcm) < 4:
            return pcm
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        fade_samples = int(fade_sec * sample_rate)
        fade_samples = min(fade_samples, len(arr) // 2)
        if fade_samples < 2:
            return pcm
        t = np.linspace(0, 1, fade_samples, endpoint=False)
        arr[-fade_samples:] *= np.cos(t * np.pi / 2)
        return np.clip(arr, -32768, 32767).astype(np.int16).tobytes()

    async def _collect_chunk_audio(self, text_chunk: str) -> bytes:
        """Synthesize a single text chunk and collect all PCM audio bytes."""
        pcm_bytes = b""
        if self._streaming:
            async for frame in self._run_streaming_tts_websocket(text_chunk):
                if self._interrupted:
                    break
                if isinstance(frame, TTSAudioRawFrame):
                    pcm_bytes += frame.audio
        else:
            async for frame in self._run_non_streaming_tts(text_chunk):
                if self._interrupted:
                    break
                if isinstance(frame, TTSAudioRawFrame):
                    pcm_bytes += frame.audio
        return pcm_bytes

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Synthesize text to speech with barge-in support.

        This is the core method required by TTSService base class.
        Pipecat's sentence aggregator usually calls this once per sentence,
        but may flush multiple sentences as one block when the LLM streams
        fast.  We chunk long text using Sarvam's proven algorithm and
        crossfade between chunks to eliminate pop/click artifacts.

        STREAMING STRATEGY:
        - Single chunk (most common): Stream audio as it arrives from Svara.
          Apply fade-in to the very first audio fragment, and fade-out to
          the last fragment, so each sentence starts/ends at zero amplitude.
          This gives near-instant first audio byte instead of waiting for
          the entire sentence to be synthesized.
        - Multi-chunk (long text): Collect + crossfade per chunk, then emit.

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

        # Chunk text using Sarvam's proven algorithm
        chunks = self._chunk_text(text)
        multi_chunk = len(chunks) > 1

        if not multi_chunk:
            # Single chunk — STREAM audio as it arrives from Svara.
            # Apply fade-in to the first audio bytes and fade-out to the last.
            import time as _time
            t_tts_start = _time.monotonic()
            logger.info(
                f"[SMOOTH] TTS start ({len(text)} chars): "
                f"'{text[:80]}{'...' if len(text) > 80 else ''}' "
                f"[voice={self._config.voice}]"
            )
            self._interrupted = False
            self._generating = True
            yield TTSStartedFrame()

            sr = self._sample_rate or 24000
            fade_samples = int(_CROSSFADE_SEC * sr)
            # We need fade_samples * 2 bytes (16-bit PCM) for fade-in
            fade_bytes = fade_samples * 2
            emit_chunk_size = 4096  # Transport chunk size

            first_chunk_emitted = False
            pending_tail = b""  # Buffer last fade_bytes for fade-out

            try:
                if self._streaming:
                    gen = self._run_streaming_tts_websocket(text)
                else:
                    gen = self._run_non_streaming_tts(text)

                async for frame in gen:
                    if self._interrupted:
                        break
                    if not isinstance(frame, TTSAudioRawFrame):
                        continue

                    pcm = frame.audio
                    if not pcm:
                        continue

                    if not first_chunk_emitted:
                        # Apply fade-in to the very first audio bytes
                        pcm = self._apply_fade_in(pcm, fade_sec=_CROSSFADE_SEC, sample_rate=sr)
                        first_chunk_emitted = True
                        ttfb_ms = (_time.monotonic() - t_tts_start) * 1000
                        logger.info(
                            f"[SMOOTH] TTS first audio byte at {ttfb_ms:.0f}ms "
                            f"({len(pcm)} bytes)"
                        )

                    # Buffer the tail for fade-out application later
                    combined = pending_tail + pcm
                    if len(combined) > fade_bytes:
                        # Emit everything except the tail buffer
                        to_emit = combined[:-fade_bytes]
                        pending_tail = combined[-fade_bytes:]

                        for i in range(0, len(to_emit), emit_chunk_size):
                            if self._interrupted:
                                break
                            yield TTSAudioRawFrame(
                                audio=to_emit[i:i + emit_chunk_size],
                                sample_rate=sr,
                                num_channels=1,
                            )
                    else:
                        pending_tail = combined

                # Flush the tail with fade-out applied
                if pending_tail and not self._interrupted:
                    pending_tail = self._apply_fade_out(
                        pending_tail, fade_sec=_CROSSFADE_SEC, sample_rate=sr
                    )
                    for i in range(0, len(pending_tail), emit_chunk_size):
                        if self._interrupted:
                            break
                        yield TTSAudioRawFrame(
                            audio=pending_tail[i:i + emit_chunk_size],
                            sample_rate=sr,
                            num_channels=1,
                        )

                total_ms = (_time.monotonic() - t_tts_start) * 1000
                logger.info(
                    f"[SMOOTH] TTS complete in {total_ms:.0f}ms "
                    f"(first byte at {ttfb_ms:.0f}ms)" if first_chunk_emitted
                    else f"[SMOOTH] TTS complete in {total_ms:.0f}ms (no audio)"
                )

            except asyncio.CancelledError:
                logger.info("Svara TTS generation cancelled")
                raise
            except Exception as e:
                logger.error(f"Svara TTS error: {e}")
                yield ErrorFrame(f"Svara TTS error: {e}")
            finally:
                self._generating = False
                self._current_websocket = None
                yield TTSStoppedFrame()
            return

        # Multi-chunk path: collect audio per chunk, crossfade, then emit
        logger.info(
            f"TTS multi-chunk: {len(text)} chars → {len(chunks)} chunks "
            f"({[len(c) for c in chunks]})"
        )

        self._interrupted = False
        self._generating = True
        yield TTSStartedFrame()

        try:
            combined_pcm = b""

            for i, chunk in enumerate(chunks):
                if self._interrupted:
                    logger.info(f"TTS skipped chunk {i+1}/{len(chunks)} - interrupted")
                    break

                logger.info(
                    f"TTS INPUT TEXT ({len(chunk)} chars, chunk {i+1}/{len(chunks)}): "
                    f"'{chunk[:60]}{'...' if len(chunk) > 60 else ''}' "
                    f"[voice={self._config.voice}]"
                )

                chunk_pcm = await self._collect_chunk_audio(chunk)

                if not chunk_pcm:
                    continue

                if combined_pcm:
                    # Crossfade with previous chunk to eliminate pops
                    combined_pcm = self._crossfade_pcm(
                        combined_pcm, chunk_pcm,
                        fade_sec=_CROSSFADE_SEC,
                        sample_rate=self._sample_rate or 24000,
                    )
                else:
                    combined_pcm = chunk_pcm

            # Apply fade edges to the combined audio, then emit as chunks
            if combined_pcm:
                combined_pcm = self._apply_fade_edges(
                    combined_pcm,
                    fade_sec=_CROSSFADE_SEC,
                    sample_rate=self._sample_rate or 24000,
                )
                chunk_size = 4096
                for i in range(0, len(combined_pcm), chunk_size):
                    if self._interrupted:
                        break
                    yield TTSAudioRawFrame(
                        audio=combined_pcm[i:i + chunk_size],
                        sample_rate=self._sample_rate or 24000,
                        num_channels=1,
                    )

        except asyncio.CancelledError:
            logger.info("Svara TTS generation cancelled")
            raise
        except Exception as e:
            logger.error(f"Svara TTS error: {e}")
            yield ErrorFrame(f"Svara TTS error: {e}")
        finally:
            self._generating = False
            self._current_websocket = None
            yield TTSStoppedFrame()

    async def _run_streaming_tts_websocket(self, text: str) -> AsyncGenerator[Frame, None]:
        """Run streaming TTS via WebSocket with interruption support."""
        import time as _time
        t0 = _time.monotonic()
        ws_url = f"{self._ws_url}/v1/audio/text-to-speech/stream"

        # Svara HTTP APIs use "prompt"; some WS deployments mirror that schema.
        # Send both keys for compatibility across versions.
        key = (self._api_key or "").strip()
        bearer = key if key.lower().startswith("bearer ") else (f"Bearer {key}" if key else "")
        config = {
            "type": "config",
            "prompt": text,
            "text": text,
            "voice": self._config.voice,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "max_tokens": self._config.max_tokens,
            "repetition_penalty": self._config.repetition_penalty,
            # Some gateways validate auth from the first message payload.
            "api_key": key,
            "token": key,
            "authorization": bearer,
        }

        try:
            # Include both Bearer and x-api-key for gateway compatibility.
            extra_headers = self._auth_headers()
            self._current_websocket = await websocket_connect(
                ws_url,
                max_size=10 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=20,
                additional_headers=extra_headers,
            )
            connect_ms = (_time.monotonic() - t0) * 1000
            logger.info(f"[SMOOTH] Svara WS connect: {connect_ms:.0f}ms")

            # Send config
            await self._current_websocket.send(json.dumps(config))

            # Wait for acknowledgment
            ack_msg = await asyncio.wait_for(self._current_websocket.recv(), timeout=10.0)
            ack_data = json.loads(ack_msg)

            if ack_data.get("type") == "error":
                raise RuntimeError(ack_data.get("message", "Unknown error"))

            if ack_data.get("type") != "config_ack":
                raise RuntimeError(f"Unexpected response: {ack_data}")

            ack_ms = (_time.monotonic() - t0) * 1000
            logger.info(
                f"[SMOOTH] Svara WS ready: {ack_ms:.0f}ms "
                f"(connect={connect_ms:.0f}ms + ack={ack_ms - connect_ms:.0f}ms)"
            )

            # Receive audio chunks
            header_stripped = False
            first_audio = True
            audio_chunks_received = 0
            total_audio_bytes = 0
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
                            audio_chunks_received += 1
                            total_audio_bytes += len(chunk)
                            if first_audio:
                                first_audio = False
                                first_byte_ms = (_time.monotonic() - t0) * 1000
                                logger.info(
                                    f"[SMOOTH] Svara first audio chunk: {first_byte_ms:.0f}ms "
                                    f"({len(chunk)} bytes)"
                                )
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
                            audio_dur = data.get('audio_duration', 0)
                            total_ms = (_time.monotonic() - t0) * 1000
                            logger.info(
                                f"[SMOOTH] Svara TTS done: {total_ms:.0f}ms total | "
                                f"text={len(text)}chars | audio={audio_dur:.2f}s | "
                                f"{audio_chunks_received} chunks | "
                                f"{total_audio_bytes}B | voice={self._config.voice}"
                            )
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
            headers = self._auth_headers()
            async with self._session.post(url, json=payload, headers=headers) as response:
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
            headers = self._auth_headers()
            async with self._session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Failed to list voices: HTTP {response.status}")
                    return []

                data = await response.json()
                return data.get("voices", [])

        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            return []
