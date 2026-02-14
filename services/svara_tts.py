"""
Async TTS Service for Svara-TTS-FastAPI — Non-blocking, ElevenLabs-style.

ARCHITECTURE (mirrors ElevenLabs' AudioContextWordTTSService):
  1. run_tts() is called once per sentence by Pipecat's TTSService base.
  2. Instead of blocking while audio is generated, run_tts() kicks off a
     background task that connects to Svara via WebSocket and streams audio
     into an ordered asyncio.Queue.
  3. A persistent "audio drainer" task pulls frames from the queue and pushes
     them into the Pipecat pipeline via push_frame().
  4. Because run_tts() returns almost immediately (yields TTSStartedFrame then
     None), the base TTSService._push_tts_frames() finishes quickly and the
     pipeline can process the *next* sentence's TextFrame while the previous
     sentence's audio is still streaming from Svara.
  5. Sentences are played back in order because the drainer processes one
     sentence-queue at a time via a contexts_queue (FIFO of sentence IDs).

This eliminates the root cause of missing/skipped audio: the old blocking
run_tts() serialized sentence generation, causing buffer underruns.

Server endpoints:
- POST /v1/audio/text-to-speech/ - Non-streaming TTS (returns complete WAV)
- WEBSOCKET /v1/audio/text-to-speech/stream - Streaming TTS via WebSocket
- GET /v1/audio/text-to-speech/voices - List available voices
"""

import asyncio
import json
import logging
import re
import time as _time
import uuid
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional

import aiohttp
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
    InterruptionFrame,
    LLMFullResponseStartFrame,
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
# ---------------------------------------------------------------------------
_CLAUSE_SPLIT_RE = re.compile(r'(?<=[,.!?;:।؟\n])\s+')
_CHUNK_TARGET = 120
_CHUNK_MAX = 200
_CHUNK_MIN = 50

# Crossfade / fade duration in seconds
_CROSSFADE_SEC = 0.04

# Sentinel: pushed into a sentence queue to signal "no more audio for this sentence"
_SENTENCE_DONE = object()


@dataclass
class SvaraTTSConfig:
    """Configuration for Svara TTS service."""
    voice: str = "en_female"
    temperature: float = 0.75
    top_p: float = 0.9
    max_tokens: int = 350
    repetition_penalty: float = 1.1


class SvaraTTSService(TTSService):
    """
    Async (non-blocking) TTS service for Svara.

    Like ElevenLabs, run_tts() returns almost immediately.  Audio is fetched
    in background tasks and played back in sentence order by a drainer task.

    Constructor passes push_stop_frames=True and pause_frame_processing=True
    to the base TTSService so that:
      - TTSStoppedFrame is auto-sent after audio stops flowing (2s idle)
      - Incoming text frames are paused while audio is being output
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
        super().__init__(
            sample_rate=sample_rate,
            push_text_frames=True,
            push_stop_frames=True,
            stop_frame_timeout_s=2.0,
            pause_frame_processing=True,
            **kwargs,
        )

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
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

        # ---- Async audio context system (mirrors ElevenLabs) ----
        # Each sentence gets a unique context_id and its own asyncio.Queue.
        # The drainer task processes them in FIFO order.
        self._contexts: Dict[str, asyncio.Queue] = {}
        self._contexts_queue: asyncio.Queue = asyncio.Queue()  # FIFO of context_ids
        self._audio_drainer_task: Optional[asyncio.Task] = None
        self._fetch_tasks: Dict[str, asyncio.Task] = {}  # context_id → fetch task

        # Barge-in / interruption state
        self._interrupted = False
        self._started = False  # Whether TTSStartedFrame has been sent for current response

        # Pre-warmed WebSocket for next sentence
        self._warm_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._warm_ws_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def voice(self) -> str:
        return self._config.voice

    async def set_voice(self, voice: str):
        self._config.voice = voice
        logger.info(f"Svara TTS voice set to: {voice}")

    def can_generate_metrics(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._session = aiohttp.ClientSession()
        self._create_drainer_task()
        logger.info(f"Svara TTS service started (async), base_url: {self._base_url}")

    async def stop(self, frame: EndFrame):
        await self._close_warm_ws()
        await self._stop_drainer_task()
        await self._cancel_all_fetches()
        if self._session:
            await self._session.close()
            self._session = None
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        logger.info(">>> TTS CANCEL: Stopping audio generation")
        self._interrupted = True
        await self._close_warm_ws()
        await self._stop_drainer_task()
        await self._cancel_all_fetches()
        self._create_drainer_task()
        await super().cancel(frame)

    # ------------------------------------------------------------------
    # Frame processing — interruption handling
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, (StartInterruptionFrame, InterruptionFrame)):
            logger.info(">>> TTS BARGE-IN: User interrupted, stopping audio")
            self._interrupted = True
            self._started = False
            await self._close_warm_ws()
            await self._stop_drainer_task()
            await self._cancel_all_fetches()
            self._create_drainer_task()
            await self.push_frame(frame, direction)
            return

        # Reset state for new LLM response
        if isinstance(frame, LLMFullResponseStartFrame):
            self._interrupted = False
            self._started = False

        await super().process_frame(frame, direction)

    # ------------------------------------------------------------------
    # run_tts — NON-BLOCKING (the key change)
    # ------------------------------------------------------------------

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Kick off background audio fetch and return immediately.

        Yields TTSStartedFrame (once per LLM response) then None.
        Audio frames are pushed by the drainer task, not by this generator.
        """
        if not text.strip():
            return

        if self._interrupted:
            logger.info("TTS skipped - already interrupted")
            return

        if not self._session:
            self._session = aiohttp.ClientSession()

        # Chunk text for Svara's token limit
        chunks = self._chunk_text(text)

        # Create a unique context for this sentence
        ctx_id = str(uuid.uuid4())[:8]
        ctx_queue: asyncio.Queue = asyncio.Queue()
        self._contexts[ctx_id] = ctx_queue

        # Enqueue this context for the drainer (FIFO order)
        await self._contexts_queue.put(ctx_id)

        logger.info(
            f"[ASYNC-TTS] Queued sentence ctx={ctx_id} ({len(text)} chars, "
            f"{len(chunks)} chunk(s)): '{text[:80]}{'...' if len(text) > 80 else ''}'"
        )

        # Kick off background fetch task
        task = asyncio.ensure_future(self._fetch_audio(ctx_id, text, chunks))
        self._fetch_tasks[ctx_id] = task

        # Yield TTSStartedFrame for EVERY sentence — the TextAudioSyncNotifier
        # uses each TTSStartedFrame to release the corresponding sentence text
        # to the client for display.
        if not self._started:
            await self.start_ttfb_metrics()
            self._started = True
        yield TTSStartedFrame()

        # Return immediately — audio will be pushed by the drainer
        yield None

    # ------------------------------------------------------------------
    # Background audio fetch (one task per sentence)
    # ------------------------------------------------------------------

    async def _fetch_audio(self, ctx_id: str, text: str, chunks: List[str]):
        """Connect to Svara, receive audio, push into the sentence queue."""
        t0 = _time.monotonic()
        sr = self._sample_rate or 24000
        ctx_queue = self._contexts.get(ctx_id)
        if not ctx_queue:
            return

        try:
            if len(chunks) == 1:
                # Single chunk — stream directly with fade edges
                await self._fetch_single_chunk(ctx_id, text, ctx_queue, sr, t0)
            else:
                # Multi-chunk — collect per chunk, crossfade, then push
                await self._fetch_multi_chunk(ctx_id, chunks, ctx_queue, sr, t0)
        except asyncio.CancelledError:
            logger.info(f"[ASYNC-TTS] Fetch cancelled ctx={ctx_id}")
        except Exception as e:
            logger.error(f"[ASYNC-TTS] Fetch error ctx={ctx_id}: {e}")
        finally:
            # Signal end of audio for this sentence
            if ctx_queue:
                await ctx_queue.put(_SENTENCE_DONE)
            self._fetch_tasks.pop(ctx_id, None)

    async def _fetch_single_chunk(
        self, ctx_id: str, text: str, ctx_queue: asyncio.Queue,
        sr: int, t0: float
    ):
        """Stream a single chunk from Svara with fade-in/fade-out."""
        fade_bytes = int(_CROSSFADE_SEC * sr) * 2  # 16-bit PCM
        emit_chunk_size = 4096
        first_chunk_emitted = False
        pending_tail = b""

        async for raw_frame in self._svara_ws_stream(text, t0):
            if self._interrupted:
                break
            if not isinstance(raw_frame, TTSAudioRawFrame) or not raw_frame.audio:
                continue

            pcm = raw_frame.audio

            if not first_chunk_emitted:
                pcm = self._apply_fade_in(pcm, sample_rate=sr)
                first_chunk_emitted = True
                ttfb_ms = (_time.monotonic() - t0) * 1000
                logger.info(f"[ASYNC-TTS] ctx={ctx_id} first audio at {ttfb_ms:.0f}ms")
                await self.stop_ttfb_metrics()

            # Buffer tail for fade-out
            combined = pending_tail + pcm
            if len(combined) > fade_bytes:
                to_emit = combined[:-fade_bytes]
                pending_tail = combined[-fade_bytes:]
                for i in range(0, len(to_emit), emit_chunk_size):
                    if self._interrupted:
                        return
                    await ctx_queue.put(TTSAudioRawFrame(
                        audio=to_emit[i:i + emit_chunk_size],
                        sample_rate=sr, num_channels=1,
                    ))
            else:
                pending_tail = combined

        # Flush tail with fade-out
        if pending_tail and not self._interrupted:
            pending_tail = self._apply_fade_out(pending_tail, sample_rate=sr)
            for i in range(0, len(pending_tail), emit_chunk_size):
                if self._interrupted:
                    return
                await ctx_queue.put(TTSAudioRawFrame(
                    audio=pending_tail[i:i + emit_chunk_size],
                    sample_rate=sr, num_channels=1,
                ))

        total_ms = (_time.monotonic() - t0) * 1000
        logger.info(f"[ASYNC-TTS] ctx={ctx_id} single-chunk done in {total_ms:.0f}ms")

    async def _fetch_multi_chunk(
        self, ctx_id: str, chunks: List[str], ctx_queue: asyncio.Queue,
        sr: int, t0: float
    ):
        """Collect audio per chunk, crossfade, then push into queue."""
        combined_pcm = b""

        for i, chunk in enumerate(chunks):
            if self._interrupted:
                break

            logger.info(
                f"[ASYNC-TTS] ctx={ctx_id} chunk {i+1}/{len(chunks)} "
                f"({len(chunk)} chars): '{chunk[:60]}...'"
            )

            chunk_pcm = b""
            async for raw_frame in self._svara_ws_stream(chunk, t0):
                if self._interrupted:
                    break
                if isinstance(raw_frame, TTSAudioRawFrame) and raw_frame.audio:
                    chunk_pcm += raw_frame.audio

            if not chunk_pcm:
                continue

            if combined_pcm:
                combined_pcm = self._crossfade_pcm(combined_pcm, chunk_pcm, sample_rate=sr)
            else:
                combined_pcm = chunk_pcm

        if combined_pcm and not self._interrupted:
            combined_pcm = self._apply_fade_edges(combined_pcm, sample_rate=sr)
            chunk_size = 4096
            for i in range(0, len(combined_pcm), chunk_size):
                if self._interrupted:
                    break
                await ctx_queue.put(TTSAudioRawFrame(
                    audio=combined_pcm[i:i + chunk_size],
                    sample_rate=sr, num_channels=1,
                ))

        total_ms = (_time.monotonic() - t0) * 1000
        logger.info(f"[ASYNC-TTS] ctx={ctx_id} multi-chunk done in {total_ms:.0f}ms")

    # ------------------------------------------------------------------
    # Audio drainer task — plays sentences in FIFO order
    # ------------------------------------------------------------------

    def _create_drainer_task(self):
        if not self._audio_drainer_task or self._audio_drainer_task.done():
            self._contexts_queue = asyncio.Queue()
            self._contexts = {}
            self._audio_drainer_task = asyncio.ensure_future(self._audio_drainer())

    async def _stop_drainer_task(self):
        if self._audio_drainer_task and not self._audio_drainer_task.done():
            self._audio_drainer_task.cancel()
            try:
                await self._audio_drainer_task
            except asyncio.CancelledError:
                pass
        self._audio_drainer_task = None

    async def _audio_drainer(self):
        """Process sentence queues in FIFO order, pushing audio to pipeline."""
        try:
            while True:
                ctx_id = await self._contexts_queue.get()

                ctx_queue = self._contexts.get(ctx_id)
                if not ctx_queue:
                    self._contexts_queue.task_done()
                    continue

                logger.info(f"[ASYNC-TTS] Drainer: playing ctx={ctx_id}")

                # Drain all audio frames for this sentence
                while True:
                    try:
                        item = await asyncio.wait_for(ctx_queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"[ASYNC-TTS] Drainer timeout ctx={ctx_id}")
                        break

                    if item is _SENTENCE_DONE:
                        break

                    if isinstance(item, TTSAudioRawFrame):
                        await self.push_frame(item)

                # Clean up this context
                self._contexts.pop(ctx_id, None)

                # Add a tiny silence gap between sentences (500 samples ≈ 21ms at 24kHz)
                sr = self._sample_rate or 24000
                silence = b"\x00" * (500 * 2)  # 500 samples × 2 bytes/sample
                await self.push_frame(TTSAudioRawFrame(
                    audio=silence, sample_rate=sr, num_channels=1,
                ))

                logger.info(f"[ASYNC-TTS] Drainer: done ctx={ctx_id}")
                self._contexts_queue.task_done()

        except asyncio.CancelledError:
            logger.info("[ASYNC-TTS] Drainer cancelled")

    # ------------------------------------------------------------------
    # Cancel all fetch tasks
    # ------------------------------------------------------------------

    async def _cancel_all_fetches(self):
        for ctx_id, task in list(self._fetch_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._fetch_tasks.clear()
        self._contexts.clear()
        self._started = False

    # ------------------------------------------------------------------
    # WebSocket pre-warming
    # ------------------------------------------------------------------

    async def _close_warm_ws(self):
        if self._warm_ws_task and not self._warm_ws_task.done():
            self._warm_ws_task.cancel()
            try:
                await self._warm_ws_task
            except asyncio.CancelledError:
                pass
            self._warm_ws_task = None
        if self._warm_ws:
            try:
                await self._warm_ws.close()
            except Exception:
                pass
            self._warm_ws = None

    def _kick_prewarm(self):
        """Start pre-warming the next WebSocket connection in the background."""
        if self._warm_ws_task and not self._warm_ws_task.done():
            return  # Already warming
        self._warm_ws_task = asyncio.ensure_future(self._prewarm_ws())

    async def _prewarm_ws(self):
        """Connect a WebSocket in the background for the next sentence."""
        try:
            ws_url = f"{self._ws_url}/v1/audio/text-to-speech/stream"
            extra_headers = self._auth_headers()
            ws = await websocket_connect(
                ws_url,
                max_size=10 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=20,
                additional_headers=extra_headers,
            )
            self._warm_ws = ws
            logger.debug("[ASYNC-TTS] Pre-warmed WS ready")
        except Exception as e:
            logger.warning(f"[ASYNC-TTS] Pre-warm failed: {e}")
            self._warm_ws = None

    # ------------------------------------------------------------------
    # Svara WebSocket streaming (low-level)
    # ------------------------------------------------------------------

    async def _svara_ws_stream(self, text: str, t0: float) -> AsyncGenerator[Frame, None]:
        """Connect to Svara WS, send text, yield TTSAudioRawFrame chunks."""
        ws_url = f"{self._ws_url}/v1/audio/text-to-speech/stream"

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
            "api_key": key,
            "token": key,
            "authorization": bearer,
        }

        ws = None
        try:
            # Try to use pre-warmed WebSocket
            if self._warm_ws:
                ws = self._warm_ws
                self._warm_ws = None
                logger.debug("[ASYNC-TTS] Using pre-warmed WS")
            else:
                extra_headers = self._auth_headers()
                ws = await websocket_connect(
                    ws_url,
                    max_size=10 * 1024 * 1024,
                    ping_interval=20,
                    ping_timeout=20,
                    additional_headers=extra_headers,
                )
                connect_ms = (_time.monotonic() - t0) * 1000
                logger.info(f"[ASYNC-TTS] Svara WS connect: {connect_ms:.0f}ms")

            # Send config
            await ws.send(json.dumps(config))

            # Wait for ack
            ack_msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
            ack_data = json.loads(ack_msg)
            if ack_data.get("type") == "error":
                raise RuntimeError(ack_data.get("message", "Unknown error"))
            if ack_data.get("type") != "config_ack":
                raise RuntimeError(f"Unexpected response: {ack_data}")

            ack_ms = (_time.monotonic() - t0) * 1000
            logger.debug(f"[ASYNC-TTS] Svara WS ready: {ack_ms:.0f}ms")

            # Kick off pre-warm for next sentence
            self._kick_prewarm()

            # Receive audio chunks
            header_stripped = False
            first_audio = True
            audio_chunks = 0
            total_bytes = 0

            while True:
                if self._interrupted:
                    break

                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30.0)

                    if isinstance(msg, bytes):
                        chunk = msg
                        if not header_stripped:
                            if len(chunk) > WAV_HEADER_SIZE:
                                chunk = chunk[WAV_HEADER_SIZE:]
                            header_stripped = True

                        if chunk:
                            audio_chunks += 1
                            total_bytes += len(chunk)
                            if first_audio:
                                first_audio = False
                                fb_ms = (_time.monotonic() - t0) * 1000
                                logger.info(
                                    f"[ASYNC-TTS] Svara first audio: {fb_ms:.0f}ms "
                                    f"({len(chunk)}B)"
                                )
                            yield TTSAudioRawFrame(
                                audio=chunk,
                                sample_rate=self._sample_rate or 24000,
                                num_channels=1,
                            )
                    else:
                        data = json.loads(msg)
                        msg_type = data.get("type")
                        if msg_type == "done":
                            dur = data.get("audio_duration", 0)
                            total_ms = (_time.monotonic() - t0) * 1000
                            logger.info(
                                f"[ASYNC-TTS] Svara done: {total_ms:.0f}ms | "
                                f"text={len(text)}c | audio={dur:.2f}s | "
                                f"{audio_chunks}chunks | {total_bytes}B"
                            )
                            break
                        elif msg_type == "error":
                            raise RuntimeError(data.get("message", "Unknown error"))

                except asyncio.TimeoutError:
                    logger.warning("[ASYNC-TTS] WS recv timeout")
                    break

        except websockets.ConnectionClosed as e:
            if not self._interrupted:
                logger.warning(f"[ASYNC-TTS] WS closed: {e}")
        except Exception as e:
            logger.error(f"[ASYNC-TTS] WS error: {e}")
        finally:
            if ws:
                try:
                    await ws.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict:
        key = (self._api_key or "").strip()
        if not key:
            return {}
        bearer = key if key.lower().startswith("bearer ") else f"Bearer {key}"
        return {
            "Authorization": bearer,
            "x-api-key": key,
            "X-API-Key": key,
        }

    # ------------------------------------------------------------------
    # Text chunking (unchanged from Sarvam's proven algorithm)
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(
        text: str,
        target_size: int = _CHUNK_TARGET,
        max_size: int = _CHUNK_MAX,
        min_size: int = _CHUNK_MIN,
    ) -> List[str]:
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
            if projected_len > max_size:
                chunks.append(buffer)
                buffer = part
                continue
            if len(buffer) >= target_size:
                chunks.append(buffer)
                buffer = part
                continue
            buffer += " " + part

        if buffer:
            chunks.append(buffer)

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

    # ------------------------------------------------------------------
    # Audio processing helpers (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _crossfade_pcm(
        a: bytes, b: bytes, fade_sec: float = _CROSSFADE_SEC, sample_rate: int = 24000
    ) -> bytes:
        if not a or not b:
            return a + b
        fade_samples = int(fade_sec * sample_rate)
        a_arr = np.frombuffer(a, dtype=np.int16).astype(np.float32)
        b_arr = np.frombuffer(b, dtype=np.int16).astype(np.float32)
        fade_samples = min(fade_samples, len(a_arr), len(b_arr))
        if fade_samples < 2:
            return a + b
        t = np.linspace(0, 1, fade_samples, endpoint=False)
        fade_out = np.cos(t * np.pi / 2)
        fade_in = np.sin(t * np.pi / 2)
        cross = a_arr[-fade_samples:] * fade_out + b_arr[:fade_samples] * fade_in
        result = np.concatenate([a_arr[:-fade_samples], cross, b_arr[fade_samples:]])
        return np.clip(result, -32768, 32767).astype(np.int16).tobytes()

    @staticmethod
    def _apply_fade_edges(
        pcm: bytes, fade_sec: float = _CROSSFADE_SEC, sample_rate: int = 24000
    ) -> bytes:
        if not pcm or len(pcm) < 4:
            return pcm
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        fade_samples = int(fade_sec * sample_rate)
        fade_samples = min(fade_samples, len(arr) // 4)
        if fade_samples < 2:
            return pcm
        t_in = np.linspace(0, 1, fade_samples, endpoint=False)
        arr[:fade_samples] *= np.sin(t_in * np.pi / 2)
        t_out = np.linspace(0, 1, fade_samples, endpoint=False)
        arr[-fade_samples:] *= np.cos(t_out * np.pi / 2)
        return np.clip(arr, -32768, 32767).astype(np.int16).tobytes()

    @staticmethod
    def _apply_fade_in(
        pcm: bytes, fade_sec: float = _CROSSFADE_SEC, sample_rate: int = 24000
    ) -> bytes:
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

    # ------------------------------------------------------------------
    # Voice listing
    # ------------------------------------------------------------------

    async def list_voices(self) -> list:
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
