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
import time as _time
import uuid
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional

import aiohttp
import websockets
from websockets.asyncio.client import connect as websocket_connect

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
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

from services.audio_utils import (
    CHUNK_MAX,
    CHUNK_MIN,
    CHUNK_TARGET,
    CROSSFADE_SEC,
    DATA_MARKER,
    WAV_HEADER_SIZE,
    apply_fade_edges,
    apply_fade_in,
    apply_fade_out,
    chunk_text,
    crossfade_pcm,
)

logger = logging.getLogger(__name__)

# Sentinel: pushed into a sentence queue to signal "no more audio for this sentence"
_SENTENCE_DONE = object()


@dataclass
class SvaraTTSConfig:
    """Configuration for Svara TTS service."""
    voice: str = "en_female"
    temperature: float = 0.75
    top_p: float = 0.9
    max_tokens: int = 4500  # Match Sarvam notebook; 2000 only gives ~4s of audio
    repetition_penalty: float = 1.1


class SvaraTTSService(TTSService):
    """
    Async (non-blocking) TTS service for Svara.

    Like ElevenLabs, run_tts() returns almost immediately.  Audio is fetched
    in background tasks and played back in sentence order by a drainer task.

    Constructor passes push_stop_frames=False because we manage TTSStoppedFrame
    ourselves in the drainer task (the base class timer fires too early since
    run_tts() returns immediately before audio is fetched).
    """

    # Class-level semaphore: Svara's vLLM backend has limited GPU concurrency.
    # Opening too many simultaneous WS connections causes "Background loop has
    # errored already" errors.  This is shared across ALL instances so that
    # multiple user sessions don't overwhelm the server.
    _svara_semaphore: asyncio.Semaphore = asyncio.Semaphore(1)

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080",
        api_key: str = None,
        voice: str = "en_female",
        temperature: float = 0.75,
        top_p: float = 0.9,
        max_tokens: int = 4500,
        repetition_penalty: float = 1.1,
        streaming: bool = True,
        sample_rate: int = 24000,
        **kwargs
    ):
        super().__init__(
            sample_rate=sample_rate,
            push_text_frames=True,
            # We manage TTSStoppedFrame ourselves in the drainer.  The base class
            # timer fires 2s after run_tts() returns, but run_tts() returns
            # immediately (async) — audio is still being fetched in background.
            push_stop_frames=False,
            # IMPORTANT: Do NOT pause frame processing. Our run_tts() is non-blocking
            # (returns immediately), so we need subsequent TextFrames to flow through
            # without waiting for BotStoppedSpeakingFrame. If paused, sentence 2+
            # would be blocked until sentence 1's audio finishes playing.
            pause_frame_processing=False,
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
        await self._stop_drainer_task()
        await self._cancel_all_fetches()
        if self._session:
            await self._session.close()
            self._session = None
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        logger.info(">>> TTS CANCEL: Stopping audio generation")
        self._interrupted = True
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
        chunks = chunk_text(text)

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
        """Connect to Svara, receive audio, push into the sentence queue.

        Concurrency is controlled per-WS-call (not per-sentence) via
        _svara_semaphore inside _fetch_single_chunk / _fetch_multi_chunk.
        This means a 3-chunk sentence releases the semaphore between chunks,
        allowing other users' sentences to interleave rather than blocking
        for the entire 21s multi-chunk duration.
        """
        t0 = _time.monotonic()
        sr = self._sample_rate or 24000
        ctx_queue = self._contexts.get(ctx_id)
        if not ctx_queue:
            return

        try:
            if self._interrupted:
                return
            if len(chunks) == 1:
                await self._fetch_single_chunk(ctx_id, text, ctx_queue, sr, t0)
            else:
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
        """Stream a single chunk from Svara with fade-in/fade-out.

        Acquires the class-level semaphore for the duration of the WS
        connection to prevent concurrent Svara connections.
        """
        fade_bytes = int(CROSSFADE_SEC * sr) * 2  # 16-bit PCM
        emit_chunk_size = 4096
        first_chunk_emitted = False
        pending_tail = b""

        async with self._svara_semaphore:
            async for raw_frame in self._svara_ws_stream(text, t0):
                if self._interrupted:
                    break
                if not isinstance(raw_frame, TTSAudioRawFrame) or not raw_frame.audio:
                    continue

                pcm = raw_frame.audio

                if not first_chunk_emitted:
                    pcm = apply_fade_in(pcm, sample_rate=sr)
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

        # Flush tail with fade-out (outside semaphore — no WS needed)
        if pending_tail and not self._interrupted:
            pending_tail = apply_fade_out(pending_tail, sample_rate=sr)
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
        """Stream audio progressively per chunk with crossfade at boundaries.

        Instead of collecting ALL chunks before pushing (which blocks TTFAB
        for 20+ seconds on 3-chunk sentences), we push audio as each chunk
        arrives, only holding back a small tail buffer for crossfading.

        The semaphore is acquired per-chunk (not per-sentence), so it's
        released between chunks — letting other users' sentences interleave.

        Flow:
          Chunk 1: [semaphore] fetch → fade-in → push → hold tail (60ms)
          Chunk 2: [semaphore] fetch → crossfade tail+head → push → hold tail
          ...
          Last chunk: [semaphore] fetch → crossfade → fade-out → push all
        """
        fade_bytes = int(CROSSFADE_SEC * sr) * 2  # 16-bit PCM, ~2880 bytes at 24kHz
        emit_chunk_size = 4096
        prev_tail = b""  # last fade_bytes of previous chunk for crossfading
        first_chunk_emitted = False
        is_first_chunk = True

        for i, chunk_text in enumerate(chunks):
            if self._interrupted:
                break

            is_last = (i == len(chunks) - 1)
            logger.info(
                f"[ASYNC-TTS] ctx={ctx_id} chunk {i+1}/{len(chunks)} "
                f"({len(chunk_text)} chars): '{chunk_text[:60]}...'"
            )

            # Collect this chunk's full PCM (each chunk is small, ~150-250 chars).
            # Semaphore is held only for the WS connection duration, then released
            # between chunks so other users/sentences can interleave.
            chunk_pcm = b""
            async with self._svara_semaphore:
                async for raw_frame in self._svara_ws_stream(chunk_text, t0):
                    if self._interrupted:
                        break
                    if isinstance(raw_frame, TTSAudioRawFrame) and raw_frame.audio:
                        chunk_pcm += raw_frame.audio

            if not chunk_pcm:
                continue

            # ── Crossfade with previous chunk's tail ──
            if prev_tail:
                # _crossfade_pcm overlaps last fade_samples of `a` with
                # first fade_samples of `b` — so pass prev_tail as `a`
                # and full chunk_pcm as `b`.
                chunk_pcm = crossfade_pcm(prev_tail, chunk_pcm, sample_rate=sr)

            # ── Apply fade-in on the very first chunk ──
            if is_first_chunk:
                chunk_pcm = apply_fade_in(chunk_pcm, sample_rate=sr)
                is_first_chunk = False

            # ── Hold back tail for crossfading with next chunk ──
            if not is_last and len(chunk_pcm) > fade_bytes:
                to_emit = chunk_pcm[:-fade_bytes]
                prev_tail = chunk_pcm[-fade_bytes:]
            elif is_last:
                # Last chunk — apply fade-out and emit everything
                chunk_pcm = apply_fade_out(chunk_pcm, sample_rate=sr)
                to_emit = chunk_pcm
                prev_tail = b""
            else:
                to_emit = chunk_pcm
                prev_tail = b""

            # ── Push frames to queue ──
            if not first_chunk_emitted and to_emit:
                first_chunk_emitted = True
                ttfb_ms = (_time.monotonic() - t0) * 1000
                logger.info(f"[ASYNC-TTS] ctx={ctx_id} first audio at {ttfb_ms:.0f}ms (multi-chunk)")
                await self.stop_ttfb_metrics()

            for j in range(0, len(to_emit), emit_chunk_size):
                if self._interrupted:
                    return
                await ctx_queue.put(TTSAudioRawFrame(
                    audio=to_emit[j:j + emit_chunk_size],
                    sample_rate=sr, num_channels=1,
                ))

        # If there's any remaining tail (shouldn't happen, but safety)
        if prev_tail and not self._interrupted:
            prev_tail = apply_fade_out(prev_tail, sample_rate=sr)
            for j in range(0, len(prev_tail), emit_chunk_size):
                await ctx_queue.put(TTSAudioRawFrame(
                    audio=prev_tail[j:j + emit_chunk_size],
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
        """Process sentence queues in FIFO order, pushing audio to pipeline.

        CRITICAL: When audio for a sentence was fully fetched in the background
        (while the previous sentence was still playing), the queue contains ALL
        frames already.  If we dump them all at once, the output transport's
        _write_audio_sleep() clock resets on every frame (because sleep_duration
        == 0 when frames arrive faster than real-time), effectively sending the
        entire sentence's audio in a burst.  The client can't play a 3-second
        sentence that arrives in 10ms — it drops or corrupts audio.

        Fix: after pushing each frame, sleep for the frame's audio duration so
        the downstream transport receives frames at roughly real-time pace.
        When audio is still *streaming* from Svara (queue drains faster than it
        fills), the `await ctx_queue.get()` itself provides the natural pacing.
        """
        try:
            while True:
                ctx_id = await self._contexts_queue.get()

                ctx_queue = self._contexts.get(ctx_id)
                if not ctx_queue:
                    self._contexts_queue.task_done()
                    continue

                logger.info(f"[ASYNC-TTS] Drainer: playing ctx={ctx_id}")

                sr = self._sample_rate or 24000
                frames_pushed = 0
                bytes_pushed = 0
                drain_t0 = _time.monotonic()

                # Drain all audio frames for this sentence.
                # Multi-chunk sentences collect ALL chunk audio before pushing,
                # which can take 30+ seconds for 3+ chunks.  Use a generous
                # timeout but check if the fetch task is still alive.
                fetch_task = self._fetch_tasks.get(ctx_id)
                while True:
                    try:
                        item = await asyncio.wait_for(ctx_queue.get(), timeout=5.0)
                    except asyncio.TimeoutError:
                        # If the fetch task is still running, keep waiting
                        fetch_task = self._fetch_tasks.get(ctx_id)
                        if fetch_task and not fetch_task.done():
                            logger.debug(
                                f"[ASYNC-TTS] Drainer: waiting for fetch ctx={ctx_id} "
                                f"(task still running)"
                            )
                            continue
                        logger.warning(f"[ASYNC-TTS] Drainer timeout ctx={ctx_id} (fetch task done)")
                        break

                    if item is _SENTENCE_DONE:
                        break

                    if isinstance(item, TTSAudioRawFrame):
                        await self.push_frame(item)
                        frames_pushed += 1
                        bytes_pushed += len(item.audio)

                        # ── Real-time pacing ──
                        # If the queue already has more items buffered, we're
                        # ahead of real-time.  Sleep for the duration of the
                        # audio we just pushed so the output transport clock
                        # stays in sync.  We use 95% of real-time to stay
                        # close to playback speed while keeping a tiny buffer
                        # ahead.  Using 80% caused 0.7x pacing ratios which
                        # flooded the client's WavStreamPlayer buffer and
                        # caused audio drops ("using sunlight, water, and
                        # carbon dioxide" missing).
                        if not ctx_queue.empty():
                            n_samples = len(item.audio) // 2  # 16-bit PCM
                            frame_duration = n_samples / sr
                            await asyncio.sleep(frame_duration * 0.95)

                # Clean up this context
                self._contexts.pop(ctx_id, None)

                # Check if more sentences are queued
                has_next = not self._contexts_queue.empty()

                if has_next:
                    # ── Keep-alive silence while waiting for next sentence's audio ──
                    # Pipecat's output transport fires BotStoppedSpeakingFrame after
                    # BOT_VAD_STOP_SECS (0.35s) of no audio frames.  If the next
                    # sentence's audio hasn't arrived yet, we pump silence to keep
                    # the stream alive and prevent the client's WavStreamPlayer
                    # from stopping/restarting (which drops initial audio).
                    #
                    # We push 100ms silence chunks every 100ms for up to 3 seconds.
                    # Once the next sentence's ctx_queue has data, we stop and let
                    # the normal drainer loop take over.
                    silence_samples = int(sr * 0.1)  # 100ms
                    silence_chunk = b"\x00" * (silence_samples * 2)
                    next_ctx_id = None
                    keepalive_pushed = 0
                    max_keepalive = 30  # 30 × 100ms = 3s max

                    # Find the next context's queue.  self._contexts is an
                    # insertion-ordered dict (Python 3.7+); after popping the
                    # current ctx_id above, the first remaining key is the
                    # next sentence.  This avoids accessing asyncio.Queue
                    # internals (_queue[0]).
                    remaining = list(self._contexts.keys())
                    next_ctx_id = remaining[0] if remaining else None
                    next_ctx_queue = self._contexts.get(next_ctx_id) if next_ctx_id else None

                    while keepalive_pushed < max_keepalive:
                        # Check if next sentence has audio ready
                        if next_ctx_queue and not next_ctx_queue.empty():
                            break
                        if self._interrupted:
                            break
                        await self.push_frame(TTSAudioRawFrame(
                            audio=silence_chunk, sample_rate=sr, num_channels=1,
                        ))
                        keepalive_pushed += 1
                        await asyncio.sleep(0.1)

                    if keepalive_pushed > 0:
                        logger.debug(
                            f"[ASYNC-TTS] Drainer: pushed {keepalive_pushed * 100}ms "
                            f"keepalive silence between ctx={ctx_id} and next"
                        )
                else:
                    # Last sentence — push a small silence gap before TTSStoppedFrame
                    silence = b"\x00" * (500 * 2)  # 21ms
                    await self.push_frame(TTSAudioRawFrame(
                        audio=silence, sample_rate=sr, num_channels=1,
                    ))

                # We manage TTSStoppedFrame ourselves (push_stop_frames=False).
                #
                # IMPORTANT: Only push TTSStoppedFrame when NO more sentences
                # are queued.  Pushing it between sentences can cause the
                # Pipecat client SDK's WavStreamPlayer AudioWorklet to stop
                # and restart, potentially dropping audio at the boundary.
                # By deferring to the last sentence, the audio stream stays
                # continuous from the client's perspective.
                #
                # The _tts_in_flight_count in TextAudioSyncNotifier will only
                # decrement once (for the final TTSStoppedFrame), but that's OK
                # because _send_complete_if_done checks in_flight==0.
                pushed_stop = False
                if not has_next:
                    await self.push_frame(TTSStoppedFrame())
                    pushed_stop = True

                audio_dur = bytes_pushed / (sr * 2)
                wall_dur = _time.monotonic() - drain_t0

                # Burst detection: if we pushed >0.5s of audio but wall clock
                # was less than half the audio duration, frames were dumped too
                # fast for the output transport to pace correctly.
                burst_flag = ""
                if audio_dur > 0.5 and wall_dur < audio_dur * 0.5:
                    burst_flag = " ⚠️ BURST"
                    logger.warning(
                        f"[ASYNC-TTS] BURST detected ctx={ctx_id}: "
                        f"{audio_dur:.2f}s audio pushed in {wall_dur:.2f}s wall "
                        f"(ratio={wall_dur/audio_dur:.2f}x) — client may drop audio"
                    )

                stop_info = "(pushed TTSStoppedFrame)" if pushed_stop else "(more queued, no TTSStoppedFrame)"
                logger.info(
                    f"[ASYNC-TTS] Drainer: done ctx={ctx_id} | "
                    f"{frames_pushed} frames, {audio_dur:.2f}s audio, "
                    f"{wall_dur:.2f}s wall (pacing={wall_dur/max(audio_dur,0.01):.1f}x)"
                    f"{burst_flag} {stop_info}"
                )
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

        MAX_RETRIES = 3
        for attempt in range(1, MAX_RETRIES + 1):
            ws = None
            try:
                if attempt > 1:
                    backoff = 0.5 * attempt
                    logger.info(f"[ASYNC-TTS] Retry {attempt}/{MAX_RETRIES} after {backoff:.1f}s backoff")
                    await asyncio.sleep(backoff)
                    t0 = _time.monotonic()  # reset timing for retry

                extra_headers = self._auth_headers()
                ws = await websocket_connect(
                    ws_url,
                    max_size=10 * 1024 * 1024,
                    ping_interval=20,
                    ping_timeout=20,
                    additional_headers=extra_headers,
                )
                connect_ms = (_time.monotonic() - t0) * 1000
                logger.info(f"[ASYNC-TTS] Svara WS connect: {connect_ms:.0f}ms (attempt {attempt})")

                # Send config
                logger.info(f"[ASYNC-TTS] Svara config: max_tokens={config.get('max_tokens')} text_len={len(text)}")
                await ws.send(json.dumps(config))

                # Wait for ack
                ack_msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                ack_data = json.loads(ack_msg)
                if ack_data.get("type") == "error":
                    err_msg = ack_data.get("message", "Unknown error")
                    if "Background loop" in err_msg and attempt < MAX_RETRIES:
                        logger.warning(f"[ASYNC-TTS] Svara ack error (attempt {attempt}): {err_msg}")
                        await ws.close()
                        continue  # retry
                    raise RuntimeError(err_msg)
                if ack_data.get("type") != "config_ack":
                    raise RuntimeError(f"Unexpected response: {ack_data}")

                ack_ms = (_time.monotonic() - t0) * 1000
                logger.debug(f"[ASYNC-TTS] Svara WS ready: {ack_ms:.0f}ms")

                # Receive audio chunks
                header_stripped = False
                header_buf = b""  # accumulate bytes until header is found & stripped
                first_audio = True
                audio_chunks = 0
                total_bytes = 0
                got_bg_loop_error = False

                while True:
                    if self._interrupted:
                        break

                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30.0)

                        if isinstance(msg, bytes):
                            chunk = msg
                            if not header_stripped:
                                # Accumulate bytes until we can find the "data" marker
                                header_buf += chunk
                                idx = header_buf.find(DATA_MARKER)
                                if idx >= 0:
                                    # Skip past "data" (4 bytes) + data-size field (4 bytes)
                                    audio_start = idx + 8
                                    chunk = header_buf[audio_start:]
                                    header_stripped = True
                                    logger.debug(
                                        f"[ASYNC-TTS] WAV header stripped: "
                                        f"{audio_start} bytes (found 'data' at offset {idx})"
                                    )
                                elif len(header_buf) > 200:
                                    # Safety: if we've accumulated 200+ bytes without
                                    # finding "data", fall back to fixed-size strip
                                    logger.warning(
                                        "[ASYNC-TTS] WAV 'data' marker not found in "
                                        f"first {len(header_buf)} bytes — "
                                        f"falling back to {WAV_HEADER_SIZE}-byte strip"
                                    )
                                    chunk = header_buf[WAV_HEADER_SIZE:]
                                    header_stripped = True
                                else:
                                    # Need more bytes to find the header
                                    continue

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
                                err_msg = data.get("message", "Unknown error")
                                if "Background loop" in err_msg and attempt < MAX_RETRIES:
                                    logger.warning(
                                        f"[ASYNC-TTS] Svara stream error (attempt {attempt}): "
                                        f"{err_msg} — will retry"
                                    )
                                    got_bg_loop_error = True
                                    break  # break inner loop to retry
                                raise RuntimeError(err_msg)

                    except asyncio.TimeoutError:
                        logger.warning("[ASYNC-TTS] WS recv timeout")
                        break

                if got_bg_loop_error:
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    continue  # retry outer loop

                # Success — exit retry loop
                return

            except websockets.ConnectionClosed as e:
                if not self._interrupted:
                    logger.warning(f"[ASYNC-TTS] WS closed (attempt {attempt}): {e}")
                if attempt >= MAX_RETRIES:
                    return
            except RuntimeError as e:
                logger.error(f"[ASYNC-TTS] WS error (attempt {attempt}): {e}")
                if attempt >= MAX_RETRIES:
                    return
            except Exception as e:
                logger.error(f"[ASYNC-TTS] WS error (attempt {attempt}): {e}")
                if attempt >= MAX_RETRIES:
                    return
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
