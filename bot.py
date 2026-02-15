"""
MiraVoiceAI Pipecat Bot - STT -> LLM -> TTS Pipeline with Barge-In Support.

This bot orchestrates:
- IndicASR-Streaming for Speech-to-Text (WebSocket)
- vLLM with OpenAI-compatible API for LLM
- Svara-TTS-FastAPI for Text-to-Speech (HTTP)

The pipeline processes user speech, generates AI responses,
and synthesizes speech output.

BARGE-IN SUPPORT:
When enabled (allow_interruptions=True), the user can interrupt the bot
mid-speech. This works as follows:
1. VAD detects user speech during bot audio playback
2. StartInterruptionFrame is sent through the pipeline
3. TTS stops generating audio immediately
4. LLM generation is cancelled
5. STT continues listening (persistent connection)
6. New user utterance is processed
"""

import asyncio
import logging
import os
import re
import time
from typing import Optional

from pipecat.frames.frames import (
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TextFrame,
    TTSSpeakFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    LLMAssistantAggregatorParams,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.interruptions.min_words_interruption_strategy import MinWordsInterruptionStrategy
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from services.svara_tts import SvaraTTSService
from services.soniox_stt import SonioxSTTService
from services.elevenlabs_tts import create_elevenlabs_tts

# Configure logging
# Control verbosity via LOG_LEVEL env var: DEBUG, INFO (default), WARNING, ERROR
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VoiceStateManager:
    """Manages the current voice state for voice switching functionality.

    Voice IDs are loaded from the ElevenLabs VOICE_PRESETS (services/elevenlabs_tts.py)
    so there is a single source of truth. If the presets are updated, voice switching
    picks up the change automatically.
    """

    def __init__(self, initial_gender: str = "female"):
        from services.elevenlabs_tts import VOICE_PRESETS
        self._voice_ids = {g: p["id"] for g, p in VOICE_PRESETS.items()}
        self._current_gender = initial_gender

    @property
    def current_gender(self) -> str:
        return self._current_gender

    @property
    def current_voice_id(self) -> str:
        return self._voice_ids.get(self._current_gender, self._voice_ids.get("female"))

    def switch_voice(self, mode: str) -> tuple[str, str]:
        """Switch voice based on mode. Returns (new_gender, new_voice_id)."""
        if mode == "switch":
            self._current_gender = "male" if self._current_gender == "female" else "female"
        elif mode in self._voice_ids:
            self._current_gender = mode
        return self._current_gender, self.current_voice_id


SELECT_VOICE_SCHEMA = FunctionSchema(
    name="select_voice",
    description="Switch the assistant's voice between male and female. Use this when the user asks to change the voice, switch voice, wants a different voice, or asks for a male/female voice.",
    properties={
        "mode": {
            "type": "string",
            "enum": ["switch", "male", "female"],
            "description": "The voice selection mode: 'switch' to toggle current voice, 'male' to use male voice, 'female' to use female voice"
        }
    },
    required=["mode"]
)


class PipelineInstrumentor(FrameProcessor):
    """
    Comprehensive pipeline instrumentation for diagnosing voice quality.

    Tracks end-to-end timing across every stage of the pipeline:
      - VAD → STT latency (how fast speech is transcribed)
      - STT → LLM latency (how fast the LLM starts responding)
      - LLM → TTS latency (how fast TTS starts after first LLM token)
      - TTS → Audio-out latency (time to first audible byte)
      - Full turn latency (user stops speaking → first bot audio)
      - LLM token cadence (inter-token timing for streaming smoothness)
      - TTS chunk cadence (audio chunk delivery pattern)
      - Barge-in metrics (interruption timing)

    All timings use monotonic time.time() for accuracy.
    Logs are prefixed with [METRICS] for easy grep/filtering.
    """

    def __init__(self, name: str = "PipelineInstrumentor",
                 metrics_collector=None, is_classroom: bool = False, **kwargs):
        super().__init__(name=name, **kwargs)
        self._turn_count = 0
        self._llm_buffer = ""
        self._metrics_collector = metrics_collector  # Optional MetricsCollector
        self._is_classroom = is_classroom  # Whether this is a classroom voice session

        # Per-turn timing anchors
        self._user_started_speaking_at: float = 0.0
        self._user_stopped_speaking_at: float = 0.0
        self._stt_final_at: float = 0.0
        self._llm_first_token_at: float = 0.0
        self._llm_response_start_at: float = 0.0
        self._llm_response_end_at: float = 0.0
        self._tts_started_at: float = 0.0
        self._tts_stopped_at: float = 0.0
        self._bot_started_speaking_at: float = 0.0
        self._bot_stopped_speaking_at: float = 0.0
        self._first_audio_out_at: float = 0.0

        # LLM streaming metrics
        self._llm_token_count: int = 0
        self._llm_token_times: list = []

        # TTS audio metrics
        self._tts_audio_chunks: int = 0
        self._tts_audio_bytes: int = 0

        # Barge-in tracking
        self._barge_in_count: int = 0

        # Session-level aggregates
        self._turn_latencies: list = []

    def _reset_turn(self):
        """Reset per-turn counters for a new conversation turn."""
        self._llm_buffer = ""
        self._llm_token_count = 0
        self._llm_token_times = []
        self._tts_audio_chunks = 0
        self._tts_audio_bytes = 0
        self._llm_first_token_at = 0.0
        self._llm_response_start_at = 0.0
        self._llm_response_end_at = 0.0
        self._tts_started_at = 0.0
        self._tts_stopped_at = 0.0
        self._bot_started_speaking_at = 0.0
        self._bot_stopped_speaking_at = 0.0
        self._first_audio_out_at = 0.0

    def _ms(self, start: float, end: float) -> float:
        """Convert time delta to milliseconds, return 0 if invalid."""
        if start > 0 and end > 0 and end >= start:
            return round((end - start) * 1000, 1)
        return 0.0

    def _log_turn_summary(self):
        """Log a comprehensive summary of the completed turn."""
        user_speech_ms = self._ms(self._user_started_speaking_at, self._user_stopped_speaking_at)
        vad_to_stt_ms = self._ms(self._user_stopped_speaking_at, self._stt_final_at)
        stt_to_llm_ms = self._ms(self._stt_final_at, self._llm_first_token_at)
        llm_ttft_ms = self._ms(self._user_stopped_speaking_at, self._llm_first_token_at)
        llm_generation_ms = self._ms(self._llm_response_start_at, self._llm_response_end_at)
        llm_to_tts_ms = self._ms(self._llm_first_token_at, self._tts_started_at)
        tts_duration_ms = self._ms(self._tts_started_at, self._tts_stopped_at)
        bot_speaking_ms = self._ms(self._bot_started_speaking_at, self._bot_stopped_speaking_at)
        full_turn_latency_ms = self._ms(self._user_stopped_speaking_at, self._first_audio_out_at)

        if full_turn_latency_ms > 0:
            self._turn_latencies.append(full_turn_latency_ms)

        avg_turn_latency = 0.0
        if self._turn_latencies:
            avg_turn_latency = round(sum(self._turn_latencies) / len(self._turn_latencies), 1)

        mode_label = "CLASSROOM" if self._is_classroom else "TUTOR"
        logger.info(f"")
        logger.info(f"{'─' * 70}")
        logger.info(f"[METRICS][{mode_label}] TURN {self._turn_count} SUMMARY")
        logger.info(f"{'─' * 70}")
        logger.info(f"[METRICS]   User speech duration:    {user_speech_ms:>8.1f} ms")
        logger.info(f"[METRICS]   VAD→STT (transcribe):    {vad_to_stt_ms:>8.1f} ms")
        logger.info(f"[METRICS]   STT→LLM (first token):   {stt_to_llm_ms:>8.1f} ms")
        logger.info(f"[METRICS]   LLM generation total:     {llm_generation_ms:>8.1f} ms  ({self._llm_token_count} tokens)")
        logger.info(f"[METRICS]   LLM→TTS (first chunk):    {llm_to_tts_ms:>8.1f} ms")
        logger.info(f"[METRICS]   TTS duration:             {tts_duration_ms:>8.1f} ms  ({self._tts_audio_chunks} chunks, {self._tts_audio_bytes} bytes)")
        logger.info(f"[METRICS]   Bot speaking duration:    {bot_speaking_ms:>8.1f} ms")
        logger.info(f"[METRICS]   ★ FULL TURN LATENCY:      {full_turn_latency_ms:>8.1f} ms  (user-stop → first-audio)")
        logger.info(f"[METRICS]   Session avg turn latency: {avg_turn_latency:>8.1f} ms  ({len(self._turn_latencies)} turns)")
        logger.info(f"[METRICS]   Barge-ins this session:   {self._barge_in_count}")
        logger.info(f"[METRICS]   Bot response: '{self._llm_buffer[:120]}{'...' if len(self._llm_buffer) > 120 else ''}'")
        logger.info(f"{'─' * 70}")
        logger.info(f"")

        # ── Feed into MetricsCollector for /metrics endpoint ──
        if self._metrics_collector:
            self._metrics_collector.record_voice_turn(
                turn_latency_ms=full_turn_latency_ms,
                stt_ms=vad_to_stt_ms,
                llm_ttft_ms=llm_ttft_ms,
                llm_total_ms=llm_generation_ms,
                llm_to_tts_ms=llm_to_tts_ms,
                tts_ms=tts_duration_ms,
                tokens=self._llm_token_count,
                tts_bytes=self._tts_audio_bytes,
                is_classroom=self._is_classroom,
            )
            # Per-call trace for voice pipeline
            self._metrics_collector.record_trace({
                "mode": "classroom_voice" if self._is_classroom else "tutor_voice",
                "query": self._llm_buffer[:80],
                "ts": self._user_stopped_speaking_at or time.time(),
                "total_ms": full_turn_latency_ms,
                "stages": [
                    {"name": "user_speech", "ms": user_speech_ms},
                    {"name": "vad_to_stt", "ms": vad_to_stt_ms},
                    {"name": "stt_to_llm_ttft", "ms": stt_to_llm_ms},
                    {"name": "llm_generation", "ms": llm_generation_ms,
                     "tokens": self._llm_token_count},
                    {"name": "llm_to_tts", "ms": llm_to_tts_ms},
                    {"name": "tts", "ms": tts_duration_ms,
                     "chunks": self._tts_audio_chunks,
                     "audio_bytes": self._tts_audio_bytes},
                    {"name": "bot_speaking", "ms": bot_speaking_ms},
                ],
            })

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        now = time.time()

        # VAD: User started speaking
        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_started_speaking_at = now
            self._reset_turn()
            logger.info(f"")
            logger.info(f"{'=' * 60}")
            logger.info(f"[METRICS] >>> VAD: USER STARTED SPEAKING <<<  t={now:.3f}")
            logger.info(f"{'=' * 60}")

        # VAD: User stopped speaking
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_stopped_speaking_at = now
            speech_dur = self._ms(self._user_started_speaking_at, now)
            logger.info(f"{'=' * 60}")
            logger.info(f"[METRICS] >>> VAD: USER STOPPED SPEAKING <<<  duration={speech_dur:.0f}ms")
            logger.info(f"{'=' * 60}")

        # STT: Final transcription
        elif isinstance(frame, TranscriptionFrame):
            self._stt_final_at = now
            self._turn_count += 1
            stt_latency = self._ms(self._user_stopped_speaking_at, now)
            logger.info(f"[METRICS] [TURN {self._turn_count}] STT FINAL: '{frame.text}'  (VAD→STT: {stt_latency:.0f}ms)")

        # STT: Interim transcription
        elif isinstance(frame, InterimTranscriptionFrame):
            interim_latency = self._ms(self._user_started_speaking_at, now)
            logger.debug(f"[METRICS] STT INTERIM: '{frame.text[:60]}...'  ({interim_latency:.0f}ms from speech start)")

        # LLM: Response stream start
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._llm_response_start_at = now
            logger.info(f"[METRICS] LLM response stream STARTED  (STT→LLM-start: {self._ms(self._stt_final_at, now):.0f}ms)")

        # LLM: Text token
        elif isinstance(frame, TextFrame):
            self._llm_token_count += 1
            self._llm_token_times.append(now)
            self._llm_buffer += frame.text

            if self._llm_token_count == 1:
                self._llm_first_token_at = now
                ttft = self._ms(self._user_stopped_speaking_at, now)
                logger.info(f"[METRICS] LLM FIRST TOKEN: '{frame.text}'  (TTFT from user-stop: {ttft:.0f}ms)")

        # LLM: Response stream end
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._llm_response_end_at = now
            gen_ms = self._ms(self._llm_response_start_at, now)
            tps = self._llm_token_count / (gen_ms / 1000) if gen_ms > 0 else 0
            logger.info(f"[METRICS] LLM response COMPLETE: {self._llm_token_count} tokens in {gen_ms:.0f}ms ({tps:.1f} tok/s)")

        # TTS: Started generating
        elif isinstance(frame, TTSStartedFrame):
            # Track gap between consecutive TTS sentences
            if self._tts_stopped_at > 0:
                inter_sentence_gap = self._ms(self._tts_stopped_at, now)
                if inter_sentence_gap > 0:
                    logger.info(
                        f"[SMOOTH] Inter-sentence gap: {inter_sentence_gap:.0f}ms "
                        f"(TTS-stop → TTS-start)"
                    )
            self._tts_started_at = now
            llm_to_tts = self._ms(self._llm_first_token_at, now)
            logger.info(f"[METRICS] TTS STARTED  (LLM-first-token → TTS-start: {llm_to_tts:.0f}ms)")

        # TTS: Stopped generating
        elif isinstance(frame, TTSStoppedFrame):
            self._tts_stopped_at = now
            tts_dur = self._ms(self._tts_started_at, now)
            logger.info(f"[METRICS] TTS STOPPED  duration={tts_dur:.0f}ms  chunks={self._tts_audio_chunks}  bytes={self._tts_audio_bytes}")
            self._log_turn_summary()

        # Bot started speaking (audio out)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_started_speaking_at = now
            full_latency = self._ms(self._user_stopped_speaking_at, now)
            logger.info(f"[METRICS] BOT STARTED SPEAKING  (user-stop → bot-speak: {full_latency:.0f}ms)")

        # Bot stopped speaking
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_stopped_speaking_at = now
            speak_dur = self._ms(self._bot_started_speaking_at, now)
            logger.info(f"[METRICS] BOT STOPPED SPEAKING  duration={speak_dur:.0f}ms")

        # Audio output frame (for first-byte tracking)
        elif isinstance(frame, AudioRawFrame):
            self._tts_audio_chunks += 1
            self._tts_audio_bytes += len(frame.audio) if hasattr(frame, 'audio') else 0

            if self._tts_audio_chunks == 1:
                self._first_audio_out_at = now
                ttfb = self._ms(self._user_stopped_speaking_at, now)
                logger.info(f"[METRICS] ★ FIRST AUDIO BYTE OUT  TTFB={ttfb:.0f}ms (user-stop → first-audio)")

        # Barge-in
        elif isinstance(frame, StartInterruptionFrame):
            self._barge_in_count += 1
            bot_interrupted_after = self._ms(self._bot_started_speaking_at, now)
            logger.warning(f"[METRICS] ⚡ BARGE-IN #{self._barge_in_count}  (bot was speaking for {bot_interrupted_after:.0f}ms)")
            self._llm_buffer = ""

        # Forward the frame downstream
        await self.push_frame(frame, direction)


# Regex to match [TEACHER_ACTION: ...] or [TUTOR_ACTION: ...] tags
_ACTION_TAG_RE = re.compile(r'\[(?:TEACHER_ACTION|TUTOR_ACTION):\s*[^\]]*\]\s*', re.IGNORECASE)


class ActionTagFilter(FrameProcessor):
    """
    Strips [TEACHER_ACTION: ...] and [TUTOR_ACTION: ...] tags from LLM output.

    The LLM sometimes echoes these command tags in its response even though the
    system prompt says not to. This filter removes them from TextFrames before
    they reach the text forwarder and TTS, so users never see raw command tags.

    If a TextFrame's entire content is just a tag (nothing left after stripping),
    the frame is dropped entirely.
    """

    def __init__(self, name: str = "ActionTagFilter", **kwargs):
        super().__init__(name=name, **kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Must call super() first so StartFrame and other system frames are handled
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            cleaned = _ACTION_TAG_RE.sub('', frame.text)
            if not cleaned.strip():
                # Entire frame was just a tag — drop it
                logger.debug(f"[ActionTagFilter] Dropped tag-only frame: '{frame.text}'")
                return
            if cleaned != frame.text:
                logger.info(f"[ActionTagFilter] Stripped tags: '{frame.text}' -> '{cleaned}'")
                frame.text = cleaned
        await self.push_frame(frame, direction)


class TextStreamForwarder(FrameProcessor):
    """
    Sends LLM text to the client as JSON messages for real-time text rendering.

    Each WebSocket connection gets its own ``TextStreamForwarder`` +
    ``TextAudioSyncNotifier`` pair, so all state is **per-receiver**.

    Behavior depends on mode:

    text_only mode:
      Streams every LLM token immediately for fast text rendering.
      TTSSpeakFrames (greetings) are consumed here — no TTS downstream.

    text_and_audio mode:
      Text is QUEUED sentence-by-sentence into an ``asyncio.Queue`` and
      only released to the client when TTS actually starts speaking that
      sentence.  A companion ``TextAudioSyncNotifier`` sits after TTS in
      the pipeline and calls ``release_next_sentence()`` on each
      ``TTSStartedFrame``.

      **TTS timeout guard** — if a queued sentence sits for longer than
      ``_TTS_SENTENCE_TIMEOUT`` seconds without TTS starting, a watchdog
      task sends the text to the client anyway (so the user can read it),
      then continues with the next sentence.  This prevents text from
      being stuck behind a hung or errored TTS call.

    JSON messages sent:
      - {"type": "bot_text", "text": "...", "streaming": true}
      - {"type": "bot_text_complete", "text": "full response"}
    """

    # Sentence-ending punctuation (covers English, Hindi Devanagari, etc.)
    _SENTENCE_ENDS = re.compile(r'[.!?।؟\n]\s*$')

    # How long to wait for TTS to start a sentence before sending text anyway
    _TTS_SENTENCE_TIMEOUT = 15.0  # seconds

    def __init__(self, websocket, text_only: bool = False, name: str = "TextStreamForwarder", **kwargs):
        super().__init__(name=name, **kwargs)
        self._websocket = websocket
        self._text_only = text_only
        self._current_response = ""
        self._sentence_buffer = ""        # Accumulates tokens until sentence boundary
        self._in_response = False
        self._needs_space_before_next = False  # Ensure space between sentences for TTS aggregator

        # ── Per-receiver sentence queue (text_and_audio only) ──
        # Each sentence is a string.  The queue is consumed by
        # release_next_sentence() which is called either by
        # TextAudioSyncNotifier (on TTSStartedFrame) or by the
        # watchdog timer if TTS is too slow.
        self._sentence_q: asyncio.Queue[str] = asyncio.Queue()
        self._watchdog_task: asyncio.Task | None = None
        self._watchdog_event = asyncio.Event()  # signalled when TTS releases a sentence

        # ── Metrics (per-response, reset on LLMFullResponseStartFrame) ──
        self._metrics_sentences_queued: int = 0       # sentences enqueued this response
        self._metrics_sentences_released: int = 0     # released by TTS (normal path)
        self._metrics_sentences_timed_out: int = 0    # released by watchdog (TTS too slow)
        self._metrics_sentences_flushed: int = 0      # flushed on end/interruption
        self._metrics_response_start_at: float = 0.0  # time of LLMFullResponseStartFrame
        self._metrics_first_sentence_queued_at: float = 0.0
        self._metrics_first_sentence_released_at: float = 0.0

        # ── Session-level metrics (across all responses) ──
        self._metrics_total_responses: int = 0
        self._metrics_total_timeouts: int = 0
        self._metrics_total_interruptions: int = 0

    def _reset_response_metrics(self):
        """Reset per-response metrics counters."""
        self._metrics_sentences_queued = 0
        self._metrics_sentences_released = 0
        self._metrics_sentences_timed_out = 0
        self._metrics_sentences_flushed = 0
        self._metrics_response_start_at = time.time()
        self._metrics_first_sentence_queued_at = 0.0
        self._metrics_first_sentence_released_at = 0.0

    def _log_response_metrics(self):
        """Log a summary of text-audio sync metrics for the completed response."""
        total = self._metrics_sentences_queued
        if total == 0 and self._text_only:
            return  # text_only mode doesn't queue sentences
        elapsed_ms = round((time.time() - self._metrics_response_start_at) * 1000, 1) if self._metrics_response_start_at else 0
        queue_to_release_ms = 0.0
        if self._metrics_first_sentence_queued_at and self._metrics_first_sentence_released_at:
            queue_to_release_ms = round(
                (self._metrics_first_sentence_released_at - self._metrics_first_sentence_queued_at) * 1000, 1
            )
        mode_label = "text_only" if self._text_only else "text_and_audio"
        logger.info(
            f"[TEXT_SYNC_METRICS] response_complete | mode={mode_label} | "
            f"sentences_queued={total} | released_by_tts={self._metrics_sentences_released} | "
            f"timed_out={self._metrics_sentences_timed_out} | flushed={self._metrics_sentences_flushed} | "
            f"first_sentence_delay={queue_to_release_ms}ms | total_elapsed={elapsed_ms}ms | "
            f"session_responses={self._metrics_total_responses} | "
            f"session_timeouts={self._metrics_total_timeouts}"
        )

    # ── Called by TextAudioSyncNotifier when TTS starts a sentence ──
    async def release_next_sentence(self):
        """Send the next queued sentence to the client (called on TTSStartedFrame)."""
        if self._sentence_q.empty():
            return
        try:
            text = self._sentence_q.get_nowait()
        except asyncio.QueueEmpty:
            return
        await self._send_sentence(text)
        self._metrics_sentences_released += 1
        if self._metrics_sentences_released == 1:
            self._metrics_first_sentence_released_at = time.time()
        # Signal watchdog that this sentence was released normally
        self._watchdog_event.set()

    async def _send_sentence(self, text: str):
        """Send a single sentence to the client."""
        try:
            await self._websocket.send_json({
                "type": "bot_text",
                "text": text,
                "streaming": True,
            })
            logger.debug(f"[TEXT_STREAM] Released sentence: '{text[:80]}'")
        except Exception as e:
            logger.warning(f"[TEXT_STREAM] Failed to send sentence: {e}")

    async def flush_all_queued(self):
        """Flush all remaining queued sentences (called on response end / interruption)."""
        while not self._sentence_q.empty():
            try:
                text = self._sentence_q.get_nowait()
                await self._send_sentence(text)
                self._metrics_sentences_flushed += 1
            except asyncio.QueueEmpty:
                break
        self._stop_watchdog()

    def _start_watchdog(self):
        """Start the TTS timeout watchdog (if not already running)."""
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    def _stop_watchdog(self):
        """Cancel the watchdog task."""
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            self._watchdog_task = None

    async def _watchdog_loop(self):
        """
        Background task: if a sentence sits in the queue longer than
        _TTS_SENTENCE_TIMEOUT without being released by TTS, send the
        text to the client anyway.  This prevents text from being stuck
        behind a hung TTS call — the user can at least read it.
        """
        try:
            while True:
                # Wait for a sentence to be enqueued
                if self._sentence_q.empty():
                    await asyncio.sleep(0.1)
                    continue

                # A sentence is waiting — give TTS time to claim it
                self._watchdog_event.clear()
                try:
                    await asyncio.wait_for(
                        self._watchdog_event.wait(),
                        timeout=self._TTS_SENTENCE_TIMEOUT,
                    )
                    # TTS released it in time — loop back
                    continue
                except asyncio.TimeoutError:
                    pass

                # TTS didn't start this sentence in time — send text anyway
                if not self._sentence_q.empty():
                    try:
                        text = self._sentence_q.get_nowait()
                    except asyncio.QueueEmpty:
                        continue
                    self._metrics_sentences_timed_out += 1
                    self._metrics_total_timeouts += 1
                    logger.warning(
                        f"[TEXT_STREAM] TTS timeout ({self._TTS_SENTENCE_TIMEOUT}s) — "
                        f"sending text without audio: '{text[:60]}' "
                        f"(timeout #{self._metrics_sentences_timed_out} this response, "
                        f"#{self._metrics_total_timeouts} session)"
                    )
                    await self._send_sentence(text)
                    # Continue loop — will pick up next sentence
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[TEXT_STREAM] Watchdog error: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._in_response = True
            self._current_response = ""
            self._sentence_buffer = ""
            self._needs_space_before_next = False
            self._metrics_total_responses += 1
            self._reset_response_metrics()
            # Drain any leftover from previous response
            while not self._sentence_q.empty():
                try:
                    self._sentence_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSSpeakFrame):
            # Greeting / direct-speak frames — send as text immediately
            try:
                await self._websocket.send_json({
                    "type": "bot_text_complete",
                    "text": frame.text,
                })
                logger.debug(f"[TEXT_STREAM] Greeting text sent: '{frame.text[:120]}'")
            except Exception as e:
                logger.debug(f"[TEXT_STREAM] Failed to send greeting text: {e}")

            if self._text_only:
                # No TTS downstream — consume the frame
                return
            # In text_and_audio mode, push downstream so TTS synthesizes it
            await self.push_frame(frame, direction)

        elif isinstance(frame, TextFrame) and self._in_response:
            token_text = frame.text
            self._current_response += token_text

            # ── Ensure space between sentences for Pipecat's TTS aggregator ──
            # LLM tokens like "Why" after "right?" may lack a leading space,
            # causing Pipecat's NLTK-based aggregator to merge "right?Why"
            # into one sentence instead of splitting at the "?".
            if self._needs_space_before_next and token_text and not token_text[0].isspace():
                frame.text = " " + token_text
                self._needs_space_before_next = False
            elif token_text and token_text[0].isspace():
                self._needs_space_before_next = False

            if self._text_only:
                # ── text_only: stream every token immediately (fast) ──
                try:
                    await self._websocket.send_json({
                        "type": "bot_text",
                        "text": token_text,  # Send original text to client
                        "streaming": True,
                    })
                    logger.debug(f"[TEXT_STREAM] Sent token: '{token_text}'")
                except Exception as e:
                    logger.warning(f"[TEXT_STREAM] Failed to send token: {e}")
            else:
                # ── text_and_audio: buffer tokens, enqueue on sentence boundary ──
                # Don't send to client yet — TextAudioSyncNotifier will call
                # release_next_sentence() when TTS starts speaking this sentence.
                self._sentence_buffer += token_text  # Use original text for our buffer
                if self._SENTENCE_ENDS.search(self._sentence_buffer):
                    sentence = self._sentence_buffer.strip()
                    if sentence:
                        self._sentence_q.put_nowait(sentence)
                        self._metrics_sentences_queued += 1
                        if self._metrics_sentences_queued == 1:
                            self._metrics_first_sentence_queued_at = time.time()
                        self._start_watchdog()
                        logger.debug(f"[TEXT_STREAM] Queued sentence #{self._metrics_sentences_queued}: '{sentence[:60]}'")
                    self._sentence_buffer = ""
                    self._needs_space_before_next = True

            # Always push downstream (to TTS in audio mode, or to assistant aggregator)
            # frame.text may have a leading space injected for TTS aggregator
            await self.push_frame(frame, direction)

        elif isinstance(frame, LLMFullResponseEndFrame):
            self._in_response = False

            if self._text_only:
                # text_only: send bot_text_complete immediately (no TTS to wait for)
                try:
                    await self._websocket.send_json({
                        "type": "bot_text_complete",
                        "text": self._current_response,
                    })
                    logger.info(f"[TEXT_STREAM] Complete response: '{self._current_response[:120]}{'...' if len(self._current_response) > 120 else ''}'")
                except Exception as e:
                    logger.debug(f"[TEXT_STREAM] Failed to send complete text: {e}")
                self._current_response = ""
                self._log_response_metrics()
            else:
                # text_and_audio: enqueue any remaining partial sentence.
                # Do NOT send bot_text_complete here — TextAudioSyncNotifier
                # will flush queued sentences first, then send the complete message
                # so the client gets text in the right order.
                leftover = self._sentence_buffer.strip()
                if leftover:
                    self._sentence_q.put_nowait(leftover)
                    self._start_watchdog()
                self._sentence_buffer = ""
                # _current_response is preserved — TextAudioSyncNotifier reads it

            await self.push_frame(frame, direction)

        elif isinstance(frame, StartInterruptionFrame):
            # On barge-in, flush any queued text immediately so the client
            # has the full text up to the interruption point.
            if not self._text_only:
                self._metrics_total_interruptions += 1
                await self.flush_all_queued()
                self._sentence_buffer = ""
                self._log_response_metrics()
            await self.push_frame(frame, direction)

        else:
            # Forward everything else unchanged
            await self.push_frame(frame, direction)


class TextAudioSyncNotifier(FrameProcessor):
    """
    Sits AFTER TTS in the text_and_audio pipeline.  Per-receiver (one
    instance per WebSocket connection, paired with a TextStreamForwarder).

    When TTS emits a ``TTSStartedFrame`` (= it began synthesising a sentence),
    this processor tells the upstream ``TextStreamForwarder`` to release the
    corresponding sentence text to the client.  The result: text appears in
    the chat window at the same moment audio starts playing.

    On ``TTSStoppedFrame`` after the last sentence (when LLM response is done),
    it sends ``bot_text_complete`` so the client finalizes the display.
    """

    def __init__(self, text_forwarder: TextStreamForwarder, name: str = "TextAudioSyncNotifier", **kwargs):
        super().__init__(name=name, **kwargs)
        self._text_forwarder = text_forwarder
        # Counter of in-flight TTS sentences (incremented on TTSStartedFrame,
        # decremented on TTSStoppedFrame).  With async Svara, each sentence
        # gets its own TTSStartedFrame/TTSStoppedFrame pair.  A simple boolean
        # would go False after sentence 1's TTSStoppedFrame, causing premature
        # bot_text_complete while sentences 2+ are still playing.
        self._tts_in_flight = 0

    async def _send_complete_if_done(self):
        """Send bot_text_complete if the LLM response is finished, all sentences
        released, AND TTS has finished generating audio for ALL sentences."""
        fwd = self._text_forwarder
        if (not fwd._in_response
                and fwd._sentence_q.empty()
                and fwd._current_response
                and self._tts_in_flight <= 0):
            fwd._stop_watchdog()
            try:
                await fwd._websocket.send_json({
                    "type": "bot_text_complete",
                    "text": fwd._current_response,
                })
                logger.info(
                    f"[TEXT_SYNC] Complete response (after TTS): "
                    f"'{fwd._current_response[:120]}{'...' if len(fwd._current_response) > 120 else ''}'"
                )
            except Exception as e:
                logger.debug(f"[TEXT_SYNC] Failed to send complete text: {e}")
            fwd._current_response = ""
            fwd._log_response_metrics()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            self._tts_in_flight += 1
            logger.debug(f"[TEXT_SYNC] TTSStartedFrame — in_flight={self._tts_in_flight}")
            # TTS just started speaking — release ALL pending sentence text
            # to the client.  We release ALL (not just one) because the TTS
            # aggregator may merge multiple TextStreamForwarder sentences into
            # a single run_tts() call (e.g. Hindi text where NLTK doesn't
            # split on '।').  In that case there's only one TTSStartedFrame
            # but N sentences queued.  Releasing all ensures text appears
            # on screen as soon as audio begins.
            fwd = self._text_forwarder
            released = 0
            while not fwd._sentence_q.empty():
                await fwd.release_next_sentence()
                released += 1
            if released > 1:
                logger.debug(f"[TEXT_SYNC] Released {released} sentences on TTSStartedFrame")
            # NOTE: Do NOT call _send_complete_if_done() here.  TTS is still
            # generating audio.  Wait for TTSStoppedFrame to finalize.

        elif isinstance(frame, TTSStoppedFrame):
            # The Svara drainer only pushes TTSStoppedFrame after the LAST
            # sentence (intermediate sentences skip it to keep the client's
            # audio stream continuous).  So when we see it, ALL sentences
            # are done — reset the counter to 0 rather than decrementing.
            prev = self._tts_in_flight
            self._tts_in_flight = 0
            logger.debug(f"[TEXT_SYNC] TTSStoppedFrame — in_flight={prev}→0")
            # Finalize now that all TTS is done
            await self._send_complete_if_done()

        elif isinstance(frame, LLMFullResponseEndFrame):
            # LLM is done producing text.  Do NOT flush remaining queued
            # sentences here — they still need to be synthesised by TTS.
            # The TTSStoppedFrame handler above will call _send_complete_if_done()
            # after the last sentence's audio finishes.
            #
            # However, if the queue is already empty (TTS already processed
            # everything), we should finalize now.
            await self._send_complete_if_done()

        elif isinstance(frame, StartInterruptionFrame):
            # On barge-in, reset in-flight counter and flush text
            self._tts_in_flight = 0
            fwd = self._text_forwarder
            await fwd.flush_all_queued()
            if fwd._current_response:
                try:
                    await fwd._websocket.send_json({
                        "type": "bot_text_complete",
                        "text": fwd._current_response,
                    })
                except Exception:
                    pass
                fwd._current_response = ""

        await self.push_frame(frame, direction)


class UserTranscriptForwarder(FrameProcessor):
    """
    Sends STT transcription results back to the client as JSON messages
    so the frontend can display what the user said.

    Strips internal metadata tags ([User is speaking X], Speaker N:) that are
    meant for the LLM pipeline, not for display.

    JSON messages sent:
      - {"type": "user_transcript", "text": "...", "final": true}   — final transcript
      - {"type": "user_transcript", "text": "...", "final": false}  — interim transcript
    """

    # Regex to strip internal pipeline metadata from display text
    _LANG_TAG_RE = re.compile(r"\[User is speaking \w+\]\s*")
    _SPEAKER_RE = re.compile(r"Speaker\s+\d+:\s*")

    def __init__(self, websocket, name: str = "UserTranscriptForwarder", **kwargs):
        super().__init__(name=name, **kwargs)
        self._websocket = websocket

    @classmethod
    def _clean_for_display(cls, text: str) -> str:
        """Strip [User is speaking X] and Speaker N: prefixes for client display."""
        cleaned = cls._LANG_TAG_RE.sub("", text)
        cleaned = cls._SPEAKER_RE.sub("", cleaned)
        return cleaned.strip()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            try:
                display_text = self._clean_for_display(frame.text)
                await self._websocket.send_json({
                    "type": "user_transcript",
                    "text": display_text,
                    "final": True,
                })
                logger.info(f"[USER_TEXT] Sent final transcript: '{display_text[:80]}'")
            except Exception as e:
                logger.warning(f"[USER_TEXT] Failed to send transcript: {e}")
            await self.push_frame(frame, direction)

        elif isinstance(frame, InterimTranscriptionFrame):
            try:
                display_text = self._clean_for_display(frame.text)
                await self._websocket.send_json({
                    "type": "user_transcript",
                    "text": display_text,
                    "final": False,
                })
            except Exception as e:
                logger.debug(f"[USER_TEXT] Failed to send interim: {e}")
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)


class STTClarityGate(FrameProcessor):
    """
    Pipeline processor that intercepts low-clarity STT transcriptions and
    asks the speaker to repeat — like a human teacher who says "I didn't
    catch that, could you say it again?"

    Sits BETWEEN STT + UserTranscriptForwarder and the LLM aggregator.

    Behavior:
      - If ``TranscriptionFrame.clarity_score >= CLARITY_THRESHOLD``:
        pass through normally (LLM processes the transcription).
      - If ``clarity_score < CLARITY_THRESHOLD``:
        1. Drop the transcription (don't send to LLM).
        2. Inject a ``TTSSpeakFrame`` asking the speaker to repeat.
        3. Send a ``stt_clarification`` JSON message to the client so the
           frontend can display a "please repeat" indicator.
        4. Log the event for metrics.

    The threshold and clarification messages are configurable via env vars:
      - ``STT_CLARITY_THRESHOLD`` (default: 0.50)
      - ``STT_CLARITY_MAX_RETRIES`` (default: 2) — after N consecutive
        low-clarity utterances, pass through anyway to avoid infinite loops.

    This processor does NOT affect:
      - Typed text (TextInputInjector) — those bypass STT entirely.
      - Interim transcriptions — only final transcriptions are gated.
      - Classroom mode — ClassroomBroadcaster handles its own transcription
        broadcasting. This gate runs in the speaker's pipeline before the
        LLM sees the transcription.
    """

    # Clarification messages by detected language
    _CLARIFICATION_MESSAGES = {
        "en": "Sorry, I didn't quite catch that. Could you say it again?",
        "hi": "माफ़ कीजिए, मुझे ठीक से सुनाई नहीं दिया। क्या आप दोबारा बोल सकते हैं?",
        "ta": "மன்னிக்கவும், எனக்கு சரியாகப் புரியவில்லை. மீண்டும் சொல்ல முடியுமா?",
    }

    # Echo-back confirmation messages (for medium-low confidence)
    _ECHO_MESSAGES = {
        "en": "Did you say: \"{text}\"?",
        "hi": "क्या आपने कहा: \"{text}\"?",
        "ta": "நீங்கள் சொன்னது: \"{text}\" என்பதா?",
    }

    def __init__(
        self,
        websocket,
        name: str = "STTClarityGate",
        clarity_threshold: float = None,
        max_retries: int = None,
        metrics_collector=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._websocket = websocket
        self._clarity_threshold = clarity_threshold or float(
            os.getenv("STT_CLARITY_THRESHOLD", "0.50")
        )
        self._max_retries = max_retries or int(
            os.getenv("STT_CLARITY_MAX_RETRIES", "2")
        )
        self._consecutive_low = 0  # Track consecutive low-clarity utterances
        self._enabled = os.getenv("STT_CLARITY_GATE_ENABLED", "true").lower() == "true"
        self._metrics_collector = metrics_collector

        # Metrics
        self._total_gated = 0
        self._total_passed = 0
        self._total_force_passed = 0  # Passed after max retries

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, TranscriptionFrame):
            # Pass everything else through unchanged
            await self.push_frame(frame, direction)
            return

        # Only gate voice transcriptions, not typed text
        if getattr(frame, "user_id", "") == "typed":
            self._consecutive_low = 0
            await self.push_frame(frame, direction)
            return

        # If gate is disabled, pass through with logging
        if not self._enabled:
            await self.push_frame(frame, direction)
            return

        clarity = getattr(frame, "clarity_score", 1.0)
        raw_text = getattr(frame, "raw_text", "")
        lang = getattr(frame, "language", "en") or "en"

        if clarity >= self._clarity_threshold:
            # High clarity — pass through normally
            self._consecutive_low = 0
            self._total_passed += 1
            if self._metrics_collector:
                self._metrics_collector.record_stt_clarity(clarity, gated=False, language=lang)
            logger.debug(
                f"[CLARITY_GATE] PASS | score={clarity:.2f} | "
                f"text='{raw_text[:60]}'"
            )
            await self.push_frame(frame, direction)
            return

        # Low clarity detected
        self._consecutive_low += 1

        # Safety valve: after max_retries consecutive low-clarity utterances,
        # pass through anyway to avoid frustrating the user
        if self._consecutive_low > self._max_retries:
            self._total_force_passed += 1
            if self._metrics_collector:
                self._metrics_collector.record_stt_clarity(clarity, gated=False, language=lang)
            logger.warning(
                f"[CLARITY_GATE] FORCE_PASS | score={clarity:.2f} | "
                f"consecutive_low={self._consecutive_low} > max_retries={self._max_retries} | "
                f"text='{raw_text[:60]}'"
            )
            self._consecutive_low = 0
            await self.push_frame(frame, direction)
            return

        # Gate the transcription — don't send to LLM
        self._total_gated += 1
        if self._metrics_collector:
            self._metrics_collector.record_stt_clarity(clarity, gated=True, language=lang)
        logger.info(
            f"[CLARITY_GATE] GATED | score={clarity:.2f} | "
            f"attempt={self._consecutive_low}/{self._max_retries} | "
            f"text='{raw_text[:60]}' | "
            f"total_gated={self._total_gated}"
        )

        # Choose clarification message based on language
        clarification = self._CLARIFICATION_MESSAGES.get(
            lang, self._CLARIFICATION_MESSAGES["en"]
        )

        # Notify the client about the clarification
        try:
            await self._websocket.send_json({
                "type": "stt_clarification",
                "message": clarification,
                "clarity_score": clarity,
                "original_text": raw_text,
                "attempt": self._consecutive_low,
                "max_retries": self._max_retries,
            })
        except Exception as e:
            logger.warning(f"[CLARITY_GATE] Failed to send clarification JSON: {e}")

        # Speak the clarification via TTS so the user hears it
        await self.push_frame(
            TTSSpeakFrame(text=clarification),
            FrameDirection.DOWNSTREAM,
        )

        # Do NOT push the original TranscriptionFrame — it's dropped.
        # The user will re-speak, and the next transcription will be
        # evaluated again by this gate.


class TextInputInjector(FrameProcessor):
    """
    Allows text to be injected into a running voice pipeline.

    When a user types text while voice mode is active, this processor
    receives the text via an asyncio.Queue and emits TranscriptionFrame(s)
    downstream, simulating speech input. The LLM then processes it and
    TTS speaks the response — giving the user a hybrid text-input / voice-output
    experience.

    The injector runs a background task that polls its queue and pushes
    frames downstream through the pipeline.
    """

    def __init__(self, session_id: str, websocket=None, name: str = "TextInputInjector", **kwargs):
        super().__init__(name=name, **kwargs)
        self._session_id = session_id
        self._websocket = websocket
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: asyncio.Task | None = None

    async def inject_text(self, text: str):
        """Put text into the injection queue."""
        logger.info(f"[TEXT_INJECT] Queuing text for session {self._session_id}: '{text[:80]}'")
        await self._queue.put(text)

    async def _poll_loop(self):
        """Background task that drains the queue and pushes TranscriptionFrames."""
        self._running = True
        logger.info(f"[TEXT_INJECT] Poll loop started for session {self._session_id}")
        while self._running:
            try:
                text = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                if text is None:
                    break  # Sentinel to stop

                logger.info(f"[TEXT_INJECT] Injecting text as transcription: '{text[:80]}'")

                # Send user_transcript to the client so the frontend shows the text
                if self._websocket:
                    try:
                        await self._websocket.send_json({
                            "type": "user_transcript",
                            "text": text,
                            "final": True,
                        })
                    except Exception as e:
                        logger.warning(f"[TEXT_INJECT] Failed to send user_transcript: {e}")

                # Simulate the STT output: UserStartedSpeaking -> Transcription -> UserStoppedSpeaking
                # This triggers the LLM context aggregator properly
                await self.push_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
                await self.push_frame(
                    TranscriptionFrame(text=text, user_id="typed", timestamp=""),
                    FrameDirection.DOWNSTREAM,
                )
                await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[TEXT_INJECT] Poll loop error: {e}")

        logger.info(f"[TEXT_INJECT] Poll loop ended for session {self._session_id}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Start background poll loop on the first StartFrame
        if isinstance(frame, StartFrame) and not self._task:
            self._task = asyncio.create_task(self._poll_loop())

        await self.push_frame(frame, direction)

    async def cleanup(self):
        """Stop the background task."""
        self._running = False
        if self._task:
            await self._queue.put(None)  # sentinel
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


# ── Global registry of active text injectors (keyed by session_id) ──
_active_text_injectors: dict[str, TextInputInjector] = {}


def register_text_injector(session_id: str, injector: TextInputInjector):
    """Register a text injector for the given session."""
    _active_text_injectors[session_id] = injector
    logger.info(f"[TEXT_INJECT] Registered injector for session {session_id}")


def unregister_text_injector(session_id: str):
    """Remove a text injector when the session ends."""
    _active_text_injectors.pop(session_id, None)
    logger.info(f"[TEXT_INJECT] Unregistered injector for session {session_id}")


def get_text_injector(session_id: str) -> TextInputInjector | None:
    """Look up the active text injector for a session."""
    return _active_text_injectors.get(session_id)


class GreetingProcessor(FrameProcessor):
    """
    Processor that speaks a greeting when the pipeline starts.
    Injects a TTSSpeakFrame on StartFrame which flows to TTS for synthesis.
    Can be skipped for reconnects where the room already has conversation history.
    """

    def __init__(self, greeting_text: str, name: str = "GreetingProcessor", skip: bool = False, **kwargs):
        super().__init__(name=name, **kwargs)
        self._greeting_text = greeting_text
        self._greeting_spoken = False
        self._skip = skip

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # On StartFrame, inject greeting text for TTS (unless skipped)
        if isinstance(frame, StartFrame) and not self._greeting_spoken:
            self._greeting_spoken = True
            if self._skip:
                logger.info("[GREETING] Skipped (reconnect with existing conversation)")
                await self.push_frame(frame, direction)
                return
            # Push StartFrame FIRST so TTS initializes before receiving text
            await self.push_frame(frame, direction)
            # Short delay for ElevenLabs WebSocket handshake
            await asyncio.sleep(0.3)
            logger.info(f"[GREETING] Speaking: '{self._greeting_text}'")
            # Use TTSSpeakFrame for immediate synthesis (bypasses text aggregator)
            await self.push_frame(TTSSpeakFrame(text=self._greeting_text), FrameDirection.DOWNSTREAM)
            return  # Don't push StartFrame again

        # Forward the original frame
        await self.push_frame(frame, direction)


# ─────────────────────────────────────────────────────────────────────
# Environment configuration — ALL provider settings are env-var driven.
# Change provider by setting the env var; no code changes needed.
# ─────────────────────────────────────────────────────────────────────

# --- LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")       # "openai" (or any OpenAI-compatible)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://vllm-gpt-oss-120b/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "DUMMY_KEY")

# --- STT ---
STT_PROVIDER = os.getenv("STT_PROVIDER", "soniox")       # "soniox" | "deepgram" | "whisper"
SONIOX_API_KEY = os.getenv("SONIOX_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
STT_LANGUAGE_HINTS = os.getenv("STT_LANGUAGE_HINTS", "en,hi,ta")  # Comma-separated
STT_SAMPLE_RATE = int(os.getenv("STT_SAMPLE_RATE", "16000"))

# --- TTS ---
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs")    # "elevenlabs" | "svara" | "openai"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
TTS_VOICE_GENDER = os.getenv("TTS_VOICE_GENDER", "female")  # "female" | "male"
TTS_WS_URL = os.getenv("TTS_WS_URL", "ws://svara-tts/v1/audio/text-to-speech/stream")
TTS_WS_API_KEY = os.getenv("TTS_WS_API_KEY", "")          # API key for Svara TTS auth
TTS_MAX_TOKENS = int(os.getenv("TTS_MAX_TOKENS", "4500"))   # Max tokens per Svara TTS request (text is chunked by sentence)
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "nova")  # For OpenAI TTS provider

# --- Voice ---
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "en_female")   # Default voice for Svara TTS
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "auto")

# VAD params as env vars for tuning without code change
VAD_CONFIDENCE = float(os.getenv("VAD_CONFIDENCE", "0.5"))
VAD_START_SECS = float(os.getenv("VAD_START_SECS", "0.2"))
VAD_STOP_SECS = float(os.getenv("VAD_STOP_SECS", "1.0"))
VAD_MIN_VOLUME = float(os.getenv("VAD_MIN_VOLUME", "0.4"))

# Minimum words the user must speak before a barge-in interruption fires.
# Prevents false barge-in from acoustic echo (bot audio leaking into mic),
# especially on mobile devices without hardware echo cancellation.
# Set to 0 to disable (immediate interruption on any speech).
INTERRUPTION_MIN_WORDS = int(os.getenv("INTERRUPTION_MIN_WORDS", "2"))

# Prompt configuration
PROMPT_DIR = os.getenv("PROMPT_DIR", os.path.join(os.path.dirname(__file__), "prompts"))
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v4")


def _load_prompt_file(filename: str) -> str:
    """Load a single prompt file from the prompts directory."""
    prompt_path = os.path.join(PROMPT_DIR, filename)
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning(f"Prompt file not found: {prompt_path}")
        return ""


def load_system_prompt(version: str = PROMPT_VERSION, mode: str = "voice") -> str:
    """Load system prompt from prompts directory.

    For v4+, composes base + mode-specific prompt.
    For older versions (v1-v3), loads the single file as before.

    Args:
        version: Prompt version (e.g. "v3", "v4").
        mode: One of "voice", "classroom", "text". Only used for v4+.
    """
    if version.startswith("v4"):
        base = _load_prompt_file(f"{version}-base.md")
        mode_prompt = _load_prompt_file(f"{version}-{mode}.md")
        if base and mode_prompt:
            return base + "\n\n" + mode_prompt
        elif base:
            return base
        elif mode_prompt:
            return mode_prompt
        else:
            logger.warning(f"No v4 prompt files found for {version}/{mode}, using fallback")
            return "You are a helpful voice assistant. Keep responses concise."
    else:
        # Legacy single-file prompts (v1, v2, v3)
        content = _load_prompt_file(f"{version}.md")
        return content or "You are a helpful voice assistant. Keep responses concise."


def get_default_system_prompt() -> str:
    """Get default system prompt for voice tutor mode."""
    return load_system_prompt(version=PROMPT_VERSION, mode="voice")


# Greeting text spoken when connection is established
# NOTE: Single sentence avoids TTS splitting into multiple audio segments
GREETING_TEXT = os.getenv("GREETING_TEXT",
    "Namaste! I'm Mira, your study buddy. I speak English, Hindi, and Tamil. Ask me anything!")


# ─────────────────────────────────────────────────────────────────────
# Provider factories — swap provider via env var, no code changes.
# ─────────────────────────────────────────────────────────────────────

def _normalize_language_hint(lang: str) -> Optional[str]:
    """Normalize language hint values to Soniox-friendly short codes."""
    if not lang:
        return None
    normalized = lang.strip().lower()
    alias_map = {
        "english": "en",
        "hindi": "hi",
        "tamil": "ta",
    }
    return alias_map.get(normalized, normalized)


def create_stt_service(sample_rate: int = None, language_hints_override: list[str] | None = None):
    """
    Create an STT service based on STT_PROVIDER env var.

    Supported providers:
      - "soniox" (default) — custom SonioxSTTService with language detection
      - "deepgram"         — Pipecat built-in DeepgramSTTService
      - "whisper"          — Pipecat built-in WhisperSTTService (local)

    Returns a FrameProcessor that emits TranscriptionFrame/InterimTranscriptionFrame.
    """
    provider = STT_PROVIDER.lower()
    sr = sample_rate or STT_SAMPLE_RATE
    raw_hints = language_hints_override or STT_LANGUAGE_HINTS.split(",")
    lang_hints = []
    for hint in raw_hints:
        normalized = _normalize_language_hint(hint)
        if normalized:
            lang_hints.append(normalized)
    if not lang_hints:
        lang_hints = ["en"]

    if provider == "soniox":
        if not SONIOX_API_KEY:
            raise ValueError("SONIOX_API_KEY is required when STT_PROVIDER=soniox")
        logger.info(f"Creating STT service: Soniox (languages={lang_hints})")
        return SonioxSTTService(
            api_key=SONIOX_API_KEY,
            language_hints=lang_hints,
            sample_rate=sr,
        )

    elif provider == "deepgram":
        if not DEEPGRAM_API_KEY:
            raise ValueError("DEEPGRAM_API_KEY is required when STT_PROVIDER=deepgram")
        from pipecat.services.deepgram.stt import DeepgramSTTService
        logger.info(f"Creating STT service: Deepgram (language={lang_hints[0] if lang_hints else 'en'})")
        return DeepgramSTTService(
            api_key=DEEPGRAM_API_KEY,
            sample_rate=sr,
        )

    elif provider == "whisper":
        from pipecat.services.whisper.stt import WhisperSTTService
        logger.info("Creating STT service: Whisper (local)")
        return WhisperSTTService()

    else:
        raise ValueError(
            f"Unknown STT_PROVIDER: '{provider}'. "
            f"Supported: soniox, deepgram, whisper"
        )


def create_tts_service(sample_rate: int = None):
    """
    Create a TTS service based on TTS_PROVIDER env var.

    Supported providers:
      - "elevenlabs" (default) — ElevenLabs multilingual TTS
      - "svara"                — Custom Svara TTS
      - "openai"               — OpenAI TTS (tts-1 / tts-1-hd)

    Returns a TTSService (or compatible FrameProcessor).
    """
    provider = TTS_PROVIDER.lower()
    sr = sample_rate or TTS_SAMPLE_RATE

    if provider == "elevenlabs" and ELEVENLABS_API_KEY:
        logger.info(f"Creating TTS service: ElevenLabs (voice_gender={TTS_VOICE_GENDER})")
        return create_elevenlabs_tts(
            api_key=ELEVENLABS_API_KEY,
            voice_gender=TTS_VOICE_GENDER,
            sample_rate=sr,
        )

    elif provider == "svara":
        logger.info("Creating TTS service: Svara")
        tts_base_url = TTS_WS_URL.replace("ws://", "http://").replace("wss://", "https://").rsplit("/v1/", 1)[0]
        return SvaraTTSService(
            base_url=tts_base_url,
            api_key=TTS_WS_API_KEY,
            voice=DEFAULT_VOICE,
            max_tokens=TTS_MAX_TOKENS,
            streaming=True,
            sample_rate=sr,
        )

    elif provider == "openai":
        from pipecat.services.openai.tts import OpenAITTSService
        logger.info(f"Creating TTS service: OpenAI (voice={OPENAI_TTS_VOICE})")
        return OpenAITTSService(
            api_key=LLM_API_KEY,
            voice=OPENAI_TTS_VOICE,
            sample_rate=sr,
        )

    else:
        raise ValueError(
            f"Unknown TTS_PROVIDER: '{provider}'. "
            f"Supported: elevenlabs, svara, openai"
        )


def create_llm_service():
    """
    Create an LLM service based on LLM_PROVIDER env var.

    Supported providers:
      - "openai" (default) — OpenAI-compatible (works with vLLM, Azure, etc.)

    Any provider that exposes an OpenAI-compatible /v1/chat/completions
    endpoint works by setting LLM_BASE_URL and LLM_API_KEY.

    Returns an LLM service with a chat completions interface.
    """
    provider = LLM_PROVIDER.lower()

    if provider == "openai":
        logger.info(f"Creating LLM service: OpenAI-compatible (model={LLM_MODEL}, base_url={LLM_BASE_URL})")
        # Only send chat_template_kwargs to vLLM endpoints (not real OpenAI).
        # vLLM reasoning models need enable_thinking=False to avoid burning tokens
        # on reasoning_content before producing actual content tokens.
        is_openai_native = "api.openai.com" in LLM_BASE_URL
        extra = {}
        if not is_openai_native:
            extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        llm_params = OpenAILLMService.InputParams(extra=extra)
        return OpenAILLMService(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
            params=llm_params,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Supported: openai (covers vLLM, Azure, OpenRouter, etc.)"
        )


async def create_bot_pipeline(
    websocket,
    system_prompt: str = None,
    context_messages: list = None,
    mode: str = "text_and_audio",
    extra_processors: list = None,
    session_id: str = None,
    skip_greeting: bool = False,
    metrics_collector=None,
    is_classroom: bool = False,
    stt_language_hints: list[str] | None = None,
) -> tuple[PipelineTask, PipelineRunner, FastAPIWebsocketTransport]:
    """
    Create and configure the bot pipeline.

    Args:
        websocket: FastAPI WebSocket connection
        system_prompt: Custom system prompt (defaults to prompts/v0.md)
        context_messages: Prior conversation context
        mode: "text_and_audio" (default) or "text_only" — when text_only, TTS
              is skipped entirely and LLM text is forwarded to the client as JSON.
        extra_processors: Optional list of FrameProcessors to insert into the pipeline
              after instrumentation (e.g., classroom broadcaster taps).
        skip_greeting: If True, skip the greeting message (for reconnects with history).
        metrics_collector: Optional MetricsCollector for aggregated /metrics endpoint.
        is_classroom: Whether this is a classroom voice session (affects metric categorization).
        stt_language_hints: Optional STT language hints override for this session.

    Returns:
        Tuple of (PipelineTask, PipelineRunner, Transport)
    """
    text_only = mode == "text_only"
    logger.info(f"Creating pipeline (mode={mode})")

    # Create transport for WebSocket communication
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=not text_only,  # Disable audio output for text_only
            audio_in_sample_rate=16000,  # Input from client (16kHz)
            audio_out_sample_rate=24000,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=VAD_CONFIDENCE,
                    start_secs=VAD_START_SECS,
                    stop_secs=VAD_STOP_SECS,
                    min_volume=VAD_MIN_VOLUME,
                )
            ),
            serializer=ProtobufFrameSerializer(),
        ),
    )

    # === STT Service (provider-agnostic factory) ===
    stt = create_stt_service(sample_rate=STT_SAMPLE_RATE, language_hints_override=stt_language_hints)

    # === TTS Service (provider-agnostic factory) — skipped in text_only mode ===
    tts = None if text_only else create_tts_service(sample_rate=TTS_SAMPLE_RATE)

    # Voice state manager for runtime voice switching
    voice_state = VoiceStateManager(initial_gender=TTS_VOICE_GENDER)

    # === LLM Service (provider-agnostic factory) ===
    llm = create_llm_service()

    # === Register Function Handler for Voice Switching (only when TTS is active) ===
    from openai import NOT_GIVEN
    tools = NOT_GIVEN
    if tts is not None:
        async def handle_select_voice(params: FunctionCallParams):
            """Handle voice switching function call from LLM."""
            switch_mode = params.arguments.get("mode", "switch")
            new_gender, new_voice_id = voice_state.switch_voice(switch_mode)

            # Directly update the TTS voice and force reconnection
            tts._voice_id = new_voice_id
            await tts._disconnect()
            await tts._connect()

            logger.info(f"[VOICE SWITCH] Mode: {switch_mode}, New voice: {new_gender} ({new_voice_id})")

            # Return result to LLM so it can acknowledge the change
            await params.result_callback({
                "success": True,
                "new_voice": new_gender,
                "message": f"Voice switched to {new_gender}"
            })

        llm.register_function("select_voice", handle_select_voice)
        tools = ToolsSchema(standard_tools=[SELECT_VOICE_SCHEMA])

    # === Modern LLM Context Setup ===
    # Build initial messages with system prompt and optional user-provided context
    effective_prompt = system_prompt if system_prompt else get_default_system_prompt()
    messages = [{"role": "system", "content": effective_prompt}]
    logger.info(f"[PIPELINE] System prompt: {len(effective_prompt)} chars (version={PROMPT_VERSION})")
    # Pre-seed the greeting only for 1:1 tutor sessions (not classroom/discussion).
    # Classroom greeting is handled separately by send_first_join_greeting().
    if not skip_greeting and not is_classroom:
        messages.append({"role": "assistant", "content": GREETING_TEXT})
    if context_messages:
        messages.extend(context_messages)
        logger.info(f"[PIPELINE] Seeded LLM context with {len(context_messages)} prior messages")

    context = LLMContext(messages=messages, tools=tools)

    aggregator_pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(),
        assistant_params=LLMAssistantAggregatorParams(),
    )

    user_aggregator = aggregator_pair.user()
    assistant_aggregator = aggregator_pair.assistant()

    # Create pipeline instrumentor for comprehensive timing diagnostics
    transcript_logger = PipelineInstrumentor(
        name="PipelineInstrumentor",
        metrics_collector=metrics_collector,
        is_classroom=is_classroom,
    )

    # Filter out any [TEACHER_ACTION: ...] / [TUTOR_ACTION: ...] tags the LLM might generate
    action_tag_filter = ActionTagFilter(name="ActionTagFilter")

    # Create greeting processor to speak welcome message on connection
    # Skip greeting on reconnects (room already has conversation history)
    greeting_processor = GreetingProcessor(
        greeting_text=GREETING_TEXT,
        name="GreetingProcessor",
        skip=skip_greeting,
    )
    if skip_greeting:
        logger.info("[PIPELINE] Greeting will be skipped (reconnect with history)")

    # TextStreamForwarder sends LLM text as JSON to client in BOTH modes
    text_forwarder = TextStreamForwarder(
        websocket=websocket,
        text_only=text_only,
        name="TextStreamForwarder",
    )

    # UserTranscriptForwarder sends STT transcripts back to client as JSON
    user_transcript_forwarder = UserTranscriptForwarder(
        websocket=websocket,
        name="UserTranscriptForwarder",
    )

    # STTClarityGate intercepts low-clarity transcriptions and asks the
    # speaker to repeat — like a human teacher who says "I didn't catch that"
    clarity_gate = STTClarityGate(
        websocket=websocket,
        name="STTClarityGate",
        metrics_collector=metrics_collector,
    )

    # TextInputInjector allows typed text to be injected into the voice pipeline
    text_injector = None
    if session_id:
        text_injector = TextInputInjector(
            session_id=session_id,
            websocket=websocket,
            name="TextInputInjector",
        )
        register_text_injector(session_id, text_injector)

    # Build pipeline
    extra_processors = extra_processors or []
    # Insert text injector after STT, before user_transcript_forwarder
    injector_list = [text_injector] if text_injector else []
    if text_only:
        logger.info("[PIPELINE] text_only mode: TTS skipped, text streamed via JSON")
        pipeline = Pipeline([
            transport.input(),              # 1. Receive audio from client
            stt,                            # 2. Speech-to-text
            *injector_list,                 # 2b. Text injection point (typed text)
            user_transcript_forwarder,      # 3. Send user transcript to client
            clarity_gate,                   # 3b. Gate low-clarity STT (ask to repeat)
            user_aggregator,                # 4. Collect user messages and trigger LLM
            llm,                            # 5. Language model
            action_tag_filter,              # 5b. Strip [TEACHER_ACTION:...] tags from output
            transcript_logger,              # 6. Log conversation turns
            *extra_processors,              # 7. Optional taps (e.g., classroom)
            greeting_processor,             # 8. Inject greeting on StartFrame
            text_forwarder,                 # 9. Stream text as JSON (TTS skipped)
            transport.output(),             # 10. Transport (audio-in still works)
            assistant_aggregator,           # 11. Collect assistant responses for context
        ])
    else:
        logger.info("[PIPELINE] text_and_audio mode: text synced with TTS audio")
        # TextAudioSyncNotifier sits after TTS and triggers text release
        # to the client when each sentence's audio actually starts playing.
        text_audio_sync = TextAudioSyncNotifier(
            text_forwarder=text_forwarder,
            name="TextAudioSyncNotifier",
        )
        pipeline = Pipeline([
            transport.input(),              # 1. Receive audio from client
            stt,                            # 2. Speech-to-text
            *injector_list,                 # 2b. Text injection point (typed text)
            user_transcript_forwarder,      # 3. Send user transcript to client
            clarity_gate,                   # 3b. Gate low-clarity STT (ask to repeat)
            user_aggregator,                # 4. Collect user messages and trigger LLM
            llm,                            # 5. Language model
            action_tag_filter,              # 5b. Strip [TEACHER_ACTION:...] tags from output
            transcript_logger,              # 6. Log conversation turns
            *extra_processors,              # 7. Optional taps (e.g., classroom)
            greeting_processor,             # 8. Inject greeting on StartFrame
            text_forwarder,                 # 9. Queue text sentences (don't send yet)
            tts,                            # 10. Text-to-speech (audio)
            text_audio_sync,                # 10b. On TTSStarted → release queued text
            transport.output(),             # 11. Send audio to client
            assistant_aggregator,           # 12. Collect assistant responses for context
        ])

    # Create task and runner
    # Build interruption strategies: require minimum words before barge-in
    # to prevent false interruptions from acoustic echo on mobile devices.
    interruption_strategies = []
    if INTERRUPTION_MIN_WORDS > 0:
        interruption_strategies.append(
            MinWordsInterruptionStrategy(min_words=INTERRUPTION_MIN_WORDS)
        )
        logger.info(
            f"[PIPELINE] Interruption guard: require {INTERRUPTION_MIN_WORDS} words "
            f"before barge-in (prevents echo false-positives)"
        )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            interruption_strategies=interruption_strategies,
        ),
    )

    runner = PipelineRunner()

    return task, runner, transport


async def run_bot(
    websocket,
    system_prompt: str = None,
    context_messages: list = None,
    mode: str = "text_and_audio",
    extra_processors: list = None,
    session_id: str = None,
    skip_greeting: bool = False,
    metrics_collector=None,
    is_classroom: bool = False,
    stt_language_hints: list[str] | None = None,
):
    """
    Run the bot for a WebSocket connection.

    Args:
        websocket: FastAPI WebSocket connection
        system_prompt: Custom system prompt (defaults to prompts/v0.md)
        context_messages: Prior conversation context
        mode: "text_and_audio" (default) or "text_only"
        session_id: Unique session ID for text injection support
        skip_greeting: If True, skip the greeting message (for reconnects)
        metrics_collector: Optional MetricsCollector for aggregated /metrics endpoint.
        is_classroom: Whether this is a classroom voice session.
        stt_language_hints: Optional STT language hints override for this session.
    """
    task, runner, transport = await create_bot_pipeline(
        websocket,
        system_prompt=system_prompt,
        context_messages=context_messages,
        mode=mode,
        extra_processors=extra_processors,
        session_id=session_id,
        skip_greeting=skip_greeting,
        metrics_collector=metrics_collector,
        is_classroom=is_classroom,
        stt_language_hints=stt_language_hints,
    )

    # Add transport event handlers for proper RTVI protocol support
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat client disconnected")
        await task.cancel()

    try:
        await runner.run(task)
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise
    finally:
        # Clean up text injector
        if session_id:
            injector = get_text_injector(session_id)
            if injector:
                await injector.cleanup()
            unregister_text_injector(session_id)
        logger.info("Bot session ended")
