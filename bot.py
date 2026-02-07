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
import time

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
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from services.svara_tts import SvaraTTSService
from services.soniox_stt import SonioxSTTService
from services.elevenlabs_tts import create_elevenlabs_tts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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

    def __init__(self, name: str = "PipelineInstrumentor", **kwargs):
        super().__init__(name=name, **kwargs)
        self._turn_count = 0
        self._llm_buffer = ""

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

        logger.info(f"")
        logger.info(f"{'─' * 70}")
        logger.info(f"[METRICS] TURN {self._turn_count} SUMMARY")
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


class TextStreamForwarder(FrameProcessor):
    """
    Sends LLM text to the client as JSON messages for real-time text rendering.

    Used in BOTH modes:
      - text_and_audio: client gets text + audio simultaneously (text for rendering,
        audio for playback). Frames still flow downstream to TTS.
      - text_only: client gets text only; TTSSpeakFrames are consumed here
        (no TTS downstream to process them).

    JSON messages sent:
      - {"type": "bot_text", "text": "chunk", "streaming": true}  — per LLM token
      - {"type": "bot_text_complete", "text": "full response"}     — end of response
    """

    def __init__(self, websocket, text_only: bool = False, name: str = "TextStreamForwarder", **kwargs):
        super().__init__(name=name, **kwargs)
        self._websocket = websocket
        self._text_only = text_only
        self._current_response = ""
        self._in_response = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._in_response = True
            self._current_response = ""
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSSpeakFrame):
            # Greeting / direct-speak frames — send as text
            try:
                await self._websocket.send_json({
                    "type": "bot_text_complete",
                    "text": frame.text,
                })
                logger.info(f"[TEXT_STREAM] Greeting text sent: '{frame.text[:120]}'")
            except Exception as e:
                logger.debug(f"[TEXT_STREAM] Failed to send greeting text: {e}")

            if self._text_only:
                # No TTS downstream — consume the frame
                return
            # In text_and_audio mode, push downstream so TTS synthesizes it
            await self.push_frame(frame, direction)

        elif isinstance(frame, TextFrame) and self._in_response:
            self._current_response += frame.text
            # Stream each token to client for real-time text rendering
            try:
                await self._websocket.send_json({
                    "type": "bot_text",
                    "text": frame.text,
                    "streaming": True,
                })
            except Exception as e:
                logger.debug(f"[TEXT_STREAM] Failed to send text chunk: {e}")
            # Always push downstream (to TTS in audio mode, or to assistant aggregator)
            await self.push_frame(frame, direction)

        elif isinstance(frame, LLMFullResponseEndFrame):
            self._in_response = False
            # Send complete response for client to finalize display
            try:
                await self._websocket.send_json({
                    "type": "bot_text_complete",
                    "text": self._current_response,
                })
                logger.info(f"[TEXT_STREAM] Complete response: '{self._current_response[:120]}{'...' if len(self._current_response) > 120 else ''}'")
            except Exception as e:
                logger.debug(f"[TEXT_STREAM] Failed to send complete text: {e}")
            self._current_response = ""
            await self.push_frame(frame, direction)

        else:
            # Forward everything else unchanged
            await self.push_frame(frame, direction)


class GreetingProcessor(FrameProcessor):
    """
    Processor that speaks a greeting when the pipeline starts.
    Injects a TTSSpeakFrame on StartFrame which flows to TTS for synthesis.
    """

    def __init__(self, greeting_text: str, name: str = "GreetingProcessor", **kwargs):
        super().__init__(name=name, **kwargs)
        self._greeting_text = greeting_text
        self._greeting_spoken = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # On StartFrame, inject greeting text for TTS
        if isinstance(frame, StartFrame) and not self._greeting_spoken:
            self._greeting_spoken = True
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
STT_LANGUAGE_HINTS = os.getenv("STT_LANGUAGE_HINTS", "en,hi,ta,kn")  # Comma-separated
STT_SAMPLE_RATE = int(os.getenv("STT_SAMPLE_RATE", "16000"))

# --- TTS ---
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs")    # "elevenlabs" | "svara" | "openai"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
TTS_VOICE_GENDER = os.getenv("TTS_VOICE_GENDER", "female")  # "female" | "male"
TTS_WS_URL = os.getenv("TTS_WS_URL", "ws://svara-tts/v1/audio/text-to-speech/stream")
TTS_WS_API_KEY = os.getenv("TTS_WS_API_KEY", "")          # API key for Svara TTS auth
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "nova")  # For OpenAI TTS provider

# --- Voice ---
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "en_female")   # Default voice for Svara TTS
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "auto")

# VAD params as env vars for tuning without code change
VAD_CONFIDENCE = float(os.getenv("VAD_CONFIDENCE", "0.7"))
VAD_START_SECS = float(os.getenv("VAD_START_SECS", "0.2"))
VAD_STOP_SECS = float(os.getenv("VAD_STOP_SECS", "0.6"))
VAD_MIN_VOLUME = float(os.getenv("VAD_MIN_VOLUME", "0.6"))

# Prompt configuration
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v3")


def load_system_prompt(version: str = PROMPT_VERSION) -> str:
    """Load system prompt from prompts directory."""
    prompt_path = os.path.join(PROMPT_DIR, f"{version}.md")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning(f"Prompt file not found: {prompt_path}, using default")
        return "You are a helpful voice assistant. Keep responses concise."


def get_default_system_prompt() -> str:
    """Get default system prompt (uses PROMPT_VERSION env var, defaults to v3)."""
    return load_system_prompt(version=PROMPT_VERSION)


# Greeting text spoken when connection is established
# NOTE: Single sentence avoids TTS splitting into multiple audio segments
GREETING_TEXT = os.getenv("GREETING_TEXT",
    "Namaste! I'm Mira, your study buddy. I speak English, Hindi, Tamil, and Kannada. Ask me anything!")


# ─────────────────────────────────────────────────────────────────────
# Provider factories — swap provider via env var, no code changes.
# ─────────────────────────────────────────────────────────────────────

def create_stt_service(sample_rate: int = None):
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
    lang_hints = [h.strip() for h in STT_LANGUAGE_HINTS.split(",") if h.strip()]

    if provider == "soniox":
        if not SONIOX_API_KEY:
            raise ValueError("SONIOX_API_KEY is required when STT_PROVIDER=soniox")
        logger.info(f"Creating STT service: Soniox (languages={lang_hints})")
        return SonioxSTTService(
            api_key=SONIOX_API_KEY,
            language_hints=lang_hints,
            enable_speaker_diarization=True,
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
        return OpenAILLMService(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
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
    stt = create_stt_service(sample_rate=STT_SAMPLE_RATE)

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
    # Pre-seed the greeting as an assistant message so the LLM knows it was spoken,
    # even if the greeting audio gets interrupted by the user speaking early.
    messages.append({"role": "assistant", "content": GREETING_TEXT})
    if context_messages:
        messages.extend(context_messages)

    context = LLMContext(messages=messages, tools=tools)

    aggregator_pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(),
        assistant_params=LLMAssistantAggregatorParams(),
    )

    user_aggregator = aggregator_pair.user()
    assistant_aggregator = aggregator_pair.assistant()

    # Create pipeline instrumentor for comprehensive timing diagnostics
    transcript_logger = PipelineInstrumentor(name="PipelineInstrumentor")

    # Create greeting processor to speak welcome message on connection
    greeting_processor = GreetingProcessor(
        greeting_text=GREETING_TEXT,
        name="GreetingProcessor",
    )

    # TextStreamForwarder sends LLM text as JSON to client in BOTH modes
    text_forwarder = TextStreamForwarder(
        websocket=websocket,
        text_only=text_only,
        name="TextStreamForwarder",
    )

    # Build pipeline
    extra_processors = extra_processors or []
    if text_only:
        logger.info("[PIPELINE] text_only mode: TTS skipped, text streamed via JSON")
        # Flow: input -> STT -> aggregator -> LLM -> logger -> greeting -> text_forwarder -> output -> assistant_aggregator
        pipeline = Pipeline([
            transport.input(),          # 1. Receive audio from client
            stt,                        # 2. Speech-to-text
            user_aggregator,            # 3. Collect user messages and trigger LLM
            llm,                        # 4. Language model
            transcript_logger,          # 5. Log conversation turns
            *extra_processors,          # 6. Optional taps (e.g., classroom)
            greeting_processor,         # 6. Inject greeting on StartFrame
            text_forwarder,             # 7. Stream text as JSON (TTS skipped)
            transport.output(),         # 8. Transport (audio-in still works)
            assistant_aggregator,       # 9. Collect assistant responses for context
        ])
    else:
        logger.info("[PIPELINE] text_and_audio mode: text streamed + TTS audio")
        # Flow: input -> STT -> aggregator -> LLM -> logger -> greeting -> text_forwarder -> TTS -> output -> assistant_aggregator
        pipeline = Pipeline([
            transport.input(),          # 1. Receive audio from client
            stt,                        # 2. Speech-to-text
            user_aggregator,            # 3. Collect user messages and trigger LLM
            llm,                        # 4. Language model
            transcript_logger,          # 5. Log conversation turns
            *extra_processors,          # 6. Optional taps (e.g., classroom)
            greeting_processor,         # 6. Inject greeting on StartFrame
            text_forwarder,             # 7. Stream text as JSON to client
            tts,                        # 8. Text-to-speech (audio)
            transport.output(),         # 9. Send audio to client
            assistant_aggregator,       # 10. Collect assistant responses for context
        ])

    # Create task and runner
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
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
):
    """
    Run the bot for a WebSocket connection.

    Args:
        websocket: FastAPI WebSocket connection
        system_prompt: Custom system prompt (defaults to prompts/v0.md)
        context_messages: Prior conversation context
        mode: "text_and_audio" (default) or "text_only"
    """
    task, runner, transport = await create_bot_pipeline(
        websocket,
        system_prompt=system_prompt,
        context_messages=context_messages,
        mode=mode,
        extra_processors=extra_processors,
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
        logger.info("Bot session ended")
