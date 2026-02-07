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
    """Manages the current voice state for voice switching functionality."""

    VOICE_IDS = {
        "female": "2zRM7PkgwBPiau2jvVXc",  # Monika Sogam
        "male": "siw1N9V8LmYeEWKyWBxv",    # Ruhaan
    }

    def __init__(self, initial_gender: str = "female"):
        self._current_gender = initial_gender

    @property
    def current_gender(self) -> str:
        return self._current_gender

    @property
    def current_voice_id(self) -> str:
        return self.VOICE_IDS[self._current_gender]

    def switch_voice(self, mode: str) -> tuple[str, str]:
        """Switch voice based on mode. Returns (new_gender, new_voice_id)."""
        if mode == "switch":
            self._current_gender = "male" if self._current_gender == "female" else "female"
        elif mode in ("male", "female"):
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


# Environment configuration
TTS_WS_URL = os.getenv("TTS_WS_URL", "ws://svara-tts/v1/audio/text-to-speech/stream")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://vllm-gpt-oss-120b/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "DUMMY_KEY")

# STT configuration (Soniox)
SONIOX_API_KEY = os.getenv("SONIOX_API_KEY", "")

# TTS Provider configuration
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs")  # "elevenlabs" or "svara"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
TTS_VOICE_GENDER = os.getenv("TTS_VOICE_GENDER", "female")  # "female" or "male"
TTS_WS_API_KEY = os.getenv("TTS_WS_API_KEY", "")  # API key for Svara TTS auth

# Voice configuration (hardcoded)
DEFAULT_VOICE = "en_female"  # Default voice for Svara TTS
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


async def create_bot_pipeline(
    websocket,
    system_prompt: str = None,
    context_messages: list = None,
) -> tuple[PipelineTask, PipelineRunner, FastAPIWebsocketTransport]:
    """
    Create and configure the bot pipeline.

    Args:
        websocket: FastAPI WebSocket connection
        system_prompt: Custom system prompt (defaults to prompts/v0.md)
        context_messages: Prior conversation context

    Returns:
        Tuple of (PipelineTask, PipelineRunner, Transport)
    """
    logger.info("Creating pipeline")

    # Create transport for WebSocket communication
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
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

    # === STT Service (Soniox only) ===
    if not SONIOX_API_KEY:
        raise ValueError("SONIOX_API_KEY environment variable is required")

    logger.info("Using Soniox STT provider")
    stt = SonioxSTTService(
        api_key=SONIOX_API_KEY,
        language_hints=["en", "hi", "ta", "kn"],
        enable_speaker_diarization=True,
        sample_rate=16000,
    )

    # === TTS Service Selection ===
    if TTS_PROVIDER == "elevenlabs":
        if not ELEVENLABS_API_KEY:
            logger.warning("ELEVENLABS_API_KEY not set, falling back to Svara TTS")
            tts_base_url = TTS_WS_URL.replace("ws://", "http://").replace("wss://", "https://").rsplit("/v1/", 1)[0]
            tts = SvaraTTSService(
                base_url=tts_base_url,
                api_key=TTS_WS_API_KEY,
                voice=DEFAULT_VOICE,
                streaming=True,
                sample_rate=24000,
            )
        else:
            logger.info(f"Using ElevenLabs TTS provider (voice_gender={TTS_VOICE_GENDER})")
            tts = create_elevenlabs_tts(
                api_key=ELEVENLABS_API_KEY,
                voice_gender=TTS_VOICE_GENDER,
                sample_rate=24000,
            )
    else:
        logger.info("Using Svara TTS provider")
        tts_base_url = TTS_WS_URL.replace("ws://", "http://").replace("wss://", "https://").rsplit("/v1/", 1)[0]
        tts = SvaraTTSService(
            base_url=tts_base_url,
            api_key=TTS_WS_API_KEY,
            voice=DEFAULT_VOICE,
            streaming=True,
            sample_rate=24000,
        )

    # Voice state manager for runtime voice switching
    voice_state = VoiceStateManager(initial_gender=TTS_VOICE_GENDER)

    # Initialize LLM service
    llm = OpenAILLMService(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
    )

    # === Register Function Handler for Voice Switching ===
    async def handle_select_voice(params: FunctionCallParams):
        """Handle voice switching function call from LLM."""
        mode = params.arguments.get("mode", "switch")
        new_gender, new_voice_id = voice_state.switch_voice(mode)

        # Directly update the TTS voice and force reconnection
        tts._voice_id = new_voice_id
        await tts._disconnect()
        await tts._connect()

        logger.info(f"[VOICE SWITCH] Mode: {mode}, New voice: {new_gender} ({new_voice_id})")

        # Return result to LLM so it can acknowledge the change
        await params.result_callback({
            "success": True,
            "new_voice": new_gender,
            "message": f"Voice switched to {new_gender}"
        })

    llm.register_function("select_voice", handle_select_voice)

    # === Tools Schema for Function Calling ===
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

    # Build pipeline with greeting support
    # Flow: input -> STT -> user_aggregator -> LLM -> logger -> greeting -> TTS -> output -> assistant_aggregator
    pipeline = Pipeline([
        transport.input(),          # 1. Receive audio from client
        stt,                        # 2. Speech-to-text (IndicASR, finalize-only mode)
        user_aggregator,            # 3. Collect user messages and trigger LLM
        llm,                        # 4. Language model (GPT-OSS, responds in English)
        transcript_logger,          # 5. Log conversation turns
        greeting_processor,         # 6. Inject greeting on StartFrame
        tts,                        # 7. Text-to-speech (Svara, hi_male voice)
        transport.output(),         # 8. Send audio to client
        assistant_aggregator,       # 9. Collect assistant responses for context
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
):
    """
    Run the bot for a WebSocket connection.

    Args:
        websocket: FastAPI WebSocket connection
        system_prompt: Custom system prompt (defaults to prompts/v0.md)
        context_messages: Prior conversation context
    """
    task, runner, transport = await create_bot_pipeline(
        websocket,
        system_prompt=system_prompt,
        context_messages=context_messages,
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
