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

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TTSStoppedFrame,
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

from services.svara_tts import SvaraTTSService
from services.soniox_stt import SonioxSTTService
from services.elevenlabs_tts import create_elevenlabs_tts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TranscriptLogger(FrameProcessor):
    """
    Enhanced logging processor for debugging the voice pipeline.
    Tracks conversation turns and logs all important frame events.
    """

    def __init__(self, name: str = "TranscriptLogger", **kwargs):
        super().__init__(name=name, **kwargs)
        self._llm_buffer = ""  # Buffer to accumulate LLM text chunks
        self._turn_count = 0   # Track conversation turns

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Log STT transcriptions (final) - increment turn count
        if isinstance(frame, TranscriptionFrame):
            self._turn_count += 1
            logger.info(f"[TURN {self._turn_count}] USER: '{frame.text}'")

        # Log STT interim transcriptions
        elif isinstance(frame, InterimTranscriptionFrame):
            logger.debug(f"[STT INTERIM] '{frame.text}'")

        # Log LLM text chunks (accumulate for full response)
        elif isinstance(frame, TextFrame):
            self._llm_buffer += frame.text
            logger.debug(f"[LLM CHUNK] '{frame.text}'")

        # Log when TTS finishes speaking - output full bot response
        elif isinstance(frame, TTSStoppedFrame):
            if self._llm_buffer:
                logger.info(f"[TURN {self._turn_count}] BOT: '{self._llm_buffer}'")
                self._llm_buffer = ""

        # Log barge-in interruptions
        elif isinstance(frame, StartInterruptionFrame):
            logger.warning(f"[BARGE-IN] User interrupted bot speech")
            self._llm_buffer = ""  # Clear partial response

        # Log VAD events - VERY CLEAR markers for debugging
        elif isinstance(frame, UserStartedSpeakingFrame):
            logger.info(f"")
            logger.info(f"{'='*60}")
            logger.info(f">>> VAD: USER STARTED SPEAKING <<<")
            logger.info(f"{'='*60}")

        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info(f"{'='*60}")
            logger.info(f">>> VAD: USER STOPPED SPEAKING <<<")
            logger.info(f"{'='*60}")
            logger.info(f"")

        # Forward the frame downstream
        await self.push_frame(frame, direction)


class GreetingProcessor(FrameProcessor):
    """
    Processor that speaks a greeting when the pipeline starts.
    Injects a TextFrame on StartFrame which flows to TTS for synthesis.
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
            logger.info(f"[GREETING] Speaking: '{self._greeting_text}'")
            # Push TextFrame downstream - will flow to TTS for synthesis
            await self.push_frame(TextFrame(text=self._greeting_text), FrameDirection.DOWNSTREAM)

        # Forward the original frame
        await self.push_frame(frame, direction)


# Environment configuration
ASR_WS_URL = os.getenv("ASR_WS_URL", "ws://localhost:8082/v1/audio/speech-to-text/stream")
TTS_WS_URL = os.getenv("TTS_WS_URL", "ws://vllm-svara-tts/v1/audio/text-to-speech/stream")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://vllm-gpt-oss-120b/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "DUMMY_KEY")

# STT configuration (Soniox)
SONIOX_API_KEY = os.getenv("SONIOX_API_KEY", "")

# TTS Provider configuration
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs")  # "elevenlabs" or "svara"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
TTS_VOICE_GENDER = os.getenv("TTS_VOICE_GENDER", "female")  # "female" or "male"

# Voice configuration (for Svara TTS fallback)
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "hi_male")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "auto")

# System prompt for the assistant - handles multilingual input, always responds in English
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """You are a helpful, friendly AI voice assistant.

IMPORTANT INSTRUCTIONS:
1. The user may speak to you in ANY language (Hindi, Kannada, Tamil, Telugu, Bengali, etc.)
2. You MUST understand what they said and ALWAYS respond in English only
3. Keep responses concise (1-3 sentences) - you are in a voice conversation
4. Be natural, warm, and conversational
5. If you don't understand something, politely ask for clarification in English

You are excellent at understanding multiple languages but you always speak English.""")

# Greeting text spoken when connection is established
GREETING_TEXT = os.getenv("GREETING_TEXT",
    "Hello! I'm your voice assistant. I can understand you in any language, but I'll respond in English. How can I help you today?")


async def create_bot_pipeline(
    websocket,
    sample_rate: int = 16000,
    voice: str = DEFAULT_VOICE,
    language: str = DEFAULT_LANGUAGE,
    context_messages: list = None,
) -> tuple[PipelineTask, PipelineRunner]:
    """
    Create and configure the bot pipeline.

    Args:
        websocket: FastAPI WebSocket connection
        sample_rate: Audio sample rate
        voice: TTS voice ID
        language: STT language code

    Returns:
        Tuple of (PipelineTask, PipelineRunner)
    """
    logger.info(f"Creating pipeline: language={language}, voice={DEFAULT_VOICE}, interim_results=False")

    # Create transport for WebSocket communication
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=sample_rate,  # Input from client (16kHz)
            audio_out_sample_rate=24000,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.7, # Confidence threshold for speech detection. Higher values make detection more strict. Must be between 0 and 1.
                    start_secs=0.2, # Time in seconds that speech must be detected before transitioning to SPEAKING state.
                    stop_secs=0.6, # Time in seconds of silence required before transitioning back to QUIET state.
                    min_volume=0.6, # Minimum audio volume threshold for speech detection. Must be between 0 and 1.
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
        language_hints=["en", "hi", "ta", "te", "bn", "kn", "mr", "ml", "gu", "pa"],
        enable_speaker_diarization=True,
        sample_rate=sample_rate,
    )

    # === TTS Service Selection ===
    if TTS_PROVIDER == "elevenlabs":
        if not ELEVENLABS_API_KEY:
            logger.warning("ELEVENLABS_API_KEY not set, falling back to Svara TTS")
            tts_base_url = TTS_WS_URL.replace("ws://", "http://").replace("wss://", "https://").rsplit("/v1/", 1)[0]
            tts = SvaraTTSService(
                base_url=tts_base_url,
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
            voice=DEFAULT_VOICE,
            streaming=True,
            sample_rate=24000,
        )

    # Initialize LLM service
    llm = OpenAILLMService(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
    )

    # === Modern LLM Context Setup ===
    # Build initial messages with system prompt and optional user-provided context
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context_messages:
        messages.extend(context_messages)

    context = LLMContext(messages=messages)

    aggregator_pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(),
        assistant_params=LLMAssistantAggregatorParams(),
    )

    user_aggregator = aggregator_pair.user()
    assistant_aggregator = aggregator_pair.assistant()

    # Create transcript logger to capture LLM responses and STT transcripts
    transcript_logger = TranscriptLogger(name="PipelineLogger")

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

    return task, runner


async def run_bot(
    websocket,
    sample_rate: int = 16000,
    voice: str = DEFAULT_VOICE,
    language: str = DEFAULT_LANGUAGE,
    context_messages: list = None,
):
    """
    Run the bot for a WebSocket connection.

    Args:
        websocket: FastAPI WebSocket connection
        sample_rate: Audio sample rate
        voice: TTS voice ID
        language: STT language code
    """
    task, runner = await create_bot_pipeline(
        websocket,
        sample_rate=sample_rate,
        voice=voice,
        language=language,
        context_messages=context_messages,
    )

    try:
        await runner.run(task)
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise
    finally:
        logger.info("Bot session ended")
