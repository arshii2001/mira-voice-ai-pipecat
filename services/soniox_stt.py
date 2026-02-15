"""
Custom STT Service for Soniox Realtime WebSocket API.

This service integrates with the Soniox realtime STT server for
multilingual ASR via WebSocket streaming.

KEY FEATURES:
- Persistent WebSocket connection (always listening)
- Language identification across 50+ languages
- Endpoint detection for utterance segmentation
- Barge-in support
- Clarity scoring — heuristic quality signal per utterance

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
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import websockets
from websockets.asyncio.client import connect as websocket_connect


def detect_language_from_script(text: str) -> Optional[str]:
    """Detect language from Unicode script of the text (fallback when STT doesn't report language).

    Strips speaker labels and ASCII before checking the dominant non-Latin script.
    """
    # Remove speaker labels like "Speaker 1: "
    cleaned = re.sub(r"Speaker\s+\d+:\s*", "", text)
    # Remove ASCII / Latin characters and punctuation — only look at non-Latin chars
    non_latin = re.sub(r"[\x00-\x7F]", "", cleaned)
    if not non_latin:
        return "en"  # All ASCII → English

    # Count characters by Unicode block
    devanagari = sum(1 for c in non_latin if "\u0900" <= c <= "\u097F")
    tamil = sum(1 for c in non_latin if "\u0B80" <= c <= "\u0BFF")

    counts = {"hi": devanagari, "ta": tamil}
    best = max(counts, key=counts.get)
    if counts[best] > 0:
        return best
    return None  # Unknown script

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
LANG_LABELS = {"en": "English", "hi": "Hindi", "ta": "Tamil"}


@dataclass
class SonioxConfig:
    """Configuration for Soniox STT service."""
    api_key: str
    model: str = "stt-rt-v3"
    sample_rate: int = 16000
    num_channels: int = 1
    include_nonfinal: bool = True
    enable_language_identification: bool = True
    enable_endpoint_detection: bool = True
    language_hints: List[str] = field(default_factory=lambda: ["en", "hi"])
    language_hints_strict: bool = True


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

    CLARITY SCORING:
    - Each final transcription gets a heuristic clarity score (0.0–1.0)
    - Score is attached to the TranscriptionFrame as ``clarity_score``
    - Downstream processors (STTClarityGate) use this to decide whether
      to pass the transcription to the LLM or ask for clarification

    This processor:
    - Receives InputAudioRawFrame from the transport
    - Sends audio to Soniox via persistent WebSocket
    - Emits TranscriptionFrame and InterimTranscriptionFrame
    - Handles interruptions without connection reset
    """

    # Pre-compiled regex for filler word detection in clarity scoring
    _FILLER_RE = re.compile(r'^(uh|um|hmm|hm|ah|oh|er|erm)$', re.IGNORECASE)

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "stt-rt-v3",
        sample_rate: int = 16000,
        num_channels: int = 1,
        language_hints: Optional[List[str]] = None,
        include_nonfinal: bool = True,
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
        self._detected_language: Optional[str] = None
        self._last_logged_raw_language: Optional[str] = None

        # Barge-in state
        self._user_speaking = False
        self._bot_speaking = False
        self._interrupted = False
        self._muted = False
        self._keepalive_task: Optional[asyncio.Task] = None
        self._idle_keepalive_task: Optional[asyncio.Task] = None
        self._pipeline_stopped = False
        self._first_speech_received = False  # True after first UserStartedSpeakingFrame
        self._pipeline_start_time: float = 0.0  # monotonic time when pipeline started

        # Reconnect audio buffer — holds audio frames received while Soniox
        # is reconnecting, so initial words are not lost.
        self._reconnect_buffer: list[bytes] = []
        _MAX_RECONNECT_BUFFER = 50  # ~1s of audio at 20ms chunks
        self._MAX_RECONNECT_BUFFER = _MAX_RECONNECT_BUFFER

        # Rate-limit tracking — Soniox returns 429 when too many concurrent
        # connections are open.  When we detect a 429, we back off proactive
        # reconnects to avoid burning the quota on idle sessions.
        self._rate_limited = False
        self._rate_limit_backoff = 30.0  # seconds to wait after a 429 before retrying
        self._last_rate_limit_time: float = 0.0  # monotonic timestamp of last 429

        # Clarity scoring state — tracks per-utterance quality signals
        self._speech_start_time: float = 0.0  # monotonic time when user started speaking
        self._audio_bytes_sent: int = 0       # bytes of audio sent for current utterance
        self._utterance_token_count: int = 0  # number of tokens in current final result

        # Clarity metrics (cumulative across session)
        self._clarity_total_utterances: int = 0
        self._clarity_low_count: int = 0
        self._clarity_scores: list[float] = []

    @staticmethod
    def _normalize_lang_code(lang: Optional[str]) -> Optional[str]:
        """Normalize provider language codes to app-supported short codes."""
        if not lang:
            return None
        raw = str(lang).strip().lower()
        # Collapse region/script variants (e.g. en-US, zh-CN)
        base = raw.split("-", 1)[0].split("_", 1)[0]
        alias_map = {
            "eng": "en",
            "english": "en",
            "hin": "hi",
            "hindi": "hi",
            "tam": "ta",
            "tamil": "ta",
        }
        return alias_map.get(base, base)

    def _resolved_language(self, text: str) -> str:
        """Resolve final language safely, constrained to configured hints."""
        hint_set = {
            self._normalize_lang_code(h) or h.strip().lower()
            for h in (self._config.language_hints or [])
            if isinstance(h, str) and h.strip()
        }

        detected = self._normalize_lang_code(self._detected_language)
        if detected and (not hint_set or detected in hint_set):
            return detected
        if detected and hint_set and detected not in hint_set:
            logger.debug(
                f"Ignoring detected language '{detected}' outside allowed hints {sorted(hint_set)}"
            )

        script_lang = detect_language_from_script(text)
        if script_lang and (not hint_set or script_lang in hint_set):
            return script_lang

        # Fall back to first configured hint, then English.
        if self._config.language_hints:
            first_hint = self._normalize_lang_code(self._config.language_hints[0])
            if first_hint:
                return first_hint
        return "en"

    def _allowed_hint_set(self) -> set[str]:
        """Normalized language hints for this session."""
        return {
            self._normalize_lang_code(h) or h.strip().lower()
            for h in (self._config.language_hints or [])
            if isinstance(h, str) and h.strip()
        }

    @staticmethod
    def _is_latin_extended(char: str) -> bool:
        """Allow accented Latin text when English is allowed."""
        return (
            "\u00C0" <= char <= "\u00FF"   # Latin-1 Supplement letters
            or "\u0100" <= char <= "\u017F"  # Latin Extended-A
            or "\u0180" <= char <= "\u024F"  # Latin Extended-B
        )

    def _contains_disallowed_script(self, text: str) -> bool:
        """True when text contains non-ASCII script outside allowed language hints."""
        allowed = self._allowed_hint_set()
        if not allowed:
            return False

        for c in text:
            # ASCII is always okay
            if ord(c) <= 0x7F:
                continue

            # Allow common Unicode punctuation/symbols if English is allowed.
            if "en" in allowed and "\u2000" <= c <= "\u206F":
                continue

            if "\u0900" <= c <= "\u097F":  # Devanagari
                if "hi" in allowed:
                    continue
                return True

            if "\u0B80" <= c <= "\u0BFF":  # Tamil
                if "ta" in allowed:
                    continue
                return True

            # Latin diacritics should be treated as English-compatible.
            if "en" in allowed and self._is_latin_extended(c):
                continue

            # Any other non-ASCII script is disallowed for this pipeline.
            return True

        return False

    def _compute_clarity_score(self, text: str, tokens: list, speech_duration_s: float) -> float:
        """Compute a heuristic clarity score (0.0–1.0) for a final transcription.

        Soniox's realtime API does not expose per-token confidence scores,
        so we use surrogate signals that correlate with transcription quality:

        1. **Word count** — very short utterances (1 word) from multi-second
           speech often indicate the STT only caught a fragment.
        2. **Speech-to-text ratio** — if the user spoke for 5 seconds but STT
           produced only 2 words, something was lost (expected ~2-3 words/sec
           for conversational speech).
        3. **Token density** — Soniox tokens include timing info. Very few
           tokens relative to speech duration suggests dropped content.
        4. **Repetition / filler** — repeated single characters or filler
           sounds ("uh", "um", "hmm") indicate unclear speech.

        Returns a score in [0.0, 1.0] where:
          - >= 0.75: High clarity — proceed normally
          - 0.50–0.75: Medium clarity — proceed but log a warning
          - < 0.50: Low clarity — should ask for clarification
        """
        score = 1.0
        words = text.split()
        word_count = len(words)

        # ── Signal 1: Very short utterance ──
        # Single-word transcriptions from >1.5s of speech are suspicious
        if word_count <= 1 and speech_duration_s > 1.5:
            score -= 0.35
        elif word_count <= 2 and speech_duration_s > 3.0:
            score -= 0.25

        # ── Signal 2: Speech-to-text ratio ──
        # Conversational speech is ~2-3 words/sec. If ratio is very low,
        # the STT likely missed content.
        if speech_duration_s > 0.5:
            words_per_sec = word_count / speech_duration_s
            if words_per_sec < 0.5:  # Less than 0.5 words/sec is very sparse
                score -= 0.30
            elif words_per_sec < 1.0:  # Less than 1 word/sec is sparse
                score -= 0.15

        # ── Signal 3: Filler / repetition detection ──
        # Single-char words (excluding common ones like "I", "a") or
        # repeated tokens suggest garbled speech
        filler_count = sum(1 for w in words if self._FILLER_RE.match(w))
        if word_count > 0 and filler_count / word_count > 0.5:
            score -= 0.25

        # ── Signal 4: Very short text from long speech ──
        # If user spoke for >3 seconds but text is <10 chars, likely garbled
        if speech_duration_s > 3.0 and len(text) < 10:
            score -= 0.20

        # ── Signal 5: Token timing gaps ──
        # If Soniox returned tokens with very large gaps, content was likely lost
        if len(tokens) >= 2:
            total_duration_ms = 0
            for t in tokens:
                total_duration_ms += t.get("duration_ms", 0)
            if speech_duration_s > 0 and total_duration_ms > 0:
                coverage = (total_duration_ms / 1000.0) / speech_duration_s
                if coverage < 0.3:  # Tokens cover less than 30% of speech
                    score -= 0.20

        return max(0.0, min(1.0, score))

    async def start(self, frame: StartFrame):
        """Start the STT service. Connect eagerly for lower first-turn latency."""
        self._pipeline_start_time = time.monotonic()
        logger.info("[SONIOX_TIMING] Pipeline START — connecting eagerly")
        t0 = time.monotonic()
        try:
            await self._connect()
            connect_ms = (time.monotonic() - t0) * 1000
            logger.info(f"[SONIOX_TIMING] Eager connect completed in {connect_ms:.0f}ms")
            # Start keepalive to prevent Soniox from dropping the connection
            # during ANY idle gap (before first speech, between turns, etc.).
            self._idle_keepalive_task = asyncio.create_task(self._idle_keepalive_loop())
            logger.info("[SONIOX_TIMING] Keepalive started (pings every 5s during idle gaps)")
        except Exception as e:
            logger.warning(f"Eager Soniox connect failed, will retry on first speech: {e}")

    async def stop(self, frame: EndFrame):
        """Stop the STT service and close connection."""
        self._pipeline_stopped = True
        # Cancel idle keepalive if still running
        if self._idle_keepalive_task and not self._idle_keepalive_task.done():
            self._idle_keepalive_task.cancel()
            self._idle_keepalive_task = None
        # Send end-of-session signal before disconnecting
        ws = self._websocket
        if self._connected and ws is not None:
            try:
                await ws.send("")
                logger.info("Sent end-of-session signal to Soniox")
            except Exception:
                pass
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel — disconnect Soniox to free the concurrent connection slot.

        Previously we kept the connection open for barge-in, but when the
        pipeline is fully cancelled (e.g. page refresh / client disconnect),
        the EndFrame never arrives and the Soniox WebSocket lingers until
        idle timeout, blocking new sessions from connecting.
        """
        self._interrupted = True
        self._pipeline_stopped = True
        # Cancel idle keepalive
        if self._idle_keepalive_task and not self._idle_keepalive_task.done():
            self._idle_keepalive_task.cancel()
            self._idle_keepalive_task = None
        await self._disconnect()

    async def _connect(self):
        """Establish persistent WebSocket connection to Soniox server."""
        async with self._connect_lock:
            if self._connected and self._websocket:
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
                    "language_hints_strict": self._config.language_hints_strict,
                    "enable_language_identification": self._config.enable_language_identification,
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

                    # Check for 429 rate-limit in initial response
                    try:
                        init_data = json.loads(init_response)
                        if init_data.get("error_code") == 429:
                            self._rate_limited = True
                            self._last_rate_limit_time = time.monotonic()
                            logger.warning(
                                f"[SONIOX] ⚠️  Rate-limited (429): {init_data.get('error_message', 'unknown')} — "
                                f"backing off proactive reconnects for {self._rate_limit_backoff}s"
                            )
                            # Close this connection immediately — Soniox will drop it anyway
                            try:
                                await self._websocket.close()
                            except Exception:
                                pass
                            self._websocket = None
                            self._connected = False
                            return  # Don't set connected=True
                    except (json.JSONDecodeError, TypeError):
                        pass  # Not JSON or not parseable — continue normally

                except asyncio.TimeoutError:
                    logger.warning("No initial response from Soniox (continuing anyway)")
                except Exception as e:
                    logger.warning(f"Error receiving initial response: {e}")

                self._connected = True
                self._rate_limited = False  # Successful connection clears rate-limit flag
                logger.info("Connected to Soniox")

                # Start receiving messages in background
                self._receive_task = asyncio.create_task(self._receive_messages())

            except Exception as e:
                logger.error(f"Failed to connect to Soniox: {e}")
                self._websocket = None
                self._connected = False
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
        """Continuously receive and process messages from Soniox.

        Auto-reconnects with exponential backoff (up to 3 retries) when the
        WebSocket connection drops **while a user is actively speaking**.

        If the connection drops while idle (no user speaking), we simply mark
        ourselves as disconnected and exit. The next UserStartedSpeakingFrame
        will trigger a fresh _connect(). This avoids the constant
        disconnect/reconnect churn that Soniox's idle timeout causes.

        IMPORTANT: This is the ONLY coroutine that should reconnect. Other
        callers (_process_audio, _keepalive_loop) must NOT attempt their own
        reconnect — they simply skip work when disconnected.
        """
        MAX_RECONNECT_RETRIES = 3
        reconnect_attempt = 0
        self._pipeline_stopped = False

        try:
            while True:
                if self._pipeline_stopped:
                    break

                # Grab a local reference to avoid races with other coroutines
                ws = self._websocket

                if not self._connected or ws is None:
                    # Connection lost — only reconnect if user is actively
                    # speaking or bot is speaking (keepalive needed). If idle,
                    # just exit and let the next speech event re-connect.
                    if not self._user_speaking and not self._bot_speaking:
                        logger.info(
                            "Soniox: connection dropped while idle — will reconnect on next speech"
                        )
                        break

                    if reconnect_attempt >= MAX_RECONNECT_RETRIES:
                        logger.error(
                            f"Soniox: exhausted {MAX_RECONNECT_RETRIES} reconnect attempts, giving up"
                        )
                        break

                    backoff = min(0.5 * (2 ** reconnect_attempt), 4.0)
                    reconnect_attempt += 1
                    logger.warning(
                        f"Soniox: auto-reconnect attempt {reconnect_attempt}/{MAX_RECONNECT_RETRIES} "
                        f"in {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)
                    try:
                        # Close old socket if it exists
                        old_ws = self._websocket
                        self._websocket = None
                        self._connected = False
                        if old_ws:
                            try:
                                await old_ws.close()
                            except Exception:
                                pass

                        # Open new connection directly (bypass _connect lock to
                        # avoid deadlock since we ARE the receive task)
                        new_ws = await websocket_connect(
                            SONIOX_WS_URL,
                            max_size=10 * 1024 * 1024,
                            ping_interval=20,
                            ping_timeout=20,
                            close_timeout=10,
                        )
                        config = {
                            "api_key": self._config.api_key,
                            "model": self._config.model,
                            "language_hints": self._config.language_hints,
                            "language_hints_strict": self._config.language_hints_strict,
                            "enable_language_identification": self._config.enable_language_identification,
                            "enable_endpoint_detection": self._config.enable_endpoint_detection,
                            "audio_format": "pcm_s16le",
                            "sample_rate": self._config.sample_rate,
                            "num_channels": self._config.num_channels,
                        }
                        await new_ws.send(json.dumps(config))
                        try:
                            init_resp = await asyncio.wait_for(new_ws.recv(), timeout=5.0)
                            # Check for 429 rate-limit
                            try:
                                init_data = json.loads(init_resp)
                                if init_data.get("error_code") == 429:
                                    self._rate_limited = True
                                    self._last_rate_limit_time = time.monotonic()
                                    logger.warning(
                                        f"[SONIOX] Auto-reconnect got 429: "
                                        f"{init_data.get('error_message', 'rate limited')}"
                                    )
                                    try:
                                        await new_ws.close()
                                    except Exception:
                                        pass
                                    continue
                            except (json.JSONDecodeError, TypeError):
                                pass
                        except (asyncio.TimeoutError, Exception):
                            pass
                        # Atomically publish the new connection
                        self._websocket = new_ws
                        self._connected = True
                        self._rate_limited = False
                        reconnect_attempt = 0
                        logger.info("Soniox: auto-reconnect succeeded")
                    except Exception as e:
                        logger.error(f"Soniox: auto-reconnect failed: {e}")
                        self._websocket = None
                        self._connected = False
                        continue

                # Re-grab after potential reconnect
                ws = self._websocket
                if ws is None:
                    continue

                try:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    await self._handle_message(data)
                    # Successful message resets the reconnect counter
                    reconnect_attempt = 0
                except websockets.ConnectionClosed as e:
                    logger.warning(f"Soniox WebSocket connection closed: {e}")
                    self._connected = False
                    self._websocket = None
                    # Loop will check user_speaking/bot_speaking on next iteration
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from Soniox: {e}")
                except Exception as e:
                    logger.error(f"Error receiving from Soniox: {e}")
                    self._connected = False
                    self._websocket = None
                    # Loop will check user_speaking/bot_speaking on next iteration
        except asyncio.CancelledError:
            pass
        finally:
            self._connected = False
            self._websocket = None
            logger.info("Soniox: _receive_messages loop exited")

    async def _handle_message(self, data: dict):
        """Handle a message from Soniox server."""
        # Check for error
        if "error" in data:
            error_msg = data.get("error", "Unknown error")
            logger.error(f"Soniox error: {error_msg}")
            await self.push_frame(ErrorFrame(error=f"Soniox error: {error_msg}"))
            return

        # Soniox response format:
        # {"tokens": [{"text": "hello", "start_ms": 0, "duration_ms": 500, "is_final": true}], ...}

        tokens = data.get("tokens", [])
        if not tokens:
            # Log raw response for debugging
            logger.debug(f"Soniox response (no tokens): {data}")
            return

        # Build text from tokens
        text_parts = []
        is_final = False

        for token in tokens:
            token_text = token.get("text", "")
            if token_text:
                text_parts.append(token_text)

            # Track if any token is final
            if token.get("is_final", False):
                is_final = True

            # Track detected language if language identification is enabled
            if "language" in token and self._config.enable_language_identification:
                self._detected_language = token["language"]
                if self._detected_language != self._last_logged_raw_language:
                    self._last_logged_raw_language = self._detected_language
                    logger.debug(f"Soniox detected language: {self._detected_language}")

        text = "".join(text_parts).strip()  # No space - tokens may include spaces
        # Remove Soniox end token
        text = text.replace("<end>", "").strip()
        if not text:
            return

        formatted_text = text
        if self._contains_disallowed_script(formatted_text):
            logger.warning(
                f"Dropping Soniox transcript with disallowed script "
                f"(hints={self._config.language_hints}): '{formatted_text[:80]}'"
            )
            return

        # Check fin_audio_proc (final audio processed) or is_final flag
        if data.get("fin_audio_proc", False) or is_final:
            if not self._muted:
                logger.info(f"Soniox final [{self._detected_language or '?'}]: {formatted_text}")
                self._interrupted = False
                self._current_text = ""

                # Resolve language per-utterance and clamp to configured hints.
                lang = self._resolved_language(formatted_text)
                logger.info(
                    f"Language resolved for final transcript: {lang} "
                    f"(raw_detected={self._detected_language}, hints={self._config.language_hints})"
                )
                lang_label = LANG_LABELS.get(lang, "English")
                tagged_text = f"[User is speaking {lang_label}] {formatted_text}"

                # ── Clarity scoring ──
                speech_duration_s = 0.0
                if self._speech_start_time > 0:
                    speech_duration_s = time.monotonic() - self._speech_start_time
                clarity_score = self._compute_clarity_score(
                    formatted_text, tokens, speech_duration_s
                )
                self._clarity_total_utterances += 1
                self._clarity_scores.append(clarity_score)
                if clarity_score < 0.50:
                    self._clarity_low_count += 1

                logger.info(
                    f"[STT_CLARITY] score={clarity_score:.2f} | "
                    f"words={len(formatted_text.split())} | "
                    f"speech_dur={speech_duration_s:.1f}s | "
                    f"text='{formatted_text[:60]}' | "
                    f"low_rate={self._clarity_low_count}/{self._clarity_total_utterances}"
                )

                frame = TranscriptionFrame(
                    text=tagged_text,
                    user_id="",
                    timestamp="",
                    language=lang,
                )
                # Attach clarity score as an extra attribute so downstream
                # processors can gate on it without modifying Pipecat internals.
                frame.clarity_score = clarity_score
                frame.raw_text = formatted_text  # untagged text for echo-back

                await self.push_frame(frame)
                # Prevent stale language from carrying into a later utterance.
                self._detected_language = None
                self._last_logged_raw_language = None
        elif not self._muted and self._config.include_nonfinal:
            # Interim result — log first interim for timing
            if not self._current_text and self._speech_start_time:
                since_speech = (time.monotonic() - self._speech_start_time) * 1000
                logger.info(
                    f"[SONIOX_TIMING] First interim transcript {since_speech:.0f}ms "
                    f"after speech start: '{formatted_text[:60]}'"
                )
            else:
                logger.debug(f"Soniox interim: {formatted_text[:50]}...")
            self._current_text = formatted_text

            await self.push_frame(
                InterimTranscriptionFrame(
                    text=formatted_text,
                    user_id="",
                    timestamp="",
                    language=self._resolved_language(formatted_text),
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
            # Track speech start for clarity scoring
            self._speech_start_time = time.monotonic()
            self._audio_bytes_sent = 0
            # Reset per-utterance language state to avoid stale carry-over.
            self._detected_language = None
            self._last_logged_raw_language = None

            idle_sec = time.monotonic() - self._pipeline_start_time
            is_first = not self._first_speech_received

            if is_first:
                self._first_speech_received = True
                logger.info(
                    f"[SONIOX_TIMING] First speech after {idle_sec:.1f}s idle | "
                    f"connected={self._connected}"
                )
            else:
                logger.info(
                    f"[SONIOX_TIMING] User speaking (turn) | "
                    f"connected={self._connected}"
                )
            # NOTE: Keepalive loop stays running — it auto-pauses when
            # user_speaking=True and resumes when both user+bot go silent.

            # Clear any stale reconnect buffer from a previous utterance
            self._reconnect_buffer.clear()

            # Reconnect if not connected. The _receive_messages loop may have
            # exited due to idle timeout, so check if the task is done too.
            # Always attempt reconnect for real speech, even if rate-limited —
            # the user is actually talking, so we must try.
            receive_task_dead = (
                self._receive_task is None or self._receive_task.done()
            )
            if not self._connected and receive_task_dead:
                logger.warning(
                    f"[SONIOX_TIMING] ⚠️  Soniox NOT connected on speech start "
                    f"(idle {idle_sec:.1f}s) — reconnecting..."
                )
                t0 = time.monotonic()
                await self._connect()
                reconnect_ms = (time.monotonic() - t0) * 1000
                logger.info(
                    f"[SONIOX_TIMING] Reconnect completed in {reconnect_ms:.0f}ms"
                )
                # Flush any audio that was buffered during reconnection
                if self._reconnect_buffer and self._connected and self._websocket:
                    buf_bytes = sum(len(b) for b in self._reconnect_buffer)
                    buf_ms = buf_bytes / (self._config.sample_rate * 2) * 1000
                    logger.info(
                        f"[SONIOX_TIMING] Flushing {len(self._reconnect_buffer)} buffered frames "
                        f"({buf_bytes} bytes ≈ {buf_ms:.0f}ms audio) to Soniox"
                    )
                    ws = self._websocket
                    for buffered_audio in self._reconnect_buffer:
                        try:
                            await ws.send(buffered_audio)
                        except Exception as e:
                            logger.warning(f"Failed to flush buffered audio: {e}")
                            break
                    self._reconnect_buffer.clear()
            elif not self._connected:
                logger.warning(
                    "[SONIOX_TIMING] ⚠️  Soniox reconnecting — audio will be buffered"
                )
            else:
                logger.info(
                    f"[SONIOX_TIMING] ✅ Soniox already connected — "
                    f"audio flows immediately (idle {idle_sec:.1f}s)"
                )
            await self.push_frame(frame, direction)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            logger.info("User stopped speaking (Soniox endpoint detection will finalize)")
            # NOTE: We intentionally do NOT send "" (empty string) here.
            # Sending "" tells Soniox "this session is done" and it closes the
            # connection shortly after. This breaks multi-turn conversations
            # because subsequent turns need to reconnect (~300ms delay).
            #
            # Instead, we rely on Soniox's enable_endpoint_detection (which is
            # enabled in our config) to detect utterance boundaries from the
            # silence that follows the user's speech. The VAD stop_secs (1.0s)
            # ensures enough silence for Soniox to finalize the utterance.
            #
            # The "" signal is only sent in stop() when the pipeline ends.

            await self.push_frame(frame, direction)

        # === Audio Frames ===
        elif isinstance(frame, InputAudioRawFrame):
            await self._process_audio(frame)
            # Don't forward audio frames downstream

        # === Other Frames ===
        else:
            await self.push_frame(frame, direction)

    async def _process_audio(self, frame: InputAudioRawFrame):
        """Process audio frame and send to Soniox.

        If the connection is dead and the user is speaking, buffer the audio
        so it can be flushed once Soniox reconnects (prevents initial word loss).
        If idle, silently drop the frame.
        """
        ws = self._websocket  # Local ref to avoid races
        if not self._connected or ws is None:
            # Buffer audio during reconnection so initial words aren't lost
            if self._user_speaking and len(self._reconnect_buffer) < self._MAX_RECONNECT_BUFFER:
                audio = frame.audio
                if isinstance(audio, np.ndarray):
                    if audio.dtype == np.float32 or audio.dtype == np.float64:
                        audio = (audio * 32767).astype(np.int16)
                    elif audio.dtype != np.int16:
                        audio = audio.astype(np.int16)
                    audio = audio.tobytes()
                if isinstance(audio, bytes):
                    self._reconnect_buffer.append(audio)
                    logger.debug(
                        f"Buffered audio frame during reconnect "
                        f"({len(self._reconnect_buffer)}/{self._MAX_RECONNECT_BUFFER})"
                    )
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
            await ws.send(audio_bytes)
            prev_sent = self._audio_bytes_sent
            self._audio_bytes_sent += len(audio_bytes)
            # Log first audio frame of each utterance for timing analysis
            if prev_sent == 0:
                since_speech = (time.monotonic() - self._speech_start_time) * 1000
                since_pipeline = (time.monotonic() - self._pipeline_start_time) * 1000 if self._pipeline_start_time else 0
                logger.info(
                    f"[SONIOX_TIMING] ✅ First audio frame sent to STT | "
                    f"{since_speech:.0f}ms after VAD trigger | "
                    f"{since_pipeline:.0f}ms after pipeline start | "
                    f"{len(audio_bytes)} bytes"
                )
            else:
                logger.debug(f"Sent {len(audio_bytes)} bytes of audio to Soniox")

        except websockets.ConnectionClosed:
            logger.warning("Soniox connection closed during audio send — _receive_messages will reconnect")
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
            while self._bot_speaking:
                try:
                    await asyncio.sleep(1.0)  # Send keepalive every second
                    ws = self._websocket  # Local ref to avoid races
                    if self._connected and ws is not None and self._bot_speaking:
                        await ws.send(silence_bytes)
                        logger.debug("Sent keepalive silence to Soniox")
                    elif not self._connected:
                        # Connection lost — skip keepalive; _receive_messages handles reconnect
                        logger.debug("Keepalive skipped — not connected, waiting for reconnect")
                except websockets.ConnectionClosed:
                    logger.warning("Connection closed during keepalive — _receive_messages will reconnect")
                    self._connected = False
                    self._websocket = None
                    # Don't break — stay in loop so keepalive resumes after reconnect
                except Exception as e:
                    logger.warning(f"Keepalive error: {e}")
                    # Don't break — transient errors shouldn't kill keepalive
        except asyncio.CancelledError:
            pass

    async def _idle_keepalive_loop(self):
        """Send periodic silence to keep Soniox alive during ALL idle gaps.

        Soniox's server drops idle connections after ~10-30s. This loop sends
        a tiny silence ping every 5s to keep the connection alive.

        It runs continuously from pipeline start until the pipeline stops.
        It pauses itself when the user or bot is actively speaking (real audio
        keeps the connection alive), and resumes when both go silent.

        RATE-LIMIT AWARENESS (429):
        When Soniox returns a 429 (max concurrent requests), we stop proactive
        reconnects for a backoff period. This prevents the tight reconnect loop
        that was burning through Soniox's concurrent connection quota on idle
        sessions. The connection will be re-established on-demand when the user
        actually starts speaking (UserStartedSpeakingFrame).
        """
        silence_samples = int(0.1 * self._config.sample_rate)  # 100ms of silence
        silence = np.zeros(silence_samples, dtype=np.int16)
        silence_bytes = silence.tobytes()

        ping_count = 0
        consecutive_reconnect_failures = 0
        MAX_CONSECUTIVE_FAILURES = 3  # Stop proactive reconnects after 3 failures
        try:
            while not self._pipeline_stopped:
                await asyncio.sleep(5.0)  # Ping every 5 seconds

                # Skip pinging when real audio is flowing — it's unnecessary
                # and would just add noise to the stream.
                if self._user_speaking or self._bot_speaking:
                    consecutive_reconnect_failures = 0  # Reset on real activity
                    continue

                ws = self._websocket
                if self._connected and ws is not None:
                    try:
                        await ws.send(silence_bytes)
                        ping_count += 1
                        consecutive_reconnect_failures = 0  # Reset on success
                        # Only log every 10th ping to reduce log spam
                        if ping_count % 10 == 1:
                            idle_sec = time.monotonic() - self._pipeline_start_time
                            logger.info(
                                f"[SONIOX_TIMING] Keepalive ping #{ping_count} "
                                f"({idle_sec:.1f}s since start) — Soniox alive ✅"
                            )
                    except websockets.ConnectionClosed:
                        logger.warning(
                            f"[SONIOX_TIMING] ⚠️  Keepalive: Soniox dropped connection "
                            f"after {ping_count} pings — will reconnect on next speech"
                        )
                        self._connected = False
                        self._websocket = None
                        # Don't break — stay in loop. If the connection is
                        # re-established (e.g. by UserStartedSpeakingFrame),
                        # keepalive should resume.
                    except Exception as e:
                        logger.warning(f"Soniox keepalive error: {e}")
                elif not self._connected:
                    # ── Rate-limit guard ──
                    # If we recently got a 429, don't try to reconnect proactively.
                    # Wait for the backoff period to expire, or for the user to
                    # actually start speaking (which triggers on-demand reconnect).
                    if self._rate_limited:
                        since_rate_limit = time.monotonic() - self._last_rate_limit_time
                        if since_rate_limit < self._rate_limit_backoff:
                            remaining = self._rate_limit_backoff - since_rate_limit
                            logger.debug(
                                f"[SONIOX] Skipping proactive reconnect — rate-limited, "
                                f"{remaining:.0f}s remaining in backoff"
                            )
                            continue
                        else:
                            logger.info(
                                f"[SONIOX] Rate-limit backoff expired ({since_rate_limit:.0f}s), "
                                f"allowing proactive reconnect"
                            )
                            self._rate_limited = False

                    # ── Consecutive failure guard ──
                    if consecutive_reconnect_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.info(
                            f"[SONIOX] Skipping proactive reconnect — "
                            f"{consecutive_reconnect_failures} consecutive failures, "
                            f"waiting for user speech to trigger on-demand reconnect"
                        )
                        continue

                    # Not connected and idle — try to reconnect proactively
                    # so the user doesn't pay the reconnect cost on next speech.
                    logger.info(
                        "[SONIOX_TIMING] Keepalive: not connected, attempting proactive reconnect..."
                    )
                    try:
                        await self._connect()
                        if self._connected:
                            logger.info("[SONIOX_TIMING] Keepalive: proactive reconnect succeeded ✅")
                            consecutive_reconnect_failures = 0
                        else:
                            # _connect returned without setting connected (e.g. 429)
                            consecutive_reconnect_failures += 1
                            logger.info(
                                f"[SONIOX] Proactive reconnect did not establish connection "
                                f"(failure {consecutive_reconnect_failures}/{MAX_CONSECUTIVE_FAILURES})"
                            )
                    except Exception as e:
                        consecutive_reconnect_failures += 1
                        logger.warning(
                            f"[SONIOX_TIMING] Keepalive: proactive reconnect failed "
                            f"({consecutive_reconnect_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}"
                        )
        except asyncio.CancelledError:
            idle_sec = time.monotonic() - self._pipeline_start_time
            logger.info(
                f"[SONIOX_TIMING] Keepalive loop ended after {ping_count} pings "
                f"({idle_sec:.1f}s) — pipeline stopping"
            )
