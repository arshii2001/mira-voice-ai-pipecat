"""
Standalone TTS service for classroom audio.

This module keeps classroom listener audio provider-agnostic so the same
`TTS_PROVIDER` env var can drive both tutor voice and classroom fanout.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional

import aiohttp

from pipecat.frames.frames import TTSAudioRawFrame
from services.audio_utils import (
    apply_fade_edges,
    chunk_text,
    crossfade_pcm,
)
from services.svara_tts import SvaraTTSService

logger = logging.getLogger(__name__)

# Voice presets — same as services/elevenlabs_tts.py
VOICE_PRESETS = {
    "female": "2zRM7PkgwBPiau2jvVXc",  # Monika Sogam
    "male": "siw1N9V8LmYeEWKyWBxv",    # Ruhaan
}

ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1"
DEFAULT_MODEL = "eleven_multilingual_v2"


@dataclass
class AudioChunk:
    """Raw PCM audio chunk."""
    audio: bytes
    sample_rate: int = 24000


class ElevenLabsClassroomTTS:
    """
    Standalone ElevenLabs TTS for classroom listeners.

    Uses the REST streaming API to synthesize speech and yields raw PCM chunks.
    No Pipecat pipeline or StartFrame required.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: Optional[str] = None,
        voice_gender: str = "female",
        model: str = DEFAULT_MODEL,
        sample_rate: int = 24000,
    ):
        self._api_key = api_key
        self._voice_id = voice_id or VOICE_PRESETS.get(voice_gender, VOICE_PRESETS["female"])
        self._model = model
        self._sample_rate = sample_rate
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info(
            f"ClassroomTTS initialized: provider=elevenlabs, voice={self._voice_id}, "
            f"model={model}, sr={sample_rate}"
        )

    async def _ensure_session(self):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def run_tts(self, text: str) -> AsyncGenerator[AudioChunk, None]:
        """
        Synthesize text to speech using ElevenLabs REST streaming API.

        Yields AudioChunk objects with raw PCM 16-bit LE audio data.
        """
        if not text.strip():
            return

        await self._ensure_session()

        url = f"{ELEVENLABS_API_URL}/text-to-speech/{self._voice_id}/stream"

        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

        payload = {
            "text": text,
            "model_id": self._model,
            "voice_settings": {
                "stability": 0.75,
                "similarity_boost": 0.85,
                "style": 0.0,
                "use_speaker_boost": True,
            },
        }

        # Add output format for raw PCM
        params = {"output_format": f"pcm_{self._sample_rate}"}

        try:
            async with self._session.post(
                url, json=payload, headers=headers, params=params
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        f"ElevenLabs REST TTS failed: HTTP {resp.status} - {error_text[:200]}"
                    )
                    return

                # Stream PCM chunks
                chunk_size = 4800  # 100ms of audio at 24kHz 16-bit mono
                buffer = b""
                async for data in resp.content.iter_any():
                    buffer += data
                    while len(buffer) >= chunk_size:
                        chunk = buffer[:chunk_size]
                        buffer = buffer[chunk_size:]
                        yield AudioChunk(audio=chunk, sample_rate=self._sample_rate)

                # Yield remaining buffer
                if buffer:
                    yield AudioChunk(audio=buffer, sample_rate=self._sample_rate)

        except asyncio.CancelledError:
            logger.info("ClassroomTTS (ElevenLabs) cancelled")
            raise
        except Exception as e:
            logger.error(f"ClassroomTTS (ElevenLabs) error: {e}")


class SvaraClassroomTTS:
    """Classroom TTS adapter that calls Svara WS directly (blocking).

    Unlike the pipeline SvaraTTSService (which is async/non-blocking),
    this class is used outside the Pipecat pipeline for classroom listeners
    and needs to yield audio frames directly from the generator.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        voice: str = "en_female",
        sample_rate: int = 24000,
    ):
        self._sample_rate = sample_rate
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._ws_url = self._base_url.replace("http://", "ws://").replace("https://", "wss://")
        self._voice = voice
        # Create a lightweight SvaraTTSService just for its helper methods
        self._service = SvaraTTSService(
            base_url=base_url,
            api_key=api_key or None,
            voice=voice,
            streaming=True,
            sample_rate=sample_rate,
        )
        logger.info(
            f"ClassroomTTS initialized: provider=svara, base_url={base_url}, "
            f"voice={voice}, sr={sample_rate}"
        )

    async def run_tts(self, text: str) -> AsyncGenerator[AudioChunk, None]:
        """Synthesize text via Svara WS and yield AudioChunk objects.

        Handles:
        - Text chunking for long text (prevents max_tokens truncation)
        - Crossfade between chunks (eliminates pops at chunk boundaries)
        - Fade-in / fade-out on edges (eliminates start/end pops)
        """
        if not text.strip():
            return

        import time as _time
        t0 = _time.monotonic()
        sr = self._sample_rate
        chunk_size = 4096  # PCM bytes per yielded AudioChunk

        try:
            chunks = chunk_text(text)
            logger.info(
                f"[ClassroomTTS] Svara: {len(text)} chars → {len(chunks)} chunk(s)"
            )

            if len(chunks) == 1:
                # ── Single chunk: collect under semaphore, then fade + yield ──
                raw_pcm = b""
                async with SvaraTTSService._svara_semaphore:
                    async for frame in self._service._svara_ws_stream(text, t0):
                        if not isinstance(frame, TTSAudioRawFrame) or not frame.audio:
                            continue
                        raw_pcm += frame.audio

                # Apply fades and yield (outside semaphore)
                if raw_pcm:
                    raw_pcm = apply_fade_edges(raw_pcm, sample_rate=sr)
                    for i in range(0, len(raw_pcm), chunk_size):
                        yield AudioChunk(
                            audio=raw_pcm[i:i + chunk_size],
                            sample_rate=sr,
                        )
            else:
                # ── Multi-chunk: collect per chunk under semaphore, crossfade, yield ──
                combined_pcm = b""

                for ci, ct in enumerate(chunks):
                    logger.info(
                        f"[ClassroomTTS] chunk {ci+1}/{len(chunks)} "
                        f"({len(ct)} chars): '{ct[:60]}...'"
                    )
                    chunk_pcm = b""
                    ct0 = _time.monotonic()
                    # Semaphore per-chunk: released between chunks so voice
                    # pipeline can interleave if needed.
                    async with SvaraTTSService._svara_semaphore:
                        async for frame in self._service._svara_ws_stream(ct, ct0):
                            if isinstance(frame, TTSAudioRawFrame) and frame.audio:
                                chunk_pcm += frame.audio

                    if not chunk_pcm:
                        continue

                    if combined_pcm:
                        combined_pcm = crossfade_pcm(
                            combined_pcm, chunk_pcm, sample_rate=sr
                        )
                    else:
                        combined_pcm = chunk_pcm

                # Apply fade edges and yield (outside semaphore)
                if combined_pcm:
                    combined_pcm = apply_fade_edges(
                        combined_pcm, sample_rate=sr
                    )
                    for i in range(0, len(combined_pcm), chunk_size):
                        yield AudioChunk(
                            audio=combined_pcm[i:i + chunk_size],
                            sample_rate=sr,
                        )

            total_ms = (_time.monotonic() - t0) * 1000
            logger.info(f"[ClassroomTTS] Svara done in {total_ms:.0f}ms")

        except asyncio.CancelledError:
            logger.info("ClassroomTTS (Svara) cancelled")
            raise
        except Exception as e:
            logger.error(f"ClassroomTTS (Svara) error: {e}")


def create_classroom_tts(sample_rate: int = 24000):
    """Factory function to create classroom TTS using TTS_PROVIDER env vars."""
    provider = os.getenv("TTS_PROVIDER", "elevenlabs").strip().lower()

    if provider == "elevenlabs":
        api_key = os.getenv("ELEVENLABS_API_KEY", "")
        if not api_key:
            logger.warning("ELEVENLABS_API_KEY not set — classroom audio disabled")
            return None

        voice_gender = os.getenv("TTS_VOICE_GENDER", "female")
        return ElevenLabsClassroomTTS(
            api_key=api_key,
            voice_gender=voice_gender,
            sample_rate=sample_rate,
        )

    if provider == "svara":
        tts_ws_url = os.getenv("TTS_WS_URL", "ws://svara-tts/v1/audio/text-to-speech/stream")
        tts_base_url = (
            tts_ws_url
            .replace("ws://", "http://")
            .replace("wss://", "https://")
            .rsplit("/v1/", 1)[0]
        )
        tts_api_key = os.getenv("TTS_WS_API_KEY", "")
        default_voice = os.getenv("DEFAULT_VOICE", "en_female")
        return SvaraClassroomTTS(
            base_url=tts_base_url,
            api_key=tts_api_key,
            voice=default_voice,
            sample_rate=sample_rate,
        )

    logger.warning(
        f"Unsupported TTS_PROVIDER='{provider}' for classroom audio "
        f"(supported: elevenlabs, svara) — classroom audio disabled"
    )
    return None
