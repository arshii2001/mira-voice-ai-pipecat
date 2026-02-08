"""
Standalone TTS service for classroom audio.

Uses ElevenLabs REST API (not Pipecat's WebSocket-based service) so it works
outside a Pipecat pipeline. Returns raw PCM 16-bit LE audio chunks.
"""

import asyncio
import io
import logging
import os
import struct
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import aiohttp

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


class ClassroomTTS:
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
        logger.info(f"ClassroomTTS initialized: voice={self._voice_id}, model={model}, sr={sample_rate}")

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
            logger.info("ClassroomTTS cancelled")
            raise
        except Exception as e:
            logger.error(f"ClassroomTTS error: {e}")


def create_classroom_tts(sample_rate: int = 24000) -> Optional[ClassroomTTS]:
    """Factory function to create a ClassroomTTS instance from env vars."""
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key:
        logger.warning("ELEVENLABS_API_KEY not set — classroom audio disabled")
        return None

    voice_gender = os.getenv("TTS_VOICE_GENDER", "female")
    return ClassroomTTS(
        api_key=api_key,
        voice_gender=voice_gender,
        sample_rate=sample_rate,
    )
