"""
ElevenLabs TTS Service Wrapper for Pipecat.

This module provides a factory function to create an ElevenLabs TTS service
with predefined voice presets for the MiraVoiceAI project.

Features:
- Wraps pipecat's built-in ElevenLabsTTSService
- Provides voice presets for male/female voices
- Uses eleven_multilingual_v2 model for multilingual support
- WebSocket streaming for low latency
"""

import logging
from typing import Optional

from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

logger = logging.getLogger(__name__)

# Voice presets for ElevenLabs
# These are high-quality multilingual voices suitable for Indian language support
VOICE_PRESETS = {
    "female": {
        "id": "2zRM7PkgwBPiau2jvVXc",
        "name": "Monika Sogam - Deep & Natural",
    },
    "male": {
        "id": "siw1N9V8LmYeEWKyWBxv",
        "name": "Ruhaan - Clear, Loud, & Cheerful",
    },
}

# Default model for multilingual support
DEFAULT_MODEL = "eleven_multilingual_v2"


def create_elevenlabs_tts(
    api_key: str,
    voice_gender: str = "female",
    voice_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    sample_rate: int = 24000,
    **kwargs
) -> ElevenLabsTTSService:
    """
    Create an ElevenLabs TTS service with preset configuration.

    Args:
        api_key: ElevenLabs API key
        voice_gender: Voice gender preset ("female" or "male")
        voice_id: Optional custom voice ID (overrides voice_gender)
        model: ElevenLabs model (default: eleven_multilingual_v2)
        sample_rate: Output audio sample rate (default: 24000)
        **kwargs: Additional arguments passed to ElevenLabsTTSService

    Returns:
        Configured ElevenLabsTTSService instance

    Example:
        tts = create_elevenlabs_tts(
            api_key="your_api_key",
            voice_gender="female",
        )
    """
    # Determine voice ID
    if voice_id:
        selected_voice_id = voice_id
        voice_name = "Custom Voice"
    else:
        voice_preset = VOICE_PRESETS.get(voice_gender, VOICE_PRESETS["female"])
        selected_voice_id = voice_preset["id"]
        voice_name = voice_preset["name"]

    logger.info(f"Creating ElevenLabs TTS: voice={voice_name} ({selected_voice_id}), model={model}")

    # Voice settings to prevent high-pitch first syllable and ensure consistent prosody.
    # - stability=0.6: Higher stability = more consistent pitch across chunks (reduces first-syllable spike)
    # - similarity_boost=0.8: Keep voice close to the original voice profile
    # - style=0.0: Disable style exaggeration which can cause pitch variation
    # - use_speaker_boost=True: Clearer voice output
    params = ElevenLabsTTSService.InputParams(
        stability=0.75,
        similarity_boost=0.85,
        style=0.0,
        use_speaker_boost=True,
    )

    logger.info(f"ElevenLabs voice params: stability={params.stability}, similarity={params.similarity_boost}, style={params.style}")

    # Create and return the service
    return ElevenLabsTTSService(
        api_key=api_key,
        voice_id=selected_voice_id,
        model=model,
        sample_rate=sample_rate,
        params=params,
        **kwargs
    )


def get_available_voices() -> dict:
    """
    Get available voice presets.

    Returns:
        Dictionary of voice presets with id and name
    """
    return VOICE_PRESETS.copy()
