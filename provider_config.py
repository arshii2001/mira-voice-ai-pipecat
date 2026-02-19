"""
Multi-provider configuration resolver.

All provider API keys and endpoints live in the environment.
A single selector env var (STT_PROVIDER, LLM_PROVIDER, TTS_PROVIDER)
picks which set of credentials to use.

To switch providers: change the selector, restart the pod.

Env var naming convention:
    LLM_<PROVIDER>_BASE_URL, LLM_<PROVIDER>_MODEL, LLM_<PROVIDER>_API_KEY
    TTS_<PROVIDER>_WS_URL,   TTS_<PROVIDER>_API_KEY, ...
    STT_<PROVIDER>_API_KEY,  STT_<PROVIDER>_LANGUAGE_HINTS, ...

Example:
    LLM_PROVIDER=groq          ← change this to switch
    LLM_OPENAI_BASE_URL=...    ← always present
    LLM_OPENAI_API_KEY=...
    LLM_GROQ_BASE_URL=...      ← always present
    LLM_GROQ_API_KEY=...
    LLM_VLLM_BASE_URL=...      ← always present
    LLM_VLLM_API_KEY=...
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Data classes for resolved config
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LLMConfig:
    provider: str       # "openai" | "groq" | "vllm"
    base_url: str
    model: str
    api_key: str
    is_vllm: bool       # True → send chat_template_kwargs

    def __repr__(self):
        return (
            f"LLMConfig(provider={self.provider!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, is_vllm={self.is_vllm})"
        )


@dataclass(frozen=True)
class TTSConfig:
    provider: str       # "elevenlabs" | "svara" | "openai"
    # ElevenLabs
    elevenlabs_api_key: str = ""
    elevenlabs_voice_gender: str = "female"
    # Svara
    svara_ws_url: str = ""
    svara_api_key: str = ""
    svara_voice: str = "en_female"
    svara_max_tokens: int = 4500
    # OpenAI TTS
    openai_tts_voice: str = "nova"
    # Common
    sample_rate: int = 24000


@dataclass(frozen=True)
class STTConfig:
    provider: str       # "soniox" | "deepgram" | "whisper"
    soniox_api_key: str = ""
    deepgram_api_key: str = ""
    language_hints: str = "en,hi,ta"
    sample_rate: int = 16000


# ─────────────────────────────────────────────────────────────────────
# LLM resolver
# ─────────────────────────────────────────────────────────────────────

# Provider presets: (env_prefix, default_base_url, default_model, is_vllm)
_LLM_PRESETS = {
    "openai": ("OPENAI", "https://api.openai.com/v1", "gpt-4o-mini", False),
    "groq":   ("GROQ",   "https://api.groq.com/openai/v1", "openai/gpt-oss-120b", False),
    "vllm":   ("VLLM",   "http://vllm-gpt-oss-120b/v1", "openai/gpt-oss-120b", True),
}


def resolve_llm_config() -> LLMConfig:
    """Resolve LLM configuration from environment variables.

    Reads LLM_PROVIDER, then looks up LLM_{PROVIDER}_BASE_URL, etc.
    Falls back to legacy LLM_BASE_URL / LLM_API_KEY for backward compat.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    preset = _LLM_PRESETS.get(provider)
    if preset:
        prefix, default_url, default_model, is_vllm = preset
        base_url = os.getenv(f"LLM_{prefix}_BASE_URL", default_url)
        model = os.getenv(f"LLM_{prefix}_MODEL", default_model)
        api_key = os.getenv(f"LLM_{prefix}_API_KEY", "")
    else:
        # Unknown provider — try generic env vars as fallback
        prefix = provider.upper()
        base_url = os.getenv(f"LLM_{prefix}_BASE_URL", "")
        model = os.getenv(f"LLM_{prefix}_MODEL", "")
        api_key = os.getenv(f"LLM_{prefix}_API_KEY", "")
        is_vllm = False

    # ── Backward compatibility ──
    # If provider-specific vars are empty, fall back to legacy LLM_BASE_URL etc.
    if not api_key:
        api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if not base_url:
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    if not model:
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    config = LLMConfig(
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        is_vllm=is_vllm,
    )
    logger.info(f"[CONFIG] LLM resolved: {config}")
    return config


# ─────────────────────────────────────────────────────────────────────
# TTS resolver
# ─────────────────────────────────────────────────────────────────────

def resolve_tts_config() -> TTSConfig:
    """Resolve TTS configuration from environment variables.

    Reads TTS_PROVIDER, then populates the relevant fields.
    All provider keys are always loaded so switching only requires
    changing TTS_PROVIDER.
    """
    provider = os.getenv("TTS_PROVIDER", "elevenlabs").strip().lower()
    sample_rate = int(os.getenv("TTS_SAMPLE_RATE", "24000"))

    config = TTSConfig(
        provider=provider,
        # ElevenLabs (always loaded)
        elevenlabs_api_key=os.getenv("TTS_ELEVENLABS_API_KEY",
                                     os.getenv("ELEVENLABS_API_KEY", "")),
        elevenlabs_voice_gender=os.getenv("TTS_VOICE_GENDER", "female"),
        # Svara (always loaded)
        svara_ws_url=os.getenv("TTS_SVARA_WS_URL",
                               os.getenv("TTS_WS_URL",
                                         "ws://svara-tts/v1/audio/text-to-speech/stream")),
        svara_api_key=os.getenv("TTS_SVARA_API_KEY",
                                os.getenv("TTS_WS_API_KEY", "")),
        svara_voice=os.getenv("TTS_SVARA_VOICE",
                              os.getenv("DEFAULT_VOICE", "en_female")),
        svara_max_tokens=int(os.getenv("TTS_MAX_TOKENS", "4500")),
        # OpenAI TTS
        openai_tts_voice=os.getenv("OPENAI_TTS_VOICE", "nova"),
        # Common
        sample_rate=sample_rate,
    )
    logger.info(
        f"[CONFIG] TTS resolved: provider={config.provider}, "
        f"sample_rate={config.sample_rate}"
    )
    return config


# ─────────────────────────────────────────────────────────────────────
# STT resolver
# ─────────────────────────────────────────────────────────────────────

def resolve_stt_config() -> STTConfig:
    """Resolve STT configuration from environment variables.

    Reads STT_PROVIDER, then populates the relevant fields.
    """
    provider = os.getenv("STT_PROVIDER", "soniox").strip().lower()

    config = STTConfig(
        provider=provider,
        soniox_api_key=os.getenv("STT_SONIOX_API_KEY",
                                 os.getenv("SONIOX_API_KEY", "")),
        deepgram_api_key=os.getenv("STT_DEEPGRAM_API_KEY",
                                   os.getenv("DEEPGRAM_API_KEY", "")),
        language_hints=os.getenv("STT_LANGUAGE_HINTS", "en,hi,ta"),
        sample_rate=int(os.getenv("STT_SAMPLE_RATE", "16000")),
    )
    logger.info(
        f"[CONFIG] STT resolved: provider={config.provider}, "
        f"language_hints={config.language_hints}"
    )
    return config


# ─────────────────────────────────────────────────────────────────────
# Module-level singletons — resolved once at import time
# ─────────────────────────────────────────────────────────────────────

LLM = resolve_llm_config()
TTS = resolve_tts_config()
STT = resolve_stt_config()
