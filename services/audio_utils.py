"""
Audio processing utilities for TTS services.

Contains PCM audio helpers (crossfade, fade-in/fade-out, WAV header
stripping) and text chunking logic shared by SvaraTTSService and
SvaraClassroomTTS.
"""

import logging
import re
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WAV header constants
# ---------------------------------------------------------------------------
WAV_HEADER_SIZE = 44
DATA_MARKER = b"data"

# ---------------------------------------------------------------------------
# Text chunking for Svara long-form generation
# ---------------------------------------------------------------------------
_CLAUSE_SPLIT_RE = re.compile(r'(?<=[,.!?;:।؟\n])\s+')
CHUNK_TARGET = 150
CHUNK_MAX = 250
CHUNK_MIN = 50

# ---------------------------------------------------------------------------
# Crossfade / fade duration in seconds.
# 60ms smooths out Svara's initial audio transients (first ~100ms can have
# abrupt waveform starts) while remaining imperceptible to the listener.
# ---------------------------------------------------------------------------
CROSSFADE_SEC = 0.06


# ===================================================================
# Text chunking
# ===================================================================

def chunk_text(
    text: str,
    target_size: int = CHUNK_TARGET,
    max_size: int = CHUNK_MAX,
    min_size: int = CHUNK_MIN,
) -> List[str]:
    """Split long text into chunks at clause boundaries for TTS."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= target_size:
        return [text]

    parts = _CLAUSE_SPLIT_RE.split(text)
    chunks: List[str] = []
    buffer = ""

    for part in parts:
        if not part:
            continue
        if not buffer:
            buffer = part
            continue
        projected_len = len(buffer) + 1 + len(part)
        if projected_len > max_size:
            chunks.append(buffer)
            buffer = part
            continue
        if len(buffer) >= target_size:
            chunks.append(buffer)
            buffer = part
            continue
        buffer += " " + part

    if buffer:
        chunks.append(buffer)

    final: List[str] = []
    for c in chunks:
        if final and len(c) < min_size and len(final[-1]) + 1 + len(c) <= max_size:
            final[-1] += " " + c
        else:
            final.append(c)

    if len(final) > 1:
        logger.info(
            f"TTS text chunked: {len(text)} chars → {len(final)} chunks "
            f"({[len(c) for c in final]})"
        )
    return final


# ===================================================================
# PCM audio processing
# ===================================================================

def crossfade_pcm(
    a: bytes, b: bytes, fade_sec: float = CROSSFADE_SEC, sample_rate: int = 24000
) -> bytes:
    """Crossfade the tail of PCM buffer `a` into the head of `b`."""
    if not a or not b:
        return a + b
    fade_samples = int(fade_sec * sample_rate)
    a_arr = np.frombuffer(a, dtype=np.int16).astype(np.float32)
    b_arr = np.frombuffer(b, dtype=np.int16).astype(np.float32)
    fade_samples = min(fade_samples, len(a_arr), len(b_arr))
    if fade_samples < 2:
        return a + b
    t = np.linspace(0, 1, fade_samples, endpoint=False)
    fade_out = np.cos(t * np.pi / 2)
    fade_in = np.sin(t * np.pi / 2)
    cross = a_arr[-fade_samples:] * fade_out + b_arr[:fade_samples] * fade_in
    result = np.concatenate([a_arr[:-fade_samples], cross, b_arr[fade_samples:]])
    return np.clip(result, -32768, 32767).astype(np.int16).tobytes()


def apply_fade_edges(
    pcm: bytes, fade_sec: float = CROSSFADE_SEC, sample_rate: int = 24000
) -> bytes:
    """Apply both fade-in and fade-out to a PCM buffer."""
    if not pcm or len(pcm) < 4:
        return pcm
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    fade_samples = int(fade_sec * sample_rate)
    fade_samples = min(fade_samples, len(arr) // 4)
    if fade_samples < 2:
        return pcm
    t_in = np.linspace(0, 1, fade_samples, endpoint=False)
    arr[:fade_samples] *= np.sin(t_in * np.pi / 2)
    t_out = np.linspace(0, 1, fade_samples, endpoint=False)
    arr[-fade_samples:] *= np.cos(t_out * np.pi / 2)
    return np.clip(arr, -32768, 32767).astype(np.int16).tobytes()


def apply_fade_in(
    pcm: bytes, fade_sec: float = CROSSFADE_SEC, sample_rate: int = 24000
) -> bytes:
    """Apply a fade-in to the start of a PCM buffer."""
    if not pcm or len(pcm) < 4:
        return pcm
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    fade_samples = int(fade_sec * sample_rate)
    fade_samples = min(fade_samples, len(arr) // 2)
    if fade_samples < 2:
        return pcm
    t = np.linspace(0, 1, fade_samples, endpoint=False)
    arr[:fade_samples] *= np.sin(t * np.pi / 2)
    return np.clip(arr, -32768, 32767).astype(np.int16).tobytes()


def apply_fade_out(
    pcm: bytes, fade_sec: float = CROSSFADE_SEC, sample_rate: int = 24000
) -> bytes:
    """Apply a fade-out to the end of a PCM buffer."""
    if not pcm or len(pcm) < 4:
        return pcm
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    fade_samples = int(fade_sec * sample_rate)
    fade_samples = min(fade_samples, len(arr) // 2)
    if fade_samples < 2:
        return pcm
    t = np.linspace(0, 1, fade_samples, endpoint=False)
    arr[-fade_samples:] *= np.cos(t * np.pi / 2)
    return np.clip(arr, -32768, 32767).astype(np.int16).tobytes()
