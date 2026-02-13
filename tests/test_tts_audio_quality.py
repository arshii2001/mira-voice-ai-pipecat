#!/usr/bin/env python3
"""
Local test for TTS audio quality: fade edges, chunking, and crossfade.

Tests the audio processing logic in svara_tts.py WITHOUT needing a running
Svara server or any external services.  Generates synthetic PCM audio and
verifies that:
  1. _apply_fade_edges produces zero-amplitude start/end (no pops)
  2. _crossfade_pcm produces smooth transitions between chunks
  3. _chunk_text splits text correctly using Sarvam's algorithm
  4. Fade edges don't distort the middle of the audio

Usage:
    python tests/test_tts_audio_quality.py
    python tests/test_tts_audio_quality.py --save-wav   # Save WAVs for manual listening
"""

import argparse
import math
import os
import struct
import sys
import wave

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.svara_tts import SvaraTTSService, _CHUNK_TARGET, _CHUNK_MAX, _CHUNK_MIN, _CROSSFADE_SEC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_sine_pcm(freq_hz: float = 440.0, duration_sec: float = 1.0,
                      sample_rate: int = 24000, amplitude: int = 16000) -> bytes:
    """Generate a pure sine wave as PCM16 bytes."""
    n_samples = int(sample_rate * duration_sec)
    samples = np.array([
        amplitude * math.sin(2 * math.pi * freq_hz * i / sample_rate)
        for i in range(n_samples)
    ], dtype=np.int16)
    return samples.tobytes()


def pcm_to_array(pcm: bytes) -> np.ndarray:
    """Convert PCM16 bytes to float array."""
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32)


def save_wav(filename: str, pcm: bytes, sample_rate: int = 24000):
    """Save PCM16 bytes as a WAV file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    print(f"  Saved: {filename} ({len(pcm)//2} samples, {len(pcm)/2/sample_rate:.2f}s)")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_chunk_text():
    """Test that _chunk_text splits text correctly."""
    print("\n=== Test: _chunk_text (Sarvam algorithm) ===")

    # Short text — no splitting
    short = "Hello, how are you?"
    chunks = SvaraTTSService._chunk_text(short)
    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    assert chunks[0] == short
    print(f"  ✅ Short text ({len(short)} chars) → 1 chunk")

    # Medium text — should stay as one chunk if under target
    medium = "This is a medium length sentence that should fit in one chunk."
    chunks = SvaraTTSService._chunk_text(medium)
    assert len(chunks) == 1, f"Expected 1 chunk for {len(medium)} chars, got {len(chunks)}"
    print(f"  ✅ Medium text ({len(medium)} chars) → 1 chunk")

    # Long text — should be split
    long_text = (
        "Applications and devices equipped with AI can see and identify objects. "
        "They can understand and respond to human language. "
        "They can learn from new information and experience. "
        "They can make detailed recommendations to users and experts. "
        "They can act independently, replacing the need for human intelligence or intervention."
    )
    chunks = SvaraTTSService._chunk_text(long_text)
    assert len(chunks) > 1, f"Expected >1 chunks for {len(long_text)} chars, got {len(chunks)}"
    for i, c in enumerate(chunks):
        assert len(c) <= _CHUNK_MAX, f"Chunk {i} exceeds max: {len(c)} > {_CHUNK_MAX}"
        print(f"  Chunk {i+1}: {len(c)} chars — '{c[:60]}...'")
    print(f"  ✅ Long text ({len(long_text)} chars) → {len(chunks)} chunks (all ≤{_CHUNK_MAX})")

    # Tiny tail merge
    text_with_tiny_tail = "This is a decent length sentence that goes on for a while. X."
    chunks = SvaraTTSService._chunk_text(text_with_tiny_tail)
    for c in chunks:
        assert len(c) >= _CHUNK_MIN or len(chunks) == 1, \
            f"Chunk too small ({len(c)} < {_CHUNK_MIN}): '{c}'"
    print(f"  ✅ Tiny tail merge works (no chunk < {_CHUNK_MIN} chars)")

    # Hindi text
    hindi = (
        "एक समय की बात है, जब हिमालय की चोटियों के बीच एक छोटा सा गाँव बसा था। "
        "वहाँ के लोग कहते थे कि हवाओं में संगीत होता है। "
        "हमारा लक्ष्य है कि हर भारतीय अपनी भाषा में गर्व से बात कर सके। "
        "चलिए इस सफर में हमारे साथ जुड़िये।"
    )
    chunks = SvaraTTSService._chunk_text(hindi)
    for i, c in enumerate(chunks):
        assert len(c) <= _CHUNK_MAX, f"Hindi chunk {i} exceeds max: {len(c)}"
    print(f"  ✅ Hindi text ({len(hindi)} chars) → {len(chunks)} chunks")

    print("  ✅ All chunk_text tests passed!")


def test_fade_edges():
    """Test that _apply_fade_edges produces zero-start/end audio."""
    print("\n=== Test: _apply_fade_edges ===")

    sr = 24000
    pcm = generate_sine_pcm(freq_hz=440, duration_sec=0.5, sample_rate=sr)
    arr_before = pcm_to_array(pcm)

    faded = SvaraTTSService._apply_fade_edges(pcm, fade_sec=_CROSSFADE_SEC, sample_rate=sr)
    arr_after = pcm_to_array(faded)

    # Same length
    assert len(arr_before) == len(arr_after), \
        f"Length changed: {len(arr_before)} → {len(arr_after)}"
    print(f"  ✅ Length preserved: {len(arr_after)} samples")

    # First sample should be ~0 (faded in)
    assert abs(arr_after[0]) < 100, \
        f"First sample not near zero: {arr_after[0]}"
    print(f"  ✅ First sample near zero: {arr_after[0]:.0f}")

    # Last sample should be ~0 (faded out)
    assert abs(arr_after[-1]) < 100, \
        f"Last sample not near zero: {arr_after[-1]}"
    print(f"  ✅ Last sample near zero: {arr_after[-1]:.0f}")

    # Middle should be mostly unchanged
    mid = len(arr_after) // 2
    mid_diff = abs(arr_before[mid] - arr_after[mid])
    assert mid_diff < 10, f"Middle distorted: diff={mid_diff}"
    print(f"  ✅ Middle undistorted: diff={mid_diff:.0f}")

    # Fade-in region should be monotonically increasing in envelope
    fade_samples = int(_CROSSFADE_SEC * sr)
    fade_region = np.abs(arr_after[:fade_samples])
    # Check that later samples are generally larger than earlier ones
    first_quarter = np.mean(fade_region[:fade_samples//4])
    last_quarter = np.mean(fade_region[3*fade_samples//4:])
    assert last_quarter > first_quarter, \
        f"Fade-in not increasing: {first_quarter:.0f} → {last_quarter:.0f}"
    print(f"  ✅ Fade-in envelope increasing: {first_quarter:.0f} → {last_quarter:.0f}")

    print("  ✅ All fade_edges tests passed!")
    return pcm, faded


def test_crossfade():
    """Test that _crossfade_pcm produces smooth transitions."""
    print("\n=== Test: _crossfade_pcm ===")

    sr = 24000
    # Two sine waves at different frequencies
    pcm_a = generate_sine_pcm(freq_hz=440, duration_sec=0.5, sample_rate=sr)
    pcm_b = generate_sine_pcm(freq_hz=880, duration_sec=0.5, sample_rate=sr)

    merged = SvaraTTSService._crossfade_pcm(pcm_a, pcm_b, fade_sec=_CROSSFADE_SEC, sample_rate=sr)
    arr_merged = pcm_to_array(merged)

    # Length should be less than sum (overlapping region)
    expected_len = len(pcm_a)//2 + len(pcm_b)//2 - int(_CROSSFADE_SEC * sr)
    assert abs(len(arr_merged) - expected_len) < 2, \
        f"Unexpected length: {len(arr_merged)} vs expected ~{expected_len}"
    print(f"  ✅ Merged length correct: {len(arr_merged)} samples (expected ~{expected_len})")

    # The crossfade region should have intermediate amplitude
    fade_samples = int(_CROSSFADE_SEC * sr)
    join_point = len(pcm_a)//2 - fade_samples
    crossfade_region = arr_merged[join_point:join_point + fade_samples]
    max_amplitude = np.max(np.abs(crossfade_region))
    assert max_amplitude < 25000, \
        f"Crossfade region too loud: {max_amplitude}"
    print(f"  ✅ Crossfade region amplitude reasonable: {max_amplitude:.0f}")

    print("  ✅ All crossfade tests passed!")
    return pcm_a, pcm_b, merged


def test_pop_elimination():
    """Simulate what happens when Pipecat concatenates two sentences.

    Without fade edges: direct concatenation produces a discontinuity (pop).
    With fade edges: the join is smooth because both ends are at zero.
    """
    print("\n=== Test: Pop elimination (sentence boundary simulation) ===")

    sr = 24000
    # Simulate two sentences from Svara
    sentence_a = generate_sine_pcm(freq_hz=440, duration_sec=0.8, sample_rate=sr, amplitude=12000)
    sentence_b = generate_sine_pcm(freq_hz=660, duration_sec=0.6, sample_rate=sr, amplitude=14000)

    # Without fade: direct concatenation
    raw_concat = sentence_a + sentence_b
    arr_raw = pcm_to_array(raw_concat)

    # Measure discontinuity at join point
    join = len(sentence_a) // 2
    raw_jump = abs(float(arr_raw[join]) - float(arr_raw[join - 1]))

    # With fade edges
    faded_a = SvaraTTSService._apply_fade_edges(sentence_a, fade_sec=_CROSSFADE_SEC, sample_rate=sr)
    faded_b = SvaraTTSService._apply_fade_edges(sentence_b, fade_sec=_CROSSFADE_SEC, sample_rate=sr)
    faded_concat = faded_a + faded_b
    arr_faded = pcm_to_array(faded_concat)
    faded_jump = abs(float(arr_faded[join]) - float(arr_faded[join - 1]))

    print(f"  Raw concatenation jump at boundary:   {raw_jump:.0f}")
    print(f"  Faded concatenation jump at boundary: {faded_jump:.0f}")

    # The faded version should have a much smaller jump
    if faded_jump < raw_jump:
        improvement = (1 - faded_jump / max(raw_jump, 1)) * 100
        print(f"  ✅ Pop reduced by {improvement:.0f}% ({raw_jump:.0f} → {faded_jump:.0f})")
    else:
        print(f"  ⚠️  Faded jump not smaller (may be okay for specific phase alignment)")

    # The key check: both ends of faded audio should be near zero
    arr_a = pcm_to_array(faded_a)
    arr_b = pcm_to_array(faded_b)
    assert abs(arr_a[-1]) < 200, f"End of sentence A not near zero: {arr_a[-1]}"
    assert abs(arr_b[0]) < 200, f"Start of sentence B not near zero: {arr_b[0]}"
    print(f"  ✅ End of sentence A: {arr_a[-1]:.0f} (near zero)")
    print(f"  ✅ Start of sentence B: {arr_b[0]:.0f} (near zero)")

    print("  ✅ Pop elimination verified!")
    return raw_concat, faded_concat


def test_edge_cases():
    """Test edge cases: empty audio, very short audio, etc."""
    print("\n=== Test: Edge cases ===")

    # Empty bytes
    result = SvaraTTSService._apply_fade_edges(b"", fade_sec=0.04, sample_rate=24000)
    assert result == b""
    print("  ✅ Empty bytes → empty bytes")

    # Very short audio (< 4 bytes)
    result = SvaraTTSService._apply_fade_edges(b"\x00\x01", fade_sec=0.04, sample_rate=24000)
    assert result == b"\x00\x01"
    print("  ✅ Very short audio → passed through unchanged")

    # Audio shorter than fade duration
    short_pcm = generate_sine_pcm(freq_hz=440, duration_sec=0.01, sample_rate=24000)
    result = SvaraTTSService._apply_fade_edges(short_pcm, fade_sec=0.04, sample_rate=24000)
    assert len(result) == len(short_pcm)
    print(f"  ✅ Short audio ({len(short_pcm)} bytes) → same length, no crash")

    # Crossfade with empty
    pcm = generate_sine_pcm(freq_hz=440, duration_sec=0.5, sample_rate=24000)
    result = SvaraTTSService._crossfade_pcm(b"", pcm, fade_sec=0.04, sample_rate=24000)
    assert result == pcm
    print("  ✅ Crossfade with empty first → returns second")

    result = SvaraTTSService._crossfade_pcm(pcm, b"", fade_sec=0.04, sample_rate=24000)
    assert result == pcm
    print("  ✅ Crossfade with empty second → returns first")

    # Chunk text edge cases
    assert SvaraTTSService._chunk_text("") == [""]  or SvaraTTSService._chunk_text("") == []
    print("  ✅ Chunk empty text → no crash")

    assert SvaraTTSService._chunk_text("Hi") == ["Hi"]
    print("  ✅ Chunk tiny text → single chunk")

    print("  ✅ All edge case tests passed!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test TTS audio quality locally")
    parser.add_argument("--save-wav", action="store_true",
                        help="Save WAV files for manual listening")
    args = parser.parse_args()

    output_dir = "/tmp/mira-tts-test"
    if args.save_wav:
        os.makedirs(output_dir, exist_ok=True)
        print(f"WAV files will be saved to: {output_dir}/")

    print("=" * 60)
    print("MIRA TTS Audio Quality Tests (Local)")
    print(f"Settings: CHUNK_TARGET={_CHUNK_TARGET}, CHUNK_MAX={_CHUNK_MAX}, "
          f"CHUNK_MIN={_CHUNK_MIN}, CROSSFADE={_CROSSFADE_SEC}s")
    print("=" * 60)

    # Run tests
    test_chunk_text()

    pcm_orig, pcm_faded = test_fade_edges()
    if args.save_wav:
        save_wav(f"{output_dir}/01_original.wav", pcm_orig)
        save_wav(f"{output_dir}/02_faded.wav", pcm_faded)

    pcm_a, pcm_b, pcm_merged = test_crossfade()
    if args.save_wav:
        save_wav(f"{output_dir}/03_chunk_a.wav", pcm_a)
        save_wav(f"{output_dir}/04_chunk_b.wav", pcm_b)
        save_wav(f"{output_dir}/05_crossfaded.wav", pcm_merged)

    raw_concat, faded_concat = test_pop_elimination()
    if args.save_wav:
        save_wav(f"{output_dir}/06_raw_concat_POPS.wav", raw_concat)
        save_wav(f"{output_dir}/07_faded_concat_SMOOTH.wav", faded_concat)

    test_edge_cases()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    if args.save_wav:
        print(f"\n📁 WAV files saved to {output_dir}/")
        print("   Listen to 06_raw_concat_POPS.wav vs 07_faded_concat_SMOOTH.wav")
        print("   to hear the difference!")
    print("=" * 60)


if __name__ == "__main__":
    main()
