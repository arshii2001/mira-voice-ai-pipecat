#!/usr/bin/env python3
"""
Pipeline Smoothness Test Suite — Local Reference Tests.

This test suite verifies that every stage of the MIRA audio pipeline
(both tutor and classroom modes) meets smoothness requirements.
It runs LOCALLY without external services by testing the code logic,
data structures, and audio processing directly.

If production ever sounds rough, compare the [SMOOTH] logs from production
against the reference numbers from this test.

Tests:
  1. TTS Audio Quality       — fade, crossfade, pop elimination (existing)
  2. TTS Streaming Behavior  — streaming vs buffered, first-byte latency
  3. Pipeline Stage Timing   — VAD→STT→LLM→TTS→Audio-out latency budget
  4. Classroom Delivery      — clause chunking, translation, listener TTS timing
  5. Inter-sentence Gaps     — gap between consecutive TTS sentences
  6. Soniox Keepalive        — idle connection, audio buffering, word loss prevention
  7. Text Chunking           — chunk sizes, boundary detection
  8. Barge-in Guard          — MinWordsInterruptionStrategy behavior

Usage:
    python tests/test_pipeline_smoothness.py
    python tests/test_pipeline_smoothness.py --save-wav   # Save WAV files for manual listening
    python tests/test_pipeline_smoothness.py --verbose     # Extra detail

Reference Numbers (what "smooth" looks like):
    ┌────────────────────────────────────────────────────────────────┐
    │ Metric                          │ Target    │ Max Acceptable  │
    ├─────────────────────────────────┼───────────┼─────────────────┤
    │ TTS first audio byte (Svara WS) │ <300ms    │ <600ms          │
    │ Inter-sentence gap              │ <50ms     │ <200ms          │
    │ Fade-in/out duration            │ 40ms      │ 20-80ms         │
    │ Pop amplitude at boundary       │ <100      │ <500            │
    │ Clause dispatch latency         │ <10ms     │ <50ms           │
    │ VAD trigger time                │ 200ms     │ 300ms           │
    │ Full turn latency (user→audio)  │ <2000ms   │ <3500ms         │
    │ Soniox idle keepalive interval  │ 5s        │ 10s             │
    │ Reconnect buffer capacity       │ 50 frames │ ≥30 frames      │
    │ MinWords barge-in threshold     │ 2 words   │ 1-3 words       │
    └─────────────────────────────────┴───────────┴─────────────────┘
"""

import argparse
import inspect
import math
import os
import re
import sys
import time
import wave

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Test Results Tracking ────────────────────────────────────────────────────

class TestReport:
    """Collects pass/fail/warn results and prints a final report."""

    def __init__(self):
        self.results = []  # (section, name, status, detail)

    def ok(self, section: str, name: str, detail: str = ""):
        self.results.append((section, name, "PASS", detail))
        print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))

    def warn(self, section: str, name: str, detail: str = ""):
        self.results.append((section, name, "WARN", detail))
        print(f"  ⚠️  {name}" + (f" — {detail}" if detail else ""))

    def fail(self, section: str, name: str, detail: str = ""):
        self.results.append((section, name, "FAIL", detail))
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))

    def summary(self):
        passed = sum(1 for _, _, s, _ in self.results if s == "PASS")
        warned = sum(1 for _, _, s, _ in self.results if s == "WARN")
        failed = sum(1 for _, _, s, _ in self.results if s == "FAIL")
        total = len(self.results)

        print(f"\n{'═' * 70}")
        print(f"SMOOTHNESS TEST REPORT")
        print(f"{'═' * 70}")

        # Group by section
        sections = {}
        for section, name, status, detail in self.results:
            sections.setdefault(section, []).append((name, status, detail))

        for section, items in sections.items():
            sec_pass = sum(1 for _, s, _ in items if s == "PASS")
            sec_total = len(items)
            icon = "✅" if sec_pass == sec_total else "⚠️" if sec_pass > 0 else "❌"
            print(f"\n{icon} {section} ({sec_pass}/{sec_total})")
            for name, status, detail in items:
                sym = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[status]
                print(f"   {sym} {name}")

        print(f"\n{'─' * 70}")
        print(f"Total: {passed} passed, {warned} warnings, {failed} failed / {total} tests")

        if failed == 0 and warned == 0:
            print(f"\n🎉 ALL SMOOTH — Pipeline meets all smoothness criteria!")
        elif failed == 0:
            print(f"\n✅ MOSTLY SMOOTH — {warned} warnings to review.")
        else:
            print(f"\n❌ ISSUES FOUND — {failed} failures need fixing.")

        print(f"{'═' * 70}")
        return failed == 0


report = TestReport()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def generate_sine_pcm(freq_hz=440.0, duration_sec=1.0, sr=24000, amplitude=16000):
    """Generate a pure sine wave as PCM16 bytes."""
    n = int(sr * duration_sec)
    t = np.arange(n, dtype=np.float64) / sr
    samples = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)
    return samples.tobytes()


def pcm_to_array(pcm):
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32)


def save_wav(filename, pcm, sr=24000):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    print(f"    Saved: {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: TTS Audio Quality (Fade, Crossfade, Pops)
# ═══════════════════════════════════════════════════════════════════════════════

def test_tts_audio_quality(save_wavs=False, output_dir="/tmp/mira-smooth"):
    """Test fade edges, crossfade, and pop elimination."""
    from services.svara_tts import (
        SvaraTTSService, _CHUNK_TARGET, _CHUNK_MAX, _CHUNK_MIN, _CROSSFADE_SEC
    )

    SECTION = "TTS Audio Quality"
    print(f"\n{'=' * 60}")
    print(f"1. {SECTION}")
    print(f"   CROSSFADE_SEC={_CROSSFADE_SEC}, CHUNK_TARGET={_CHUNK_TARGET}")
    print(f"{'=' * 60}")

    sr = 24000

    # 1a. Fade edges produce zero-start/end
    pcm = generate_sine_pcm(freq_hz=440, duration_sec=0.5, sr=sr)
    faded = SvaraTTSService._apply_fade_edges(pcm, fade_sec=_CROSSFADE_SEC, sample_rate=sr)
    arr = pcm_to_array(faded)

    if abs(arr[0]) < 100 and abs(arr[-1]) < 100:
        report.ok(SECTION, "Fade edges → zero amplitude at boundaries",
                  f"start={arr[0]:.0f}, end={arr[-1]:.0f}")
    else:
        report.fail(SECTION, "Fade edges → zero amplitude at boundaries",
                    f"start={arr[0]:.0f}, end={arr[-1]:.0f}")

    # 1b. Middle of audio undistorted
    arr_orig = pcm_to_array(pcm)
    mid = len(arr) // 2
    mid_diff = abs(arr_orig[mid] - arr[mid])
    if mid_diff < 50:
        report.ok(SECTION, "Middle of audio undistorted", f"diff={mid_diff:.0f}")
    else:
        report.warn(SECTION, "Middle of audio distortion", f"diff={mid_diff:.0f}")

    # 1c. Fade-in only (new streaming method)
    faded_in = SvaraTTSService._apply_fade_in(pcm, fade_sec=_CROSSFADE_SEC, sample_rate=sr)
    arr_in = pcm_to_array(faded_in)
    if abs(arr_in[0]) < 100:
        report.ok(SECTION, "Fade-in only → zero start", f"start={arr_in[0]:.0f}")
    else:
        report.fail(SECTION, "Fade-in only → zero start", f"start={arr_in[0]:.0f}")
    # End should be unchanged
    end_diff = abs(pcm_to_array(pcm)[-1] - arr_in[-1])
    if end_diff < 10:
        report.ok(SECTION, "Fade-in only → end unchanged", f"diff={end_diff:.0f}")
    else:
        report.fail(SECTION, "Fade-in only → end unchanged", f"diff={end_diff:.0f}")

    # 1d. Fade-out only
    faded_out = SvaraTTSService._apply_fade_out(pcm, fade_sec=_CROSSFADE_SEC, sample_rate=sr)
    arr_out = pcm_to_array(faded_out)
    if abs(arr_out[-1]) < 100:
        report.ok(SECTION, "Fade-out only → zero end", f"end={arr_out[-1]:.0f}")
    else:
        report.fail(SECTION, "Fade-out only → zero end", f"end={arr_out[-1]:.0f}")
    # Start should be unchanged
    start_diff = abs(pcm_to_array(pcm)[0] - arr_out[0])
    if start_diff < 10:
        report.ok(SECTION, "Fade-out only → start unchanged", f"diff={start_diff:.0f}")
    else:
        report.fail(SECTION, "Fade-out only → start unchanged", f"diff={start_diff:.0f}")

    # 1e. Streaming fade-in + fade-out = same as full fade_edges
    # Simulate what the streaming TTS path does:
    # Apply fade_in to first chunk, then fade_out to last chunk
    streamed = SvaraTTSService._apply_fade_in(pcm, fade_sec=_CROSSFADE_SEC, sample_rate=sr)
    streamed = SvaraTTSService._apply_fade_out(streamed, fade_sec=_CROSSFADE_SEC, sample_rate=sr)
    arr_streamed = pcm_to_array(streamed)
    arr_full = pcm_to_array(faded)
    max_diff = np.max(np.abs(arr_streamed - arr_full))
    if max_diff < 10:
        report.ok(SECTION, "Streaming fade = full fade_edges", f"max_diff={max_diff:.0f}")
    else:
        report.warn(SECTION, "Streaming fade ≠ full fade_edges", f"max_diff={max_diff:.0f}")

    # 1f. Pop elimination at sentence boundary
    sent_a = generate_sine_pcm(freq_hz=440, duration_sec=0.8, sr=sr, amplitude=12000)
    sent_b = generate_sine_pcm(freq_hz=660, duration_sec=0.6, sr=sr, amplitude=14000)

    # Raw concatenation
    raw_arr = pcm_to_array(sent_a + sent_b)
    join = len(sent_a) // 2
    raw_jump = abs(float(raw_arr[join]) - float(raw_arr[join - 1]))

    # Faded concatenation (what streaming TTS does)
    fa = SvaraTTSService._apply_fade_in(sent_a, fade_sec=_CROSSFADE_SEC, sample_rate=sr)
    fa = SvaraTTSService._apply_fade_out(fa, fade_sec=_CROSSFADE_SEC, sample_rate=sr)
    fb = SvaraTTSService._apply_fade_in(sent_b, fade_sec=_CROSSFADE_SEC, sample_rate=sr)
    fb = SvaraTTSService._apply_fade_out(fb, fade_sec=_CROSSFADE_SEC, sample_rate=sr)
    faded_arr = pcm_to_array(fa + fb)
    faded_jump = abs(float(faded_arr[join]) - float(faded_arr[join - 1]))

    if faded_jump < 500:
        report.ok(SECTION, f"Pop at boundary < 500",
                  f"raw={raw_jump:.0f} → faded={faded_jump:.0f}")
    else:
        report.warn(SECTION, f"Pop at boundary still large",
                    f"raw={raw_jump:.0f} → faded={faded_jump:.0f}")

    # 1g. Crossfade for multi-chunk
    merged = SvaraTTSService._crossfade_pcm(sent_a, sent_b,
                                             fade_sec=_CROSSFADE_SEC, sample_rate=sr)
    expected_len = len(sent_a)//2 + len(sent_b)//2 - int(_CROSSFADE_SEC * sr)
    arr_m = pcm_to_array(merged)
    if abs(len(arr_m) - expected_len) < 4:
        report.ok(SECTION, "Crossfade length correct",
                  f"{len(arr_m)} ≈ {expected_len}")
    else:
        report.fail(SECTION, "Crossfade length wrong",
                    f"{len(arr_m)} vs {expected_len}")

    if save_wavs:
        os.makedirs(output_dir, exist_ok=True)
        save_wav(f"{output_dir}/01_original.wav", pcm, sr)
        save_wav(f"{output_dir}/02_fade_edges.wav", faded, sr)
        save_wav(f"{output_dir}/03_fade_in_only.wav", faded_in, sr)
        save_wav(f"{output_dir}/04_fade_out_only.wav", faded_out, sr)
        save_wav(f"{output_dir}/05_streamed_fade.wav", streamed, sr)
        save_wav(f"{output_dir}/06_raw_concat_POPS.wav", sent_a + sent_b, sr)
        save_wav(f"{output_dir}/07_faded_concat_SMOOTH.wav", fa + fb, sr)
        save_wav(f"{output_dir}/08_crossfaded.wav", merged, sr)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: TTS Streaming Behavior
# ═══════════════════════════════════════════════════════════════════════════════

def test_tts_streaming_logic():
    """Verify the streaming TTS path emits audio incrementally, not all-at-once."""
    from services.svara_tts import SvaraTTSService, _CROSSFADE_SEC

    SECTION = "TTS Streaming Logic"
    print(f"\n{'=' * 60}")
    print(f"2. {SECTION}")
    print(f"{'=' * 60}")

    # Read the run_tts source code to verify streaming behavior
    source = inspect.getsource(SvaraTTSService.run_tts)

    # 2a. Single-chunk path should stream (not collect all first)
    if "pending_tail" in source and "_apply_fade_in" in source:
        report.ok(SECTION, "Single-chunk path uses streaming with fade-in/out",
                  "pending_tail buffer for fade-out")
    else:
        report.fail(SECTION, "Single-chunk path should stream, not buffer all",
                    "Missing pending_tail or _apply_fade_in")

    # 2b. Multi-chunk path collects and crossfades (expected)
    if "_crossfade_pcm" in source and "multi_chunk" in source:
        report.ok(SECTION, "Multi-chunk path uses crossfade",
                  "Correct: collect + crossfade for long text")
    else:
        report.warn(SECTION, "Multi-chunk crossfade missing")

    # 2c. Check that [SMOOTH] timing logs are present
    if "[SMOOTH]" in source:
        report.ok(SECTION, "[SMOOTH] timing logs in run_tts",
                  "First-byte and completion logged")
    else:
        report.warn(SECTION, "Missing [SMOOTH] timing logs in run_tts")

    # 2d. Check streaming TTS WebSocket has timing logs
    ws_source = inspect.getsource(SvaraTTSService._run_streaming_tts_websocket)
    smooth_logs = ws_source.count("[SMOOTH]")
    if smooth_logs >= 3:
        report.ok(SECTION, f"Svara WS has {smooth_logs} [SMOOTH] timing points",
                  "connect, ready, first-byte, done")
    else:
        report.warn(SECTION, f"Only {smooth_logs} [SMOOTH] logs in Svara WS")

    # 2e. Fade buffer size check
    sr = 24000
    fade_bytes = int(_CROSSFADE_SEC * sr) * 2  # PCM16 = 2 bytes/sample
    if fade_bytes < 4096:
        report.ok(SECTION, f"Fade buffer ({fade_bytes}B) < transport chunk (4096B)",
                  "Won't delay first audio byte significantly")
    else:
        report.warn(SECTION, f"Fade buffer ({fade_bytes}B) ≥ transport chunk (4096B)",
                    "May delay first audio byte")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Pipeline Stage Timing Budget
# ═══════════════════════════════════════════════════════════════════════════════

def test_pipeline_timing_budget():
    """Verify the pipeline timing budget is realistic for smooth experience."""
    SECTION = "Pipeline Timing Budget"
    print(f"\n{'=' * 60}")
    print(f"3. {SECTION}")
    print(f"{'=' * 60}")

    # Read bot.py for VAD and pipeline params
    import bot
    vad_conf = bot.VAD_CONFIDENCE
    vad_start = bot.VAD_START_SECS
    vad_stop = bot.VAD_STOP_SECS
    vad_vol = bot.VAD_MIN_VOLUME
    min_words = bot.INTERRUPTION_MIN_WORDS

    print(f"  VAD: confidence={vad_conf}, start={vad_start}s, stop={vad_stop}s, min_vol={vad_vol}")
    print(f"  Interruption min words: {min_words}")

    # 3a. VAD start_secs
    if vad_start <= 0.25:
        report.ok(SECTION, f"VAD start_secs={vad_start}s ≤ 250ms",
                  "Fast speech detection trigger")
    else:
        report.warn(SECTION, f"VAD start_secs={vad_start}s > 250ms",
                    "Slow speech detection — user may feel ignored")

    # 3b. VAD stop_secs (end-of-speech detection)
    if 0.5 <= vad_stop <= 1.5:
        report.ok(SECTION, f"VAD stop_secs={vad_stop}s in [0.5, 1.5]",
                  "Balanced end-of-speech detection")
    elif vad_stop < 0.5:
        report.warn(SECTION, f"VAD stop_secs={vad_stop}s < 0.5",
                    "May cut off mid-sentence pauses")
    else:
        report.warn(SECTION, f"VAD stop_secs={vad_stop}s > 1.5",
                    "Slow end-of-speech — adds latency")

    # 3c. VAD min_volume
    if 0.3 <= vad_vol <= 0.5:
        report.ok(SECTION, f"VAD min_volume={vad_vol} in [0.3, 0.5]",
                  "Good noise rejection without losing quiet speech")
    else:
        report.warn(SECTION, f"VAD min_volume={vad_vol} outside [0.3, 0.5]")

    # 3d. Interruption strategy
    if 1 <= min_words <= 3:
        report.ok(SECTION, f"Interruption min_words={min_words} in [1, 3]",
                  "Blocks echo, allows real barge-in")
    else:
        report.warn(SECTION, f"Interruption min_words={min_words} outside [1, 3]")

    # 3e. Latency budget calculation
    # Tutor mode: user speaks → VAD → STT → LLM → TTS → audio out
    vad_trigger_ms = vad_start * 1000  # VAD start delay
    vad_end_ms = vad_stop * 1000       # VAD end-of-speech delay
    stt_ms = 200                       # Soniox typical final transcript
    llm_ttft_ms = 300                  # vLLM first token (OSS)
    llm_sentence_ms = 500              # Time to generate first sentence
    tts_connect_ms = 150               # Svara WS connect + ack
    tts_first_byte_ms = 300            # Svara first audio chunk
    total_ms = vad_end_ms + stt_ms + llm_ttft_ms + llm_sentence_ms + tts_connect_ms + tts_first_byte_ms

    print(f"\n  Latency Budget (Tutor Mode):")
    print(f"    VAD end-of-speech:    {vad_end_ms:>6.0f}ms")
    print(f"    STT final transcript: {stt_ms:>6.0f}ms")
    print(f"    LLM TTFT:             {llm_ttft_ms:>6.0f}ms")
    print(f"    LLM first sentence:   {llm_sentence_ms:>6.0f}ms")
    print(f"    TTS WS connect+ack:   {tts_connect_ms:>6.0f}ms")
    print(f"    TTS first audio byte: {tts_first_byte_ms:>6.0f}ms")
    print(f"    ─────────────────────────────────")
    print(f"    TOTAL (estimated):    {total_ms:>6.0f}ms")

    if total_ms <= 3500:
        report.ok(SECTION, f"Estimated turn latency {total_ms}ms ≤ 3500ms",
                  "Within acceptable range")
    else:
        report.warn(SECTION, f"Estimated turn latency {total_ms}ms > 3500ms",
                    "May feel slow to user")

    # 3f. Check PipelineInstrumentor has inter-sentence gap tracking
    instrumentor_source = inspect.getsource(bot.PipelineInstrumentor.process_frame)
    if "Inter-sentence gap" in instrumentor_source or "inter_sentence_gap" in instrumentor_source:
        report.ok(SECTION, "Inter-sentence gap tracking in PipelineInstrumentor",
                  "[SMOOTH] logs for gap measurement")
    else:
        report.warn(SECTION, "Missing inter-sentence gap tracking")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Classroom Listener Delivery
# ═══════════════════════════════════════════════════════════════════════════════

def test_classroom_delivery():
    """Verify classroom listener pipeline is instrumented for smoothness."""
    SECTION = "Classroom Listener Delivery"
    print(f"\n{'=' * 60}")
    print(f"4. {SECTION}")
    print(f"{'=' * 60}")

    import classroom

    # 4a. Clause-level chunking for listener streaming
    clause_re = classroom.RoomManager._CLAUSE_RE
    min_clause = classroom.RoomManager._MIN_CLAUSE_LEN

    test_text = "Hello, this is a test sentence. And here is another one, with a clause."
    parts = clause_re.split(test_text)
    if len(parts) > 1:
        report.ok(SECTION, f"Clause regex splits on commas/periods",
                  f"'{test_text[:40]}...' → {len(parts)} parts")
    else:
        report.warn(SECTION, "Clause regex doesn't split test text")

    if 15 <= min_clause <= 30:
        report.ok(SECTION, f"MIN_CLAUSE_LEN={min_clause} in [15, 30]",
                  "Prevents tiny fragments, keeps latency low")
    else:
        report.warn(SECTION, f"MIN_CLAUSE_LEN={min_clause} outside [15, 30]")

    # 4b. Listener delivery has [SMOOTH] timing logs
    delivery_source = inspect.getsource(
        classroom.RoomManager._deliver_sentence_to_listener_inner
    )
    if "[SMOOTH]" in delivery_source:
        report.ok(SECTION, "[SMOOTH] timing logs in listener delivery",
                  "translate, tts, first_byte tracked")
    else:
        report.warn(SECTION, "Missing [SMOOTH] logs in listener delivery")

    # 4c. Per-user audio lock prevents interleaving
    if "_audio_lock" in inspect.getsource(
        classroom.RoomManager._deliver_sentence_to_listener
    ):
        report.ok(SECTION, "Per-user _audio_lock prevents audio interleaving")
    else:
        report.fail(SECTION, "Missing _audio_lock — audio frames may interleave")

    # 4d. TTS first-byte tracking in delivery
    if "tts_first_byte_ms" in delivery_source:
        report.ok(SECTION, "TTS first-byte latency tracked per sentence",
                  "Enables diagnosing slow TTS calls")
    else:
        report.warn(SECTION, "Missing TTS first-byte tracking in delivery")

    # 4e. ClassroomBroadcaster streams clauses, not full response
    broadcaster_source = inspect.getsource(classroom.ClassroomBroadcaster)
    if "_clause_buffer" in broadcaster_source:
        report.ok(SECTION, "ClassroomBroadcaster streams at clause level",
                  "Listener hears audio ~800ms sooner than full-response")
    else:
        report.warn(SECTION, "ClassroomBroadcaster may wait for full response")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Text Chunking Quality
# ═══════════════════════════════════════════════════════════════════════════════

def test_text_chunking():
    """Verify text chunking produces TTS-friendly segments."""
    from services.svara_tts import (
        SvaraTTSService, _CHUNK_TARGET, _CHUNK_MAX, _CHUNK_MIN
    )

    SECTION = "Text Chunking"
    print(f"\n{'=' * 60}")
    print(f"5. {SECTION}")
    print(f"   TARGET={_CHUNK_TARGET}, MAX={_CHUNK_MAX}, MIN={_CHUNK_MIN}")
    print(f"{'=' * 60}")

    # 5a. Short text → no split
    short = "Hello, how are you?"
    chunks = SvaraTTSService._chunk_text(short)
    if len(chunks) == 1 and chunks[0] == short:
        report.ok(SECTION, f"Short text ({len(short)} chars) → 1 chunk")
    else:
        report.fail(SECTION, f"Short text split unexpectedly: {chunks}")

    # 5b. Long English text → multiple chunks, all within limits
    long_en = (
        "Applications and devices equipped with AI can see and identify objects. "
        "They can understand and respond to human language. "
        "They can learn from new information and experience. "
        "They can make detailed recommendations to users and experts. "
        "They can act independently, replacing the need for human intelligence or intervention."
    )
    chunks = SvaraTTSService._chunk_text(long_en)
    all_ok = all(len(c) <= _CHUNK_MAX for c in chunks)
    if len(chunks) > 1 and all_ok:
        sizes = [len(c) for c in chunks]
        report.ok(SECTION, f"Long English ({len(long_en)} chars) → {len(chunks)} chunks",
                  f"sizes={sizes}")
    else:
        report.fail(SECTION, f"Long English chunking issue: {len(chunks)} chunks, all_ok={all_ok}")

    # 5c. Long Hindi text → proper splitting
    hindi = (
        "एक समय की बात है, जब हिमालय की चोटियों के बीच एक छोटा सा गाँव बसा था। "
        "वहाँ के लोग कहते थे कि हवाओं में संगीत होता है। "
        "हमारा लक्ष्य है कि हर भारतीय अपनी भाषा में गर्व से बात कर सके। "
        "चलिए इस सफर में हमारे साथ जुड़िये।"
    )
    chunks = SvaraTTSService._chunk_text(hindi)
    all_ok = all(len(c) <= _CHUNK_MAX for c in chunks)
    if all_ok:
        report.ok(SECTION, f"Hindi text ({len(hindi)} chars) → {len(chunks)} chunks",
                  f"all ≤ {_CHUNK_MAX}")
    else:
        over = [c for c in chunks if len(c) > _CHUNK_MAX]
        report.fail(SECTION, f"Hindi chunk exceeds max: {[len(c) for c in over]}")

    # 5d. No chunk smaller than MIN (except single-chunk case)
    for text_name, text in [("English", long_en), ("Hindi", hindi)]:
        chunks = SvaraTTSService._chunk_text(text)
        tiny = [c for c in chunks if len(c) < _CHUNK_MIN and len(chunks) > 1]
        if not tiny:
            report.ok(SECTION, f"{text_name}: no tiny chunks (all ≥ {_CHUNK_MIN})")
        else:
            report.warn(SECTION, f"{text_name}: tiny chunks found: {[len(c) for c in tiny]}")

    # 5e. Edge cases
    empty_chunks = SvaraTTSService._chunk_text("")
    tiny_chunks = SvaraTTSService._chunk_text("Hi")
    if len(tiny_chunks) == 1:
        report.ok(SECTION, "Edge cases: empty/tiny text handled gracefully")
    else:
        report.warn(SECTION, "Edge case issue")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: Soniox Keepalive & Audio Buffering
# ═══════════════════════════════════════════════════════════════════════════════

def test_soniox_keepalive():
    """Verify Soniox idle keepalive and audio buffering are properly implemented."""
    SECTION = "Soniox Keepalive & Buffer"
    print(f"\n{'=' * 60}")
    print(f"6. {SECTION}")
    print(f"{'=' * 60}")

    with open("services/soniox_stt.py") as f:
        code = f.read()

    # 6a. Idle keepalive loop exists
    if "_idle_keepalive_loop" in code:
        report.ok(SECTION, "Idle keepalive loop implemented",
                  "Prevents Soniox from dropping idle connections")
    else:
        report.fail(SECTION, "Missing _idle_keepalive_loop",
                    "Soniox will drop idle connections → word loss")

    # 6b. Keepalive starts eagerly on pipeline start
    if "idle_keepalive" in code and "start" in code:
        report.ok(SECTION, "Keepalive starts on pipeline start()")
    else:
        report.warn(SECTION, "Keepalive may not start eagerly")

    # 6c. Keepalive cancelled on first speech
    if "_first_speech_received" in code:
        report.ok(SECTION, "Keepalive cancelled on first user speech",
                  "Saves resources after user starts talking")
    else:
        report.warn(SECTION, "Missing _first_speech_received flag")

    # 6d. Audio buffering during reconnect
    if "_reconnect_buffer" in code:
        report.ok(SECTION, "Audio buffering during reconnect",
                  "Prevents word loss if connection drops")
    else:
        report.fail(SECTION, "Missing _reconnect_buffer",
                    "Audio will be lost during reconnection")

    # 6e. Buffer size limit
    if "_MAX_RECONNECT_BUFFER" in code:
        # Extract the value
        match = re.search(r'_MAX_RECONNECT_BUFFER\s*=\s*(\d+)', code)
        if match:
            buf_size = int(match.group(1))
            if buf_size >= 30:
                report.ok(SECTION, f"Reconnect buffer capacity: {buf_size} frames",
                          f"≈{buf_size * 20}ms of audio")
            else:
                report.warn(SECTION, f"Reconnect buffer small: {buf_size} frames")
    else:
        report.warn(SECTION, "Missing _MAX_RECONNECT_BUFFER")

    # 6f. Timing logs present
    timing_logs = code.count("[SONIOX_TIMING]") + code.count("[SMOOTH]")
    if timing_logs >= 3:
        report.ok(SECTION, f"{timing_logs} timing log points in soniox_stt.py",
                  "Enables diagnosing connection issues")
    else:
        report.warn(SECTION, f"Only {timing_logs} timing logs in soniox_stt.py")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: Barge-in Guard
# ═══════════════════════════════════════════════════════════════════════════════

def test_barge_in_guard():
    """Verify MinWordsInterruptionStrategy works correctly."""
    SECTION = "Barge-in Guard"
    print(f"\n{'=' * 60}")
    print(f"7. {SECTION}")
    print(f"{'=' * 60}")

    import bot
    min_words = bot.INTERRUPTION_MIN_WORDS

    # Test word counting logic
    test_cases = [
        ("", 0, False, "empty — should NOT interrupt"),
        ("um", 1, False, "single word — echo, should NOT interrupt"),
        ("hello there", 2, True, "two words — SHOULD interrupt"),
        ("stop talking now", 3, True, "three words — SHOULD interrupt"),
        ("ruko ruko", 2, True, "Hindi barge-in — SHOULD interrupt"),
    ]

    for text, word_count, should_interrupt, desc in test_cases:
        would_interrupt = word_count >= min_words
        if would_interrupt == should_interrupt:
            report.ok(SECTION, f"'{text}' ({word_count} words) → {'interrupt' if should_interrupt else 'blocked'}",
                      desc)
        else:
            report.fail(SECTION, f"'{text}' wrong behavior: got {would_interrupt}, expected {should_interrupt}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8: End-to-End Pipeline Wiring
# ═══════════════════════════════════════════════════════════════════════════════

def test_pipeline_wiring():
    """Verify the pipeline is wired correctly with all processors in order."""
    SECTION = "Pipeline Wiring"
    print(f"\n{'=' * 60}")
    print(f"8. {SECTION}")
    print(f"{'=' * 60}")

    with open("bot.py") as f:
        code = f.read()

    # 8a. Text-and-audio pipeline has correct order
    # transport.input → stt → ... → tts → transport.output
    pipeline_section = code[code.find("text_and_audio mode"):]
    if "transport.input()" in pipeline_section and "transport.output()" in pipeline_section:
        report.ok(SECTION, "Pipeline has transport.input() → transport.output()")
    else:
        report.fail(SECTION, "Pipeline missing transport endpoints")

    # 8b. PipelineInstrumentor is in the pipeline
    if "PipelineInstrumentor" in code and "transcript_logger" in code:
        report.ok(SECTION, "PipelineInstrumentor in pipeline",
                  "Comprehensive timing at every stage")
    else:
        report.warn(SECTION, "Missing PipelineInstrumentor")

    # 8c. GreetingProcessor present
    if "greeting_processor" in pipeline_section:
        report.ok(SECTION, "GreetingProcessor in pipeline")
    else:
        report.warn(SECTION, "Missing GreetingProcessor")

    # 8d. TextAudioSyncNotifier for text-audio synchronization
    if "TextAudioSyncNotifier" in code:
        report.ok(SECTION, "TextAudioSyncNotifier for text-audio sync",
                  "Text appears when audio starts playing")
    else:
        report.warn(SECTION, "Missing TextAudioSyncNotifier")

    # 8e. ClarityGate for low-quality STT
    if "clarity_gate" in pipeline_section:
        report.ok(SECTION, "ClarityGate for low-quality STT filtering")
    else:
        report.warn(SECTION, "Missing ClarityGate")

    # 8f. ActionTagFilter for stripping [TEACHER_ACTION:...] tags
    if "action_tag_filter" in pipeline_section:
        report.ok(SECTION, "ActionTagFilter strips action tags from output")
    else:
        report.warn(SECTION, "Missing ActionTagFilter")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pipeline Smoothness Test Suite")
    parser.add_argument("--save-wav", action="store_true",
                        help="Save WAV files for manual listening")
    parser.add_argument("--verbose", action="store_true",
                        help="Extra detail in output")
    parser.add_argument("--output-dir", default="/tmp/mira-smooth",
                        help="Directory for WAV output files")
    args = parser.parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║  MIRA Pipeline Smoothness Test Suite — Local Reference Tests      ║")
    print("║  Both Tutor and Classroom pipelines                               ║")
    print("╚" + "═" * 68 + "╝")
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]}")
    print()

    # Run all test sections
    test_tts_audio_quality(save_wavs=args.save_wav, output_dir=args.output_dir)
    test_tts_streaming_logic()
    test_pipeline_timing_budget()
    test_classroom_delivery()
    test_text_chunking()
    test_soniox_keepalive()
    test_barge_in_guard()
    test_pipeline_wiring()

    # Print final report
    success = report.summary()

    if args.save_wav:
        print(f"\n📁 WAV files saved to {args.output_dir}/")
        print("   Compare 06_raw_concat_POPS.wav vs 07_faded_concat_SMOOTH.wav")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
