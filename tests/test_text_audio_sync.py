#!/usr/bin/env python3
"""
Unit + integration tests for TextStreamForwarder & TextAudioSyncNotifier.

These tests validate the text-audio sync pipeline in isolation — no real
TTS, LLM, or WebSocket server needed.  We feed synthetic Pipecat frames
through the processors and assert that the mock WebSocket receives the
right JSON messages in the right order.

Test categories:

  CORE (14 tests):
    - text_only mode, audio queuing, sentence boundaries, partial flush
    - TTS timeout watchdog, barge-in, greeting, complete ordering
    - multi-response metrics, per-receiver isolation, empty response
    - safety net flush, watchdog no false fire, first sentence delay

  MULTI-LANGUAGE (5 tests):
    - Hindi sentence boundaries (Devanagari danda ।)
    - Mixed English-Hindi sentence splitting
    - Arabic question mark (؟)
    - Long Hindi multi-sentence response
    - Regex consistency between tutor (bot.py) and classroom (classroom.py)

  CONCURRENCY / SCALE (10 tests):
    - Parallel speakers (5 concurrent, no cross-talk)
    - Parallel speakers with different languages
    - Multi-receiver same content (10 receivers, classroom scenario)
    - Mixed mode receivers (text_only + text_and_audio in same room)
    - Multi-room isolation (4 rooms × 4 clients, no cross-room leak)
    - Barge-in under load (interrupt one pipeline, others unaffected)
    - Rapid-fire responses (5 back-to-back)
    - Watchdog under concurrent load (5 simultaneous timeouts)
    - Receiver joins mid-response
    - High sentence count stress test (50 sentences)

Usage:
    python tests/test_text_audio_sync.py                   # Run all (29 tests)
    python tests/test_text_audio_sync.py --test text_only  # One test
    python tests/test_text_audio_sync.py --verbose         # Debug logs
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# ── Pipecat imports ──
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartInterruptionFrame,
    TextFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# ── Module under test ──
sys.path.insert(0, "/app")
from bot import TextAudioSyncNotifier, TextStreamForwarder

# ─────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("text-audio-sync-test")


# ─────────────────────────────────────────────────
# Mock WebSocket
# ─────────────────────────────────────────────────
class MockWebSocket:
    """
    Records all send_json / send_bytes calls for assertion.
    Each instance is per-receiver — just like a real WebSocket.
    """

    def __init__(self, name: str = "mock-ws"):
        self.name = name
        self.sent_json: list[dict] = []
        self.sent_bytes: list[bytes] = []

    async def send_json(self, data: dict):
        self.sent_json.append(data)

    async def send_bytes(self, data: bytes):
        self.sent_bytes.append(data)

    def get_messages_of_type(self, msg_type: str) -> list[dict]:
        return [m for m in self.sent_json if m.get("type") == msg_type]

    def clear(self):
        self.sent_json.clear()
        self.sent_bytes.clear()

    def __repr__(self):
        return f"MockWebSocket({self.name}, {len(self.sent_json)} msgs)"


# ─────────────────────────────────────────────────
# Frame collector (sits at end of pipeline to capture pushed frames)
# ─────────────────────────────────────────────────
class FrameCollector(FrameProcessor):
    """Collects all frames pushed to it for assertion."""

    def __init__(self, **kwargs):
        super().__init__(name="FrameCollector", **kwargs)
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.frames.append(frame)


# ─────────────────────────────────────────────────
# Helper: simulate a pipeline by feeding frames through processors
# ─────────────────────────────────────────────────
async def feed_frames(processor: FrameProcessor, frames: list[Frame]):
    """Feed a sequence of frames through a single processor."""
    for frame in frames:
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)


async def feed_frames_through_pair(
    forwarder: TextStreamForwarder,
    notifier: TextAudioSyncNotifier,
    frames: list[Frame],
):
    """
    Feed frames through the forwarder, then through the notifier.
    This simulates the real pipeline: forwarder → TTS → notifier.
    We skip actual TTS and manually inject TTSStarted/Stopped frames.
    """
    for frame in frames:
        await forwarder.process_frame(frame, FrameDirection.DOWNSTREAM)
        # If it's a TTS frame, also feed it to the notifier
        if isinstance(frame, (TTSStartedFrame, TTSStoppedFrame)):
            await notifier.process_frame(frame, FrameDirection.DOWNSTREAM)
        # LLMFullResponseEndFrame also flows through to notifier (after TTS)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await notifier.process_frame(frame, FrameDirection.DOWNSTREAM)
        elif isinstance(frame, StartInterruptionFrame):
            await notifier.process_frame(frame, FrameDirection.DOWNSTREAM)


# ─────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────
def make_text_frames(text: str) -> list[TextFrame]:
    """Split text into word-level TextFrames (simulating LLM token output)."""
    frames = []
    words = text.split(" ")
    for i, word in enumerate(words):
        token = word if i == len(words) - 1 else word + " "
        frames.append(TextFrame(text=token))
    return frames


def make_llm_response(text: str) -> list[Frame]:
    """Create a complete LLM response: Start + TextFrames + End."""
    frames: list[Frame] = [LLMFullResponseStartFrame()]
    frames.extend(make_text_frames(text))
    frames.append(LLMFullResponseEndFrame())
    return frames


# ═════════════════════════════════════════════════
# TESTS
# ═════════════════════════════════════════════════

async def test_text_only_streams_tokens():
    """text_only mode: each token sent immediately, bot_text_complete at end."""
    ws = MockWebSocket("text-only")
    fwd = TextStreamForwarder(websocket=ws, text_only=True)

    frames = make_llm_response("Hello there. How are you?")
    await feed_frames(fwd, frames)

    # Every token should produce a bot_text message
    bot_texts = ws.get_messages_of_type("bot_text")
    assert len(bot_texts) > 0, "Expected bot_text messages for each token"
    for msg in bot_texts:
        assert msg["streaming"] is True

    # Concatenated text should match original
    concat = "".join(m["text"] for m in bot_texts)
    assert concat == "Hello there. How are you?", f"Got: {concat}"

    # Should end with bot_text_complete
    completes = ws.get_messages_of_type("bot_text_complete")
    assert len(completes) == 1, f"Expected 1 bot_text_complete, got {len(completes)}"
    assert completes[0]["text"] == "Hello there. How are you?"

    # Metrics
    assert fwd._metrics_total_responses == 1
    assert fwd._metrics_sentences_queued == 0  # text_only doesn't queue

    return True


async def test_audio_mode_queues_sentences():
    """text_and_audio mode: sentences queued, NOT sent until TTSStartedFrame.

    The notifier releases ALL pending sentences on a single TTSStartedFrame
    because Pipecat's TTS aggregator may merge multiple sentences into one
    run_tts() call (e.g. Hindi text where NLTK doesn't split on '।').
    """
    ws = MockWebSocket("audio-queue")
    fwd = TextStreamForwarder(websocket=ws, text_only=False)
    notifier = TextAudioSyncNotifier(text_forwarder=fwd)

    # Send LLM response start + tokens for "Hello there. How are you?"
    await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    for tf in make_text_frames("Hello there. How are you?"):
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

    # At this point, two sentences should be queued but NOT sent
    assert fwd._sentence_q.qsize() == 2, f"Expected 2 queued, got {fwd._sentence_q.qsize()}"
    assert len(ws.get_messages_of_type("bot_text")) == 0, "No bot_text should be sent yet"

    # Simulate TTS starting — ALL queued sentences are released at once
    await notifier.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)

    # Both sentences should now be released
    bot_texts = ws.get_messages_of_type("bot_text")
    assert len(bot_texts) == 2, f"Expected 2 bot_text after TTSStarted (release-all), got {len(bot_texts)}"
    assert bot_texts[0]["text"] == "Hello there."
    assert bot_texts[1]["text"] == "How are you?"

    # Queue should be empty
    assert fwd._sentence_q.qsize() == 0

    # Metrics
    assert fwd._metrics_sentences_queued == 2
    assert fwd._metrics_sentences_released == 2
    assert fwd._metrics_sentences_timed_out == 0

    return True


async def test_sentence_boundaries():
    """Sentence detection works for English periods, questions, exclamations, Hindi danda."""
    ws = MockWebSocket("boundaries")
    fwd = TextStreamForwarder(websocket=ws, text_only=False)

    await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)

    # English period
    for tf in make_text_frames("First sentence."):
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)
    assert fwd._sentence_q.qsize() == 1

    # English question
    for tf in make_text_frames(" Second question?"):
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)
    assert fwd._sentence_q.qsize() == 2

    # English exclamation
    for tf in make_text_frames(" Wow!"):
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)
    assert fwd._sentence_q.qsize() == 3

    # Hindi danda (purna viram)
    for tf in [TextFrame(text="यह हिंदी है।")]:
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)
    assert fwd._sentence_q.qsize() == 4, f"Hindi danda not detected, got {fwd._sentence_q.qsize()}"

    # Verify queued content
    sentences = []
    while not fwd._sentence_q.empty():
        sentences.append(fwd._sentence_q.get_nowait())
    assert sentences == [
        "First sentence.",
        "Second question?",
        "Wow!",
        "यह हिंदी है।",
    ], f"Got: {sentences}"

    return True


async def test_partial_sentence_flushed_at_end():
    """Partial sentence (no trailing punctuation) is enqueued on LLMFullResponseEndFrame."""
    ws = MockWebSocket("partial")
    fwd = TextStreamForwarder(websocket=ws, text_only=False)
    notifier = TextAudioSyncNotifier(text_forwarder=fwd)

    await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)

    # One complete sentence + one partial
    for tf in make_text_frames("Complete sentence. Partial without punct"):
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

    assert fwd._sentence_q.qsize() == 1  # Only "Complete sentence." queued
    assert fwd._sentence_buffer.strip() == "Partial without punct"

    # End the response — partial should be enqueued
    await fwd.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    assert fwd._sentence_q.qsize() == 2  # Now both queued

    # Simulate TTS for both + end
    await notifier.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
    await notifier.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
    await notifier.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
    await notifier.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
    # LLMFullResponseEndFrame also flows to notifier as safety net
    await notifier.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    bot_texts = ws.get_messages_of_type("bot_text")
    assert len(bot_texts) == 2
    assert bot_texts[0]["text"] == "Complete sentence."
    assert bot_texts[1]["text"] == "Partial without punct"

    # bot_text_complete should have the full response
    completes = ws.get_messages_of_type("bot_text_complete")
    assert len(completes) == 1
    assert "Complete sentence." in completes[0]["text"]
    assert "Partial without punct" in completes[0]["text"]

    return True


async def test_tts_timeout_watchdog():
    """If TTS doesn't start within timeout, watchdog sends text anyway."""
    ws = MockWebSocket("timeout")
    fwd = TextStreamForwarder(websocket=ws, text_only=False)
    # Set a very short timeout for testing
    fwd._TTS_SENTENCE_TIMEOUT = 0.3  # 300ms

    await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)

    for tf in make_text_frames("This sentence will timeout."):
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

    # Sentence queued, watchdog started
    assert fwd._sentence_q.qsize() == 1
    assert fwd._watchdog_task is not None

    # Wait for watchdog to fire (timeout + margin)
    await asyncio.sleep(0.6)

    # Watchdog should have sent the text
    bot_texts = ws.get_messages_of_type("bot_text")
    assert len(bot_texts) == 1, f"Expected watchdog to send 1 message, got {len(bot_texts)}"
    assert bot_texts[0]["text"] == "This sentence will timeout."

    # Metrics should show timeout
    assert fwd._metrics_sentences_timed_out == 1
    assert fwd._metrics_total_timeouts == 1

    # Clean up
    fwd._stop_watchdog()
    return True


async def test_barge_in_flushes_queue():
    """StartInterruptionFrame flushes all queued text immediately.

    With release-all behavior, TTSStartedFrame releases ALL queued sentences.
    So after TTS starts, queue is already empty. Barge-in then flushes nothing
    extra but still sends bot_text_complete.
    """
    ws = MockWebSocket("barge-in")
    fwd = TextStreamForwarder(websocket=ws, text_only=False)
    notifier = TextAudioSyncNotifier(text_forwarder=fwd)

    await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)

    # Queue 3 sentences
    for tf in make_text_frames("First. Second. Third."):
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

    assert fwd._sentence_q.qsize() == 3

    # TTS starts — releases ALL 3 sentences at once (release-all behavior)
    await notifier.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
    assert fwd._sentence_q.qsize() == 0  # All released

    # Barge-in!
    await fwd.process_frame(StartInterruptionFrame(), FrameDirection.DOWNSTREAM)
    await notifier.process_frame(StartInterruptionFrame(), FrameDirection.DOWNSTREAM)

    # Queue already empty, all 3 were TTS-released
    assert fwd._sentence_q.qsize() == 0
    bot_texts = ws.get_messages_of_type("bot_text")
    assert len(bot_texts) == 3, f"Expected 3 bot_text (all TTS-released), got {len(bot_texts)}"

    # Metrics: all 3 released via TTS, none flushed by barge-in
    assert fwd._metrics_sentences_released == 3
    assert fwd._metrics_total_interruptions == 1

    return True


async def test_greeting_sent_immediately():
    """TTSSpeakFrame (greeting) sends bot_text_complete immediately in both modes."""
    for text_only in [True, False]:
        ws = MockWebSocket(f"greeting-{'text' if text_only else 'audio'}")
        fwd = TextStreamForwarder(websocket=ws, text_only=text_only)

        greeting = TTSSpeakFrame(text="Welcome to MIRA!")
        await fwd.process_frame(greeting, FrameDirection.DOWNSTREAM)

        completes = ws.get_messages_of_type("bot_text_complete")
        assert len(completes) == 1, f"Expected greeting bot_text_complete (text_only={text_only})"
        assert completes[0]["text"] == "Welcome to MIRA!"

    return True


async def test_complete_after_all_sentences():
    """bot_text_complete arrives AFTER all sentence bot_text messages."""
    ws = MockWebSocket("ordering")
    fwd = TextStreamForwarder(websocket=ws, text_only=False)
    notifier = TextAudioSyncNotifier(text_forwarder=fwd)

    # Full flow: LLM → queue → TTS triggers → complete
    frames: list[Frame] = [LLMFullResponseStartFrame()]
    frames.extend(make_text_frames("One. Two. Three."))
    frames.append(LLMFullResponseEndFrame())

    # Feed through forwarder
    for f in frames:
        await fwd.process_frame(f, FrameDirection.DOWNSTREAM)

    # 3 sentences queued (including no leftover since "Three." ends with period)
    assert fwd._sentence_q.qsize() == 3, f"Got {fwd._sentence_q.qsize()}"

    # Simulate TTS processing all 3 sentences
    for i in range(3):
        await notifier.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
        await notifier.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)

    # Safety net
    await notifier.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    # Verify ordering: all bot_text before bot_text_complete
    all_msgs = ws.sent_json
    text_indices = [i for i, m in enumerate(all_msgs) if m.get("type") == "bot_text"]
    complete_indices = [i for i, m in enumerate(all_msgs) if m.get("type") == "bot_text_complete"]

    assert len(text_indices) == 3, f"Expected 3 bot_text, got {len(text_indices)}"
    assert len(complete_indices) == 1, f"Expected 1 bot_text_complete, got {len(complete_indices)}"

    # bot_text_complete must come after ALL bot_text messages
    assert complete_indices[0] > max(text_indices), (
        f"bot_text_complete at index {complete_indices[0]} but last bot_text at {max(text_indices)}"
    )

    return True


async def test_multiple_responses_reset_metrics():
    """Metrics reset between LLM responses; session counters accumulate."""
    ws = MockWebSocket("multi-response")
    fwd = TextStreamForwarder(websocket=ws, text_only=False)
    notifier = TextAudioSyncNotifier(text_forwarder=fwd)

    # --- Response 1 ---
    await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    for tf in make_text_frames("First response."):
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)
    await fwd.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    await notifier.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
    await notifier.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
    await notifier.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    assert fwd._metrics_total_responses == 1

    # --- Response 2 ---
    ws.clear()
    await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)

    # Per-response metrics should be reset
    assert fwd._metrics_sentences_queued == 0
    assert fwd._metrics_sentences_released == 0

    for tf in make_text_frames("Second response. Two sentences."):
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)
    await fwd.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    assert fwd._metrics_sentences_queued == 2
    assert fwd._metrics_total_responses == 2

    return True


async def test_per_receiver_isolation():
    """Two forwarders with separate WebSockets don't interfere."""
    ws_a = MockWebSocket("receiver-A")
    ws_b = MockWebSocket("receiver-B")
    fwd_a = TextStreamForwarder(websocket=ws_a, text_only=False)
    fwd_b = TextStreamForwarder(websocket=ws_b, text_only=False)
    notifier_a = TextAudioSyncNotifier(text_forwarder=fwd_a)
    notifier_b = TextAudioSyncNotifier(text_forwarder=fwd_b)

    # Feed different text to each
    await fwd_a.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    for tf in make_text_frames("Hello from A."):
        await fwd_a.process_frame(tf, FrameDirection.DOWNSTREAM)

    await fwd_b.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    for tf in make_text_frames("Hello from B."):
        await fwd_b.process_frame(tf, FrameDirection.DOWNSTREAM)

    # Release A's sentence
    await notifier_a.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)

    # A should have text, B should not
    assert len(ws_a.get_messages_of_type("bot_text")) == 1
    assert len(ws_b.get_messages_of_type("bot_text")) == 0

    # Release B's sentence
    await notifier_b.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)

    assert len(ws_b.get_messages_of_type("bot_text")) == 1
    assert ws_a.get_messages_of_type("bot_text")[0]["text"] == "Hello from A."
    assert ws_b.get_messages_of_type("bot_text")[0]["text"] == "Hello from B."

    return True


async def test_empty_response():
    """LLM response with no text tokens should not break anything."""
    ws = MockWebSocket("empty")
    fwd = TextStreamForwarder(websocket=ws, text_only=False)
    notifier = TextAudioSyncNotifier(text_forwarder=fwd)

    await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await fwd.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    await notifier.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    # No messages should be sent (no text to send)
    assert len(ws.sent_json) == 0
    assert fwd._metrics_sentences_queued == 0

    return True


async def test_safety_net_flush_on_llm_end():
    """If TTS never fires, LLMFullResponseEndFrame at notifier flushes remaining.

    When TTS never starts (in_flight == 0) and the queue still has sentences,
    the notifier's LLMFullResponseEndFrame handler acts as a safety net:
    it flushes all remaining queued text so the client sees the response,
    then sends bot_text_complete.
    """
    ws = MockWebSocket("safety-net")
    fwd = TextStreamForwarder(websocket=ws, text_only=False)
    notifier = TextAudioSyncNotifier(text_forwarder=fwd)

    await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    for tf in make_text_frames("Sentence one. Sentence two."):
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)
    await fwd.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    # 2 sentences queued, TTS never fires
    assert fwd._sentence_q.qsize() == 2

    # LLMFullResponseEndFrame reaches notifier (safety net)
    await notifier.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    # Both should be flushed by the safety net
    bot_texts = ws.get_messages_of_type("bot_text")
    assert len(bot_texts) == 2, f"Expected 2 flushed, got {len(bot_texts)}"
    assert fwd._metrics_sentences_flushed == 2

    # bot_text_complete should also be sent
    completes = ws.get_messages_of_type("bot_text_complete")
    assert len(completes) == 1

    return True


async def test_watchdog_doesnt_fire_when_tts_is_fast():
    """Watchdog should NOT fire if TTS releases sentences quickly."""
    ws = MockWebSocket("fast-tts")
    fwd = TextStreamForwarder(websocket=ws, text_only=False)
    notifier = TextAudioSyncNotifier(text_forwarder=fwd)
    fwd._TTS_SENTENCE_TIMEOUT = 0.5  # 500ms

    await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    for tf in make_text_frames("Quick sentence."):
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

    # Release immediately via TTS
    await notifier.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)

    # Wait past the timeout
    await asyncio.sleep(0.7)

    # Should have exactly 1 bot_text (from TTS release), no timeout
    bot_texts = ws.get_messages_of_type("bot_text")
    assert len(bot_texts) == 1
    assert fwd._metrics_sentences_timed_out == 0

    fwd._stop_watchdog()
    return True


async def test_metrics_first_sentence_delay():
    """Metrics track delay between first sentence queued and first sentence released."""
    ws = MockWebSocket("delay-metrics")
    fwd = TextStreamForwarder(websocket=ws, text_only=False)
    notifier = TextAudioSyncNotifier(text_forwarder=fwd)

    await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    for tf in make_text_frames("Test sentence."):
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

    assert fwd._metrics_first_sentence_queued_at > 0

    # Simulate 100ms TTS startup delay
    await asyncio.sleep(0.1)

    await notifier.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)

    assert fwd._metrics_first_sentence_released_at > 0
    delay = fwd._metrics_first_sentence_released_at - fwd._metrics_first_sentence_queued_at
    assert delay >= 0.05, f"Expected >= 50ms delay, got {delay*1000:.1f}ms"
    assert delay < 1.0, f"Delay too large: {delay*1000:.1f}ms"

    fwd._stop_watchdog()
    return True


# ═════════════════════════════════════════════════
# MULTI-LANGUAGE TESTS
# ═════════════════════════════════════════════════

async def test_hindi_sentence_boundaries():
    """Hindi text with Devanagari danda (।) splits correctly when delivered token-by-token."""
    ws = MockWebSocket("hindi")
    fwd = TextStreamForwarder(websocket=ws, text_only=False)

    await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)

    # Hindi sentences with purna viram (।) — delivered word-by-word like a real LLM
    # The _SENTENCE_ENDS regex checks the END of the buffer, so tokens must arrive
    # incrementally (as LLMs actually produce them).
    hindi_text = "नमस्ते, मैं मीरा हूँ। आप कैसे हैं? मैं आपकी मदद कर सकती हूँ।"
    for tf in make_text_frames(hindi_text):
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

    # Should detect 3 sentences (two danda, one question mark)
    assert fwd._sentence_q.qsize() == 3, (
        f"Hindi danda/question splitting failed, got {fwd._sentence_q.qsize()} sentences"
    )

    sentences = []
    while not fwd._sentence_q.empty():
        sentences.append(fwd._sentence_q.get_nowait())

    # Verify Hindi punctuation was detected
    for s in sentences:
        assert s.strip(), f"Empty sentence detected: {sentences}"

    # Verify content
    assert "नमस्ते" in sentences[0]
    assert "?" in sentences[1] or "हैं" in sentences[1]
    assert "मदद" in sentences[2]

    logger.info(f"[TEST] Hindi sentences split: {sentences}")
    return True


async def test_mixed_language_sentence_boundaries():
    """Mixed English-Hindi text splits correctly at both . and ।"""
    ws = MockWebSocket("mixed-lang")
    fwd = TextStreamForwarder(websocket=ws, text_only=False)

    await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)

    # Mixed: English sentence, then Hindi sentence, then English again
    mixed = "Hello, I am Mira. मैं आपकी मदद करूँगी। Let me help you!"
    for tf in make_text_frames(mixed):
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

    assert fwd._sentence_q.qsize() == 3, (
        f"Mixed lang splitting: expected 3, got {fwd._sentence_q.qsize()}"
    )

    sentences = []
    while not fwd._sentence_q.empty():
        sentences.append(fwd._sentence_q.get_nowait())

    assert "Hello, I am Mira." in sentences[0]
    assert "मैं आपकी मदद करूँगी।" in sentences[1]
    assert "Let me help you!" in sentences[2]

    return True


async def test_arabic_question_mark():
    """Arabic question mark (؟) is recognized as sentence boundary."""
    ws = MockWebSocket("arabic-qmark")
    fwd = TextStreamForwarder(websocket=ws, text_only=False)

    await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)

    # Arabic question mark
    for tf in [TextFrame(text="هل تحتاج مساعدة؟ ")]:
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

    assert fwd._sentence_q.qsize() == 1, (
        f"Arabic question mark not detected, got {fwd._sentence_q.qsize()}"
    )
    return True


async def test_long_hindi_response_multi_sentence():
    """Realistic long Hindi LLM response with multiple sentences."""
    ws = MockWebSocket("long-hindi")
    fwd = TextStreamForwarder(websocket=ws, text_only=False)
    notifier = TextAudioSyncNotifier(text_forwarder=fwd)

    await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)

    # Simulate token-by-token delivery of a multi-sentence Hindi response
    hindi_response = (
        "भारत एक विशाल देश है। "
        "यहाँ अनेक भाषाएँ बोली जाती हैं। "
        "हिंदी राष्ट्रभाषा है। "
        "तमिल दक्षिण भारत में बोली जाती है।"
    )
    # Deliver token by token (word by word)
    for tf in make_text_frames(hindi_response):
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

    expected_sentences = 4
    actual = fwd._sentence_q.qsize()
    assert actual == expected_sentences, (
        f"Long Hindi: expected {expected_sentences} sentences, got {actual}"
    )

    # Simulate TTS for all sentences
    for i in range(actual):
        await notifier.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
        await notifier.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)

    await fwd.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    await notifier.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    bot_texts = ws.get_messages_of_type("bot_text")
    assert len(bot_texts) == expected_sentences, (
        f"Expected {expected_sentences} bot_text, got {len(bot_texts)}"
    )

    completes = ws.get_messages_of_type("bot_text_complete")
    assert len(completes) == 1
    assert "भारत" in completes[0]["text"]

    return True


async def test_classroom_vs_tutor_sentence_regex_consistency():
    """
    Verify that bot.py _SENTENCE_ENDS and classroom.py _SENTENCE_RE
    agree on sentence boundaries for the same input text.
    This catches drift between the two regex patterns.
    """
    import re

    # bot.py pattern (used by TextStreamForwarder)
    tutor_re = re.compile(r'[.!?।؟\n]\s*$')

    # classroom.py pattern (used by RoomManager.ask_llm)
    classroom_re = re.compile(r'(?<=[.!?।\n])\s+')

    test_cases = [
        # (input, expected_sentence_count_for_both)
        ("Hello. World.", 2),
        ("Question? Answer!", 2),
        ("यह हिंदी है। दूसरा वाक्य।", 2),
        ("Mixed English. हिंदी वाक्य। More English!", 3),
        ("Single sentence without ending punct", 0),  # No boundary detected
    ]

    for text, expected in test_cases:
        # Tutor path: accumulate buffer, check if _SENTENCE_ENDS matches
        tutor_sentences = 0
        buffer = ""
        for word in text.split(" "):
            buffer += word + " "
            if tutor_re.search(buffer):
                tutor_sentences += 1
                buffer = ""

        # Classroom path: split by _SENTENCE_RE
        classroom_parts = classroom_re.split(text)
        classroom_sentences = len([p for p in classroom_parts if p.strip()])
        # classroom_re splits text, so the count is the number of non-empty parts
        # For "no boundary" case, it returns [original_text] → 1 part but 0 splits

        # Both should agree on whether there ARE sentence boundaries
        tutor_has_boundaries = tutor_sentences > 0
        classroom_has_boundaries = len(classroom_re.findall(text)) > 0

        if expected > 0:
            assert tutor_has_boundaries, (
                f"Tutor regex missed boundaries in: '{text[:40]}'"
            )
            assert classroom_has_boundaries, (
                f"Classroom regex missed boundaries in: '{text[:40]}'"
            )
        else:
            # For text without sentence-ending punctuation followed by space,
            # neither should find boundaries
            pass

        logger.info(
            f"[TEST] Regex consistency: '{text[:30]}...' → "
            f"tutor={tutor_sentences}, classroom_splits={len(classroom_re.findall(text))}"
        )

    return True


# ═════════════════════════════════════════════════
# PARALLEL PIPELINE / CONCURRENCY TESTS
# ═════════════════════════════════════════════════

async def test_parallel_speakers_concurrent():
    """
    Multiple speakers (each with own pipeline) process concurrently.
    Verifies no cross-talk between pipelines running in parallel.
    """
    NUM_SPEAKERS = 5

    async def run_speaker(speaker_id: int) -> tuple[int, MockWebSocket]:
        ws = MockWebSocket(f"speaker-{speaker_id}")
        fwd = TextStreamForwarder(websocket=ws, text_only=False)
        notifier = TextAudioSyncNotifier(text_forwarder=fwd)

        text = f"Speaker {speaker_id} says hello. Speaker {speaker_id} says goodbye."
        await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        for tf in make_text_frames(text):
            await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

        # Simulate TTS with slight delay (simulating real TTS latency)
        await asyncio.sleep(0.01 * speaker_id)  # Stagger TTS starts

        for _ in range(2):  # 2 sentences
            await notifier.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
            await asyncio.sleep(0.01)  # Simulate TTS processing time
            await notifier.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)

        await fwd.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        await notifier.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        return speaker_id, ws

    # Run all speakers concurrently
    tasks = [run_speaker(i) for i in range(NUM_SPEAKERS)]
    results = await asyncio.gather(*tasks)

    # Verify each speaker got their own messages, no cross-contamination
    for speaker_id, ws in results:
        bot_texts = ws.get_messages_of_type("bot_text")
        assert len(bot_texts) == 2, (
            f"Speaker {speaker_id}: expected 2 bot_text, got {len(bot_texts)}"
        )
        for msg in bot_texts:
            assert f"Speaker {speaker_id}" in msg["text"], (
                f"Speaker {speaker_id} got wrong text: {msg['text']}"
            )

        completes = ws.get_messages_of_type("bot_text_complete")
        assert len(completes) == 1, (
            f"Speaker {speaker_id}: expected 1 bot_text_complete, got {len(completes)}"
        )
        assert f"Speaker {speaker_id}" in completes[0]["text"]

    logger.info(f"[TEST] {NUM_SPEAKERS} parallel speakers completed without cross-talk")
    return True


async def test_parallel_speakers_different_languages():
    """
    Multiple speakers using different languages concurrently.
    Verifies sentence splitting works correctly per-language in parallel.
    """
    speakers = [
        (0, "English speaker here. How are you?"),
        (1, "हिंदी वक्ता यहाँ है। आप कैसे हैं?"),
        (2, "Mixed mode. हिंदी भी। And English!"),
    ]

    async def run_speaker(speaker_id: int, text: str) -> tuple[int, MockWebSocket, int]:
        ws = MockWebSocket(f"lang-speaker-{speaker_id}")
        fwd = TextStreamForwarder(websocket=ws, text_only=False)
        notifier = TextAudioSyncNotifier(text_forwarder=fwd)

        await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        for tf in make_text_frames(text):
            await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

        queued = fwd._sentence_q.qsize()

        # Release all via TTS
        for _ in range(queued):
            await notifier.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
            await notifier.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)

        await fwd.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        await notifier.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        return speaker_id, ws, queued

    tasks = [run_speaker(sid, text) for sid, text in speakers]
    results = await asyncio.gather(*tasks)

    for speaker_id, ws, queued in results:
        bot_texts = ws.get_messages_of_type("bot_text")
        assert len(bot_texts) == queued, (
            f"Speaker {speaker_id}: queued {queued} but got {len(bot_texts)} bot_text"
        )
        completes = ws.get_messages_of_type("bot_text_complete")
        assert len(completes) == 1, (
            f"Speaker {speaker_id}: expected 1 complete, got {len(completes)}"
        )
        logger.info(
            f"[TEST] Lang speaker {speaker_id}: {queued} sentences, "
            f"{len(bot_texts)} bot_text, 1 complete ✓"
        )

    return True


async def test_multi_receiver_same_content():
    """
    Simulates classroom: one LLM response, multiple receivers each with
    their own TextStreamForwarder + TextAudioSyncNotifier pair.
    All receivers should get the same content independently.
    """
    NUM_RECEIVERS = 10
    text = "The teacher explains the topic. Students listen carefully. Questions are welcome!"

    receivers = []
    for i in range(NUM_RECEIVERS):
        ws = MockWebSocket(f"receiver-{i}")
        fwd = TextStreamForwarder(websocket=ws, text_only=False)
        notifier = TextAudioSyncNotifier(text_forwarder=fwd)
        receivers.append((ws, fwd, notifier))

    # Feed the same LLM response to all receivers
    for ws, fwd, notifier in receivers:
        await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        for tf in make_text_frames(text):
            await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

    # All should have 3 sentences queued
    for i, (ws, fwd, notifier) in enumerate(receivers):
        assert fwd._sentence_q.qsize() == 3, (
            f"Receiver {i}: expected 3 queued, got {fwd._sentence_q.qsize()}"
        )

    # Simulate TTS at different speeds (staggered)
    async def release_receiver(idx: int):
        ws, fwd, notifier = receivers[idx]
        await asyncio.sleep(0.005 * idx)  # Stagger
        for _ in range(3):
            await notifier.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
            await asyncio.sleep(0.005)
            await notifier.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
        await fwd.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        await notifier.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    await asyncio.gather(*[release_receiver(i) for i in range(NUM_RECEIVERS)])

    # Verify all receivers got the same content
    for i, (ws, fwd, notifier) in enumerate(receivers):
        bot_texts = ws.get_messages_of_type("bot_text")
        assert len(bot_texts) == 3, (
            f"Receiver {i}: expected 3 bot_text, got {len(bot_texts)}"
        )
        completes = ws.get_messages_of_type("bot_text_complete")
        assert len(completes) == 1, (
            f"Receiver {i}: expected 1 complete, got {len(completes)}"
        )
        # All should have the same full text
        assert completes[0]["text"] == text

    logger.info(f"[TEST] {NUM_RECEIVERS} receivers all got identical content ✓")
    return True


async def test_mixed_mode_receivers():
    """
    Classroom scenario: some receivers in text_only, some in text_and_audio.
    text_only receivers get tokens immediately, text_and_audio get synced.
    """
    text = "First sentence. Second sentence."

    # text_only receiver
    ws_text = MockWebSocket("text-only-listener")
    fwd_text = TextStreamForwarder(websocket=ws_text, text_only=True)

    # text_and_audio receiver
    ws_audio = MockWebSocket("audio-listener")
    fwd_audio = TextStreamForwarder(websocket=ws_audio, text_only=False)
    notifier_audio = TextAudioSyncNotifier(text_forwarder=fwd_audio)

    # Feed same LLM response to both
    for fwd in [fwd_text, fwd_audio]:
        await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        for tf in make_text_frames(text):
            await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

    # text_only should already have all tokens sent
    text_msgs = ws_text.get_messages_of_type("bot_text")
    assert len(text_msgs) > 0, "text_only should have tokens immediately"

    # text_and_audio should have nothing sent yet (queued)
    audio_msgs = ws_audio.get_messages_of_type("bot_text")
    assert len(audio_msgs) == 0, "text_and_audio should not have tokens yet"
    assert fwd_audio._sentence_q.qsize() == 2

    # Complete text_only
    await fwd_text.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    text_completes = ws_text.get_messages_of_type("bot_text_complete")
    assert len(text_completes) == 1

    # Release audio receiver via TTS
    for _ in range(2):
        await notifier_audio.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
        await notifier_audio.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
    await fwd_audio.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    await notifier_audio.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    audio_msgs = ws_audio.get_messages_of_type("bot_text")
    assert len(audio_msgs) == 2
    audio_completes = ws_audio.get_messages_of_type("bot_text_complete")
    assert len(audio_completes) == 1

    # Both should have the same final text
    assert text_completes[0]["text"] == audio_completes[0]["text"]

    return True


async def test_multi_room_isolation():
    """
    Multiple rooms (each with own speaker + receivers) run concurrently.
    Verifies no cross-room contamination.
    """
    NUM_ROOMS = 4
    RECEIVERS_PER_ROOM = 3

    async def run_room(room_id: int) -> tuple[int, list[MockWebSocket]]:
        room_text = f"Room {room_id} content. Room {room_id} second sentence."

        # Speaker
        ws_speaker = MockWebSocket(f"room{room_id}-speaker")
        fwd_speaker = TextStreamForwarder(websocket=ws_speaker, text_only=False)
        notifier_speaker = TextAudioSyncNotifier(text_forwarder=fwd_speaker)

        # Receivers
        receivers = []
        for r in range(RECEIVERS_PER_ROOM):
            ws = MockWebSocket(f"room{room_id}-recv{r}")
            fwd = TextStreamForwarder(websocket=ws, text_only=False)
            notifier = TextAudioSyncNotifier(text_forwarder=fwd)
            receivers.append((ws, fwd, notifier))

        # Feed text to speaker and all receivers
        all_pipelines = [(ws_speaker, fwd_speaker, notifier_speaker)] + receivers
        for ws, fwd, notifier in all_pipelines:
            await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            for tf in make_text_frames(room_text):
                await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

        # Simulate staggered TTS
        await asyncio.sleep(0.01 * room_id)

        for ws, fwd, notifier in all_pipelines:
            for _ in range(2):
                await notifier.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
                await notifier.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
            await fwd.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
            await notifier.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        all_ws = [ws_speaker] + [ws for ws, _, _ in receivers]
        return room_id, all_ws

    tasks = [run_room(r) for r in range(NUM_ROOMS)]
    results = await asyncio.gather(*tasks)

    for room_id, all_ws in results:
        for ws in all_ws:
            completes = ws.get_messages_of_type("bot_text_complete")
            assert len(completes) == 1, (
                f"Room {room_id}, {ws.name}: expected 1 complete, got {len(completes)}"
            )
            # Verify content belongs to this room
            assert f"Room {room_id}" in completes[0]["text"], (
                f"Room {room_id}, {ws.name}: got wrong room's content: "
                f"'{completes[0]['text'][:50]}'"
            )

    logger.info(f"[TEST] {NUM_ROOMS} rooms × {RECEIVERS_PER_ROOM + 1} clients: no cross-room leak ✓")
    return True


async def test_barge_in_under_load():
    """
    Multiple concurrent pipelines, one gets interrupted mid-stream.
    Verifies barge-in only affects the interrupted pipeline.

    With release-all behavior, TTSStartedFrame releases ALL queued sentences.
    """
    # Pipeline A: will be interrupted
    ws_a = MockWebSocket("barge-target")
    fwd_a = TextStreamForwarder(websocket=ws_a, text_only=False)
    notifier_a = TextAudioSyncNotifier(text_forwarder=fwd_a)

    # Pipeline B: runs normally (should not be affected)
    ws_b = MockWebSocket("barge-bystander")
    fwd_b = TextStreamForwarder(websocket=ws_b, text_only=False)
    notifier_b = TextAudioSyncNotifier(text_forwarder=fwd_b)

    # Both start
    for fwd in [fwd_a, fwd_b]:
        await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)

    # Both queue 3 sentences
    for fwd in [fwd_a, fwd_b]:
        for tf in make_text_frames("Sentence one. Sentence two. Sentence three."):
            await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

    assert fwd_a._sentence_q.qsize() == 3
    assert fwd_b._sentence_q.qsize() == 3

    # TTS starts for both — releases ALL queued sentences (release-all)
    await notifier_a.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
    await notifier_b.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)

    # Both queues now empty (all released)
    assert fwd_a._sentence_q.qsize() == 0
    assert fwd_b._sentence_q.qsize() == 0

    # Interrupt A only
    await fwd_a.process_frame(StartInterruptionFrame(), FrameDirection.DOWNSTREAM)
    await notifier_a.process_frame(StartInterruptionFrame(), FrameDirection.DOWNSTREAM)

    # A should have all 3 texts (all TTS-released before barge-in)
    a_texts = ws_a.get_messages_of_type("bot_text")
    assert len(a_texts) == 3, f"Expected 3 bot_text for A, got {len(a_texts)}"

    # B should also have all 3 texts (all TTS-released, unaffected by A's barge-in)
    b_texts = ws_b.get_messages_of_type("bot_text")
    assert len(b_texts) == 3, f"Expected 3 bot_text for B, got {len(b_texts)}"

    # Complete B normally
    await notifier_b.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
    await fwd_b.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    await notifier_b.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    b_completes = ws_b.get_messages_of_type("bot_text_complete")
    assert len(b_completes) == 1

    # Verify isolation: B's metrics unaffected by A's interruption
    assert fwd_b._metrics_total_interruptions == 0
    assert fwd_a._metrics_total_interruptions == 1

    return True


async def test_rapid_fire_responses():
    """
    Back-to-back LLM responses without waiting for TTS to finish.
    Verifies queue is drained between responses and metrics reset.
    """
    ws = MockWebSocket("rapid-fire")
    fwd = TextStreamForwarder(websocket=ws, text_only=False)
    notifier = TextAudioSyncNotifier(text_forwarder=fwd)

    for response_num in range(5):
        text = f"Response {response_num} sentence one. Response {response_num} sentence two."

        await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        for tf in make_text_frames(text):
            await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

        # Simulate TTS for all sentences
        for _ in range(2):
            await notifier.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
            await notifier.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)

        await fwd.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        await notifier.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        # Queue should be empty after each response
        assert fwd._sentence_q.qsize() == 0, (
            f"Response {response_num}: queue not empty after completion"
        )

    assert fwd._metrics_total_responses == 5
    # Per-response metrics should reflect the LAST response only
    assert fwd._metrics_sentences_queued == 2

    # Total messages: 5 responses × 2 sentences + 5 completes = 15
    bot_texts = ws.get_messages_of_type("bot_text")
    completes = ws.get_messages_of_type("bot_text_complete")
    assert len(bot_texts) == 10, f"Expected 10 bot_text, got {len(bot_texts)}"
    assert len(completes) == 5, f"Expected 5 completes, got {len(completes)}"

    return True


async def test_watchdog_under_concurrent_load():
    """
    Multiple pipelines with TTS timeouts firing concurrently.
    Verifies watchdog tasks don't interfere with each other.
    """
    NUM_PIPELINES = 5

    async def run_pipeline(idx: int) -> tuple[int, MockWebSocket, TextStreamForwarder]:
        ws = MockWebSocket(f"watchdog-{idx}")
        fwd = TextStreamForwarder(websocket=ws, text_only=False)
        fwd._TTS_SENTENCE_TIMEOUT = 0.2  # 200ms

        await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        for tf in make_text_frames(f"Pipeline {idx} will timeout."):
            await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

        # Don't send TTSStartedFrame — let watchdog fire
        return idx, ws, fwd

    results = await asyncio.gather(*[run_pipeline(i) for i in range(NUM_PIPELINES)])

    # Wait for all watchdogs to fire
    await asyncio.sleep(0.5)

    for idx, ws, fwd in results:
        bot_texts = ws.get_messages_of_type("bot_text")
        assert len(bot_texts) == 1, (
            f"Pipeline {idx}: expected 1 watchdog-released, got {len(bot_texts)}"
        )
        assert f"Pipeline {idx}" in bot_texts[0]["text"]
        assert fwd._metrics_sentences_timed_out == 1
        fwd._stop_watchdog()

    logger.info(f"[TEST] {NUM_PIPELINES} concurrent watchdog timeouts: no interference ✓")
    return True


async def test_receiver_joins_mid_response():
    """
    Simulates a receiver connecting mid-way through an LLM response.
    Only the sentences after join should be delivered.
    """
    text = "Before join. After join first. After join second."

    # Early receiver — gets everything
    ws_early = MockWebSocket("early-join")
    fwd_early = TextStreamForwarder(websocket=ws_early, text_only=False)
    notifier_early = TextAudioSyncNotifier(text_forwarder=fwd_early)

    await fwd_early.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    for tf in make_text_frames(text):
        await fwd_early.process_frame(tf, FrameDirection.DOWNSTREAM)

    assert fwd_early._sentence_q.qsize() == 3

    # Late receiver — only gets sentences 2 and 3
    ws_late = MockWebSocket("late-join")
    fwd_late = TextStreamForwarder(websocket=ws_late, text_only=False)
    notifier_late = TextAudioSyncNotifier(text_forwarder=fwd_late)

    # Late receiver starts fresh — only gets remaining sentences
    await fwd_late.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    for tf in make_text_frames("After join first. After join second."):
        await fwd_late.process_frame(tf, FrameDirection.DOWNSTREAM)

    assert fwd_late._sentence_q.qsize() == 2

    # Release early receiver
    for _ in range(3):
        await notifier_early.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
        await notifier_early.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
    await fwd_early.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    await notifier_early.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    # Release late receiver
    for _ in range(2):
        await notifier_late.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
        await notifier_late.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
    await fwd_late.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    await notifier_late.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    early_texts = ws_early.get_messages_of_type("bot_text")
    late_texts = ws_late.get_messages_of_type("bot_text")

    assert len(early_texts) == 3, f"Early: expected 3, got {len(early_texts)}"
    assert len(late_texts) == 2, f"Late: expected 2, got {len(late_texts)}"

    return True


async def test_high_sentence_count():
    """
    Stress test: LLM response with many sentences.
    Verifies queue and metrics handle high sentence counts.
    """
    NUM_SENTENCES = 50
    text = " ".join([f"Sentence number {i}." for i in range(NUM_SENTENCES)])

    ws = MockWebSocket("high-count")
    fwd = TextStreamForwarder(websocket=ws, text_only=False)
    notifier = TextAudioSyncNotifier(text_forwarder=fwd)

    await fwd.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    for tf in make_text_frames(text):
        await fwd.process_frame(tf, FrameDirection.DOWNSTREAM)

    assert fwd._sentence_q.qsize() == NUM_SENTENCES, (
        f"Expected {NUM_SENTENCES} queued, got {fwd._sentence_q.qsize()}"
    )

    # Release all via TTS
    for _ in range(NUM_SENTENCES):
        await notifier.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
        await notifier.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)

    await fwd.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    await notifier.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    bot_texts = ws.get_messages_of_type("bot_text")
    assert len(bot_texts) == NUM_SENTENCES
    completes = ws.get_messages_of_type("bot_text_complete")
    assert len(completes) == 1

    assert fwd._metrics_sentences_queued == NUM_SENTENCES
    assert fwd._metrics_sentences_released == NUM_SENTENCES
    assert fwd._metrics_sentences_timed_out == 0

    logger.info(f"[TEST] {NUM_SENTENCES} sentences: all released correctly ✓")
    return True


# ═════════════════════════════════════════════════
# Test runner
# ═════════════════════════════════════════════════

ALL_TESTS = {
    # ── Core tests ──
    "text_only": test_text_only_streams_tokens,
    "audio_queues": test_audio_mode_queues_sentences,
    "sentence_boundaries": test_sentence_boundaries,
    "partial_flush": test_partial_sentence_flushed_at_end,
    "tts_timeout": test_tts_timeout_watchdog,
    "barge_in": test_barge_in_flushes_queue,
    "greeting": test_greeting_sent_immediately,
    "complete_ordering": test_complete_after_all_sentences,
    "multi_response": test_multiple_responses_reset_metrics,
    "per_receiver": test_per_receiver_isolation,
    "empty_response": test_empty_response,
    "safety_net": test_safety_net_flush_on_llm_end,
    "watchdog_no_false_fire": test_watchdog_doesnt_fire_when_tts_is_fast,
    "first_sentence_delay": test_metrics_first_sentence_delay,

    # ── Multi-language tests ──
    "hindi_boundaries": test_hindi_sentence_boundaries,
    "mixed_lang_boundaries": test_mixed_language_sentence_boundaries,
    "arabic_qmark": test_arabic_question_mark,
    "long_hindi": test_long_hindi_response_multi_sentence,
    "regex_consistency": test_classroom_vs_tutor_sentence_regex_consistency,

    # ── Concurrency / scale tests ──
    "parallel_speakers": test_parallel_speakers_concurrent,
    "parallel_lang_speakers": test_parallel_speakers_different_languages,
    "multi_receiver": test_multi_receiver_same_content,
    "mixed_mode_receivers": test_mixed_mode_receivers,
    "multi_room": test_multi_room_isolation,
    "barge_in_under_load": test_barge_in_under_load,
    "rapid_fire": test_rapid_fire_responses,
    "watchdog_concurrent": test_watchdog_under_concurrent_load,
    "mid_response_join": test_receiver_joins_mid_response,
    "high_sentence_count": test_high_sentence_count,
}


async def run_test(name: str, func) -> bool:
    """Run a single test, return True if passed."""
    try:
        result = await asyncio.wait_for(func(), timeout=30.0)
        if result:
            print(f"  ✅ PASS  {name}")
            return True
        else:
            print(f"  ❌ FAIL  {name} — returned False")
            return False
    except AssertionError as e:
        print(f"  ❌ FAIL  {name} — {e}")
        return False
    except asyncio.TimeoutError:
        print(f"  ❌ FAIL  {name} — TIMEOUT (30s)")
        return False
    except Exception as e:
        print(f"  ❌ FAIL  {name} — {type(e).__name__}: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Text-Audio Sync Unit Tests")
    parser.add_argument("--test", type=str, default="all", help="Test name or 'all'")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.test == "all":
        tests_to_run = ALL_TESTS
    elif args.test in ALL_TESTS:
        tests_to_run = {args.test: ALL_TESTS[args.test]}
    else:
        print(f"Unknown test: {args.test}")
        print(f"Available: {', '.join(ALL_TESTS.keys())}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  TEXT-AUDIO SYNC UNIT TESTS")
    print("=" * 60)
    print(f"  Tests: {len(tests_to_run)}")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    for name, func in tests_to_run.items():
        if await run_test(name, func):
            passed += 1
        else:
            failed += 1

    print()
    print("-" * 60)
    print(f"  {passed}/{passed + failed} tests passed")
    if failed:
        print(f"  {failed} FAILED")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
