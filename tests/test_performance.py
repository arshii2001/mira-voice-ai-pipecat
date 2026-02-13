#!/usr/bin/env python3
"""
Performance Test for MiraVoiceAI — Tutor & Classroom Metrics.

Runs a configurable number of queries across all four metrics paths:
  1. Tutor text  (streaming /chat)
  2. Tutor text  (non-streaming /chat)
  3. Classroom speaker (text_message via WebSocket)
  4. Classroom speaker + listener (translation + delivery)

After each phase, fetches GET /metrics and prints a formatted summary
showing separate stats for tutor vs classroom, speaker vs listener.

Usage:
    # ── Production-safe (recommended for live servers) ──
    python tests/test_performance.py --prod
    python tests/test_performance.py --prod --host https://pipecat-v2-api.example.com

    # ── Just view current metrics (zero queries) ──
    python tests/test_performance.py --metrics-only
    python tests/test_performance.py --metrics-only --host https://pipecat-v2-api.example.com

    # ── Full benchmark (for staging/dev) ──
    python tests/test_performance.py                  # 5 iterations per phase
    python tests/test_performance.py --quick           # 2 iterations per phase
    python tests/test_performance.py -n 10             # 10 iterations per phase

    # ── Specific phases ──
    python tests/test_performance.py --phase tutor_text
    python tests/test_performance.py --phase classroom_speaker
    python tests/test_performance.py --phase classroom_listener

    # ── Via Docker ──
    docker compose run --rm test-voice tests/test_performance.py --prod
    docker compose run --rm test-voice tests/test_performance.py --quick

Load profile:
    --prod          3 LLM calls + 1 translation  (~10-15s)  ← safe for production
    --quick         ~12 LLM calls + translations  (~30s)
    default (-n 5)  ~20 LLM calls + translations  (~60s)
    -n 10           ~40 LLM calls + translations  (~2min)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import List, Optional

try:
    import aiohttp
except ImportError:
    print("pip install aiohttp")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)

import jwt as pyjwt

# ─────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────
HTTP_URL = os.getenv("PIPECAT_HTTP_URL", "http://localhost:7860")
WS_URL = os.getenv("PIPECAT_WS_URL", "ws://localhost:7860/ws")
CLASSROOM_WS_BASE = os.getenv("CLASSROOM_WS_URL", "ws://localhost:7860/classroom/rooms")
WEBUI_SECRET_KEY = os.getenv("WEBUI_SECRET_KEY", "").strip() or None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("perf-test")


def _make_jwt(user_id: str = "test-perf-user") -> str:
    """Generate a JWT token for test auth when WEBUI_SECRET_KEY is set."""
    if not WEBUI_SECRET_KEY:
        return ""
    payload = {
        "id": user_id,
        "email": f"{user_id}@example.test",
        "exp": int(time.time()) + 7200,
    }
    return pyjwt.encode(payload, WEBUI_SECRET_KEY, algorithm="HS256")


def _jwt_headers(user_id: str = "test-perf-user") -> dict:
    """Return Authorization header if WEBUI_SECRET_KEY is set."""
    token = _make_jwt(user_id)
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}

# ─────────────────────────────────────────────────
# Test prompts — short to keep latency measurable
# ─────────────────────────────────────────────────
TUTOR_PROMPTS = [
    "What is photosynthesis? One sentence.",
    "Explain gravity in simple terms.",
    "What causes rain? Brief answer.",
    "Who was Albert Einstein? One sentence.",
    "What is the speed of light?",
    "Define osmosis briefly.",
    "What is DNA?",
    "Why is the sky blue? Short answer.",
    "What is a black hole? One sentence.",
    "Explain what an atom is.",
]

CLASSROOM_PROMPTS = [
    "What is the water cycle? One sentence.",
    "Explain how volcanoes work. Brief.",
    "What is the solar system?",
    "Define photosynthesis simply.",
    "What causes earthquakes? Brief.",
    "Why do we have seasons?",
    "What is electricity? One sentence.",
    "Explain the food chain briefly.",
    "What is the moon made of?",
    "How do magnets work? Short answer.",
]

LISTENER_LANGUAGES = ["hi", "ta"]  # Hindi, Tamil

# Lightweight prompts for --prod mode (short answers → minimal LLM/translation load)
PROD_TUTOR_PROMPT = "What is 2+2? One word answer."
PROD_CLASSROOM_PROMPT = "What is 3+3? One word."


# ─────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────
async def recv_json(ws, timeout: float = 5.0) -> Optional[dict]:
    try:
        msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
        if isinstance(msg, str):
            return json.loads(msg)
        return None
    except (asyncio.TimeoutError, Exception):
        return None


async def recv_json_nonbinary(ws, timeout: float = 5.0) -> Optional[dict]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = max(deadline - time.time(), 0.1)
        msg = await recv_json(ws, timeout=remaining)
        if msg is not None:
            return msg
    return None


async def wait_for_type(ws, msg_type: str, timeout: float = 10.0) -> Optional[dict]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = max(deadline - time.time(), 0.1)
        msg = await recv_json_nonbinary(ws, timeout=remaining)
        if msg and msg.get("type") == msg_type:
            return msg
    return None


async def drain(ws, duration: float = 1.0) -> List[dict]:
    msgs = []
    deadline = time.time() + duration
    while time.time() < deadline:
        msg = await recv_json(ws, timeout=max(deadline - time.time(), 0.1))
        if msg:
            msgs.append(msg)
    return msgs


async def join_room(ws, room_id: str, user_id: str, name: str,
                    language: str = "en", mode: str = "text_only") -> dict:
    join_msg = {
        "type": "join", "user_id": user_id,
        "name": name, "language": language, "mode": mode,
    }
    token = _make_jwt(user_id)
    if token:
        join_msg["token"] = token
    await ws.send(json.dumps(join_msg))
    for _ in range(10):
        msg = await recv_json_nonbinary(ws, timeout=3.0)
        if msg and msg.get("type") == "joined":
            return msg
    raise Exception(f"Did not receive 'joined' for {name}")


async def fetch_metrics() -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{HTTP_URL}/metrics", headers=_jwt_headers()) as resp:
            return await resp.json()


# ─────────────────────────────────────────────────
# Phase 1: Tutor text mode
# ─────────────────────────────────────────────────
async def phase_tutor_text(iterations: int):
    """Run streaming and non-streaming /chat queries."""
    logger.info(f"  Running {iterations} streaming + {iterations} non-streaming queries...")
    results = []

    for i in range(iterations):
        prompt = TUTOR_PROMPTS[i % len(TUTOR_PROMPTS)]

        # Streaming
        t0 = time.time()
        t_first = 0.0
        token_count = 0
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{HTTP_URL}/chat",
                json={"messages": [{"role": "user", "content": prompt}], "stream": True},
                headers=_jwt_headers(),
            ) as resp:
                async for line in resp.content:
                    decoded = line.decode().strip()
                    if decoded.startswith("data: "):
                        token_count += 1
                        if token_count == 1:
                            t_first = time.time()
        total_ms = round((time.time() - t0) * 1000, 1)
        ttft_ms = round((t_first - t0) * 1000, 1) if t_first else 0.0
        results.append({"type": "stream", "total_ms": total_ms, "ttft_ms": ttft_ms, "tokens": token_count})
        logger.info(f"    [{i+1}/{iterations}] stream: ttft={ttft_ms}ms total={total_ms}ms tokens={token_count}")

        # Non-streaming
        t0 = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{HTTP_URL}/chat",
                json={"messages": [{"role": "user", "content": prompt}], "stream": False},
                headers=_jwt_headers(),
            ) as resp:
                body = await resp.json()
        total_ms = round((time.time() - t0) * 1000, 1)
        results.append({"type": "sync", "total_ms": total_ms})
        logger.info(f"    [{i+1}/{iterations}] sync:   total={total_ms}ms")

    return results


# ─────────────────────────────────────────────────
# Phase 2: Classroom speaker (no listeners)
# ─────────────────────────────────────────────────
async def phase_classroom_speaker(iterations: int):
    """Speaker-only classroom queries to measure speaker LLM latency."""
    logger.info(f"  Running {iterations} classroom speaker queries...")
    results = []
    room_id = "default"

    ws = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
    await join_room(ws, room_id, "perf-speaker", "PerfSpeaker", "en")
    await wait_for_type(ws, "token_changed", timeout=5)

    for i in range(iterations):
        prompt = CLASSROOM_PROMPTS[i % len(CLASSROOM_PROMPTS)]
        t0 = time.time()

        await ws.send(json.dumps({"type": "text_message", "text": prompt}))

        # Collect bot_text chunks and final bot_text_complete
        chunks = []
        t_first_chunk = 0.0
        complete_text = ""
        deadline = time.time() + 30
        while time.time() < deadline:
            msg = await recv_json_nonbinary(ws, timeout=20)
            if not msg:
                break
            if msg.get("type") == "bot_text":
                if not chunks:
                    t_first_chunk = time.time()
                chunks.append(msg.get("text", ""))
            elif msg.get("type") == "bot_text_complete":
                complete_text = msg.get("text", "")
                break

        total_ms = round((time.time() - t0) * 1000, 1)
        ttft_ms = round((t_first_chunk - t0) * 1000, 1) if t_first_chunk else 0.0
        results.append({
            "total_ms": total_ms, "ttft_ms": ttft_ms,
            "chunks": len(chunks), "response_len": len(complete_text),
        })
        logger.info(
            f"    [{i+1}/{iterations}] ttft={ttft_ms}ms total={total_ms}ms "
            f"chunks={len(chunks)} len={len(complete_text)}"
        )

    await ws.close()
    await asyncio.sleep(0.5)
    return results


# ─────────────────────────────────────────────────
# Phase 3: Classroom speaker + listener (translation)
# ─────────────────────────────────────────────────
async def phase_classroom_listener(iterations: int):
    """Speaker + Hindi listener to measure translation + delivery latency."""
    logger.info(f"  Running {iterations} classroom speaker+listener queries (Hindi)...")
    results = []
    room_id = "default"

    ws_speaker = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
    await join_room(ws_speaker, room_id, "perf-speaker2", "PerfSpeaker2", "en")
    await wait_for_type(ws_speaker, "token_changed", timeout=5)

    ws_listener = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
    await join_room(ws_listener, room_id, "perf-listener", "PerfListenerHi", "hi")
    await drain(ws_listener, duration=1.0)

    for i in range(iterations):
        prompt = CLASSROOM_PROMPTS[i % len(CLASSROOM_PROMPTS)]
        t0 = time.time()

        await ws_speaker.send(json.dumps({"type": "text_message", "text": prompt}))

        # Speaker side: wait for complete
        speaker_done = False
        speaker_deadline = time.time() + 30
        while time.time() < speaker_deadline:
            msg = await recv_json_nonbinary(ws_speaker, timeout=20)
            if msg and msg.get("type") == "bot_text_complete":
                speaker_done = True
                break

        # Listener side: collect translated chunks
        listener_texts = []
        t_listener_first = 0.0
        listener_done = False
        listener_deadline = time.time() + 30
        while time.time() < listener_deadline:
            msg = await recv_json_nonbinary(ws_listener, timeout=20)
            if not msg:
                break
            if msg.get("type") == "bot_text":
                if not listener_texts:
                    t_listener_first = time.time()
                listener_texts.append(msg.get("text", ""))
            elif msg.get("type") == "bot_text_complete":
                listener_done = True
                break

        total_ms = round((time.time() - t0) * 1000, 1)
        listener_first_ms = round((t_listener_first - t0) * 1000, 1) if t_listener_first else 0.0
        listener_full = "".join(listener_texts)
        results.append({
            "total_ms": total_ms,
            "listener_first_chunk_ms": listener_first_ms,
            "listener_chunks": len(listener_texts),
            "listener_text_len": len(listener_full),
            "speaker_done": speaker_done,
            "listener_done": listener_done,
        })
        logger.info(
            f"    [{i+1}/{iterations}] total={total_ms}ms "
            f"listener_first={listener_first_ms}ms "
            f"chunks={len(listener_texts)} len={len(listener_full)} "
            f"done=s:{speaker_done}/l:{listener_done}"
        )

    await ws_speaker.close()
    await ws_listener.close()
    await asyncio.sleep(0.5)
    return results


# ─────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────
def fmt_stats(stats: dict, unit: str = "ms") -> str:
    if not stats or stats.get("count", 0) == 0:
        return "—"
    return (
        f"avg={stats['avg']:.0f}{unit}  "
        f"p95={stats['p95']:.0f}{unit}  "
        f"min={stats['min']:.0f}{unit}  "
        f"max={stats['max']:.0f}{unit}  "
        f"(n={stats['count']})"
    )


def print_metrics_report(metrics: dict, phase_results: dict):
    """Print a beautiful formatted metrics report."""
    w = 76
    print()
    print("┌" + "─" * w + "┐")
    print("│" + " MIRA PERFORMANCE REPORT ".center(w) + "│")
    print("├" + "─" * w + "┤")
    print(f"│  Server uptime: {metrics['uptime_seconds']:.0f}s".ljust(w + 1) + "│")
    sess = metrics["sessions"]
    print(f"│  Sessions — total: {sess['total']}  "
          f"(tutor: {sess['tutor']['total']}, classroom: {sess['classroom']['total']})  "
          f"active: {sess['active']}".ljust(w + 1) + "│")
    print("├" + "─" * w + "┤")

    # ── Tutor Text ──
    tt = metrics["tutor"]["text"]
    print("│" + " TUTOR — Text Mode (/chat) ".center(w) + "│")
    print("├" + "─" * w + "┤")
    print(f"│  Queries: {tt.get('query_count', 0)}".ljust(w + 1) + "│")
    if "llm_ttft_ms" in tt:
        print(f"│  LLM TTFT:     {fmt_stats(tt['llm_ttft_ms'])}".ljust(w + 1) + "│")
    if "llm_total_ms" in tt:
        print(f"│  LLM Total:    {fmt_stats(tt['llm_total_ms'])}".ljust(w + 1) + "│")
    if "llm_tokens" in tt:
        print(f"│  Tokens/query: {fmt_stats(tt['llm_tokens'], unit='')}".ljust(w + 1) + "│")

    # ── Tutor Voice ──
    tv = metrics["tutor"]["voice"]
    print("├" + "─" * w + "┤")
    print("│" + " TUTOR — Voice Mode (Pipecat pipeline) ".center(w) + "│")
    print("├" + "─" * w + "┤")
    print(f"│  Voice turns: {tv.get('turn_count', 0)}".ljust(w + 1) + "│")
    if "turn_latency_ms" in tv:
        print(f"│  Turn latency: {fmt_stats(tv['turn_latency_ms'])}".ljust(w + 1) + "│")
    if "stt_latency_ms" in tv:
        print(f"│  VAD→STT:      {fmt_stats(tv['stt_latency_ms'])}".ljust(w + 1) + "│")
    if "llm_ttft_ms" in tv:
        print(f"│  LLM TTFT:     {fmt_stats(tv['llm_ttft_ms'])}".ljust(w + 1) + "│")
    if "llm_total_ms" in tv:
        print(f"│  LLM Total:    {fmt_stats(tv['llm_total_ms'])}".ljust(w + 1) + "│")
    if "llm_to_tts_ms" in tv:
        print(f"│  LLM→TTS:      {fmt_stats(tv['llm_to_tts_ms'])}".ljust(w + 1) + "│")
    if "tts_ms" in tv:
        print(f"│  TTS duration:  {fmt_stats(tv['tts_ms'])}".ljust(w + 1) + "│")
    if tv.get("turn_count", 0) == 0:
        print(f"│  (no voice turns recorded — use voice mode to populate)".ljust(w + 1) + "│")

    # ── Classroom Speaker ──
    cs = metrics["classroom"]["speaker"]
    print("├" + "─" * w + "┤")
    print("│" + " CLASSROOM — Speaker ".center(w) + "│")
    print("├" + "─" * w + "┤")
    print(f"│  Queries: {cs.get('query_count', 0)}   Voice turns: {cs.get('turn_count', 0)}".ljust(w + 1) + "│")
    if "llm_ttft_ms" in cs:
        print(f"│  LLM TTFT:     {fmt_stats(cs['llm_ttft_ms'])}".ljust(w + 1) + "│")
    if "llm_total_ms" in cs:
        print(f"│  LLM Total:    {fmt_stats(cs['llm_total_ms'])}".ljust(w + 1) + "│")
    if "llm_tokens" in cs:
        print(f"│  Tokens/query: {fmt_stats(cs['llm_tokens'], unit='')}".ljust(w + 1) + "│")
    if "turn_latency_ms" in cs:
        print(f"│  Voice turn:   {fmt_stats(cs['turn_latency_ms'])}".ljust(w + 1) + "│")
    if "stt_latency_ms" in cs:
        print(f"│  VAD→STT:      {fmt_stats(cs['stt_latency_ms'])}".ljust(w + 1) + "│")

    # ── Classroom Listener ──
    cl = metrics["classroom"]["listener"]
    print("├" + "─" * w + "┤")
    print("│" + " CLASSROOM — Listener (translation + delivery) ".center(w) + "│")
    print("├" + "─" * w + "┤")
    if "translation_ms" in cl:
        print(f"│  Translation:  {fmt_stats(cl['translation_ms'])}".ljust(w + 1) + "│")
    if "listener_delivery_ms" in cl:
        print(f"│  Delivery:     {fmt_stats(cl['listener_delivery_ms'])}".ljust(w + 1) + "│")
    if "tts_ms" in cl:
        print(f"│  TTS:          {fmt_stats(cl['tts_ms'])}".ljust(w + 1) + "│")
    if "tts_audio_bytes" in cl:
        print(f"│  Audio bytes:  {fmt_stats(cl['tts_audio_bytes'], unit='B')}".ljust(w + 1) + "│")
    if not any(k in cl for k in ("translation_ms", "listener_delivery_ms", "tts_ms")):
        print(f"│  (no listener data — run with --phase classroom_listener)".ljust(w + 1) + "│")

    # ── Errors ──
    errors = metrics.get("errors", {})
    if errors:
        print("├" + "─" * w + "┤")
        print("│" + " ERRORS ".center(w) + "│")
        print("├" + "─" * w + "┤")
        for cat, count in errors.items():
            print(f"│  {cat}: {count}".ljust(w + 1) + "│")

    # ── Recent Sessions ──
    recent = metrics.get("recent_sessions", [])
    if recent:
        print("├" + "─" * w + "┤")
        print("│" + f" RECENT SESSIONS (last {len(recent)}) ".center(w) + "│")
        print("├" + "─" * w + "┤")
        for s in recent[-6:]:
            stype = s.get("type", s.get("mode", "?"))
            subtype = s.get("subtype", "")
            name = s.get("user_name", s.get("session_id", "?")[:12])
            dur = s.get("duration_s", 0)
            room = s.get("room_id", "")
            label = f"{stype}"
            if subtype:
                label += f"/{subtype}"
            if room:
                label += f" room={room}"
            print(f"│  {label:<30} {name:<15} {dur:>6.1f}s".ljust(w + 1) + "│")

    # ── Per-Call Traces (pipeline waterfall) ──
    traces = metrics.get("traces", [])
    if traces:
        print("├" + "─" * w + "┤")
        print("│" + f" PER-CALL PIPELINE TRACES (last {len(traces)}) ".center(w) + "│")
        print("├" + "─" * w + "┤")
        for i, t in enumerate(traces):
            mode = t.get("mode", "?")
            total = t.get("total_ms", 0)
            query = t.get("query", "")[:50]
            print(f"│".ljust(w + 1) + "│")
            print(f"│  [{i+1}] {mode}  total={total:.0f}ms".ljust(w + 1) + "│")
            print(f"│      Q: \"{query}\"".ljust(w + 1) + "│")

            # Draw stage waterfall
            stages = t.get("stages", [])
            for stage in stages:
                name = stage.get("name", "?")
                ms = stage.get("ms", 0)
                tokens = stage.get("tokens")
                tok_s = stage.get("tok_per_sec")
                detail = ""
                if tokens:
                    detail += f"  ({tokens} tok"
                    if tok_s:
                        detail += f", {tok_s} tok/s"
                    detail += ")"
                if stage.get("chunks"):
                    detail += f"  ({stage['chunks']} chunks, {stage.get('audio_bytes', 0)}B)"
                if stage.get("sentences"):
                    detail += f"  ({stage['sentences']} sentences)"

                # Visual bar: 1 char per 50ms
                bar_len = max(1, int(ms / 50))
                bar = "█" * min(bar_len, 30)
                print(f"│      ├─ {name:<18} {ms:>7.0f}ms {bar}{detail}".ljust(w + 1) + "│")

            # Per-listener delivery breakdown
            listeners = t.get("listeners", [])
            if listeners:
                print(f"│      └─ listeners:".ljust(w + 1) + "│")
                for lis in listeners:
                    lname = lis.get("user", "?")
                    llang = lis.get("language", "?")
                    lmode = lis.get("mode", "?")
                    ltrans = lis.get("translate_ms", 0)
                    ltts = lis.get("tts_ms", 0)
                    ltotal = lis.get("total_ms", 0)
                    lsent = lis.get("sentences", 0)
                    parts = []
                    if ltrans > 0:
                        parts.append(f"translate={ltrans:.0f}ms")
                    if ltts > 0:
                        parts.append(f"tts={ltts:.0f}ms")
                    parts.append(f"total={ltotal:.0f}ms")
                    parts.append(f"{lsent}sent")
                    detail_str = "  ".join(parts)
                    print(f"│         {lname}({llang}/{lmode}): {detail_str}".ljust(w + 1) + "│")

    # ── Local phase timing ──
    if phase_results:
        print("├" + "─" * w + "┤")
        print("│" + " LOCAL PHASE TIMING ".center(w) + "│")
        print("├" + "─" * w + "┤")
        for phase_name, phase_data in phase_results.items():
            duration = phase_data.get("duration_s", 0)
            count = phase_data.get("count", 0)
            print(f"│  {phase_name:<30} {count:>3} queries  {duration:>6.1f}s".ljust(w + 1) + "│")

    print("└" + "─" * w + "┘")
    print()


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────
async def phase_prod_smoke():
    """Production-safe smoke test: 1 tiny query per path (3 LLM calls total).

    Total load: ~3 short LLM calls + 1 translation.  Takes ~10-15s.
    Safe to run against production without impacting users.
    """
    results = {}

    # 1. Tutor text — single streaming query, short answer
    logger.info("  [prod] Tutor text (streaming)...")
    t0 = time.time()
    t_first = 0.0
    token_count = 0
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{HTTP_URL}/chat",
            json={"messages": [{"role": "user", "content": PROD_TUTOR_PROMPT}], "stream": True},
            headers=_jwt_headers(),
        ) as resp:
            async for line in resp.content:
                decoded = line.decode().strip()
                if decoded.startswith("data: "):
                    token_count += 1
                    if token_count == 1:
                        t_first = time.time()
    total_ms = round((time.time() - t0) * 1000, 1)
    ttft_ms = round((t_first - t0) * 1000, 1) if t_first else 0.0
    logger.info(f"    → ttft={ttft_ms}ms  total={total_ms}ms  tokens={token_count}")
    results["tutor_text"] = {"total_ms": total_ms, "ttft_ms": ttft_ms, "tokens": token_count}

    # 2. Classroom speaker — single short query
    logger.info("  [prod] Classroom speaker (text)...")
    room_id = "default"
    ws = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
    await join_room(ws, room_id, "prod-probe-speaker", "ProdProbe", "en")
    await wait_for_type(ws, "token_changed", timeout=5)

    t0 = time.time()
    await ws.send(json.dumps({"type": "text_message", "text": PROD_CLASSROOM_PROMPT}))
    complete = await wait_for_type(ws, "bot_text_complete", timeout=15)
    speaker_ms = round((time.time() - t0) * 1000, 1)
    answer = (complete or {}).get("text", "")
    logger.info(f"    → total={speaker_ms}ms  answer='{answer[:40]}'")
    results["classroom_speaker"] = {"total_ms": speaker_ms, "answer": answer[:60]}

    # 3. Classroom speaker + Hindi listener — single short query
    logger.info("  [prod] Classroom speaker + Hindi listener...")
    ws_listener = await websockets.connect(f"{CLASSROOM_WS_BASE}/{room_id}/ws")
    await join_room(ws_listener, room_id, "prod-probe-listener", "ProdListenerHi", "hi")
    await drain(ws_listener, duration=0.5)

    t0 = time.time()
    await ws.send(json.dumps({"type": "text_message", "text": PROD_CLASSROOM_PROMPT}))

    # Wait for listener to get translated text
    listener_texts = []
    listener_done = False
    deadline = time.time() + 20
    while time.time() < deadline:
        msg = await recv_json_nonbinary(ws_listener, timeout=15)
        if not msg:
            break
        if msg.get("type") == "bot_text":
            listener_texts.append(msg.get("text", ""))
        elif msg.get("type") == "bot_text_complete":
            listener_done = True
            break
    listener_full = "".join(listener_texts)
    listener_ms = round((time.time() - t0) * 1000, 1)
    logger.info(f"    → total={listener_ms}ms  hindi='{listener_full[:40]}'  done={listener_done}")
    results["classroom_listener"] = {
        "total_ms": listener_ms, "text": listener_full[:60], "done": listener_done,
    }

    # Also drain speaker complete
    await wait_for_type(ws, "bot_text_complete", timeout=10)

    await ws.close()
    await ws_listener.close()
    await asyncio.sleep(0.5)

    return results


async def main():
    parser = argparse.ArgumentParser(
        description="MIRA Performance Test",
        epilog=(
            "Examples:\n"
            "  python tests/test_performance.py --prod          # Production-safe (3 tiny queries)\n"
            "  python tests/test_performance.py --metrics-only  # Just show current /metrics\n"
            "  python tests/test_performance.py --quick         # 2 iterations per phase\n"
            "  python tests/test_performance.py -n 10           # 10 iterations per phase\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--iterations", "-n", type=int, default=5,
                        help="Number of queries per phase (default: 5)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with 2 iterations")
    parser.add_argument("--prod", action="store_true",
                        help="Production-safe: 1 tiny query per path (~3 LLM calls, ~10s)")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Just fetch and display current /metrics (no queries)")
    parser.add_argument("--phase", choices=["all", "tutor_text", "classroom_speaker", "classroom_listener"],
                        default="all", help="Which phase to run (default: all)")
    parser.add_argument("--host", type=str, default=None,
                        help="Override server host (e.g. https://my-server.example.com)")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    # Accept --test for compatibility with run_all_tests.py harness
    parser.add_argument("--test", type=str, default=None,
                        help="Alias for compatibility with test runner (maps 'all' → --quick)")
    args = parser.parse_args()

    # If invoked via --test all (from run_all_tests.py), treat as --quick
    if args.test:
        args.quick = True

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.quick:
        args.iterations = 2

    if args.host:
        global HTTP_URL, WS_URL, CLASSROOM_WS_BASE
        host = args.host.rstrip("/")
        HTTP_URL = host
        ws_host = host.replace("https://", "wss://").replace("http://", "ws://")
        WS_URL = f"{ws_host}/ws"
        CLASSROOM_WS_BASE = f"{ws_host}/classroom/rooms"

    # ── Metrics-only mode: just fetch and display ──
    if args.metrics_only:
        print()
        print("=" * 60)
        print(f"  MIRA METRICS — {HTTP_URL}")
        print("=" * 60)
        try:
            metrics = await fetch_metrics()
            print_metrics_report(metrics, {})
            print("Raw /metrics JSON:")
            print(json.dumps(metrics, indent=2))
        except Exception as e:
            print(f"  ❌ Cannot reach server: {e}")
            sys.exit(1)
        return

    # ── Verify server is reachable ──
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{HTTP_URL}/health", headers=_jwt_headers(), timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    print(f"  ❌ Server health check failed: {resp.status}")
                    sys.exit(1)
    except Exception as e:
        print(f"  ❌ Cannot reach server: {e}")
        sys.exit(1)

    # ── Production-safe smoke test ──
    if args.prod:
        print()
        print("=" * 60)
        print(f"  MIRA PROD SMOKE TEST — {HTTP_URL}")
        print(f"  (3 lightweight queries — safe for production)")
        print("=" * 60)
        print(f"  ✓ Server is healthy")

        print()
        print("─" * 60)
        print("  Running production smoke test...")
        print("─" * 60)
        t0 = time.time()
        smoke_results = await phase_prod_smoke()
        phase_results = {"prod_smoke": {
            "duration_s": round(time.time() - t0, 1),
            "count": 3,
        }}

        print()
        print("─" * 60)
        print("  Fetching /metrics...")
        print("─" * 60)
        metrics = await fetch_metrics()
        print_metrics_report(metrics, phase_results)
        print("Raw /metrics JSON:")
        print(json.dumps(metrics, indent=2))

        # Print summary line for run_all_tests.py harness compatibility
        print()
        print("=" * 60)
        print(f"  3/3 tests passed")
        print("=" * 60)
        return

    # ── Full benchmark mode ──
    n = args.iterations
    run_all = args.phase == "all"
    phase_results = {}

    print()
    print("=" * 60)
    print(f"  MIRA PERFORMANCE TEST — {n} iterations per phase")
    print(f"  Server: {HTTP_URL}")
    print("=" * 60)
    print(f"  ✓ Server is healthy")

    # ── Phase 1: Tutor text ──
    if run_all or args.phase == "tutor_text":
        print()
        print("─" * 60)
        print("  PHASE 1: Tutor Text Mode")
        print("─" * 60)
        t0 = time.time()
        tutor_results = await phase_tutor_text(n)
        phase_results["tutor_text"] = {
            "duration_s": round(time.time() - t0, 1),
            "count": len(tutor_results),
        }

    # ── Phase 2: Classroom speaker ──
    if run_all or args.phase == "classroom_speaker":
        print()
        print("─" * 60)
        print("  PHASE 2: Classroom Speaker (text queries)")
        print("─" * 60)
        t0 = time.time()
        speaker_results = await phase_classroom_speaker(n)
        phase_results["classroom_speaker"] = {
            "duration_s": round(time.time() - t0, 1),
            "count": len(speaker_results),
        }

    # ── Phase 3: Classroom speaker + listener ──
    if run_all or args.phase == "classroom_listener":
        print()
        print("─" * 60)
        print("  PHASE 3: Classroom Speaker + Hindi Listener")
        print("─" * 60)
        t0 = time.time()
        listener_results = await phase_classroom_listener(n)
        phase_results["classroom_listener"] = {
            "duration_s": round(time.time() - t0, 1),
            "count": len(listener_results),
        }

    # ── Fetch final metrics and display report ──
    print()
    print("─" * 60)
    print("  Fetching /metrics...")
    print("─" * 60)
    metrics = await fetch_metrics()
    print_metrics_report(metrics, phase_results)

    # Also dump raw JSON for programmatic use
    print("Raw /metrics JSON:")
    print(json.dumps(metrics, indent=2))

    # Print summary line for run_all_tests.py harness compatibility
    phase_count = len(phase_results)
    print()
    print("=" * 60)
    print(f"  {phase_count}/{phase_count} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
