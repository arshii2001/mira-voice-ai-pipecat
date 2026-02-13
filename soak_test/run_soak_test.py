#!/usr/bin/env python3
"""
30-Minute Text-Audio Sync Drift Soak Test
==========================================

Proves that Mira's TextStreamForwarder + TextAudioSyncNotifier pipeline
does NOT drift over a sustained 30-minute conversation with realistic,
diverse student personas.

Architecture:
  1. Connects to Mira via WebSocket (voice pipeline)
  2. Sends scripted dialog turns via /inject_text (text injection)
  3. Listens for bot_text / bot_text_complete JSON messages
  4. Measures per-turn timing: inject → first bot_text → bot_text_complete
  5. Polls /metrics for server-side sync stats
  6. Generates a drift report (console + CSV)

Uses sdialog-inspired student personas (Ananya, Ravi, Priya, Arjun, Karthik)
with different languages, emotional arcs, and communication styles.

Usage:
  # Against OSS deployment
  python soak_test/run_soak_test.py --target oss

  # Against local Docker
  python soak_test/run_soak_test.py --target local

  # Custom endpoint
  python soak_test/run_soak_test.py --http-url https://mira-oss.inf7ks8.com/pipecat

  # Quick 5-minute smoke test
  python soak_test/run_soak_test.py --target oss --quick
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import jwt  # PyJWT
import httpx
import websockets

# Add parent dir to path so we can import personas
sys.path.insert(0, str(Path(__file__).parent))
from personas import DIALOG_SCRIPT, ALL_PERSONAS, StudentPersona

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("soak_test")

# ── Deployment targets ──
TARGETS = {
    "local": {
        "http": "http://localhost:7860",
        "ws": "ws://localhost:7860/ws",
    },
    "oss": {
        "http": "https://mira-oss.inf7ks8.com/pipecat",
        "ws": "wss://mira-oss.inf7ks8.com/pipecat/ws",
    },
    "elevenlabs": {
        "http": "https://mira-ai.westus2.cloudapp.azure.com/pipecat",
        "ws": "wss://mira-ai.westus2.cloudapp.azure.com/pipecat/ws",
    },
}


# ── Data structures ──

@dataclass
class TurnMetrics:
    """Timing metrics for a single dialog turn."""
    turn_index: int
    persona_name: str
    language: str
    utterance_preview: str          # first 60 chars of the sent text
    expected_behavior: str
    inject_time: float = 0.0       # time.time() when text was injected
    first_bot_text_time: float = 0.0  # time of first bot_text JSON received
    complete_time: float = 0.0     # time of bot_text_complete received
    sentence_count: int = 0        # number of bot_text messages received
    total_response_text: str = ""  # full bot response
    error: str = ""                # error message if any

    @property
    def inject_to_first_text_ms(self) -> float:
        if self.first_bot_text_time and self.inject_time:
            return (self.first_bot_text_time - self.inject_time) * 1000
        return -1

    @property
    def inject_to_complete_ms(self) -> float:
        if self.complete_time and self.inject_time:
            return (self.complete_time - self.inject_time) * 1000
        return -1

    @property
    def first_text_to_complete_ms(self) -> float:
        if self.complete_time and self.first_bot_text_time:
            return (self.complete_time - self.first_bot_text_time) * 1000
        return -1

    def elapsed_since_start(self, test_start: float) -> float:
        """Minutes elapsed from test start to this turn's injection."""
        return (self.inject_time - test_start) / 60.0


@dataclass
class SoakTestResult:
    """Aggregated results from the full soak test."""
    test_start: float = 0.0
    test_end: float = 0.0
    target: str = ""
    turns: List[TurnMetrics] = field(default_factory=list)
    server_metrics_snapshots: List[dict] = field(default_factory=list)

    @property
    def duration_minutes(self) -> float:
        return (self.test_end - self.test_start) / 60.0

    @property
    def successful_turns(self) -> List[TurnMetrics]:
        return [t for t in self.turns if not t.error and t.complete_time > 0]

    @property
    def failed_turns(self) -> List[TurnMetrics]:
        return [t for t in self.turns if t.error]


# ── JWT Helper ──

def generate_jwt(secret_key: str) -> str:
    """Generate a JWT token compatible with OpenWebUI."""
    payload = {
        "id": "soak-test-user",
        "exp": int(time.time()) + 7200,  # 2 hours
        "jti": f"soak-{int(time.time())}",
    }
    return jwt.encode(payload, secret_key, algorithm="HS256")


# ── WebSocket Client ──

class MiraSoakClient:
    """
    WebSocket client that connects to Mira's voice pipeline,
    sends text via /inject_text, and collects bot responses
    with precise timing for drift analysis.
    """

    def __init__(self, http_url: str, ws_url: str, jwt_token: str = ""):
        self.http_url = http_url.rstrip("/")
        self.ws_url = ws_url
        self.jwt_token = jwt_token
        self.session_id: Optional[str] = None
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._listener_task: Optional[asyncio.Task] = None

    async def connect(self, language: str = "en") -> bool:
        """Connect to Mira WebSocket and authenticate."""
        try:
            logger.info(f"Connecting to {self.ws_url} ...")
            self.ws = await websockets.connect(
                self.ws_url,
                open_timeout=30,
                close_timeout=10,
                ping_interval=20,
                ping_timeout=60,
                max_size=10 * 1024 * 1024,
            )

            # Send config message
            config_msg = {
                "type": "config",
                "config": {
                    "enable_greeting": True,
                    "language": language,
                },
            }
            if self.jwt_token:
                config_msg["token"] = self.jwt_token

            await self.ws.send(json.dumps(config_msg))
            logger.info("Config sent, waiting for session_id...")

            # Start listener
            self._listener_task = asyncio.create_task(self._listen_loop())

            # Wait for session_id (up to 30s)
            deadline = time.time() + 30
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(self._message_queue.get(), timeout=5.0)
                    if isinstance(msg, dict) and msg.get("type") == "session_id":
                        self.session_id = msg["session_id"]
                        logger.info(f"Got session_id: {self.session_id}")
                        return True
                    elif isinstance(msg, dict) and msg.get("type") == "error":
                        logger.error(f"Server error: {msg.get('message')}")
                        return False
                except asyncio.TimeoutError:
                    continue

            logger.error("Timed out waiting for session_id")
            return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def _listen_loop(self):
        """Background task to receive WebSocket messages."""
        try:
            async for raw in self.ws:
                if isinstance(raw, str):
                    try:
                        msg = json.loads(raw)
                        await self._message_queue.put(msg)
                    except json.JSONDecodeError:
                        pass
                # Ignore binary (audio) frames
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.warning(f"Listener error: {e}")

    async def inject_text(self, text: str) -> bool:
        """Send text via /inject_text HTTP endpoint."""
        if not self.session_id:
            logger.error("No session_id — cannot inject text")
            return False

        headers = {}
        if self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.http_url}/inject_text",
                    json={"session_id": self.session_id, "text": text},
                    headers=headers,
                )
                if resp.status_code == 200:
                    return True
                else:
                    logger.error(f"inject_text failed: {resp.status_code} {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"inject_text error: {e}")
            return False

    async def collect_response(self, timeout: float = 60.0) -> Dict:
        """
        Collect bot response messages until bot_text_complete or timeout.

        Returns dict with:
          - first_text_time: time of first bot_text
          - complete_time: time of bot_text_complete
          - sentence_count: number of bot_text messages
          - full_text: concatenated response text
          - error: error string if any
        """
        result = {
            "first_text_time": 0.0,
            "complete_time": 0.0,
            "sentence_count": 0,
            "full_text": "",
            "error": "",
        }

        text_parts = []
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                remaining = max(0.1, deadline - time.time())
                msg = await asyncio.wait_for(self._message_queue.get(), timeout=remaining)

                if not isinstance(msg, dict):
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "bot_text":
                    now = time.time()
                    result["sentence_count"] += 1
                    if result["sentence_count"] == 1:
                        result["first_text_time"] = now
                    text = msg.get("text", "")
                    text_parts.append(text)

                elif msg_type == "bot_text_complete":
                    result["complete_time"] = time.time()
                    result["full_text"] = msg.get("text", " ".join(text_parts))
                    return result

                elif msg_type == "error":
                    result["error"] = msg.get("message", "unknown error")
                    return result

                elif msg_type == "stt_clarification":
                    # STT clarity gate fired — this is expected for short/unclear inputs
                    logger.info(f"  STT clarification requested: {msg.get('message', '')[:60]}")

            except asyncio.TimeoutError:
                break

        if text_parts and not result["complete_time"]:
            result["full_text"] = " ".join(text_parts)
            result["error"] = "timeout_waiting_for_complete"
            result["complete_time"] = time.time()

        if not text_parts:
            result["error"] = "no_response"

        return result

    async def fetch_server_metrics(self) -> Optional[dict]:
        """Pull /metrics from the server."""
        headers = {}
        if self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(f"{self.http_url}/metrics", headers=headers)
                if resp.status_code == 200:
                    return resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch metrics: {e}")
        return None

    async def disconnect(self):
        """Close WebSocket connection."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass


# ── Main Soak Test Runner ──

async def run_soak_test(
    http_url: str,
    ws_url: str,
    jwt_token: str = "",
    quick: bool = False,
) -> SoakTestResult:
    """
    Execute the 30-minute soak test.

    Args:
        http_url: Mira Pipecat HTTP base URL
        ws_url: Mira Pipecat WebSocket URL
        jwt_token: JWT for authentication
        quick: If True, run only first 10 turns (5-min smoke test)

    Returns:
        SoakTestResult with all timing data
    """
    result = SoakTestResult(target=http_url)
    result.test_start = time.time()

    # Select dialog turns
    script = DIALOG_SCRIPT[:10] if quick else DIALOG_SCRIPT

    logger.info("=" * 70)
    logger.info(f"MIRA TEXT-AUDIO SYNC SOAK TEST")
    logger.info(f"Target: {http_url}")
    logger.info(f"Turns: {len(script)} ({'quick mode' if quick else 'full 30-min'})")
    logger.info(f"Personas: {', '.join(p.name for p in ALL_PERSONAS)}")
    logger.info("=" * 70)

    # Connect to Mira
    client = MiraSoakClient(http_url, ws_url, jwt_token)
    connected = await client.connect(language="en")
    if not connected:
        logger.error("FATAL: Could not connect to Mira. Aborting.")
        result.test_end = time.time()
        return result

    # Drain the greeting (wait for bot_text_complete from greeting)
    logger.info("Waiting for greeting to complete...")
    greeting_resp = await client.collect_response(timeout=30.0)
    if greeting_resp["full_text"]:
        logger.info(f"Greeting received: '{greeting_resp['full_text'][:80]}...'")
    else:
        logger.warning("No greeting received (may be disabled)")

    # Fetch initial server metrics
    initial_metrics = await client.fetch_server_metrics()
    if initial_metrics:
        result.server_metrics_snapshots.append({
            "time": time.time(),
            "label": "start",
            "data": initial_metrics,
        })

    # ── Execute dialog turns ──
    for i, (persona, utterance, expected, delay) in enumerate(script):
        turn_num = i + 1
        elapsed_min = (time.time() - result.test_start) / 60.0

        logger.info(f"\n{'─' * 60}")
        logger.info(
            f"Turn {turn_num}/{len(script)} | "
            f"{persona.name} ({persona.language}) | "
            f"t={elapsed_min:.1f}min | "
            f"expect={expected}"
        )
        logger.info(f"  → \"{utterance[:70]}{'...' if len(utterance) > 70 else ''}\"")

        # Create turn metrics
        turn = TurnMetrics(
            turn_index=i,
            persona_name=persona.name,
            language=persona.language,
            utterance_preview=utterance[:60],
            expected_behavior=expected,
        )

        # Wait for pacing delay (simulates real conversation rhythm)
        if delay > 0 and i > 0:
            logger.info(f"  ⏳ Waiting {delay}s (simulating student think time)...")
            await asyncio.sleep(delay)

        # Inject the text
        turn.inject_time = time.time()
        success = await client.inject_text(utterance)
        if not success:
            turn.error = "inject_failed"
            result.turns.append(turn)
            logger.error(f"  ✗ Text injection failed")
            continue

        # Collect response
        resp = await client.collect_response(timeout=90.0)
        turn.first_bot_text_time = resp["first_text_time"]
        turn.complete_time = resp["complete_time"]
        turn.sentence_count = resp["sentence_count"]
        turn.total_response_text = resp["full_text"]
        turn.error = resp["error"]

        result.turns.append(turn)

        # Log turn results
        if turn.error:
            logger.warning(
                f"  ✗ {turn.error} | "
                f"sentences={turn.sentence_count}"
            )
        else:
            logger.info(
                f"  ✓ inject→first_text: {turn.inject_to_first_text_ms:.0f}ms | "
                f"inject→complete: {turn.inject_to_complete_ms:.0f}ms | "
                f"sentences: {turn.sentence_count} | "
                f"response: '{turn.total_response_text[:60]}...'"
            )

        # Periodic metrics snapshot (every 5 turns)
        if turn_num % 5 == 0:
            metrics = await client.fetch_server_metrics()
            if metrics:
                result.server_metrics_snapshots.append({
                    "time": time.time(),
                    "label": f"turn_{turn_num}",
                    "data": metrics,
                })

    # Final metrics snapshot
    final_metrics = await client.fetch_server_metrics()
    if final_metrics:
        result.server_metrics_snapshots.append({
            "time": time.time(),
            "label": "end",
            "data": final_metrics,
        })

    # Disconnect
    await client.disconnect()

    result.test_end = time.time()
    logger.info(f"\n{'=' * 70}")
    logger.info(f"SOAK TEST COMPLETE — Duration: {result.duration_minutes:.1f} minutes")
    logger.info(f"{'=' * 70}")

    return result


# ── Report Generation ──

def generate_report(result: SoakTestResult, output_dir: str = "soak_test/output"):
    """
    Generate drift analysis report from soak test results.

    Creates:
      - Console summary
      - CSV with per-turn timing data
      - JSON with full results + drift analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    successful = result.successful_turns
    failed = result.failed_turns

    # ── Console Report ──
    print("\n" + "=" * 70)
    print("DRIFT ANALYSIS REPORT")
    print("=" * 70)
    print(f"Target:    {result.target}")
    print(f"Duration:  {result.duration_minutes:.1f} minutes")
    print(f"Turns:     {len(result.turns)} total, {len(successful)} successful, {len(failed)} failed")
    print()

    if not successful:
        print("ERROR: No successful turns — cannot compute drift metrics.")
        return

    # Per-turn latencies
    inject_to_first = [t.inject_to_first_text_ms for t in successful if t.inject_to_first_text_ms > 0]
    inject_to_complete = [t.inject_to_complete_ms for t in successful if t.inject_to_complete_ms > 0]
    first_to_complete = [t.first_text_to_complete_ms for t in successful if t.first_text_to_complete_ms > 0]

    print("─── Latency Summary (ms) ───")
    for label, data in [
        ("Inject → First Text", inject_to_first),
        ("Inject → Complete", inject_to_complete),
        ("First Text → Complete", first_to_complete),
    ]:
        if data:
            print(f"  {label}:")
            print(f"    mean={statistics.mean(data):.0f}  median={statistics.median(data):.0f}  "
                  f"p95={sorted(data)[int(len(data)*0.95)]:.0f}  "
                  f"min={min(data):.0f}  max={max(data):.0f}  stddev={statistics.stdev(data):.0f}" if len(data) > 1
                  else f"    mean={statistics.mean(data):.0f}")
    print()

    # ── Drift Analysis ──
    # Split turns into first half and second half and compare latencies
    half = len(successful) // 2
    first_half = successful[:half]
    second_half = successful[half:]

    first_half_latencies = [t.inject_to_first_text_ms for t in first_half if t.inject_to_first_text_ms > 0]
    second_half_latencies = [t.inject_to_first_text_ms for t in second_half if t.inject_to_first_text_ms > 0]

    print("─── Drift Analysis (First Half vs Second Half) ───")
    if first_half_latencies and second_half_latencies:
        mean_first = statistics.mean(first_half_latencies)
        mean_second = statistics.mean(second_half_latencies)
        drift_ms = mean_second - mean_first
        drift_pct = (drift_ms / mean_first * 100) if mean_first > 0 else 0

        print(f"  First half  (turns 1-{half}):  mean={mean_first:.0f}ms")
        print(f"  Second half (turns {half+1}-{len(successful)}): mean={mean_second:.0f}ms")
        print(f"  Drift: {drift_ms:+.0f}ms ({drift_pct:+.1f}%)")

        if abs(drift_pct) < 10:
            print(f"  ✅ PASS — Drift is within ±10% tolerance")
        elif abs(drift_pct) < 25:
            print(f"  ⚠️  WARNING — Drift is {drift_pct:.1f}%, approaching threshold")
        else:
            print(f"  ❌ FAIL — Drift exceeds 25% threshold")
    print()

    # ── Per-Persona Breakdown ──
    print("─── Per-Persona Breakdown ───")
    persona_turns: Dict[str, List[TurnMetrics]] = {}
    for t in successful:
        persona_turns.setdefault(t.persona_name, []).append(t)

    for name, turns in sorted(persona_turns.items()):
        latencies = [t.inject_to_first_text_ms for t in turns if t.inject_to_first_text_ms > 0]
        lang = turns[0].language if turns else "?"
        if latencies:
            print(f"  {name} ({lang}): {len(turns)} turns, "
                  f"mean={statistics.mean(latencies):.0f}ms, "
                  f"max={max(latencies):.0f}ms, "
                  f"sentences/turn={statistics.mean([t.sentence_count for t in turns]):.1f}")
    print()

    # ── Queue Leak Check ──
    # Check if any turn had 0 sentences (potential queue leak)
    zero_sentence_turns = [t for t in successful if t.sentence_count == 0]
    print("─── Queue Health ───")
    print(f"  Zero-sentence responses: {len(zero_sentence_turns)}/{len(successful)}")
    if zero_sentence_turns:
        print(f"  ⚠️  {len(zero_sentence_turns)} turns got no bot_text messages (potential queue issue)")
        for t in zero_sentence_turns:
            print(f"    Turn {t.turn_index}: {t.persona_name} — '{t.utterance_preview}'")
    else:
        print(f"  ✅ All turns received at least one bot_text message")
    print()

    # ── Failed Turns ──
    if failed:
        print("─── Failed Turns ───")
        for t in failed:
            print(f"  Turn {t.turn_index}: {t.persona_name} — {t.error} — '{t.utterance_preview}'")
    print()

    # ── Trend Line (latency over time) ──
    print("─── Latency Trend (inject → first text) ───")
    bucket_size = max(1, len(successful) // 6)  # ~6 buckets
    for i in range(0, len(successful), bucket_size):
        bucket = successful[i:i+bucket_size]
        bucket_latencies = [t.inject_to_first_text_ms for t in bucket if t.inject_to_first_text_ms > 0]
        if bucket_latencies:
            elapsed = bucket[0].elapsed_since_start(result.test_start)
            mean_lat = statistics.mean(bucket_latencies)
            bar = "█" * int(mean_lat / 100)
            print(f"  t={elapsed:5.1f}min | mean={mean_lat:6.0f}ms | {bar}")
    print()

    # ── CSV Output ──
    csv_path = os.path.join(output_dir, f"soak_turns_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "turn_index", "persona", "language", "expected_behavior",
            "inject_to_first_text_ms", "inject_to_complete_ms",
            "first_text_to_complete_ms", "sentence_count",
            "elapsed_minutes", "error", "utterance_preview",
            "response_preview",
        ])
        for t in result.turns:
            writer.writerow([
                t.turn_index, t.persona_name, t.language, t.expected_behavior,
                f"{t.inject_to_first_text_ms:.0f}" if t.inject_to_first_text_ms > 0 else "",
                f"{t.inject_to_complete_ms:.0f}" if t.inject_to_complete_ms > 0 else "",
                f"{t.first_text_to_complete_ms:.0f}" if t.first_text_to_complete_ms > 0 else "",
                t.sentence_count,
                f"{t.elapsed_since_start(result.test_start):.2f}",
                t.error,
                t.utterance_preview,
                t.total_response_text[:100],
            ])
    print(f"CSV written: {csv_path}")

    # ── JSON Output ──
    json_path = os.path.join(output_dir, f"soak_results_{timestamp}.json")
    json_data = {
        "test_start": datetime.fromtimestamp(result.test_start).isoformat(),
        "test_end": datetime.fromtimestamp(result.test_end).isoformat(),
        "duration_minutes": result.duration_minutes,
        "target": result.target,
        "total_turns": len(result.turns),
        "successful_turns": len(successful),
        "failed_turns": len(failed),
        "drift_analysis": {},
        "per_persona": {},
        "latency_summary": {},
        "server_metrics_snapshots": result.server_metrics_snapshots,
    }

    # Add drift analysis
    if first_half_latencies and second_half_latencies:
        json_data["drift_analysis"] = {
            "first_half_mean_ms": statistics.mean(first_half_latencies),
            "second_half_mean_ms": statistics.mean(second_half_latencies),
            "drift_ms": statistics.mean(second_half_latencies) - statistics.mean(first_half_latencies),
            "drift_pct": ((statistics.mean(second_half_latencies) - statistics.mean(first_half_latencies))
                          / statistics.mean(first_half_latencies) * 100) if statistics.mean(first_half_latencies) > 0 else 0,
            "pass": abs((statistics.mean(second_half_latencies) - statistics.mean(first_half_latencies))
                        / statistics.mean(first_half_latencies) * 100) < 10 if statistics.mean(first_half_latencies) > 0 else True,
        }

    # Add latency summary
    if inject_to_first:
        json_data["latency_summary"]["inject_to_first_text_ms"] = {
            "mean": statistics.mean(inject_to_first),
            "median": statistics.median(inject_to_first),
            "min": min(inject_to_first),
            "max": max(inject_to_first),
            "stddev": statistics.stdev(inject_to_first) if len(inject_to_first) > 1 else 0,
        }

    # Per-persona
    for name, turns in persona_turns.items():
        latencies = [t.inject_to_first_text_ms for t in turns if t.inject_to_first_text_ms > 0]
        json_data["per_persona"][name] = {
            "language": turns[0].language,
            "turns": len(turns),
            "mean_latency_ms": statistics.mean(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "avg_sentences": statistics.mean([t.sentence_count for t in turns]),
        }

    # Per-turn data
    json_data["turns"] = [
        {
            "index": t.turn_index,
            "persona": t.persona_name,
            "language": t.language,
            "inject_to_first_text_ms": t.inject_to_first_text_ms,
            "inject_to_complete_ms": t.inject_to_complete_ms,
            "sentence_count": t.sentence_count,
            "elapsed_min": t.elapsed_since_start(result.test_start),
            "error": t.error,
        }
        for t in result.turns
    ]

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"JSON written: {json_path}")

    # ── Final Verdict ──
    print()
    drift_pass = json_data.get("drift_analysis", {}).get("pass", True)
    fail_rate = len(failed) / len(result.turns) * 100 if result.turns else 0
    queue_healthy = len(zero_sentence_turns) == 0

    if drift_pass and fail_rate < 15 and queue_healthy:
        print("🟢 OVERALL: PASS — No significant drift, queue healthy, low failure rate")
    elif drift_pass and (fail_rate < 30 or not queue_healthy):
        print("🟡 OVERALL: WARN — Drift OK but some concerns (failures or queue issues)")
    else:
        print("🔴 OVERALL: FAIL — Significant drift detected or high failure rate")

    return json_data


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="Mira Text-Audio Sync Soak Test")
    parser.add_argument("--target", choices=["local", "oss", "elevenlabs"],
                        default="oss", help="Deployment target (default: oss)")
    parser.add_argument("--http-url", help="Override HTTP base URL")
    parser.add_argument("--ws-url", help="Override WebSocket URL")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: only 10 turns (~5 min)")
    parser.add_argument("--output-dir", default="soak_test/output",
                        help="Output directory for reports")
    args = parser.parse_args()

    # Resolve URLs
    target = TARGETS.get(args.target, TARGETS["oss"])
    http_url = args.http_url or target["http"]
    ws_url = args.ws_url or target["ws"]

    # JWT token
    secret_key = os.environ.get("WEBUI_SECRET_KEY", "")
    jwt_token = generate_jwt(secret_key) if secret_key else ""
    if not jwt_token:
        logger.warning("No WEBUI_SECRET_KEY set — running without JWT auth")

    # Run the soak test
    result = asyncio.run(run_soak_test(http_url, ws_url, jwt_token, quick=args.quick))

    # Generate report
    generate_report(result, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
