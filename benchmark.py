#!/usr/bin/env python3
"""
Performance Benchmark for MiraVoiceAI Pipecat Pipeline.

Measures:
- TTFS (Time to First Speech): Time until first audio byte from TTS
- Voice-to-Voice (V2V) Latency: End-to-end from audio input to audio output
- Component latencies: STT, LLM (with TTFT), TTS
- Throughput: Requests per second under load

Usage:
    # Full benchmark
    python benchmark.py

    # With specific audio file
    python benchmark.py --audio sample.wav

    # Quick benchmark (fewer iterations)
    python benchmark.py --quick

    # Throughput test with concurrency
    python benchmark.py --throughput --concurrency 5
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
import wave
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import aiohttp
import numpy as np

try:
    import websockets
except ImportError:
    print("Error: websockets package required. Install with: pip install websockets")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
INDICASR_WS_URL = os.getenv("INDICASR_WS_URL", "ws://localhost:8082/v1/audio/stream")
SVARA_TTS_URL = os.getenv("SVARA_TTS_URL", "http://localhost:8080")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://gpt-oss-120b/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You are Idli, a helpful AI assistant. Keep responses to 1-2 sentences."""

# Test prompts for benchmarking
TEST_PROMPTS = [
    "नमस्ते, आप कैसे हैं?",
    "What is the weather like today?",
    "मुझे एक कहानी सुनाओ।",
    "Tell me a joke.",
    "ಹೇಗಿದ್ದೀರಿ?",
]


@dataclass
class LatencyMetrics:
    """Latency metrics for a single request."""
    stt_latency: float = 0.0
    llm_ttft: float = 0.0  # Time to First Token
    llm_total: float = 0.0
    tts_ttfs: float = 0.0  # Time to First Speech (actual audio, not WAV header)
    tts_total: float = 0.0
    v2v_latency: float = 0.0  # Voice-to-Voice: end of user speech → first TTS audio

    # Audio metrics
    input_audio_duration: float = 0.0
    output_audio_duration: float = 0.0

    # Text
    transcription: str = ""
    response: str = ""


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    iterations: int = 0
    metrics: List[LatencyMetrics] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add(self, m: LatencyMetrics):
        self.metrics.append(m)
        self.iterations += 1

    def _stats(self, values: List[float]) -> dict:
        if not values:
            return {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "std": 0}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            "mean": statistics.mean(values),
            "p50": sorted_vals[int(n * 0.5)],
            "p95": sorted_vals[int(n * 0.95)] if n >= 20 else sorted_vals[-1],
            "p99": sorted_vals[int(n * 0.99)] if n >= 100 else sorted_vals[-1],
            "min": min(values),
            "max": max(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
        }

    def summary(self) -> dict:
        if not self.metrics:
            return {}

        return {
            "iterations": self.iterations,
            "errors": len(self.errors),
            "stt_latency": self._stats([m.stt_latency for m in self.metrics if m.stt_latency > 0]),
            "llm_ttft": self._stats([m.llm_ttft for m in self.metrics if m.llm_ttft > 0]),
            "llm_total": self._stats([m.llm_total for m in self.metrics if m.llm_total > 0]),
            "tts_ttfs": self._stats([m.tts_ttfs for m in self.metrics if m.tts_ttfs > 0]),
            "tts_total": self._stats([m.tts_total for m in self.metrics if m.tts_total > 0]),
            "v2v_latency": self._stats([m.v2v_latency for m in self.metrics if m.v2v_latency > 0]),
        }


async def benchmark_stt(audio_data: bytes, sample_rate: int, language: str = "auto") -> Tuple[str, float]:
    """
    Benchmark STT latency.

    Returns:
        Tuple of (transcription, latency_seconds)
    """
    start_time = time.perf_counter()

    async with websockets.connect(INDICASR_WS_URL, max_size=10 * 1024 * 1024) as ws:
        # Send config
        config = {"language": language, "interim_results": False, "sample_rate": sample_rate}
        await ws.send(json.dumps(config))

        # Wait for ack
        ack = await asyncio.wait_for(ws.recv(), timeout=10.0)
        ack_data = json.loads(ack)
        if ack_data.get("type") != "config_ack":
            raise RuntimeError(f"Unexpected response: {ack_data}")

        # Send all audio at once (faster than real-time for benchmarking)
        chunk_size = 32000  # 1 second chunks at 16kHz
        offset = 0
        while offset < len(audio_data):
            chunk = audio_data[offset:offset + chunk_size]
            await ws.send(chunk)
            offset += chunk_size

        # Send END
        await ws.send("END")

        # Get final result
        final_text = ""
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                data = json.loads(msg)
                if data.get("type") == "done":
                    final_text = data.get("text", "")
                    break
                elif data.get("type") == "final":
                    final_text = data.get("text", "")
                elif data.get("type") == "error":
                    raise RuntimeError(data.get("message"))
            except asyncio.TimeoutError:
                break

        latency = time.perf_counter() - start_time
        return final_text, latency


async def benchmark_llm(text: str, stream: bool = True) -> Tuple[str, float, float]:
    """
    Benchmark LLM latency.

    Returns:
        Tuple of (response_text, ttft_seconds, total_latency_seconds)
    """
    url = f"{LLM_BASE_URL}/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": stream,
    }

    start_time = time.perf_counter()
    ttft = 0.0
    response_text = ""

    async with aiohttp.ClientSession() as session:
        if stream:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    raise RuntimeError(f"LLM request failed: {response.status} - {error}")

                first_chunk = True
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if first_chunk:
                                ttft = time.perf_counter() - start_time
                                first_chunk = False

                            delta = data.get('choices', [{}])[0].get('delta', {})
                            if 'content' in delta:
                                response_text += delta['content']
                        except json.JSONDecodeError:
                            continue
        else:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    raise RuntimeError(f"LLM request failed: {response.status} - {error}")

                data = await response.json()
                ttft = time.perf_counter() - start_time
                response_text = data["choices"][0]["message"]["content"]

    total_latency = time.perf_counter() - start_time
    return response_text, ttft, total_latency


async def benchmark_tts(text: str, voice: str = "hi_male", stream: bool = True) -> Tuple[bytes, float, float]:
    """
    Benchmark TTS latency.

    TTFS is measured as time to first ACTUAL audio samples, not the WAV header.
    The WAV header is 44 bytes and is sent immediately, which would be misleading.

    Returns:
        Tuple of (audio_bytes, ttfs_seconds, total_latency_seconds)
    """
    WAV_HEADER_SIZE = 44

    if stream:
        url = f"{SVARA_TTS_URL}/v1/speech/text-to-speech/stream"
    else:
        url = f"{SVARA_TTS_URL}/v1/speech/text-to-speech"

    payload = {
        "prompt": text,
        "voice": voice,
        "temperature": 0.75,
        "top_p": 0.9,
        "max_tokens": 1500,
        "repetition_penalty": 1.1,
    }

    start_time = time.perf_counter()
    ttfs = 0.0
    audio_chunks = []
    bytes_received = 0
    ttfs_recorded = False

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error = await response.text()
                raise RuntimeError(f"TTS request failed: {response.status} - {error}")

            async for chunk in response.content.iter_chunked(4096):
                audio_chunks.append(chunk)
                prev_bytes = bytes_received
                bytes_received += len(chunk)

                # TTFS = time when we receive first actual audio AFTER the WAV header
                if not ttfs_recorded and bytes_received > WAV_HEADER_SIZE:
                    ttfs = time.perf_counter() - start_time
                    ttfs_recorded = True

    total_latency = time.perf_counter() - start_time
    audio_data = b"".join(audio_chunks)

    return audio_data, ttfs, total_latency


async def benchmark_v2v_with_audio(
    audio_data: bytes,
    sample_rate: int,
    language: str = "auto",
    voice: str = "hi_male",
) -> LatencyMetrics:
    """
    Benchmark full Voice-to-Voice pipeline with audio input.

    V2V Latency = Time from END of user speech until FIRST actual TTS audio chunk.
    This measures: STT finalization + LLM TTFT + TTS TTFS
    """
    WAV_HEADER_SIZE = 44
    metrics = LatencyMetrics()
    metrics.input_audio_duration = len(audio_data) / 2 / sample_rate

    # === STT Phase ===
    stt_start = time.perf_counter()

    async with websockets.connect(INDICASR_WS_URL, max_size=10 * 1024 * 1024) as ws:
        config = {"language": language, "interim_results": False, "sample_rate": sample_rate}
        await ws.send(json.dumps(config))
        ack = await asyncio.wait_for(ws.recv(), timeout=10.0)
        ack_data = json.loads(ack)
        if ack_data.get("type") != "config_ack":
            raise RuntimeError(f"Unexpected response: {ack_data}")

        # Send all audio
        chunk_size = 32000
        offset = 0
        while offset < len(audio_data):
            chunk = audio_data[offset:offset + chunk_size]
            await ws.send(chunk)
            offset += chunk_size

        # Send END - THIS IS THE END OF USER SPEECH
        await ws.send("END")
        v2v_start = time.perf_counter()  # V2V timer starts here!

        # Get final result
        transcription = ""
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                data = json.loads(msg)
                if data.get("type") == "done":
                    transcription = data.get("text", "")
                    break
                elif data.get("type") == "final":
                    transcription = data.get("text", "")
                elif data.get("type") == "error":
                    raise RuntimeError(data.get("message"))
            except asyncio.TimeoutError:
                break

    metrics.stt_latency = time.perf_counter() - stt_start
    metrics.transcription = transcription

    if not transcription:
        return metrics

    # === LLM Phase ===
    llm_start = time.perf_counter()
    url = f"{LLM_BASE_URL}/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcription}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": True,
    }

    response_text = ""
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error = await response.text()
                raise RuntimeError(f"LLM request failed: {response.status} - {error}")

            first_token = True
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if first_token:
                            metrics.llm_ttft = time.perf_counter() - llm_start
                            first_token = False
                        delta = data.get('choices', [{}])[0].get('delta', {})
                        if 'content' in delta:
                            response_text += delta['content']
                    except json.JSONDecodeError:
                        continue

    metrics.llm_total = time.perf_counter() - llm_start
    metrics.response = response_text

    if not response_text:
        return metrics

    # === TTS Phase ===
    tts_start = time.perf_counter()
    tts_url = f"{SVARA_TTS_URL}/v1/speech/text-to-speech/stream"
    tts_payload = {
        "prompt": response_text,
        "voice": voice,
        "temperature": 0.75,
        "top_p": 0.9,
        "max_tokens": 1500,
        "repetition_penalty": 1.1,
    }

    audio_chunks = []
    bytes_received = 0

    async with aiohttp.ClientSession() as session:
        async with session.post(tts_url, json=tts_payload) as response:
            if response.status != 200:
                error = await response.text()
                raise RuntimeError(f"TTS request failed: {response.status} - {error}")

            async for chunk in response.content.iter_chunked(4096):
                audio_chunks.append(chunk)
                prev_bytes = bytes_received
                bytes_received += len(chunk)

                # Record TTFS when we get first actual audio (past WAV header)
                if prev_bytes <= WAV_HEADER_SIZE < bytes_received:
                    metrics.tts_ttfs = time.perf_counter() - tts_start
                    # V2V ends here - first actual audio received!
                    metrics.v2v_latency = time.perf_counter() - v2v_start

    metrics.tts_total = time.perf_counter() - tts_start
    audio_data_out = b"".join(audio_chunks)
    metrics.output_audio_duration = (len(audio_data_out) - WAV_HEADER_SIZE) / 2 / 24000

    return metrics


async def benchmark_v2v_with_text(
    text: str,
    voice: str = "hi_male",
) -> LatencyMetrics:
    """
    Benchmark LLM -> TTS pipeline (text input, skips STT).

    V2V Latency = Time from text input until FIRST actual TTS audio chunk.
    This measures: LLM TTFT + remaining LLM generation + TTS TTFS
    """
    WAV_HEADER_SIZE = 44
    metrics = LatencyMetrics()
    metrics.transcription = text

    v2v_start = time.perf_counter()  # Simulates "end of user speech"

    # === LLM Phase ===
    llm_start = time.perf_counter()
    url = f"{LLM_BASE_URL}/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": True,
    }

    response_text = ""
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error = await response.text()
                raise RuntimeError(f"LLM request failed: {response.status} - {error}")

            first_token = True
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if first_token:
                            metrics.llm_ttft = time.perf_counter() - llm_start
                            first_token = False
                        delta = data.get('choices', [{}])[0].get('delta', {})
                        if 'content' in delta:
                            response_text += delta['content']
                    except json.JSONDecodeError:
                        continue

    metrics.llm_total = time.perf_counter() - llm_start
    metrics.response = response_text

    if not response_text:
        return metrics

    # === TTS Phase ===
    tts_start = time.perf_counter()
    tts_url = f"{SVARA_TTS_URL}/v1/speech/text-to-speech/stream"
    tts_payload = {
        "prompt": response_text,
        "voice": voice,
        "temperature": 0.75,
        "top_p": 0.9,
        "max_tokens": 1500,
        "repetition_penalty": 1.1,
    }

    audio_chunks = []
    bytes_received = 0

    async with aiohttp.ClientSession() as session:
        async with session.post(tts_url, json=tts_payload) as response:
            if response.status != 200:
                error = await response.text()
                raise RuntimeError(f"TTS request failed: {response.status} - {error}")

            async for chunk in response.content.iter_chunked(4096):
                audio_chunks.append(chunk)
                prev_bytes = bytes_received
                bytes_received += len(chunk)

                # Record TTFS when we get first actual audio (past WAV header)
                if prev_bytes <= WAV_HEADER_SIZE < bytes_received:
                    metrics.tts_ttfs = time.perf_counter() - tts_start
                    # V2V ends here - first actual audio received!
                    metrics.v2v_latency = time.perf_counter() - v2v_start

    metrics.tts_total = time.perf_counter() - tts_start
    audio_data_out = b"".join(audio_chunks)
    metrics.output_audio_duration = (len(audio_data_out) - WAV_HEADER_SIZE) / 2 / 24000

    return metrics


async def run_latency_benchmark(
    iterations: int = 10,
    audio_path: Optional[str] = None,
    language: str = "auto",
    voice: str = "hi_male",
    warmup: int = 2,
) -> BenchmarkResults:
    """
    Run latency benchmark.
    """
    results = BenchmarkResults()

    # Load audio if provided
    audio_data = None
    sample_rate = 16000
    if audio_path:
        with wave.open(audio_path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
            if wf.getnchannels() == 2:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
                audio_data = audio_array.tobytes()

    total_iterations = warmup + iterations

    for i in range(total_iterations):
        is_warmup = i < warmup
        prefix = "[WARMUP]" if is_warmup else f"[{i - warmup + 1}/{iterations}]"

        try:
            if audio_data:
                print(f"{prefix} Running V2V benchmark with audio...", end=" ", flush=True)
                metrics = await benchmark_v2v_with_audio(audio_data, sample_rate, language, voice)
            else:
                text = TEST_PROMPTS[i % len(TEST_PROMPTS)]
                print(f"{prefix} Running V2V benchmark with text: '{text[:30]}...'", end=" ", flush=True)
                metrics = await benchmark_v2v_with_text(text, voice)

            if not is_warmup:
                results.add(metrics)

            print(f"V2V: {metrics.v2v_latency*1000:.0f}ms")

        except Exception as e:
            print(f"ERROR: {e}")
            if not is_warmup:
                results.errors.append(str(e))

    return results


async def run_throughput_benchmark(
    duration: int = 30,
    concurrency: int = 5,
    voice: str = "hi_male",
) -> dict:
    """
    Run throughput benchmark with concurrent requests.
    """
    print(f"\nRunning throughput benchmark for {duration}s with {concurrency} concurrent requests...")

    completed = 0
    errors = 0
    latencies = []
    stop_event = asyncio.Event()

    async def worker(worker_id: int):
        nonlocal completed, errors
        while not stop_event.is_set():
            try:
                text = TEST_PROMPTS[completed % len(TEST_PROMPTS)]
                start = time.perf_counter()

                # Just LLM + TTS for throughput (STT would bottleneck)
                _, _, llm_latency = await benchmark_llm(text, stream=False)
                response = "Test response for throughput benchmark."
                _, _, tts_latency = await benchmark_tts(response, voice, stream=False)

                latency = time.perf_counter() - start
                latencies.append(latency)
                completed += 1

            except Exception as e:
                errors += 1
                logger.warning(f"Worker {worker_id} error: {e}")

    # Start workers
    start_time = time.perf_counter()
    workers = [asyncio.create_task(worker(i)) for i in range(concurrency)]

    # Run for duration
    await asyncio.sleep(duration)
    stop_event.set()

    # Wait for workers to finish current requests
    await asyncio.gather(*workers, return_exceptions=True)

    elapsed = time.perf_counter() - start_time

    return {
        "duration_s": elapsed,
        "concurrency": concurrency,
        "completed_requests": completed,
        "errors": errors,
        "throughput_rps": completed / elapsed,
        "avg_latency_ms": statistics.mean(latencies) * 1000 if latencies else 0,
        "p50_latency_ms": sorted(latencies)[len(latencies)//2] * 1000 if latencies else 0,
        "p95_latency_ms": sorted(latencies)[int(len(latencies)*0.95)] * 1000 if len(latencies) >= 20 else 0,
    }


def print_results(results: BenchmarkResults):
    """Print benchmark results."""
    summary = results.summary()
    if not summary:
        print("No results to display")
        return

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\nIterations: {summary['iterations']}")
    print(f"Errors: {summary['errors']}")

    def print_metric(name: str, stats: dict, unit: str = "ms", multiplier: float = 1000):
        print(f"\n{name}:")
        print(f"  Mean:  {stats['mean']*multiplier:>8.1f} {unit}")
        print(f"  P50:   {stats['p50']*multiplier:>8.1f} {unit}")
        print(f"  P95:   {stats['p95']*multiplier:>8.1f} {unit}")
        print(f"  P99:   {stats['p99']*multiplier:>8.1f} {unit}")
        print(f"  Min:   {stats['min']*multiplier:>8.1f} {unit}")
        print(f"  Max:   {stats['max']*multiplier:>8.1f} {unit}")
        print(f"  Std:   {stats['std']*multiplier:>8.1f} {unit}")

    if summary['stt_latency']['mean'] > 0:
        print_metric("STT Latency", summary['stt_latency'])

    print_metric("LLM TTFT (Time to First Token)", summary['llm_ttft'])
    print_metric("LLM Total Latency", summary['llm_total'])
    print_metric("TTS TTFS (Time to First Speech - actual audio)", summary['tts_ttfs'])
    print_metric("TTS Total Latency", summary['tts_total'])
    print_metric("V2V (End of Speech → First Audio)", summary['v2v_latency'])

    # Calculate component breakdown for V2V
    if results.metrics:
        avg_llm_total = statistics.mean([m.llm_total for m in results.metrics if m.llm_total > 0])
        avg_tts_ttfs = statistics.mean([m.tts_ttfs for m in results.metrics if m.tts_ttfs > 0])
        avg_tts_total = statistics.mean([m.tts_total for m in results.metrics if m.tts_total > 0])
        avg_stt = statistics.mean([m.stt_latency for m in results.metrics if m.stt_latency > 0]) if any(m.stt_latency > 0 for m in results.metrics) else 0
        avg_v2v = statistics.mean([m.v2v_latency for m in results.metrics if m.v2v_latency > 0])

        print("\n" + "-" * 70)
        print("V2V LATENCY BREAKDOWN")
        print("(End of user speech → First TTS audio)")
        print("-" * 70)
        if avg_stt > 0:
            # For audio input: V2V = STT_remaining + LLM_total + TTS_TTFS
            stt_remaining = avg_v2v - avg_llm_total - avg_tts_ttfs
            print(f"  STT (after END):  {stt_remaining*1000:>7.1f} ms ({stt_remaining/avg_v2v*100:>5.1f}%)")
            print(f"  LLM (total):      {avg_llm_total*1000:>7.1f} ms ({avg_llm_total/avg_v2v*100:>5.1f}%)")
            print(f"  TTS (TTFS):       {avg_tts_ttfs*1000:>7.1f} ms ({avg_tts_ttfs/avg_v2v*100:>5.1f}%)")
        else:
            # For text input: V2V = LLM_total + TTS_TTFS
            print(f"  LLM (total):      {avg_llm_total*1000:>7.1f} ms ({avg_llm_total/avg_v2v*100:>5.1f}%)")
            print(f"  TTS (TTFS):       {avg_tts_ttfs*1000:>7.1f} ms ({avg_tts_ttfs/avg_v2v*100:>5.1f}%)")
        print(f"  ─────────────────────────────────────")
        print(f"  V2V Total:        {avg_v2v*1000:>7.1f} ms")

        print("\n" + "-" * 70)
        print("COMPONENT TOTALS (for reference)")
        print("-" * 70)
        if avg_stt > 0:
            print(f"  STT Total:        {avg_stt*1000:>7.1f} ms")
        print(f"  LLM Total:        {avg_llm_total*1000:>7.1f} ms")
        print(f"  TTS Total:        {avg_tts_total*1000:>7.1f} ms")

    print("\n" + "=" * 70)


def print_throughput_results(results: dict):
    """Print throughput results."""
    print("\n" + "=" * 70)
    print("THROUGHPUT BENCHMARK RESULTS")
    print("=" * 70)
    print(f"\n  Duration:           {results['duration_s']:.1f} s")
    print(f"  Concurrency:        {results['concurrency']}")
    print(f"  Completed Requests: {results['completed_requests']}")
    print(f"  Errors:             {results['errors']}")
    print(f"\n  Throughput:         {results['throughput_rps']:.2f} req/s")
    print(f"  Avg Latency:        {results['avg_latency_ms']:.1f} ms")
    print(f"  P50 Latency:        {results['p50_latency_ms']:.1f} ms")
    print(f"  P95 Latency:        {results['p95_latency_ms']:.1f} ms")
    print("\n" + "=" * 70)


async def main():
    parser = argparse.ArgumentParser(description="Benchmark MiraVoiceAI Pipecat Pipeline")
    parser.add_argument("--audio", help="Audio file for V2V benchmark")
    parser.add_argument("--iterations", "-n", type=int, default=10, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--language", default="auto", help="STT language code")
    parser.add_argument("--voice", default="hi_male", help="TTS voice ID")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (3 iterations)")
    parser.add_argument("--throughput", action="store_true", help="Run throughput benchmark")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrency for throughput test")
    parser.add_argument("--duration", type=int, default=30, help="Duration for throughput test (seconds)")

    args = parser.parse_args()

    if args.quick:
        args.iterations = 3
        args.warmup = 1

    print("\n" + "=" * 70)
    print("MiraVoiceAI Pipecat Pipeline Benchmark")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  STT:  {INDICASR_WS_URL}")
    print(f"  LLM:  {LLM_BASE_URL} ({LLM_MODEL})")
    print(f"  TTS:  {SVARA_TTS_URL}")
    print(f"  Voice: {args.voice}")
    if args.audio:
        print(f"  Audio: {args.audio}")
    print()

    # Latency benchmark
    results = await run_latency_benchmark(
        iterations=args.iterations,
        audio_path=args.audio,
        language=args.language,
        voice=args.voice,
        warmup=args.warmup,
    )
    print_results(results)

    # Throughput benchmark
    if args.throughput:
        throughput_results = await run_throughput_benchmark(
            duration=args.duration,
            concurrency=args.concurrency,
            voice=args.voice,
        )
        print_throughput_results(throughput_results)


if __name__ == "__main__":
    asyncio.run(main())
