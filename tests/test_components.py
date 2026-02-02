#!/usr/bin/env python3
"""
Component tests for MiraVoiceAI Pipecat services.

Tests individual components (STT, LLM, TTS) before full pipeline integration.
Run this first to verify all backend services are accessible.

Usage:
    # Test all components
    python test_components.py

    # Test specific component
    python test_components.py --stt
    python test_components.py --llm
    python test_components.py --tts

    # Test with custom URLs
    python test_components.py --stt-url ws://localhost:8082/v1/audio/speech-to-text/stream
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import wave
from pathlib import Path

import aiohttp
import numpy as np

try:
    import websockets
except ImportError:
    print("Error: websockets package required. Install with: pip install websockets")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_stt(
    ws_url: str = "ws://localhost:8082/v1/audio/speech-to-text/stream",
    audio_path: str = None,
    language: str = "auto",
) -> bool:
    """
    Test IndicASR STT service.

    Args:
        ws_url: WebSocket URL for IndicASR
        audio_path: Path to test audio file (optional, generates test audio if not provided)
        language: Language code for transcription

    Returns:
        True if test passed, False otherwise
    """
    print("\n" + "=" * 60)
    print("Testing IndicASR STT Service")
    print("=" * 60)

    if audio_path and os.path.exists(audio_path):
        # Read audio file
        with wave.open(audio_path, "rb") as wf:
            sample_rate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())
        audio_duration = len(audio_data) / 2 / sample_rate
        logger.info(f"Using audio file: {audio_path} ({audio_duration:.2f}s)")
    else:
        # Generate test silence with some noise
        logger.info("No audio file provided, generating test silence...")
        sample_rate = 16000
        duration = 3.0
        samples = int(sample_rate * duration)
        # Generate low-level noise
        noise = np.random.randn(samples) * 100
        audio_data = noise.astype(np.int16).tobytes()
        audio_duration = duration

    try:
        logger.info(f"Connecting to {ws_url}")
        async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
            # Send config
            config = {
                "language": language,
                "interim_results": True,
                "sample_rate": sample_rate,
            }
            await ws.send(json.dumps(config))
            logger.info(f"Sent config: {config}")

            # Wait for ack
            ack = await asyncio.wait_for(ws.recv(), timeout=10.0)
            ack_data = json.loads(ack)

            if ack_data.get("type") != "config_ack":
                logger.error(f"Unexpected response: {ack_data}")
                return False

            session_id = ack_data.get("session_id")
            logger.info(f"Connected! Session ID: {session_id}")

            # Send audio in chunks
            chunk_size = 3200  # 100ms at 16kHz
            offset = 0
            while offset < len(audio_data):
                chunk = audio_data[offset:offset + chunk_size]
                await ws.send(chunk)
                offset += chunk_size
                await asyncio.sleep(0.05)

            logger.info(f"Sent {len(audio_data)} bytes of audio")

            # Send END
            await ws.send("END")
            logger.info("Sent END signal")

            # Collect results
            results = []
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    data = json.loads(msg)
                    results.append(data)

                    if data.get("type") == "language":
                        logger.info(f"  Language: {data.get('language')} ({data.get('probability', 0):.2f})")
                    elif data.get("type") == "interim":
                        logger.info(f"  Interim: {data.get('text', '')[:50]}...")
                    elif data.get("type") == "final":
                        logger.info(f"  Final: {data.get('text', '')}")
                    elif data.get("type") == "done":
                        logger.info(f"  Done! Text: {data.get('text', '')[:100]}")
                        break
                    elif data.get("type") == "error":
                        logger.error(f"  Error: {data.get('message')}")
                        return False

                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for results")
                    break

            print("\n[PASS] IndicASR STT test passed")
            return True

    except Exception as e:
        logger.error(f"STT test failed: {e}")
        print(f"\n[FAIL] IndicASR STT test failed: {e}")
        return False


async def test_llm(
    base_url: str = "http://gpt-oss-120b/v1",
    model: str = "openai/gpt-oss-120b",
    api_key: str = "not-needed",
) -> bool:
    """
    Test vLLM LLM service.

    Args:
        base_url: Base URL for vLLM API
        model: Model name
        api_key: API key (usually not needed for vLLM)

    Returns:
        True if test passed, False otherwise
    """
    print("\n" + "=" * 60)
    print("Testing vLLM LLM Service")
    print("=" * 60)

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you? Reply in 1 sentence."}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }

    try:
        logger.info(f"Connecting to {url}")
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            async with session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                latency = time.time() - start_time

                if response.status != 200:
                    error = await response.text()
                    logger.error(f"LLM request failed: {response.status} - {error}")
                    print(f"\n[FAIL] vLLM test failed: HTTP {response.status}")
                    return False

                data = await response.json()

                # Extract response
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0].get("message", {}).get("content", "")
                    logger.info(f"Response: {content}")
                    logger.info(f"Latency: {latency:.2f}s")

                    # Check usage
                    usage = data.get("usage", {})
                    logger.info(f"Tokens: {usage.get('total_tokens', 'N/A')}")

                    print("\n[PASS] vLLM LLM test passed")
                    return True
                else:
                    logger.error(f"Unexpected response format: {data}")
                    print("\n[FAIL] vLLM test failed: unexpected response")
                    return False

    except Exception as e:
        logger.error(f"LLM test failed: {e}")
        print(f"\n[FAIL] vLLM test failed: {e}")
        return False


async def test_tts(
    base_url: str = "http://localhost:8080",
    voice: str = "hi_male",
    text: str = "नमस्ते, मैं इडली हूं।",
    output_path: str = "test_tts_output.wav",
) -> bool:
    """
    Test Svara TTS service.

    Args:
        base_url: Base URL for Svara TTS
        voice: Voice ID
        text: Text to synthesize
        output_path: Path to save output audio

    Returns:
        True if test passed, False otherwise
    """
    print("\n" + "=" * 60)
    print("Testing Svara TTS Service")
    print("=" * 60)

    # First check health
    health_url = f"{base_url}/health"
    tts_url = f"{base_url}/v1/audio/text-to-speech"

    try:
        async with aiohttp.ClientSession() as session:
            # Health check
            logger.info(f"Checking health at {health_url}")
            async with session.get(health_url) as response:
                if response.status != 200:
                    logger.error(f"Health check failed: {response.status}")
                    print(f"\n[FAIL] Svara TTS health check failed")
                    return False

                health = await response.json()
                logger.info(f"Health: {health.get('status')}")
                logger.info(f"Model: {health.get('model', 'N/A')}")

            # List voices
            voices_url = f"{base_url}/v1/audio/text-to-speech/voices"
            logger.info(f"Listing voices at {voices_url}")
            async with session.get(voices_url) as response:
                if response.status == 200:
                    voices = await response.json()
                    voice_list = voices.get("voices", [])
                    logger.info(f"Available voices: {len(voice_list)}")
                    for v in voice_list[:5]:
                        logger.info(f"  - {v.get('voice_id', v)}: {v.get('name', 'N/A')}")

            # Synthesize text
            logger.info(f"Synthesizing: '{text}'")
            payload = {
                "prompt": text,
                "voice": voice,
                "temperature": 0.75,
                "top_p": 0.9,
                "max_tokens": 1500,
                "repetition_penalty": 1.1,
            }

            start_time = time.time()
            async with session.post(tts_url, json=payload) as response:
                latency = time.time() - start_time

                if response.status != 200:
                    error = await response.text()
                    logger.error(f"TTS failed: {response.status} - {error}")
                    print(f"\n[FAIL] Svara TTS test failed: HTTP {response.status}")
                    return False

                audio_data = await response.read()
                audio_duration = (len(audio_data) - 44) / 2 / 24000  # Rough estimate

                logger.info(f"Received {len(audio_data)} bytes of audio")
                logger.info(f"Latency: {latency:.2f}s")
                logger.info(f"Estimated duration: {audio_duration:.2f}s")

                # Save output
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                logger.info(f"Saved to {output_path}")

                print("\n[PASS] Svara TTS test passed")
                return True

    except Exception as e:
        logger.error(f"TTS test failed: {e}")
        print(f"\n[FAIL] Svara TTS test failed: {e}")
        return False


async def run_all_tests(args):
    """Run all component tests."""
    results = {}

    if args.all or args.stt:
        results["STT"] = await test_stt(
            ws_url=args.stt_url,
            audio_path=args.audio,
            language=args.language,
        )

    if args.all or args.llm:
        results["LLM"] = await test_llm(
            base_url=args.llm_url,
            model=args.llm_model,
        )

    if args.all or args.tts:
        results["TTS"] = await test_tts(
            base_url=args.tts_url,
            voice=args.voice,
            text=args.text or "नमस्ते, मैं इडली हूं। आप कैसे हैं?",
        )

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for component, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {component}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed! Ready for pipeline integration.")
        return 0
    else:
        print("\nSome tests failed. Please check service connectivity.")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Test MiraVoiceAI Pipecat components")

    # Component selection
    parser.add_argument("--all", action="store_true", default=True,
                        help="Test all components (default)")
    parser.add_argument("--stt", action="store_true", help="Test STT only")
    parser.add_argument("--llm", action="store_true", help="Test LLM only")
    parser.add_argument("--tts", action="store_true", help="Test TTS only")

    # URLs
    parser.add_argument("--stt-url", default="ws://localhost:8082/v1/audio/speech-to-text/stream",
                        help="IndicASR WebSocket URL")
    parser.add_argument("--llm-url", default="http://gpt-oss-120b/v1",
                        help="vLLM API base URL")
    parser.add_argument("--tts-url", default="http://localhost:8080",
                        help="Svara TTS base URL")

    # Options
    parser.add_argument("--audio", help="Audio file for STT test")
    parser.add_argument("--language", default="auto", help="STT language")
    parser.add_argument("--voice", default="hi_male", help="TTS voice")
    parser.add_argument("--text", help="Text for TTS test")
    parser.add_argument("--llm-model", default="openai/gpt-oss-120b",
                        help="LLM model name")

    args = parser.parse_args()

    # If specific component selected, disable all
    if args.stt or args.llm or args.tts:
        args.all = False

    return asyncio.run(run_all_tests(args))


if __name__ == "__main__":
    sys.exit(main())
