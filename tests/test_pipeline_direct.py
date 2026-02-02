#!/usr/bin/env python3
"""
Direct Pipeline Test for IdliDemo Pipecat.

This script tests the STT -> LLM -> TTS pipeline directly without going through
the WebSocket transport. This allows testing the core pipeline logic.

Usage:
    python test_pipeline_direct.py --audio sample.wav
    python test_pipeline_direct.py --text "Hello, how are you?"
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
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Configuration
ASR_WS_URL = os.getenv("ASR_WS_URL", "ws://localhost:8082/v1/audio/speech-to-text/stream")
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://localhost:8080")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://gpt-oss-120b/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You are Idli, a helpful AI assistant that speaks Hindi and English.
Keep your responses concise and conversational - typically 1-3 sentences.
You can understand and respond in multiple Indian languages.
Be friendly, helpful, and natural in conversation."""


async def transcribe_audio(audio_path: str, language: str = "auto") -> str:
    """
    Transcribe audio using IndicASR WebSocket API.

    Args:
        audio_path: Path to audio file
        language: Language code or "auto"

    Returns:
        Transcribed text
    """
    logger.info(f"Step 1: Transcribing audio from {audio_path}")

    # Read audio file
    with wave.open(audio_path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)

    # Convert to mono if needed
    if wf.getnchannels() == 2:
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
        audio_data = audio_array.tobytes()

    audio_duration = len(audio_data) / 2 / sample_rate
    logger.info(f"  Audio duration: {audio_duration:.2f}s, sample rate: {sample_rate}Hz")

    start_time = time.time()

    try:
        async with websockets.connect(ASR_WS_URL, max_size=10 * 1024 * 1024) as ws:
            # Send config
            config = {
                "language": language,
                "interim_results": False,
                "sample_rate": sample_rate,
            }
            await ws.send(json.dumps(config))

            # Wait for ack
            ack = await asyncio.wait_for(ws.recv(), timeout=10.0)
            ack_data = json.loads(ack)
            if ack_data.get("type") != "config_ack":
                raise RuntimeError(f"Unexpected response: {ack_data}")

            session_id = ack_data.get("session_id")
            logger.info(f"  Connected to IndicASR, session: {session_id[:8]}...")

            # Send audio in chunks
            chunk_size = int(sample_rate * 0.1 * 2)  # 100ms chunks
            offset = 0
            while offset < len(audio_data):
                chunk = audio_data[offset:offset + chunk_size]
                await ws.send(chunk)
                offset += chunk_size
                await asyncio.sleep(0.02)

            # Send END
            await ws.send("END")

            # Collect final result
            final_text = ""
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    data = json.loads(msg)

                    if data.get("type") == "final":
                        final_text = data.get("text", "")
                    elif data.get("type") == "done":
                        final_text = data.get("text", final_text)
                        break
                    elif data.get("type") == "error":
                        raise RuntimeError(data.get("message"))

                except asyncio.TimeoutError:
                    break

            latency = time.time() - start_time
            logger.info(f"  Transcription: '{final_text}'")
            logger.info(f"  Latency: {latency:.2f}s")

            return final_text

    except Exception as e:
        logger.error(f"  Transcription failed: {e}")
        raise


async def generate_response(user_text: str) -> str:
    """
    Generate LLM response using vLLM.

    Args:
        user_text: User's transcribed text

    Returns:
        LLM response text
    """
    logger.info(f"Step 2: Generating LLM response for: '{user_text}'")

    url = f"{LLM_BASE_URL}/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ],
        "max_tokens": 200,
        "temperature": 0.7,
    }

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error = await response.text()
                raise RuntimeError(f"LLM request failed: {response.status} - {error}")

            data = await response.json()

            if "choices" not in data or len(data["choices"]) == 0:
                raise RuntimeError(f"Unexpected LLM response: {data}")

            response_text = data["choices"][0]["message"]["content"]
            latency = time.time() - start_time

            logger.info(f"  Response: '{response_text}'")
            logger.info(f"  Latency: {latency:.2f}s")

            return response_text


async def synthesize_speech(text: str, voice: str = "hi_male", output_path: str = "output.wav") -> str:
    """
    Synthesize speech using Svara TTS.

    Args:
        text: Text to synthesize
        voice: Voice ID
        output_path: Output file path

    Returns:
        Output file path
    """
    logger.info(f"Step 3: Synthesizing speech: '{text}'")

    url = f"{TTS_BASE_URL}/v1/audio/text-to-speech/"
    payload = {
        "prompt": text,
        "voice": voice,
        "temperature": 0.75,
        "top_p": 0.9,
        "max_tokens": 1500,
        "repetition_penalty": 1.1,
    }

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error = await response.text()
                raise RuntimeError(f"TTS request failed: {response.status} - {error}")

            audio_data = await response.read()
            latency = time.time() - start_time

            # Estimate audio duration (rough)
            audio_duration = (len(audio_data) - 44) / 2 / 24000

            # Save to file
            with open(output_path, "wb") as f:
                f.write(audio_data)

            logger.info(f"  Audio duration: ~{audio_duration:.2f}s")
            logger.info(f"  Saved to: {output_path}")
            logger.info(f"  Latency: {latency:.2f}s")

            return output_path


async def run_pipeline(
    audio_path: str = None,
    text: str = None,
    voice: str = "hi_male",
    language: str = "auto",
    output_path: str = "pipeline_output.wav",
):
    """
    Run the full STT -> LLM -> TTS pipeline.

    Args:
        audio_path: Input audio file (for full pipeline)
        text: Direct text input (skips STT)
        voice: TTS voice ID
        language: STT language code
        output_path: Output audio file path
    """
    print("\n" + "=" * 60)
    print("IdliDemo Pipecat - Direct Pipeline Test")
    print("=" * 60)

    total_start = time.time()

    # Step 1: STT (if audio provided)
    if audio_path:
        user_text = await transcribe_audio(audio_path, language)
    else:
        user_text = text
        logger.info(f"Step 1: Using direct text input: '{user_text}'")

    if not user_text:
        logger.error("No text to process!")
        return

    # Step 2: LLM
    response_text = await generate_response(user_text)

    # Step 3: TTS
    output = await synthesize_speech(response_text, voice, output_path)

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    print(f"  User said: {user_text}")
    print(f"  Bot replied: {response_text}")
    print(f"  Output audio: {output}")
    print(f"  Total pipeline time: {total_time:.2f}s")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Direct pipeline test for IdliDemo Pipecat")
    parser.add_argument("--audio", help="Input audio file (WAV format)")
    parser.add_argument("--text", help="Direct text input (skips STT)")
    parser.add_argument("--voice", default="hi_male", help="TTS voice ID")
    parser.add_argument("--language", default="auto", help="STT language code")
    parser.add_argument("--output", default="pipeline_output.wav", help="Output audio file")

    args = parser.parse_args()

    if not args.audio and not args.text:
        # Default test with Hindi text
        args.text = "नमस्ते, आप कैसे हैं?"
        logger.info("No input provided, using default Hindi text")

    if args.audio and not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        sys.exit(1)

    asyncio.run(run_pipeline(
        audio_path=args.audio,
        text=args.text,
        voice=args.voice,
        language=args.language,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
