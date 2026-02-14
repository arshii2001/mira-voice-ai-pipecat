#!/usr/bin/env python3
"""
Mira Voice Eval — End-to-End Audio Pipeline Evaluation (v2).

Sends real audio through the full Pipecat pipeline (WebSocket protobuf):
  Audio In → STT → LLM → TTS → Audio Out

Reliability improvements over v1:
  - Captures user_transcript JSON messages for STT accuracy (not just protobuf)
  - Properly waits for greeting to finish (audio drain + bot_audio_end sentinel)
  - Reuses a single WebSocket for multi-turn conversation (like a real user)
  - 15+ test cases for statistical significance
  - Validates response relevance (keyword matching)
  - Records all raw JSON/protobuf messages for debugging

Measures:
  - STT accuracy (what Mira heard vs what we said)
  - Time to First Audio Byte (TTFAB) — from end-of-speech to first bot audio
  - Total round-trip time
  - Response quality (GPT-4o judge, same dimensions as eval_mira.py)
  - Response relevance (keyword match)
  - Response audio duration

Usage:
    # Against OSS production
    docker compose run --rm test-voice tests/eval_voice.py --target oss

    # Against ElevenLabs production
    docker compose run --rm test-voice tests/eval_voice.py --target elevenlabs

    # Custom URL
    PIPECAT_WS_URL=wss://mira-oss.inf7ks8.com/pipecat/ws python tests/eval_voice.py
"""

import argparse
import asyncio
import io
import json
import logging
import math
import os
import struct
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import httpx

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")

try:
    import jwt as pyjwt
except ImportError:
    pyjwt = None

# Pipecat protobuf frames
try:
    import pipecat.frames.protobufs.frames_pb2 as frame_protos
    HAS_PROTO = True
except ImportError:
    HAS_PROTO = False
    print("⚠  pipecat protobuf not available — cannot run voice eval")
    sys.exit(1)

# ─────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────
TARGETS = {
    "local": {
        "description": "Local Docker stack",
        "ws_url": "ws://localhost:7860/ws",
        "http_url": "http://localhost:7860",
    },
    "oss": {
        "description": "OSS production (Svara TTS + OSS LLM)",
        "ws_url": "wss://mira-oss.inf7ks8.com/pipecat/ws",
        "http_url": "https://mira-oss.inf7ks8.com/pipecat",
    },
    "elevenlabs": {
        "description": "ElevenLabs production (GPT-4o-mini + ElevenLabs TTS)",
        "ws_url": "wss://mira-ai.westus2.cloudapp.azure.com/pipecat/ws",
        "http_url": "https://mira-ai.westus2.cloudapp.azure.com/pipecat",
    },
}

WS_URL = os.getenv("PIPECAT_WS_URL", "ws://localhost:7860/ws")
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
CHUNK_DURATION_MS = 100

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")
JUDGE_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1")
WEBUI_SECRET_KEY = os.getenv("WEBUI_SECRET_KEY", "").strip() or None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voice-eval")


# ─────────────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────────────
def _make_jwt(user_id: str = "eval-voice-user") -> str:
    if not WEBUI_SECRET_KEY or not pyjwt:
        return ""
    payload = {
        "id": user_id,
        "email": f"{user_id}@example.test",
        "exp": int(time.time()) + 7200,
    }
    return pyjwt.encode(payload, WEBUI_SECRET_KEY, algorithm="HS256")


# ─────────────────────────────────────────────────
# Protobuf helpers
# ─────────────────────────────────────────────────
def make_audio_frame(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw PCM16 audio bytes in a Pipecat protobuf Frame."""
    frame = frame_protos.Frame()
    frame.audio.audio = pcm_bytes
    frame.audio.sample_rate = sample_rate
    frame.audio.num_channels = NUM_CHANNELS
    return frame.SerializeToString()


def parse_frame(data: bytes) -> dict:
    """Parse a binary protobuf message from the server."""
    try:
        proto = frame_protos.Frame.FromString(data)
        which = proto.WhichOneof("frame")
        if which == "audio":
            return {"type": "audio", "data": proto.audio.audio, "length": len(proto.audio.audio)}
        elif which == "text":
            return {"type": "text", "text": proto.text.text}
        elif which == "transcription":
            return {"type": "transcription", "text": proto.transcription.text}
        elif which == "message":
            try:
                return {"type": "message", "data": json.loads(proto.message.message)}
            except Exception:
                return {"type": "message", "data": proto.message.message}
        return {"type": "unknown", "which": which}
    except Exception as e:
        return {"type": "parse_error", "error": str(e)}


def generate_silence(duration_sec: float = 1.5) -> bytes:
    """Generate silence as PCM16 bytes."""
    return np.zeros(int(SAMPLE_RATE * duration_sec), dtype=np.int16).tobytes()


# ─────────────────────────────────────────────────
# TTS-based audio generation for test prompts
# ─────────────────────────────────────────────────
def text_to_pcm16(text: str, lang: str = "en") -> bytes:
    """Convert text to PCM16 16kHz mono audio using gTTS."""
    try:
        from gtts import gTTS
        from pydub import AudioSegment

        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_buf = io.BytesIO()
        tts.write_to_fp(mp3_buf)
        mp3_buf.seek(0)

        # Convert MP3 → PCM16 16kHz mono
        audio = AudioSegment.from_mp3(mp3_buf)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        return audio.raw_data

    except ImportError:
        logger.warning("gTTS/pydub not available — using pre-recorded WAV files only")
        return None


def load_wav_file(path: str) -> Tuple[bytes, int]:
    """Load a WAV file and return (pcm_bytes, sample_rate)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        data = wf.readframes(n)
        if wf.getnchannels() == 2:
            arr = np.frombuffer(data, dtype=np.int16).reshape(-1, 2).mean(axis=1).astype(np.int16)
            data = arr.tobytes()
        return data, sr


# ─────────────────────────────────────────────────
# Test cases — 15+ for statistical significance
# ─────────────────────────────────────────────────
@dataclass
class VoiceTestCase:
    id: str
    label: str
    spoken_text: str           # What the student says
    lang: str = "en"           # Language for TTS generation
    wav_file: Optional[str] = None  # Pre-recorded WAV (overrides TTS)
    user_name: Optional[str] = None
    topic: Optional[str] = None
    category: str = "general"
    expected_transcription: Optional[str] = None  # For STT accuracy check
    relevance_keywords: list = field(default_factory=list)  # Keywords expected in response


VOICE_TEST_CASES = [
    # ── English questions ──
    VoiceTestCase(
        "en_greeting", "English greeting",
        "Hello, can you help me with my science homework?",
        lang="en", user_name="Aarav", topic="Science", category="english",
        expected_transcription="hello can you help me with my science homework",
        relevance_keywords=["help", "science", "homework"],
    ),
    VoiceTestCase(
        "en_photosynthesis", "English science question",
        "How do plants make food from sunlight?",
        lang="en", user_name="Meera", topic="Photosynthesis", category="english",
        expected_transcription="how do plants make food from sunlight",
        relevance_keywords=["photosynthesis", "sunlight", "plant", "food", "leaf", "leaves", "chlorophyll"],
    ),
    VoiceTestCase(
        "en_fractions", "English math question",
        "I don't understand fractions, can you explain?",
        lang="en", user_name="Ravi", topic="Fractions", category="english",
        expected_transcription="i don't understand fractions can you explain",
        relevance_keywords=["fraction", "part", "whole", "numerator", "denominator", "pizza", "half"],
    ),
    VoiceTestCase(
        "en_excited", "English excited student",
        "I got full marks in my test today!",
        lang="en", user_name="Ananya", topic="Science", category="eq",
        expected_transcription="i got full marks in my test today",
        relevance_keywords=["congrat", "amazing", "proud", "awesome", "great", "well done", "fantastic"],
    ),
    VoiceTestCase(
        "en_confused", "English confused student",
        "I'm really confused about gravity, why do things fall down?",
        lang="en", user_name="Rohan", topic="Physics", category="english",
        expected_transcription="i'm really confused about gravity why do things fall down",
        relevance_keywords=["gravity", "earth", "pull", "force", "newton", "fall"],
    ),
    VoiceTestCase(
        "en_history", "English history question",
        "Who was Mahatma Gandhi and why is he important?",
        lang="en", user_name="Priya", topic="History", category="english",
        expected_transcription="who was mahatma gandhi and why is he important",
        relevance_keywords=["gandhi", "india", "freedom", "independence", "nonviolence", "peace"],
    ),
    VoiceTestCase(
        "en_math_area", "English math area question",
        "How do I find the area of a triangle?",
        lang="en", user_name="Vikram", topic="Geometry", category="english",
        expected_transcription="how do i find the area of a triangle",
        relevance_keywords=["triangle", "area", "base", "height", "half", "formula"],
    ),
    VoiceTestCase(
        "en_water_cycle", "English water cycle",
        "Can you explain the water cycle?",
        lang="en", user_name="Diya", topic="Science", category="english",
        expected_transcription="can you explain the water cycle",
        relevance_keywords=["water", "evaporation", "condensation", "rain", "cloud", "cycle"],
    ),
    VoiceTestCase(
        "en_sad_student", "English sad student",
        "I failed my exam and I feel really bad about it.",
        lang="en", user_name="Arjun", topic="General", category="eq",
        expected_transcription="i failed my exam and i feel really bad about it",
        relevance_keywords=["okay", "try", "learn", "next", "mistake", "improve", "happen"],
    ),
    VoiceTestCase(
        "en_solar_system", "English solar system",
        "How many planets are there in our solar system?",
        lang="en", user_name="Neha", topic="Science", category="english",
        expected_transcription="how many planets are there in our solar system",
        relevance_keywords=["planet", "eight", "solar", "mercury", "venus", "earth", "mars", "jupiter"],
    ),
    VoiceTestCase(
        "en_decimal", "English decimal question",
        "What is the difference between a fraction and a decimal?",
        lang="en", user_name="Karan", topic="Math", category="english",
        expected_transcription="what is the difference between a fraction and a decimal",
        relevance_keywords=["fraction", "decimal", "point", "part", "number"],
    ),

    # ── English LONG-RESPONSE questions (force 4+ sentences) ──
    VoiceTestCase(
        "en_long_photosynthesis", "Long: explain photosynthesis with analogy",
        "Can you explain photosynthesis to me with a fun analogy?",
        lang="en", user_name="Meera", topic="Photosynthesis", category="english_long",
        expected_transcription="can you explain photosynthesis to me with a fun analogy",
        relevance_keywords=["photosynthesis", "sunlight", "plant", "food"],
    ),
    VoiceTestCase(
        "en_long_digestive", "Long: digestive system step by step",
        "What happens to food after I eat it? Tell me step by step.",
        lang="en", user_name="Aarav", topic="Digestive System", category="english_long",
        expected_transcription="what happens to food after i eat it tell me step by step",
        relevance_keywords=["stomach", "digest", "food", "intestine", "mouth", "enzyme"],
    ),
    VoiceTestCase(
        "en_long_water_cycle", "Long: water cycle detailed",
        "Explain the complete water cycle from the ocean to rain and back again.",
        lang="en", user_name="Diya", topic="Water Cycle", category="english_long",
        expected_transcription="explain the complete water cycle from the ocean to rain and back again",
        relevance_keywords=["evaporation", "condensation", "rain", "cloud", "ocean", "water"],
    ),
    VoiceTestCase(
        "en_long_gravity", "Long: gravity with examples",
        "Why does gravity exist and what would happen if there was no gravity on Earth?",
        lang="en", user_name="Rohan", topic="Physics", category="english_long",
        expected_transcription="why does gravity exist and what would happen if there was no gravity on earth",
        relevance_keywords=["gravity", "earth", "float", "pull", "force", "fall"],
    ),
    VoiceTestCase(
        "en_long_fractions", "Long: fractions with pizza example",
        "I still don't get fractions. Can you explain with a real life example like pizza or cake?",
        lang="en", user_name="Ravi", topic="Fractions", category="english_long",
        expected_transcription="i still don't get fractions can you explain with a real life example like pizza or cake",
        relevance_keywords=["fraction", "pizza", "cake", "piece", "part", "whole", "half"],
    ),

    # ── Hindi questions (gTTS) ──
    VoiceTestCase(
        "hi_science", "Hindi science question",
        "पानी का सूत्र क्या है?",
        lang="hi", user_name="Aarti", topic="Chemistry", category="hindi",
        expected_transcription="pani ka sutra kya hai",
        relevance_keywords=["H2O", "पानी", "water", "hydrogen", "oxygen", "हाइड्रोजन", "ऑक्सीजन"],
    ),
    VoiceTestCase(
        "hi_math", "Hindi math question",
        "दो और तीन का गुणा कितना होता है?",
        lang="hi", user_name="Suresh", topic="Math", category="hindi",
        expected_transcription="do aur teen ka guna kitna hota hai",
        relevance_keywords=["6", "छह", "गुणा", "multiply", "six"],
    ),

    # ── Hindi pre-recorded WAVs ──
    VoiceTestCase(
        "hi_greeting_wav", "Hindi greeting (pre-recorded)",
        "नमस्ते, मुझे भारत के बारे में बताइए",
        lang="hi", wav_file="user_greeting.wav", category="hindi",
        expected_transcription="namaste mujhe bharat ke baare mein bataiye",
        relevance_keywords=["india", "भारत", "country", "देश"],
    ),
    VoiceTestCase(
        "hi_short_wav", "Hindi short response (pre-recorded)",
        "हाँ",
        lang="hi", wav_file="user_short.wav", category="hindi",
        expected_transcription="haan",
        relevance_keywords=[],  # "yes" is too short for keyword matching
    ),

    # ── English pre-recorded WAVs ──
    VoiceTestCase(
        "en_weather_wav", "English pre-recorded weather",
        "Hello, tell me about the weather today",
        lang="en", wav_file="user_english.wav", category="english",
        expected_transcription="hello tell me about the weather today",
        relevance_keywords=["weather"],
    ),
]


# ─────────────────────────────────────────────────
# Judge (same as eval_mira.py)
# ─────────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for an AI tutor called Mira, designed for Indian students in Grades 5-8.

You will be given:
- The student's message (with optional context: name, topic, language)
- Mira's response

Score Mira's response on these 6 dimensions (1-5 scale each):

1. **FRIENDLY_WARM** — Does Mira sound like a warm, fun older sister? Not robotic, not textbook-ish?
   - 1: Cold/robotic  2: Polite but distant  3: Friendly enough  4: Warm and engaging  5: Genuinely delightful

2. **EDUCATIONAL_ENCOURAGING** — Does Mira encourage learning and curiosity?
   - 1: Discouraging  2: Neutral  3: Somewhat encouraging  4: Clearly encouraging  5: Makes learning exciting

3. **INDIAN_CULTURAL** — Does Mira use Indian cultural references?
   - 1: No Indian references  2: Generic global  3: One vague reference  4: Good Indian examples  5: Deeply Indian

4. **EMOTIONAL_INTELLIGENCE** — Does Mira read the student's emotional state correctly?
   - 1: Ignores emotions  2: Acknowledges but moves on  3: Decent read  4: Good attunement  5: Perfect EQ

5. **SOCRATIC_TEACHING** — Does Mira guide rather than dump answers?
   - 1: Dumps answer  2: Explains then token question  3: Some guiding  4: Good Socratic  5: Masterful

6. **BREVITY_VOICE_READY** — Is the response concise enough for voice? Would it sound natural read aloud?
   - 1: Way too long  2: Too long  3: Acceptable  4: Good brevity  5: Perfect for voice

Return ONLY a JSON object:
{"friendly_warm": N, "educational_encouraging": N, "indian_cultural": N, "emotional_intelligence": N, "socratic_teaching": N, "brevity_voice_ready": N, "overall": N, "one_line_feedback": "..."}
"""


def judge_response(test: VoiceTestCase, mira_response: str) -> dict:
    """Use GPT-4o to judge Mira's response."""
    user_context = f"Student name: {test.user_name or 'unknown'}"
    if test.topic:
        user_context += f"\nTopic: {test.topic}"
    user_context += f"\nCategory: {test.category}"
    user_context += f"\nNote: This was a VOICE interaction — student spoke aloud and heard the response."

    judge_prompt = f"""{user_context}

Student message (spoken): {test.spoken_text}

Mira's response: {mira_response}

Score this response."""

    for attempt in range(3):
        try:
            client = httpx.Client(timeout=90)
            r = client.post(
                f"{JUDGE_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {JUDGE_API_KEY}"},
                json={
                    "model": JUDGE_MODEL,
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": judge_prompt},
                    ],
                    "temperature": 0.0,
                },
            )
            data = r.json()
            raw = data["choices"][0]["message"]["content"].strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(raw)
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return {
                    "friendly_warm": 0, "educational_encouraging": 0,
                    "indian_cultural": 0, "emotional_intelligence": 0,
                    "socratic_teaching": 0, "brevity_voice_ready": 0,
                    "overall": 0, "one_line_feedback": f"JUDGE ERROR: {str(e)[:60]}",
                }


# ─────────────────────────────────────────────────
# Persistent WebSocket client (reuses connection)
# ─────────────────────────────────────────────────
class VoiceEvalClient:
    """
    Persistent WebSocket client that reuses a single connection
    for multiple test turns, simulating a real user session.
    """

    def __init__(self, ws_url: str, audio_dir: str):
        self.ws_url = ws_url
        self.audio_dir = audio_dir
        self.ws = None
        self.connected = False
        self.turn_count = 0

    async def connect(self, lang: str = "en"):
        """Connect to the Pipecat pipeline and wait for greeting to finish."""
        logger.info(f"Connecting to {self.ws_url}")
        self.ws = await asyncio.wait_for(
            websockets.connect(self.ws_url, max_size=10 * 1024 * 1024),
            timeout=15.0,
        )

        # Send config — enable greeting on first connect
        config = {
            "type": "config",
            "mode": "text_and_audio",
            "language": lang,
            "enable_greeting": True,
        }
        token = _make_jwt()
        if token:
            config["token"] = token
        await self.ws.send(json.dumps(config))
        logger.info(f"  Config sent (lang={lang}, greeting=True, jwt={'yes' if token else 'no'})")

        # Wait for greeting to fully complete
        await self._drain_greeting()
        self.connected = True
        self.turn_count = 0

    async def _drain_greeting(self):
        """
        Wait for the greeting to fully complete.

        The server sends:
          1. session_id JSON
          2. bot_text chunks (streaming) via protobuf or JSON
          3. Audio frames (protobuf)
          4. bot_text_complete JSON
          5. More audio frames
          6. Eventually silence / no more frames

        We wait for bot_text_complete AND then drain remaining audio
        until we get 2 consecutive recv timeouts (= audio stream ended).
        """
        got_session_id = False
        got_text_complete = False
        greeting_text = ""
        greeting_audio_chunks = 0
        greeting_audio_bytes = 0
        consecutive_timeouts = 0
        deadline = time.time() + 25.0

        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=2.0)
                consecutive_timeouts = 0
            except asyncio.TimeoutError:
                consecutive_timeouts += 1
                # After text_complete, 2 timeouts = audio stream ended
                if got_text_complete and consecutive_timeouts >= 2:
                    break
                # Without text_complete, 5 timeouts = pipeline probably didn't greet
                if not got_text_complete and consecutive_timeouts >= 5:
                    break
                continue
            except websockets.ConnectionClosed:
                logger.warning("  Connection closed during greeting")
                break

            if isinstance(msg, bytes):
                parsed = parse_frame(msg)
                if parsed["type"] == "audio":
                    greeting_audio_chunks += 1
                    greeting_audio_bytes += parsed["length"]
            else:
                try:
                    data = json.loads(msg)
                    msg_type = data.get("type", "")
                    if msg_type == "session_id":
                        got_session_id = True
                        logger.info(f"  Session: {data.get('session_id', '?')[:8]}...")
                    elif msg_type == "bot_text_complete":
                        got_text_complete = True
                        greeting_text = data.get("text", "")
                        logger.info(f"  Greeting text: \"{greeting_text[:60]}...\"")
                    elif msg_type == "bot_text":
                        pass  # Streaming token — ignore
                except json.JSONDecodeError:
                    pass

        greeting_audio_sec = round(greeting_audio_bytes / (24000 * 2), 1) if greeting_audio_bytes > 0 else 0
        logger.info(
            f"  Greeting complete: text={'✓' if got_text_complete else '✗'} | "
            f"audio={greeting_audio_chunks} chunks ({greeting_audio_sec}s) | "
            f"session={'✓' if got_session_id else '✗'}"
        )

    async def send_turn(self, test: VoiceTestCase) -> dict:
        """
        Send one student turn and collect the full response.

        Returns dict with:
          - transcription: what STT heard (from user_transcript JSON)
          - bot_text: Mira's full response text
          - bot_audio_bytes: total audio bytes received
          - ttfab_ms: time to first audio byte
          - total_rt_ms: total round-trip
          - all_json_messages: raw JSON messages for debugging
        """
        self.turn_count += 1

        # ── 1. Get audio ──
        pcm_audio = None
        if test.wav_file:
            wav_path = os.path.join(self.audio_dir, test.wav_file)
            if os.path.exists(wav_path):
                pcm_audio, sr = load_wav_file(wav_path)
                logger.info(f"  Loaded WAV: {wav_path} ({len(pcm_audio)/(sr*2):.1f}s)")

        if pcm_audio is None:
            pcm_audio = text_to_pcm16(test.spoken_text, lang=test.lang)
            if pcm_audio is None:
                return {"error": "No audio source (no WAV file and gTTS not available)"}
            logger.info(f"  Generated TTS audio: {len(pcm_audio)/(SAMPLE_RATE*2):.1f}s")

        silence = generate_silence(duration_sec=1.5)

        # ── 2. Send student audio ──
        chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
        chunk_bytes = chunk_samples * 2

        logger.info(f"  Sending student audio ({len(pcm_audio)/(SAMPLE_RATE*2):.1f}s)...")
        send_start = time.time()
        offset = 0
        while offset < len(pcm_audio):
            chunk = pcm_audio[offset:offset + chunk_bytes]
            proto_data = make_audio_frame(chunk)
            await self.ws.send(proto_data)
            offset += chunk_bytes
            await asyncio.sleep(CHUNK_DURATION_MS / 1000.0 * 0.8)

        # Send trailing silence for VAD to detect end-of-speech
        offset = 0
        while offset < len(silence):
            chunk = silence[offset:offset + chunk_bytes]
            proto_data = make_audio_frame(chunk)
            await self.ws.send(proto_data)
            offset += chunk_bytes
            await asyncio.sleep(CHUNK_DURATION_MS / 1000.0 * 0.8)

        speech_done_time = time.time()
        logger.info(f"  Audio sent in {(speech_done_time - send_start)*1000:.0f}ms")

        # ── 3. Receive response ──
        first_audio_at = None
        last_audio_at = None
        bot_audio_bytes = 0
        bot_audio_chunks = 0
        bot_audio_raw = bytearray()   # Accumulate raw PCM for WAV saving
        transcriptions = []           # From user_transcript JSON messages
        proto_transcriptions = []     # From protobuf transcription frames
        bot_text_tokens = []          # Streaming bot_text tokens
        bot_text_complete = None       # Final complete text
        all_json_messages = []

        # ── Per-chunk timing for audio segment analysis ──
        audio_chunk_times = []         # (timestamp, chunk_bytes) for each audio chunk
        text_token_times = []          # (timestamp, token_text) for each bot_text token

        response_deadline = time.time() + 50.0
        no_data_count = 0

        while time.time() < response_deadline:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=2.5)
                no_data_count = 0
            except asyncio.TimeoutError:
                no_data_count += 1
                # If we got response text + some audio, we're done
                if (bot_audio_chunks > 5 or bot_text_complete) and no_data_count >= 2:
                    break
                if no_data_count >= 5:
                    break
                continue
            except websockets.ConnectionClosed:
                logger.warning("  Connection closed during response")
                break

            if isinstance(msg, bytes):
                parsed = parse_frame(msg)
                if parsed["type"] == "audio":
                    bot_audio_chunks += 1
                    bot_audio_bytes += parsed["length"]
                    bot_audio_raw.extend(parsed["data"])
                    now = time.time()
                    audio_chunk_times.append((now, parsed["length"]))
                    if first_audio_at is None:
                        first_audio_at = now
                    last_audio_at = now
                elif parsed["type"] == "transcription":
                    proto_transcriptions.append(parsed["text"])
                    logger.info(f"  🎤 STT (proto): '{parsed['text']}'")
                elif parsed["type"] == "text":
                    bot_text_tokens.append(parsed["text"])
            else:
                # JSON text message
                try:
                    data = json.loads(msg)
                    all_json_messages.append(data)
                    msg_type = data.get("type", "")

                    if msg_type == "user_transcript":
                        # This is the key message! STT result sent as JSON
                        text = data.get("text", "")
                        is_final = data.get("final", False)
                        if is_final and text:
                            transcriptions.append(text)
                            logger.info(f"  🎤 STT (JSON): '{text}'")
                        elif text:
                            logger.info(f"  🎤 STT interim: '{text}'")

                    elif msg_type == "bot_text":
                        token = data.get("text", "")
                        bot_text_tokens.append(token)
                        text_token_times.append((time.time(), token))

                    elif msg_type == "bot_text_complete":
                        bot_text_complete = data.get("text", "")
                        logger.info(f"  📝 Bot: '{bot_text_complete[:80]}...'")

                except json.JSONDecodeError:
                    pass

        # ── 4. Compute metrics ──
        # Prefer JSON user_transcript, fall back to protobuf transcription
        stt_text = " ".join(transcriptions) if transcriptions else " ".join(proto_transcriptions)
        bot_text = bot_text_complete or "".join(bot_text_tokens)
        bot_audio_sec = round(bot_audio_bytes / (24000 * 2), 2) if bot_audio_bytes > 0 else 0

        ttfab_ms = round((first_audio_at - speech_done_time) * 1000) if first_audio_at else -1
        total_rt_ms = round((last_audio_at - send_start) * 1000) if last_audio_at else -1

        # ── 4b. Audio segment analysis (detect sentence boundaries by gaps) ──
        audio_segments = []
        if len(audio_chunk_times) >= 2:
            GAP_THRESHOLD_SEC = 0.15  # 150ms gap = new segment (sentence boundary)
            seg_start_t = audio_chunk_times[0][0]
            seg_chunks = 1
            seg_bytes = audio_chunk_times[0][1]
            for i in range(1, len(audio_chunk_times)):
                gap = audio_chunk_times[i][0] - audio_chunk_times[i - 1][0]
                if gap > GAP_THRESHOLD_SEC:
                    # End of segment
                    seg_dur = audio_chunk_times[i - 1][0] - seg_start_t
                    seg_audio_dur = seg_bytes / (24000 * 2)
                    audio_segments.append({
                        "seg": len(audio_segments) + 1,
                        "chunks": seg_chunks,
                        "bytes": seg_bytes,
                        "audio_sec": round(seg_audio_dur, 2),
                        "recv_sec": round(seg_dur, 2),
                        "gap_after_ms": round(gap * 1000),
                    })
                    seg_start_t = audio_chunk_times[i][0]
                    seg_chunks = 1
                    seg_bytes = audio_chunk_times[i][1]
                else:
                    seg_chunks += 1
                    seg_bytes += audio_chunk_times[i][1]
            # Final segment
            seg_dur = audio_chunk_times[-1][0] - seg_start_t
            seg_audio_dur = seg_bytes / (24000 * 2)
            audio_segments.append({
                "seg": len(audio_segments) + 1,
                "chunks": seg_chunks,
                "bytes": seg_bytes,
                "audio_sec": round(seg_audio_dur, 2),
                "recv_sec": round(seg_dur, 2),
                "gap_after_ms": 0,
            })
        elif len(audio_chunk_times) == 1:
            audio_segments.append({
                "seg": 1, "chunks": 1, "bytes": audio_chunk_times[0][1],
                "audio_sec": round(audio_chunk_times[0][1] / (24000 * 2), 2),
                "recv_sec": 0, "gap_after_ms": 0,
            })

        # ── 4c. Text token stats ──
        text_first_token_ms = -1
        text_last_token_ms = -1
        text_token_count = len(text_token_times)
        if text_token_times:
            text_first_token_ms = round((text_token_times[0][0] - speech_done_time) * 1000)
            text_last_token_ms = round((text_token_times[-1][0] - speech_done_time) * 1000)

        # ── Print detailed segment stats ──
        if audio_segments:
            logger.info(f"  📊 Audio segments: {len(audio_segments)} | Text tokens: {text_token_count}")
            for seg in audio_segments:
                gap_str = f" → gap {seg['gap_after_ms']}ms" if seg['gap_after_ms'] > 0 else ""
                logger.info(
                    f"     Seg {seg['seg']}: {seg['chunks']} chunks, "
                    f"{seg['audio_sec']}s audio, recv in {seg['recv_sec']}s{gap_str}"
                )
        if text_token_count > 0:
            logger.info(f"  📝 Text: first token at {text_first_token_ms}ms, last at {text_last_token_ms}ms ({text_token_count} tokens)")

        logger.info(
            f"  ✅ TTFAB={ttfab_ms}ms | RT={total_rt_ms}ms | "
            f"Audio={bot_audio_sec}s ({bot_audio_chunks} chunks) | "
            f"Segments={len(audio_segments)} | "
            f"STT={'✓' if stt_text else '✗'} | "
            f"BotText={'✓' if bot_text else '✗'}"
        )

        return {
            "transcription": stt_text,
            "bot_text": bot_text,
            "bot_audio_bytes": bot_audio_bytes,
            "bot_audio_chunks": bot_audio_chunks,
            "bot_audio_duration_sec": bot_audio_sec,
            "bot_audio_raw": bytes(bot_audio_raw),
            "audio_segments": audio_segments,
            "text_token_count": text_token_count,
            "text_first_token_ms": text_first_token_ms,
            "text_last_token_ms": text_last_token_ms,
            "ttfab_ms": ttfab_ms,
            "total_rt_ms": total_rt_ms,
            "all_json_messages": all_json_messages,
        }

    async def disconnect(self):
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
            self.connected = False


# ─────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────
@dataclass
class VoiceResult:
    test_id: str
    label: str
    category: str
    spoken_text: str
    turn_number: int = 0
    # STT
    transcription: str = ""
    stt_match: float = 0.0  # 0-1 word overlap ratio
    # Latency
    ttfab_ms: float = 0.0   # Time to first audio byte (from end of speech)
    total_rt_ms: float = 0.0  # Total round-trip
    # Response
    bot_text: str = ""
    bot_audio_bytes: int = 0
    bot_audio_duration_sec: float = 0.0
    # Relevance
    relevance_hit: bool = False  # Did response contain expected keywords?
    relevance_keywords_found: list = field(default_factory=list)
    # Quality scores
    scores: dict = field(default_factory=dict)
    error: Optional[str] = None


def compute_word_overlap(expected: str, actual: str) -> float:
    """Compute word-level overlap ratio between expected and actual transcription."""
    if not expected or not actual:
        return 0.0
    exp_words = set(expected.lower().split())
    act_words = set(actual.lower().split())
    if not exp_words:
        return 0.0
    overlap = exp_words & act_words
    return len(overlap) / len(exp_words)


def check_relevance(response: str, keywords: list) -> Tuple[bool, list]:
    """Check if response contains any of the expected keywords."""
    if not keywords:
        return True, []  # No keywords to check = pass
    response_lower = response.lower()
    found = [kw for kw in keywords if kw.lower() in response_lower]
    return len(found) > 0, found


# ─────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────
def score_bar(score, max_score=5):
    filled = int(score * 4)
    empty = 20 - filled
    return f"[{'█' * filled}{'░' * empty}] {score:.1f}/5"


def print_colored(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m")


def score_color(score):
    if score >= 4.5: return "92"
    if score >= 3.5: return "32"
    if score >= 2.5: return "33"
    return "31"


# ─────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────
async def run_all(ws_url: str, audio_dir: str, target_name: str, run_judge: bool = True, limit: int = None):
    """Run all voice test cases and report results."""
    auth_status = "JWT ✓" if WEBUI_SECRET_KEY else "no auth"

    active_tests = VOICE_TEST_CASES[:limit] if limit else VOICE_TEST_CASES

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       MIRA VOICE EVAL v2 — End-to-End Audio Pipeline       ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Target: {target_name:<50} ║")
    print(f"║  WS:     {ws_url:<50} ║")
    print(f"║  Auth:   {auth_status:<50} ║")
    print(f"║  Judge:  {JUDGE_MODEL if run_judge else 'DISABLED':<50} ║")
    print(f"║  Tests:  {len(active_tests):<50} ║")
    print(f"║  Mode:   {'Single session (multi-turn)':<50} ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    results: List[VoiceResult] = []

    # ── Strategy: Run tests in batches sharing a single WebSocket ──
    # Group by language to avoid language switching mid-session
    en_tests = [t for t in active_tests if t.lang == "en"]
    hi_tests = [t for t in active_tests if t.lang == "hi"]

    for batch_label, batch_tests, batch_lang in [
        ("English", en_tests, "en"),
        ("Hindi", hi_tests, "hi"),
    ]:
        if not batch_tests:
            continue

        print(f"\n{'═' * 60}")
        print(f"  BATCH: {batch_label} ({len(batch_tests)} tests, single session)")
        print(f"{'═' * 60}")

        client = VoiceEvalClient(ws_url, audio_dir)
        try:
            await client.connect(lang=batch_lang)
        except Exception as e:
            logger.error(f"  ❌ Connection failed for {batch_label} batch: {e}")
            for test in batch_tests:
                results.append(VoiceResult(
                    test_id=test.id, label=test.label, category=test.category,
                    spoken_text=test.spoken_text, error=f"Connection failed: {e}",
                ))
            continue

        for i, test in enumerate(batch_tests):
            print(f"\n{'─' * 60}")
            print(f"[{batch_label} {i+1}/{len(batch_tests)}] {test.label}")
            print(f"  🗣  \"{test.spoken_text[:60]}\"")

            result = VoiceResult(
                test_id=test.id,
                label=test.label,
                category=test.category,
                spoken_text=test.spoken_text,
                turn_number=client.turn_count + 1,
            )

            try:
                turn_data = await client.send_turn(test)

                if "error" in turn_data:
                    result.error = turn_data["error"]
                    print_colored(f"  ❌ ERROR: {result.error}", "31")
                    results.append(result)
                    continue

                # Fill in result
                result.transcription = turn_data["transcription"]
                result.bot_text = turn_data["bot_text"]
                result.bot_audio_bytes = turn_data["bot_audio_bytes"]
                result.bot_audio_duration_sec = turn_data["bot_audio_duration_sec"]
                result.ttfab_ms = turn_data["ttfab_ms"]
                result.total_rt_ms = turn_data["total_rt_ms"]

                # ── Save bot audio as WAV file ──
                raw_pcm = turn_data.get("bot_audio_raw", b"")
                if raw_pcm:
                    wav_out_dir = f"tests/eval_audio_output/{target_name}"
                    os.makedirs(wav_out_dir, exist_ok=True)
                    wav_path = os.path.join(wav_out_dir, f"{test.id}_turn{client.turn_count}.wav")
                    try:
                        with wave.open(wav_path, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)  # PCM16
                            wf.setframerate(24000)  # Bot audio is 24kHz
                            wf.writeframes(raw_pcm)
                        logger.info(f"  💾 Saved audio: {wav_path} ({len(raw_pcm)/(24000*2):.1f}s)")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Failed to save audio: {e}")

                    # ── Quick waveform analysis for clicks/gaps ──
                    try:
                        arr = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32)
                        # Detect silence gaps (>100ms of near-zero amplitude)
                        window_ms = 100
                        window_samples = int(24000 * window_ms / 1000)
                        silence_threshold = 200  # ~-50dB for int16
                        gaps = []
                        gap_start = None
                        for s in range(0, len(arr) - window_samples, window_samples // 2):
                            window = arr[s:s + window_samples]
                            if np.max(np.abs(window)) < silence_threshold:
                                if gap_start is None:
                                    gap_start = s
                            else:
                                if gap_start is not None:
                                    gap_dur_ms = (s - gap_start) / 24000 * 1000
                                    if gap_dur_ms >= 150:  # Only report gaps >= 150ms
                                        gap_at_sec = gap_start / 24000
                                        gaps.append((gap_at_sec, gap_dur_ms))
                                    gap_start = None
                        # Check for clicks (sudden large amplitude spikes)
                        diff = np.abs(np.diff(arr))
                        click_threshold = 20000  # Large sudden jump
                        click_indices = np.where(diff > click_threshold)[0]
                        clicks = [idx / 24000 for idx in click_indices]

                        if gaps:
                            print(f"  🔇 Silence gaps detected: {len(gaps)}")
                            for gap_at, gap_dur in gaps[:5]:
                                print(f"      at {gap_at:.2f}s — {gap_dur:.0f}ms gap")
                        if clicks:
                            print(f"  ⚡ Potential clicks detected: {len(clicks)}")
                            for c in clicks[:5]:
                                print(f"      at {c:.3f}s")
                        if not gaps and not clicks:
                            print(f"  ✅ Audio waveform: no gaps or clicks detected")
                    except Exception as e:
                        logger.debug(f"  Waveform analysis error: {e}")

                # ── Print audio segment & text token stats ──
                segments = turn_data.get("audio_segments", [])
                if segments:
                    print(f"  📊 Audio segments (sentence-level):")
                    for seg in segments:
                        gap_str = f" → then {seg['gap_after_ms']}ms gap" if seg['gap_after_ms'] > 0 else ""
                        print(f"      Seg {seg['seg']}: {seg['chunks']:>3} chunks | "
                              f"{seg['audio_sec']:>5.2f}s audio | "
                              f"recv {seg['recv_sec']:>5.2f}s{gap_str}")
                tk_count = turn_data.get("text_token_count", 0)
                tk_first = turn_data.get("text_first_token_ms", -1)
                tk_last = turn_data.get("text_last_token_ms", -1)
                if tk_count > 0:
                    print(f"  📝 Text tokens: {tk_count} | first at {tk_first}ms | last at {tk_last}ms")

                # STT accuracy
                if test.expected_transcription and result.transcription:
                    result.stt_match = round(compute_word_overlap(
                        test.expected_transcription, result.transcription
                    ), 2)
                    mark = "✅" if result.stt_match >= 0.5 else "⚠️"
                    print(f"  {mark} STT accuracy: {result.stt_match:.0%} — heard: \"{result.transcription[:60]}\"")
                elif result.transcription:
                    print(f"  🎤 Heard: \"{result.transcription[:60]}\"")
                else:
                    print(f"  ⚠️  No transcription captured")

                # Relevance check
                if result.bot_text and test.relevance_keywords:
                    result.relevance_hit, result.relevance_keywords_found = check_relevance(
                        result.bot_text, test.relevance_keywords
                    )
                    if result.relevance_hit:
                        print(f"  ✅ Relevance: found [{', '.join(result.relevance_keywords_found[:3])}]")
                    else:
                        print_colored(f"  ⚠️  Relevance: none of {test.relevance_keywords[:3]} found in response", "33")

                # Judge quality
                if run_judge and result.bot_text and JUDGE_API_KEY:
                    print(f"  ⚖️  Judging...", end=" ", flush=True)
                    result.scores = judge_response(test, result.bot_text)
                    overall = int(result.scores.get("overall", 0))
                    print_colored(
                        f"{'★' * overall}{'☆' * (5-overall)} ({overall}/5) — {result.scores.get('one_line_feedback', '')[:50]}",
                        score_color(overall),
                    )

            except websockets.ConnectionClosed:
                result.error = "WebSocket closed mid-turn"
                logger.warning(f"  ❌ Connection closed, reconnecting for remaining tests...")
                # Try to reconnect for remaining tests
                try:
                    await client.disconnect()
                    await client.connect(lang=batch_lang)
                except Exception as e2:
                    result.error = f"Reconnect failed: {e2}"
            except Exception as e:
                result.error = str(e)
                logger.error(f"  ❌ Error: {e}")

            results.append(result)

            # Brief pause between turns
            await asyncio.sleep(1.5)

        await client.disconnect()

    # ══════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("VOICE EVAL v2 RESULTS")
    print("=" * 70)

    successful = [r for r in results if not r.error and r.bot_text]
    failed = [r for r in results if r.error]

    if not successful:
        print_colored("\n❌ No successful tests!", "31")
        return

    # ── Reliability Summary ──
    print(f"\n📊 RELIABILITY ({len(successful)}/{len(results)} tests passed)")
    print("─" * 50)
    stt_captured = [r for r in successful if r.transcription]
    audio_received = [r for r in successful if r.bot_audio_bytes > 0]
    text_received = [r for r in successful if r.bot_text]
    relevant = [r for r in successful if r.relevance_hit]
    relevant_total = [r for r in successful if r.relevance_keywords_found is not None and len([kw for kw in (VOICE_TEST_CASES[i].relevance_keywords if i < len(VOICE_TEST_CASES) else []) for i2, t in enumerate(VOICE_TEST_CASES) if t.id == r.test_id]) > 0]

    # Count tests that had keywords to check
    tests_with_keywords = [r for r in successful if any(t.relevance_keywords for t in VOICE_TEST_CASES if t.id == r.test_id)]
    relevant_of_checkable = [r for r in tests_with_keywords if r.relevance_hit]

    print(f"  STT transcription captured:  {len(stt_captured)}/{len(successful)} {'✅' if len(stt_captured) > len(successful)*0.7 else '⚠️'}")
    print(f"  Bot audio received:          {len(audio_received)}/{len(successful)} {'✅' if len(audio_received) == len(successful) else '⚠️'}")
    print(f"  Bot text received:           {len(text_received)}/{len(successful)} {'✅' if len(text_received) == len(successful) else '⚠️'}")
    if tests_with_keywords:
        print(f"  Response relevance:          {len(relevant_of_checkable)}/{len(tests_with_keywords)} {'✅' if len(relevant_of_checkable) > len(tests_with_keywords)*0.8 else '⚠️'}")

    # ── Latency Summary ──
    print(f"\n📊 LATENCY METRICS")
    print("─" * 50)

    ttfab_values = [r.ttfab_ms for r in successful if r.ttfab_ms > 0]
    rt_values = [r.total_rt_ms for r in successful if r.total_rt_ms > 0]
    audio_dur_values = [r.bot_audio_duration_sec for r in successful if r.bot_audio_duration_sec > 0]

    if ttfab_values:
        avg_ttfab = sum(ttfab_values) / len(ttfab_values)
        min_ttfab = min(ttfab_values)
        max_ttfab = max(ttfab_values)
        p50 = sorted(ttfab_values)[len(ttfab_values)//2]
        p90 = sorted(ttfab_values)[int(len(ttfab_values)*0.9)]
        color = "92" if avg_ttfab < 2000 else ("33" if avg_ttfab < 4000 else "31")
        print_colored(f"  Time to First Audio Byte (TTFAB):", color)
        print_colored(f"    Avg: {avg_ttfab:.0f}ms  |  P50: {p50:.0f}ms  |  P90: {p90:.0f}ms", color)
        print_colored(f"    Min: {min_ttfab:.0f}ms  |  Max: {max_ttfab:.0f}ms  |  N={len(ttfab_values)}", color)

    if rt_values:
        avg_rt = sum(rt_values) / len(rt_values)
        print(f"  Total Round-Trip:")
        print(f"    Avg: {avg_rt:.0f}ms  |  Min: {min(rt_values):.0f}ms  |  Max: {max(rt_values):.0f}ms")

    if audio_dur_values:
        avg_dur = sum(audio_dur_values) / len(audio_dur_values)
        print(f"  Bot Audio Duration:")
        print(f"    Avg: {avg_dur:.1f}s  |  Min: {min(audio_dur_values):.1f}s  |  Max: {max(audio_dur_values):.1f}s")

    # ── STT Accuracy ──
    stt_values = [r.stt_match for r in successful if r.stt_match > 0]
    print(f"\n📊 STT ACCURACY")
    print("─" * 50)
    if stt_values:
        avg_stt = sum(stt_values) / len(stt_values)
        color = "92" if avg_stt >= 0.8 else ("33" if avg_stt >= 0.5 else "31")
        print_colored(f"  Avg word overlap: {avg_stt:.0%} ({len(stt_values)} tests with transcription)", color)
        for r in successful:
            if r.transcription:
                mark = "✅" if r.stt_match >= 0.7 else ("⚠️" if r.stt_match >= 0.3 else "❌")
                print(f"    {mark} {r.test_id}: {r.stt_match:.0%} — heard: \"{r.transcription[:50]}\"")
    else:
        print_colored(f"  ⚠️  No STT transcriptions captured (0/{len(successful)} tests)", "33")
        print(f"  Note: user_transcript JSON messages may not be arriving.")
        print(f"  Check server logs for [USER_TEXT] entries.")

    # ── Per-test detail ──
    print(f"\n📊 PER-TEST DETAIL ({len(successful)} tests)")
    print("─" * 70)
    print(f"  {'Test ID':<25} {'Turn':>4} {'TTFAB':>7} {'RT':>8} {'Audio':>6} {'STT':>5} {'Rel':>4}")
    print(f"  {'─'*25} {'─'*4} {'─'*7} {'─'*8} {'─'*6} {'─'*5} {'─'*4}")
    for r in successful:
        rel_mark = "✅" if r.relevance_hit else ("—" if not any(t.relevance_keywords for t in VOICE_TEST_CASES if t.id == r.test_id) else "⚠️")
        stt_mark = f"{r.stt_match:.0%}" if r.stt_match > 0 else ("—" if not r.transcription else "0%")
        print(
            f"  {r.test_id:<25} {r.turn_number:>4} "
            f"{r.ttfab_ms:>6.0f}ms {r.total_rt_ms:>7.0f}ms "
            f"{r.bot_audio_duration_sec:>5.1f}s {stt_mark:>5} {rel_mark:>4}"
        )
        if r.bot_text:
            print(f"    🤖 \"{r.bot_text[:90]}{'...' if len(r.bot_text)>90 else ''}\"")

    # ── Quality Summary ──
    scored = [r for r in successful if r.scores]
    if scored:
        print(f"\n📊 QUALITY SCORES (GPT-4o Judge, N={len(scored)})")
        print("─" * 50)

        dims = ["friendly_warm", "educational_encouraging", "indian_cultural",
                "emotional_intelligence", "socratic_teaching", "brevity_voice_ready", "overall"]

        all_scores = [r.scores for r in scored]
        for dim in dims:
            vals = [s.get(dim, 0) for s in all_scores]
            avg = sum(vals) / len(vals) if vals else 0
            label = dim.replace("_", " ").title()
            color = score_color(avg)
            print_colored(f"   {label:<25} {score_bar(avg)}", color)

    # ── Failures ──
    if failed:
        print(f"\n⚠️  FAILED TESTS ({len(failed)}):")
        for r in failed:
            print(f"  ❌ {r.test_id}: {r.error}")

    # ── Confidence Assessment ──
    print(f"\n📊 CONFIDENCE ASSESSMENT")
    print("─" * 50)
    checks = []
    # Check 1: Enough tests
    if len(successful) >= 12:
        checks.append(("✅", f"Sample size: {len(successful)} tests (≥12)"))
    else:
        checks.append(("⚠️", f"Sample size: {len(successful)} tests (<12, need more)"))

    # Check 2: STT working
    if len(stt_captured) >= len(successful) * 0.5:
        checks.append(("✅", f"STT capture: {len(stt_captured)}/{len(successful)} tests have transcription"))
    else:
        checks.append(("⚠️", f"STT capture: {len(stt_captured)}/{len(successful)} tests — user_transcript not arriving"))

    # Check 3: Audio received
    if len(audio_received) == len(successful):
        checks.append(("✅", f"Audio pipeline: all {len(successful)} tests received bot audio"))
    else:
        checks.append(("⚠️", f"Audio pipeline: {len(audio_received)}/{len(successful)} tests received audio"))

    # Check 4: TTFAB consistency
    if ttfab_values and max(ttfab_values) / max(min(ttfab_values), 1) < 5:
        checks.append(("✅", f"TTFAB consistency: max/min ratio = {max(ttfab_values)/max(min(ttfab_values),1):.1f}x"))
    elif ttfab_values:
        checks.append(("⚠️", f"TTFAB inconsistent: max/min ratio = {max(ttfab_values)/max(min(ttfab_values),1):.1f}x"))

    # Check 5: Relevance
    if tests_with_keywords and len(relevant_of_checkable) >= len(tests_with_keywords) * 0.8:
        checks.append(("✅", f"Response relevance: {len(relevant_of_checkable)}/{len(tests_with_keywords)} on-topic"))
    elif tests_with_keywords:
        checks.append(("⚠️", f"Response relevance: {len(relevant_of_checkable)}/{len(tests_with_keywords)} on-topic"))

    # Check 6: Multi-turn (conversation context)
    multi_turn = [r for r in successful if r.turn_number > 1]
    if multi_turn:
        checks.append(("✅", f"Multi-turn: {len(multi_turn)} tests ran with conversation history"))
    else:
        checks.append(("⚠️", "Multi-turn: all tests ran as first turn (no conversation context)"))

    for mark, desc in checks:
        print(f"  {mark} {desc}")

    all_pass = all(c[0] == "✅" for c in checks)
    if all_pass:
        print_colored(f"\n  ✅ HIGH CONFIDENCE — all reliability checks passed", "92")
    else:
        warnings = sum(1 for c in checks if c[0] == "⚠️")
        print_colored(f"\n  ⚠️  MEDIUM CONFIDENCE — {warnings} reliability warning(s)", "33")

    # ── Save results ──
    output_path = f"tests/eval_voice_results_{target_name}.json"
    json_results = []
    for r in results:
        json_results.append({
            "test_id": r.test_id,
            "label": r.label,
            "category": r.category,
            "spoken_text": r.spoken_text,
            "turn_number": r.turn_number,
            "transcription": r.transcription,
            "stt_match": r.stt_match,
            "ttfab_ms": r.ttfab_ms,
            "total_rt_ms": r.total_rt_ms,
            "bot_text": r.bot_text,
            "bot_audio_bytes": r.bot_audio_bytes,
            "bot_audio_duration_sec": r.bot_audio_duration_sec,
            "relevance_hit": r.relevance_hit,
            "relevance_keywords_found": r.relevance_keywords_found,
            "scores": r.scores,
            "error": r.error,
        })

    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "target": target_name,
            "ws_url": ws_url,
            "judge_model": JUDGE_MODEL if scored else None,
            "test_count": len(results),
            "success_count": len(successful),
            "mode": "single_session_multi_turn",
            "results": json_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Full results saved to {output_path}")

    # ── Comparison with text eval ──
    text_eval_path = f"tests/eval_results_{target_name}.json"
    if os.path.exists(text_eval_path) and scored:
        try:
            with open(text_eval_path) as f:
                text_eval = json.load(f)
            text_scores = text_eval.get("results", [])
            if text_scores:
                text_avg_latency = sum(r.get("latency_ms", 0) for r in text_scores) / len(text_scores)
                voice_avg_ttfab = sum(ttfab_values) / max(len(ttfab_values), 1) if ttfab_values else 0

                text_quality = sum(r.get("scores", {}).get("overall", 0) for r in text_scores) / len(text_scores)
                voice_quality = sum(r.scores.get("overall", 0) for r in scored) / len(scored)

                print(f"\n📊 TEXT vs VOICE COMPARISON")
                print("─" * 50)
                print(f"  {'Metric':<30} {'Text Mode':<15} {'Voice Mode':<15}")
                print(f"  {'─' * 58}")
                print(f"  {'Avg Latency / TTFAB':<30} {text_avg_latency:.0f}ms{'':<10} {voice_avg_ttfab:.0f}ms")
                print(f"  {'Quality (Overall)':<30} {text_quality:.1f}/5{'':<10} {voice_quality:.1f}/5")
        except Exception:
            pass

    print("\n🏁 VOICE EVAL v2 COMPLETE!")


def main():
    global WS_URL

    parser = argparse.ArgumentParser(description="Mira Voice Eval v2 — End-to-End Audio Pipeline")
    parser.add_argument(
        "--target", choices=list(TARGETS.keys()), default=None,
        help="Deployment target (local, oss, elevenlabs).",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip GPT-4o quality judging (latency-only mode).",
    )
    parser.add_argument(
        "--audio-dir", default=None,
        help="Directory containing pre-recorded WAV files.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of test cases to run (useful for quick sanity checks).",
    )
    args = parser.parse_args()

    if args.target:
        target = TARGETS[args.target]
        WS_URL = target["ws_url"]
        target_name = args.target
    else:
        target_name = "custom" if WS_URL != "ws://localhost:7860/ws" else "local"

    audio_dir = args.audio_dir or os.getenv("AUDIO_DIR", "tests/test_audio")
    if not os.path.isabs(audio_dir):
        # Try /app/tests/test_audio inside Docker
        if os.path.exists("/app/tests/test_audio"):
            audio_dir = "/app/tests/test_audio"

    run_judge = not args.no_judge
    if run_judge and (not JUDGE_API_KEY or JUDGE_API_KEY == "DUMMY_KEY"):
        print("⚠️  No JUDGE_API_KEY — running in latency-only mode (no quality scoring)")
        run_judge = False

    asyncio.run(run_all(WS_URL, audio_dir, target_name, run_judge, limit=args.limit))


if __name__ == "__main__":
    main()
