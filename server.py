"""
MiraVoiceAI Pipecat Server - FastAPI WebSocket Server

This server provides a WebSocket endpoint for voice interaction with the
MiraVoiceAI bot that orchestrates STT -> LLM -> TTS pipeline.

Endpoints:
- WebSocket /ws - Main bot interaction endpoint
- GET /health - Health check
- GET /config - Get current configuration
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bot import run_bot, DEFAULT_VOICE, DEFAULT_LANGUAGE, ASR_WS_URL, TTS_WS_URL, LLM_BASE_URL, LLM_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting MiraVoiceAI Pipecat server...")
    yield
    logger.info("Shutting down MiraVoiceAI Pipecat server...")


app = FastAPI(
    title="MiraVoiceAI Pipecat Server",
    description="Voice AI bot with STT -> LLM -> TTS pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mira-voice-ai-pipecat"}


@app.post("/connect")
async def connect():
    """Return WebSocket URL for Pipecat client connection."""
    return {"wsUrl": f"ws://localhost:{PORT}/ws"}


@app.get("/config")
async def get_config():
    """Get current server configuration."""
    return {
        "asr_ws_url": ASR_WS_URL,
        "tts_ws_url": TTS_WS_URL,
        "llm_base_url": LLM_BASE_URL,
        "llm_model": LLM_MODEL,
        "default_voice": DEFAULT_VOICE,
        "default_language": DEFAULT_LANGUAGE,
    }


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    voice: str = Query(default=DEFAULT_VOICE, description="TTS voice ID"),
    language: str = Query(default=DEFAULT_LANGUAGE, description="STT language code"),
    sample_rate: int = Query(default=16000, description="Audio sample rate"),
):
    """
    WebSocket endpoint for voice interaction.

    Protocol:
    1. Client connects with optional query params (voice, language, sample_rate)
    2. Client sends audio as binary PCM16 frames
    3. Server sends transcriptions and audio responses
    4. Connection stays open for continuous conversation

    Query Parameters:
        voice: TTS voice ID (default: hi_male)
        language: STT language (default: auto)
        sample_rate: Audio sample rate (default: 16000)
    """
    await websocket.accept()
    import uuid
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"[{session_id}] Client connected: language={language}, sample_rate={sample_rate}Hz (voice forced to hi_male)")

    try:
        await run_bot(
            websocket=websocket,
            sample_rate=sample_rate,
            voice=voice,
            language=language,
        )
    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Client disconnected")
    except Exception as e:
        logger.error(f"[{session_id}] WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/voices")
async def list_voices():
    """List available TTS voices."""
    # Common voice IDs for Svara TTS
    voices = [
        {"id": "hi_male", "name": "Hindi (Male)", "language": "Hindi"},
        {"id": "hi_female", "name": "Hindi (Female)", "language": "Hindi"},
        {"id": "en_male", "name": "English (Male)", "language": "English"},
        {"id": "en_female", "name": "English (Female)", "language": "English"},
        {"id": "ta_male", "name": "Tamil (Male)", "language": "Tamil"},
        {"id": "ta_female", "name": "Tamil (Female)", "language": "Tamil"},
        {"id": "te_male", "name": "Telugu (Male)", "language": "Telugu"},
        {"id": "te_female", "name": "Telugu (Female)", "language": "Telugu"},
        {"id": "bn_male", "name": "Bengali (Male)", "language": "Bengali"},
        {"id": "bn_female", "name": "Bengali (Female)", "language": "Bengali"},
        {"id": "mr_male", "name": "Marathi (Male)", "language": "Marathi"},
        {"id": "mr_female", "name": "Marathi (Female)", "language": "Marathi"},
        {"id": "kn_male", "name": "Kannada (Male)", "language": "Kannada"},
        {"id": "kn_female", "name": "Kannada (Female)", "language": "Kannada"},
    ]
    return {"voices": voices}


@app.get("/languages")
async def list_languages():
    """List supported STT languages."""
    languages = [
        {"code": "auto", "name": "Auto-detect"},
        {"code": "hi", "name": "Hindi"},
        {"code": "en", "name": "English"},
        {"code": "ta", "name": "Tamil"},
        {"code": "te", "name": "Telugu"},
        {"code": "bn", "name": "Bengali"},
        {"code": "mr", "name": "Marathi"},
        {"code": "gu", "name": "Gujarati"},
        {"code": "kn", "name": "Kannada"},
        {"code": "ml", "name": "Malayalam"},
        {"code": "pa", "name": "Punjabi"},
        {"code": "ur", "name": "Urdu"},
        {"code": "or", "name": "Odia"},
        {"code": "as", "name": "Assamese"},
    ]
    return {"languages": languages}


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        workers=1,
        loop="asyncio",
        access_log=True,
        log_level="info",
    )
