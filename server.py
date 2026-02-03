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
import signal
import sys
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bot import (
    run_bot,
    DEFAULT_VOICE,
    DEFAULT_LANGUAGE,
    TTS_WS_URL,
    LLM_BASE_URL,
    LLM_MODEL,
    SONIOX_API_KEY,
    TTS_PROVIDER,
    ELEVENLABS_API_KEY,
    TTS_VOICE_GENDER,
)
from services.elevenlabs_tts import VOICE_PRESETS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))


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
    """Return WebSocket URL for direct connection to backend."""
    public_host = os.getenv("PUBLIC_HOST", "hp-fury")

    # Check if SSL is enabled (same logic as run_server)
    ssl_keyfile = os.getenv("SSL_KEYFILE", "hp-fury+1-key.pem")
    ssl_certfile = os.getenv("SSL_CERTFILE", "hp-fury+1.pem")
    use_ssl = os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile)

    protocol = "wss" if use_ssl else "ws"
    return {"ws_url": f"{protocol}://{public_host}:{PORT}/ws"}


@app.get("/config")
async def get_config():
    """Get current server configuration."""
    return {
        "stt_provider": "soniox",
        "soniox_api_key_set": bool(SONIOX_API_KEY),
        "tts_provider": TTS_PROVIDER,
        "tts_ws_url": TTS_WS_URL,
        "elevenlabs_api_key_set": bool(ELEVENLABS_API_KEY),
        "tts_voice_gender": TTS_VOICE_GENDER,
        "llm_base_url": LLM_BASE_URL,
        "llm_model": LLM_MODEL,
        "default_voice": DEFAULT_VOICE,
        "default_language": DEFAULT_LANGUAGE,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for voice interaction."""
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    try:
        await run_bot(websocket=websocket)
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


@app.get("/voices")
async def list_voices():
    """List available TTS voices."""
    # Svara TTS voices
    svara_voices = [
        {"id": "hi_male", "name": "Hindi (Male)", "language": "Hindi", "provider": "svara"},
        {"id": "hi_female", "name": "Hindi (Female)", "language": "Hindi", "provider": "svara"},
        {"id": "en_male", "name": "English (Male)", "language": "English", "provider": "svara"},
        {"id": "en_female", "name": "English (Female)", "language": "English", "provider": "svara"},
        {"id": "ta_male", "name": "Tamil (Male)", "language": "Tamil", "provider": "svara"},
        {"id": "ta_female", "name": "Tamil (Female)", "language": "Tamil", "provider": "svara"},
        {"id": "te_male", "name": "Telugu (Male)", "language": "Telugu", "provider": "svara"},
        {"id": "te_female", "name": "Telugu (Female)", "language": "Telugu", "provider": "svara"},
        {"id": "bn_male", "name": "Bengali (Male)", "language": "Bengali", "provider": "svara"},
        {"id": "bn_female", "name": "Bengali (Female)", "language": "Bengali", "provider": "svara"},
        {"id": "mr_male", "name": "Marathi (Male)", "language": "Marathi", "provider": "svara"},
        {"id": "mr_female", "name": "Marathi (Female)", "language": "Marathi", "provider": "svara"},
        {"id": "kn_male", "name": "Kannada (Male)", "language": "Kannada", "provider": "svara"},
        {"id": "kn_female", "name": "Kannada (Female)", "language": "Kannada", "provider": "svara"},
    ]

    # ElevenLabs voices
    elevenlabs_voices = [
        {
            "id": VOICE_PRESETS["female"]["id"],
            "name": VOICE_PRESETS["female"]["name"],
            "language": "Multilingual",
            "provider": "elevenlabs",
            "gender": "female",
        },
        {
            "id": VOICE_PRESETS["male"]["id"],
            "name": VOICE_PRESETS["male"]["name"],
            "language": "Multilingual",
            "provider": "elevenlabs",
            "gender": "male",
        },
    ]

    return {
        "voices": svara_voices + elevenlabs_voices,
        "current_provider": TTS_PROVIDER,
        "current_voice_gender": TTS_VOICE_GENDER,
    }


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


def run_server():
    """Run the server with proper signal handling."""
    # Handle shutdown signals - use os._exit for immediate termination
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, forcing shutdown...")
        os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Check for SSL certificates
    ssl_keyfile = os.getenv("SSL_KEYFILE", "hp-fury+1-key.pem")
    ssl_certfile = os.getenv("SSL_CERTFILE", "hp-fury+1.pem")
    use_ssl = os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile)

    config_kwargs = {
        "app": app,
        "host": HOST,
        "port": PORT,
        "access_log": True,
        "log_level": "info",
    }

    if use_ssl:
        config_kwargs["ssl_keyfile"] = ssl_keyfile
        config_kwargs["ssl_certfile"] = ssl_certfile
        logger.info(f"Starting server with HTTPS on port {PORT}")
    else:
        logger.info(f"Starting server with HTTP on port {PORT}")

    config = uvicorn.Config(**config_kwargs)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    run_server()
