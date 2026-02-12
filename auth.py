"""
JWT authentication for Pipecat ↔ OpenWebUI trust.

When WEBUI_SECRET_KEY is set, all classroom REST endpoints and WebSocket
connections require a valid OpenWebUI JWT (HS256, same secret).

When WEBUI_SECRET_KEY is NOT set (local dev / tests), identity falls back
to x-user-id / x-user-name / x-user-role headers (no verification).

OpenWebUI JWT payload: {"id": "<user-uuid>", "exp": ..., "jti": "..."}
We decode the JWT to get the trusted user ID.  Name / email / role are
still accepted from headers as supplementary (display) info — only the
user_id is cryptographically verified.
"""

import logging
import os
from typing import Optional

import jwt  # PyJWT

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────
WEBUI_SECRET_KEY: Optional[str] = os.getenv("WEBUI_SECRET_KEY", "").strip() or None
JWT_ALGORITHM = "HS256"

# Convenience flag
AUTH_ENABLED = WEBUI_SECRET_KEY is not None

if AUTH_ENABLED:
    logger.info("[AUTH] JWT auth ENABLED — WEBUI_SECRET_KEY is set")
else:
    logger.info("[AUTH] JWT auth DISABLED — falling back to header-based identity (dev/test mode)")


def decode_openwebui_jwt(token: str) -> Optional[dict]:
    """Decode and verify an OpenWebUI JWT.

    Returns the decoded payload dict on success, or None on failure.
    The payload contains at minimum: {"id": "<user-uuid>"}.
    """
    if not WEBUI_SECRET_KEY:
        return None

    try:
        payload = jwt.decode(
            token,
            WEBUI_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"verify_exp": True},
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("[AUTH] JWT expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"[AUTH] Invalid JWT: {e}")
        return None


def extract_bearer_token(auth_header: Optional[str]) -> Optional[str]:
    """Extract the token from an 'Authorization: Bearer <token>' header."""
    if not auth_header:
        return None
    parts = auth_header.strip().split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def get_verified_user_id(auth_header: Optional[str]) -> Optional[str]:
    """Extract and verify user ID from a Bearer JWT.

    Returns the user_id string if the JWT is valid, or None if:
    - No auth header provided
    - JWT is invalid/expired
    - AUTH is disabled (WEBUI_SECRET_KEY not set)
    """
    token = extract_bearer_token(auth_header)
    if not token:
        return None

    payload = decode_openwebui_jwt(token)
    if payload and "id" in payload:
        return payload["id"]
    return None
