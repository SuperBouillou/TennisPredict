"""Authentication helpers — cookie signing, password hashing, flash messages."""
from __future__ import annotations

import os
import bcrypt
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from fastapi import Request
from fastapi.responses import Response

_SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
_SALT = "tennispredict-session"
_COOKIE_NAME = "tp_session"
_FLASH_COOKIE = "tp_flash"
_MAX_AGE = 30 * 24 * 3600  # 30 days in seconds
_SECURE = os.getenv("SECURE_COOKIE", "false").lower() == "true"

_serializer = URLSafeTimedSerializer(_SECRET_KEY)


# ── Session ───────────────────────────────────────────────────────────────────

def create_session_cookie(response: Response, user_id: int) -> None:
    """Sign and set the session cookie on a response."""
    token = _serializer.dumps({"user_id": user_id}, salt=_SALT)
    response.set_cookie(
        key=_COOKIE_NAME,
        value=token,
        max_age=_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=_SECURE,
    )


def delete_session_cookie(response: Response) -> None:
    """Clear the session cookie."""
    response.delete_cookie(key=_COOKIE_NAME, samesite="lax", secure=_SECURE)


def get_user_id(request: Request) -> int | None:
    """Return user_id from session cookie, or None if missing/invalid/expired."""
    token = request.cookies.get(_COOKIE_NAME)
    if not token:
        return None
    try:
        data = _serializer.loads(token, salt=_SALT, max_age=_MAX_AGE)
        return int(data["user_id"])
    except (BadSignature, SignatureExpired, KeyError, ValueError):
        return None


# ── Password ──────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Return bcrypt hash of password (cost factor 12)."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Return True if password matches hash."""
    return bcrypt.checkpw(password.encode(), password_hash.encode())


# ── Flash messages ────────────────────────────────────────────────────────────

def set_flash(response: Response, message: str) -> None:
    """Store a one-time flash message in a short-lived cookie."""
    response.set_cookie(
        key=_FLASH_COOKIE,
        value=message,
        max_age=60,
        httponly=False,
        samesite="lax",
        secure=_SECURE,
    )


def get_flash(request: Request, response: Response) -> str | None:
    """Read and clear the flash message cookie. Returns message or None.
    IMPORTANT: pass the response object that will actually be returned to the client.
    """
    message = request.cookies.get(_FLASH_COOKIE)
    if message:
        response.delete_cookie(key=_FLASH_COOKIE, samesite="lax")
    return message
