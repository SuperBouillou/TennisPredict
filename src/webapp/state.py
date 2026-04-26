"""Shared APP_STATE accessor.

Avoids repeating the same deferred-import boilerplate in every router.
"""
from __future__ import annotations


def get_state() -> dict:
    """Return the global APP_STATE dict (deferred import prevents circular dep)."""
    from src.webapp.main import APP_STATE  # noqa: PLC0415
    return APP_STATE
