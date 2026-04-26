"""Shared utility helpers for webapp routers."""
from __future__ import annotations
import math


def safe_float(v, default: float = 0.0) -> float:
    """Safely convert v to float, returning default on None/NaN/Inf/error."""
    try:
        if v is None:
            return default
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


def safe_get(d: dict, key: str, default: float = 0.0) -> float:
    """Safely extract a float from a dict by key."""
    return safe_float(d.get(key), default)
