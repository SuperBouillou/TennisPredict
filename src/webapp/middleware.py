"""AuthMiddleware — protects all routes except public ones."""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.webapp.auth import get_user_id, set_flash


def _is_public(path: str) -> bool:
    """Return True if the path is publicly accessible without auth."""
    if path == "/":
        return True
    return any(path.startswith(p) for p in ("/login", "/static"))


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if _is_public(request.url.path):
            return await call_next(request)

        user_id = get_user_id(request)
        if user_id is None:
            redirect = RedirectResponse(url="/login", status_code=302)
            set_flash(redirect, "Session expirée, veuillez vous reconnecter")
            return redirect

        request.state.user_id = user_id
        return await call_next(request)
