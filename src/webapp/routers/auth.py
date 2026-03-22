"""Auth routes — GET/POST /login, POST /logout."""
from __future__ import annotations

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from src.webapp.auth import (
    verify_password, create_session_cookie,
    delete_session_cookie, set_flash, get_flash,
)
from src.webapp.db import get_connection
from src.webapp.limiter import limiter

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    # Create the response first, then pass it to get_flash so the deletion
    # Set-Cookie header is included in the response actually sent to the browser.
    resp = templates.TemplateResponse("login.html", {"request": request, "flash": None})
    flash = get_flash(request, resp)
    resp.context["flash"] = flash
    return resp


@router.post("/login")
@limiter.limit("5/minute")
async def login_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
    finally:
        conn.close()

    if row is None or not verify_password(password, row["password_hash"]):
        redirect = RedirectResponse(url="/login", status_code=302)
        set_flash(redirect, "Email ou mot de passe incorrect")
        return redirect

    redirect = RedirectResponse(url="/today", status_code=302)
    create_session_cookie(redirect, user_id=row["id"])
    return redirect


@router.post("/logout")
async def logout(request: Request):
    redirect = RedirectResponse(url="/login", status_code=302)
    delete_session_cookie(redirect)
    set_flash(redirect, "Deconnexion reussie")
    return redirect
