"""FastAPI pages and endpoints for the live Jetson detector."""

import asyncio
import hmac
import json
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .db import (
    add_audit,
    get_user,
    init_db,
    list_detection_events,
    seed_defaults,
)
from .pipeline import pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
SESSION_SECRET = os.environ.get("INSPECT_WEB_SESSION_SECRET", "")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if len(SESSION_SECRET) < 32:
        raise RuntimeError(
            "Set INSPECT_WEB_SESSION_SECRET to a random value of at least 32 characters."
        )
    init_db()
    seed_defaults()
    yield


app = FastAPI(title="Jetson Inspection Dashboard", lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET or secrets.token_urlsafe(48),
    https_only=os.environ.get("INSPECT_WEB_HTTPS_ONLY", "0") == "1",
    same_site="lax",
)
app.mount(
    "/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static"
)


def signed_in(request: Request) -> str | None:
    username = request.session.get("username")
    if not username or get_user(username) is None:
        return None
    return str(username)


def csrf_token(request: Request) -> str:
    token = request.session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf_token"] = token
    return token


def check_csrf(request: Request, submitted: str | None) -> None:
    expected = request.session.get("csrf_token", "")
    if not expected or not submitted or not hmac.compare_digest(expected, submitted):
        raise HTTPException(status_code=403, detail="Invalid request token")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse(
        "/dashboard" if signed_in(request) else "/login", status_code=302
    )


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        request, "login.html", {"request": request, "error": "",
                                 "csrf_token": csrf_token(request)}
    )


@app.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    csrf: str = Form(...),
):
    check_csrf(request, csrf)
    from .db import verify_password
    user = get_user(username.strip())
    if not user or not verify_password(password, user["password_hash"]):
        return templates.TemplateResponse(
            request, "login.html", {"request": request,
                                     "error": "Invalid username or password",
                                     "csrf_token": csrf_token(request)},
            status_code=401,
        )
    request.session.clear()
    request.session["username"] = user["username"]
    csrf_token(request)
    add_audit(user["username"], "login")
    return RedirectResponse("/dashboard", status_code=303)


@app.post("/logout")
async def logout(request: Request, csrf: str = Form(...)):
    check_csrf(request, csrf)
    username = signed_in(request) or "unknown"
    request.session.clear()
    add_audit(username, "logout")
    return RedirectResponse("/login", status_code=303)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    username = signed_in(request)
    if not username:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(
        request, "dashboard.html", {"request": request, "username": username,
                                     "snapshot": await pipeline.snapshot(),
                                     "csrf_token": csrf_token(request)}
    )


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    username = signed_in(request)
    if not username:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(
        request, "history.html", {"request": request, "username": username,
                                  "rows": list_detection_events(200),
                                  "csrf_token": csrf_token(request)}
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    username = signed_in(request)
    if not username:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(
        request, "settings.html", {"request": request, "username": username,
                                   "snapshot": await pipeline.snapshot(),
                                   "csrf_token": csrf_token(request)}
    )


@app.get("/api/status")
async def api_status(request: Request):
    if not signed_in(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return JSONResponse(await pipeline.snapshot())


@app.get("/api/history")
async def api_history(request: Request, limit: int = 50):
    if not signed_in(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return JSONResponse({"items": list_detection_events(limit)})


@app.post("/api/capture")
async def api_capture(request: Request):
    username = signed_in(request)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    check_csrf(request, request.headers.get("x-csrf-token"))
    try:
        filename = await pipeline.capture_frame()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    add_audit(username, "capture", filename)
    return JSONResponse({"saved": filename})


@app.get("/api/frame.jpg")
async def api_frame(request: Request, camera: str = "merged"):
    if not signed_in(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if camera not in {"cam1", "cam2", "merged"}:
        raise HTTPException(status_code=400, detail="Unknown camera view")
    content = await pipeline.frame_jpeg(camera)
    if not content:
        snapshot = await pipeline.snapshot()
        raise HTTPException(status_code=503, detail=snapshot["status"])
    return Response(
        content=content,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket):
    if not websocket.session.get("username"):
        await websocket.close(code=1008, reason="Login required")
        return
    await websocket.accept()
    try:
        while True:
            await websocket.send_text(json.dumps(await pipeline.snapshot()))
            await asyncio.sleep(1.0)
    except Exception:
        return
