import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .db import (
    add_audit,
    get_setting_json,
    get_user,
    init_db,
    list_detection_events,
    seed_defaults,
    set_setting_json,
    verify_password,
)
from .pipeline import PipelineService

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

pipeline = PipelineService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    seed_defaults()
    task = asyncio.create_task(pipeline.run())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Inspect Web App", lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("INSPECT_WEB_SESSION_SECRET", "change-this-secret"),
)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


def user_or_redirect(request: Request):
    username = request.session.get("username")
    if not username:
        return None
    return username


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if request.session.get("username"):
        return RedirectResponse(url="/dashboard", status_code=302)
    return RedirectResponse(url="/login", status_code=302)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": ""})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    user = get_user(username)
    if not user or not verify_password(password, user["password_hash"]):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid username or password"},
            status_code=401,
        )
    request.session["username"] = username
    add_audit(username, "login", "")
    return RedirectResponse(url="/dashboard", status_code=302)


@app.post("/logout")
async def logout(request: Request):
    username = request.session.get("username", "unknown")
    request.session.clear()
    add_audit(username, "logout", "")
    return RedirectResponse(url="/login", status_code=302)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    username = user_or_redirect(request)
    if not username:
        return RedirectResponse(url="/login", status_code=302)
    snapshot = await pipeline.snapshot()
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "username": username, "snapshot": snapshot},
    )


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    username = user_or_redirect(request)
    if not username:
        return RedirectResponse(url="/login", status_code=302)
    rows = list_detection_events(limit=200)
    return templates.TemplateResponse(
        "history.html", {"request": request, "username": username, "rows": rows}
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    username = user_or_redirect(request)
    if not username:
        return RedirectResponse(url="/login", status_code=302)
    expected = get_setting_json("expected_bom", {"A": 5, "B": 0, "C": 0, "D": 0, "E": 2, "F": 10})
    pipeline_mode = get_setting_json("pipeline_mode", "demo")
    return templates.TemplateResponse(
        "settings.html",
        {"request": request, "username": username, "expected": expected, "pipeline_mode": pipeline_mode},
    )


@app.post("/settings")
async def save_settings(
    request: Request,
    pipeline_mode: str = Form("demo"),
    A: int = Form(5),
    B: int = Form(0),
    C: int = Form(0),
    D: int = Form(0),
    E: int = Form(2),
    F: int = Form(10),
):
    username = user_or_redirect(request)
    if not username:
        return RedirectResponse(url="/login", status_code=302)
    expected = {"A": max(0, A), "B": max(0, B), "C": max(0, C), "D": max(0, D), "E": max(0, E), "F": max(0, F)}
    set_setting_json("expected_bom", expected)
    set_setting_json("pipeline_mode", pipeline_mode)
    add_audit(username, "settings.update", json.dumps({"expected": expected, "pipeline_mode": pipeline_mode}))
    return RedirectResponse(url="/settings", status_code=302)


@app.get("/api/status")
async def api_status(request: Request):
    if not user_or_redirect(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return JSONResponse(await pipeline.snapshot())


@app.get("/api/history")
async def api_history(request: Request, limit: int = 50):
    if not user_or_redirect(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return JSONResponse({"items": list_detection_events(limit=max(1, min(500, limit)))})


@app.post("/api/capture")
async def api_capture(request: Request):
    username = user_or_redirect(request)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    path = await pipeline.capture_frame()
    add_audit(username, "capture", path)
    return JSONResponse({"saved": path})


@app.get("/api/frame.jpg")
async def api_frame(request: Request):
    if not user_or_redirect(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    content = await pipeline.frame_jpeg()
    if not content:
        snapshot = await pipeline.snapshot()
        raise HTTPException(status_code=503, detail=f"Frame unavailable ({snapshot['status']})")
    return Response(content=content, media_type="image/jpeg")


@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_text(json.dumps(await pipeline.snapshot()))
            await asyncio.sleep(1.0)
    except Exception:
        await websocket.close()
