# Inspect Web App (V1)

A minimal web app for the inspection system using FastAPI + server-rendered pages.

## Features in this V1

- Login session (default seeded user)
- Dashboard with live status and image frame endpoint
- Detection history page backed by SQLite
- Manual capture endpoint saving to `component_dataset/inbox`
- Basic settings page for expected BOM values
- WebSocket status updates

## Run

From repository root:

```bash
python -m pip install -r /home/runner/work/inspect_v1.1/inspect_v1.1/web_app/requirements.txt
uvicorn web_app.app.main:app --reload --host 0.0.0.0 --port 8000
```

Open: `http://localhost:8000`

Default login:
- username: `admin`
- password: `admin123`

Set a different seed password before first run:

```bash
export INSPECT_WEB_DEFAULT_PASSWORD='your-strong-password'
```

## Notes

- Current pipeline mode is `demo` and emits synthetic live updates.
- Captured frames are saved to:
  `/home/runner/work/inspect_v1.1/inspect_v1.1/component_dataset/inbox`
- SQLite DB path:
  `/home/runner/work/inspect_v1.1/inspect_v1.1/web_app/data/inspect_web.db`
