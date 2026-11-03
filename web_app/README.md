# Jetson Inspection Web Dashboard

The browser dashboard reads frames and detection results from the same
`combined_Cameras_.py` process that owns the RTSP streams, stitching and YOLO
inference. It does not open a second camera connection or run another model.

## Start on the Jetson

Install the web dependencies in the Python environment already used by the
camera application:

```bash
python -m pip install -r web_app/requirements.txt
```

Set a session secret and an admin password, then start the existing
camera application. The desktop Tk view and browser dashboard run together:

```bash
export INSPECT_WEB_SESSION_SECRET="replace-with-a-random-secret-at-least-32-characters"
export INSPECT_WEB_DEFAULT_PASSWORD="choose-a-strong-password"
python combined_Cameras_.py
```

Open `http://<jetson-address>:8000` from a browser on the inspection network.
The username is `admin`; the password is the value of
`INSPECT_WEB_DEFAULT_PASSWORD`. The database is stored under `web_app/data/`
and is excluded from Git.

Optional settings:

- `INSPECT_WEB_HOST` (default `0.0.0.0`)
- `INSPECT_WEB_PORT` (default `8000`)
- `INSPECT_WEB_ENABLED=0` disables the embedded server
- `INSPECT_WEB_HTTPS_ONLY=1` marks the session cookie secure when HTTPS is
  terminated in front of the app
- `BOM_CAM1_URL` and `BOM_CAM2_URL` override the camera RTSP URLs

The camera process remains the source of truth. The web server shares its
latest camera frames, merged annotated view, counts, stability state and model
status in memory. Closing the camera application also stops the web server.

## Current scope

- Two camera preview frames and the merged detector view
- Live camera connection state, resolution, FPS and reconnect count
- A–F detections, confidence, stable counts and BOM PASS / ERROR / INCOMPLETE
- Manual capture of the unannotated merged detector input to
  `component_dataset/inbox/`, with a history and audit entry
- Login-protected pages and endpoints; status WebSocket requires a session

Dimensional analysis is intentionally not included in this phase. The web app
does not calculate measurements; those can be added after the detection view is
connected and stable.

## Limitations

The dashboard and Tk interface currently run in the same Python process, so
the Jetson display session must remain running. For headless startup, the next
step is to move the camera/stitch/detection workers behind a service process
and keep the same browser API. Protect port 8000 with the factory network or a
TLS reverse proxy before allowing access beyond the inspection LAN.
