"""Thread-safe latest inspection snapshot shared by Tk and the web server."""

from datetime import datetime, timezone
import threading
import time

import cv2


class LiveState:
    def __init__(self):
        self._lock = threading.RLock()
        self._snapshot = {
            "cameras": {
                "cam1": {"connected": False, "width": 0, "height": 0,
                         "fps": 0.0, "reconnects": 0, "error": ""},
                "cam2": {"connected": False, "width": 0, "height": 0,
                         "fps": 0.0, "reconnects": 0, "error": ""},
            },
            "counts": {name: 0 for name in "ABCDEF"},
            "expected": {"A": 5, "B": 0, "C": 0, "D": 0, "E": 2, "F": 10},
            "stable": False,
            "has_prediction": False,
            "merged_available": False,
            "result": "INCOMPLETE",
            "status": "Waiting for the Jetson detector",
            "model_status": "",
            "stitch_status": "",
            "detections": [],
            "updated_at": "",
        }
        self._frames = {"cam1": None, "cam2": None, "merged": None, "capture": None}
        self._last_requested_at = {}
        self._frame_versions = {}
        self._pending_frames = {}
        self._encode_event = threading.Event()
        self._last_publish_monotonic = 0.0
        self._encoder = threading.Thread(
            target=self._encode_loop, daemon=True, name="web-frame-encoder"
        )
        self._encoder.start()

    def _encode_loop(self):
        while True:
            self._encode_event.wait()
            self._encode_event.clear()
            with self._lock:
                pending = self._pending_frames
                self._pending_frames = {}
            for name, (frame, version) in pending.items():
                try:
                    quality = 94 if name == "capture" else 82
                    ok, buffer = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
                    )
                    if not ok:
                        continue
                    with self._lock:
                        if self._frame_versions.get(name) == version:
                            self._frames[name] = buffer.tobytes()
                except Exception:
                    continue

    def publish(self, *, cameras, counts, expected, stable, has_prediction,
                model_status, stitch_status, detections, frames=None,
                capture_frame=None, merged_available=False):
        now = time.monotonic()
        queue_encode = False
        with self._lock:
            for name, frame in (frames or {}).items():
                if frame is None:
                    version = self._frame_versions.get(name, 0) + 1
                    self._frame_versions[name] = version
                    self._frames[name] = None
                    self._pending_frames.pop(name, None)
                    self._last_requested_at[name] = now
                    continue
                min_interval = 0.22 if name in ("cam1", "cam2") else 0.18
                if now - self._last_requested_at.get(name, 0.0) < min_interval:
                    continue
                version = self._frame_versions.get(name, 0) + 1
                self._frame_versions[name] = version
                self._pending_frames[name] = (frame.copy(), version)
                self._last_requested_at[name] = now
                queue_encode = True

            if capture_frame is False:
                version = self._frame_versions.get("capture", 0) + 1
                self._frame_versions["capture"] = version
                self._frames["capture"] = None
                self._pending_frames.pop("capture", None)
                self._last_requested_at["capture"] = now
            elif (capture_frame is not None
                  and now - self._last_requested_at.get("capture", 0.0) >= 0.28):
                version = self._frame_versions.get("capture", 0) + 1
                self._frame_versions["capture"] = version
                self._pending_frames["capture"] = (capture_frame.copy(), version)
                self._last_requested_at["capture"] = now
                queue_encode = True
            if queue_encode:
                self._encode_event.set()

        camera_values = {
            name: {
                "connected": bool(value["connected"]),
                "width": int(value["width"]),
                "height": int(value["height"]),
                "fps": round(float(value["fps"]), 1),
                "reconnects": int(value["reconnects"]),
                "error": str(value["error"] or ""),
            }
            for name, value in cameras.items()
        }
        count_values = {name: int(counts.get(name, 0)) for name in "ABCDEF"}
        expected_values = {name: int(expected.get(name, 0)) for name in "ABCDEF"}
        all_cameras_ready = all(item["connected"] for item in camera_values.values())
        if (not all_cameras_ready or not merged_available or not has_prediction
                or "inference error" in str(model_status).lower()):
            result = "INCOMPLETE"
        elif not stable:
            result = "CHECKING"
        else:
            result = "PASS" if count_values == expected_values else "ERROR"
        status = "Both cameras online" if all_cameras_ready else "Waiting for both cameras"
        if has_prediction:
            status = f"{status} | {result}"
        if model_status:
            status = f"{status} | {model_status}"
        if stitch_status:
            status = f"{status} | {stitch_status}"
        clean_detections = []
        for item in detections:
            try:
                clean_detections.append({
                    "class": str(item["class"]),
                    "confidence": round(float(item["confidence"]), 4),
                    "box": [float(value) for value in item["box"]],
                })
            except (KeyError, TypeError, ValueError):
                continue

        with self._lock:
            self._snapshot = {
                "cameras": camera_values,
                "counts": count_values,
                "expected": expected_values,
                "stable": bool(stable),
                "has_prediction": bool(has_prediction),
                "merged_available": bool(merged_available),
                "result": result,
                "status": status,
                "model_status": str(model_status or ""),
                "stitch_status": str(stitch_status or ""),
                "detections": clean_detections,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._last_publish_monotonic = now

    def snapshot(self):
        with self._lock:
            result = {
                **self._snapshot,
                "cameras": {key: dict(value) for key, value in self._snapshot["cameras"].items()},
                "counts": dict(self._snapshot["counts"]),
                "expected": dict(self._snapshot["expected"]),
                "detections": [dict(value) for value in self._snapshot["detections"]],
                "frame_available": {
                    name: self._frames.get(name) is not None
                    for name in ("cam1", "cam2", "merged")
                },
            }
            if (not self._last_publish_monotonic
                    or time.monotonic() - self._last_publish_monotonic > 3.0):
                result["result"] = "INCOMPLETE"
                result["status"] = "Detector publisher is stale or unavailable"
                result["stable"] = False
                result["has_prediction"] = False
                result["frame_available"] = {
                    name: False for name in ("cam1", "cam2", "merged")
                }
            return result

    def frame_jpeg(self, camera="merged"):
        with self._lock:
            return self._frames.get(camera)

    def mark_stopped(self):
        with self._lock:
            for camera in self._snapshot["cameras"].values():
                camera["connected"] = False
            self._snapshot["result"] = "INCOMPLETE"
            self._snapshot["status"] = "Jetson detector stopped"
            self._snapshot["has_prediction"] = False
            self._snapshot["merged_available"] = False
            self._snapshot["stable"] = False
            self._snapshot["updated_at"] = datetime.now(timezone.utc).isoformat()


live_state = LiveState()
