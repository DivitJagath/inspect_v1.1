
#!/usr/bin/env python3
"""
Two-camera RTSP viewer with a live stitched product view.

Controls:
- Fullscreen button above the merged view; Esc exits fullscreen.
- Global green dashed center lines ON/OFF at top-right.
- Independent ROI ON/OFF and FEED TEXT ON/OFF for each camera.
- Independent ROI adjustment, lock/unlock and save.
- Status text appears in a black band above each image.
- Aspect ratio is preserved when displaying the cameras.

This is a separate test program. It does not modify the protected
camera1_inspection.py program or its autostart configuration.
"""

import os
import cv2
import json
import math
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from collections import Counter, deque

# TCP retransmits damaged/missing RTSP packets instead of allowing lossy UDP
# delivery, which is a common source of H.264 macroblock decode errors.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import numpy as np
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# ============================================================
# CONFIGURATION
# ============================================================

CAM1_URL = "rtsp://admin:admin@192.168.1.10:554/profile1"
CAM2_URL = "rtsp://admin:admin@192.168.1.20:554/profile1"

PANEL_WIDTH = 640
PANEL_HEIGHT = 480
STATUS_HEIGHT = 48
CAMERA_PREVIEW_HEIGHT = 300
TOP_VIEW_LONG_EDGE = 2400
GUI_REFRESH_MS = 67
STITCH_REFRESH_SECONDS = 0.20
DETECTION_REFRESH_SECONDS = 0.40

GREEN = (0, 255, 0)       # OpenCV BGR
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)

DASH_LENGTH = 16
DASH_GAP = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "roi_settings.json")
STITCH_PATH = os.path.join(SCRIPT_DIR, "stitch_settings.json")
MODEL_RELATIVE_PATH = "best.pt"
MODEL_PATH = os.environ.get(
    "BOM_MODEL_PATH", os.path.join(SCRIPT_DIR, MODEL_RELATIVE_PATH)
)
DATASET_INBOX = os.path.join(SCRIPT_DIR, "component_dataset", "inbox")
EXPECTED_BOM = {"A": 5, "B": 0, "C": 0, "D": 0, "E": 2, "F": 10}
COMPONENTS = tuple(EXPECTED_BOM)
MODEL_CONFIDENCE = 0.30
MODEL_IMAGE_SIZE = 1280
MODEL_MAX_DETECTIONS = 40
CLASS_CONFIDENCE_THRESHOLDS = {
    "A": 0.45, "B": 0.45, "C": 0.45, "D": 0.45, "E": 0.45, "F": 0.62,
}
CLASS_COLORS = {
    "A": (255, 150, 0), "B": (0, 220, 255), "C": (255, 0, 255),
    "D": (0, 255, 0), "E": (0, 100, 255), "F": (255, 255, 0),
}

CAMERA_DEFAULT = {
    "x1": 280,
    "x2": 280,
    "x3": 280,
    "x4": 280,
    "top_y": 80,
    "bottom_y": 400,
}

DEFAULT_SETTINGS = {
    "cam1": CAMERA_DEFAULT.copy(),
    "cam2": CAMERA_DEFAULT.copy(),
}


# ============================================================
# SETTINGS
# ============================================================

def load_settings():
    settings = {
        name: values.copy()
        for name, values in DEFAULT_SETTINGS.items()
    }

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            saved = json.load(file)

        # Support older settings containing only one ROI.
        if all(key in saved for key in CAMERA_DEFAULT):
            for camera in settings:
                for key in CAMERA_DEFAULT:
                    settings[camera][key] = int(saved.get(key, CAMERA_DEFAULT[key]))
        else:
            for camera in settings:
                values = saved.get(camera, {})
                if isinstance(values, dict):
                    for key in CAMERA_DEFAULT:
                        settings[camera][key] = int(
                            values.get(key, CAMERA_DEFAULT[key])
                        )

    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass

    return settings


def save_settings(settings):
    temp_path = CONFIG_PATH + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2)

    os.replace(temp_path, CONFIG_PATH)


# ============================================================
# RTSP CAMERA THREAD
# ============================================================

class CameraStream:
    def __init__(self, name, url):
        self.name = name
        self.url = url

        self.lock = threading.Lock()
        self.frame = None
        self.connected = False
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self.running = True

        self.thread = threading.Thread(
            target=self._reader,
            daemon=True
        )
        self.thread.start()

    def _reader(self):
        capture = None
        fps_count = 0
        fps_start = time.monotonic()
        reconnect_delay = 0.5

        def mark_disconnected():
            with self.lock:
                self.connected = False
                self.frame = None

        while self.running:
            try:
                if capture is None or not capture.isOpened():
                    if capture is not None:
                        capture.release()

                    # Bound network stalls so an RTSP outage does not leave a
                    # reader thread stuck for OpenCV's default 30 seconds.
                    timeout_params = []
                    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                        timeout_params.extend((cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000))
                    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                        timeout_params.extend((cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000))
                    try:
                        capture = (cv2.VideoCapture(self.url, cv2.CAP_FFMPEG,
                                                    timeout_params)
                                   if timeout_params else
                                   cv2.VideoCapture(self.url, cv2.CAP_FFMPEG))
                    except (TypeError, cv2.error):
                        # Older OpenCV builds do not accept open-time options.
                        capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

                    if not capture.isOpened():
                        mark_disconnected()
                        time.sleep(reconnect_delay)
                        reconnect_delay = min(10.0, reconnect_delay * 1.7)
                        continue

                    # These settings are best-effort and backend-dependent.
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                ok, frame = capture.read()

                if not ok or frame is None:
                    mark_disconnected()
                    capture.release()
                    capture = None
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(10.0, reconnect_delay * 1.7)
                    continue

                height, width = frame.shape[:2]
                fps_count += 1
                now = time.monotonic()

                if now - fps_start >= 1.0:
                    fps_value = fps_count / (now - fps_start)
                    fps_count = 0
                    fps_start = now
                else:
                    fps_value = None

                with self.lock:
                    self.frame = frame
                    self.connected = True
                    self.width = width
                    self.height = height
                    if fps_value is not None:
                        self.fps = fps_value
                reconnect_delay = 0.5

            except Exception:
                mark_disconnected()
                if capture is not None:
                    capture.release()
                    capture = None
                time.sleep(reconnect_delay)
                reconnect_delay = min(10.0, reconnect_delay * 1.7)

        if capture is not None:
            capture.release()

    def get_data(self):
        with self.lock:
            # Captured frames are replaced, never mutated. Returning the
            # immutable current reference avoids copying two full-resolution
            # RTSP frames on every GUI refresh.
            frame = self.frame
            return (
                frame,
                self.connected,
                self.fps,
                self.width,
                self.height,
            )

    def stop(self):
        self.running = False
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=2.0)


# ============================================================
# YOLO COMPONENT DETECTION AND TEMPORAL STABILITY
# ============================================================

def detection_iou(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(
        0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(
        0.0, float(box_b[3]) - float(box_b[1]))
    return intersection / max(area_a + area_b - intersection, 1e-6)


def suppress_detection_duplicates(detections, frame_width, frame_height):
    """Remove overlapping YOLO boxes; counts never force or add detections."""
    kept = []
    diagonal = max(math.hypot(frame_width, frame_height), 1.0)
    for det in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        duplicate = False
        for other in kept:
            if detection_iou(det["box"], other["box"]) >= 0.45:
                duplicate = True
                break
            if det["class"] == other["class"]:
                distance = math.hypot(det["cx"] - other["cx"], det["cy"] - other["cy"])
                if distance / diagonal <= 0.018:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(det)
    return kept


def plausible_component_detection(frame, name, box):
    """Reject overlay text, tiny marks, and non-round F lookalikes."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        return False

    # The camera timestamp and feed text sit at the top edge, outside the
    # useful product area. The product/components in this installation are
    # well below that overlay band.
    if (y1 + y2) * 0.5 < height * 0.085:
        return False

    if name == "F":
        # Studs are small and approximately circular. Chalk strokes and date
        # glyphs may score as F, but fail these scale/shape checks.
        if not (width * 0.009 <= box_w <= width * 0.055 and
                height * 0.012 <= box_h <= height * 0.065):
            return False
        aspect = box_w / box_h
        if not 0.68 <= aspect <= 1.46:
            return False

        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        side = max(box_w, box_h) * 1.65
        left = max(0, int(round(cx - side * 0.5)))
        top = max(0, int(round(cy - side * 0.5)))
        right = min(width, int(round(cx + side * 0.5)))
        bottom = min(height, int(round(cy + side * 0.5)))
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_CUBIC)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.0, minDist=20,
            param1=100, param2=19, minRadius=12, maxRadius=39,
        )
        if circles is None:
            return False
        # Require a circular rim near the predicted object's center. A chalk
        # cross can be bright and square, but does not form this circular rim.
        return any(math.hypot(float(circle[0]) - 48.0,
                              float(circle[1]) - 48.0) <= 17.0
                   for circle in circles[0])

    # Reject tiny timestamp/date fragments for other classes as well.
    if box_w < width * 0.008 or box_h < height * 0.010:
        return False
    return True


class TemporalComponentTracker:
    """Associate by position over time while retaining YOLO class evidence."""

    def __init__(self):
        self.next_id = 1
        self.tracks = {}

    def update(self, detections, frame_width, frame_height):
        pairs = []
        diagonal = max(math.hypot(frame_width, frame_height), 1.0)
        for track_id, track in self.tracks.items():
            tx = (track["box"][0] + track["box"][2]) / 2.0
            ty = (track["box"][1] + track["box"][3]) / 2.0
            for index, det in enumerate(detections):
                distance = math.hypot(det["cx"] - tx, det["cy"] - ty) / diagonal
                if distance <= 0.025:
                    pairs.append((distance, track_id, index))

        used_tracks, used_detections = set(), set()
        for _, track_id, index in sorted(pairs):
            if track_id in used_tracks or index in used_detections:
                continue
            track = self.tracks[track_id]
            det = detections[index]
            alpha = 0.30
            track["box"] = (1.0 - alpha) * track["box"] + alpha * det["box"]
            track["hits"] += 1
            track["misses"] = 0
            track["votes"][det["class"]] += max(0.05, det["confidence"] ** 2)
            track["confidence"] = det["confidence"]
            used_tracks.add(track_id)
            used_detections.add(index)

        for track_id, track in list(self.tracks.items()):
            if track_id not in used_tracks:
                track["misses"] += 1
            if track["misses"] > 4:
                del self.tracks[track_id]

        for index, det in enumerate(detections):
            if index in used_detections:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {
                "box": np.asarray(det["box"], dtype=np.float32),
                "hits": 1,
                "misses": 0,
                "confidence": det["confidence"],
                "votes": Counter({det["class"]: max(0.05, det["confidence"] ** 2)}),
            }

        output = []
        for track_id, track in self.tracks.items():
            if track["hits"] < 3 or track["misses"] > 2 or not track["votes"]:
                continue
            label, vote = track["votes"].most_common(1)[0]
            total = sum(track["votes"].values())
            box = track["box"].copy()
            x1, y1, x2, y2 = box.tolist()
            output.append({
                "class": label,
                "confidence": min(0.99, vote / max(total, 1e-6)),
                "box": box,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "track_id": track_id,
                "hits": track["hits"],
            })
        return output


class DetectionWorker:
    """Run YOLO off the Tk thread and always process the newest merged frame."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop_event = threading.Event()
        self.pending_frame = None
        self.detections = []
        self.counts = {name: 0 for name in COMPONENTS}
        self.status = "Loading detector..."
        self.has_prediction = False
        self.bom_stable = False
        self.history = deque(maxlen=6)
        self.tracker = TemporalComponentTracker()
        self.generation = 0
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name="component-yolo-worker")
        self.thread.start()

    def submit(self, frame):
        if frame is None or self.stop_event.is_set():
            return
        with self.lock:
            self.pending_frame = frame.copy()
        self.wake.set()

    def snapshot(self):
        with self.lock:
            detections = [dict(item, box=item["box"].copy())
                          for item in self.detections]
            return (detections, dict(self.counts), self.status,
                    self.has_prediction, self.bom_stable)

    def reset_for_camera_loss(self):
        """Discard old counts when a camera drops so stale BOM data cannot pass."""
        with self.lock:
            self.generation += 1
            self.pending_frame = None
            self.detections = []
            self.counts = {name: 0 for name in COMPONENTS}
            self.has_prediction = False
            self.bom_stable = False
            self.history.clear()
            self.tracker = TemporalComponentTracker()
            self.status = "Waiting for both camera feeds"

    def close(self):
        self.stop_event.set()
        self.wake.set()

    def _run(self):
        if YOLO is None:
            with self.lock:
                self.status = "Ultralytics is not installed"
            return

        candidates = [self.model_path, os.path.join(SCRIPT_DIR, "best.pt")]
        path = next((candidate for candidate in candidates if candidate and
                     os.path.isfile(candidate)), None)
        if path is None:
            with self.lock:
                self.status = "Model weights missing; put best.pt beside this script"
            return

        try:
            model = YOLO(path)
            try:
                import torch
                device = 0 if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
            with self.lock:
                self.status = f"Model ready ({os.path.basename(path)})"
        except Exception as exc:
            with self.lock:
                self.status = f"Model load failed: {exc}"
            return

        while not self.stop_event.is_set():
            self.wake.wait(0.5)
            self.wake.clear()
            if self.stop_event.is_set():
                break
            with self.lock:
                frame = self.pending_frame
                self.pending_frame = None
                generation = self.generation
            if frame is None:
                continue
            try:
                results = model.predict(
                    source=frame, conf=MODEL_CONFIDENCE, iou=0.50,
                    imgsz=MODEL_IMAGE_SIZE, max_det=MODEL_MAX_DETECTIONS,
                    device=device, verbose=False,
                )
                result = results[0]
                names = result.names
                raw = []
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        name = (names.get(class_id, "") if isinstance(names, dict)
                                else names[class_id] if class_id < len(names) else "")
                        name = str(name).upper()
                        confidence = float(box.conf.item())
                        if (name not in COMPONENTS or confidence <
                                CLASS_CONFIDENCE_THRESHOLDS.get(name, MODEL_CONFIDENCE)):
                            continue
                        coords = box.xyxy[0].detach().cpu().numpy().astype(np.float32)
                        x1, y1, x2, y2 = coords.tolist()
                        if not plausible_component_detection(frame, name, coords):
                            continue
                        raw.append({
                            "class": name, "confidence": confidence, "box": coords,
                            "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
                        })
                frame_h, frame_w = frame.shape[:2]
                raw = suppress_detection_duplicates(raw, frame_w, frame_h)
                with self.lock:
                    if generation != self.generation:
                        continue
                    stable = self.tracker.update(raw, frame_w, frame_h)
                    counts = {name: sum(item["class"] == name for item in stable)
                              for name in COMPONENTS}
                    signature = tuple(counts[name] for name in COMPONENTS)
                    self.history.append(signature)
                    bom_stable = (len(self.history) == self.history.maxlen and
                                  len(set(self.history)) == 1)
                    self.detections = stable
                    self.counts = counts
                    self.has_prediction = True
                    self.bom_stable = bom_stable
                    self.status = "Live detection | BOM stable" if bom_stable else "Live detection | stabilizing"
            except Exception as exc:
                with self.lock:
                    self.status = f"Inference error: {exc}"


# ============================================================
# AUTOMATIC HORIZONTAL STITCHING AND OVERLAP REMOVAL
# ============================================================

class LiveStitcher:
    """Automatically align the fixed camera views and composite them once."""

    def __init__(self):
        self.H = None
        self.T = None
        self.output_size = None
        self.mask1 = None
        self.mask2 = None
        self.previous_seam = None
        self.fixed_seam = None
        self.rectified_corners = None
        self.rectify_samples = []
        self.rectify_angle = None
        self.status = "Automatically aligning live camera views..."
        self.last_attempt = 0.0

    def reset(self):
        self.H = None
        self.T = None
        self.output_size = None
        self.mask1 = None
        self.mask2 = None
        self.previous_seam = None
        self.fixed_seam = None
        self.rectified_corners = None
        self.rectify_samples = []
        self.rectify_angle = None
        self.status = "Automatically aligning the two live views..."
        self.last_attempt = 0.0

    @staticmethod
    def _corners(width, height):
        return np.float32([[0, 0], [width - 1, 0],
                           [width - 1, height - 1], [0, height - 1]]).reshape(-1, 1, 2)

    def _estimate(self, frame1, frame2):
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        if (w1, h1) != (w2, h2):
            raise RuntimeError("Camera frame sizes differ; automatic alignment paused")

        scale = min(1.0, 1500.0 / max(w1, h1))
        small1 = cv2.resize(frame1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small2 = cv2.resize(frame2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        g1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
        # Ignore camera text at the very top and edges while finding matches.
        feature_mask = np.zeros_like(g1)
        margin_x = int(g1.shape[1] * 0.025)
        margin_y = int(g1.shape[0] * 0.055)
        feature_mask[margin_y:g1.shape[0] - margin_y,
                     margin_x:g1.shape[1] - margin_x] = 255
        orb = cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8,
                             edgeThreshold=19, fastThreshold=7)
        kp1, des1 = orb.detectAndCompute(g1, feature_mask)
        kp2, des2 = orb.detectAndCompute(g2, feature_mask)
        if des1 is None or des2 is None:
            raise RuntimeError("Not enough image detail in the camera overlap")

        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des2, des1, k=2)
        good = [pair[0] for pair in pairs
                if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
        if len(good) < 20:
            raise RuntimeError(f"Only {len(good)} matching features; need a clearer shared view")

        inv_scale = 1.0 / scale
        points2 = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) * inv_scale
        points1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2) * inv_scale
        H, inliers = cv2.findHomography(points2, points1, cv2.RANSAC, 8.0)
        inlier_count = int(inliers.sum()) if inliers is not None else 0
        inlier_ratio = inlier_count / max(1, len(good))
        if H is None or inlier_count < 18 or inlier_ratio < 0.22:
            raise RuntimeError(f"Automatic alignment confidence is low ({inlier_count} matches)")

        corners1 = self._corners(w1, h1)
        corners2 = cv2.perspectiveTransform(self._corners(w2, h2), H)
        all_corners = np.concatenate((corners1, corners2), axis=0)
        min_xy = np.floor(all_corners.min(axis=0).ravel()).astype(int)
        max_xy = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
        out_w, out_h = (max_xy - min_xy).tolist()
        if (out_w < w1 or out_h < h1 * 0.65 or
                out_w > w1 * 2.5 or out_h > h1 * 2.0):
            raise RuntimeError("Automatic transform is unstable; no merge was shown")

        T = np.array([[1.0, 0.0, -min_xy[0]],
                      [0.0, 1.0, -min_xy[1]],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        H2 = T @ H
        m1 = cv2.warpPerspective(np.full((h1, w1), 255, np.uint8), T, (out_w, out_h)) > 0
        m2 = cv2.warpPerspective(np.full((h2, w2), 255, np.uint8), H2, (out_w, out_h)) > 0
        overlap = m1 & m2
        overlap_ratio = float(overlap.sum()) / max(1, min(m1.sum(), m2.sum()))
        if overlap_ratio < 0.08:
            raise RuntimeError("The two feeds do not show enough shared area to stitch")

        self.H = H2
        self.T = T
        self.output_size = (out_w, out_h)
        self.mask1 = m1
        self.mask2 = m2
        self.previous_seam = None
        self.fixed_seam = None
        self.rectified_corners = None
        self.rectify_samples = []
        self.status = f"Auto-aligned | {inlier_count} matches | overlap shown once"

    def _rectify_product(self, image):
        """Find the reddish product panel and warp its outline to a rectangle."""
        height, width = image.shape[:2]
        if self.rectified_corners is None:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, saturation, value = cv2.split(hsv)
            red = (((hue <= 16) | (hue >= 170)) &
                   (saturation >= 28) & (value >= 35)).astype(np.uint8) * 255
            k = max(9, int(min(height, width) * 0.025))
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < image.shape[0] * image.shape[1] * 0.12:
                    continue
                hull = cv2.convexHull(contour)
                perimeter = cv2.arcLength(hull, True)
                for epsilon_ratio in (0.01, 0.015, 0.02, 0.03, 0.045, 0.06):
                    approx = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        candidates.append((area, approx.reshape(4, 2).astype(np.float32)))
                        break
            if not candidates:
                return image, False

            _, points = max(candidates, key=lambda item: item[0])
            # Order the four corners consistently: top-left, top-right,
            # bottom-right, bottom-left.
            sums = points.sum(axis=1)
            diffs = points[:, 0] - points[:, 1]
            corners = np.array([
                points[np.argmin(sums)], points[np.argmax(diffs)],
                points[np.argmax(sums)], points[np.argmin(diffs)]
            ], dtype=np.float32)
            # Require five consecutive detections and take their median so a
            # single noisy frame cannot lock a bad crop.
            self.rectify_samples.append(corners.copy())
            self.rectify_samples = self.rectify_samples[-5:]
            if len(self.rectify_samples) < 5:
                return image, None
            corners = np.median(np.stack(self.rectify_samples), axis=0).astype(np.float32)
        else:
            corners = self.rectified_corners.copy()
        raw_corners = corners.copy()
        # The red panel mask can stop a little inside the outer metal rim.
        # Expand the detected quadrilateral so the complete top and bottom
        # edges remain in the corrected image.
        center = corners.mean(axis=0)
        # Expand beyond the detected red face to retain the full outer frame,
        # especially the lower-left corner when the color mask misses a rim.
        corners[:, 0] = center[0] + 1.12 * (corners[:, 0] - center[0])
        corners[:, 1] = center[1] + 1.18 * (corners[:, 1] - center[1])
        corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)
        tl, tr, br, bl = corners
        out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        if out_w < 100 or out_h < 100 or out_w > width * 1.2 or out_h > height * 1.2:
            return image, False
        ratio = out_w / float(out_h)
        if not 0.35 <= ratio <= 3.5:
            return image, False
        if self.rectified_corners is None:
            # Lock only after a multi-frame boundary has passed geometry checks.
            self.rectified_corners = raw_corners
        target = np.float32([[0, 0], [out_w - 1, 0],
                             [out_w - 1, out_h - 1], [0, out_h - 1]])
        transform = cv2.getPerspectiveTransform(corners, target)
        rectified = cv2.warpPerspective(image, transform, (out_w, out_h))
        if self.rectify_angle is None:
            # The red boundary can be a few pixels irregular. Estimate the
            # residual roll from long horizontal rails/board lines once, then
            # hold that correction for the rest of the live session.
            gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 45, 135)
            min_length = max(100, int(out_w * 0.30))
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 720, threshold=max(60, min_length // 3),
                minLineLength=min_length, maxLineGap=max(20, out_w // 80)
            )
            angles = []
            if lines is not None:
                # OpenCV normally returns (N, 1, 4), but some builds return
                # (N, 4). Flatten the result so both layouts are handled.
                line_values = np.asarray(lines).reshape(-1, 4)
                for x1, y1, x2, y2 in line_values:
                    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    if abs(angle) <= 12.0:
                        angles.append(angle)
            self.rectify_angle = (
                float(np.median(angles)) if len(angles) >= 2 else 0.0
            )

        angle = self.rectify_angle
        if abs(angle) < 0.15:
            return rectified, True
        # Rotate onto an expanded canvas to keep the complete product visible.
        pad_x = max(8, int(out_w * 0.04))
        pad_y = max(8, int(out_h * 0.08))
        canvas_w, canvas_h = out_w + 2 * pad_x, out_h + 2 * pad_y
        rotation = cv2.getRotationMatrix2D((out_w / 2.0, out_h / 2.0), angle, 1.0)
        rotation[0, 2] += pad_x
        rotation[1, 2] += pad_y
        leveled = cv2.warpAffine(
            rectified, rotation, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        return leveled, True

    def _find_seam(self, first, second, overlap):
        # Find a low-detail vertical path through the common view so the cut
        # avoids component edges where possible and never blends a part twice.
        seam_w = min(700, overlap.shape[1])
        seam_h = min(360, overlap.shape[0])
        small1 = cv2.resize(first, (seam_w, seam_h), interpolation=cv2.INTER_AREA)
        small2 = cv2.resize(second, (seam_w, seam_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(overlap.astype(np.uint8), (seam_w, seam_h),
                          interpolation=cv2.INTER_NEAREST) > 0
        gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(gray1 - gray2)
        edge1x = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
        edge1y = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
        edge2x = cv2.Sobel(gray2, cv2.CV_32F, 1, 0, ksize=3)
        edge2y = cv2.Sobel(gray2, cv2.CV_32F, 0, 1, ksize=3)
        energy = diff + 0.18 * (cv2.magnitude(edge1x, edge1y) +
                                cv2.magnitude(edge2x, edge2y))
        energy[~mask] = 1e6
        # Keep the seam near the center of the actual overlap.
        overlap_x = np.where(mask.any(axis=0))[0]
        if len(overlap_x) == 0:
            return np.full(overlap.shape[0], overlap.shape[1] // 2, dtype=np.int32)
        center = (int(overlap_x[0]) + int(overlap_x[-1])) // 2
        band = max(10, int((overlap_x[-1] - overlap_x[0]) * 0.2))
        energy[:, :max(0, center - band)] += 1e5
        energy[:, min(seam_w, center + band + 1):] += 1e5

        costs = np.full_like(energy, 1e12, dtype=np.float32)
        back = np.zeros((seam_h, seam_w), dtype=np.int16)
        costs[0] = energy[0]
        for y in range(1, seam_h):
            prev = costs[y - 1]
            for step in (-2, -1, 0, 1, 2):
                shifted = np.roll(prev, step)
                better = shifted < costs[y]
                costs[y, better] = shifted[better]
                back[y, better] = step
            costs[y] += energy[y]
        path = np.zeros(seam_h, dtype=np.float32)
        path[-1] = int(np.argmin(costs[-1]))
        for y in range(seam_h - 1, 0, -1):
            path[y - 1] = path[y] - back[y, int(path[y])]
        cols = np.interp(np.linspace(0, seam_h - 1, overlap.shape[0]),
                         np.arange(seam_h), path)
        # Smooth row-to-row changes and keep the seam stable between frames.
        cols = cv2.GaussianBlur(cols.astype(np.float32).reshape(-1, 1), (1, 21), 0).ravel()
        cols *= overlap.shape[1] / float(seam_w)
        if self.previous_seam is not None and len(self.previous_seam) == len(cols):
            cols = 0.75 * self.previous_seam + 0.25 * cols
        self.previous_seam = cols
        return np.clip(np.round(cols), 0, overlap.shape[1] - 1).astype(np.int32)

    def stitch(self, frame1, frame2):
        if frame1 is None or frame2 is None:
            self.status = "Waiting for both camera frames"
            return None
        if self.H is None:
            now = time.monotonic()
            if now - self.last_attempt < 2.0:
                return None
            self.last_attempt = now
            try:
                self._estimate(frame1, frame2)
            except Exception as exc:
                self.status = f"Auto-align pending: {exc}"
                return None

        first = cv2.warpPerspective(frame1, self.T, self.output_size)
        second = cv2.warpPerspective(frame2, self.H, self.output_size)
        overlap = self.mask1 & self.mask2
        # Select the cut once after alignment and hold it in panorama
        # coordinates. Recomputing it from every live frame makes it flicker
        # as lighting, timestamps, and small scene changes alter image costs.
        if self.fixed_seam is None or len(self.fixed_seam) != overlap.shape[0]:
            self.fixed_seam = self._find_seam(first, second, overlap)
        seam = self.fixed_seam
        yy, xx = np.indices(self.output_size[::-1])
        choose1 = self.mask1 & ((xx <= seam[:, None]) | ~self.mask2)
        choose2 = self.mask2 & ~choose1
        merged = np.zeros_like(first)
        merged[choose1] = first[choose1]
        merged[choose2] = second[choose2]
        valid = choose1 | choose2
        ys, xs = np.where(valid)
        if len(xs) == 0:
            self.status = "Automatic merge has no usable image area"
            return None
        stitched = merged[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        product, rectified = self._rectify_product(stitched)
        if rectified is True:
            self.status = "Automatic merge | cut line locked | product corrected to rectangle"
        elif rectified is None:
            self.status = "Automatic merge | stabilizing product boundary..."
        else:
            self.status = "Automatic merge | cut line locked | product boundary not found"
        return product


# MAIN GUI
# ============================================================

class CombinedCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Combined Camera Viewer")
        self.root.geometry("1600x950")
        self.root.configure(bg="#202020")

        self.closing = False
        self.fullscreen = False
        self.show_center_lines = True

        self.settings = load_settings()

        self.roi_visible = {"cam1": True, "cam2": True}
        self.text_visible = {"cam1": True, "cam2": True}
        self.locked = {"cam1": False, "cam2": False}

        self.roi_vars = {}
        self.roi_buttons = {}
        self.text_buttons = {}
        self.camera_labels = {}
        self.tk_images = {}
        self.merged_toplevel = None
        self.merged_fullscreen_label = None
        self.bom_toplevel = None
        self.bom_fullscreen_table = None
        self.stitcher = LiveStitcher()
        self.detector = DetectionWorker(MODEL_PATH)
        self.last_stitch_time = 0.0
        self.last_detector_submit_time = 0.0
        self.last_merged = None
        self._feeds_ready = False
        self._last_gui_error_time = 0.0
        self._last_bom_rows = None
        self._last_bom_summary = None

        self.cam1 = CameraStream("CAM 1", CAM1_URL)
        self.cam2 = CameraStream("CAM 2", CAM2_URL)

        self._build_ui()

        self.root.bind("<Escape>", self._escape_fullscreen)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.update_view()

    def _build_ui(self):
        # Top-level toolbar.
        self.toolbar = tk.Frame(self.root, bg="#202020")
        self.toolbar.pack(fill="x", padx=8, pady=8)

        ttk.Label(
            self.toolbar,
            text="LIVE CAMERA FEEDS + AUTOMATIC HORIZONTAL STITCH",
            font=("Arial", 14, "bold"),
            background="#202020",
            foreground="white"
        ).pack(side="left", padx=12)

        # Top-right actions.
        self.recalibrate_button = ttk.Button(
            self.toolbar,
            text="RESTART AUTO ALIGNMENT",
            command=self.restart_auto_alignment
        )
        self.recalibrate_button.pack(side="right", padx=(8, 0))

        self.center_button = ttk.Button(
            self.toolbar,
            text="CENTER LINES: ON",
            command=self.toggle_center_lines
        )
        self.center_button.pack(side="right", padx=(8, 0))

        # Main body.
        self.body = tk.Frame(self.root, bg="#202020")
        self.body.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.controls = tk.Frame(self.body, bg="#303030", width=310)
        self.controls.pack(side="left", fill="y", padx=(0, 8))
        self.controls.pack_propagate(False)

        self.view = tk.Frame(self.body, bg="black")
        self.view.pack(side="left", fill="both", expand=True)

        self.view_header_var = tk.StringVar(value="Waiting for cameras...")
        ttk.Label(
            self.view,
            textvariable=self.view_header_var,
            background="#202020",
            foreground="white",
            font=("Arial", 10, "bold")
        ).pack(fill="x", pady=(0, 5))

        # Live feeds appear beside the horizontal merged product view.
        self.display_area = tk.Frame(self.view, bg="#202020")
        self.display_area.pack(fill="both", expand=True)
        self.live_column = tk.Frame(self.display_area, bg="#202020")
        self.live_column.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.merge_column = tk.Frame(self.display_area, bg="#111111", width=640)
        self.merge_column.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self.merge_column.pack_propagate(False)

        for camera, title in (("cam1", "CAMERA 1 - LIVE"),
                              ("cam2", "CAMERA 2 - LIVE")):
            pane = tk.Frame(self.live_column, bg="#111111")
            pane.pack(fill="both", expand=True, pady=4)
            tk.Label(pane, text=title, bg="#202020", fg="white",
                     font=("Arial", 10, "bold")).pack(fill="x", padx=4, pady=4)
            label = tk.Label(pane, bg="black", bd=0)
            label.pack(fill="both", expand=True, padx=4, pady=(0, 4))
            self.camera_labels[camera] = label

        merge_toolbar = tk.Frame(self.merge_column, bg="#111111")
        merge_toolbar.pack(fill="x", padx=5, pady=5)
        self.merge_fullscreen_button = ttk.Button(
            merge_toolbar, text="FULL SCREEN MERGED PHOTO",
            command=self.toggle_merged_fullscreen
        )
        self.merge_fullscreen_button.pack(side="left", fill="x", expand=True)
        ttk.Button(merge_toolbar, text="SAVE TRAINING FRAME",
                   command=self.save_training_frame).pack(side="left", padx=(6, 0))
        self.merge_status_var = tk.StringVar(value="Waiting for camera frames")
        tk.Label(self.merge_column, textvariable=self.merge_status_var,
                 bg="#111111", fg="#b9e8b9", font=("Arial", 9),
                 wraplength=410, justify="left").pack(fill="x", padx=6, pady=(0, 4))
        tk.Label(self.merge_column, text="HORIZONTAL MERGED PRODUCT VIEW",
                 bg="#202020", fg="white", font=("Arial", 10, "bold")).pack(fill="x", padx=5, pady=(2, 4))
        bom_frame = tk.LabelFrame(self.merge_column, text="BOM_RESULT TABLE",
                                  bg="#202020", fg="white", padx=4, pady=3)
        bom_frame.pack(side="bottom", fill="x", padx=5, pady=(0, 5))
        bom_header = tk.Frame(bom_frame, bg="#202020")
        bom_header.pack(fill="x", pady=(0, 3))
        tk.Label(bom_header, text="BOM_RESULT TABLE", bg="#202020", fg="white",
                 font=("Arial", 9, "bold")).pack(side="left")
        ttk.Button(bom_header, text="FULL SCREEN TABLE",
                   command=self.toggle_bom_fullscreen).pack(side="right")
        bom_style = ttk.Style()
        bom_style.configure("BOM.Treeview", rowheight=22, font=("Arial", 8))
        bom_style.configure("BOM.Treeview.Heading", font=("Arial", 8, "bold"))
        columns = ("sr", "component", "bom", "detected", "result")
        self.bom_table = ttk.Treeview(bom_frame, columns=columns, show="headings",
                                      height=5, style="BOM.Treeview")
        for column, title, width in (
            ("sr", "Sr.No", 48),
            ("component", "Component", 82),
            ("bom", "BOM_Quantity", 105),
            ("detected", "Detected_quantity", 120),
            ("result", "PASS/ERROR result", 145),
        ):
            self.bom_table.heading(column, text=title)
            self.bom_table.column(column, width=width, minwidth=42,
                                  anchor="center", stretch=True)
        self.bom_table.tag_configure("pass", foreground="#147a25")
        self.bom_table.tag_configure("error", foreground="#b3261e")
        self.bom_table.tag_configure("waiting", foreground="#8a6700")
        self.bom_row_ids = {}
        for index, name in enumerate(COMPONENTS, 1):
            self.bom_row_ids[name] = self.bom_table.insert(
                "", "end", values=(str(index), name,
                                    str(EXPECTED_BOM[name]) if EXPECTED_BOM[name] is not None else "Set qty",
                                    "—", "WAITING"), tags=("waiting",)
            )
        self.bom_table.pack(fill="x", expand=True)
        self.bom_summary_var = tk.StringVar(value="Waiting for model weights")
        tk.Label(bom_frame, textvariable=self.bom_summary_var,
                 bg="#202020", fg="#f0d060", font=("Arial", 9, "bold"),
                 anchor="w").pack(fill="x", padx=3, pady=(3, 0))
        self.merged_label = tk.Label(self.merge_column, bg="black", bd=0,
                                     text="Waiting for automatic camera alignment...",
                                     fg="white")
        self.merged_label.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        self._build_controls()

    def _build_controls(self):
        tk.Label(
            self.controls,
            text="CAMERA CONTROLS",
            bg="#303030",
            fg="white",
            font=("Arial", 12, "bold")
        ).pack(fill="x", padx=8, pady=(10, 8))

        for camera, title in (
            ("cam1", "CAMERA 1 ROI"),
            ("cam2", "CAMERA 2 ROI"),
        ):
            section = tk.LabelFrame(
                self.controls,
                text=title,
                bg="#303030",
                fg="white",
                padx=6,
                pady=5
            )
            section.pack(fill="x", padx=7, pady=6)

            button_row = tk.Frame(section, bg="#303030")
            button_row.pack(fill="x", pady=3)

            self.roi_buttons[camera] = ttk.Button(
                button_row,
                text="ROI: ON",
                command=lambda c=camera: self.toggle_roi(c)
            )
            self.roi_buttons[camera].pack(
                side="left", expand=True, fill="x", padx=(0, 3)
            )

            self.text_buttons[camera] = ttk.Button(
                button_row,
                text="FEED TEXT: ON",
                command=lambda c=camera: self.toggle_text(c)
            )
            self.text_buttons[camera].pack(
                side="left", expand=True, fill="x", padx=(3, 0)
            )

            for key in CAMERA_DEFAULT:
                row = tk.Frame(section, bg="#303030")
                row.pack(fill="x", pady=2)

                label = key.replace("_", " ").upper()
                tk.Label(
                    row,
                    text=label,
                    width=12,
                    anchor="w",
                    bg="#303030",
                    fg="white"
                ).pack(side="left")

                var = tk.IntVar(
                    value=self.settings[camera][key]
                )
                self.roi_vars[(camera, key)] = var

                ttk.Button(
                    row,
                    text="-",
                    width=3,
                    command=lambda c=camera, k=key: self.adjust_roi(c, k, -1)
                ).pack(side="left", padx=2)

                tk.Label(
                    row,
                    textvariable=var,
                    width=5,
                    bg="#303030",
                    fg="white"
                ).pack(side="left", padx=3)

                ttk.Button(
                    row,
                    text="+",
                    width=3,
                    command=lambda c=camera, k=key: self.adjust_roi(c, k, 1)
                ).pack(side="left", padx=2)

            lock_row = tk.Frame(section, bg="#303030")
            lock_row.pack(fill="x", pady=(6, 2))

            ttk.Button(
                lock_row,
                text="LOCK / UNLOCK ROI",
                command=lambda c=camera: self.toggle_lock(c)
            ).pack(side="left", expand=True, fill="x", padx=(0, 3))

            ttk.Button(
                lock_row,
                text="SAVE ROI",
                command=self.save_roi
            ).pack(side="left", expand=True, fill="x", padx=(3, 0))

        ttk.Button(
            self.controls,
            text="SAVE ALL SETTINGS",
            command=self.save_roi
        ).pack(fill="x", padx=8, pady=10)

        ttk.Label(
            self.controls,
            text=f"Settings file:\n{CONFIG_PATH}",
            wraplength=280,
            background="#303030",
            foreground="white"
        ).pack(fill="x", padx=8, pady=5)

    # --------------------------------------------------------
    # BUTTON ACTIONS
    # --------------------------------------------------------

    def toggle_center_lines(self):
        self.show_center_lines = not self.show_center_lines
        self.center_button.configure(
            text="CENTER LINES: ON" if self.show_center_lines
            else "CENTER LINES: OFF"
        )

    def restart_auto_alignment(self):
        """Discard the current live estimate and automatically find it again."""
        self.stitcher.reset()
        self.last_merged = None
        self.last_stitch_time = 0.0
        self.merge_status_var.set("Restarting automatic alignment from live camera views...")

    def toggle_merged_fullscreen(self):
        if self.merged_toplevel is not None:
            self._close_merged_fullscreen()
            return

        window = tk.Toplevel(self.root)
        window.configure(bg="black")
        window.attributes("-fullscreen", True)
        window.bind("<Escape>", lambda _event: self._close_merged_fullscreen())
        window.protocol("WM_DELETE_WINDOW", self._close_merged_fullscreen)
        header = tk.Frame(window, bg="#151515", height=44)
        header.pack(fill="x", side="top")
        tk.Label(header, text="MERGED PRODUCT VIEW", bg="#151515", fg="white",
                 font=("Arial", 12, "bold")).pack(side="left", padx=12, pady=7)
        tk.Button(header, text="✕", command=self._close_merged_fullscreen,
                  bg="#c62828", fg="white", activebackground="#ef5350",
                  activeforeground="white", bd=0, font=("Arial", 16, "bold"),
                  width=4, cursor="hand2").pack(side="right", padx=8, pady=5)
        label = tk.Label(window, bg="black", bd=0)
        label.pack(fill="both", expand=True)
        self.merged_toplevel = window
        self.merged_fullscreen_label = label

    def _close_merged_fullscreen(self):
        window = self.merged_toplevel
        self.merged_toplevel = None
        self.merged_fullscreen_label = None
        if window is not None:
            try:
                if window.winfo_exists():
                    window.destroy()
            except tk.TclError:
                pass

    def toggle_roi(self, camera):
        self.roi_visible[camera] = not self.roi_visible[camera]
        self.roi_buttons[camera].configure(
            text=f"ROI: {'ON' if self.roi_visible[camera] else 'OFF'}"
        )

    def toggle_text(self, camera):
        self.text_visible[camera] = not self.text_visible[camera]
        self.text_buttons[camera].configure(
            text=f"FEED TEXT: {'ON' if self.text_visible[camera] else 'OFF'}"
        )

    def toggle_lock(self, camera):
        self.locked[camera] = not self.locked[camera]
        state = "LOCKED" if self.locked[camera] else "UNLOCKED"
        messagebox.showinfo(
            "ROI",
            f"{camera.upper()} ROI is now {state}."
        )

    def adjust_roi(self, camera, key, delta):
        if self.locked[camera]:
            messagebox.showwarning(
                "ROI Locked",
                f"Unlock {camera.upper()} ROI before editing."
            )
            return

        var = self.roi_vars[(camera, key)]
        value = int(var.get()) + delta

        if key in ("top_y", "bottom_y"):
            value = max(0, min(PANEL_HEIGHT - 1, value))
        else:
            value = max(0, min(PANEL_WIDTH - 1, value))

        var.set(value)
        self.settings[camera][key] = value

    def save_roi(self):
        for camera in ("cam1", "cam2"):
            for key in CAMERA_DEFAULT:
                self.settings[camera][key] = int(
                    self.roi_vars[(camera, key)].get()
                )

        try:
            save_settings(self.settings)
            messagebox.showinfo(
                "Settings Saved",
                f"ROI settings saved to:\n{CONFIG_PATH}"
            )
        except OSError as exc:
            messagebox.showerror(
                "Save Error",
                f"Could not save ROI settings:\n{exc}"
            )

    # --------------------------------------------------------
    # DRAWING HELPERS
    # --------------------------------------------------------

    @staticmethod
    def draw_dashed_line(image, start, end, color, thickness=2):
        x1, y1 = start
        x2, y2 = end

        length = int(np.hypot(x2 - x1, y2 - y1))
        if length <= 0:
            return

        for offset in range(0, length, DASH_LENGTH + DASH_GAP):
            end_offset = min(offset + DASH_LENGTH, length)

            sx = int(x1 + (x2 - x1) * offset / length)
            sy = int(y1 + (y2 - y1) * offset / length)
            ex = int(x1 + (x2 - x1) * end_offset / length)
            ey = int(y1 + (y2 - y1) * end_offset / length)

            cv2.line(
                image, (sx, sy), (ex, ey), color, thickness
            )

    def draw_center_lines(self, image):
        if not self.show_center_lines:
            return image

        height, width = image.shape[:2]

        # Vertical green dashed center line.
        self.draw_dashed_line(
            image,
            (width // 2, 0),
            (width // 2, height - 1),
            GREEN,
            2
        )

        # Horizontal green dashed center line.
        self.draw_dashed_line(
            image,
            (0, height // 2),
            (width - 1, height // 2),
            GREEN,
            2
        )

        return image

    def draw_roi(self, image, camera):
        if not self.roi_visible[camera]:
            return image

        height, width = image.shape[:2]
        values = {
            key: int(self.roi_vars[(camera, key)].get())
            for key in CAMERA_DEFAULT
        }

        x1 = max(0, min(width - 1, values["x1"]))
        x2 = max(0, min(width - 1, width - 1 - values["x2"]))
        x3 = max(0, min(width - 1, values["x3"]))
        x4 = max(0, min(width - 1, width - 1 - values["x4"]))
        top_y = max(0, min(height - 1, values["top_y"]))
        bottom_y = max(0, min(height - 1, values["bottom_y"]))

        polygon = np.array([
            [x1, top_y],
            [x2, top_y],
            [x4, bottom_y],
            [x3, bottom_y],
        ], dtype=np.int32)

        color = GREEN if self.locked[camera] else YELLOW

        cv2.polylines(
            image,
            [polygon],
            True,
            color,
            2
        )

        return image

    def _show_frame(self, label, frame, key, max_w=None, max_h=None):
        if frame is None:
            label.configure(image="", text="NO CAMERA FRAME", fg="white")
            return
        height, width = frame.shape[:2]
        label_w = max_w or max(1, label.winfo_width())
        label_h = max_h or max(1, label.winfo_height())
        scale = min(label_w / width, label_h / height)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        preview = cv2.resize(frame, new_size,
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        self.tk_images[key] = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.configure(image=self.tk_images[key], text="")

    def save_training_frame(self):
        """Save the current unannotated stitched frame for later labeling."""
        if self.last_merged is None:
            self.merge_status_var.set("No merged frame available to save yet")
            return
        os.makedirs(DATASET_INBOX, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"merged_{stamp}_{int(time.time() * 1000) % 1000:03d}.jpg"
        path = os.path.join(DATASET_INBOX, filename)
        if not cv2.imwrite(path, self.last_merged, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            self.merge_status_var.set("Could not save training frame")
            return
        self.merge_status_var.set(f"Saved training frame: {filename}")

    @staticmethod
    def _annotate_detections(frame, detections):
        output = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(round(value)) for value in det["box"]]
            x1 = max(0, min(output.shape[1] - 1, x1))
            x2 = max(0, min(output.shape[1] - 1, x2))
            y1 = max(0, min(output.shape[0] - 1, y1))
            y2 = max(0, min(output.shape[0] - 1, y2))
            color = CLASS_COLORS.get(det["class"], (255, 255, 255))
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            label = f"{det['class']} {det['confidence']:.2f} T{det['track_id']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y = max(th + 8, y1)
            cv2.rectangle(output, (x1, y - th - 8), (x1 + tw + 8, y), color, -1)
            cv2.putText(output, label, (x1 + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (15, 15, 15), 2, cv2.LINE_AA)
        return output

    def _update_bom_table(self, detections, counts, model_status,
                          has_prediction, bom_stable):
        if not has_prediction:
            display_counts = {name: "—" for name in COMPONENTS}
            row_result = "WAITING" if "ready" not in model_status.lower() else "CHECKING"
            tag = "waiting"
            summary = model_status
        else:
            display_counts = counts
            if not bom_stable:
                row_result, tag = "CHECKING", "waiting"
                summary = "Checking stable detections..."
            else:
                configured = all(EXPECTED_BOM[name] is not None for name in COMPONENTS)
                passes = configured and all(counts.get(name, 0) == EXPECTED_BOM[name]
                                            for name in COMPONENTS)
                row_result = "PASS" if passes else "ERROR" if configured else "SET BOM QTY"
                tag = "pass" if passes else "error" if configured else "waiting"
                summary = ("PASS" if passes else "ERROR — BOM quantities do not match"
                           if configured else "Set BOM quantity for component F")

        rows = tuple((str(index), name,
                      str(EXPECTED_BOM[name]) if EXPECTED_BOM[name] is not None else "Set qty",
                      str(display_counts[name]), row_result)
                     for index, name in enumerate(COMPONENTS, 1))
        if rows != self._last_bom_rows:
            for item_id, values in zip(self.bom_row_ids.values(), rows):
                self.bom_table.item(item_id, values=values, tags=(tag,))
            if self.bom_fullscreen_table is not None:
                try:
                    if self.bom_fullscreen_table.winfo_exists():
                        for item_id, values in zip(
                                self.bom_fullscreen_table.get_children(), rows):
                            self.bom_fullscreen_table.item(
                                item_id, values=values, tags=(tag,)
                            )
                    else:
                        self.bom_fullscreen_table = None
                        self.bom_toplevel = None
                except tk.TclError:
                    # The user may close the fullscreen table between frames.
                    self.bom_fullscreen_table = None
                    self.bom_toplevel = None
            self._last_bom_rows = rows
        if summary != self._last_bom_summary:
            self.bom_summary_var.set(summary)
            self._last_bom_summary = summary

    # --------------------------------------------------------
    # FULLSCREEN
    # --------------------------------------------------------

    def toggle_bom_fullscreen(self):
        if self.bom_toplevel is not None and self.bom_toplevel.winfo_exists():
            self.bom_toplevel.destroy()
            self.bom_toplevel = None
            self.bom_fullscreen_table = None
            return

        window = tk.Toplevel(self.root)
        self.bom_toplevel = window
        window.title("BOM Result Table")
        window.configure(bg="#202020")
        window.attributes("-fullscreen", True)
        window.bind("<Escape>", lambda _event: self._close_bom_fullscreen())
        header = tk.Frame(window, bg="#202020")
        header.pack(fill="x", padx=24, pady=18)
        tk.Label(header, text="BOM_RESULT TABLE", bg="#202020", fg="white",
                 font=("Arial", 22, "bold")).pack(side="left")
        ttk.Button(header, text="EXIT FULL SCREEN",
                   command=self._close_bom_fullscreen).pack(side="right")
        style = ttk.Style(window)
        style.configure("BOMFull.Treeview", rowheight=56, font=("Arial", 18))
        style.configure("BOMFull.Treeview.Heading", font=("Arial", 17, "bold"))
        table = ttk.Treeview(window, columns=("sr", "component", "bom", "detected", "result"),
                             show="headings", style="BOMFull.Treeview")
        self.bom_fullscreen_table = table
        for column, title in (("sr", "Sr.No"), ("component", "Component"),
                              ("bom", "BOM_Quantity"),
                              ("detected", "Detected_quantity"),
                              ("result", "PASS/ERROR result")):
            table.heading(column, text=title)
            table.column(column, anchor="center", width=220, stretch=True)
        table.tag_configure("pass", foreground="#147a25")
        table.tag_configure("error", foreground="#b3261e")
        table.tag_configure("waiting", foreground="#8a6700")
        for name in COMPONENTS:
            source = self.bom_table.item(self.bom_row_ids[name])
            table.insert("", "end", values=source["values"], tags=source["tags"])
        table.pack(fill="both", expand=True, padx=24, pady=(0, 24))
        window.protocol("WM_DELETE_WINDOW", lambda: self._close_bom_fullscreen())

    def _close_bom_fullscreen(self):
        if self.bom_toplevel is not None and self.bom_toplevel.winfo_exists():
            self.bom_toplevel.destroy()
        self.bom_toplevel = None
        self.bom_fullscreen_table = None

    def toggle_fullscreen(self):
        if not self.fullscreen:
            self.fullscreen = True
            self.root.attributes("-fullscreen", True)
            self.toolbar.pack_forget()
            self.controls.pack_forget()
        else:
            self._leave_fullscreen()

    def _leave_fullscreen(self):
        if not self.fullscreen:
            return

        self.root.attributes("-fullscreen", False)
        self.fullscreen = False

        self.toolbar.pack(fill="x", padx=8, pady=8)
        self.controls.pack(
            side="left", fill="y", padx=(0, 8)
        )

    def _escape_fullscreen(self, _event=None):
        self._leave_fullscreen()

    # --------------------------------------------------------
    # LIVE VIEW
    # --------------------------------------------------------

    def update_view(self):
        if self.closing:
            return

        try:
            self._update_view_once()
        except Exception as exc:
            # A transient Tk/OpenCV error must not silently stop the scheduled
            # refresh loop during a long inspection shift.
            now = time.monotonic()
            if now - self._last_gui_error_time >= 5.0:
                print(f"[viewer recovered from update error] {exc}", flush=True)
                self._last_gui_error_time = now
            try:
                self.merge_status_var.set(f"Recovering from viewer error: {exc}")
            except tk.TclError:
                pass
        finally:
            if not self.closing:
                try:
                    self.root.after(GUI_REFRESH_MS, self.update_view)
                except tk.TclError:
                    self.closing = True

    def _update_view_once(self):

        frame1, connected1, fps1, w1, h1 = self.cam1.get_data()
        frame2, connected2, fps2, w2, h2 = self.cam2.get_data()
        feeds_ready = frame1 is not None and frame2 is not None
        if feeds_ready != self._feeds_ready:
            self.detector.reset_for_camera_loss()
            self._feeds_ready = feeds_ready
        if not feeds_ready:
            self.last_merged = None

        for camera, frame, connected, fps, width, height in (
            ("cam1", frame1, connected1, fps1, w1, h1),
            ("cam2", frame2, connected2, fps2, w2, h2),
        ):
            preview = None if frame is None else cv2.resize(
                frame, (PANEL_WIDTH, CAMERA_PREVIEW_HEIGHT),
                interpolation=cv2.INTER_AREA
            )
            if preview is not None:
                preview = self.draw_roi(preview, camera)
                preview = self.draw_center_lines(preview)
            self._show_frame(self.camera_labels[camera], preview, camera)

        now = time.monotonic()
        render_stitch = now - self.last_stitch_time >= STITCH_REFRESH_SECONDS
        if render_stitch:
            self.last_stitch_time = now
            self.last_merged = self.stitcher.stitch(frame1, frame2)
            if (self.last_merged is not None and
                    now - self.last_detector_submit_time >= DETECTION_REFRESH_SECONDS):
                self.detector.submit(self.last_merged)
                self.last_detector_submit_time = now
        (detections, counts, model_status,
         has_prediction, bom_stable) = self.detector.snapshot()
        self._update_bom_table(detections, counts, model_status,
                               has_prediction, bom_stable)
        if self.last_merged is not None and render_stitch:
            annotated = self._annotate_detections(self.last_merged, detections)
            self._show_frame(self.merged_label, annotated, "merged")
            if self.merged_fullscreen_label is not None:
                self._show_frame(self.merged_fullscreen_label, annotated,
                                 "merged_fullscreen")
        elif self.last_merged is None and render_stitch:
            self.merged_label.configure(image="", text=self.stitcher.status, fg="white")
            if self.merged_fullscreen_label is not None:
                self.merged_fullscreen_label.configure(image="", text=self.stitcher.status,
                                                       fg="white")
        self.merge_status_var.set(f"{self.stitcher.status} | {model_status}")

        self.view_header_var.set(
            f"CAM 1: {'ONLINE' if connected1 else 'RECONNECTING'} "
            f"{w1}x{h1} | {fps1:.1f} FPS     "
            f"CAM 2: {'ONLINE' if connected2 else 'RECONNECTING'} "
            f"{w2}x{h2} | {fps2:.1f} FPS"
        )

    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    def close(self):
        if self.closing:
            return

        self.closing = True
        self.cam1.stop()
        self.cam2.stop()
        self.detector.close()
        if self.merged_toplevel is not None:
            self.merged_toplevel.destroy()
        self.root.destroy()


# ============================================================
# START APPLICATION
# ============================================================

def main():
    root = tk.Tk()
    app = CombinedCameraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
