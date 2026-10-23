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
# Prefer reliable interleaved RTSP over TCP. Override by setting
# BOM_OPENCV_FFMPEG_CAPTURE_OPTIONS before launching if a different transport
# is required for a specific camera/network.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = os.environ.get(
    "BOM_OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp"
)
import cv2
cv2.setNumThreads(2)
import json
import math
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque

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
GUI_REFRESH_MS = 80
STITCH_REFRESH_SECONDS = 0.35
DETECTION_REFRESH_SECONDS = 0.80
CAMERA_PREVIEW_REFRESH_SECONDS = 0.13
MERGED_DISPLAY_MAX_EDGE = 1600
CAMERA_FRAME_STALE_SECONDS = 15.0

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
MODEL_ROTATION_TTA = (0, 90, 180, 270)
CLASS_CONFIDENCE_THRESHOLDS = {
    # Keep post-filtering permissive so a lighting change cannot discard a
    # prediction that the appearance model already recognizes.
    "A": 0.50, "B": 0.50, "C": 0.50, "D": 0.50, "E": 0.65, "F": 0.50,
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


def remove_camera_osd(frame):
    """Inpaint small bright camera logo/time glyphs in the top image band."""
    if frame is None or frame.size == 0:
        return frame
    height, width = frame.shape[:2]
    band_h = max(1, int(height * 0.075))
    band = frame[:band_h]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    # Camera OSD is bright and nearly neutral in color. Exclude large rail
    # regions so only compact text/logo shapes are removed.
    candidates = ((gray >= 165) & (hsv[:, :, 1] <= 150)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates, connectivity=8
    )
    mask = np.zeros((band_h, width), dtype=np.uint8)
    for label in range(1, count):
        x, y, box_w, box_h, area = stats[label]
        if (3 <= area <= 3500 and 4 <= box_h <= max(12, int(height * 0.055))
                and box_w <= int(width * 0.38)):
            mask[labels == label] = 255
    if not np.any(mask):
        return frame
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    cleaned = frame.copy()
    cleaned[:band_h] = cv2.inpaint(band, mask, 4, cv2.INPAINT_TELEA)
    return cleaned


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
        self.last_frame_at = 0.0
        self.last_error = ""
        self.reconnect_count = 0
        self.frame_sequence = 0
        self.ever_connected = False
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

        def mark_disconnected(reason):
            with self.lock:
                self.connected = False
                self.frame = None
                self.last_error = reason

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
                        mark_disconnected("RTSP open failed")
                        time.sleep(reconnect_delay)
                        reconnect_delay = min(10.0, reconnect_delay * 1.7)
                        continue

                    # These settings are best-effort and backend-dependent.
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                ok, frame = capture.read()

                if not ok or frame is None:
                    mark_disconnected("RTSP read failed or timed out")
                    capture.release()
                    capture = None
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(10.0, reconnect_delay * 1.7)
                    continue

                height, width = frame.shape[:2]
                if width < 16 or height < 16:
                    mark_disconnected("Camera returned an invalid frame")
                    capture.release()
                    capture = None
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(10.0, reconnect_delay * 1.7)
                    continue
                fps_count += 1
                now = time.monotonic()

                if now - fps_start >= 1.0:
                    fps_value = fps_count / (now - fps_start)
                    fps_count = 0
                    fps_start = now
                else:
                    fps_value = None

                with self.lock:
                    if self.ever_connected and not self.connected:
                        self.reconnect_count += 1
                    self.frame = frame
                    self.frame_sequence += 1
                    self.connected = True
                    self.last_frame_at = now
                    self.last_error = ""
                    self.ever_connected = True
                    self.width = width
                    self.height = height
                    if fps_value is not None:
                        self.fps = fps_value
                reconnect_delay = 0.5

            except Exception as exc:
                mark_disconnected(f"{type(exc).__name__}: {exc}")
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
            connected = self.connected
            if (frame is not None and
                    time.monotonic() - self.last_frame_at > CAMERA_FRAME_STALE_SECONDS):
                # Keep the GUI honest if the backend stops yielding frames even
                # when its read call has not returned yet.
                frame = None
                connected = False
            return (
                frame,
                connected,
                self.fps,
                self.width,
                self.height,
                self.reconnect_count,
                self.last_error,
                self.frame_sequence,
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


def suppress_detection_duplicates(detections):
    """Remove only boxes that overlap the same object; never use location priors."""
    kept = []
    for det in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        duplicate = False
        for other in kept:
            if detection_iou(det["box"], other["box"]) >= 0.45:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def map_rotated_box_to_original(box, angle, original_width, original_height):
    """Map an axis-aligned box from a rotated image back to the source frame."""
    x1, y1, x2, y2 = map(float, box)
    if angle == 90:
        return np.asarray((y1, original_height - x2,
                           y2, original_height - x1), dtype=np.float32)
    if angle == 180:
        return np.asarray((original_width - x2, original_height - y2,
                           original_width - x1, original_height - y1),
                          dtype=np.float32)
    if angle == 270:
        return np.asarray((original_width - y2, x1,
                           original_width - y1, x2), dtype=np.float32)
    return np.asarray((x1, y1, x2, y2), dtype=np.float32)


def rotate_mask_to_original(mask, angle):
    if angle == 90:
        return cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(mask, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    return mask


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
        self.segmentation_available = False
        self.has_prediction = False
        self.bom_stable = False
        self.history = deque(maxlen=6)
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
            # Results are replaced atomically after inference and never mutated
            # afterward, so shallow snapshots avoid copying every segmentation
            # mask on every GUI refresh.
            detections = [dict(item) for item in self.detections]
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
            self.status = "Waiting for both camera feeds"

    def reset_for_view_change(self):
        """Discard predictions made against the previous merged orientation."""
        with self.lock:
            self.generation += 1
            self.pending_frame = None
            self.detections = []
            self.counts = {name: 0 for name in COMPONENTS}
            self.has_prediction = False
            self.bom_stable = False
            self.history.clear()
            self.status = "Updating detection for rotated merged view"

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
            segmentation_available = getattr(model, "task", None) == "segment"
            try:
                import torch
                device = 0 if torch.cuda.is_available() else "cpu"
                if device == "cpu":
                    # Leave CPU capacity for the camera readers and responsive UI.
                    torch.set_num_threads(2)
                    try:
                        torch.set_num_interop_threads(1)
                    except RuntimeError:
                        pass
            except Exception:
                device = "cpu"
            with self.lock:
                self.segmentation_available = segmentation_available
                mode = "segmentation masks" if segmentation_available else "detection boxes only"
                self.status = f"Model ready ({os.path.basename(path)}; {mode})"
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
                raw = []
                frame_h, frame_w = frame.shape[:2]
                masks_present = False
                for angle in MODEL_ROTATION_TTA:
                    if angle == 0:
                        inference_frame = frame
                    elif angle == 90:
                        inference_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif angle == 180:
                        inference_frame = cv2.rotate(frame, cv2.ROTATE_180)
                    else:
                        inference_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    results = model.predict(
                        source=inference_frame, conf=MODEL_CONFIDENCE, iou=0.50,
                        imgsz=MODEL_IMAGE_SIZE, max_det=MODEL_MAX_DETECTIONS,
                        device=device, verbose=False,
                    )
                    result = results[0]
                    names = result.names
                    masks = getattr(result, "masks", None)
                    mask_data = masks.data if masks is not None else None
                    masks_present = masks_present or masks is not None
                    if result.boxes is None:
                        continue
                    for box_index, box in enumerate(result.boxes):
                        class_id = int(box.cls.item())
                        name = (names.get(class_id, "") if isinstance(names, dict)
                                else names[class_id] if class_id < len(names) else "")
                        name = str(name).upper()
                        confidence = float(box.conf.item())
                        if (name not in COMPONENTS or confidence <
                                CLASS_CONFIDENCE_THRESHOLDS.get(name, MODEL_CONFIDENCE)):
                            continue
                        rotated_coords = box.xyxy[0].detach().cpu().numpy().astype(np.float32)
                        coords = map_rotated_box_to_original(
                            rotated_coords, angle, frame_w, frame_h
                        )
                        coords[[0, 2]] = np.clip(coords[[0, 2]], 0, frame_w - 1)
                        coords[[1, 3]] = np.clip(coords[[1, 3]], 0, frame_h - 1)
                        x1, y1, x2, y2 = coords.tolist()
                        instance_mask = None
                        instance_mask_box = None
                        if mask_data is not None and box_index < len(mask_data):
                            instance_mask = mask_data[box_index].detach().cpu().numpy()
                            instance_mask = cv2.resize(
                                instance_mask.astype(np.float32),
                                (inference_frame.shape[1], inference_frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            instance_mask = (instance_mask >= 0.5).astype(np.uint8)
                            instance_mask = rotate_mask_to_original(instance_mask, angle)
                            left = max(0, min(frame_w - 1, int(np.floor(x1))))
                            top = max(0, min(frame_h - 1, int(np.floor(y1))))
                            right = max(left + 1, min(frame_w, int(np.ceil(x2))))
                            bottom = max(top + 1, min(frame_h, int(np.ceil(y2))))
                            instance_mask = instance_mask[top:bottom, left:right].copy()
                            instance_mask_box = (left, top, right, bottom)
                        raw.append({
                            "class": name, "confidence": confidence, "box": coords,
                            "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
                            "mask": instance_mask,
                            "mask_box": instance_mask_box,
                        })
                raw = suppress_detection_duplicates(raw)
                with self.lock:
                    if generation != self.generation:
                        continue
                    # Detection and class assignment are made independently on
                    # each current merged frame from visual evidence. No fixed
                    # product region, expected location, or temporal coordinate
                    # association is allowed to add, remove, or relabel boxes.
                    counts = {name: sum(item["class"] == name for item in raw)
                              for name in COMPONENTS}
                    signature = tuple(counts[name] for name in COMPONENTS)
                    self.history.append(signature)
                    bom_stable = (len(self.history) == self.history.maxlen and
                                  len(set(self.history)) == 1)
                    self.detections = raw
                    self.counts = counts
                    self.has_prediction = True
                    self.bom_stable = bom_stable
                    mask_status = ("segmentation masks" if masks_present
                                   else "boxes only; segmentation weights required")
                    stability_status = "BOM stable" if bom_stable else "stabilizing"
                    self.status = f"Live detection | {mask_status} | {stability_status}"
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
        self.full_view_angle = None
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
        self.full_view_angle = None
        self.status = "Automatically aligning the two live views..."
        self.last_attempt = 0.0

    @staticmethod
    def _corners(width, height):
        return np.float32([[0, 0], [width - 1, 0],
                           [width - 1, height - 1], [0, height - 1]]).reshape(-1, 1, 2)

    @staticmethod
    def _product_feature_mask(frame):
        """Prefer shared features on the product plane over rails/background."""
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        panel = (((hue <= 20) | (hue >= 168)) &
                 (saturation >= 22) & (value >= 30)).astype(np.uint8) * 255
        kernel_size = max(9, int(min(height, width) * 0.025))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        panel = cv2.morphologyEx(panel, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(panel, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < height * width * 0.08:
            return None
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        # Keep matches away from the panel border, where parallax and frame
        # metal can pull the homography away from the flat inspection surface.
        erode_size = max(3, int(min(height, width) * 0.008))
        if erode_size % 2 == 0:
            erode_size += 1
        erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_size, erode_size)
        )
        return cv2.erode(mask, erode, iterations=2)

    def _estimate(self, frame1, frame2, roi_mask1=None, roi_mask2=None):
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        scale = min(1.0, 1500.0 / max(w1, h1, w2, h2))
        small1 = cv2.resize(frame1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small2 = cv2.resize(frame2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        g1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
        # Match on the shared product plane so the homography aligns the
        # product itself, rather than camera rails or distant background.
        product_mask1 = self._product_feature_mask(frame1)
        product_mask2 = self._product_feature_mask(frame2)
        feature_mask1 = np.zeros_like(g1)
        feature_mask2 = np.zeros_like(g2)
        margin_x = int(g1.shape[1] * 0.025)
        margin_y = int(g1.shape[0] * 0.055)
        product_masks_used = product_mask1 is not None and product_mask2 is not None
        if product_masks_used:
            feature_mask1 = cv2.resize(
                product_mask1, (g1.shape[1], g1.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            feature_mask2 = cv2.resize(
                product_mask2, (g2.shape[1], g2.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            feature_mask1[:margin_y] = 0
            feature_mask2[:margin_y] = 0
        else:
            # Fall back to broad interior features when the product color
            # cannot be separated reliably in either feed.
            feature_mask1[margin_y:g1.shape[0] - margin_y,
                          margin_x:g1.shape[1] - margin_x] = 255
            feature_mask2[margin_y:g2.shape[0] - margin_y,
                          margin_x:g2.shape[1] - margin_x] = 255
        roi_masks_used = roi_mask1 is not None and roi_mask2 is not None
        if roi_masks_used:
            scaled_roi1 = cv2.resize(roi_mask1, (g1.shape[1], g1.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            scaled_roi2 = cv2.resize(roi_mask2, (g2.shape[1], g2.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            # ROI controls define where to find reliable cross-camera matches;
            # the final panorama still uses the complete camera frames.
            product_feature1 = cv2.bitwise_and(feature_mask1, scaled_roi1)
            product_feature2 = cv2.bitwise_and(feature_mask2, scaled_roi2)
            if cv2.countNonZero(product_feature1) >= 500 and cv2.countNonZero(product_feature2) >= 500:
                feature_mask1, feature_mask2 = product_feature1, product_feature2
            else:
                feature_mask1, feature_mask2 = scaled_roi1, scaled_roi2
        orb = cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8,
                             edgeThreshold=19, fastThreshold=7)
        def match_features(mask1, mask2):
            keypoints1, descriptors1 = orb.detectAndCompute(g1, mask1)
            keypoints2, descriptors2 = orb.detectAndCompute(g2, mask2)
            if descriptors1 is None or descriptors2 is None:
                return keypoints1, keypoints2, []
            pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
                descriptors2, descriptors1, k=2
            )
            matches = [pair[0] for pair in pairs
                       if len(pair) == 2 and
                       pair[0].distance < 0.72 * pair[1].distance]
            return keypoints1, keypoints2, matches

        kp1, kp2, good = match_features(feature_mask1, feature_mask2)
        if len(good) < 20 and product_masks_used and not roi_masks_used:
            # If the colored product surface is too plain or too different in
            # lighting between feeds, retry with all interior image features.
            broad1 = np.zeros_like(g1)
            broad2 = np.zeros_like(g2)
            broad1[margin_y:g1.shape[0] - margin_y,
                   margin_x:g1.shape[1] - margin_x] = 255
            broad2[margin_y:g2.shape[0] - margin_y,
                   margin_x:g2.shape[1] - margin_x] = 255
            kp1, kp2, good = match_features(broad1, broad2)
        if len(good) < 20:
            area = "selected ROIs" if roi_masks_used else "shared view"
            raise RuntimeError(f"Only {len(good)} matching features in {area}; adjust the ROI values")

        inv_scale = 1.0 / scale
        points2 = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) * inv_scale
        points1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2) * inv_scale
        # Prefer an affine camera-to-camera transform. The views are fixed and
        # nearly top-down, so projective extrapolation from a narrow ROI can
        # bend the far edges of the product. Use a homography only when the
        # affine fit cannot explain enough reliable ROI matches.
        affine, affine_inliers = cv2.estimateAffine2D(
            points2.reshape(-1, 2), points1.reshape(-1, 2),
            method=cv2.RANSAC, ransacReprojThreshold=8.0,
            maxIters=5000, confidence=0.995, refineIters=10,
        )
        affine_count = (int(affine_inliers.sum())
                        if affine_inliers is not None else 0)
        affine_ratio = affine_count / max(1, len(good))
        transform_kind = "affine"
        if affine is not None and affine_count >= 18 and affine_ratio >= 0.22:
            H = np.vstack((affine, [0.0, 0.0, 1.0])).astype(np.float64)
            inliers = affine_inliers
        else:
            H, inliers = cv2.findHomography(points2, points1, cv2.RANSAC, 8.0)
            transform_kind = "perspective"
        inlier_count = int(inliers.sum()) if inliers is not None else 0
        inlier_ratio = inlier_count / max(1, len(good))
        if H is None or inlier_count < 18 or inlier_ratio < 0.22:
            raise RuntimeError(f"Automatic alignment confidence is low ({inlier_count} matches)")
        if transform_kind == "perspective":
            perspective_strength = max(abs(H[2, 0]) * w2,
                                       abs(H[2, 1]) * h2)
            if perspective_strength > 0.35:
                if affine is not None and affine_count >= 12 and affine_ratio >= 0.18:
                    H = np.vstack((affine, [0.0, 0.0, 1.0])).astype(np.float64)
                    inlier_count, inlier_ratio = affine_count, affine_ratio
                    transform_kind = "affine"
                else:
                    raise RuntimeError("Perspective fit is unstable; widen the shared ROI")

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
        self.status = (f"Auto-aligned ({transform_kind}) | {inlier_count} matches | "
                       "overlap shown once")

    def _level_full_view(self, image):
        """Deskew the entire panorama on an expanded canvas, without cropping."""
        height, width = image.shape[:2]
        if self.full_view_angle is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 45, 135)
            min_length = max(100, int(width * 0.28))
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 720, threshold=max(50, min_length // 3),
                minLineLength=min_length, maxLineGap=max(24, width // 70),
            )
            angles = []
            if lines is not None:
                for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
                    angle = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
                    if abs(angle) <= 15.0:
                        angles.append(angle)
            self.full_view_angle = float(np.median(angles)) if len(angles) >= 2 else 0.0

        angle = self.full_view_angle
        if abs(angle) < 0.15:
            return image
        radians = math.radians(angle)
        canvas_w = int(math.ceil(width * abs(math.cos(radians)) +
                                 height * abs(math.sin(radians))))
        canvas_h = int(math.ceil(height * abs(math.cos(radians)) +
                                 width * abs(math.sin(radians))))
        rotation = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        rotation[0, 2] += (canvas_w - width) / 2.0
        rotation[1, 2] += (canvas_h - height) / 2.0
        return cv2.warpAffine(
            image, rotation, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )

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
                # A worn or shadowed edge often prevents polygon approximation
                # from returning exactly four corners. Fit a rotated rectangle
                # to the largest credible product-color region as a fallback.
                largest = max(contours, key=cv2.contourArea, default=None)
                if (largest is None or
                        cv2.contourArea(largest) < image.shape[0] * image.shape[1] * 0.12):
                    return self._level_full_view(image), False
                fitted = cv2.minAreaRect(largest)
                candidates.append((cv2.contourArea(largest),
                                   cv2.boxPoints(fitted).astype(np.float32)))

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
                return self._level_full_view(image), None
            corners = np.median(np.stack(self.rectify_samples), axis=0).astype(np.float32)
        else:
            corners = self.rectified_corners.copy()
        raw_corners = corners.copy()
        # The red panel mask can stop a little inside the outer metal rim.
        # Expand the detected quadrilateral so the complete top and bottom
        # edges remain in the corrected image.
        center = corners.mean(axis=0)
        # Expand beyond the detected red face to retain the frame around it.
        corners[:, 0] = center[0] + 1.10 * (corners[:, 0] - center[0])
        corners[:, 1] = center[1] + 1.14 * (corners[:, 1] - center[1])
        corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)
        # Use the unexpanded face boundary when the extra frame margin runs
        # outside the panorama. This still rectifies the product instead of
        # falling back to a tilted full scene with large black corners.
        edge_margin = max(3, int(min(width, height) * 0.004))
        touches_canvas_edge = (
            np.any(corners[:, 0] <= edge_margin) or
            np.any(corners[:, 1] <= edge_margin) or
            np.any(corners[:, 0] >= width - 1 - edge_margin) or
            np.any(corners[:, 1] >= height - 1 - edge_margin)
        )
        if touches_canvas_edge:
            raw_touches_edge = (
                np.any(raw_corners[:, 0] <= edge_margin) or
                np.any(raw_corners[:, 1] <= edge_margin) or
                np.any(raw_corners[:, 0] >= width - 1 - edge_margin) or
                np.any(raw_corners[:, 1] >= height - 1 - edge_margin)
            )
            if raw_touches_edge:
                return self._level_full_view(image), False
            corners = raw_corners
        tl, tr, br, bl = corners
        out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        if out_w < 100 or out_h < 100 or out_w > width * 1.2 or out_h > height * 1.2:
            return self._level_full_view(image), False
        ratio = out_w / float(out_h)
        if not 0.35 <= ratio <= 3.5:
            return self._level_full_view(image), False
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

    def stitch(self, frame1, frame2, roi_masks=None):
        if frame1 is None or frame2 is None:
            self.status = "Waiting for both camera frames"
            return None
        if self.H is None:
            now = time.monotonic()
            if now - self.last_attempt < 2.0:
                return None
            self.last_attempt = now
            try:
                masks = roi_masks or (None, None)
                self._estimate(frame1, frame2, masks[0], masks[1])
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
            self.status = "Automatic merge | full view leveled to preserve product edges"
        return product


# ============================================================
# BACKGROUND STITCH WORKER
# ============================================================

def roi_polygon_from_values(values, width, height):
    """Map logical 640x480 ROI settings to an image's dimensions."""
    sx, sy = width / float(PANEL_WIDTH), height / float(PANEL_HEIGHT)
    x1 = values["x1"] * sx
    x2 = (PANEL_WIDTH - 1 - values["x2"]) * sx
    x3 = values["x3"] * sx
    x4 = (PANEL_WIDTH - 1 - values["x4"]) * sx
    y1, y2 = values["top_y"] * sy, values["bottom_y"] * sy
    return np.round([[x1, y1], [x2, y1], [x4, y2], [x3, y2]]).astype(np.int32)


def roi_mask_from_values(frame, values):
    """Build a full-resolution alignment mask from logical ROI settings."""
    height, width = frame.shape[:2]
    polygon = roi_polygon_from_values(values, width, height)
    if abs(cv2.contourArea(polygon.reshape(-1, 1, 2))) < height * width * 0.002:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


class StitchWorker:
    """Keep inpainting, alignment, warping and leveling off the Tk event loop."""

    def __init__(self):
        self.stitcher = LiveStitcher()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop_event = threading.Event()
        self.pending = None
        self.reset_requested = False
        self.revision = 0
        self.sequence = 0
        self.image = None
        self.status = self.stitcher.status
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name="camera-stitch-worker")
        self.thread.start()

    def submit(self, frame1, frame2, settings):
        if self.stop_event.is_set():
            return
        settings = {camera: dict(values) for camera, values in settings.items()}
        with self.lock:
            self.pending = (self.revision, frame1, frame2, settings)
        self.wake.set()

    def request_reset(self):
        with self.lock:
            self.revision += 1
            self.pending = None
            self.reset_requested = True
            self.image = None
            self.status = "Restarting automatic alignment..."
            self.sequence += 1
        self.wake.set()

    def snapshot(self):
        with self.lock:
            return self.image, self.status, self.sequence

    def _run(self):
        while not self.stop_event.is_set():
            self.wake.wait(0.5)
            self.wake.clear()
            if self.stop_event.is_set():
                break
            with self.lock:
                reset = self.reset_requested
                self.reset_requested = False
                item = self.pending
                self.pending = None
                revision = self.revision
            if reset:
                self.stitcher.reset()
            if item is None:
                continue
            item_revision, frame1, frame2, settings = item
            if item_revision != revision:
                continue
            if frame1 is None or frame2 is None:
                image, status = None, "Waiting for both camera frames"
            else:
                try:
                    clean1 = remove_camera_osd(frame1)
                    clean2 = remove_camera_osd(frame2)
                    masks = (roi_mask_from_values(clean1, settings["cam1"]),
                             roi_mask_from_values(clean2, settings["cam2"]))
                    image = self.stitcher.stitch(clean1, clean2, roi_masks=masks)
                    status = self.stitcher.status
                except Exception as exc:
                    image = None
                    status = f"Stitch error: {exc}"
            with self.lock:
                if item_revision != self.revision:
                    continue
                self.image = image
                self.status = status
                self.sequence += 1

    def close(self):
        self.stop_event.set()
        self.wake.set()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=2.0)


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
        self.merged_rotation = 0

        self.settings = load_settings()

        self.roi_visible = {"cam1": True, "cam2": True}
        self.text_visible = {"cam1": True, "cam2": True}
        self.locked = {"cam1": False, "cam2": False}

        self.roi_vars = {}
        self.roi_edit_jobs = {}
        self.roi_buttons = {}
        self.text_buttons = {}
        self.camera_labels = {}
        self.tk_images = {}
        self.merged_toplevel = None
        self.merged_fullscreen_label = None
        self.merged_fullscreen_rotate_button = None
        self.bom_toplevel = None
        self.bom_fullscreen_table = None
        self.stitch_worker = StitchWorker()
        self.detector = DetectionWorker(MODEL_PATH)
        self.last_stitch_time = 0.0
        self.last_stitch_submit_time = 0.0
        self.last_stitch_sequence = -1
        self.last_detector_submit_time = 0.0
        self.last_merged = None
        self.last_merged_source = None
        self._feeds_ready = False
        self._last_gui_error_time = 0.0
        self._last_bom_rows = None
        self._last_bom_summary = None
        self._last_merge_status_text = None
        self._last_view_header_text = None
        self.camera_preview_sequence = {"cam1": -1, "cam2": -1}
        self.camera_preview_time = {"cam1": 0.0, "cam2": 0.0}

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
        self.rotate_merged_button = ttk.Button(
            merge_toolbar, text="ROTATE 90° (0°)", command=self.rotate_merged_view
        )
        self.rotate_merged_button.pack(side="left", padx=(6, 0))
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

                label = {
                    "x1": "X1", "x2": "X2", "x3": "X3", "x4": "X4",
                    "top_y": "Y1", "bottom_y": "Y2",
                }[key]
                tk.Label(
                    row,
                    text=label,
                    width=12,
                    anchor="w",
                    bg="#303030",
                    fg="white"
                ).pack(side="left")

                var = tk.StringVar(value=str(self.settings[camera][key]))
                self.roi_vars[(camera, key)] = var

                ttk.Button(
                    row,
                    text="-",
                    width=3,
                    command=lambda c=camera, k=key: self.adjust_roi(c, k, -1)
                ).pack(side="left", padx=2)

                value_entry = ttk.Entry(row, textvariable=var, width=5,
                                        justify="center")
                value_entry.pack(side="left", padx=3)
                value_entry.bind(
                    "<KeyRelease>",
                    lambda _event, c=camera, k=key: self._schedule_roi_edit(c, k)
                )
                value_entry.bind(
                    "<FocusOut>",
                    lambda _event, c=camera, k=key: self._apply_roi_edit(c, k)
                )

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
        self.camera_preview_sequence = {"cam1": -1, "cam2": -1}
        self.center_button.configure(
            text="CENTER LINES: ON" if self.show_center_lines
            else "CENTER LINES: OFF"
        )

    def restart_auto_alignment(self):
        """Discard the current live estimate and automatically find it again."""
        self.stitch_worker.request_reset()
        self.last_merged = None
        self.last_merged_source = None
        self.last_stitch_time = 0.0
        self.last_stitch_submit_time = 0.0
        self.merge_status_var.set("Restarting automatic alignment from live camera views...")

    def _rotate_merged_image(self, image):
        if image is None:
            return None
        if self.merged_rotation == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if self.merged_rotation == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if self.merged_rotation == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def rotate_merged_view(self):
        """Rotate the displayed/inferred merged image clockwise by 90 degrees."""
        self.merged_rotation = (self.merged_rotation + 90) % 360
        self.detector.reset_for_view_change()
        self.last_detector_submit_time = 0.0
        self.rotate_merged_button.configure(
            text=f"ROTATE 90° ({self.merged_rotation}°)"
        )
        if self.merged_fullscreen_rotate_button is not None:
            try:
                if self.merged_fullscreen_rotate_button.winfo_exists():
                    self.merged_fullscreen_rotate_button.configure(
                        text=f"ROTATE 90° ({self.merged_rotation}°)"
                    )
            except tk.TclError:
                self.merged_fullscreen_rotate_button = None

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
        self.merged_fullscreen_rotate_button = tk.Button(
            header, text=f"ROTATE 90° ({self.merged_rotation}°)",
            command=self.rotate_merged_view, bg="#333333", fg="white",
            activebackground="#555555", activeforeground="white", bd=0,
            font=("Arial", 10, "bold"), padx=12, pady=7, cursor="hand2",
        )
        self.merged_fullscreen_rotate_button.pack(side="right", padx=8, pady=5)
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
        self.merged_fullscreen_rotate_button = None
        if window is not None:
            try:
                if window.winfo_exists():
                    window.destroy()
            except tk.TclError:
                pass

    def toggle_roi(self, camera):
        self.roi_visible[camera] = not self.roi_visible[camera]
        self.camera_preview_sequence[camera] = -1
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
        self.camera_preview_sequence[camera] = -1
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

        value = self._read_roi_value(camera, key) + delta
        value = self._clamp_roi_value(key, value)
        self.roi_vars[(camera, key)].set(str(value))
        self._apply_roi_edit(camera, key)

    @staticmethod
    def _clamp_roi_value(key, value):
        maximum = PANEL_HEIGHT - 1 if key in ("top_y", "bottom_y") else PANEL_WIDTH - 1
        return max(0, min(maximum, int(value)))

    def _read_roi_value(self, camera, key):
        try:
            raw = self.roi_vars[(camera, key)].get().strip()
            if not raw or not raw.lstrip("+-").isdigit():
                raise ValueError
            value = int(raw)
        except (ValueError, tk.TclError):
            value = int(self.settings[camera].get(key, CAMERA_DEFAULT[key]))
        return self._clamp_roi_value(key, value)

    def _schedule_roi_edit(self, camera, key):
        if self.locked[camera]:
            return
        previous = self.roi_edit_jobs.pop((camera, key), None)
        if previous is not None:
            try:
                self.root.after_cancel(previous)
            except tk.TclError:
                pass
        raw = self.roi_vars[(camera, key)].get().strip()
        if raw and not raw.lstrip("+-").isdigit():
            return
        self.roi_edit_jobs[(camera, key)] = self.root.after(
            450, lambda c=camera, k=key: self._apply_roi_edit(c, k)
        )

    def _apply_roi_edit(self, camera, key):
        job = self.roi_edit_jobs.pop((camera, key), None)
        if job is not None:
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass
        if self.locked[camera]:
            self.roi_vars[(camera, key)].set(str(self.settings[camera][key]))
            return
        previous_value = int(self.settings[camera].get(key, CAMERA_DEFAULT[key]))
        value = self._read_roi_value(camera, key)
        self.roi_vars[(camera, key)].set(str(value))
        self.settings[camera][key] = value
        top = self._read_roi_value(camera, "top_y")
        bottom = self._read_roi_value(camera, "bottom_y")
        if bottom - top < 8:
            self.roi_vars[(camera, key)].set(str(previous_value))
            self.settings[camera][key] = previous_value
            return
        self.stitch_worker.request_reset()
        self.last_merged = None
        self.last_merged_source = None
        self.last_stitch_time = 0.0
        self.last_stitch_submit_time = 0.0
        self.last_detector_submit_time = 0.0
        self.detector.reset_for_view_change()
        self.camera_preview_sequence[camera] = -1

    def save_roi(self):
        for camera in ("cam1", "cam2"):
            for key in CAMERA_DEFAULT:
                self.settings[camera][key] = self._read_roi_value(camera, key)
                self.roi_vars[(camera, key)].set(str(self.settings[camera][key]))

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
        polygon = self._roi_polygon(camera, width, height)

        color = GREEN if self.locked[camera] else YELLOW

        cv2.polylines(
            image,
            [polygon],
            True,
            color,
            2
        )

        return image

    def _roi_polygon(self, camera, width, height):
        """Map the logical 640x480 ROI coordinates onto an image's size."""
        values = {key: self._read_roi_value(camera, key) for key in CAMERA_DEFAULT}
        return roi_polygon_from_values(values, width, height)

    def _roi_mask(self, frame, camera):
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = self._roi_polygon(camera, width, height)
        if abs(cv2.contourArea(polygon.reshape(-1, 1, 2))) < height * width * 0.002:
            return None
        cv2.fillPoly(mask, [polygon], 255)
        return mask

    def _roi_settings_snapshot(self):
        return {
            camera: {
                key: self._read_roi_value(camera, key)
                for key in CAMERA_DEFAULT
            }
            for camera in ("cam1", "cam2")
        }

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
        scale = min(1.0, MERGED_DISPLAY_MAX_EDGE / float(max(frame.shape[:2])))
        if scale < 1.0:
            original_h, original_w = frame.shape[:2]
            frame = cv2.resize(
                frame,
                (max(1, int(round(original_w * scale))),
                 max(1, int(round(original_h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            scaled_detections = []
            for detection in detections:
                item = dict(detection)
                item["box"] = detection["box"] * scale
                mask = detection.get("mask")
                mask_box = detection.get("mask_box")
                if mask is not None and mask_box is not None:
                    left, top, right, bottom = mask_box
                    left, top = int(round(left * scale)), int(round(top * scale))
                    right, bottom = int(round(right * scale)), int(round(bottom * scale))
                    item["mask_box"] = (left, top, right, bottom)
                    item["mask"] = cv2.resize(
                        mask, (max(1, right - left), max(1, bottom - top)),
                        interpolation=cv2.INTER_NEAREST,
                    )
                scaled_detections.append(item)
            detections = scaled_detections
        output = frame.copy()
        for det in detections:
            color = CLASS_COLORS.get(det["class"], (255, 255, 255))
            mask = det.get("mask")
            mask_box = det.get("mask_box")
            if mask is not None and mask_box is not None:
                left, top, right, bottom = mask_box
                left = max(0, min(output.shape[1], int(left)))
                right = max(left, min(output.shape[1], int(right)))
                top = max(0, min(output.shape[0], int(top)))
                bottom = max(top, min(output.shape[0], int(bottom)))
                patch = output[top:bottom, left:right]
                if patch.shape[:2] == mask.shape[:2] and patch.size:
                    selected = mask.astype(bool)
                    if np.any(selected):
                        blended = patch.copy()
                        blended[selected] = color
                        output[top:bottom, left:right] = cv2.addWeighted(
                            blended, 0.38, patch, 0.62, 0
                        )
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        cv2.drawContours(output, contours, -1, color, 2,
                                         offset=(left, top))
            x1, y1, x2, y2 = [int(round(value)) for value in det["box"]]
            x1 = max(0, min(output.shape[1] - 1, x1))
            x2 = max(0, min(output.shape[1] - 1, x2))
            y1 = max(0, min(output.shape[0] - 1, y1))
            y2 = max(0, min(output.shape[0] - 1, y2))
            if x2 > x1 and y2 > y1:
                roi = output[y1:y2 + 1, x1:x2 + 1]
                if roi.size:
                    color_fill = np.empty_like(roi)
                    color_fill[:] = color
                    output[y1:y2 + 1, x1:x2 + 1] = cv2.addWeighted(
                        color_fill, 0.24, roi, 0.76, 0
                    )
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']} {det['confidence']:.2f}"
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

        (frame1, connected1, fps1, w1, h1,
         reconnects1, camera_error1, sequence1) = self.cam1.get_data()
        (frame2, connected2, fps2, w2, h2,
         reconnects2, camera_error2, sequence2) = self.cam2.get_data()
        feeds_ready = frame1 is not None and frame2 is not None
        if feeds_ready != self._feeds_ready:
            self.detector.reset_for_camera_loss()
            self.stitch_worker.request_reset()
            self.last_stitch_submit_time = 0.0
            self.last_stitch_sequence = -1
            self._feeds_ready = feeds_ready
        if not feeds_ready:
            self.last_merged = None
            self.last_merged_source = None

        now = time.monotonic()
        for camera, frame, connected, fps, width, height, sequence in (
            ("cam1", frame1, connected1, fps1, w1, h1, sequence1),
            ("cam2", frame2, connected2, fps2, w2, h2, sequence2),
        ):
            current_sequence = -1 if frame is None else sequence
            should_draw = (
                current_sequence != self.camera_preview_sequence[camera] and
                now - self.camera_preview_time[camera] >= CAMERA_PREVIEW_REFRESH_SECONDS
            )
            if frame is None and self.camera_preview_sequence[camera] != -2:
                self._show_frame(self.camera_labels[camera], None, camera)
                self.camera_preview_sequence[camera] = -2
            elif frame is not None and should_draw:
                preview = cv2.resize(
                    frame, (PANEL_WIDTH, CAMERA_PREVIEW_HEIGHT),
                    interpolation=cv2.INTER_AREA
                )
                preview = self.draw_roi(preview, camera)
                preview = self.draw_center_lines(preview)
                self._show_frame(self.camera_labels[camera], preview, camera)
                self.camera_preview_sequence[camera] = current_sequence
                self.camera_preview_time[camera] = now

        if feeds_ready and now - self.last_stitch_submit_time >= STITCH_REFRESH_SECONDS:
            self.stitch_worker.submit(frame1, frame2, self._roi_settings_snapshot())
            self.last_stitch_submit_time = now
        elif not feeds_ready and now - self.last_stitch_submit_time >= STITCH_REFRESH_SECONDS:
            self.stitch_worker.submit(None, None, self._roi_settings_snapshot())
            self.last_stitch_submit_time = now

        stitch_image, stitch_status, stitch_sequence = self.stitch_worker.snapshot()
        render_stitch = stitch_sequence != self.last_stitch_sequence
        if render_stitch:
            self.last_stitch_sequence = stitch_sequence
            self.last_merged_source = stitch_image
            # Keep detector input in the unrotated source coordinate system.
            self.last_merged = self._rotate_merged_image(stitch_image)
            if (stitch_image is not None and
                    now - self.last_detector_submit_time >= DETECTION_REFRESH_SECONDS):
                self.detector.submit(stitch_image)
                self.last_detector_submit_time = now
        (detections, counts, model_status,
         has_prediction, bom_stable) = self.detector.snapshot()
        self._update_bom_table(detections, counts, model_status,
                               has_prediction, bom_stable)
        if self.last_merged is not None and render_stitch:
            annotated_source = self._annotate_detections(
                self.last_merged_source, detections
            )
            annotated = self._rotate_merged_image(annotated_source)
            self._show_frame(self.merged_label, annotated, "merged")
            if self.merged_fullscreen_label is not None:
                self._show_frame(self.merged_fullscreen_label, annotated,
                                 "merged_fullscreen")
        elif self.last_merged is None and render_stitch:
            self.merged_label.configure(image="", text=stitch_status, fg="white")
            if self.merged_fullscreen_label is not None:
                self.merged_fullscreen_label.configure(image="", text=stitch_status,
                                                       fg="white")
        merge_status_text = f"{stitch_status} | {model_status}"
        if merge_status_text != self._last_merge_status_text:
            self.merge_status_var.set(merge_status_text)
            self._last_merge_status_text = merge_status_text

        cam1_state = "ONLINE" if connected1 else f"RECONNECTING: {camera_error1 or 'no fresh frames'}"
        cam2_state = "ONLINE" if connected2 else f"RECONNECTING: {camera_error2 or 'no fresh frames'}"
        retry1 = f" | reconnects {reconnects1}" if reconnects1 else ""
        retry2 = f" | reconnects {reconnects2}" if reconnects2 else ""
        header_text = (
            f"CAM 1: {cam1_state} {w1}x{h1} | {fps1:.1f} FPS{retry1}     "
            f"CAM 2: {cam2_state} {w2}x{h2} | {fps2:.1f} FPS{retry2}"
        )
        if header_text != self._last_view_header_text:
            self.view_header_var.set(header_text)
            self._last_view_header_text = header_text

    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    def close(self):
        if self.closing:
            return

        self.closing = True
        self.cam1.stop()
        self.cam2.stop()
        self.stitch_worker.close()
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
