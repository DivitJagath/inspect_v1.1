
#!/usr/bin/env python3
"""
Two-camera RTSP viewer with independent ROI controls.

Controls:
- Fullscreen button at top-left; Esc exits fullscreen.
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
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from PIL import Image, ImageTk


# ============================================================
# CONFIGURATION
# ============================================================

CAM1_URL = "rtsp://admin:admin@192.168.1.10:554/profile1"
CAM2_URL = "rtsp://admin:admin@192.168.1.20:554/profile1"

PANEL_WIDTH = 640
PANEL_HEIGHT = 480
STATUS_HEIGHT = 75

GREEN = (0, 255, 0)       # OpenCV BGR
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)

DASH_LENGTH = 16
DASH_GAP = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "roi_settings.json")

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

        while self.running:
            try:
                if capture is None or not capture.isOpened():
                    if capture is not None:
                        capture.release()

                    capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

                    if not capture.isOpened():
                        with self.lock:
                            self.connected = False
                        time.sleep(2)
                        continue

                    # These settings are best-effort and backend-dependent.
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                ok, frame = capture.read()

                if not ok or frame is None:
                    with self.lock:
                        self.connected = False
                    capture.release()
                    capture = None
                    time.sleep(0.5)
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

            except Exception:
                with self.lock:
                    self.connected = False
                time.sleep(1)

        if capture is not None:
            capture.release()

    def get_data(self):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            return (
                frame,
                self.connected,
                self.fps,
                self.width,
                self.height,
            )

    def stop(self):
        self.running = False


# ============================================================
# MAIN GUI
# ============================================================

class CombinedCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Combined Camera Viewer")
        self.root.geometry("1450x850")
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
        self.video_labels = {}
        self.tk_images = {}

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

        # Top-left fullscreen button.
        self.fullscreen_button = ttk.Button(
            self.toolbar,
            text="FULL SCREEN VIEW",
            command=self.toggle_fullscreen
        )
        self.fullscreen_button.pack(side="left", padx=(0, 8))

        ttk.Label(
            self.toolbar,
            text="TWO-CAMERA LIVE VIEW",
            font=("Arial", 14, "bold"),
            background="#202020",
            foreground="white"
        ).pack(side="left", padx=12)

        # Top-right global center-line button.
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

        # Camera view header is created BEFORE buttons that use it.
        self.view_header = tk.Frame(self.view, bg="#202020")
        self.view_header.pack(fill="x", pady=(0, 5))

        self.view_header_var = tk.StringVar(value="Waiting for cameras...")
        ttk.Label(
            self.view_header,
            textvariable=self.view_header_var,
            background="#202020",
            foreground="white",
            font=("Arial", 11, "bold")
        ).pack(side="left", padx=5)

        self.video_label = tk.Label(
            self.view,
            bg="black",
            bd=0
        )
        self.video_label.pack(fill="both", expand=True)

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

    def make_panel(self, frame, camera, connected, fps, width, height):
        # The panel is fixed-size for consistent side-by-side layout.
        panel = np.zeros(
            (STATUS_HEIGHT + PANEL_HEIGHT, PANEL_WIDTH, 3),
            dtype=np.uint8
        )

        # Black status band ABOVE the image.
        if self.text_visible[camera]:
            status = (
                f"{camera.upper()} | "
                f"{'CONNECTED' if connected else 'RECONNECTING'}"
            )
            detail = (
                f"FPS: {fps:.1f} | SOURCE: {width} x {height}"
                if width and height else "Waiting for camera frames..."
            )

            cv2.putText(
                panel, status, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2
            )
            cv2.putText(
                panel, detail, (10, 57),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, YELLOW, 1
            )

        if frame is None:
            cv2.putText(
                panel, f"{camera.upper()} - NO SIGNAL",
                (90, STATUS_HEIGHT + PANEL_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2
            )
            return panel

        # Resize the camera image into a 640x480 preview.
        # ROI values are defined relative to this preview.
        frame = cv2.resize(
            frame,
            (PANEL_WIDTH, PANEL_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        # Draw ROI and then the global center lines.
        frame = self.draw_roi(frame, camera)
        frame = self.draw_center_lines(frame)

        panel[
            STATUS_HEIGHT:STATUS_HEIGHT + PANEL_HEIGHT,
            :
        ] = frame

        return panel

    # --------------------------------------------------------
    # FULLSCREEN
    # --------------------------------------------------------

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

        frame1, connected1, fps1, w1, h1 = self.cam1.get_data()
        frame2, connected2, fps2, w2, h2 = self.cam2.get_data()

        panel1 = self.make_panel(
            frame1, "cam1", connected1, fps1, w1, h1
        )
        panel2 = self.make_panel(
            frame2, "cam2", connected2, fps2, w2, h2
        )

        combined = np.hstack((panel1, panel2))

        if self.fullscreen:
            available_width = max(1, self.root.winfo_width())
            available_height = max(1, self.root.winfo_height())
        else:
            available_width = max(1, self.video_label.winfo_width())
            available_height = max(1, self.video_label.winfo_height())

        source_h, source_w = combined.shape[:2]

        # Preserve aspect ratio: never stretch or crop the combined view.
        scale = min(
            available_width / source_w,
            available_height / source_h
        )

        new_width = max(1, int(source_w * scale))
        new_height = max(1, int(source_h * scale))

        resized = cv2.resize(
            combined,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        )

        canvas = np.zeros(
            (available_height, available_width, 3),
            dtype=np.uint8
        )

        x_offset = (available_width - new_width) // 2
        y_offset = (available_height - new_height) // 2

        canvas[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width
        ] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.tk_images["combined"] = ImageTk.PhotoImage(image)

        self.video_label.configure(image=self.tk_images["combined"])

        self.view_header_var.set(
            f"CAM 1: {w1}x{h1} | {fps1:.1f} FPS     "
            f"CAM 2: {w2}x{h2} | {fps2:.1f} FPS"
        )

        self.root.after(30, self.update_view)

    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    def close(self):
        if self.closing:
            return

        self.closing = True
        self.cam1.stop()
        self.cam2.stop()
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
