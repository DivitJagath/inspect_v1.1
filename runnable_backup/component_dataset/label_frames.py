"""Draw YOLO bounding boxes on saved merged-camera frames."""

from pathlib import Path
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
INBOX = ROOT / "inbox"
CLASSES = ("A", "B", "C", "D", "E", "F")
EXTS = {".jpg", ".jpeg", ".png"}


class Labeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Merged Camera A-F Dataset Labeler")
        self.root.geometry("1280x900")
        self.files = sorted(p for p in INBOX.glob("*") if p.suffix.lower() in EXTS)
        self.index = 0
        self.boxes = []
        self.drag_start = None
        self.scale = 1.0
        self.image_size = (0, 0)
        self.class_var = tk.StringVar(value="A")
        self.status = tk.StringVar(value="Open the camera app and save merged training frames first.")

        bar = ttk.Frame(root, padding=8)
        bar.pack(fill="x")
        ttk.Label(bar, text="Class:").pack(side="left")
        self.selector = ttk.Combobox(bar, textvariable=self.class_var,
                                     values=CLASSES, state="readonly", width=6)
        self.selector.pack(side="left", padx=6)
        for i, name in enumerate(CLASSES, 1):
            root.bind(str(i), lambda _event, n=name: self.class_var.set(n))
        ttk.Button(bar, text="Undo last box", command=self.undo).pack(side="left", padx=5)
        ttk.Button(bar, text="Previous", command=lambda: self.change(-1)).pack(side="left", padx=5)
        ttk.Button(bar, text="Next", command=lambda: self.change(1)).pack(side="left", padx=5)
        ttk.Button(bar, text="Save to TRAIN", command=lambda: self.save("train")).pack(side="right", padx=5)
        ttk.Button(bar, text="Save to VALIDATION", command=lambda: self.save("val")).pack(side="right", padx=5)
        ttk.Button(bar, text="Save EMPTY TRAIN", command=lambda: self.save("train", allow_empty=True)).pack(side="right", padx=5)
        ttk.Button(bar, text="Save EMPTY VAL", command=lambda: self.save("val", allow_empty=True)).pack(side="right", padx=5)
        self.canvas = tk.Canvas(root, bg="#202020", width=1200, height=760,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=8)
        self.canvas.bind("<ButtonPress-1>", self.begin_box)
        self.canvas.bind("<B1-Motion>", self.drag_box)
        self.canvas.bind("<ButtonRelease-1>", self.end_box)
        ttk.Label(root, textvariable=self.status, padding=8).pack(fill="x")
        root.bind("<BackSpace>", lambda _event: self.undo())
        root.bind("<Left>", lambda _event: self.change(-1))
        root.bind("<Right>", lambda _event: self.change(1))
        root.update_idletasks()
        self.load()

    def load(self):
        self.canvas.delete("all")
        self.boxes.clear()
        if not self.files:
            self.status.set(f"No images found in {INBOX}. Save training frames from the camera app.")
            return
        self.index %= len(self.files)
        path = self.files[self.index]
        image = Image.open(path).convert("RGB")
        self.image_size = image.size
        max_w, max_h = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.scale = min(max_w / image.width, max_h / image.height, 1.0)
        size = (max(1, round(image.width * self.scale)), max(1, round(image.height * self.scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.config(width=size[0], height=size[1])
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw", tags="base")
        self.status.set(f"{self.index + 1}/{len(self.files)}  {path.name} — draw a tight box around each visible target; choose its class first.")

    def begin_box(self, event):
        self.drag_start = (event.x, event.y)
        self.canvas.delete("draft")

    def drag_box(self, event):
        if self.drag_start:
            self.canvas.delete("draft")
            self.canvas.create_rectangle(*self.drag_start, event.x, event.y,
                                         outline="#00ff40", width=2, tags="draft")

    def end_box(self, event):
        if not self.drag_start:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        self.drag_start = None
        if abs(x1 - x0) < 4 or abs(y1 - y0) < 4:
            self.canvas.delete("draft")
            return
        self.canvas.delete("draft")
        box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), self.class_var.get())
        self.boxes.append(box)
        self.draw_boxes()

    def draw_boxes(self):
        self.canvas.delete("labelbox")
        for i, (x0, y0, x1, y1, name) in enumerate(self.boxes, 1):
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#00ff40", width=2, tags="labelbox")
            self.canvas.create_text(x0 + 3, y0 + 3, text=f"{name} {i}", anchor="nw",
                                    fill="#ffff00", font=("Arial", 11, "bold"), tags="labelbox")

    def undo(self):
        if self.boxes:
            self.boxes.pop()
            self.draw_boxes()

    def change(self, offset):
        if self.files:
            self.index = (self.index + offset) % len(self.files)
            self.load()

    def save(self, split, allow_empty=False):
        if not self.files or (not self.boxes and not allow_empty):
            messagebox.showwarning("No boxes", "Draw component boxes, or use an EMPTY button for a clear negative frame.")
            return
        image_path = self.files[self.index]
        image_out = ROOT / "images" / split / image_path.name
        label_out = ROOT / "labels" / split / f"{image_path.stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        width, height = self.image_size
        rows = []
        for x0, y0, x1, y1, name in self.boxes:
            left, right = x0 / self.scale, x1 / self.scale
            top, bottom = y0 / self.scale, y1 / self.scale
            xc, yc = (left + right) / (2 * width), (top + bottom) / (2 * height)
            bw, bh = (right - left) / width, (bottom - top) / height
            rows.append(f"{CLASSES.index(name)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        shutil.copy2(image_path, image_out)
        label_out.write_text("\n".join(rows) + "\n", encoding="utf-8")
        self.status.set(f"Saved {len(rows)} boxes to {split}: {image_path.name}")
        self.change(1)


if __name__ == "__main__":
    INBOX.mkdir(parents=True, exist_ok=True)
    app = tk.Tk()
    Labeler(app)
    app.mainloop()
