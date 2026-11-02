import asyncio
import io
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

from PIL import Image, ImageDraw

from .db import add_detection_event, get_setting_json

REPO_ROOT = "/home/runner/work/inspect_v1.1/inspect_v1.1"
INBOX_DIR = os.path.join(REPO_ROOT, "component_dataset", "inbox")
os.makedirs(INBOX_DIR, exist_ok=True)


@dataclass
class PipelineState:
    counts: dict = field(default_factory=lambda: {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0})
    expected: dict = field(default_factory=lambda: {"A": 5, "B": 0, "C": 0, "D": 0, "E": 2, "F": 10})
    stable: bool = False
    result: str = "UNKNOWN"
    status: str = "Starting"
    last_frame_jpeg: bytes = b""
    updated_at: str = ""


class PipelineService:
    def __init__(self):
        self.state = PipelineState()
        self._lock = asyncio.Lock()
        self._tick = 0

    async def run(self):
        while True:
            await asyncio.sleep(1.0)
            await self._update_demo()

    async def _update_demo(self):
        async with self._lock:
            self._tick += 1
            self.state.expected = get_setting_json("expected_bom", self.state.expected)
            counts = {}
            for key, target in self.state.expected.items():
                if self._tick % 7 == 0:
                    counts[key] = max(0, int(target) + random.choice([-1, 0, 1]))
                else:
                    counts[key] = int(target)
            self.state.counts = counts
            self.state.stable = self._tick % 2 == 0
            self.state.result = "PASS" if self.state.stable and counts == self.state.expected else "FAIL"
            self.state.status = "Demo pipeline active"
            self.state.updated_at = datetime.now(timezone.utc).isoformat()
            self.state.last_frame_jpeg = self._render_frame_bytes()

            if self._tick % 3 == 0:
                add_detection_event(
                    counts=self.state.counts,
                    expected=self.state.expected,
                    stable=self.state.stable,
                    result=self.state.result,
                    note="demo stream",
                )

    def _render_frame_bytes(self) -> bytes:
        w, h = 1280, 720
        img = Image.new("RGB", (w, h), (20, 24, 30))
        d = ImageDraw.Draw(img)
        d.rectangle((30, 30, w - 30, h - 30), outline=(90, 200, 255), width=3)
        d.text((50, 50), "Merged View (Demo)", fill=(255, 255, 255))
        d.text((50, 85), f"Result: {self.state.result} | Stable: {self.state.stable}", fill=(180, 230, 180))
        y = 130
        for key in ("A", "B", "C", "D", "E", "F"):
            d.text(
                (50, y),
                f"{key}: detected={self.state.counts.get(key,0)} expected={self.state.expected.get(key,0)}",
                fill=(230, 230, 230),
            )
            y += 32
        bio = io.BytesIO()
        img.save(bio, format="JPEG", quality=85)
        return bio.getvalue()

    async def snapshot(self) -> dict:
        async with self._lock:
            return {
                "counts": dict(self.state.counts),
                "expected": dict(self.state.expected),
                "stable": self.state.stable,
                "result": self.state.result,
                "status": self.state.status,
                "updated_at": self.state.updated_at,
            }

    async def frame_jpeg(self) -> bytes:
        async with self._lock:
            return self.state.last_frame_jpeg

    async def capture_frame(self) -> str:
        data = await self.frame_jpeg()
        if not data:
            await self._update_demo()
            data = await self.frame_jpeg()
        filename = f"web_capture_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(INBOX_DIR, filename)
        with open(path, "wb") as f:
            f.write(data)
        return path
