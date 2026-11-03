"""Browser-facing adapter for the live Jetson detector process."""

from datetime import datetime, timezone
from pathlib import Path
import uuid

from .db import add_detection_event
from .live_state import live_state

REPO_ROOT = Path(__file__).resolve().parents[2]
INBOX_DIR = REPO_ROOT / "component_dataset" / "inbox"


class PipelineService:
    async def snapshot(self) -> dict:
        return live_state.snapshot()

    async def frame_jpeg(self, camera="merged") -> bytes | None:
        return live_state.frame_jpeg(camera)

    async def capture_frame(self) -> str:
        data = live_state.frame_jpeg("capture")
        if not data:
            raise RuntimeError("No merged detector frame is available yet")
        INBOX_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"web_capture_{stamp}_{uuid.uuid4().hex[:6]}.jpg"
        destination = INBOX_DIR / filename
        destination.write_bytes(data)
        state = live_state.snapshot()
        add_detection_event(
            counts=state["counts"],
            expected=state["expected"],
            stable=state["stable"],
            result=state["result"],
            note=f"web capture: {filename}",
        )
        return filename


pipeline = PipelineService()
