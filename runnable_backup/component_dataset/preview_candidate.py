"""Render a merged-image preview using the viewer's duplicate suppression."""

from collections import Counter
from pathlib import Path
import sys

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from combined_Cameras_ import suppress_detection_duplicates  # noqa: E402


def main():
    weights = ROOT / "runs" / "af_detector" / "weights" / "best.pt"
    source = ROOT / "inbox" / "Merged_photo.png"
    image = cv2.imread(str(source))
    model = YOLO(str(weights))
    result = model.predict(image, imgsz=1280, conf=0.05, iou=0.50,
                           max_det=80, verbose=False)[0]
    detections = []
    for box in result.boxes:
        name = model.names[int(box.cls[0])]
        coords = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = coords.tolist()
        detections.append({
            "class": name,
            "confidence": float(box.conf[0]),
            "box": coords,
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
        })

    kept = suppress_detection_duplicates(detections, image.shape[1], image.shape[0])
    colors = {"A": (0, 255, 0), "E": (0, 255, 255), "F": (255, 0, 255)}
    for item in kept:
        x1, y1, x2, y2 = [int(round(value)) for value in item["box"]]
        color = colors[item["class"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        text = f"{item['class']} {item['confidence']:.2f}"
        cv2.putText(image, text, (x1, max(24, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    output = ROOT / "runs" / "detect" / "final_candidate_check" / "Merged_photo.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)
    print("Filtered counts:", dict(Counter(item["class"] for item in kept)))
    print("Saved preview:", output)


if __name__ == "__main__":
    main()
