"""Make zoomed merged-view training crops so small E parts occupy more pixels."""

from pathlib import Path
import random

from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parent
SOURCE_IMAGE = ROOT / "inbox" / "Merged_photo.png"
SOURCE_LABEL = ROOT / "inbox" / "Merged_photo.txt"
OUT_IMAGES = ROOT / "images" / "merged_mix"
OUT_LABELS = ROOT / "labels" / "merged_mix"
SEED = 610152
COUNT = 120
CROP_W, CROP_H = 1200, 700


def main():
    rng = random.Random(SEED)
    image = Image.open(SOURCE_IMAGE).convert("RGB")
    width, height = image.size
    if CROP_W >= width or CROP_H >= height:
        raise ValueError("Zoom crop must be smaller than the source image")

    boxes = []
    for line in SOURCE_LABEL.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        cls = int(fields[0])
        cx, cy, bw, bh = map(float, fields[1:])
        boxes.append((cls, (cx - bw / 2) * width, (cy - bh / 2) * height,
                      (cx + bw / 2) * width, (cy + bh / 2) * height))

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)
    for index in range(COUNT):
        center_x = width / 2 + rng.uniform(-70, 70)
        center_y = height / 2 + rng.uniform(-35, 35)
        x0 = int(max(0, min(width - CROP_W, center_x - CROP_W / 2)))
        y0 = int(max(0, min(height - CROP_H, center_y - CROP_H / 2)))
        x1, y1 = x0 + CROP_W, y0 + CROP_H
        crop = image.crop((x0, y0, x1, y1))
        crop = ImageEnhance.Brightness(crop).enhance(rng.uniform(0.78, 1.22))
        crop = ImageEnhance.Contrast(crop).enhance(rng.uniform(0.85, 1.18))
        crop = ImageEnhance.Color(crop).enhance(rng.uniform(0.78, 1.22))

        output_labels = []
        for cls, bx0, by0, bx1, by1 in boxes:
            ix0, iy0 = max(bx0, x0), max(by0, y0)
            ix1, iy1 = min(bx1, x1), min(by1, y1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            visible = ((ix1 - ix0) * (iy1 - iy0)) / max(
                (bx1 - bx0) * (by1 - by0), 1.0
            )
            if visible < 0.65:
                continue
            cx = ((ix0 + ix1) / 2 - x0) / CROP_W
            cy = ((iy0 + iy1) / 2 - y0) / CROP_H
            bw = (ix1 - ix0) / CROP_W
            bh = (iy1 - iy0) / CROP_H
            output_labels.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        stem = f"merged_zoom_{SEED}_{index:04d}"
        crop.save(OUT_IMAGES / f"{stem}.jpg", quality=95, subsampling=0)
        (OUT_LABELS / f"{stem}.txt").write_text(
            "\n".join(output_labels) + "\n", encoding="utf-8"
        )
    print(f"Created {COUNT} labeled zoom crops in {OUT_IMAGES}")


if __name__ == "__main__":
    main()
