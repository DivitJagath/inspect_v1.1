"""Create labeled synthetic training variations from one annotated merged frame.

All generated images stay in the training split. They are useful augmentation,
not independent validation data. Paint overlays are constrained to A/E boxes.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ("A", "B", "C", "D", "E", "F")
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def parse_yolo(path: Path, width: int, height: int):
    boxes = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no}: expected class x_center y_center width height")
        cls = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        if cls not in range(len(CLASS_NAMES)):
            raise ValueError(f"{path}:{line_no}: class id must be 0..5")
        if not all(0 <= v <= 1 for v in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
            raise ValueError(f"{path}:{line_no}: invalid normalized box")
        boxes.append((cls, cx, cy, bw, bh))
    return boxes


def paint_overlay(image: Image.Image, boxes, rng: random.Random, coverage: str):
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for cls, cx, cy, bw, bh in boxes:
        if cls not in (0, 4) or rng.random() > 0.78:
            continue
        x0 = (cx - bw / 2) * image.width
        y0 = (cy - bh / 2) * image.height
        x1 = (cx + bw / 2) * image.width
        y1 = (cy + bh / 2) * image.height
        # Keep a narrow edge visible so the synthetic example still has shape cues.
        margin_x = max(1, (x1 - x0) * 0.08)
        margin_y = max(1, (y1 - y0) * 0.08)
        x0, x1 = x0 + margin_x, x1 - margin_x
        y0, y1 = y0 + margin_y, y1 - margin_y
        if x1 <= x0 or y1 <= y0:
            continue
        fraction = rng.uniform(0.18, 0.42) if coverage == "light" else (
            rng.uniform(0.42, 0.70) if coverage == "partial" else rng.uniform(0.70, 0.96)
        )
        pw, ph = (x1 - x0) * fraction, (y1 - y0) * fraction
        px = rng.uniform(x0, max(x0, x1 - pw))
        py = rng.uniform(y0, max(y0, y1 - ph))
        color = rng.choice(((145, 63, 55), (183, 76, 62), (112, 77, 57),
                            (157, 129, 104), (90, 91, 86)))
        alpha = rng.randint(180, 255)
        points = []
        for i in range(12):
            angle = 2 * math.pi * i / 12
            radius = rng.uniform(0.78, 1.08)
            points.append((px + pw / 2 + math.cos(angle) * pw / 2 * radius,
                           py + ph / 2 + math.sin(angle) * ph / 2 * radius))
        draw.polygon(points, fill=(*color, alpha))
        if rng.random() < 0.4:
            draw.ellipse((px + pw * 0.2, py + ph * 0.25,
                          px + pw * 0.8, py + ph * 0.75), fill=(*color, alpha))
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def add_shadow(image: Image.Image, rng: random.Random):
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = image.size
    if rng.random() < 0.65:
        x0, x1 = sorted((rng.randint(0, w - 1), rng.randint(0, w - 1)))
        y0, y1 = sorted((rng.randint(0, h - 1), rng.randint(0, h - 1)))
        draw.polygon([(x0, y0), (x1, max(0, y0 - h * .15)),
                      (min(w, x1 + w * .2), y1), (max(0, x0 - w * .1), y1)],
                     fill=(15, 13, 12, rng.randint(12, 48)))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=rng.uniform(18, 75)))
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def add_light_artifacts(image: Image.Image, rng: random.Random):
    """Simulate uneven exposure and soft reflected glare without changing labels.

    The overlays are intentionally soft and limited so the generated target still
    contains some shape evidence. Fully clipped/featureless objects cannot be
    learned from pixels, so this does not pretend to synthesize that information.
    """
    width, height = image.size
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # Local camera/metal glare: a soft, broad highlight, sometimes crossing the
    # product area but not routinely erasing an entire component.
    if rng.random() < 0.62:
        cx = rng.uniform(0.08, 0.92) * width
        cy = rng.uniform(0.10, 0.90) * height
        rx = rng.uniform(0.035, 0.15) * width
        ry = rng.uniform(0.025, 0.11) * height
        alpha = rng.randint(12, 62)
        draw.ellipse((cx-rx, cy-ry, cx+rx, cy+ry),
                     fill=(255, 250, 235, alpha))
        layer = layer.filter(ImageFilter.GaussianBlur(radius=max(3, min(width, height)*0.018)))

    # A mild directional cast can shift the metal from cool gray to warm white.
    if rng.random() < 0.45:
        tint = rng.choice(((255, 244, 224), (224, 238, 255), (255, 255, 255)))
        tint_layer = Image.new("RGBA", image.size, (*tint, rng.randint(5, 20)))
        layer = Image.alpha_composite(layer, tint_layer)
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def transform(source: Image.Image, boxes, rng: random.Random, index: int):
    image = source.copy().convert("RGB")
    paint_mode = ("none", "none", "none", "light", "partial", "heavy")[index % 6]
    if paint_mode != "none":
        image = paint_overlay(image, boxes, rng, paint_mode)
    image = add_shadow(image, rng)
    # Cover underexposure, ordinary exposure, and bright camera exposure. Some
    # bright cases include local glare to mimic uneven light on reflective metal.
    lighting = index % 5
    if lighting == 0:
        brightness = rng.uniform(0.48, 0.75)
    elif lighting == 1:
        brightness = rng.uniform(0.76, 1.12)
    elif lighting in (2, 3):
        brightness = rng.uniform(1.12, 1.68)
    else:
        brightness = rng.uniform(0.72, 1.52)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.62, 1.48))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.35, 1.55))
    if lighting in (2, 3, 4) and rng.random() < 0.82:
        image = add_light_artifacts(image, rng)
    if rng.random() < 0.28:
        image = ImageOps.grayscale(image).convert("RGB")
    if rng.random() < 0.22:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 1.1)))
    if rng.random() < 0.18:
        image = ImageEnhance.Sharpness(image).enhance(rng.uniform(1.2, 2.0))
    if rng.random() < 0.28:
        gamma = rng.uniform(0.72, 1.35)
        lut = [min(255, max(0, int((v / 255) ** gamma * 255))) for v in range(256)]
        image = image.point(lut * 3)
    return image, paint_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Merged source frame")
    parser.add_argument("--count", type=int, default=240, help="Number of synthetic training images")
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    image_path = args.image.resolve()
    label_path = image_path.with_suffix(".txt")
    if not image_path.is_file() or not label_path.is_file():
        raise SystemExit(f"Need both {image_path.name} and its same-name YOLO label file in {image_path.parent}")
    if args.count < 1:
        raise SystemExit("--count must be at least 1")

    source = Image.open(image_path).convert("RGB")
    boxes = parse_yolo(label_path, *source.size)
    if not boxes:
        raise SystemExit("The source labels contain no objects.")
    image_dir = ROOT / "images" / "train"
    label_dir = ROOT / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    labels = label_path.read_text(encoding="utf-8").strip() + "\n"
    rng = random.Random(args.seed)
    generated = []
    for i in range(args.count):
        name = f"merged_synth_{args.seed}_{i:04d}.jpg"
        out_image = image_dir / name
        out_label = label_dir / f"{Path(name).stem}.txt"
        if out_image.exists() and out_label.exists():
            generated.append((name, "already generated"))
            continue
        if out_image.exists() != out_label.exists():
            raise SystemExit(f"Incomplete existing pair; inspect before continuing: {name}")
        augmented, mode = transform(source, boxes, rng, i)
        augmented.save(out_image, quality=94, subsampling=0)
        out_label.write_text(labels, encoding="utf-8")
        generated.append((name, mode))

    report = ROOT / f"synthetic_generation_{args.seed}.txt"
    report.write_text(
        "Synthetic merged-image augmentation report\n"
        f"Source: {image_path}\nCount: {len(generated)}\nSeed: {args.seed}\n"
        "All generated images are training-only. Do not copy these into validation.\n"
        "Paint is a visual simulation constrained to A/E boxes; it does not replace real painted captures.\n"
        "Lighting variants include dark/bright exposure, contrast/color changes, and soft local glare.\n"
        "They are synthetic and do not establish accuracy on real glare or fully clipped white pixels.\n\n"
        + "\n".join(f"{name}: {mode} paint" for name, mode in generated) + "\n",
        encoding="utf-8",
    )
    completed = sum(mode != "already generated" for _, mode in generated)
    print(f"Created {completed} new variants; {len(generated)} total labeled variants in {image_dir}")
    print(f"Source labels: {label_path}")
    print(f"Generation log: {report}")
    print("These variants are correlated to one source frame and are not validation evidence.")


if __name__ == "__main__":
    main()
