"""Train a six-class A-F detector after collecting and labeling merged frames.

This intentionally refuses to train on the six close-up reference photos. They
are visual references, not representative merged-camera detection examples.
"""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
CLASSES = ("A", "B", "C", "D", "E", "F")


def read_labels(label_dir):
    counts = {name: 0 for name in CLASSES}
    files = list(label_dir.glob("*.txt"))
    for path in files:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{path.name}:{line_no}: expected YOLO class x y w h")
            class_id = int(fields[0])
            if class_id < 0 or class_id >= len(CLASSES):
                raise ValueError(f"{path.name}:{line_no}: class id must be 0..5")
            coords = [float(value) for value in fields[1:]]
            if any(value < 0 or value > 1 for value in coords):
                raise ValueError(f"{path.name}:{line_no}: normalized coordinates must be 0..1")
            counts[CLASSES[class_id]] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of training images per epoch; use lower values for CPU laptops")
    parser.add_argument("--weights", default="yolo11n.pt",
                        help="Starting model checkpoint (local .pt file or Ultralytics model name)")
    parser.add_argument("--train-images", default="images/train",
                        help="Training image folder relative to dataset root")
    args = parser.parse_args()
    train_images = ROOT / args.train_images
    val_images = ROOT / "images" / "val"
    train_labels = ROOT / "labels" / Path(args.train_images).name
    val_labels = ROOT / "labels" / "val"
    image_exts = {".jpg", ".jpeg", ".png"}
    train_files = [p for p in train_images.iterdir() if p.suffix.lower() in image_exts]
    val_files = [p for p in val_images.iterdir() if p.suffix.lower() in image_exts]
    if len(train_files) < 30 or len(val_files) < 8:
        sys.exit("Add at least 30 labeled training frames and 8 separate validation frames first. "
                 "Capture different product positions, sizes, lighting, and camera sessions.")
    for split, files, label_dir in (("train", train_files, train_labels),
                                    ("val", val_files, val_labels)):
        missing = [p.name for p in files if not (label_dir / f"{p.stem}.txt").is_file()]
        if missing:
            sys.exit(f"{split}: {len(missing)} image(s) have no matching label text file.")
    train_counts = read_labels(train_labels)
    val_counts = read_labels(val_labels)
    insufficient = [name for name in CLASSES if train_counts[name] < 20]
    if insufficient:
        sys.exit("Need at least 20 labeled training instances per class. Missing/low classes: "
                 + ", ".join(insufficient))
    unvalidated = [name for name in CLASSES if val_counts[name] < 3]
    if unvalidated:
        print("WARNING: validation has fewer than 3 instances for: " + ", ".join(unvalidated))
        print("Any weights from this run are exploratory; this run cannot establish accuracy for those classes.")

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        sys.exit(f"Missing training dependency: {exc}. Install the packages listed in README.md.")

    device = 0 if torch.cuda.is_available() else "cpu"
    batch = -1 if device != "cpu" else 2
    model = YOLO(args.weights)
    runtime_yaml = ROOT / "data_runtime.yaml"
    runtime_yaml.write_text(
        f"path: {ROOT.as_posix()}\n"
        f"train: {args.train_images.replace(chr(92), '/')}\n"
        "val: images/val\n"
        "names:\n"
        "  0: A\n  1: B\n  2: C\n  3: D\n  4: E\n  5: F\n",
        encoding="utf-8",
    )
    result = model.train(
        data=str(runtime_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=min(15, max(5, args.epochs // 3)),
        fraction=args.fraction,
        batch=batch,
        device=device,
        workers=0,
        hsv_h=0.035,
        hsv_s=0.60,
        hsv_v=0.50,
        degrees=4.0,
        translate=0.05,
        scale=0.30,
        fliplr=0.0,
        mosaic=0.35,
        project=str(ROOT / "runs"),
        name="af_detector",
        exist_ok=True,
        seed=42,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best model: {best}")
    print("Validate it on held-out images before using it for production BOM decisions.")


if __name__ == "__main__":
    main()
