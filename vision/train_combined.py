"""
Train a single YOLOv11n model that detects both people and shopping carts.

Strategy:
  - Person data: COCO128 (built into ultralytics), filtered to person class only
  - Cart data:   Roboflow shopping-cart dataset, remapped to class 1
  - Train a single YOLOv11n on both → combined.pt

Usage:
    python3 train_combined.py --api-key <YOUR_ROBOFLOW_KEY>
"""

import argparse
import os
import shutil
from pathlib import Path

import yaml
from roboflow import Roboflow
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

_DIR       = Path(__file__).parent
WORKSPACE  = "furkan-bakkal"
PROJECT    = "shopping-cart-1r48s"
VERSION    = 1

BASE_MODEL = "yolo26n.pt"
OUT_NAME   = "combined_detector"
EPOCHS     = 100
IMG_SIZE   = 640


# ---------------------------------------------------------------------------
# Person data from COCO128
# ---------------------------------------------------------------------------

def extract_coco_persons(dst: Path) -> int:
    """Download COCO128 and copy person-only images+labels to dst."""
    data = check_det_dataset("coco128.yaml")
    coco = Path(data["path"])

    img_src = coco / "images" / "train2017"
    lbl_src = coco / "labels" / "train2017"

    img_dst = dst / "images"
    lbl_dst = dst / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for img in img_src.glob("*.jpg"):
        lbl = lbl_src / (img.stem + ".txt")
        if not lbl.exists():
            continue
        person_lines = [l for l in lbl.read_text().splitlines()
                        if l.strip().startswith("0 ")]
        if not person_lines:
            continue
        shutil.copy(img, img_dst / img.name)
        (lbl_dst / (img.stem + ".txt")).write_text("\n".join(person_lines) + "\n")
        count += 1

    print(f"  COCO128 persons: {count} images")
    return count


# ---------------------------------------------------------------------------
# Cart data from Roboflow
# ---------------------------------------------------------------------------

def download_cart_dataset(api_key: str) -> Path:
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    try:
        version = project.version(VERSION)
    except Exception:
        version = project.version(project.versions[-1].version)
        print(f"Version {VERSION} not found — using version {version.version}")
    dataset = version.download("yolov8")
    return Path(dataset.location)


def copy_cart_split(cart_ds: Path, split: str, dst: Path):
    """Copy cart images + remap labels (0 → 1) for one split."""
    src_img = cart_ds / split / "images"
    src_lbl = cart_ds / split / "labels"
    if not src_img.exists():          # flat layout (darknet download)
        src_img = cart_ds / split
        src_lbl = cart_ds / split

    img_dst = dst / "images"
    lbl_dst = dst / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    images = list(src_img.glob("*.jpg")) + list(src_img.glob("*.png"))
    for img in images:
        shutil.copy(img, img_dst / img.name)

        src_l = src_lbl / (img.stem + ".txt")
        dst_l = lbl_dst / (img.stem + ".txt")
        if src_l.exists():
            remapped = []
            for line in src_l.read_text().splitlines():
                parts = line.split()
                if parts:
                    parts[0] = "1"   # cart → class 1
                    remapped.append(" ".join(parts))
            dst_l.write_text("\n".join(remapped) + "\n")
        else:
            dst_l.touch()

    print(f"  Cart {split}: {len(images)} images")


# ---------------------------------------------------------------------------
# Build combined dataset
# ---------------------------------------------------------------------------

def build_combined_dataset(cart_ds: Path) -> Path:
    combined = _DIR / "combined_dataset"
    if combined.exists():
        shutil.rmtree(combined)

    print("Extracting COCO128 persons...")
    extract_coco_persons(combined / "train")

    print("Copying cart dataset...")
    copy_cart_split(cart_ds, "train", combined / "train")
    copy_cart_split(cart_ds, "valid", combined / "valid")

    data_yaml = combined / "data.yaml"
    data_yaml.write_text(yaml.dump({
        "path": str(combined),
        "train": "train/images",
        "val":   "valid/images",
        "names": {0: "person", 1: "cart"},
    }))
    print(f"  data.yaml → {data_yaml}")
    return combined


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(combined_ds: Path):
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(combined_ds / "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        name=OUT_NAME,
        exist_ok=True,
        patience=20,
        batch=-1,
        device="mps",
    )

    best = Path(f"runs/detect/{OUT_NAME}/weights/best.pt")
    if best.exists():
        dest = _DIR / "combined.pt"
        shutil.copy(best, dest)
        print(f"\nSaved combined model → {dest.resolve()}")
    else:
        print("\nTraining complete. Find weights in runs/detect/combined_detector/weights/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        args.api_key = input("Roboflow API key: ").strip()

    print("Downloading cart dataset...")
    cart_ds = download_cart_dataset(args.api_key)
    print(f"Dataset at: {cart_ds}\n")

    print("Building combined dataset...")
    combined_ds = build_combined_dataset(cart_ds)

    print("\nTraining combined model...")
    train(combined_ds)


if __name__ == "__main__":
    main()
