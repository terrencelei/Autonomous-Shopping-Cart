"""
Download the shopping-cart dataset from Roboflow and fine-tune YOLOv11n on it.

Usage:
    python3 train_cart.py --api-key <YOUR_ROBOFLOW_KEY>

Get a free API key at https://app.roboflow.com (Account → Settings → Roboflow API).
"""

import argparse
import os
from pathlib import Path
from roboflow import Roboflow
from ultralytics import YOLO


WORKSPACE   = "furkan-bakkal"
PROJECT     = "shopping-cart-1r48s"
VERSION     = 1          # bump if Roboflow reports a higher version
BASE_MODEL  = "yolo11n.pt"
EPOCHS      = 50
IMG_SIZE    = 640
OUT_NAME    = "cart_detector"


def download_dataset(api_key: str) -> Path:
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)

    # Try the requested version; fall back to the latest if it doesn't exist.
    try:
        version = project.version(VERSION)
    except Exception:
        version = project.version(project.versions[-1].version)
        print(f"Version {VERSION} not found — using version {version.version}")

    dataset = version.download("yolov8")   # yolov8 format works with ultralytics
    return Path(dataset.location)


def train(dataset_path: Path):
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    model = YOLO(BASE_MODEL)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        name=OUT_NAME,
        exist_ok=True,
        patience=15,       # early stop if no improvement for 15 epochs
        batch=-1,          # auto batch size
    )

    best = Path(f"runs/detect/{OUT_NAME}/weights/best.pt")
    if best.exists():
        import shutil
        dest = Path("cart.pt")
        shutil.copy(best, dest)
        print(f"\nTrained model saved to: {dest.resolve()}")
    else:
        print("\nTraining complete. Find weights in runs/detect/cart_detector/weights/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"),
                        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    args = parser.parse_args()

    if not args.api_key:
        args.api_key = input("Roboflow API key: ").strip()

    print("Downloading dataset...")
    dataset_path = download_dataset(args.api_key)
    print(f"Dataset at: {dataset_path}\n")

    print("Training...")
    train(dataset_path)


if __name__ == "__main__":
    main()
