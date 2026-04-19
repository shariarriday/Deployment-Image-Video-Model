"""
Inference with a trained YOLO26 model on a single image.
Draws bounding boxes and saves the annotated image.

Usage:
    python infer_yolo26.py --source path/to/image.jpg
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

DATASET_ROOT    = Path(__file__).parent
DEFAULT_WEIGHTS = DATASET_ROOT / "runs" / "yolo26x_no_aug" / "weights" / "best.pt"
OUTPUT_DIR      = DATASET_ROOT / "inference_output"

CONF_THRESH = 0.25
IOU_THRESH  = 0.45
IMG_SIZE    = 640


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",  required=True, type=str, help="Path to input image")
    p.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    p.add_argument("--conf",    type=float, default=CONF_THRESH)
    p.add_argument("--iou",     type=float, default=IOU_THRESH)
    p.add_argument("--output",  type=str, default=str(OUTPUT_DIR))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}\nTrain first with train_yolo26m.py")

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Image not found: {source}")

    print(f"Weights : {weights}")
    print(f"Image   : {source}\n")

    model = YOLO(str(weights))

    results = model.predict(
        source=str(source),
        conf=args.conf,
        iou=args.iou,
        imgsz=IMG_SIZE,
        save=True,
        save_txt=False,
        project=str(args.output),
        name="results",
        exist_ok=True,
        line_width=2,
    )

    r = results[0]
    print(f"{len(r.boxes)} detection(s) found")
    print(f"Annotated image saved to: {Path(args.output) / 'results' / source.name}")
