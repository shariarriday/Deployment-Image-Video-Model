"""
Inference with a trained YOLO26 model on a single image.
Draws bounding boxes and saves the annotated image.

Usage:
    python infer_yolo26.py --source path/to/image.jpg
"""

import argparse
from pathlib import Path
from PIL import Image
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
        save=False,
        save_txt=False,
        verbose=False,
    )

    r = results[0]
    names = model.names

    print(f"{len(r.boxes)} detection(s) found\n")

    if len(r.boxes) == 0:
        print("No detections — nothing to crop.")
    else:
        image = Image.open(source).convert("RGB")
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls)
            label  = names[cls_id]
            conf   = float(box.conf)
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())

            crop = image.crop((x1, y1, x2, y2))

            # e.g. image_dog_0_0.91.jpg
            out_name = f"result.jpg"
            out_path = output_dir / out_name
            crop.save(out_path)

            print(f"  [{i}] {label} ({conf:.2f}) → {out_path}")

        print(f"\nAll crops saved to: {output_dir}")

    # r = results[0]
    # print(f"{len(r.boxes)} detection(s) found")
    # print(f"Annotated image saved to: {Path(args.output) / 'results' / source.name}")
