import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import numpy as np

import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core import function


# ===============================
# 2. GIVE IMAGE PATH ONLY
# ===============================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",  required=True, type=str, help="Path to input image")
    p.add_argument("--weights", required=True, type=str, help="Path to model weights")
    p.add_argument("--output",  required=True, type=str, help="Path to save output image")
    p.add_argument("--cfg",     required=True, type=str, help="Path to config file")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # ===============================
    # 1. SET YOUR PROJECT PATHS
    # ===============================

    CFG_FILE =  args.cfg

    MODEL_FILE = args.weights

    OUTPUT_FOLDER = args.output

    IMAGE_PATH = args.source
    
    # ===============================
    # 3. LOAD CONFIG
    # ===============================

    args = argparse.Namespace(
        cfg=CFG_FILE,
        model_file=MODEL_FILE,
        source=IMAGE_PATH
    )

    update_config(config, args)

    
    # ===============================
    # 4. PREPARE IMAGE DATA
    # ===============================

    img = Image.open(IMAGE_PATH).convert("RGB")
    w, h = img.size

    myData = {
        "path": IMAGE_PATH,
        "center_w": w / 2,
        "center_h": h / 2,
        "scale": max(w, h) / 200.0
    }


    # ===============================
    # 5. LOAD MODEL
    # ===============================

    device = torch.device("cpu")

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    model = models.get_face_alignment_net(config)
    model = nn.DataParallel(model).to(device)

    checkpoint = torch.load(MODEL_FILE, map_location=device)

    if hasattr(checkpoint, "state_dict"):
        checkpoint = checkpoint.state_dict()

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    try:
        model.load_state_dict(checkpoint)
    except:
        model.module.load_state_dict(checkpoint)

    model.eval()


    # ===============================
    # 6. RUN INFERENCE
    # ===============================

    dataset_type = get_dataset(config)

    test_loader = DataLoader(
        dataset=dataset_type(config, myData, is_train=False),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    nme, predictions = function.inference(config, test_loader, model)


    # ===============================
    # 7. CONVERT OUTPUT TO DICTIONARY
    # ===============================

    if isinstance(predictions, torch.Tensor):
        pred = predictions.detach().cpu().numpy()
    else:
        pred = np.asarray(predictions)

    pred = pred.flatten()

    landmarks_dict = {}

    for i in range(0, len(pred), 2):
        point_no = (i // 2) + 1
        landmarks_dict[f"landmark_{point_no}"] = {
            "x": float(pred[i]),
            "y": float(pred[i + 1])
        }
    print(pred)
    print("\nOutput tensor shape:", pred.shape)
    print("\nLandmarks dictionary:")
    print(landmarks_dict)


    # ===============================
    # 8. SAVE ANNOTATED IMAGE
    # ===============================

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    annotated_img = Image.open(IMAGE_PATH).convert("RGB")
    draw = ImageDraw.Draw(annotated_img)

    radius = max(2, int(max(w, h) / 250))

    for point in landmarks_dict.values():
        x = point["x"]
        y = point["y"]

        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 0, 0)
        )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        os.path.splitext(os.path.basename(IMAGE_PATH))[0] + "_landmarks.png"
    )

    annotated_img.save(output_path)

    print("\nAnnotated image saved at:")
    print(output_path)