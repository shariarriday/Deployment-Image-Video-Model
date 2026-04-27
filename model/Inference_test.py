#Import packages
# PyTorch
import torch
from torchvision import models
from torch import cuda 
import torch.nn as nn
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
import numpy as np
from selfonn import SelfONNLayer
import timm
# PyTorch
import torch
from torchvision import transforms, models
from torch import optim, cuda, tensor
from torch.utils.data import DataLoader
import torch.nn as nn
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
import numpy as np
import os
from os import path
from importlib import import_module
# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
# customized functions 
from utils import *
from models import *

from torch.serialization import SourceChangeWarning
for warning in [UserWarning, SourceChangeWarning, Warning]:
    warnings.filterwarnings("ignore", category=warning)



#Load Models
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam.utils.image import preprocess_image
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",  required=True, type=str, help="Path to input image")
    p.add_argument("--weights", type=str, help="Path to model weights (pt file)")
    p.add_argument("--output",  type=str, default="result.json", help="Directory to save output plots")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_path = args.source
    pt_file    = args.weights
    output_file = args.output

    name_to_idx  = {
        'Grade 1': 0,
        'Grade 2': 1,
        'Grade 3': 2,
        'Grade 4': 3,
    }
    num_classes = max(name_to_idx.values()) + 1
    idx_to_name = [''] * num_classes
    for k, v in name_to_idx.items():
        idx_to_name[v] = k

    target_size = (224, 224)
    MEAN = [0.62553333, 0.5221, 0.507]
    STD  = [0.2594, 0.1777, 0.1354]

    # ---- 2) Load model ----
    ckpt = torch.load(pt_file, weights_only= False, map_location='cpu')
    model = ckpt['model']
    del ckpt
    device = torch.device('cpu')

    # Remove DataParallel wrapper if model was saved with multiple GPUs
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model = model.to(device)
    model.eval()


    ###Got prediction
    # ---- 3) Prepare image (RGB) ----
    img = Image.open(image_path).convert('RGB').resize(target_size, Image.BILINEAR)
    img_np = np.float32(img) / 255.0                      # HxWx3 in [0,1]
    input_tensor = preprocess_image(img_np, mean=MEAN, std=STD).to(device)  # [1,3,H,W]

    # ---- 4) Inference ----
    with torch.no_grad():
        out = model(input_tensor)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, dict):
            for key in ['logits', 'out', 'pred', 'y', 'cls_logits']:
                if key in out:
                    out = out[key]
                    break
        if out.ndim == 4:  # [B,C,H,W] -> [B,C]
            out = torch.nn.functional.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
        probs = torch.softmax(out, dim=1).squeeze(0).detach().cpu().numpy()

    assert probs.shape[0] == num_classes, f"Model outputs {probs.shape[0]} classes, but mapping has {num_classes}."

    # ---- 5) Pretty plotting helpers ----
    def plot_pred_image(image_rgb_np, title_text=None, save_path=None):
        """Show the input image with an optional title."""
        plt.figure(figsize=(4.5, 4.5))
        plt.imshow(image_rgb_np)
        plt.axis('off')
        if title_text:
            plt.title(title_text, fontsize=12, pad=10)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.show()

    def plot_topk_bar(probs, idx_to_name, topk=5, save_path=None):
        """Horizontal bar chart of Top-K predictions with percentages."""
        top_idx = probs.argsort()[::-1][:topk]
        labels = [idx_to_name[i] for i in top_idx]
        values = probs[top_idx]

        plt.figure(figsize=(7, 3 + 0.35 * topk))
        y_pos = np.arange(len(labels))[::-1]  # highest on top visually
        plt.barh(y_pos, values)
        plt.yticks(y_pos, labels, fontsize=10)
        plt.xlabel("Probability", fontsize=10)
        plt.xlim(0.0, 1.0)
        plt.title(f"Top-{topk} Predictions", fontsize=12, pad=8)

        # annotate with % at end of bars
        for yi, v in zip(y_pos, values):
            plt.text(v + 0.01, yi, f"{v*100:.1f}%", va='center', fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.show()

    # ---- 6) Report + plots ----
    pred_idx = int(probs.argmax())
    pred_name = idx_to_name[pred_idx]
    pred_p = probs[pred_idx]

    out = probs.argsort()[::-1][:5]

    #Write to json
    import json
    output_data = {
        "predicted_label": pred_name,
        "predicted_probability": float(pred_p),
        "top_5_predictions": [
            {"label": idx_to_name[i], "probability": float(probs[i])}
            for i in out
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    # Figure 1: input image with predicted label
    # plot_pred_image(img_np, title_text=f"Predicted: {pred_name}  (p={pred_p:.3f})")

    # Figure 2: Top-K bar chart
    # plot_topk_bar(probs, idx_to_name, topk=5)


