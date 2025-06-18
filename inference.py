#!/usr/bin/env python3
"""
inference_origin.py

Run real-vs-synthetic (origin track) inference on your own image list.

Usage:
    python inference_origin.py \
        --data-root . \
        --ann     annotations/testgen_images.txt \
        --ckpt    linear_imaginet_save_origin_calib/model_5.pt \
        --bs      64 \
        --out     results_origin_finetuning_10.csv
"""

import os
import sys
import csv
import argparse
import torch
import torch.nn as nn
from torchvision import transforms

# 1) Make `training_utils` importable
repo_root   = os.path.dirname(os.path.abspath(__file__))
train_utils = os.path.join(repo_root, "training_utils")
sys.path.insert(0, train_utils)

# 2) Import the repo’s classes
from datasets            import ImagiNet
from networks.resnet_big import ConResNet

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", help="Repo or data root",    default=".")
    p.add_argument("--ann",       help="5-col annotation txt", required=True)
    p.add_argument("--ckpt",      help="checkpoint .pt",       required=True)
    p.add_argument("--bs",        help="batch size",           type=int, default=64)
    p.add_argument("--out",       help="output CSV path",      default="results_origin.csv")
    return p.parse_args()

def build_classifier():
    # exactly as in train_linear_classifier_selfcon.py
    return nn.Sequential(
        nn.BatchNorm1d(2048),
        nn.Linear(2048, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(1024, 1)
    )

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # normalize only (no extra test-time aug)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load annotation paths
    all_paths = [L.split(",")[0] for L in open(args.ann, "r").read().splitlines()]

    #  --------- Dataset & Loader ---------
    ds = ImagiNet(
        root_dir         = args.data_root,
        annotations_file = args.ann,
        track            = "origin",
        train            = False,
        test_aug         = False,
        resize           = False,
        anchor           = False,
        transform        = transform
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.bs, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # --------- Backbone ---------
    model = ConResNet(
        selfcon_pos  = [False, True, False],
        selfcon_size = "fc",
        dataset      = "imaginet"
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    # --------- Classifier Head ---------
    cls = build_classifier().to(device).eval()
    cls.load_state_dict(ckpt["classifier"])

    # --------- Inference ---------
    sigmoid = nn.Sigmoid()
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "prob_synthetic", "pred_label"])

        idx = 0
        with torch.no_grad():
            for imgs, _ in loader:
                b = imgs.size(0)
                imgs = imgs.to(device)
                # forward
                _, feat = model.encoder(imgs)
                logits  = cls(feat)
                probs   = sigmoid(logits).view(-1).cpu().tolist()
                preds   = [int(p>0.5) for p in probs]

                # write rows
                for i in range(b):
                    writer.writerow([all_paths[idx], f"{probs[i]:.4f}", preds[i]])
                    idx += 1

    print(f"✅ Done — wrote {idx} rows to {args.out}")

if __name__ == "__main__":
    main()
