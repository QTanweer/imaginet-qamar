#!/usr/bin/env python3
"""
Quick inference script for ImagiNet origin-track.
Place it in the repo root and run:
python inference_imaginet.py \
       --root . \
       --ann  annotations/my_sora_test.txt \
       --ckpt ImagiNet\ Checkpoints/imaginet_selfcon_origin.pt \
       --bs   64
"""

import os, sys, csv, argparse, torch, torch.nn as nn
from torchvision import transforms

# --- make repo code importable ---
repo_path = os.path.abspath("training_utils")
sys.path.insert(0, repo_path)

# --- local libs from repo ---
from training_utils.datasets import ImagiNet          # :contentReference[oaicite:0]{index=0}
from training_utils.networks.resnet_big import ConResNet  # same as in eval script

# ---------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".", help="repo root or dataset root")
    p.add_argument("--ann",  required=True, help="annotation txt you created")
    p.add_argument("--ckpt", default="ImagiNet Checkpoints/imaginet_selfcon_origin.pt")
    p.add_argument("--bs",   type=int, default=64, help="batch size")
    p.add_argument("--out",  default="results_origin.csv")
    return p.parse_args()

def main():
    args = get_args()

    # --- identical normalisation used in repo -------------
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

    # ---------- DATASET (track='origin') ------------------
    ds = ImagiNet(args.root, args.ann,
                  track="origin", train=False, test_aug=False,
                  transform=transform)                              # :contentReference[oaicite:1]{index=1}
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs,
                                     shuffle=False, num_workers=4)

    # --------------- MODEL -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConResNet(selfcon_pos=[False, True, False],
                      selfcon_size="fc", dataset="imaginet")        # :contentReference[oaicite:2]{index=2}
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval().to(device)

    # classifier head from checkpoint
    classifier = nn.Sequential(
        nn.Linear(2048, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 2)
    )
    classifier.load_state_dict(state["classifier"])
    classifier.eval().to(device)

    # --------------- INFERENCE LOOP ----------------------
    out_rows = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for imgs, _ in dl:                        # labels optional
            imgs = imgs.to(device)
            feats = model.encoder(imgs)[1]        # returns (backbone_out, pooled)
            logits = classifier(feats)
            probs = softmax(logits)[:, 1].cpu()   # prob of synthetic
            preds = (probs > 0.5).int()

            for path, p_fake, pred in zip(_, probs, preds):
                out_rows.append([path, f"{p_fake:.4f}", int(pred)])

    # --------------- SAVE CSV ----------------------------
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "prob_synthetic", "pred_label"])
        writer.writerows(out_rows)

    print(f"Saved {len(out_rows)} results to {args.out}")

if __name__ == "__main__":
    main()
