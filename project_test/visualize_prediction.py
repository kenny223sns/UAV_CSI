#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize CSI tensor (angle-delay) with prediction vs ground truth
"""
import matplotlib
matplotlib.use("Agg")  # 無頭模式

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import CSIDataset
from models.locnet import LocNet


def visualize_sample(
    x: torch.Tensor, pred: np.ndarray, gt: np.ndarray, title: str = ""
):
    """Visualize CSI angle-delay map + predicted vs GT position."""
    x_np = x.numpy()  # shape: (C, H, W)

    # --- plot CSI image (e.g., channel 0: mag of Tx0) ------------------
    csi_img = x_np[0]  # shape (H, W)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(csi_img, aspect="auto", origin="lower", cmap="turbo")
    plt.colorbar(label="magnitude")
    plt.title("CSI Angle–Delay Map (mag, ch 0)")
    plt.xlabel("Delay bins")
    plt.ylabel("Angle bins")

    # --- plot predicted vs ground truth position ------------------------
    plt.subplot(1, 2, 2)
    plt.scatter(gt[0], gt[1], color="green", label="Ground Truth", marker="o")
    plt.scatter(pred[0], pred[1], color="red", label="Prediction", marker="x")
    plt.legend()
    plt.grid(True)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"Predicted vs GT\n{title}")

    plt.tight_layout()
    plt.savefig("vis_sample.png", dpi=300)
    print("Saved to vis_sample.png")



def main(cfg):
    device = "cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu"

    # --- Load dataset
    ds = CSIDataset(cfg.data_root, f"{cfg.data_root}/labels.csv", supervised_only=True)
    print(f"[INFO] Loaded {len(ds)} labelled samples.")

    x, y = ds[cfg.index]
    x = x.unsqueeze(0).to(device)

    # --- Load model
    net = LocNet(in_ch=cfg.in_ch, backbone=cfg.backbone, emb_dim=cfg.emb_dim).to(device)
    ckpt = torch.load(cfg.ckpt, map_location=device)
    if "model_state" in ckpt:
        net.load_state_dict(ckpt["model_state"])
        scale = ckpt.get("scale", 1.0)
    else:
        net.load_state_dict(ckpt)
        scale = 1.0
    net.eval()

    with torch.no_grad():
        pred = net(x).cpu().numpy()[0] * scale

    gt = y.numpy()
    visualize_sample(x.cpu()[0], pred, gt, f"Sample #{cfg.index}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="dataset_three_sector")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--backbone", default="fusion", choices=["fusion", "resnet"])
    parser.add_argument("--in_ch", type=int, default=4)
    parser.add_argument("--emb_dim", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    cfg = parser.parse_args()
    main(cfg)
