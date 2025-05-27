#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained CSI-based localisation model.

Features
--------
* Supports plug-in backbone  (resnet | fusion).
* Reads scale factor saved in checkpoint for automatic de-normalisation.
* Prints p50 / p67 / p80 / p90 and mean error.
* Optionally saves (pred_x, pred_y, gt_x, gt_y, err) as CSV.
* Optionally saves CDF plot instead of / in addition to showing.
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import CSIDataset
from models.locnet import LocNet


# --------------------------------------------------------------------------- #
def compute_metrics(err: np.ndarray) -> dict[str, float]:
    """Return common localisation error statistics (in metres)."""
    return {
        "p50":  np.percentile(err, 50),
        "p67":  np.percentile(err, 67),
        "p80":  np.percentile(err, 80),
        "p90":  np.percentile(err, 90),
        "mean": err.mean(),
    }


def save_csv(path: Path, preds: np.ndarray, gts: np.ndarray, err: np.ndarray) -> None:
    """Save predictions / GT / error to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pred_x", "pred_y", "gt_x", "gt_y", "err_m"])
        w.writerows(np.hstack([preds, gts, err[:, None]]))
    print(f"[INFO] Predictions saved to {path.resolve()}")


def plot_cdf(err: np.ndarray, title: str, save_path: Path | None = None) -> None:
    """Plot (and optionally save) the CDF of the localisation error."""
    plt.figure(figsize=(5, 4))
    plt.hist(
        err,
        bins=200,
        density=True,
        cumulative=True,
        histtype="step",
        label="CDF",
    )
    plt.xlabel("Localisation error (m)")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] CDF figure saved to {save_path.resolve()}")
    else:
        plt.show()
    plt.close()


# --------------------------------------------------------------------------- #
def main(cfg):
    device = "cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu"
    print(f"[INFO] Using device: {device}")

    # ---------- Dataset ----------------------------------------------------- #
    ds = CSIDataset(
        cfg.data_root,
        Path(cfg.data_root) / "labels.csv",
        supervised_only=True,
    )
    print(f"[INFO] Loaded {len(ds)} labelled samples for evaluation.")
    dl = DataLoader(ds, batch_size=cfg.batch, shuffle=False, num_workers=cfg.workers)

    # ---------- Model ------------------------------------------------------- #
    net = LocNet(
        in_ch=cfg.in_ch,
        backbone=cfg.backbone,
        emb_dim=cfg.emb_dim,
        out_dim=cfg.out_dim,
    ).to(device)

    ckpt = torch.load(cfg.ckpt, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        net.load_state_dict(ckpt["model_state"])
        label_scale = ckpt.get("scale", 1.0)
    else:  # raw state_dict
        net.load_state_dict(ckpt)
        label_scale = 1.0
    net.eval()

    # ---------- Inference --------------------------------------------------- #
    preds, gts = [], []
    with torch.no_grad():
        for x, y in dl:
            preds.append(net(x.to(device)).cpu())
            gts.append(y)
    preds = torch.cat(preds).numpy() * label_scale
    gts = torch.cat(gts).numpy()

    # ---------- Metrics ----------------------------------------------------- #
    err = np.linalg.norm(preds - gts, axis=1)
    metr = compute_metrics(err)
    print("\n=====  Error statistics  =====")
    for k, v in metr.items():
        print(f"{k:<4}: {v:6.2f} m")

    # ---------- Save optional outputs -------------------------------------- #
    if cfg.save_csv:
        save_csv(Path(cfg.save_csv), preds, gts, err)
    if cfg.no_plot is False or cfg.save_fig:
        plot_cdf(
            err,
            title=f"{cfg.backbone}  (mean={metr['mean']:.2f} m)",
            save_path=Path(cfg.save_fig) if cfg.save_fig else None,
        )


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="dataset_three_sector", help="root folder")
    p.add_argument("--ckpt", required=True, help="checkpoint .pth")
    p.add_argument("--backbone", default="resnet", choices=["resnet", "fusion"])
    p.add_argument("--in_ch", type=int, default=4, help="# CSI channels")
    p.add_argument("--emb_dim", type=int, default=512, help="embedding dim (fusion)")
    p.add_argument("--out_dim", type=int, default=2, help="output dim (usually 2)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true", help="force CPU inference")
    p.add_argument("--save_csv", default="", help="path to save *.csv predictions")
    p.add_argument("--save_fig", default="", help="path to save CDF fig (png/pdf)")
    p.add_argument("--no_plot", action="store_true", help="do not pop up figure window")
    cfg = p.parse_args()
    main(cfg)
