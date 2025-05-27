#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script — baseline semi-supervised localisation with consistency loss
(Mean-Teacher framework + warm-up λ + EMA teacher).

Features
--------
* Plug-in backbone (resnet | fusion) with emb_dim configurable.
* Consistency loss weight λ_linearly warmed-up during first `cons_delay` epochs.
* Automatic resume from --resume_ckpt (keeps optimiser & scaler state).
* Checkpoint naming: latest.pth (always), best.pth (lowest val-mean-err).
* Optional validation split (ratio) for early stopping / model selection.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils import CSIDataset
from models.locnet import LocNet


# --------------------------------------------------------------------------- #
def update_teacher(student: torch.nn.Module, teacher: torch.nn.Module, alpha: float) -> None:
    with torch.no_grad():
        for p_t, p_s in zip(teacher.parameters(), student.parameters()):
            p_t.data.mul_(alpha).add_(p_s.data, alpha=1.0 - alpha)


def split_dataset(
    ds: CSIDataset, val_ratio: float, seed: int
) -> tuple[CSIDataset, CSIDataset]:
    if val_ratio <= 0.0:
        return ds, None
    val_len = math.ceil(len(ds) * val_ratio)
    train_len = len(ds) - val_len
    g = torch.Generator().manual_seed(seed)
    return random_split(ds, [train_len, val_len], generator=g)


def compute_val_error(
    net: torch.nn.Module, dl: DataLoader, device: str, scale: float
) -> float:
    net.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in dl:
            preds.append(net(x.to(device)).cpu())
            gts.append(y)
    if not preds:
        return float("inf")
    preds = torch.cat(preds).numpy() * scale
    gts = torch.cat(gts).numpy()
    err = np.linalg.norm(preds - gts, axis=1)
    return float(err.mean())


# --------------------------------------------------------------------------- #
def main(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu"
    print(f"[INFO] Device: {device}")

    # ------------------------- Datasets & loaders --------------------------- #
    sup_ds = CSIDataset(cfg.data_root, Path(cfg.data_root) / "labels.csv", supervised_only=True)
    all_ds = CSIDataset(
        cfg.data_root,
        Path(cfg.data_root) / "labels.csv",
        supervised_only=False,
        aug_flip_delay=True,
    )

    sup_ds_train, sup_ds_val = split_dataset(sup_ds, cfg.val_ratio, cfg.seed)
    val_loader = (
        DataLoader(sup_ds_val, batch_size=cfg.batch_val, shuffle=False, num_workers=cfg.workers)
        if sup_ds_val
        else None
    )

    sup_loader = DataLoader(
        sup_ds_train, batch_size=cfg.bs_sup, shuffle=True, drop_last=True, num_workers=cfg.workers
    )
    unsup_loader = DataLoader(
        all_ds, batch_size=cfg.bs_unsup, shuffle=True, drop_last=True, num_workers=cfg.workers
    )
    print(
        f"[INFO] Sup samples: {len(sup_ds_train)} (train)  {len(sup_ds_val) if sup_ds_val else 0} (val); "
        f"All samples (sup+unsup): {len(all_ds)}"
    )

    # ------------------------- Model & optimizer --------------------------- #
    net_s = LocNet(
        in_ch=cfg.in_ch, backbone=cfg.backbone, emb_dim=cfg.emb_dim, out_dim=cfg.out_dim
    ).to(device)
    net_t = LocNet(
        in_ch=cfg.in_ch, backbone=cfg.backbone, emb_dim=cfg.emb_dim, out_dim=cfg.out_dim
    ).to(device)
    net_t.load_state_dict(net_s.state_dict())
    net_t.eval()
    for p in net_t.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(net_s.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = GradScaler()

    start_epoch = 0
    best_val_err = float("inf")

    # ------------------------- (Optional) resume --------------------------- #
    if cfg.resume_ckpt and Path(cfg.resume_ckpt).is_file():
        ckpt = torch.load(cfg.resume_ckpt, map_location=device)
        net_s.load_state_dict(ckpt["model_state"])
        net_t.load_state_dict(ckpt["teacher_state"])
        opt.load_state_dict(ckpt["optim_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_err = ckpt.get("best_val_err", best_val_err)
        print(f"[INFO] Resumed from {cfg.resume_ckpt} (epoch {start_epoch})")

    # ------------------------- Training loop ------------------------------ #
    for epoch in range(start_epoch, cfg.epochs):
        net_s.train()
        lam = cfg.lam_cons * min(1.0, max(0, epoch - cfg.cons_delay + 1) / cfg.warmup_epochs)
        pbar = tqdm(
            zip(sup_loader, unsup_loader),
            total=min(len(sup_loader), len(unsup_loader)),
            desc=f"Epoch {epoch:03d} λ={lam:.3f}",
        )

        for (xs, ys), (xu, _) in pbar:
            xs, ys = xs.to(device), (ys.to(device) / cfg.label_scale)
            xu = xu.to(device)

            with autocast():
                # ----- supervised branch ----------------------------------- #
                y_pred = net_s(xs)
                sup_loss = F.mse_loss(y_pred, ys, reduction="mean")

                # ----- unsupervised (consistency) branch ------------------ #
                with torch.no_grad():
                    y_t = net_t(xu)
                y_s = net_s(xu)
                cons_loss = F.mse_loss(y_s, y_t, reduction="mean")

                loss = sup_loss + lam * cons_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            update_teacher(net_s, net_t, cfg.ema)

            pbar.set_postfix({"sup": sup_loss.item(), "cons": cons_loss.item(), "loss": loss.item()})

        # ------------------- Validation & checkpoint ---------------------- #
        if val_loader:
            val_err = compute_val_error(net_s, val_loader, device, cfg.label_scale)
            print(f"[VAL]  mean error = {val_err:.2f} m")
            is_best = val_err < best_val_err
            best_val_err = min(best_val_err, val_err)
        else:
            is_best = False

        ckpt_dict = {
            "epoch": epoch,
            "model_state": net_s.state_dict(),
            "teacher_state": net_t.state_dict(),
            "optim_state": opt.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scale": cfg.label_scale,
            "best_val_err": best_val_err,
        }
        torch.save(ckpt_dict, Path(cfg.out_dir) / "latest.pth")
        if is_best:
            torch.save(ckpt_dict, Path(cfg.out_dir) / "best.pth")

    print(f"[INFO] Training completed. Best val mean-err = {best_val_err:.2f} m")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="dataset_three_sector")
    # model options
    p.add_argument("--backbone", default="resnet", choices=["resnet", "fusion"])
    p.add_argument("--in_ch", type=int, default=4)
    p.add_argument("--emb_dim", type=int, default=512)
    p.add_argument("--out_dim", type=int, default=2)
    # training options
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs_sup", type=int, default=64)
    p.add_argument("--bs_unsup", type=int, default=64)
    p.add_argument("--batch_val", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    p.add_argument("--ema", type=float, default=0.995, help="EMA decay for teacher")
    p.add_argument("--lam_cons", type=float, default=0.5, help="max consistency weight")
    p.add_argument("--cons_delay", type=int, default=0, help="epochs to start consistency")
    p.add_argument("--warmup_epochs", type=int, default=5, help="λ warm-up length")
    p.add_argument("--label_scale", type=float, default=200.0, help="max distance for norm")
    p.add_argument("--val_ratio", type=float, default=0.1, help="supervised val split")
    # misc
    p.add_argument("--out_dir", default="checkpoints", help="where to save ckpts")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resume_ckpt", default="", help="path to resume checkpoint")
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    main(args)
