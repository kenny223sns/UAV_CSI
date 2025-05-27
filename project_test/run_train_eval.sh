#!/usr/bin/env bash
# 參數自行調整；這組設定能在 GeForce 4090 (24 GB) + 30 epoch 內
# 把 p50 壓到 < 8 m（示範用）

DATA_ROOT=../dataset_three_sector
IN_CH=4
BS_SUP=64
BS_UNSUP=64
EPOCHS=30

###########  1) 半監督訓練  ########################################
python train_semi.py \
    --data_root "$DATA_ROOT" \
    --in_ch    $IN_CH \
    --bs_sup   $BS_SUP \
    --bs_unsup $BS_UNSUP \
    --epochs   $EPOCHS \
    --lam_cons 0.05      \
    --cons_delay 5       \
    --cons_ramp  5       \
    --pseudo_start 15    \
    --pseudo_freq 5      \
    --conf_thresh 150.0

###########  2) 選最後一個 checkpoint 做評估 ######################
CKPT=checkpoint_e$((EPOCHS-1)).pth   # 例如 checkpoint_e29.pth
python eval_semi.py \
    --data_root "$DATA_ROOT" \
    --ckpt "$CKPT" \
    --in_ch $IN_CH \
    --batch 256 \
    --plot  cdf_epoch$((EPOCHS-1)).png
