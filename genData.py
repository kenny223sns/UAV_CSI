# -------------------------------------
#  Generate Three-Sector CSI Dataset
# -------------------------------------
import os, math, csv, random, numpy as np
import sionna, tensorflow as tf    
import matplotlib.pyplot as plt
from sionna.rt import (load_scene, PlanarArray, Transmitter, Receiver,
                       PathSolver, subcarrier_frequencies)

# ========== 0. 基本參數 =====================================================
OUT_DIR       = "dataset_three_sector_0519"          # 影像輸出資料夾
N_PER_SECTOR  = [21745, 6764, 12308]            # 與論文相同樣本數
CARRIER_FREQ  = 3.5e9
BANDWIDTH     = 100e6
DELTA_F       = 30e3
N_SC          = int(BANDWIDTH/DELTA_F)          # 3333
SUBSAMPLE     = 4                               # 2-comb → 每 4 RE 取 1
FINAL_RE      = 408
SNR_DB_RANGE  = (23, 27)                        # 隨機 SNR ≈ 25±2 dB
TA_MAX_BIN    = 5                               # ±5*32ns

# ========== 1. 場景 & 天線 ===================================================
scene = load_scene(sionna.rt.scene.etoile)
scene.tx_array = PlanarArray(num_rows=4, num_cols=8,
                             horizontal_spacing=0.5,
                             vertical_spacing=2.0,
                             pattern="tr38901", polarization="cross")
scene.rx_array = PlanarArray(num_rows=1, num_cols=2,
                             horizontal_spacing=0.5, vertical_spacing=0.5,
                             pattern="tr38901", polarization="cross")
scene.remove("bs"); scene.remove("ue")
scene.add(Transmitter("bs", [0, 0, 30]))
scene.add(Receiver   ("ue", [0, 0, 30]))

freqs = subcarrier_frequencies(N_SC, DELTA_F)
p_solver = PathSolver()

# ========== 2. util: TA + AWGN =============================================
def add_ta_and_awgn(H, ta_max_bin=5, snr_db=25):
    """H:(2,32,408) numpy complex64"""
    C,Hant,W = H.shape
    # --- TA -------------------------------------------------
    k   = np.arange(W)
    dly = np.random.randint(-ta_max_bin, ta_max_bin+1)
    H   = H * np.exp(-1j*2*np.pi*dly*k/W)
    # --- AWGN ----------------------------------------------
    p_sig = np.mean(np.abs(H)**2)
    sigma = math.sqrt(p_sig / (10**(snr_db/10)))
    noise = sigma/np.sqrt(2) * (np.random.randn(*H.shape)+1j*np.random.randn(*H.shape))
    return H + noise

# ========== 3. 角-延遲轉換 ================================================
def to_angle_delay(csi):          # (32,408)  → (32,408)
    h_tau = np.fft.ifft(csi, axis=1, norm="ortho")
    return np.fft.fftshift(np.fft.fft(h_tau, axis=0), 0)

# ========== 4. 生成並保存 ===================================================
os.makedirs(OUT_DIR, exist_ok=True)
label_f = open(os.path.join(OUT_DIR, "labels.csv"), "w", newline="")
csv_w   = csv.writer(label_f)
csv_w.writerow(["file","sector","x","y","los"])

sector_center_deg = [0, 120, 240]

for s, N in enumerate(N_PER_SECTOR):
    sec_dir = os.path.join(OUT_DIR, f"sector{s}")
    os.makedirs(sec_dir, exist_ok=True)
    center_deg = sector_center_deg[s]
    for idx in range(N):
        # ----- UE 隨機位置 (極座標) ---------------------------
        r   = random.uniform(5, 200)       # 5–200 m
        ang = math.radians(center_deg + random.uniform(-60, 60))
        x   = r*math.cos(ang)
        y   = r*math.sin(ang)
        scene.get("ue").position = [x, y, 30]

        # ----- ray tracing ----------------------------------
        paths = p_solver(scene=scene, max_depth=4, los=True,
                         specular_reflection=True, diffuse_reflection=False,
                         synthetic_array=True)
       


       
        # ----- CSI pipeline ---------------------------------
        H = paths.cfr(freqs, normalize=False, normalize_delays=False,
                      out_type="numpy").squeeze()         
        H = H[..., ::SUBSAMPLE]                          
        mid = (H.shape[-1]-FINAL_RE)//2
        H = H[..., mid:mid+FINAL_RE]                     

        # snr_db = random.uniform(*SNR_DB_RANGE)
        # H = add_ta_and_awgn(H, TA_MAX_BIN, snr_db)        

        H_ad = np.stack([to_angle_delay(H[p]) for p in range(2)], 0)  
        mag   = np.abs(H_ad)
        phase = np.angle(H_ad)
        tensor = np.concatenate([mag, phase], axis=0).astype(np.float32) 

        # ----- save -----------------------------------------
        f_name = f"{s}_{idx:05d}.npy"
        np.save(os.path.join(sec_dir, f_name), tensor)
        csv_w.writerow([f_name, s, f"{x:.2f}", f"{y:.2f}"])

        if (idx+1) % 1000 == 0 or idx == N-1:
            print(f"Sector {s} 生成 {idx+1}/{N}")

label_f.close()
print("Finished. 根目錄:", OUT_DIR)