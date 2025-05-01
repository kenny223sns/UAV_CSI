"""
Full-band CFR demo – Python 3.11 · TF 2.19 · Sionna 1.0.2
"""

# --------------------------------------------------------------------#
# 0) Imports & GPU housekeeping
# --------------------------------------------------------------------#
import os, numpy as np, matplotlib.pyplot as plt, tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import sionna, sionna.rt
except ImportError:
    os.system("pip install -q --upgrade sionna")
    import sionna, sionna.rt

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera, PathSolver, subcarrier_frequencies


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import sys
    
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera, watt_to_dbm

from sionna.phy.channel import OFDMChannel, CIRDataset
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.phy.utils import ebnodb2no, PlotBER
from sionna.phy.ofdm import KBestDetector, LinearDetector
from sionna.phy.mimo import StreamManagement

# Import Sionna RT components
from sionna.rt import load_scene, Camera, Transmitter, Receiver, PlanarArray,\
                      PathSolver, RadioMapSolver
import random
"""
Full-band CFR demo – Python 3.11 · TF 2.19 · Sionna 1.0.2
"""

# ── 參數區 ───────────────────────────────────────────────────────────────
# 場景與天線
SCENE_NAME      = sionna.rt.scene.etoile
TX_ARRAY_CONFIG = dict(num_rows=1, num_cols=1,
                       vertical_spacing=0.5, horizontal_spacing=0.5,
                       pattern="iso", polarization="V")
RX_ARRAY_CONFIG = TX_ARRAY_CONFIG

# 發射機設定： (name, position, orientation, role)

# (name, pos, ori, role, power_dbm)
TX_LIST = [
  ("tx0",  [-100,-100,20], [np.pi*5/6,0,0],     "desired", 10),
  ("tx1",  [-100,  50,20], [np.pi/6,  0,0],     "desired", 10),
  ("tx2",  [ 100,-100,20], [-np.pi/2,0,0],      "desired", 10),
  ("jam1", [ 100,  50,20], [np.pi/2,  0,0],     "jammer",  100),
  ("jam2", [ 50,  50,20], [np.pi/2,  0,0],     "jammer",  100),
  ("jam3", [ -50, -50,20], [np.pi/2,  0,0],     "jammer",  100),

 
]



RX_CONFIG      = ("rx", [0,0,50])  # (name, position)

# PathSolver 參數
PATHSOLVER_ARGS = dict(max_depth=6,
                       los=True,
                       specular_reflection=True,
                       diffuse_reflection=False,
                       refraction=True,
                       synthetic_array=False,
                       seed=41)

# RadioMapSolver 參數
RMSOLVER_ARGS   = dict(max_depth=5,
                       cell_size=(1.,1.),
                       samples_per_tx=10**7)

# OFDM / QPSK 參數
N_SYMBOLS       = 1
N_SUBCARRIERS   = 1024
SUBCARRIER_SPACING = 30e3  # Hz
num_ofdm_symbols = 1024 
num_subcarriers = 1024
subcarrier_spacing = 30e3

# 通道品質參數
JNR_dB          = 5.0
EBN0_dB         = 20.0

# 繪圖範圍（SINR dB）
SINR_VMIN       = -40
SINR_VMAX       =   0

# ── 程式區 ───────────────────────────────────────────────────────────────
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sionna, sionna.rt
from sionna.rt import (load_scene, PlanarArray, Transmitter, Receiver,
                       PathSolver, RadioMapSolver, subcarrier_frequencies)

# GPU 設定
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.list_physical_devices("GPU")
if gpus: tf.config.experimental.set_memory_growth(gpus[0], True)

# 1) 建立場景與天線配置
scene = load_scene(SCENE_NAME)
scene.tx_array = PlanarArray(**TX_ARRAY_CONFIG)
scene.rx_array = PlanarArray(**RX_ARRAY_CONFIG)
for tx_name in scene.transmitters.copy():  
    scene.remove(tx_name)
# 再把所有 receiver name 拿出來，一個個 remove
for rx_name in scene.receivers.copy():
    scene.remove(rx_name)

# 確認都清空了
assert len(scene.transmitters)==0 and len(scene.receivers)==0

# 2) 新增 Tx (含 role 標籤)
def add_tx(scene, name, pos, ori, role, power_dbm):
    tx = Transmitter(name=name, position=pos,
                     orientation=ori, power_dbm=power_dbm)
    tx.role = role
    scene.add(tx)
    return tx

# 迴圈時 unpack 五個欄位
for name, pos, ori, role, p_dbm in TX_LIST:
    add_tx(scene, name, pos, ori, role, p_dbm)


# 3) 新增 Rx
rx_name, rx_pos = RX_CONFIG
rx = Receiver(name=rx_name, position=rx_pos)
scene.add(rx)

# 4) 自動分組 indices
tx_names = scene.transmitters
all_txs   = [scene.get(n) for n in tx_names]
idx_des   = [i for i,tx in enumerate(all_txs) if getattr(tx,'role',None)=='desired']
idx_jam   = [i for i,tx in enumerate(all_txs) if getattr(tx,'role',None)=='jammer']

# 5) RadioMap 計算
rm_solver = RadioMapSolver()
rm = rm_solver(scene, **RMSOLVER_ARGS)

# 6) PathSolver 函式
solver = PathSolver()
def solve():
    return solver(scene, **PATHSOLVER_ARGS)

# 7) 計算 CFR
freqs = subcarrier_frequencies(N_SUBCARRIERS, SUBCARRIER_SPACING)
for name in scene.transmitters:
    scene.get(name).velocity = [30, 0, 0]   # 或者 jam1 用 [-30,0,0]
paths = solve()

def dbm2w(dbm):
    return 10**(dbm/10) / 1000

tx_powers = [ dbm2w(scene.get(n).power_dbm) 
              for n in scene.transmitters ]

# H     = paths.cfr(frequencies=freqs,
#                   normalize=False,
#                   normalize_delays=True,
#                   out_type="numpy").squeeze()


ofdm_symbol_duration = 1/subcarrier_spacing
delay_resolution = ofdm_symbol_duration/num_subcarriers
doppler_resolution = subcarrier_spacing/num_ofdm_symbols
# H= paths.cfr(frequencies=freqs,
#                   sampling_frequency    = 1/ofdm_symbol_duration,
#                   num_time_steps        = num_ofdm_symbols,
#                   normalize_delays      = False,
#                   normalize             = False,
#                   out_type              = "numpy").squeeze()
# 讓所有 desired/jammer 都動一點，不一定要同方向

H_unit = paths.cfr(
    frequencies         = freqs,
    sampling_frequency  = 1/ofdm_symbol_duration,
    num_time_steps      = num_ofdm_symbols,   # ← 讓 Sionna 跑時間演變 (多普勒)
    normalize_delays    = True,
    normalize           = False,
    out_type            = "numpy"
    ).squeeze()           # shape: (num_tx, T, F)
# h_main = np.sum(H[idx_des, :], axis=0)
# h_intf = np.sum(H[idx_jam, :], axis=0)
print("H_unit.shape", H_unit.shape)

H_all = np.sqrt(np.array(tx_powers)[:,None,None]) * H_unit

H_des = H_all[idx_des].sum(axis=0)   # (T, F)
H_jam = H_all[idx_jam].sum(axis=0)   # (T, F)
print("H_des.shape", H_des.shape)
print("H_jam.shape", H_jam.shape)
H = H_unit[:,0,:]
print("H.shape", H.shape)






h_main = sum( np.sqrt(tx_powers[i]) * H[i] 
                for i in idx_des )
h_intf = sum( np.sqrt(tx_powers[i]) * H[i] 
                for i in idx_jam )
print("h_main.shape", h_main.shape)
print("h_intf.shape", h_intf.shape)


# 8) 產生 QPSK+OFDM 符號
bits       = np.random.randint(0,2,(N_SYMBOLS, N_SUBCARRIERS, 2))
bits_jam       = np.random.randint(0,2,(N_SYMBOLS, N_SUBCARRIERS, 2))
X_qpsk     = (1-2*bits[...,0] + 1j*(1-2*bits[...,1]))/np.sqrt(2)
X_qpsk_jam     = (1-2*bits_jam[...,0] + 1j*(1-2*bits_jam[...,1]))/np.sqrt(2)

Y_sig      = X_qpsk * h_main[np.newaxis,:]
Y_int      = X_qpsk_jam * h_intf[np.newaxis,:]
p_sig      = np.mean(np.abs(Y_sig)**2)
p_int      = np.mean(np.abs(Y_int)**2)
# scale      = np.sqrt(p_sig/p_int/10**(JNR_dB/10)) if p_int>0 else 0
# Y_int     *= scale
N0         = p_sig/(10**(EBN0_dB/10)*2)
noise      = np.sqrt(N0/2)*(np.random.randn(*Y_sig.shape)+1j*np.random.randn(*Y_sig.shape))
Y_tot      = Y_sig + Y_int + noise
y_eq_no_i  = (Y_sig + noise)   / h_main
y_eq_with_i= (Y_sig + Y_int + noise) / h_main
print("Y_sig.shape", Y_sig.shape)
print("Y_int.shape", Y_int.shape)
print("y_eq_no_i.shape", y_eq_no_i.shape)
print("y_eq_with_i.shape", y_eq_with_i.shape)


# 9) 繪製星座 & CFR
fig,ax=plt.subplots(1,3,figsize=(15,4))
ax[0].scatter(y_eq_no_i.real, y_eq_no_i.imag, s=4, alpha=.25)
ax[1].scatter(y_eq_with_i.real, y_eq_with_i.imag, s=4, alpha=.25)
ax[0].set(title="No interference"); ax[0].grid(True)
ax[1].set(title="With interferer "); ax[1].grid(True)
ax[2].plot(np.abs(h_main), label="|H_main|")
ax[2].plot(np.abs(h_intf), label="|H_intf|")
ax[2].set(title="CFR Magnitude", xlabel="Subcarrier Index"); ax[2].legend(); ax[2].grid(True)
plt.tight_layout(); plt.show()





# 10) 計算並繪製 SINR Map
cc        = rm.cell_centers.numpy()
x_unique  = cc[0,:,0]; y_unique = cc[:,0,1]
rss_list  = [rm.rss[i].numpy() for i in range(len(all_txs))]
N0_map    = 1e-12
rss_des   = sum(rss_list[i] for i in idx_des)
rss_jam   = sum(rss_list[i] for i in idx_jam)
sinr_db   = 10*np.log10(np.clip(rss_des/(rss_des+rss_jam+N0_map),1e-12,None))

X,Y = np.meshgrid(x_unique,y_unique)
plt.figure(figsize=(7,5))
pcm = plt.pcolormesh(X, Y, sinr_db, shading='nearest',
                     vmin=SINR_VMIN, vmax=SINR_VMAX)
plt.colorbar(pcm, label="SINR (dB)")
plt.scatter([t.position[0] for t in all_txs if t.role=='desired'],
            [t.position[1] for t in all_txs if t.role=='desired'],
            c='g', marker='^', s=100, label='Tx')
plt.scatter([t.position[0] for t in all_txs if t.role=='jammer'],
            [t.position[1] for t in all_txs if t.role=='jammer'],
            c='red', marker='x', s=100, label='Jam')
plt.scatter(rx.position[0], rx.position[1],
            c='red', marker='o', s=50, label='Rx')
plt.legend(); plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.title("SINR Map"); plt.tight_layout(); plt.show()
H_unit = paths.cfr(
    frequencies         = freqs,
    sampling_frequency  = 1/ofdm_symbol_duration,
    num_time_steps      = num_ofdm_symbols,   # ← 讓 Sionna 跑時間演變 (多普勒)
    normalize_delays    = True,
    normalize           = False,
    out_type            = "numpy"
    ).squeeze()           # shape: (num_tx, T, F)
# h_main = np.sum(H[idx_des, :], axis=0)
# h_intf = np.sum(H[idx_jam, :], axis=0)
print("H_unit.shape", H_unit.shape)

print(idx_des)
print(idx_jam)
H_all = H_unit.sum(axis=0)
H_des = H_unit[idx_des].sum(axis=0)   # (T, F)
H_jam = H_unit[idx_jam].sum(axis=0)   # (T, F)


def to_delay_doppler(H_tf):
    Hf      = np.fft.fftshift(H_tf, axes=1)            # F shift
    h_delay = np.fft.ifft(Hf, axis=1 , norm="ortho")   # F→delay
    h_dd    = np.fft.fft(h_delay, axis=0 , norm="ortho")# t→doppler
    h_dd    = np.fft.fftshift(h_dd, axes=0)            # doppler shift
    return h_dd

Hdd_all = to_delay_doppler(H_all)
Hdd_des = to_delay_doppler(H_des)
Hdd_jam = to_delay_doppler(H_jam)
print("Hdd_des.shape", Hdd_des.shape)
print("Hdd_jam.shape", Hdd_jam.shape)
print("Hdd_all.shape", Hdd_all.shape)
T, F = Hdd_des.shape                # =1024,1024
offset = 20

# Delay：真正有多徑的前 40 bins
d_start, d_end = 0, offset*2        # [0:40]

# Doppler：fftshift 之後的 0 Hz 在 row=T//2
t_mid = T//2                        # 512
t_start, t_end = t_mid-offset, t_mid+offset  # [492:532]

# 座標軸
delay_bins   = np.arange(F) * ((1/subcarrier_spacing)/F)*1e9  # ns
doppler_bins = np.fft.fftshift(np.fft.fftfreq(T, d=1/subcarrier_spacing))

X, Y = np.meshgrid(delay_bins[d_start:d_end],
                   doppler_bins[t_start:t_end])

Z_des = np.abs(Hdd_des[t_start:t_end, d_start:d_end])
Z_jam = np.abs(Hdd_jam[t_start:t_end, d_start:d_end])
Z_all = np.abs(Hdd_all[t_start:t_end, d_start:d_end])
fig = plt.figure(figsize=(18,5))

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z_des, cmap='viridis', edgecolor='none')
ax1.set(title='Delay–Doppler |Desired|', xlabel='Delay (ns)', ylabel='Doppler (Hz)')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z_jam, cmap='viridis', edgecolor='none')
ax2.set(title='Delay–Doppler |Jammer|', xlabel='Delay (ns)')

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, Z_all, cmap='viridis', edgecolor='none')
ax3.set(title='Delay–Doppler |All|', xlabel='Delay (ns)')

plt.tight_layout()
plt.show()



import tensorflow as tf
from sionna.phy.utils import compute_ber            # 已在最前面 import

# 將硬判決後的比特包成 Tensor
b_hat_no_i_tf   = tf.constant(b_hat_no_i.reshape(-1))
b_hat_with_i_tf = tf.constant(b_hat_with_i.reshape(-1))
# ❷ 與原始 bits 比對並計算 BER
bits_ref = bits.reshape(-1)  
b_ref_tf        = tf.constant(bits_ref)

ber_no_i_tf   = compute_ber(b_ref_tf, b_hat_no_i_tf)
ber_with_i_tf = compute_ber(b_ref_tf, b_hat_with_i_tf)

print("BER (No-I)  :", ber_no_i_tf.numpy())
print("BER (With-I):", ber_with_i_tf.numpy())

