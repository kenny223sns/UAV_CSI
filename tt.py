import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Sionna / Sionna RT Imports
from sionna.rt import Scene, Transmitter, Receiver, PlanarArray, load_scene
from sionna.channel import (
    cir_to_ofdm_channel,
    subcarrier_frequencies,
)
from sionna.channel import OFDMChannel
# -----------------------------------------------------
# 1. 建立場景與 Transmitter / Receiver
# -----------------------------------------------------

# 你可以改為任何內建或自訂的場景，如 sionna.rt.scene.munich 或自建場景
scene = load_scene("outdoor_open")  # 範例名稱，請視實際情況變更
scene.frequency = 3.5e9
scene.synthetic_array = True  # 每個 Tx/Rx 使用面天線陣列 (快速)

# 設定天線陣列參數（可視需求自行調整）
scene.tx_array = PlanarArray(num_rows=4, num_cols=4,
                             vertical_spacing=0.5, horizontal_spacing=0.5,
                             pattern="iso", polarization="V")
scene.rx_array = PlanarArray(num_rows=2, num_cols=2,
                             vertical_spacing=0.5, horizontal_spacing=0.5,
                             pattern="iso", polarization="cross")

# 建立「主信號」的 Transmitter
tx_main = Transmitter(
    name="tx_main",
    position=[0.0, 0.0, 10.0],    # 主信號位置 (可自行修改)
    orientation=[0.0, 0.0, 0.0],
    tx_power=0.0  # dBW，可根據想要的功率設定
)

# 建立「干擾源」的 Transmitter
tx_intf = Transmitter(
    name="tx_intf",
    position=[50.0, 0.0, 10.0],   # 干擾源位置 (可自行修改)
    orientation=[0.0, 0.0, 0.0],
    tx_power=-5.0  # dBW，假設干擾功率小於主信號，也可自行調整
)

# 建立 Receiver
rx = Receiver(
    name="rx",
    position=[30.0, 0.0, 1.5],   # 接收端位置
    orientation=[0.0, 0.0, 0.0]
)

# 加入物件到場景
scene.add(tx_main)
scene.add(tx_intf)
scene.add(rx)

# -----------------------------------------------------
# 2. 計算射線追蹤路徑
# -----------------------------------------------------
paths = scene.compute_paths(
    max_depth=3,      # 反射/繞射/散射深度，可自行調整
    num_samples=1e6,  # 光線數量
)

# -----------------------------------------------------
# 3. 分離「主信號路徑」與「干擾路徑」
# -----------------------------------------------------
# 假設 paths 內部根據 tx_name 或其他標籤區分來源
# 以 pseudo-code 方式示範：實際可用 filter(...) 或布林索引
paths_main = paths.filter(tx_name="tx_main")
paths_intf = paths.filter(tx_name="tx_intf")

# 取得 CIR (複數增益 a、延遲 tau)
a_main, tau_main = paths_main.cir()
a_intf, tau_intf = paths_intf.cir()

# -----------------------------------------------------
# 4. 設置 OFDM 參數並轉為頻域通道
# -----------------------------------------------------
subcarrier_spacing = 30e3
fft_size = 128  # 依系統需求調整

# 計算各子載波頻率
freqs = subcarrier_frequencies(fft_size, subcarrier_spacing)

# 轉為頻域通道
h_freq_main = cir_to_ofdm_channel(
    frequencies=freqs,
    a=a_main,
    tau=tau_main,
    normalize=True
)
h_freq_intf = cir_to_ofdm_channel(
    frequencies=freqs,
    a=a_intf,
    tau=tau_intf,
    normalize=True
)

# （選用）合成後的通道，可視為 h_main + h_intf
# 以表示「主信號 + 干擾」對接收天線的總通道
h_freq_combined = h_freq_main + h_freq_intf

# -----------------------------------------------------
# 5. 在角度-延遲域分析（2D FFT：空域×頻域 → 角度×延遲）
# -----------------------------------------------------
# h_freq_x 的 shape 通常是 (batch, rx_array, tx_array, num_ofdm_symbols, num_subcarriers)
# 這裡示範取某個 batch, rx_port, tx_port, ofdm_symbol，視實際形狀做索引
h_main_np = h_freq_main.numpy()[0, 0, :, 0, :]  # (TxAnt, Subcarrier)
h_intf_np = h_freq_intf.numpy()[0, 0, :, 0, :]
h_comb_np = h_freq_combined.numpy()[0, 0, :, 0, :]

def angle_delay_transform(h_2d):
    """
    對「天線×子載波」的 2D 信道響應做 FFT，取得角度-延遲域。
    回傳 fft_shift 後的幅度，以利可視化。
    """
    h_fft = np.fft.fft2(h_2d, axes=(0, 1))        # 對(天線,子載波)做2D-FFT
    h_fft = np.fft.fftshift(h_fft, axes=(0, 1))   # 方便讓(0,0)在中心
    return np.abs(h_fft)

csi_main_ad = angle_delay_transform(h_main_np)
csi_intf_ad = angle_delay_transform(h_intf_np)
csi_comb_ad = angle_delay_transform(h_comb_np)

# -----------------------------------------------------
# 6. 繪圖比較：主信號 / 干擾 / 合成
# -----------------------------------------------------
plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.title("Main Signal in Angle-Delay Domain")
plt.imshow(csi_main_ad, aspect='auto', cmap='jet')
plt.colorbar(label="Amplitude")
plt.xlabel("Delay axis")
plt.ylabel("Angle axis")

plt.subplot(1,3,2)
plt.title("Interference Signal in Angle-Delay Domain")
plt.imshow(csi_intf_ad, aspect='auto', cmap='jet')
plt.colorbar(label="Amplitude")
plt.xlabel("Delay axis")
plt.ylabel("Angle axis")

plt.subplot(1,3,3)
plt.title("Combined (Main + Intf)")
plt.imshow(csi_comb_ad, aspect='auto', cmap='jet')
plt.colorbar(label="Amplitude")
plt.xlabel("Delay axis")
plt.ylabel("Angle axis")

plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 7. (選用) 與端到端模組串接
# -----------------------------------------------------
# 上述只示範「角度-延遲域觀察」；若要進行 EVM/BER 等端到端評估，
# 可使用 Sionna 的 PUSCHTransmitter, PUSCHReceiver, OFDMChannel 等類別，
# 把 h_freq_combined 實際套用到訊號傳輸流程，最後量化 BER/EVM。
#
# 例如:
#
#   pusch_tx = PUSCHTransmitter(...)
#   pusch_rx = PUSCHReceiver(...)
#   ofdm_channel = OFDMChannel(...)
#   # 送波形 -> 加入 h_freq_combined -> 接收 -> 解調 -> 取得 BER/EVM
#
# 依官方範例及文件進行細部參數配置即可。
