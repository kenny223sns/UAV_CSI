import matplotlib.pyplot as plt
import numpy as np
import time

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, cir_to_time_channel,subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement
import tensorflow as tf
#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import time
import sionna
# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Antenna

# For link-level simulations
from sionna.channel import OFDMChannel

from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement
scene = load_scene(sionna.rt.scene.munich) # Try also sionna.rt.scene.etoile

# 先刪除舊的接收機，避免名稱衝突
# for i in range(1, 65):  # 假設舊的接收機是 rx1, rx2, ..., rx64
#     scene.remove(f"rx{i}")
    
# num_rx = 64  # 設定接收機數量
# # 生成隨機接收機位置，確保 z 在 1 到 10 之間
# rx_positions = np.random.uniform([[0, 0, 1]], [[100, 100, 10]], size=(num_rx, 3))

# 創建並添加接收機
# for i, pos in enumerate(rx_positions):
#     rx = Receiver(name=f"rx{i+1}", position=pos.tolist(), orientation=[0, 0, 0])
#     scene.add(rx)

# print(f"已添加 {num_rx} 個隨機接收機到場景中")
scene.remove("tx")
scene.remove("rx")
scene.remove("rx1")
# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=8,
                             num_cols=8,
                             vertical_spacing=2.0,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="V")

# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=2,
                             num_cols=4,
                             vertical_spacing=2.0,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="cross")

# Create transmitter
tx = Transmitter(name="tx",
                 position=[8.5,21,27])

# Add transmitter instance to scene
scene.add(tx)

# Create a receiver
rx = Receiver(name="rx",
              position=[45,90,1.5],
              orientation=[0,0,0])

# Add receiver instance to scene
scene.add(rx)

scene.frequency = 3.5e9 # in Hz; implicitly updates RadioMaterials
scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)

# Compute propagation paths
paths = scene.compute_paths(max_depth=2,
                            num_samples=1e6)

# Default parameters in the PUSCHConfig
subcarrier_spacing = 30e3
fft_size = 408

print("Shape of `a` before applying Doppler shifts: ", paths.a.shape)

# Apply Doppler shifts
paths.apply_doppler(sampling_frequency=subcarrier_spacing, # Set to 15e3 Hz
                    num_time_steps=14, # Number of OFDM symbols
                    tx_velocities=[3.,0,0], # We can set additional tx speeds
                    rx_velocities=[0,7.,0]) # Or rx speeds

print("Shape of `a` after applying Doppler shifts: ", paths.a.shape)

a, tau = paths.cir()
print("Shape of tau: ", tau.shape)

t = tau[0,0,0,:]/1e-9 # Scale to ns
a_abs = np.abs(a)[0,0,0,0,0,:,0]
a_max = np.max(a_abs)
# Add dummy entry at start/end for nicer figure
t = np.concatenate([(0.,), t, (np.max(t)*1.1,)])
a_abs = np.concatenate([(np.nan,), a_abs, (np.nan,)])

# And plot the CIR
plt.figure()
plt.title("Channel impulse response realization")

plt.stem(t, a_abs)
plt.xlim([0, np.max(t)])
plt.ylim([-2e-6, a_max*1.1])
plt.xlabel(r"$\tau$ [ns]")
plt.ylabel(r"$|a|$")

# Disable normalization of delays
paths.normalize_delays = False





# Compute frequencies of subcarriers and center around carrier frequency
frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing)

# Compute the frequency response of the channel at frequencies.
h_freq = cir_to_ofdm_channel(frequencies,
                             a,
                             tau,
                             normalize=True) # Non-normalized includes path-loss

# Verify that the channel power is normalized

print("Shape of h_freq: ", h_freq.shape)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# h_freq shape: (1, 1, 2, 1, 64, 14, 408)
# 選擇某個 OFDM symbol (例如 0)
csi_complex = h_freq.numpy()[0, 0, 0, 0, :, 0, :]  # Shape: (64, 408) (Tx Antennas, Subcarriers)

##=====================================================================
##=====================================================================
##展示「空間-頻率域 CSI」
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 從 h_freq 中提取空間-頻率域 CSI (形狀: 64 (天線) x 408 (子載波))
csi_complex = h_freq.numpy()[0, 0, 0, 0, :, 0, :]  # Shape: (64, 408)

# 1. 空間-頻率域 CSI 的幅度與網格準備
csi_magnitude = np.abs(csi_complex)
num_antennas = csi_magnitude.shape[0]
num_subcarriers = csi_magnitude.shape[1]
antennas = np.arange(num_antennas)
subcarriers = np.arange(num_subcarriers)
X, Y = np.meshgrid(subcarriers, antennas)

# 2. 進行 2D FFT 得到角度-延遲域 CSI (先計算一次 FFT2，再做 fftshift)
csi_fft = np.fft.fft2(csi_complex, axes=(-2, -1))
csi_fft = np.fft.fftshift(csi_fft, axes=(-2, -1))
csi_xx = np.abs(csi_fft)

# 為了保證左右兩個子圖都在同一個 figure 中，這裡只建立一個 figure 並分成 1 行 2 列
fig = plt.figure(figsize=(16, 6))

# 左側子圖：空間-頻率域 CSI 3D 曲面圖
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, Y, csi_magnitude, cmap='viridis')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, label="Magnitude")
ax1.set_xlabel("Subcarrier Index")
ax1.set_ylabel("Antenna Index")
ax1.set_zlabel("CSI Magnitude")
ax1.set_title("Space-Frequency Domain CSI (3D Surface)")

# 右側子圖：角度-延遲域 CSI 3D 曲面圖
# 對應的網格依然可以用相同維度 (64, 408)
ax2 = fig.add_subplot(122, projection='3d')
X2, Y2 = np.meshgrid(np.arange(csi_xx.shape[1]), np.arange(csi_xx.shape[0]))
surf2 = ax2.plot_surface(X2, Y2, csi_xx, cmap='viridis')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5, label="Magnitude")
ax2.set_xlabel("Delay (τ)")
ax2.set_ylabel("Angle (θ)")
ax2.set_zlabel("CSI Magnitude")
ax2.set_title("Angle-Delay Domain CSI (3D Surface)")

plt.tight_layout()
plt.show()
