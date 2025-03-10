
import matplotlib.pyplot as plt
import numpy as np
import time

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, CoverageMap

# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, cir_to_time_channel,subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement

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

scene.frequency = 3.5e9 # in Hz; implicitly updates RadioMaterials

scene.synthetic_array = False # If set to False, ray tracing will be done per antenna element (slower for large arrays)
scene.remove("tx")
scene.remove("rx")
scene.remove("tx1")
# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=8,
                        num_cols=8,
                        vertical_spacing=2.0,
                        horizontal_spacing=0.5,
                        pattern="tr38901",
                        polarization="V")

# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=4,
                        num_cols=2,
                        vertical_spacing=2.0,
                        horizontal_spacing=0.5,
                        pattern="tr38901",
                        polarization="cross")

# Create transmitter


tx = Transmitter(name="tx",
            position=[500,220,30])
# Create a receiver
scene.add(tx)

rx = Receiver(name="rx",
            position=[50,50,30])

# Add transmitter instance to scene
scene.add(rx)


scene.remove("tx1")

tx1 = Transmitter(name="tx1",
                 position=[8.5,21,27])
# Add transmitter instance to scene
scene.add(tx1)

paths = scene.compute_paths(max_depth=5) 
# 取得 Tx 到 Rx 的 CIR
subcarrier_spacing = 30e3
fft_size = 408
print("Shape of `a` before applying Doppler shifts: ", paths.a.shape)

# Apply Doppler shifts
paths.apply_doppler(sampling_frequency=subcarrier_spacing,num_time_steps=10, tx_velocities=[3.,0,0],rx_velocities=[0,7.,0]) 
print("Shape of `a` after applying Doppler shifts: ", paths.a.shape)

a, tau = paths.cir()
print("Shape of tau: ", tau.shape)

frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing)


h_freq= cir_to_ofdm_channel(frequencies,
                             a,
                             tau,
                             normalize=False) # Non-normalized includes path-loss

# Verify that the channel power is normalized


print("Shape of h_freq: ", h_freq.shape)



#Rx VS subcarrier (angle delay)
import numpy as np
import matplotlib.pyplot as plt

def angle_delay_transform(h_freq):
    """
    - 先對 subcarriers (頻域) 做 IFFT -> 時延域 (Delay)
    - 再對 rx_antenna (空域) 做 FFT -> 角度域 (AOA)
    """
    # Step 1: IFFT over subcarriers (Convert to Delay domain)
    h_delay = np.fft.ifft(h_freq, axis=-1)

    # Step 2: FFT over rx_antenna (Convert to Angle domain)
    h_angle_delay = np.fft.fftshift(np.fft.fft(h_delay, axis=0), axes=0)

    return h_angle_delay

h_tx1 =h_freq.numpy()[0, 0, 0, 1, :, 0, :]   # Tx1 in No Jamming scenario

H = angle_delay_transform(h_tx1)  # 無干擾 Tx1
import matplotlib

csi_magnitude = np.abs(H)  
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# 設定 X, Y 軸
angles = np.arange(csi_magnitude.shape[0])   # 0~63 (Angle Index)
delays = np.arange(csi_magnitude.shape[1])   # 0~407 (Delay Index)
X, Y = np.meshgrid(delays, angles)
# 畫 3D 圖
surf = ax.plot_surface(X, Y, csi_magnitude, cmap='viridis')
cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
cbar.set_label("CSI Magnitude (dB)")
# 設定標籤
ax.set_xlabel("Delay Index")
ax.set_ylabel("Angle Index")
ax.set_zlabel("CSI Magnitude")

import numpy as np
import matplotlib.pyplot as plt

csi = np.abs(H)  # shape=(M, N)

# (1) Delay 索引轉成微秒
N = csi.shape[1]
subcarrier_spacing = 30e3
t_s = 1/(N*subcarrier_spacing)
delay_us = np.arange(N)*t_s*1e6

# (2) Angle 索引轉成度數 (假設 spacing=0.5 λ, M=64)
M = csi.shape[0]
d_ = 0.5  # 0.5 λ
k_array = np.arange(-M//2, M//2)
angle_deg = np.degrees(np.arcsin(k_array/(M*d_)))

# 建立網格
X, Y = np.meshgrid(delay_us, angle_deg)

# 繪圖 (3D 為例)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, csi, cmap='viridis')
fig.colorbar(surf, shrink=0.5, aspect=10, label="CSI Magnitude (linear)")

ax.set_xlabel("Delay (µs)")
ax.set_ylabel("Angle (deg)")
ax.set_zlabel("Magnitude")
plt.show()

plt.show()
plt.savefig("angle_delay_3d.png")
print("shape of csi_magnitude",csi_magnitude.shape)