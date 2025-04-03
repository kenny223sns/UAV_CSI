
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
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=2.0,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="V")

# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=4,
                             num_cols=4,
                             vertical_spacing=2.0,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="cross")

# Create transmitter


tx = Transmitter(name="tx",
                 position=[100,120,30])
# Create a receiver
scene.add(tx)

rx = Receiver(name="rx",
                 position=[50,50,30])

# Add transmitter instance to scene
scene.add(rx)


 # Transmitter points towards receiver