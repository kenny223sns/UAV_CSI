#!/usr/bin/env python3
"""
Interference-Aware UAV Swarm – channel & map generator
-----------------------------------------------------
* Python 3.11 · TensorFlow 2.19 · Sionna RT 1.0.2
* Generates paths, CFR, delay-Doppler spectra and SINR radio-map for
  multiple desired transmitters and jammers.
* All parameters are centralised in the CONFIG dict → easy to tweak.
* No Jupyter-only magics; safe to run as script **or** import as module.
* Heavy calls (PathSolver, RadioMapSolver) are wrapped in functions so they
  execute exactly once per run.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import sionna
import sionna.rt as rt
from sionna.rt import (
    load_scene, PlanarArray, Transmitter, Receiver,
    PathSolver, RadioMapSolver, subcarrier_frequencies,
)

# ──────────────────────────── GPU & TF housekeeping ───────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
GPUS = tf.config.list_physical_devices("GPU")
if GPUS:
    tf.config.experimental.set_memory_growth(GPUS[0], True)

# ───────────────────────────────────── CONFIG ──────────────────────────────────
CONFIG: Dict[str, Any] = {
    # Scene & arrays
    "scene":               sionna.rt.scene.etoile,          # or rt.scene.etoile / munich …
    "tx_array":            dict(num_rows=1, num_cols=1,
                                 vertical_spacing=.5, horizontal_spacing=.5,
                                 pattern="iso", polarization="V"),
    "rx_array":            "same_as_tx",       # reuse tx_array settings

    # (name, pos[m], ori[rad], role, P_tx[dBm])
    "tx_list": [
        ("tx0",  [-100, -100, 40], [ 5*np.pi/6, 0, 0],  "desired", 30),
        ("tx1",  [-100,   50, 40], [  np.pi/6, 0, 0],  "desired", 30),
        ("tx2",  [ 100, -100, 40], [-np.pi/2, 0, 0],  "desired", 30),
        # jammers
        ("jam1", [ 100,   50, 40], [ np.pi/2,  0, 0],  "jammer", 40),
        ("jam2", [  50,   50, 40], [ np.pi/2,  0, 0],  "jammer", 40),
        ("jam3", [ -50,  -50, 40], [ np.pi/2,  0, 0],  "jammer", 40),
    ],
    "rx":                  ("rx", [0, 0, 40]),

    # Path & radio-map solver params (keep moderate for GPU memory)
    "pathsolver":          dict(max_depth=4, los=True, specular_reflection=True,
                                 diffuse_reflection=False, refraction=False,
                                 synthetic_array=False, seed=41),
    "radiosolver":         dict(max_depth=8, cell_size=(1., 1.),
                                 samples_per_tx=int(1e6)),

    # OFDM parameters
    "num_subcarriers":     1024,
    "num_symbols":         1024,
    "subcarrier_spacing":  30e3,      # Hz

    # Plot settings
    "sinr_vmin":           -40,       # dB
    "sinr_vmax":             0,
}

# ───────────────────────────── Helper functions ────────────────────────────────

def dbm2w(dbm: float | np.ndarray) -> np.ndarray:
    """dBm → W"""
    return 1e-3 * 10 ** (np.asarray(dbm) / 10)


def add_tx(scene: rt.Scene, name: str, pos: List[float], ori: List[float],
           role: str, power_dbm: float) -> Transmitter:
    """Utility wrapper that *does* forward the orientation to the Tx object."""
    tx = Transmitter(name=name, position=pos, orientation=ori,
                     power_dbm=power_dbm)
    tx.role = role  # custom attribute for easy grouping
    scene.add(tx)
    return tx


def build_scene(cfg: Dict[str, Any]) -> Tuple[rt.Scene, Receiver, List[Transmitter]]:
    """Create scene, arrays, add all Tx/Rx and return them."""
    scene = load_scene(cfg["scene"])

    tx_arr_cfg = cfg["tx_array"]
    rx_arr_cfg = tx_arr_cfg if cfg["rx_array"] == "same_as_tx" else cfg["rx_array"]
    scene.tx_array = PlanarArray(**tx_arr_cfg)
    scene.rx_array = PlanarArray(**rx_arr_cfg)

    # Remove default actors (if any) to start clean
    for name in list(scene.transmitters):
        scene.remove(name)
    for name in list(scene.receivers):
        scene.remove(name)

    # Add transmitters
    txs: List[Transmitter] = []
    for name, pos, ori, role, p_dbm in cfg["tx_list"]:
        txs.append(add_tx(scene, name, pos, ori, role, p_dbm))

    # Add receiver
    rx_name, rx_pos = cfg["rx"]
    rx = Receiver(name=rx_name, position=rx_pos)
    scene.add(rx)

    return scene, rx, txs


def solve_paths(scene: rt.Scene, cfg: Dict[str, Any]) -> rt.Paths:
    """Run PathSolver once (velocities must already be set)"""
    solver = PathSolver()
    return solver(scene, **cfg["pathsolver"])


def compute_cfr(paths: rt.Paths, cfg: Dict[str, Any],
                norm_delay: bool = True) -> np.ndarray:
    """Return CFR with time evolution: shape (num_tx, T, F)."""
    F = cfg["num_subcarriers"]
    freqs = subcarrier_frequencies(F, cfg["subcarrier_spacing"])
    sym_dur = 1 / cfg["subcarrier_spacing"]
    return paths.cfr(
        frequencies=freqs,
        sampling_frequency=1 / sym_dur,
        num_time_steps=cfg["num_symbols"],
        normalize_delays=norm_delay,
        normalize=False,
        out_type="numpy",
    ).squeeze()


def to_delay_doppler(h_tf: np.ndarray) -> np.ndarray:
    """Convert (T,F) CFR to delay–Doppler spectrum."""
    h_shift = np.fft.fftshift(h_tf, axes=1)            # F shift
    h_delay = np.fft.ifft(h_shift, axis=1, norm="ortho")
    h_dd    = np.fft.fft(h_delay, axis=0,  norm="ortho")
    return np.fft.fftshift(h_dd, axes=0)               # Doppler shift

# ─────────────────────────────────── Main ──────────────────────────────────────

def main(cfg: Dict[str, Any] = CONFIG) -> None:
    scene, rx, txs = build_scene(cfg)

    # Example velocity assignment (all Tx fly along +x @30 m/s)
    for tx in txs:
        tx.velocity = [30, 0, 0]

    # == Radio-map (SINR) ======================================================
    rm_solver = RadioMapSolver()
    radio_map = rm_solver(scene, **cfg["radiosolver"])

    # == Propagation paths =====================================================
    paths = solve_paths(scene, cfg)

    # == CFR ===================================================================
    h_unit = compute_cfr(paths, cfg, norm_delay=False)   # (N_tx,T,F)
    print(h_unit.shape)  # (N_tx,T,F)
    tx_p_lin = dbm2w([tx.power_dbm for tx in txs])      # (N_tx,)
    print("tx_p_lin.shape", tx_p_lin.shape)  # (N_tx,)
    sqrtP    = np.sqrt(tx_p_lin)[:, None]            #
    print("sqrtP.shape", sqrtP.shape)   
    h_all = h_unit * sqrtP 
    print(h_all.shape)  # (N_tx,T,F)

    # Group indices
    idx_des = [i for i, tx in enumerate(txs) if tx.role == "desired"]
    idx_jam = [i for i, tx in enumerate(txs) if tx.role == "jammer"]

    h_des = h_all[idx_des].sum(axis=0)  # (T,F)
    h_jam = h_all[idx_jam].sum(axis=0)
    print(h_des.shape, h_jam.shape)  # (T,F)

    # == Delay–Doppler small window plot ======================================
    hdd_des = to_delay_doppler(h_des)
    hdd_jam = to_delay_doppler(h_jam)

    T, F = hdd_des.shape
    offset = 20  # half-window size
    d_bins_ns = np.arange(F) * (1/cfg["subcarrier_spacing"])/F * 1e9
    doppler_bins = np.fft.fftshift(np.fft.fftfreq(T, d=1/cfg["subcarrier_spacing"]))

    d_slice = slice(0, offset*2)
    t_mid = T//2
    t_slice = slice(t_mid-offset, t_mid+offset)

    X, Y = np.meshgrid(d_bins_ns[d_slice], doppler_bins[t_slice])
    Z_des = np.abs(hdd_des[t_slice, d_slice])
    Z_jam = np.abs(hdd_jam[t_slice, d_slice])

    fig = plt.figure(figsize=(12,4))
    for k, (Z, title) in enumerate([(Z_des, "Desired"), (Z_jam, "Jammer")], 1):
        ax = fig.add_subplot(1, 2, k, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
        ax.set_title(f"Delay–Doppler |{title}|")
        ax.set_xlabel("Delay (ns)"); ax.set_ylabel("Doppler (Hz)")
    plt.tight_layout(); plt.show()

    # == SINR radio-map ========================================================
    cc = radio_map.cell_centers.numpy()  # (Ny,Nx,3)
    x_unique, y_unique = cc[0,:,0], cc[:,0,1]
    rss = [radio_map.rss[i].numpy() for i in range(len(txs))]
    rss_des = sum(rss[i] for i in idx_des)
    rss_jam = sum(rss[i] for i in idx_jam)
    N0 = 1e-12
    sinr_db = 10*np.log10(np.clip(rss_des/(rss_des + rss_jam + N0), 1e-12, None))

    Xg, Yg = np.meshgrid(x_unique, y_unique)
    plt.figure(figsize=(6,5))
    pcm = plt.pcolormesh(Xg, Yg, sinr_db, shading="nearest",
                         vmin=cfg["sinr_vmin"], vmax=cfg["sinr_vmax"])
    plt.colorbar(pcm, label="SINR (dB)")
    plt.scatter([t.position[0] for t in txs if t.role=="desired"],
                [t.position[1] for t in txs if t.role=="desired"],
                marker="^", c="lime", edgecolor="k", label="Tx")
    plt.scatter([t.position[0] for t in txs if t.role=="jammer"],
                [t.position[1] for t in txs if t.role=="jammer"],
                marker="x", c="red", label="Jam")
    plt.scatter(rx.position[0], rx.position[1], marker="o", c="gold", label="Rx")
    plt.legend(); plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.title("SINR Map"); plt.tight_layout(); plt.show()

    # == (Optional) quick 3-D scene preview ====================================
    scene.preview(paths=paths)


if __name__ == "__main__":
    main()