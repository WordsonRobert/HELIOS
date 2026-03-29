"""
train/datasets.py

Dataset loaders for HELIOS training and validation.

Train datasets (used during SGD):
  suzuki2010   — Suzuki et al. JCP 132, 174302 (2010), Fig. 3a
  kdc_lindblad — Generated: KDC-Lindblad, Woywod params, tau_S2=22fs
  savith       — MQB-Kraus Lindblad (Savith), tau_S2~22fs

Test datasets (never seen during training):
  fssh         — Leone FSSH, ADC(2)/cc-pVDZ (Scutelnic 2021 Zenodo)
  horio2016    — Horio et al. JCP 145, 044306 (2016), Fig. 6b
  scutelnic    — Scutelnic et al. Nat. Commun. 12, 5003 (2021), N K-edge

Why Leone FSSH is test-only:
  ADC(2) has no gamma parameter -> gradient conflict
  Wigner initial conditions != KDC |S2,v=0> assumption
  Au state present -> outside 2-state KDC model
"""

import numpy as np
import os, sys

DATA_DIR     = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
XRAY_EXP_DIR = '/home/w22linux/quantumchem2/xray_extracted/ncommun_21_03851_data/exp'
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def load_fssh():
    path = os.path.join(DATA_DIR, 'populations_corrected.npz')
    if not os.path.exists(path):
        print(f"  FSSH not found at {path}"); return None
    d = np.load(path)
    times = d['times_fs']; P_S1 = d['P_S1']
    valid = np.isfinite(times) & np.isfinite(P_S1)
    return {
        'times_fs': times[valid], 'observable': P_S1[valid],
        'observable_name': 'P_S1', 'sigma': 0.05, 'role': 'test',
        'source': 'Scutelnic 2021 Nat.Commun. SHARC ADC(2)/cc-pVDZ 200trajs',
    }


def load_suzuki2010():
    from data.digitize_figures import SUZUKI_S1_TIMES, SUZUKI_S1_SIGNAL
    mask = SUZUKI_S1_TIMES >= 0
    return {
        'times_fs': SUZUKI_S1_TIMES[mask], 'observable': SUZUKI_S1_SIGNAL[mask],
        'observable_name': 'P_S1', 'sigma': 0.05, 'role': 'train',
        'source': 'Suzuki et al. JCP 132 174302 (2010) Fig.3a digitized',
    }


def load_kdc_lindblad():
    path = os.path.join(DATA_DIR, 'kdc_lindblad_populations.npz')
    if not os.path.exists(path):
        print(f"  kdc_lindblad not found. Run: python data_generation/kdc_lindblad_gen.py")
        return None
    d = np.load(path)
    times = d['times_fs']; P_S1 = d['P_S1']
    valid = np.isfinite(times) & np.isfinite(P_S1)
    return {
        'times_fs': times[valid], 'observable': P_S1[valid],
        'observable_name': 'P_S1', 'sigma': 0.02, 'role': 'train', 'weight': 0.3,
        'source': 'KDC-Lindblad Woywod1994MRCI tau_S2=22fs (this repo)',
    }


def load_savith():
    path = os.path.join(DATA_DIR, 'savith_populations.npz')
    if not os.path.exists(path):
        print(f"  savith not found at {path}"); return None
    d = np.load(path)
    times = d['times_fs']; P_S1 = d['P_S1']
    mask  = np.isfinite(times) & np.isfinite(P_S1) & (times <= 150.0)
    idx   = np.linspace(0, mask.sum()-1, min(50, mask.sum())).astype(int)
    return {
        'times_fs': times[mask][idx], 'observable': P_S1[mask][idx],
        'observable_name': 'P_S1', 'sigma': 0.05, 'role': 'train', 'weight': 0.2,
        'source': 'Savith MQB-Kraus Lindblad tau_S2~22fs Colab A100',
    }


def load_horio2016():
    from data.digitize_figures import HORIO_TIMES, HORIO_38EV
    mask = HORIO_TIMES >= 0
    return {
        'times_fs': HORIO_TIMES[mask], 'observable': HORIO_38EV[mask],
        'observable_name': 'P_S1', 'sigma': 0.05, 'role': 'test',
        'source': 'Horio et al. JCP 145 044306 (2016) Fig.6b 3.8eV digitized',
    }


def load_scutelnic2021():
    short_path  = os.path.join(XRAY_EXP_DIR, 'Transient_scan_short.dat')
    energy_path = os.path.join(XRAY_EXP_DIR, 'Energy_calibration.dat')
    delays_path = os.path.join(XRAY_EXP_DIR, 'Delays_short.dat')
    for p in [short_path, energy_path, delays_path]:
        if not os.path.exists(p):
            print(f"  Scutelnic not found at {p}"); return None
    dA_raw   = np.loadtxt(short_path)
    energies = np.loadtxt(energy_path)
    delays   = np.loadtxt(delays_path)
    e_mask   = (energies >= 395.0) & (energies <= 410.0)
    dA_int   = dA_raw[e_mask, :].sum(axis=0)
    valid    = np.isfinite(dA_int) & np.isfinite(delays)
    dA_norm  = dA_int[valid] / (np.std(dA_int[valid]) + 1e-8)
    return {
        'times_fs': delays[valid], 'observable': dA_norm,
        'observable_name': 'dA_NK_integrated', 'sigma': 0.1, 'role': 'test',
        'source': 'Scutelnic et al. Nat.Commun.12 5003 (2021) N K-edge 395-410eV',
    }


def load_all_datasets():
    print("Loading all datasets...")
    loaders = {
        'fssh'        : load_fssh,
        'suzuki2010'  : load_suzuki2010,
        'kdc_lindblad': load_kdc_lindblad,
        'savith'      : load_savith,
        'horio2016'   : load_horio2016,
        'scutelnic'   : load_scutelnic2021,
    }
    all_ds = {}
    for name, loader in loaders.items():
        try:
            ds = loader()
            if ds is not None:
                ds['_name'] = name
                all_ds[name] = ds
                role = ds.get('role', '?')
                obs  = ds.get('observable_name', '?')
                n    = len(ds.get('times_fs', []))
                print(f"  ✓ {name:15s} [{role:5s}] {obs:20s} n={n}")
            else:
                print(f"  ✗ {name:15s} — skipped")
        except Exception as e:
            print(f"  ✗ {name:15s} — error: {e}")
    train_ds = {k: v for k, v in all_ds.items() if v.get('role') == 'train'}
    test_ds  = {k: v for k, v in all_ds.items() if v.get('role') == 'test'}
    print(f"\n  Train: {list(train_ds.keys())}")
    print(f"  Test:  {list(test_ds.keys())}")
    return {'train': train_ds, 'test': test_ds, 'all': all_ds}


if __name__ == '__main__':
    load_all_datasets()
