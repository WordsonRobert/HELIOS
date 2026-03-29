# Dataset Provenance

## Training Datasets

### kdc_lindblad_populations.npz
- **File:** `data/kdc_lindblad_populations.npz`
- **Generator:** `data_generation/kdc_lindblad_gen.py`
- **Device:** Google Colab A100-SXM4-40GB
- **Time:** ~5 minutes
- **Initial state:** |S2, v_6a=0, v_10a=0> (exact cold beam)
- **All KDC params fixed to:** Woywod et al. JCP 100, 1400 (1994) Table IV CASSCF/MRCI
- **tau_S2 constraint:** 22 fs from Suzuki JCP 132 174302 (2010) + Horio JCP 145 044306 (2016)
- **t range:** 0-150 fs, 151 points, dt=1 fs
- **Role:** train · weight=0.3

### savith_populations.npz
- **File:** `data/savith_populations.npz`
- **Generator:** Savith, MQB-Kraus Lindblad model
- **Device:** Google Colab A100-SXM4-40GB
- **Time:** ~30 minutes
- **tau_S2:** ~22 fs
- **t range:** 0-150 fs, 50 points (subsampled)
- **Role:** train · weight=0.2

### suzuki2010 (digitized, no file needed)
- **Paper:** Suzuki, Horio, Whitaker & Suzuki
- **Journal:** J. Chem. Phys. 132, 174302 (2010)
- **DOI:** 10.1063/1.3395206
- **Figure:** Fig. 3a — D1 PE signal vs time, 267 nm pump
- **Points:** 13 time points, t=0 to 300 fs
- **Key measurement:** tau_S2 = 22 +/- 2 fs
- **Initial conditions:** Supersonic cold beam, near-pure |S2,v=0>
- **Role:** train

---

## Test Datasets (never seen during training)

### horio2016 (digitized, no file needed)
- **Paper:** Horio, Suzuki, Mikosch & Suzuki
- **Journal:** J. Chem. Phys. 145, 044306 (2016)
- **DOI:** 10.1063/1.4955296
- **Figure:** Fig. 6b — two-channel PE beats, 3.8 eV channel used
- **Points:** 15 time points
- **Key finding:** tau_S2 = 22 fs confirmed; no Au state signature
- **Role:** test

### populations_corrected.npz (Leone FSSH)
- **Paper:** Scutelnic et al.
- **Journal:** Nat. Commun. 12, 5003 (2021)
- **DOI:** 10.1038/s41467-021-25407-0
- **Source:** SHARC 2.1 FSSH trajectories, Zenodo supplementary dataset
- **Method:** ADC(2)/cc-pVDZ, 200 trajectories, Wigner initial conditions
- **State numbering:** SHARC State1=S2, State2=S1, State3=Au (corrected)
- **Why test only:**
  - ADC(2) has no gamma parameter -> gradient conflict on gamma
  - Wigner initial conditions != KDC |S2,v=0> assumption
  - Au state present -> outside 2-state KDC model
  - Including in training: gamma 177.8% error, kap1_S1 96.7% error
- **Role:** test

### scutelnic X-ray dA
- **Paper:** Scutelnic et al. Nat. Commun. 12, 5003 (2021)
- **Source:** Experimental N K-edge transient absorption, Leone lab Berkeley
- **Energy range:** 395-410 eV (N K-edge), integrated over energy
- **Why test only:** Au state contamination at t~200 fs; pump-probe init
- **Role:** test

---

## Ground Truth Reference

All 8 GT parameter values:

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| E_S1 | 3.9950 | eV | Woywod 1994 Table IV / Hahn & Stock 2001 Table 1 |
| E_S2 | 4.9183 | eV | Woywod 1994 Table IV / Hahn & Stock 2001 Table 1 |
| om1 | 0.1273 | eV | Hahn & Stock 2001 Table 1 |
| om10a | 0.1133 | eV | Hahn & Stock 2001 Table 1 |
| kap1_S1 | -0.0470 | eV | Stock 1995 Table II MRCI |
| kap1_S2 | -0.2012 | eV | Stock 1995 Table II MRCI |
| gamma | -0.0180 | eV | Stock 1995 Table III |
| lam | 0.1825 | eV | Woywod 1994 / Stock 1995 |

**Woywod et al.** JCP 100, 1400 (1994) — doi:10.1063/1.466615
**Hahn & Stock** PCCP 3, 2331 (2001) — doi:10.1039/b008782p
**Stock et al.** JCP 103, 6851 (1995) — doi:10.1063/1.470689
**Udagawa et al.** Chem.Phys. 46, 237 (1980) — doi:10.1016/0301-0104(80)85102-6
