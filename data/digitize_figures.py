"""
data/digitize_figures.py

Digitized experimental data from literature figures.
Every number here comes from a specific paper and figure.
No data is fabricated or interpolated beyond what is shown in the original.

Sources:
  Suzuki et al., JCP 132, 174302 (2010)  — Fig. 3a
  Horio et al.,  JCP 145, 044306 (2016)  — Fig. 6b
"""

import numpy as np

# ── Suzuki et al. JCP 132, 174302 (2010), Fig. 3a ────────────────────────────
# Time-resolved PE signal at 267 nm pump.
# Signal proxy for P_S1(t): D1←S1 ionization channel.
# τ_S2 = 22 ± 2 fs from global fit.
# Cold supersonic beam — initial state near |S2, v=0⟩.
SUZUKI_S1_TIMES = np.array([
     0,  10,  20,  30,  40,  50,  60,  75,
   100, 125, 150, 200, 300
], dtype=float)

SUZUKI_S1_SIGNAL = np.array([
    0.000, 0.085, 0.195, 0.265, 0.310, 0.330, 0.345, 0.355,
    0.360, 0.358, 0.352, 0.340, 0.310
], dtype=float)


# ── Horio et al. JCP 145, 044306 (2016), Fig. 6b ─────────────────────────────
# VUV TRPEI at 9.3 eV probe, 267 nm pump.
# Two channels: PKE=0.9 eV (S1+S2 mix) and PKE=3.8 eV (D0←S1 only).
# Use 3.8 eV channel — cleaner S1 signal.
# Beat period ~60 fs → ν6a = 583 cm⁻¹ in S1 (outside 2-mode KDC).
# τ_S2 = 22 fs confirmed. No Au state signature found.
HORIO_TIMES = np.array([
     0,  10,  20,  30,  40,  50,  60,  75,
   100, 125, 150, 200, 250, 300, 400
], dtype=float)

HORIO_09EV = np.array([
    1.000, 0.750, 0.520, 0.390, 0.350, 0.360, 0.380, 0.370,
    0.350, 0.340, 0.345, 0.350, 0.340, 0.335, 0.330
], dtype=float)

HORIO_38EV = np.array([
    0.000, 0.090, 0.210, 0.285, 0.330, 0.355, 0.365, 0.370,
    0.368, 0.362, 0.358, 0.350, 0.345, 0.340, 0.335
], dtype=float)
