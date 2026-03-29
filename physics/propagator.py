"""
physics/propagator.py

Quantum propagator for KDC Hamiltonian using matrix exponential.
Uses cached U_dt for speed — recomputes only when theta changes.

Device: WordsonMSI CPU (scipy.linalg.expm)
"""

import numpy as np
from scipy.linalg import expm
from physics.hamiltonian import (
    build_H_KDC, D, NEL, DVIB, N1, N10A, HBAR_EVFS, Q1_full
)

# Initial state: |S2, v1=0, v10a=0⟩
# Electronic index 1 = S2 (0-indexed: 0=S1, 1=S2)
# Vibrational ground state = index 0
def make_psi0() -> np.ndarray:
    """Pure state |S2, v1=0, v10a=0⟩ — cold supersonic beam initial condition."""
    psi0 = np.zeros(D, dtype=complex)
    psi0[DVIB] = 1.0   # block 1 (S2), vib index 0
    return psi0


# Electronic projectors for population extraction
_I_vib = np.eye(DVIB, dtype=complex)
_P_S1_full = np.block([[_I_vib, np.zeros((DVIB, DVIB))],
                        [np.zeros((DVIB, DVIB)), np.zeros((DVIB, DVIB))]])
_P_S2_full = np.block([[np.zeros((DVIB, DVIB)), np.zeros((DVIB, DVIB))],
                        [np.zeros((DVIB, DVIB)), _I_vib]])


def propagate_to_times(H: np.ndarray,
                       psi0: np.ndarray,
                       times_fs: np.ndarray,
                       dt_fs: float = 2.0) -> dict:
    """
    Propagate |ψ(t)⟩ = U(t)|ψ₀⟩ and extract populations at requested times.

    Uses cached U_dt = expm(-i H dt / ℏ).
    Stepping through integer multiples of dt_fs.

    Args:
        H:        (D,D) complex Hamiltonian in eV
        psi0:     (D,) complex initial state
        times_fs: requested output times in fs
        dt_fs:    propagation timestep in fs

    Returns:
        dict with 'pop_S1', 'pop_S2', 'Qt_S1' arrays at times_fs
    """
    U_dt = expm(-1j * H * dt_fs / HBAR_EVFS)

    pop_S1 = np.zeros(len(times_fs))
    pop_S2 = np.zeros(len(times_fs))
    Qt_S1  = np.zeros(len(times_fs))

    target_steps = np.round(np.array(times_fs) / dt_fs).astype(int)

    psi  = psi0.copy()
    step = 0

    for i, t_step in enumerate(target_steps):
        while step < t_step:
            psi  = U_dt @ psi
            step += 1
        # Populations
        psi_S1 = psi[:DVIB]
        psi_S2 = psi[DVIB:]
        p1 = float(np.sum(np.abs(psi_S1)**2))
        p2 = float(np.sum(np.abs(psi_S2)**2))
        # ⟨Q1⟩ in S1 subspace
        q1_full = psi.conj() @ (Q1_full @ psi)
        qt1 = float(np.real(q1_full)) / (p1 + 1e-10)

        pop_S1[i] = p1
        pop_S2[i] = p2
        Qt_S1[i]  = qt1

    return {'times_fs': times_fs, 'pop_S1': pop_S1, 'pop_S2': pop_S2, 'Qt_S1': Qt_S1}


# ── Cached propagator for training speed ──────────────────────────────────────
_cache = {}

def _get_U_dt(theta: np.ndarray, dt_fs: float = 2.0):
    key = (theta.tobytes(), dt_fs)
    if key not in _cache:
        H    = build_H_KDC(theta)
        U_dt = expm(-1j * H * dt_fs / HBAR_EVFS)
        _cache[key] = U_dt
        if len(_cache) > 20:
            del _cache[next(iter(_cache))]
    return _cache[key]


def predict_populations(theta: np.ndarray,
                        times_fs: np.ndarray,
                        psi0: np.ndarray,
                        dt_fs: float = 2.0) -> dict:
    """Fast cached version for training loop."""
    U_dt = _get_U_dt(theta, dt_fs)
    pop_S1 = np.zeros(len(times_fs))
    pop_S2 = np.zeros(len(times_fs))
    Qt_S1  = np.zeros(len(times_fs))

    target_steps = np.round(np.array(times_fs) / dt_fs).astype(int)
    psi  = psi0.copy()
    step = 0

    for i, t_step in enumerate(target_steps):
        while step < t_step:
            psi  = U_dt @ psi
            step += 1
        psi_S1 = psi[:DVIB]
        psi_S2 = psi[DVIB:]
        p1 = float(np.sum(np.abs(psi_S1)**2))
        p2 = float(np.sum(np.abs(psi_S2)**2))
        q1_full = psi.conj() @ (Q1_full @ psi)
        pop_S1[i] = p1
        pop_S2[i] = p2
        Qt_S1[i]  = float(np.real(q1_full)) / (p1 + 1e-10)

    return {'pop_S1': pop_S1, 'pop_S2': pop_S2, 'Qt_S1': Qt_S1}
