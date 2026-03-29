"""
physics/hamiltonian.py

KDC (Köppel-Domcke-Cederbaum) diabatic Hamiltonian for pyrazine S2→S1 IC.
Builds a 640×640 complex Hermitian matrix from 8 parameters.

Basis: 2 electronic states (S1, S2) × 8 ν1 modes × 5 ν10a modes
       → Hilbert space dimension D = 2 × 8 × 5 = 80
       Wait — standard KDC uses N1=8, N10a=5 → D=80 per state → 160 total
       With ground state: 3 × 8 × 5 = 120... check propagator for actual D.

Ground truth: Woywod et al. JCP 100, 1400 (1994) CASSCF/MRCI
              = Hahn & Stock PCCP 3, 2331 (2001) Table 1
"""

import numpy as np
from scipy.linalg import expm

# ── Ground truth (Hahn & Stock 2001, Table 1) ─────────────────────────────────
HAHN_STOCK_2001 = {
    'E_S1'    : 3.9950,   # eV  S1 excitation energy
    'E_S2'    : 4.9183,   # eV  S2 excitation energy
    'om1'     : 0.1273,   # eV  ν1  ring breathing  (1015 cm⁻¹)
    'om10a'   : 0.1133,   # eV  ν10a out-of-plane   (919  cm⁻¹)
    'kap1_S1' : -0.0470,  # eV  linear coupling S1
    'kap1_S2' : -0.2012,  # eV  linear coupling S2
    'gamma'   : -0.0180,  # eV  quadratic off-diagonal coupling
    'lam'     : 0.1825,   # eV  interstate coupling
}

PARAM_NAMES = ['E_S1', 'E_S2', 'om1', 'om10a', 'kap1_S1', 'kap1_S2', 'gamma', 'lam']
N_PARAMS    = len(PARAM_NAMES)

HBAR_EVFS   = 0.6582119569   # eV·fs


def params_to_vec(params: dict) -> np.ndarray:
    return np.array([params[k] for k in PARAM_NAMES], dtype=np.float64)


def vec_to_params(vec: np.ndarray) -> dict:
    return {k: float(vec[i]) for i, k in enumerate(PARAM_NAMES)}


# ── Fock space operators ──────────────────────────────────────────────────────
def _ladder(n):
    """Return (Q, N, I) for n-level harmonic oscillator."""
    a = np.zeros((n, n), dtype=complex)
    for k in range(1, n):
        a[k-1, k] = np.sqrt(float(k))
    adag = a.conj().T
    Q = (a + adag) / np.sqrt(2.0)
    N = adag @ a
    I = np.eye(n, dtype=complex)
    return Q, N, I


# Basis sizes
N1   = 8    # ν1  levels
N10A = 5    # ν10a levels
NEL  = 2    # electronic states (S1, S2)
DVIB = N1 * N10A
D    = NEL  * DVIB   # total Hilbert space dimension = 80

# Build vibrational operators once
_Q1_1d,   _N1_1d,   _I1   = _ladder(N1)
_Q10a_1d, _N10a_1d, _I10a = _ladder(N10A)

_Q1_vib   = np.kron(_Q1_1d,   _I10a)
_N1_vib   = np.kron(_N1_1d,   _I10a)
_Q10a_vib = np.kron(_I1,      _Q10a_1d)
_N10a_vib = np.kron(_I1,      _N10a_1d)
_Ivib     = np.eye(DVIB, dtype=complex)

# Electronic projectors
_P_S1 = np.array([[1, 0], [0, 0]], dtype=complex)  # |S1><S1|
_P_S2 = np.array([[0, 0], [0, 1]], dtype=complex)  # |S2><S2|
_Poff = np.array([[0, 1], [1, 0]], dtype=complex)  # |S1><S2| + |S2><S1|

# Full-space operators
Q1_full   = np.kron(np.eye(NEL, dtype=complex), _Q1_vib)
Q10a_full = np.kron(np.eye(NEL, dtype=complex), _Q10a_vib)


def build_H_KDC(theta: np.ndarray) -> np.ndarray:
    """
    Build 80×80 KDC Hamiltonian from 8-parameter vector.

    H = E_S1·|S1><S1| + E_S2·|S2><S2|
      + ω₁·N₁ + ω₁₀ₐ·N₁₀ₐ
      + κ₁(S1)·|S1><S1|·Q₁ + κ₁(S2)·|S2><S2|·Q₁
      + λ·(|S1><S2|+h.c.)·Q₁₀ₐ
      + γ·(|S1><S2|+h.c.)·Q₁₀ₐ²

    Args:
        theta: np.ndarray of shape (8,) = [E_S1, E_S2, om1, om10a,
                                            kap1_S1, kap1_S2, gamma, lam]
    Returns:
        H: complex np.ndarray of shape (80, 80)
    """
    E_S1, E_S2, om1, om10a, kap1_S1, kap1_S2, gamma, lam = theta

    P_S1_full   = np.kron(_P_S1, _Ivib)
    P_S2_full   = np.kron(_P_S2, _Ivib)
    Poff_full   = np.kron(_Poff, _Ivib)
    N1_full     = np.kron(np.eye(NEL, dtype=complex), _N1_vib)
    N10a_full   = np.kron(np.eye(NEL, dtype=complex), _N10a_vib)
    Q10a_sq     = Q10a_full @ Q10a_full

    H = (E_S1    * P_S1_full
       + E_S2    * P_S2_full
       + om1     * N1_full
       + om10a   * N10a_full
       + kap1_S1 * P_S1_full @ Q1_full
       + kap1_S2 * P_S2_full @ Q1_full
       + lam     * Poff_full  @ Q10a_full
       + gamma   * Poff_full  @ Q10a_sq)

    return H
