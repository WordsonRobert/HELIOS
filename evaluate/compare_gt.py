"""
evaluate/compare_gt.py

Validate learned H_KDC(theta*) against Hahn & Stock 2001 ground truth.

Device: WordsonMSI CPU
Time:   ~30 seconds

Ground truth: Woywod et al. JCP 100, 1400 (1994) CASSCF/MRCI
              = Hahn & Stock PCCP 3, 2331 (2001) Table 1

runs3 results:
  Matrix fidelity: 0.8756
  ||DeltaH||_F:    20.85 eV
  Eigenvalue MAE:  0.2584 eV
  P_S1 MAE:        0.0690
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from physics.hamiltonian import build_H_KDC, HAHN_STOCK_2001, params_to_vec, PARAM_NAMES
from physics.propagator import make_psi0, propagate_to_times


def main():
    path = 'trained_params.npz'
    if not os.path.exists(path):
        print("No trained_params.npz found. Run training first:")
        print("  python -u train/sgd_datasets.py")
        sys.exit(1)

    data       = np.load(path, allow_pickle=True)
    theta_star = data['theta_star']
    theta_gt   = params_to_vec(HAHN_STOCK_2001)
    H_learned  = build_H_KDC(theta_star)
    H_gt       = build_H_KDC(theta_gt)

    print("=" * 60)
    print("  HELIOS — H_KDC vs Hahn & Stock 2001")
    print("=" * 60)
    print(f"\n{'Param':12s} {'GT':>10s} {'Learned':>10s} {'Error%':>8s}")
    print("-" * 44)
    for i, name in enumerate(PARAM_NAMES):
        gt  = theta_gt[i]; pr = theta_star[i]
        pct = 100 * abs(pr - gt) / (abs(gt) + 1e-8)
        print(f"{name:12s} {gt:>10.4f} {pr:>10.4f} {pct:>7.1f}%")

    fro = np.linalg.norm(H_learned - H_gt, 'fro')
    fid = 1.0 - fro / np.linalg.norm(H_gt, 'fro')
    print(f"\nMatrix fidelity : {fid:.6f}")
    print(f"||DeltaH||_F    : {fro:.4f} eV")

    evals_gt  = np.linalg.eigvalsh(H_gt)[:20]
    evals_lrn = np.linalg.eigvalsh(H_learned)[:20]
    print(f"Eigenvalue MAE  : {np.mean(np.abs(evals_gt - evals_lrn)):.4f} eV")

    psi0  = make_psi0()
    times = np.array([10, 20, 30, 50, 75, 100, 150, 200, 300, 500], dtype=float)
    rgt   = propagate_to_times(H_gt,      psi0, times)
    rlrn  = propagate_to_times(H_learned, psi0, times)

    print(f"\n{'t(fs)':>8s} {'P_S1(GT)':>10s} {'P_S1(lrn)':>10s} {'|diff|':>8s}")
    print("-" * 42)
    for i, t in enumerate(times):
        print(f"{t:>8.0f} {rgt['pop_S1'][i]:>10.3f} "
              f"{rlrn['pop_S1'][i]:>10.3f} "
              f"{abs(rgt['pop_S1'][i]-rlrn['pop_S1'][i]):>8.3f}")

    print(f"\nP_S1 MAE: {np.mean(np.abs(rgt['pop_S1'] - rlrn['pop_S1'])):.4f}")
    print(f"\nRef: Woywod JCP 100 1400 (1994) | Hahn & Stock PCCP 3 2331 (2001)")


if __name__ == '__main__':
    main()
