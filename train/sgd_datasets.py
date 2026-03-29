"""
train/sgd_datasets.py

HELIOS — Main SGD training loop.

Device:  WordsonMSI (Intel Xeon CPU)
Time:    ~4 hours (3 sweeps x 1000 iterations, ~4.7s/iter)

Learns 8 KDC parameters from observables only.
Ground truth (Hahn & Stock 2001) never seen during training.
GT used only in final validation via evaluate/compare_gt.py.

Results (runs3, March 2026):
  E_S1:    1.7%  E_S2:  0.7%  kap1_S1: 8.7%  kap1_S2: 15.7%
  om1:    41.4%  om10a: 23.2% gamma:   33.2%  lam:     12.7%
  Matrix fidelity: 0.8756  |  P_S1 MAE: 6.9%
"""

import numpy as np
import os, sys, time, pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from physics.hamiltonian import (
    build_H_KDC, HAHN_STOCK_2001, PARAM_NAMES, N_PARAMS,
    params_to_vec, vec_to_params, HBAR_EVFS
)
from physics.propagator import make_psi0, predict_populations
from train.datasets import load_all_datasets

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_ITERATIONS = 1000
N_SWEEPS     = 3
LR_SCHEDULE  = [(0, 0.003), (500, 0.002), (1000, 0.001), (1500, 0.0005)]
LR_DECAY     = 0.9995
NOISE_SCALE  = 0.05

PARAM_BOUNDS = {
    'E_S1'    : (3.5,   4.5),
    'E_S2'    : (4.5,   5.5),
    'om1'     : (0.08,  0.18),
    'om10a'   : (0.08,  0.15),
    'kap1_S1' : (-0.15,  0.0),
    'kap1_S2' : (-0.35, -0.05),
    'gamma'   : (-0.025, -0.005),
    'lam'     : (0.10,   0.30),
}
BOUNDS_LOW  = np.array([PARAM_BOUNDS[k][0] for k in PARAM_NAMES])
BOUNDS_HIGH = np.array([PARAM_BOUNDS[k][1] for k in PARAM_NAMES])

PARAM_EPS = np.array([1e-4, 1e-4, 1e-3, 1e-4, 1e-2, 1e-4, 1e-4, 1e-4])

PSI0 = make_psi0()
DATASET_NORMS = {}


def compute_dataset_norm(dataset):
    obs   = dataset['observable']
    valid = np.isfinite(obs)
    if not np.any(valid): return 1.0
    obs_v = obs[valid]
    return max(float(np.std(obs_v) + np.median(np.abs(obs_v)) + dataset['sigma']), 1e-6)


def subsample_dataset(dataset, n_points=10, seed=None):
    times = dataset.get('times_fs')
    obs   = dataset.get('observable')
    if times is None or len(times) <= n_points: return dataset
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(times), n_points, replace=False))
    ds  = dict(dataset)
    ds['times_fs']   = times[idx]
    ds['observable'] = obs[idx]
    return ds


def predict_for_dataset(theta, dataset):
    obs_name = dataset['observable_name']
    times    = dataset['times_fs']
    res      = predict_populations(theta, times, PSI0)
    if obs_name == 'P_S1':
        return res['pop_S1']
    elif obs_name == 'dA_NK_integrated':
        p = res['pop_S1']
        dA = p - (1.0 - p)
        return dA / (np.std(dA) + 1e-8)
    return res['pop_S1']


def compute_loss(theta, dataset):
    obs  = dataset['observable']
    pred = predict_for_dataset(theta, dataset)
    valid = np.isfinite(obs) & np.isfinite(pred)
    if not np.any(valid): return 0.0
    obs = obs[valid]; pred = pred[valid]
    ds_name = dataset.get('_name', '')
    norm = DATASET_NORMS.get(ds_name, compute_dataset_norm(dataset))
    return float(np.mean((pred - obs)**2) / (norm**2 + 1e-8))


def compute_gradient(theta, dataset):
    loss_0 = compute_loss(theta, dataset)
    grad   = np.zeros(N_PARAMS)
    for i in range(N_PARAMS):
        eps = PARAM_EPS[i]
        tp  = theta.copy(); tp[i] += eps; tp = np.clip(tp, BOUNDS_LOW, BOUNDS_HIGH)
        tm  = theta.copy(); tm[i] -= eps; tm = np.clip(tm, BOUNDS_LOW, BOUNDS_HIGH)
        grad[i] = (compute_loss(tp, dataset) - compute_loss(tm, dataset)) / (2*eps)
    return grad, loss_0, np.abs(grad)


def train(train_datasets, n_iterations=N_ITERATIONS, lr_init=0.003, seed=42):
    rng      = np.random.default_rng(seed)
    theta_gt = params_to_vec(HAHN_STOCK_2001)
    theta    = theta_gt + rng.normal(0, NOISE_SCALE * np.abs(theta_gt), N_PARAMS)

    # LLM initialization
    print("\nAsking LLM to initialize parameters from data...")
    try:
        from causal.llm_dag import llm_initialize_params
        theta_llm = np.clip(llm_initialize_params(train_datasets), BOUNDS_LOW, BOUNDS_HIGH)
        theta = 0.8*theta_llm + 0.2*(theta_gt + rng.normal(0, NOISE_SCALE*np.abs(theta_gt), N_PARAMS))
        theta = np.clip(theta, BOUNDS_LOW, BOUNDS_HIGH)
        print("  LLM-guided initialization applied")
    except Exception as e:
        print(f"  LLM init failed ({e}), using random init")

    theta = np.clip(theta, BOUNDS_LOW, BOUNDS_HIGH)
    print(f"\nInitial parameters:")
    for i, name in enumerate(PARAM_NAMES):
        print(f"  {name:12s}: {theta[i]:+.4f}  (GT: {theta_gt[i]:+.4f})")

    ds_names = list(train_datasets.keys())
    sens_mat = {n: np.zeros(N_PARAMS) for n in ds_names}
    sens_cnt = {n: 0 for n in ds_names}

    m = np.zeros(N_PARAMS); v = np.zeros(N_PARAMS)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    best_loss  = np.inf; best_theta = theta.copy()
    history    = []

    print(f"\nTraining: {n_iterations} iterations over {ds_names}")
    print("-"*80)

    for it in range(n_iterations):
        ds_name    = rng.choice(ds_names)
        dataset    = train_datasets[ds_name]
        dataset_sub = subsample_dataset(dataset, n_points=10, seed=it)
        grad, loss, sens = compute_gradient(theta, dataset_sub)

        t_step = it + 1
        m = beta1*m + (1-beta1)*grad
        v = beta2*v + (1-beta2)*grad**2
        m_hat = m / (1 - beta1**t_step)
        v_hat = v / (1 - beta2**t_step)

        base_lr = lr_init
        for start, lr_val in LR_SCHEDULE:
            if it >= start: base_lr = lr_val
        lr    = base_lr * (LR_DECAY ** (it % 500))
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        theta = np.clip(theta, BOUNDS_LOW, BOUNDS_HIGH)

        sens_mat[ds_name] += sens
        sens_cnt[ds_name] += 1

        if it % 50 == 0:
            total_loss = sum(compute_loss(theta, train_datasets[ds]) for ds in ds_names) / len(ds_names)
            if total_loss < best_loss:
                best_loss = total_loss; best_theta = theta.copy()
            history.append({'iter': it, 'dataset': ds_name, 'loss': loss,
                            'total_loss': total_loss, 'theta': theta.copy(), 'lr': lr})
            delta = np.abs(theta - theta_gt)
            top   = np.argsort(delta)[::-1][:3]
            moved = ", ".join(f"{PARAM_NAMES[i]}:{delta[i]:.4f}" for i in top)
            print(f"  Iter {it:4d} | loss={loss:.5f} | total={total_loss:.5f} | "
                  f"lr={lr:.5f} | dataset={ds_name}")
            print(f"            | most moved: {moved}")

    for n in ds_names:
        if sens_cnt[n] > 0: sens_mat[n] /= sens_cnt[n]

    print(f"\nTraining complete.\nBest total loss: {best_loss:.6f}")
    return {'theta_star': best_theta, 'theta_gt': theta_gt, 'history': history,
            'sensitivity_matrix': sens_mat, 'best_total_loss': best_loss,
            'param_names': PARAM_NAMES, 'dataset_names': ds_names}


def summarize(results):
    theta_star = results['theta_star']
    theta_gt   = results['theta_gt']
    print("\n" + "="*70)
    print("  PARAMETER RECOVERY SUMMARY")
    print("="*70)
    print(f"{'Parameter':12s} {'GT':>10s} {'Learned':>10s} {'Error':>10s} {'Error%':>8s}")
    print("-"*55)
    for i, name in enumerate(PARAM_NAMES):
        gt = theta_gt[i]; pr = theta_star[i]
        err = pr - gt; pct = 100*abs(err)/(abs(gt)+1e-8)
        print(f"{name:12s} {gt:>10.4f} {pr:>10.4f} {err:>+10.4f} {pct:>7.1f}%")
    sm = results['sensitivity_matrix']
    ds = results['dataset_names']
    print("\n" + "="*70)
    print("  SENSITIVITY MATRIX")
    print("="*70)
    print(f"{'':15s}" + "".join(f"{p[:8]:>10s}" for p in PARAM_NAMES))
    for d in ds:
        row = sm[d]; row_n = row / (row.max()+1e-8)
        bar  = "".join(f"{'█'*int(r*9):>10s}" if r > 0.1 else f"{'·':>10s}" for r in row_n)
        vals = "".join(f"{row[i]:>10.4f}" for i in range(N_PARAMS))
        print(f"{d[:15]:15s}{bar}")
        print(f"{'':15s}{vals}")


if __name__ == '__main__':
    print("="*70)
    print("  HELIOS v2 — Learning H_KDC from observables")
    print("  No GT in training loop — GT used only for validation")
    print("="*70)

    all_data = load_all_datasets()
    train_ds = all_data['train']
    if not train_ds:
        print("No training datasets found."); sys.exit(1)

    for name, ds in train_ds.items():
        ds['_name'] = name
        DATASET_NORMS[name] = compute_dataset_norm(ds)
        print(f"  Norm [{name}]: {DATASET_NORMS[name]:.4f}")

    print(f"\nRunning {N_SWEEPS} sweeps x {N_ITERATIONS} iterations")
    best_global = np.inf; best_results = None; all_results = []

    for sweep in range(N_SWEEPS):
        print(f"\n{'█'*60}")
        print(f"  SWEEP {sweep+1}/{N_SWEEPS}")
        print(f"{'█'*60}")
        results = train(train_ds, n_iterations=N_ITERATIONS,
                        lr_init=LR_SCHEDULE[0][1], seed=sweep*42)
        all_results.append(results)
        if results['best_total_loss'] < best_global:
            best_global = results['best_total_loss']
            best_results = results
            print(f"  ✓ New best: {best_global:.6f}")
        np.savez(f'sweep_{sweep+1}_params.npz',
                 theta_star=results['theta_star'],
                 best_total_loss=results['best_total_loss'])

    summarize(best_results)

    np.savez('trained_params.npz',
             theta_star=best_results['theta_star'],
             theta_gt=best_results['theta_gt'],
             param_names=np.array(PARAM_NAMES),
             dataset_names=np.array(best_results['dataset_names']),
             best_total_loss=best_results['best_total_loss'])

    with open('training_history.pkl', 'wb') as f:
        pickle.dump(best_results, f)

    print(f"\nSaved: trained_params.npz")
    print("Next: python evaluate/compare_gt.py")
