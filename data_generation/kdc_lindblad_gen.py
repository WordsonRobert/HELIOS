"""
data_generation/kdc_lindblad_gen.py

Savith KDC-Lindblad population generator.

Device:  Google Colab A100-SXM4-40GB
Time:    ~5 minutes
Output:  data/kdc_lindblad_populations.npz

All KDC parameters fixed to:
  Woywod et al. JCP 100, 1400 (1994) CASSCF/MRCI
  = Hahn & Stock PCCP 3, 2331 (2001) Table 1

One learnable parameter: Lindblad S2->S1 dissipation rate
Constraint: tau_S2 = 22 fs
  Suzuki et al. JCP 132, 174302 (2010) Fig. 3a
  Horio  et al. JCP 145, 044306 (2016) Fig. 6b

Initial state: |S2, v_6a=0, v_10a=0> (cold supersonic beam)
Output:        t in [0, 150] fs, 151 points, dt=1 fs

Run on Colab:
  !python data_generation/kdc_lindblad_gen.py
"""

import math, time, os
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

HBAR  = 0.6582119569
CM2EV = 1.2398419e-4

# Woywod 1994 CASSCF/MRCI = Hahn & Stock 2001 Table 1
KDC = {
    "E_S1"     : 3.9950,
    "E_S2"     : 4.9183,
    "om6a"     : 596.0  * CM2EV,
    "om10a"    : 919.0  * CM2EV,
    "kap6a_S1" : -0.0964,
    "kap6a_S2" :  0.1193,
    "lam"      :  0.1825,
    "gamma"    : -0.0180,
}
TAU_TARGET = 22.0


def ladder(n, device):
    cd = torch.complex128
    a  = torch.zeros(n, n, dtype=cd, device=device)
    for k in range(1, n):
        a[k-1, k] = math.sqrt(float(k))
    adag = a.conj().T.contiguous()
    return (a + adag) / math.sqrt(2.0), adag @ a, torch.eye(n, dtype=cd, device=device)


def build_ops(N6a, N10a, device):
    cd   = torch.complex128
    Nel  = 3; Dvib = N6a * N10a
    Q6a_1d,  N6a_1d,  I6a  = ladder(N6a,  device)
    Q10a_1d, N10a_1d, I10a = ladder(N10a, device)
    Ivib = torch.eye(Dvib, dtype=cd, device=device)
    Iel  = torch.eye(Nel,  dtype=cd, device=device)
    lift = lambda op: torch.kron(Iel, op)
    Q6a  = lift(torch.kron(Q6a_1d,  I10a))
    Q10a = lift(torch.kron(I6a,     Q10a_1d))
    N6a_ = lift(torch.kron(N6a_1d,  I10a))
    N10a_= lift(torch.kron(I6a,     N10a_1d))
    P = {}
    for k in range(Nel):
        ek = torch.zeros(Nel, dtype=cd, device=device); ek[k] = 1.0
        P[k] = torch.kron(torch.outer(ek, ek), Ivib)
    e1 = torch.zeros(Nel, dtype=cd, device=device); e1[1] = 1.0
    e2 = torch.zeros(Nel, dtype=cd, device=device); e2[2] = 1.0
    Poff   = torch.kron(torch.outer(e1, e2) + torch.outer(e2, e1), Ivib)
    L_S2S1 = torch.kron(torch.outer(e1, e2), Ivib)
    D = Nel * Dvib
    return {"D": D, "_N6a": N6a, "_N10a": N10a, "P": P, "Poff": Poff,
            "L_S2S1": L_S2S1, "Q6a": Q6a, "Q10a": Q10a,
            "Q10a_sq": Q10a @ Q10a, "N6a": N6a_, "N10a": N10a_,
            "I": torch.eye(D, dtype=cd, device=device)}


def build_H(ops, p):
    return (p["E_S1"]     * ops["P"][1] + p["E_S2"]  * ops["P"][2]
          + p["om6a"]     * ops["N6a"]  + p["om10a"] * ops["N10a"]
          + p["kap6a_S1"] * ops["P"][1] @ ops["Q6a"]
          + p["kap6a_S2"] * ops["P"][2] @ ops["Q6a"]
          + p["lam"]      * ops["Poff"] @ ops["Q10a"]
          + p["gamma"]    * ops["Poff"] @ ops["Q10a_sq"])


def build_Lsup(ops, H, g_S2):
    I = ops["I"]; Lj = ops["L_S2S1"]
    LdL = Lj.conj().T.contiguous() @ Lj
    Lc  = -1j * (torch.kron(I, H) - torch.kron(H.conj().T.contiguous(), I))
    Ld  = g_S2 * (torch.kron(Lj.conj(), Lj)
                 - 0.5 * torch.kron(I, LdL)
                 - 0.5 * torch.kron(LdL.conj().T.contiguous(), I))
    return Lc + Ld


def make_rho0(ops):
    D = ops["D"]; N6a = ops["_N6a"]; N10a = ops["_N10a"]
    psi = torch.zeros(D, dtype=torch.complex128, device=ops["I"].device)
    psi[2 * N6a * N10a] = 1.0
    return torch.outer(psi, psi.conj()).reshape(-1)


def propagate(ops, U_dt, rho0, n_steps):
    trP = [ops["P"][k].T.reshape(-1) for k in range(3)]
    rho = rho0.clone(); P0, P1, P2 = [], [], []
    for _ in range(n_steps):
        P0.append(torch.real(trP[0] @ rho))
        P1.append(torch.real(trP[1] @ rho))
        P2.append(torch.real(trP[2] @ rho))
        rho = U_dt @ rho
    return torch.stack(P0).float(), torch.stack(P1).float(), torch.stack(P2).float()


class KDCDataGen(nn.Module):
    def __init__(self, N6a=3, N10a=5, device=DEVICE):
        super().__init__()
        self.device = device
        ops = build_ops(N6a, N10a, device)
        self.ops = ops
        print(f"D={ops['D']}, Liouvillian={ops['D']**2}x{ops['D']**2}")
        for k, v in KDC.items():
            self.register_buffer(k, torch.tensor(v, dtype=torch.float64))
        self.log_g = nn.Parameter(torch.tensor(math.log(1/40.0), dtype=torch.float32))
        self.register_buffer("rho0", make_rho0(ops))

    def tau(self):
        return 1.0 / torch.exp(self.log_g).item()

    def forward(self, n_steps, dt=1.0):
        p    = {k: getattr(self, k) for k in KDC}
        H    = build_H(self.ops, p)
        g_S2 = torch.exp(self.log_g).double()
        Lsup = build_Lsup(self.ops, H, g_S2)
        U_dt = torch.linalg.matrix_exp(Lsup * dt / HBAR)
        return propagate(self.ops, U_dt, self.rho0, n_steps)


def loss_tau(P_S2, t_fs, target=TAU_TARGET):
    dt  = float(t_fs[1] - t_fs[0])
    i1  = min(int(10.0/dt), len(P_S2)-2)
    i2  = min(int(60.0/dt), len(P_S2)-1)
    tau = ((t_fs[i2]-t_fs[i1]) /
           torch.log(P_S2[i1].clamp(1e-6)/P_S2[i2].clamp(1e-6)).clamp(0.1))
    i_t = min(int(target/dt), len(P_S2)-1)
    anchor = (P_S2[i_t].clamp(1e-6) - math.exp(-1.0))**2
    return (tau - target)**2 + 5.0*anchor, tau


def run():
    DT = 1.0; N_TRAIN = 200; N_EVAL = 600; EPOCHS = 400
    t_train = torch.arange(N_TRAIN, dtype=torch.float32, device=DEVICE) * DT
    t_eval  = torch.arange(N_EVAL,  dtype=torch.float32, device=DEVICE) * DT

    model = KDCDataGen(N6a=3, N10a=5, device=DEVICE).to(DEVICE)
    with torch.no_grad():
        P0_, P1_, P2_ = model(5, DT)
    assert abs(P2_[0].item()-1.0) < 0.01, "P_S2(t=0) != 1"
    print(f"P_S2(t=0)={P2_[0].item():.4f} ✓  init tau_S2={model.tau():.1f}fs")

    t0    = time.time()
    opt   = torch.optim.Adam([model.log_g], lr=3e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-4)
    print(f"\nTraining: tau_S2 -> {TAU_TARGET:.0f} fs ({EPOCHS} epochs)")

    for ep in range(EPOCHS):
        opt.zero_grad()
        P0, P1, P2 = model(N_TRAIN, DT)
        l, tau_est = loss_tau(P2, t_train)
        l.backward()
        torch.nn.utils.clip_grad_norm_([model.log_g], 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            model.log_g.clamp_(math.log(1/60), math.log(1/8))
        if ep % 100 == 0 or ep == EPOCHS-1:
            print(f"  ep {ep:4d}  tau_S2={tau_est.item():.1f}fs  loss={l.item():.4f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min  |  Final tau_S2={model.tau():.1f}fs")

    model.eval()
    t_np = t_eval.cpu().numpy()
    with torch.no_grad():
        P0, P1, P2 = model(N_EVAL, DT)
    p0=P0.cpu().numpy(); p1=P1.cpu().numpy(); p2=P2.cpu().numpy()

    mask_fit = (t_np >= 5) & (t_np <= 60)
    tau_meas = -1.0 / np.polyfit(t_np[mask_fit], np.log(p2[mask_fit].clip(1e-8)), 1)[0]
    print(f"Measured tau_S2 from output: {tau_meas:.1f}fs  (target: {TAU_TARGET:.0f})")

    mask     = t_np <= 150.0
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data',
                            'kdc_lindblad_populations.npz')
    np.savez(out_path,
             times_fs        = t_np[mask],
             P_S0            = p0[mask],
             P_S1            = p1[mask],
             P_S2            = p2[mask],
             tau_S2_measured = np.float32(tau_meas),
             tau_S2_target   = np.float32(TAU_TARGET),
             gamma_eV        = np.float32(KDC["gamma"]),
             lam_eV          = np.float32(KDC["lam"]),
             device          = str(DEVICE),
             elapsed_min     = np.float32(elapsed/60),
             note = ("KDC-Lindblad. Init: |S2,v=0,v=0>. t<150fs. "
                     "Params: Woywod1994 MRCI = Hahn&Stock2001. "
                     "tau_S2 fitted to Suzuki2010+Horio2016 (22fs)."))
    print(f"Saved -> {out_path}")
    print(f"  Points: {mask.sum()} (0-150 fs, dt=1 fs)")
    print(f"  Device: {DEVICE}  |  Time: {elapsed/60:.1f} min")


if __name__ == '__main__':
    run()
