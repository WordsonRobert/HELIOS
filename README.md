# HELIOS
### Hamiltonian Equivariant Learning for Inferring Observable-constrained States

Recovers the 8-parameter KDC diabatic Hamiltonian for pyrazine S2→S1 internal
conversion from experimental and computational observables. Ground truth never
seen during training — only used for final validation.

---

## Results (runs3 · WordsonMSI · March 2026)

| Parameter | Symbol | GT (Hahn & Stock 2001) | Learned | Error |
|-----------|--------|----------------------|---------|-------|
| S1 energy | E_S1 | 3.9950 eV | 4.0620 eV | 1.7% |
| S2 energy | E_S2 | 4.9183 eV | 4.9511 eV | **0.7%** |
| ν1 freq | om1 | 0.1273 eV | 0.1800 eV | 41.4% |
| ν10a freq | om10a | 0.1133 eV | 0.1396 eV | 23.2% |
| κ1(S1) | kap1_S1 | −0.0470 eV | −0.0429 eV | 8.7% |
| κ1(S2) | kap1_S2 | −0.2012 eV | −0.2327 eV | 15.7% |
| γ | gamma | −0.0180 eV | −0.0120 eV | 33.2% |
| λ | lam | 0.1825 eV | 0.1593 eV | 12.7% |

**Matrix fidelity:** 0.8756 · **||ΔH||_F:** 20.85 eV · **P_S1 MAE:** 6.9%

**Training:** 3 sweeps × 1000 iters · ~4 hours · WordsonMSI (Intel Xeon CPU)
**Data gen:** ~5 min · Google Colab A100-SXM4-40GB

---

## Repo Structure

```
HELIOS/
├── physics/
│   ├── hamiltonian.py       build_H_KDC(): 80x80 from 8 params
│   └── propagator.py        expm(-iHt/hbar), populations
├── data/
│   ├── digitize_figures.py  Suzuki 2010 Fig.3a + Horio 2016 Fig.6b
│   └── README.md
├── data_generation/
│   └── kdc_lindblad_gen.py  Savith generator (Colab A100, ~5 min)
├── train/
│   ├── sgd_datasets.py      SGD training loop (Adam + FD gradients)
│   └── datasets.py          All dataset loaders
├── causal/
│   ├── llm_dag.py           Groq LLM init (llama-3.3-70b)
│   └── .groq_key            Your key here (gitignored)
├── evaluate/
│   └── compare_gt.py        Validate vs Hahn & Stock 2001
├── results/
│   └── runs3_summary.md     Full runs3 output
└── docs/
    └── datasets.md          Dataset provenance + all citations
```

---

## Data Sources

| Dataset | Paper | Figure | Role | Device | Time |
|---------|-------|--------|------|--------|------|
| kdc_lindblad | Generated (this repo) | — | train | Colab A100 | ~5 min |
| savith | MQB-Kraus Lindblad | — | train | Colab A100 | ~30 min |
| suzuki2010 | Suzuki et al. JCP 132 174302 (2010) | Fig. 3a | train | digitized | — |
| horio2016 | Horio et al. JCP 145 044306 (2016) | Fig. 6b | test | digitized | — |
| leone_fssh | Scutelnic et al. Nat.Commun. 12 5003 (2021) | Zenodo SHARC | test | Zenodo | — |

**Ground truth:** Woywod et al. JCP 100 1400 (1994) CASSCF/MRCI
                  → Hahn & Stock PCCP 3 2331 (2001) Table 1

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/HELIOS
cd HELIOS
pip install -r requirements.txt

# 1. Generate KDC-Lindblad training data (GPU, ~5 min)
python data_generation/kdc_lindblad_gen.py

# 2. (Optional) LLM init — get key at console.groq.com
echo "gsk_YOUR_KEY" > causal/.groq_key && chmod 600 causal/.groq_key

# 3. Train (~4 hours CPU)
python -u train/sgd_datasets.py 2>&1 | tee results/run.log

# 4. Validate
python evaluate/compare_gt.py
```

---

## Authors

- **Wordson** (w22linux) — ML, training, data pipeline · WordsonMSI
- **Savith** — MQB-Kraus Lindblad model · Colab A100

---

## Citations

```
Woywod et al.    JCP 100, 1400 (1994)           doi:10.1063/1.466615
Hahn & Stock     PCCP 3, 2331 (2001)            doi:10.1039/b008782p
Stock et al.     JCP 103, 6851 (1995)           doi:10.1063/1.470689
Udagawa et al.   Chem.Phys. 46, 237 (1980)      doi:10.1016/0301-0104(80)85102-6
Suzuki et al.    JCP 132, 174302 (2010)         doi:10.1063/1.3395206
Horio et al.     JCP 145, 044306 (2016)         doi:10.1063/1.4955296
Scutelnic et al. Nat.Commun. 12, 5003 (2021)   doi:10.1038/s41467-021-25407-0
```
