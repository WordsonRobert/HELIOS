# runs3 Results

**Run:** wordson's_runs3
**Machine:** WordsonMSI · w22linux · Intel Xeon CPU (no GPU used)
**Total time:** ~4 hours
**Date:** March 29, 2026

## Config

```
Train: suzuki2010, kdc_lindblad, savith
Test:  fssh (Leone), horio2016, scutelnic
Sweeps: 3 x 1000 iterations
Optimizer: Adam (b1=0.9, b2=0.999)
LR: 0.003 -> 0.001 cosine staircase
FD eps: om1=1e-3, kap1_S1=1e-2, others=1e-4
Bounds: gamma in [-0.025, -0.005]
LLM init: Groq llama-3.3-70b-versatile (curl subprocess)
```

## Parameter Recovery

```
Parameter         GT      Learned    Error    Error%
E_S1          3.9950      4.0620   +0.0670     1.7%
E_S2          4.9183      4.9511   +0.0328     0.7%  <- best ever
om1           0.1273      0.1800   +0.0527    41.4%  <- needs FSSH
om10a         0.1133      0.1396   +0.0263    23.2%
kap1_S1      -0.0470     -0.0429   +0.0041     8.7%  <- best ever
kap1_S2      -0.2012     -0.2327   -0.0315    15.7%
gamma        -0.0180     -0.0120   +0.0060    33.2%  <- best ever (was 177%)
lam           0.1825      0.1593   -0.0232    12.7%

Best total loss: 0.164177 (Sweep 2)
```

## Matrix Comparison

```
Matrix fidelity:     0.875570
||H_learned-H_GT||:  20.8541 eV
Eigenvalue MAE:       0.2584 eV (first 20)
P_S1 MAE:            0.0690
```

## P_S1(t) Comparison

```
t(fs)   P_S1(GT)  P_S1(lrn)  |diff|
   10      0.060      0.137   0.078
   20      0.164      0.172   0.008
   30      0.132      0.180   0.048
   50      0.152      0.235   0.083
   75      0.183      0.259   0.075
  100      0.165      0.327   0.162
  150      0.255      0.218   0.036
  200      0.357      0.345   0.012
  300      0.462      0.340   0.123
  500      0.428      0.362   0.066
```

## Sweep History

| Sweep | Best Loss | Notes |
|-------|-----------|-------|
| 1 | 0.165176 | LLM init |
| 2 | **0.164177** | Best overall |
| 3 | 0.168110 | E_S2 started 0.17 off |

## Key Findings

1. Dropping Leone FSSH: gamma 177.8% -> 33.2%, kap1_S1 96.7% -> 8.7%
2. Dropping Leone FSSH: om1 3.7% -> 41.4% (FSSH was constraining om1 correctly)
3. Single-initial-state populations give near-zero gradient through gamma by construction
4. ADC(2)/KDC inconsistency is the primary failure mode, not model architecture
5. om1 and gamma require different observables: quantum beat at 560 cm-1 (Suzuki 2010)
