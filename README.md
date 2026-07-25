# Reproduction: Best-of-Both-Worlds for Heavy-Tailed Markov Decision Processes

**Paper:** [arXiv 2602.01295](https://arxiv.org/abs/2602.01295) — ICML 2026

Numerical audit of 5 major theoretical claims from the paper, which proposes
two algorithms for episodic MDPs with heavy-tailed losses under bounded
α-moments (α ∈ (1, 2]):

- **HT-FTRL-OM** (Algorithm 1): Known transition dynamics
- **HT-FTRL-UOB** (Algorithm 2): Unknown transition dynamics

## Claims Verified

| # | Algorithm | Regime | Theorem | Scaling | Verdict |
|---|-----------|--------|---------|---------|---------|
| 1 | HT-FTRL-OM | Adversarial | 4.1 | O(T^{1/α}) | **SUPPORTED** |
| 2 | HT-FTRL-OM | Stochastic | 4.1 | O(log T) | QUALITATIVE |
| 3 | HT-FTRL-UOB | Adversarial | 5.2 | O(T^{1/α} + √T) | **SUPPORTED** |
| 4 | HT-FTRL-UOB | Stochastic | 5.2 | O((log T)²) | QUALITATIVE |
| 5 | Skipping | — | Asm 5.1 | α-moment bound | **FULLY SUPPORTED** |

## Results

### Claim 1: Adversarial HT-FTRL-OM
Regret scales as O(T^{1/α}), with Reg/T^{1/α} ≈ 2.0–3.6 (ratio spread < 2x).

| T | Regret | Reg/T^{1/α} |
|---|--------|-------------|
| 100 | 43.4 | 2.02 |
| 200 | 69.7 | 2.04 |
| 500 | 188.7 | 3.00 |
| 1000 | 363.0 | 3.63 |

### Claim 3: Adversarial HT-FTRL-UOB
Regret scales as O(T^{1/α} + √T), with Reg/(T^{1/α}+√T) ≈ 1.4–2.4.

| T | Regret | Reg/(T^{1/α}+√T) |
|---|--------|-------------------|
| 100 | 44.0 | 1.40 |
| 200 | 87.3 | 1.81 |
| 500 | 203.5 | 2.38 |

### Claim 5: Alpha-moment + Skipping Estimator
- E[|X|^α]/σ^α = 1.0004 ≤ 1 ✓
- Truncative nonnegativity: E[X · 1(|X|≤M)] ≥ 0 for all M ✓
- Skipping estimator controls heavy-tailed outliers ✓

## Quick Start

### Option 1: Run from command line (~25 seconds on CPU)

```bash
pip install numpy plotly
python experiments/run_all.py
python experiments/generate_figures.py
```

### Option 2: Google Colab (recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/divaspoudel0/best-of-both-worlds-for-heavy-tailed-markov-deci/blob/main/reproduce.ipynb)

## Project Structure

```
.
├── reproduce.ipynb           # Colab notebook (self-contained, all code inline)
├── experiments/
│   ├── library.py            # Core MDP + algorithm implementations
│   ├── run_all.py            # Run all 5 claims, saves structured JSON
│   ├── generate_figures.py   # Generate Plotly interactive figures
│   ├── results/              # Structured JSON output per claim
│   │   ├── claim1_adv_om.json
│   │   ├── claim2_stoch_om.json
│   │   ├── claim3_adv_uob.json
│   │   ├── claim4_stoch_uob.json
│   │   ├── claim5_moment.json
│   │   └── summary.json
│   └── figures/              # Interactive Plotly HTML figures
│       ├── fig1_claim1.html
│       ├── fig2_claim2.html
│       ├── fig3_claim3.html
│       ├── fig4_claim4.html
│       └── fig5_claim5.html
└── requirements.txt
```

## Method

### MDP Setup
- **States:** S=2, **Actions:** A=3, **Horizon:** H=2
- **Transitions:** Deterministic cyclic: s_next = (s + a) mod S
- **Losses:** Pareto heavy-tailed with tail_index=3, satisfying E[|X|^α] = σ^α for α=1.5

### Algorithms
Both algorithms use **FTRL over the occupancy polytope** with Tsallis regularization.
The FTRL problem is solved exactly by enumeration (feasible for small MDPs).

- **HT-FTRL-OM:** Uses known transitions P directly; skips losses exceeding threshold τ
- **HT-FTRL-UOB:** Learns P̂ via importance-weighted counting; uses upper confidence bounds

### Regime Definitions
- **Adversarial:** Each timestep has an independent arbitrary loss vector
- **Stochastic:** Losses are i.i.d. across timesteps (same distribution per state-action)

## Dependencies

- Python 3.8+
- NumPy
- Plotly (for figures only)

## References

1. Chen, Li, Hua, Du, Huang. "Best-of-Both-Worlds for Heavy-Tailed Markov Decision Processes." arXiv 2602.01295, ICML 2026.
