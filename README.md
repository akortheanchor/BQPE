# Adaptive Bayesian Quantum Phase Estimation (BQPE)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Frontiers](https://img.shields.io/badge/Journal-Frontiers%20in%20QST-orange)](https://www.frontiersin.org/)
[![DOI](https://img.shields.io/badge/arXiv-2604.05456-red)](https://arxiv.org/abs/2604.05456)

**Reproducibility repository for:**

> Akoramurthy B., Surendiran B., Cheng X. (2026)  
> *Bayesian Quantum Phase Estimation Beyond the Eigenstate Approximation:
> Sample Complexity, Cramér–Rao Bounds, and Adaptive Measurement Design
> for Non-Eigenstate and Mixed-State Inputs on NISQ Devices*  
> **Frontiers in Quantum Science and Technology**  
> DOI: [pending upon acceptance]

---
<img width="1280" height="720" alt="Graphical abstract_Final" src="https://github.com/user-attachments/assets/99ddeda4-bfa4-4ff0-bcec-cecf3f0edf48" />
To read full story of Bayesian Quantum Phase Estimation : ----https://medium.com/p/bd98e8cddee4?postPublishedType=initial
<img width="981" height="475" alt="anim3_crlb_landscape" src="https://github.com/user-attachments/assets/5419d760-5b16-41d4-8e50-630ff184b996" />


## Overview

This repository contains the complete, peer-reviewed implementation of the
adaptive BQPE protocol described in the manuscript. All figures, tables, and
numerical results in the paper can be reproduced exactly from this code.

**Key contributions reproduced here:**

| Contribution | File | Equation |
|---|---|---|
| Corrected CRLB (Theorem 1) | `src/adaptive_protocol.py` | `crlb()` |
| True adaptive protocol | `src/adaptive_protocol.py` | `run_adaptive_trial()` |
| Sample complexity bounds | `src/adaptive_protocol.py` | `sample_complexity_lower/upper()` |
| MCP-DLD hardware correction | `src/adaptive_protocol.py` | `effective_pg()` |
| Von Mises posterior update | `src/adaptive_protocol.py` | `vonmises_update()` |
| All manuscript figures | `src/figures.py` | `figure1()` – `figure7()` |

---

## Quick Start

```bash
git clone https://github.com/akoramurthy/bqpe-frontiers-2026.git
cd bqpe-frontiers-2026
pip install -r requirements.txt

# Run all unit tests
python -m pytest tests/ -v

# Reproduce Figure 7 (corrected Monte Carlo)
cd src && python figures.py --fig 7 --outdir ../figures

# Reproduce ALL figures
python figures.py --fig all --outdir ../figures --dpi 300

# Interactive notebook
jupyter notebook notebooks/BQPE_Reproducibility.ipynb
```

---

## Repository Structure

```
bqpe-frontiers-2026/
├── src/
│   ├── adaptive_protocol.py   # Core BQPE implementation (all algorithms)
│   └── figures.py             # Figure reproduction script
├── tests/
│   ├── test_theory.py         # Unit tests for all theoretical claims
│   └── conftest.py            # Pytest configuration
├── notebooks/
│   └── BQPE_Reproducibility.ipynb   # Interactive exploration
├── figures/                   # Generated figures (300 DPI PNG)
├── data/
│   └── digitised/             # Digitised experimental data (Santagati 2022,
│                              #   Valeri 2020, Pogorelov 2021)
├── docs/
│   └── corrections.md         # Record of all manuscript corrections
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Installation

### Option A — pip

```bash
pip install -r requirements.txt
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate bqpe
```

### Requirements

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.26 | Numerical computation |
| `scipy` | ≥ 1.13 | Bessel functions, statistics |
| `matplotlib` | ≥ 3.8 | Figure generation |
| `pytest` | ≥ 7.0 | Unit tests |
| `jupyter` | ≥ 1.0 | Interactive notebook |

---

## Reproducing Specific Results

### Theorem 1 — Corrected CRLB

```python
from src.adaptive_protocol import crlb

# Corrected: 1 / (N * p^2 * gamma^2 * nbar^2)
bound = crlb(N=100, p=0.7, gamma=0.9)
print(f'CRLB = {bound:.6f}')
# Expected: 0.022676...

# The original (erroneous) formula (1-pg)/(N*pg) gives:
p, g = 0.7, 0.9
original = (1 - p*g) / (100 * p * g)
print(f'Original (erroneous): {original:.6f}')
print(f'Ratio (corrected/original): {bound/original:.2f}x larger')
```

### Figure 7 — Corrected Monte Carlo

```python
from src.adaptive_protocol import run_monte_carlo

# TRUE adaptive protocol (n_t* = floor(kappa/(2pg)) + 1)
result = run_monte_carlo(
    N_shots=100, p=0.7, gamma=0.9,
    theta_true=3.14159/3,
    N_trials=2000, seed=2024,
    protocol='adaptive'   # NOT 'simplified'
)
print(f"MSE/CRLB efficiency ratio: {result['efficiency_ratio']:.3f}")
# Expected: ~1.15 ± 0.06 at N=100

# Compare with simplified (non-CRLB-achieving) schedule
result_simp = run_monte_carlo(
    N_shots=100, p=0.7, gamma=0.9,
    theta_true=3.14159/3,
    N_trials=2000, seed=2024,
    protocol='simplified'
)
print(f"Simplified efficiency ratio: {result_simp['efficiency_ratio']:.3f}")
# Expected: ~2.0+ (does NOT converge to CRLB)
```

### MCP-DLD Correction

```python
from src.adaptive_protocol import effective_pg

# NV-centre parameters (Santagati et al. 2022)
p, gamma = 0.85, 0.92
eta      = 0.50   # Standard MCP quantum efficiency

pg_ideal = p * gamma
pg_eff   = effective_pg(p, gamma, eta=eta)
print(f'Ideal:    p*gamma = {pg_ideal:.4f}')
print(f'With MCP: (p*gamma)_eff = {pg_eff:.4f}  ({pg_eff/pg_ideal*100:.0f}% of ideal)')
```

---

## Critical Corrections Implemented

This repository incorporates three critical corrections identified during
peer review. See `docs/corrections.md` for full details.

| Issue | Description | Implementation |
|---|---|---|
| **T1** | Fisher information formula at optimal basis | `crlb()` uses exact formula |
| **T2** | Sample complexity constants C₁, C₂ unspecified | `sample_complexity_lower/upper()` |
| **E1** | Monte Carlo used simplified n_t=t schedule | `protocol='adaptive'` is the default |

**Warning:** Using `protocol='simplified'` in `run_monte_carlo()` replicates
the erroneous prior-draft schedule for comparison purposes only. It does NOT
implement the adaptive protocol of Theorems 1–3 and does NOT achieve the CRLB.

---

## Unit Tests

All 25 unit tests verify the theoretical claims of the manuscript:

```bash
# Run all tests
python -m pytest tests/ -v

# Run only fast tests (skip Monte Carlo)
python -m pytest tests/ -v -m "not slow"

# Run a specific theorem test
python -m pytest tests/test_theory.py::TestCRLB -v
```

Expected output:

```
tests/test_theory.py::TestCRLB::test_eigenstate_unit_shots        PASSED
tests/test_theory.py::TestCRLB::test_scaling_in_N                 PASSED
tests/test_theory.py::TestCRLB::test_corrected_differs_from_original PASSED
tests/test_theory.py::TestEffectivePG::test_ideal_detector        PASSED
tests/test_theory.py::TestSampleComplexity::test_lower_constant   PASSED
tests/test_theory.py::TestSampleComplexity::test_upper_constant   PASSED
...
25 passed in X.XXs
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Akoramurthy2026,
  author  = {Akoramurthy, B. and Surendiran, B. and Cheng, Xiaochun},
  title   = {Bayesian Quantum Phase Estimation Beyond the Eigenstate
             Approximation: Sample Complexity, Cram\'{e}r--Rao Bounds,
             and Adaptive Measurement Design for Non-Eigenstate and
             Mixed-State Inputs on {NISQ} Devices},
  journal = {Frontiers in Quantum Science and Technology},
  year    = {2026},
  doi     = {pending}
}
```

Also cite the companion manuscript:

```bibtex
@misc{Akoramurthy2026PFA,
  author       = {Akoramurthy, B. and Surendiran, B.},
  title        = {Phase-Fidelity-Aware Truncated Quantum {F}ourier Transform
                  for Scalable Phase Estimation on {NISQ} Hardware},
  howpublished = {arXiv:2604.05456},
  year         = {2026}
}
```

---

## Authors

| Author | Affiliation | Email |
|---|---|---|
| Akoramurthy B. | NIT Puducherry, India | akoramurthy@nitpy.ac.in |
| Surendiran B. | NIT Puducherry, India | surendiran@nitpy.ac.in |
| Xiaochun Cheng | Swansea University, UK | x.cheng@swansea.ac.uk |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Figures are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
consistent with the Frontiers open-access licence.
