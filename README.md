# smooth-windowing-quantum

Companion code for the paper:

> **Parameter-Free Absorbing Boundaries for Pseudo-Spectral Quantum Dynamics via C&#8734; Windowing**
> David Ariza-Ruiz
> *Computer Physics Communications* (2026)

## Overview

This repository provides a self-contained Python implementation of a parameter-free absorbing boundary condition for FFT-based split-step solvers of the one-dimensional time-dependent Schrodinger equation. The boundary treatment consists of multiplying the wave function at each time step by a compactly supported C-infinity bump function that equals unity on an interior plateau and vanishes smoothly at the computational boundaries, exploiting the super-algebraic convergence guarantees established by Bergold and Lasser (2020).

## Repository contents

| File | Description |
|------|-------------|
| `smooth_windowing_abc.py` | Main script: runs all 13 numerical experiments and generates figures 1--14 |
| `pml_convergence_study.py` | PML convergence study with spatial refinement, higher-order stencils (FD2/FD4/FD6), and ultra-refined reference solutions |
| `convergence_extended.py` | Extended convergence analysis using 80-digit (`mpmath`) arithmetic |
| `convergence_extended.csv` | Pre-computed convergence data (80-digit precision) |
| `figure_1.pdf` -- `figure_14.pdf` | All figures reported in the paper |
| `requirements.txt` | Python dependencies |
| `CITATION.cff` | Citation metadata |
| `LICENSE` | MIT license |

## Numerical experiments (main script)

The 13 experiments in `smooth_windowing_abc.py` are:

1. Fourier approximation and Gibbs suppression
2. Wave packet absorption
3. Plateau parameter study
4. Window regularity comparison
5. Complex absorbing potential (CAP) comparison
6. CAP robustness test across momenta
7. Temporal convergence
8. Gaussian barrier scattering (above-barrier)
9. Gaussian barrier scattering (tunneling)
10. Comprehensive tunneling and strong-barrier campaign with barrier-shape universality
11. Delta-t anomaly diagnosis
12. 1D PML benchmark (single traversal)
13. Multiple-reflection benchmark (C-infinity windowing vs PML)

## Requirements

- Python 3.8+
- NumPy >= 1.22
- Matplotlib >= 3.5
- mpmath >= 1.3
- SciPy >= 1.10 (required for PML finite-difference solves; optional for main script)

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main experiments:

```bash
python smooth_windowing_abc.py
```

Run the PML convergence study:

```bash
python pml_convergence_study.py
```

Run the extended convergence analysis (computationally intensive):

```bash
python convergence_extended.py
```

All figures are saved as `figure_X.pdf` in the working directory.

## License

MIT License. See [LICENSE](LICENSE).

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ArizaRuiz2026,
  author  = {Ariza-Ruiz, David},
  title   = {Parameter-Free Absorbing Boundaries for Pseudo-Spectral Quantum
             Dynamics via {$C^\infty$} Windowing},
  journal = {Computer Physics Communications},
  year    = {2026}
}
```
