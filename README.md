<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Journal-Comput.%20Phys.%20Commun.-orange" alt="CPC"/>
  <img src="https://img.shields.io/badge/Open%20Science-Reproducible-brightgreen?logo=opensourceinitiative" alt="Open Science"/>
</p>

# Smooth Windowing for Pseudo-Spectral Quantum Dynamics

**Companion code for:**
> D. Ariza-Ruiz, *"Eliminating Wrap-Around Errors in Pseudo-Spectral Quantum Dynamics via C-infinity Windowing,"* Computer Physics Communications (2026).

---

## Overview

FFT-based pseudo-spectral methods are the workhorse of computational quantum dynamics, but their implicit periodic boundary conditions cause **wrap-around (aliasing) errors** whenever the wave function reaches the edge of the computational domain. This repository provides a complete Python implementation of a new absorbing boundary condition (ABC) that eliminates these artifacts using **compactly supported C-infinity bump functions**.

The method is grounded in the rigorous windowed Fourier series theory of [Bergold & Lasser (2020)](https://doi.org/10.1007/s00041-020-09773-3) and achieves:

- **Super-algebraic convergence** of the Fourier approximation error (faster than any polynomial rate).
- **Nine orders of magnitude** suppression of wrap-around artifacts in Schrodinger wave packet simulations.
- **O(N) overhead** per time step on top of the standard O(N log N) FFT solver -- a single pointwise multiplication.
- **Zero free parameters** beyond the absorbing layer width fraction.

## Repository structure

```
smooth-windowing-quantum/
|-- smooth_windowing_abc.py   # Main simulation script (all experiments)
|-- main.tex                  # LaTeX manuscript (elsarticle format)
|-- requirements.txt          # Python dependencies
|-- CITATION.cff              # GitHub citation metadata
|-- LICENSE                   # MIT License
|-- figures/                  # Generated automatically by the script
|   |-- figure1_convergence.png
|   |-- figure2_schrodinger.png
|   |-- figure3_norm_evolution.png
|   |-- figure4_convergence_rho.png
|   +-- figure5_window_comparison.png
+-- README.md
```

## Installation

```bash
git clone https://github.com/davidarizaruiz-VIU/smooth-windowing-quantum.git
cd smooth-windowing-quantum
pip install -r requirements.txt
```

**Requirements:** Python 3.9 or later. Only standard scientific Python packages are needed (NumPy, Matplotlib, SciPy).

## Quick start

Run all four numerical experiments and generate the five publication figures:

```bash
python smooth_windowing_abc.py
```

This produces:

| Figure | Description |
|--------|-------------|
| `figure1_convergence.png` | L2 convergence: standard vs. Hann (C1) vs. C-infinity bump |
| `figure2_schrodinger.png` | Wave packet dynamics: wrap-around error vs. clean absorption |
| `figure3_norm_evolution.png` | Norm evolution: unitarity vs. physical absorption |
| `figure4_convergence_rho.png` | Effect of the plateau parameter on convergence rate |
| `figure5_window_comparison.png` | Hann / Tukey / C-infinity bump comparison (4 panels) |

A summary table with quantitative diagnostics is printed to stdout.

## Method in brief

At each time step of the split-step Fourier solver, the wave function is multiplied by a C-infinity bump window:

```
psi^{n+1} = W * IFFT[ T_hat * FFT[ psi^n ] ]
```

where `W(x) = 1` in the physical interior and decays smoothly to `0` at the domain boundaries. The infinite differentiability of the bump ensures that no spurious reflections are generated at the transition interface.

## Experiments

1. **Gibbs suppression and convergence** -- The test function f(x) = x on [-1, 1] is approximated by truncated Fourier sums with and without windowing. The C-infinity bump achieves errors below 10^-6 with 50 coefficients.

2. **Schrodinger wave packet absorption** -- A Gaussian wave packet propagates on [-10, 10] under the free-particle TDSE. The standard FFT produces catastrophic wrap-around; the windowed method suppresses the artifact by a factor of ~10^9.

3. **Plateau parameter study** -- The non-degenerate bump with varying plateau widths reveals the theoretically predicted trade-off between plateau size and convergence rate.

4. **Window regularity comparison** -- Hann (C1), Tukey (C1), and C-infinity windows are compared on the Schrodinger problem, demonstrating that infinite differentiability is essential for optimal absorption.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ArizaRuiz2026smoothwindowing,
  author  = {Ariza-Ruiz, David},
  title   = {Eliminating Wrap-Around Errors in Pseudo-Spectral Quantum
             Dynamics via {$C^\infty$} Windowing},
  journal = {Computer Physics Communications},
  year    = {2026},
  note    = {Submitted},
  doi     = {}
}
```

and the foundational theory paper:

```bibtex
@article{BergoldLasser2020,
  author  = {Bergold, Paul and Lasser, Caroline},
  title   = {Fourier Series Windowed by a Bump Function},
  journal = {Journal of Fourier Analysis and Applications},
  volume  = {26},
  pages   = {65},
  year    = {2020},
  doi     = {10.1007/s00041-020-09773-3}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <em>Valencian International University (VIU), Spain</em>
</p>
