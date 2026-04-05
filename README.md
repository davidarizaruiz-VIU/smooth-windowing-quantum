# smooth-windowing-quantum

Reference implementation accompanying the manuscript

> D. Ariza-Ruiz, *Parameter-Free Absorbing Boundaries for Pseudo-Spectral
> Quantum Dynamics via C∞ Windowing*, submitted to *Computer Physics
> Communications* (2026).

The code implements a split-step Fourier solver for the one-dimensional
time-dependent Schrödinger equation in which the wave function is
multiplied at each time step by a compactly supported C∞ bump function
with a plateau, following the quantitative Fourier-analytic theory of
Bergold and Lasser (*J. Fourier Anal. Appl.*, 2020). The window acts as
a parameter-free absorbing boundary condition that eliminates
wrap-around errors without auxiliary fields, modified Hamiltonians, or
tuning parameters.

## Repository contents

| File | Description |
|------|-------------|
| `smooth_windowing_abc.py` | Main script. Runs the eleven numerical experiments reported in the paper (Fourier approximation and Gibbs suppression, plateau-parameter study, window-regularity comparison, CAP benchmark, temporal convergence, above-barrier and tunneling scattering) and generates all figures and tables. |
| `convergence_extended.py` | Companion script. Generates `convergence_extended.csv` using 80-digit `mpmath` arbitrary-precision arithmetic, extending the convergence analysis far beyond the `float64` machine-epsilon floor. |
| `pml_benchmark_1d.py` | Independent 1D PML benchmark (Crank–Nicolson, 2nd-order finite differences, Thomas algorithm). Produces `pml_results.txt`. |
| `convergence_extended.csv` | Precomputed high-precision convergence data (80-digit `mpmath`). |
| `figure_1.png` … `figure_8.png` | Figures reproduced in the manuscript. |
| `RESULTADOS.txt` | Console log of a reference run of `smooth_windowing_abc.py`. |
| `pml_results.txt` | Console log of the PML benchmark. |
| `LICENSE` | MIT License. |

## Requirements

- Python 3.9 or later
- `numpy`, `matplotlib`, `mpmath`
- Optional: `gmpy2` (C backend for `mpmath`; provides ~5× speed-up for
  the extended-precision convergence script)

Install with:

```bash
pip install numpy matplotlib mpmath
pip install gmpy2   # optional, recommended
```

## Usage

Reproduce all figures and tables of the paper:

```bash
python3 smooth_windowing_abc.py
```

Regenerate the high-precision convergence table (80-digit arithmetic):

```bash
python3 convergence_extended.py
```

Run the PML benchmark:

```bash
python3 pml_benchmark_1d.py
```

## License

MIT License. See `LICENSE`.

## Citation

If you use this code in academic work, please cite the accompanying
manuscript (to appear in *Computer Physics Communications*). An
archival snapshot of the repository will be deposited on Zenodo upon
acceptance.
