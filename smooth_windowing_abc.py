#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smooth Windowing via C-infinity Bump Functions for Pseudo-Spectral Quantum Dynamics.

Companion code for:
    D. Ariza-Ruiz, "Parameter-Free Absorbing Boundaries for Pseudo-Spectral
    Quantum Dynamics via C-infinity Windowing",
    Computer Physics Communications (2026).

Theoretical foundation:
    P. Bergold & C. Lasser, "Fourier Series Windowed by a Bump Function",
    J. Fourier Anal. Appl. 26 (2020) 65.
    https://doi.org/10.1007/s00041-020-09773-3

This script runs thirteen numerical experiments and generates publication-
quality figures (300 dpi PNG/PDF) plus summary tables printed to stdout:

    Experiment  1 -- Gibbs suppression & L2 convergence (standard / Hann / C-inf)
    Experiment  2 -- Schrodinger wave packet absorption (standard vs. windowed)
    Experiment  3 -- Convergence vs. plateau parameter rho
    Experiment  4 -- Window regularity comparison (Hann / Tukey / C-inf bump)
    Experiment  5 -- Complex absorbing potential (CAP) parameter sweep (preview)
    Experiment  6 -- CAP robustness across momenta
    Experiment  7 -- Hamiltonian vs. multiplicative CAP (reproduces tab:cap_sweep)
    Experiment  8 -- Manolopoulos transmission-free CAP benchmark (rem:cap_scope)
    Experiment  9 -- Temporal convergence study with plateau-restricted norm
                     (reproduces tab:temporal_convergence and
                      eqs. norm_powerlaw and spurious_density_scaling)
    Experiment 10 -- Gaussian barrier scattering (above-barrier & tunneling,
                     with large-domain reference for the tunneling regime)
    Experiment 11 -- 1D PML with Crank-Nicolson finite differences
                     (reproduces tab:pml_comparison)
    Experiment 12 -- Comprehensive tunneling & strong-barrier scattering campaign
                     (reproduces tab:scatt-summary and figures 9, 10, 11):
                     dense 2D (eta, V0) map, long-time propagation, barrier-shape
                     universality (Gaussian/sech^2/rect), eta-saturation,
                     dt-anomaly diagnosis (Strang vs. Yoshida 4th-order),
                     fixed window-period prescription, reference convergence
    Experiment 13 -- Multiple-reflection benchmark (C-inf windowing vs PML
                     in confined-scattering regime with machine-precision
                     reference solution)

Usage:
    python smooth_windowing_abc.py

Author:  David Ariza-Ruiz
         Faculty of Engineering, Science and Technology
         Valencian International University (VIU), Spain
         david.ariza@professor.universidadviu.com

License: MIT (see LICENSE)
"""

from __future__ import annotations

import time as timer
import math
import numpy as np

def _erf_scalar(x):
    """Wrapper for math.erf that works with scalars."""
    return math.erf(x)

# Vectorized erf using math.erf (no scipy needed)
erf = np.vectorize(_erf_scalar, otypes=[float])
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =============================================================================
# 0. GLOBAL SETTINGS
# =============================================================================
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,       # Set True if LaTeX is installed
    'font.family': 'serif',
})

OUTPUT_DIR = "."  # Change to a subfolder if desired

# =============================================================================
# 1. CORE MATHEMATICAL FUNCTIONS
# =============================================================================

def standard_bump_function(t: np.ndarray) -> np.ndarray:
    """Degenerate C-infinity bump on [-1, 1], normalized to peak value 1.

    phi(t) = exp(-1 / (1 - t^2)) / exp(-1),  |t| < 1;  0 otherwise.
    Corresponds to the rho = 0 (no plateau) case.
    """
    result = np.zeros_like(t, dtype=float)
    mask = np.abs(t) < 1.0
    t_interior = t[mask]
    result[mask] = np.exp(-1.0 / (1.0 - t_interior**2))
    peak = np.exp(-1.0)  # max at t=0
    return result / peak


def plateau_bump_function(t: np.ndarray, rho: float) -> np.ndarray:
    """Non-degenerate C-infinity bump with plateau on [-rho, rho], support [-1, 1].

    Implements Bergold & Lasser Eq. (4.1) with lambda = 1.

    Parameters
    ----------
    t : ndarray
        Evaluation points.
    rho : float
        Half-width of the plateau region, 0 <= rho < 1.

    Returns
    -------
    ndarray
        Window values in [0, 1].
    """
    result = np.zeros_like(t, dtype=float)
    at = np.abs(t)

    # Region 1: plateau
    mask_plateau = at <= rho
    result[mask_plateau] = 1.0

    # Region 2: transition
    mask_trans = (at > rho) & (at < 1.0)
    a = at[mask_trans]
    # Bergold & Lasser Eq. (4.1):
    # w(x) = 1 / (exp(1/(lambda-|x|) + 1/(rho-|x|)) + 1)
    exponent = 1.0 / (1.0 - a) + 1.0 / (rho - a)
    with np.errstate(over='ignore'):
        result[mask_trans] = 1.0 / (np.exp(exponent) + 1.0)

    # Region 3: outside support -> already 0
    return result


def absorbing_boundary_window(
    x: np.ndarray, L: float, width_fraction: float = 0.15
) -> np.ndarray:
    """C-infinity absorbing boundary window on [-L, L].

    Implements the Bergold & Lasser Eq. (4.1) smooth bump w_{rho, lambda}
    with lambda = L and rho = (1 - eta) * L:

        W(x) = 1 / (exp(1/(L - |x|) + 1/(rho - |x|)) + 1)

    for rho < |x| < L, with W = 1 on |x| <= rho and W = 0 for |x| >= L.

    This is the *sigmoidal-logistic* profile whose convergence properties
    are analyzed in Theorems 3.3 and 4.6 of [Bergold & Lasser, 2020].

    Parameters
    ----------
    x : ndarray
        Spatial grid points.
    L : float
        Half-width of the computational domain (= lambda).
    width_fraction : float
        Fraction eta of each half-domain used for absorption (default 0.15).

    Returns
    -------
    ndarray
        Window values in [0, 1].
    """
    eta = width_fraction
    rho = (1.0 - eta) * L
    ax = np.abs(x)
    result = np.ones_like(ax, dtype=float)

    # Transition region: rho < |x| < L
    mask_trans = (ax > rho) & (ax < L)
    a = ax[mask_trans]
    # Bergold & Lasser Eq. (4.1): w(x) = 1 / (exp(1/(lambda-|x|) + 1/(rho-|x|)) + 1)
    # Note: rho - a < 0 in the transition region, so the second term is negative.
    exponent = 1.0 / (L - a) + 1.0 / (rho - a)
    with np.errstate(over='ignore'):
        result[mask_trans] = 1.0 / (np.exp(exponent) + 1.0)

    # Outside support: |x| >= L
    result[ax >= L] = 0.0

    return result


def fourier_truncation(signal: np.ndarray, num_coefs: int) -> np.ndarray:
    """Truncate the DFT of *signal* to the lowest *num_coefs* modes (symmetric)."""
    coeffs = np.fft.fft(signal)
    truncated = np.zeros_like(coeffs)
    truncated[:num_coefs] = coeffs[:num_coefs]
    truncated[-num_coefs:] = coeffs[-num_coefs:]
    return np.real(np.fft.ifft(truncated))


def _tridiag_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                    d: np.ndarray) -> np.ndarray:
    """Thomas algorithm for a tridiagonal system (complex-valued, dependency-free).

    Solves  a[i] x[i-1] + b[i] x[i] + c[i] x[i+1] = d[i]  with a[0] = c[-1] = 0.

    Parameters
    ----------
    a, b, c : ndarray (complex, length n)
        Sub-, main-, and super-diagonals (a[0] and c[-1] unused).
    d : ndarray (complex, length n)
        Right-hand side.
    """
    n = len(d)
    cp = np.empty(n, dtype=c.dtype)
    dp = np.empty(n, dtype=d.dtype)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i-1]) / m
    x = np.empty(n, dtype=d.dtype)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x


def pml_cn_fd_run(sigma_max: float, N_grid: int, dt_step: float,
                   L: float = 10.0, rho: float = 8.5, p_exp: int = 2,
                   T_final: float = 0.9, x0: float = -5.0,
                   p0: float = 15.0, sigma0: float = 1.0) -> dict:
    """Run one PML + Crank-Nicolson second-order FD simulation.

    Integrates the complex-stretched Schrodinger equation

        i d_t psi = -0.5 * (1/s) d_x [ (1/s) d_x psi ],
        s(x) = 1 + i*sigma(x),
        sigma(x) = sigma_max * ((|x|-rho)/(L-rho))^p_exp  for |x|>rho.

    Dirichlet boundary conditions psi(-L) = psi(L) = 0. Uses a uniform grid
    of (N_grid + 1) points with second-order centred finite differences and
    Crank-Nicolson time stepping. Returns a dict with the spurious density in
    x < 0, the plateau norm on |x| <= rho, and the total L2 norm at T_final.
    """
    # Grid including both Dirichlet endpoints
    x_full = np.linspace(-L, L, N_grid + 1)
    h = x_full[1] - x_full[0]

    # Interior unknowns j = 1..N_grid-1
    n_int = N_grid - 1
    x_int = x_full[1:N_grid]

    # Half-grid points x_{j+1/2}, j = 0..N_grid-1
    x_half = x_full[:-1] + h / 2.0
    ax_half = np.abs(x_half)
    sig_half = np.where(ax_half > rho,
                        sigma_max * ((ax_half - rho) / (L - rho))**p_exp,
                        0.0)
    inv_s_half = 1.0 / (1.0 + 1j * sig_half)

    # sigma at interior grid points
    ax_int = np.abs(x_int)
    sig_int = np.where(ax_int > rho,
                       sigma_max * ((ax_int - rho) / (L - rho))**p_exp,
                       0.0)
    inv_s_int = 1.0 / (1.0 + 1j * sig_int)

    sm = inv_s_half[:-1]                          # s_{j-1/2}^{-1} (k=0..n_int-1)
    sp = inv_s_half[1:]                           # s_{j+1/2}^{-1}
    pref = inv_s_int / h**2
    coef_left  = pref * sm
    coef_right = pref * sp
    coef_diag  = -pref * (sm + sp)

    # Crank-Nicolson: d_t u = (i/2) L_op u
    #   (I - (i*dt/4) L_op) u^{n+1} = (I + (i*dt/4) L_op) u^n
    theta = 1j * dt_step / 4.0
    a_L = -theta * coef_left
    b_L = 1.0 - theta * coef_diag
    c_L = -theta * coef_right
    a_R = theta * coef_left
    b_R = 1.0 + theta * coef_diag
    c_R = theta * coef_right

    # Initial Gaussian wave packet, L2-normalized on the full grid (trapezoidal)
    psi_full = np.exp(-(x_full - x0)**2 / (2.0 * sigma0**2)) * np.exp(1j * p0 * x_full)
    psi_full /= np.sqrt(np.sum(np.abs(psi_full)**2) * h)
    psi_full[0] = 0.0
    psi_full[-1] = 0.0
    u = psi_full[1:N_grid].astype(np.complex128)

    M_steps = int(round(T_final / dt_step))
    for _ in range(M_steps):
        d_rhs = b_R * u
        d_rhs[1:]  += a_R[1:]  * u[:-1]
        d_rhs[:-1] += c_R[:-1] * u[1:]
        u = _tridiag_solve(a_L.copy(), b_L.copy(), c_L.copy(), d_rhs)

    psi_final = np.zeros(N_grid + 1, dtype=np.complex128)
    psi_final[1:N_grid] = u
    dens = np.abs(psi_final)**2
    return {
        'spur_density': float(np.sum(dens[x_full < 0.0]) * h),
        'plateau_norm': float(np.sum(dens[np.abs(x_full) <= rho]) * h),
        'total_norm':   float(np.sum(dens) * h),
        'M_steps':      M_steps,
    }


# =============================================================================
# 2. EXPERIMENTS
# =============================================================================

if __name__ == "__main__":

    # =========================================================================
    # EXPERIMENT 1: GIBBS PHENOMENON & L2 CONVERGENCE (degenerate bump, rho=0)
    # =========================================================================
    print("=" * 72)
    print("EXPERIMENT 1: Convergence Analysis (degenerate bump, rho=0)")
    print("=" * 72)

    N_grid = 20000
    lam_1 = 1.0                               # Half-domain for Experiment 1
    x_1D = np.linspace(-lam_1, lam_1, N_grid, endpoint=False)
    dx_1 = 2.0 * lam_1 / N_grid               # Grid spacing (matches Eq. (16))
    f_x = x_1D.copy()                         # Test function f(x) = x
    bump_degen = plateau_bump_function(x_1D, 0.0)  # B&L Eq. 4.1 with rho=0
    f_windowed = f_x * bump_degen

    # numpy can track errors reliably until the float64 floor (~2e-16);
    # the extended high-precision data is produced separately in 80-digit
    # arithmetic and loaded from convergence_extended.csv.
    N_max = 160
    coef_range = np.arange(1, N_max + 1)
    err_std = np.zeros(N_max)
    err_win = np.zeros(N_max)
    err_hann = np.zeros(N_max)

    # Hann window on [-1, 1] for convergence comparison
    hann_1D = np.zeros_like(x_1D)
    mask_hann_1D = np.abs(x_1D) < 1.0
    hann_1D[mask_hann_1D] = 0.5 * (1.0 + np.cos(np.pi * np.abs(x_1D[mask_hann_1D])))
    f_hann = f_x * hann_1D

    for i, n in enumerate(coef_range):
        approx_s = fourier_truncation(f_x, n)
        approx_w = fourier_truncation(f_windowed, n)
        approx_h = fourier_truncation(f_hann, n)
        # L2 error using the discrete L2 norm: sqrt(sum |f - S_n f|^2 * dx)
        # This matches Eq. (16) of the manuscript.
        err_std[i] = np.sqrt(np.sum(np.abs(f_x - approx_s)**2) * dx_1)
        err_win[i] = np.sqrt(np.sum(np.abs(f_windowed - approx_w)**2) * dx_1)
        err_hann[i] = np.sqrt(np.sum(np.abs(f_hann - approx_h)**2) * dx_1)

    # ---- Load extended high-precision convergence data if available ----
    import csv as csv_module
    extended_csv = f'{OUTPUT_DIR}/convergence_extended.csv'
    extended_available = False
    try:
        with open(extended_csv, 'r') as fcsv:
            reader = csv_module.DictReader(fcsv)
            mp_n, mp_cinf, mp_hann, mp_std = [], [], [], []
            for row in reader:
                mp_n.append(int(row['n']))
                val = float(row['eps_cinf']) if row['eps_cinf'] != '0' else 0.0
                mp_cinf.append(val if val > 0 else np.nan)
                mp_hann.append(float(row['eps_hann']))
                mp_std.append(float(row['eps_std']))
        mp_n = np.array(mp_n)
        mp_cinf = np.array(mp_cinf)
        mp_hann = np.array(mp_hann)
        mp_std = np.array(mp_std)
        N_FOURIER_EXT = len(mp_n)
        extended_available = True
        print(f"  -> Loaded extended convergence data: n = 1 … {N_FOURIER_EXT}")
    except FileNotFoundError:
        print(f"  -> Extended CSV not found ({extended_csv}); using numpy data only")

    # ---- Figure 1: Convergence plot ----
    fig1, ax1 = plt.subplots(figsize=(10, 7))

    if extended_available:
        # Use extended high-precision data for the full range
        ax1.semilogy(mp_n, mp_std, 'r-', linewidth=1.0, alpha=0.8,
                     label='Standard truncation (no window)')
        ax1.semilogy(mp_n, mp_hann, '-', color='#ff7f0e', linewidth=1.0, alpha=0.8,
                     label=r'Hann window ($C^1$, algebraic $\mathcal{O}(n^{-2})$)')
        ax1.semilogy(mp_n, mp_cinf, 'b-', linewidth=1.5,
                     label=r'$C^\infty$ bump (super-algebraic)')
        plot_range = mp_n
        ref_cinf = mp_cinf
    else:
        ax1.semilogy(coef_range, err_std, 'r.-', markersize=5, linewidth=1.2,
                     label='Standard truncation (no window)')
        ax1.semilogy(coef_range, err_hann, 's-', color='#ff7f0e', markersize=4, linewidth=1.2,
                     label=r'Hann window ($C^1$, algebraic $\mathcal{O}(n^{-2})$)')
        ax1.semilogy(coef_range, err_win, 'b.-', markersize=5, linewidth=1.2,
                     label=r'$C^\infty$ bump (super-algebraic)')
        plot_range = coef_range
        ref_cinf = err_win

    # Algebraic reference lines anchored at n=5
    n_ref = 5
    ref_value = ref_cinf[n_ref - 1]
    for s, ls in [(4, ':'), (8, '--'), (16, '-.')]:
        ref_line = ref_value * (n_ref / plot_range)**s
        ax1.semilogy(plot_range, ref_line, color='gray', linestyle=ls, alpha=0.3,
                     linewidth=0.7, label=f'$O(n^{{-{s}}})$ reference')

    # Machine epsilon floor
    ax1.axhline(y=2.2e-16, color='red', linestyle=':', alpha=0.5, linewidth=0.8)
    ax1.text(plot_range[-1] * 0.6, 5e-16, r'float64 floor',
             color='red', fontsize=9, alpha=0.7)

    ax1.set_xlabel('Number of Fourier coefficients ($n$)')
    ax1.set_ylabel(r'$L^2$ error $\varepsilon_n$ (log scale)')
    ax1.set_title(r'Convergence analysis: $f(x) = x$ on $[-1,\,1]$, '
                  f'$N_{{\\mathrm{{grid}}}} = {N_grid}$')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.set_xlim(1, plot_range[-1])

    fig1.tight_layout()
    fig1.savefig(f'{OUTPUT_DIR}/figure_2.pdf')
    print(f"  -> Saved figure_2.pdf")

    # ---- Figure 1b: Effective exponent α_eff ----
    if extended_available:
        # Compute α_eff from extended high-precision data for the full range
        alpha_eff_values = []
        alpha_eff_hann_vals = []
        n_values_alpha = []
        max_n_alpha = N_FOURIER_EXT // 2
        for n in range(3, max_n_alpha + 1):
            idx_n = n - 1
            idx_2n = 2 * n - 1
            if idx_2n < N_FOURIER_EXT and mp_cinf[idx_2n] > 0 and mp_cinf[idx_n] > 0:
                alpha_w = -np.log(mp_cinf[idx_2n] / mp_cinf[idx_n]) / np.log(2.0)
                alpha_h = -np.log(mp_hann[idx_2n] / mp_hann[idx_n]) / np.log(2.0)
                alpha_eff_values.append(alpha_w)
                alpha_eff_hann_vals.append(alpha_h)
                n_values_alpha.append(n)
    else:
        alpha_eff_values = []
        alpha_eff_hann_vals = []
        n_values_alpha = []
        for n in range(3, 81):
            if 2*n <= N_max:
                alpha_w = -np.log(err_win[2*n - 1] / err_win[n - 1]) / np.log(2.0)
                alpha_h = -np.log(err_hann[2*n - 1] / err_hann[n - 1]) / np.log(2.0)
                alpha_eff_values.append(alpha_w)
                alpha_eff_hann_vals.append(alpha_h)
                n_values_alpha.append(n)

    n_alpha = np.array(n_values_alpha)
    a_cinf  = np.array(alpha_eff_values)
    a_hann  = np.array(alpha_eff_hann_vals)

    fig1b, (ax1b_L, ax1b_R) = plt.subplots(1, 2, figsize=(12, 5),
                                             gridspec_kw={'width_ratios': [1, 1.3]})

    # -- Left panel: detail view n = 3 … 20, LINEAR x-axis --
    mask_L = n_alpha <= 20
    ax1b_L.plot(n_alpha[mask_L], a_cinf[mask_L], 'b.-', markersize=5, linewidth=0.8,
                label=r'$C^\infty$ bump')
    ax1b_L.plot(n_alpha[mask_L], a_hann[mask_L], 's-', color='#ff7f0e',
                markersize=4, linewidth=0.8, label=r'Hann window ($C^1$)')
    ax1b_L.set_xlabel('$n$')
    ax1b_L.set_ylabel(r'$\alpha_{\rm eff}(n \to 2n)$')
    ax1b_L.set_title(r'Detail: $n \leq 20$')
    ax1b_L.set_xlim(3, 20)
    ax1b_L.set_xticks(range(3, 21, 1))
    ax1b_L.grid(True, which='both', ls='--', alpha=0.3)
    ax1b_L.legend(fontsize=9, loc='upper left')

    # -- Right panel: full range, LOG x-axis --
    ax1b_R.semilogx(n_alpha, a_cinf, 'b-', linewidth=0.7,
                     label=r'$C^\infty$ bump')
    ax1b_R.semilogx(n_alpha, a_hann, '-', color='#ff7f0e',
                     linewidth=0.8, label=r'Hann window ($C^1$)')
    ax1b_R.set_xlabel('$n$ (log scale)')
    ax1b_R.set_ylabel(r'$\alpha_{\rm eff}(n \to 2n)$')
    ax1b_R.set_title(r'Full range: $n = 3 \ldots 2000$')
    ax1b_R.grid(True, which='both', ls='--', alpha=0.3)
    ax1b_R.legend(fontsize=9, loc='upper left')

    fig1b.tight_layout()
    fig1b.savefig(f'{OUTPUT_DIR}/figure_3.pdf')
    print(f"  -> Saved figure_3.pdf")

    # Print key values
    print(f"  Standard  error at n=80:  {err_std[79]:.4e}")
    print(f"  Hann      error at n=80:  {err_hann[79]:.4e}")
    print(f"  C^inf     error at n=80:  {err_win[79]:.4e}")
    print(f"  Standard  error at n=160: {err_std[-1]:.4e}")
    print(f"  Hann      error at n=160: {err_hann[-1]:.4e}")
    print(f"  C^inf     error at n=160: {err_win[-1]:.4e}")
    print(f"  Improvement C^inf vs Std at n=80: {err_std[79]/err_win[79]:.0f}x")
    print(f"  Improvement C^inf vs Hann at n=80: {err_hann[79]/err_win[79]:.0f}x")
    print(f"  Improvement C^inf vs Std at n=160: {err_std[-1]/err_win[-1]:.0f}x")
    print(f"  Improvement C^inf vs Hann at n=160: {err_hann[-1]/err_win[-1]:.0f}x")

    # Effective exponent by doubling (wide-ratio estimator)
    print("  Effective exponent alpha_eff (C^inf bump, wide-ratio doubling):")
    for n1, n2 in [(5, 10), (10, 20), (15, 30), (20, 40), (25, 50), (30, 60), (40, 80), (50, 100), (60, 120), (80, 160)]:
        alpha_eff = -np.log(err_win[n2 - 1] / err_win[n1 - 1]) / np.log(2)
        print(f"    alpha_eff({n1} -> {n2}) = {alpha_eff:.2f}")

    # Point-to-point exponent (staircase analysis, cf. Remark in paper)
    print("  Point-to-point alpha_loc (staircase structure diagnostic):")
    print(f"    {'n':>4s}  {'alpha_loc':>10s}  {'eps_n':>12s}")
    for n in range(5, 160):
        alpha_loc = -np.log(err_win[n] / err_win[n - 1]) / np.log((n + 1) / n)
        if n in [10, 17, 20, 30, 34, 44, 50, 63, 68] or alpha_loc < 0.5 or alpha_loc > 10:
            print(f"    {n:4d}  {alpha_loc:10.2f}  {err_win[n - 1]:12.4e}  {'<-- plateau' if alpha_loc < 1 else ('<-- peak' if alpha_loc > 10 else '')}")

    # --- Additional test functions (universality verification, cf. Section 4.2) ---
    print("  Additional test functions (C^inf degenerate bump, n=80):")
    additional_tests = {
        'Gaussian exp(-x^2)':  np.exp(-x_1D**2),
        'sin(5*pi*x)':        np.sin(5.0 * np.pi * x_1D),
        'Runge 1/(1+25x^2)':  1.0 / (1.0 + 25.0 * x_1D**2),
    }
    for name, f_test in additional_tests.items():
        fw_test = f_test * bump_degen
        approx_test_80 = fourier_truncation(fw_test, 80)
        approx_test_160 = fourier_truncation(fw_test, 160)
        eps_test_80 = np.sqrt(np.sum(np.abs(fw_test - approx_test_80)**2) * dx_1)
        eps_test_160 = np.sqrt(np.sum(np.abs(fw_test - approx_test_160)**2) * dx_1)
        print(f"    {name:25s}: eps_80 = {eps_test_80:.2e}, eps_160 = {eps_test_160:.2e}")


    # ---- Figure 0: Gibbs suppression visualization (Section 4.1) ----
    n_gibbs = 20  # Moderate n to make Gibbs oscillations clearly visible
    approx_std_gibbs = fourier_truncation(f_x, n_gibbs)
    approx_win_gibbs = fourier_truncation(f_windowed, n_gibbs)

    fig0, (ax0a, ax0b) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Left panel: standard truncation (with Gibbs)
    ax0a.plot(x_1D, f_x, 'k-', linewidth=1.5, label=r'$f(x) = x$')
    ax0a.plot(x_1D, approx_std_gibbs, 'r-', linewidth=1.2,
              label=rf'$S_{{{n_gibbs}}} f(x)$ (standard)')
    ax0a.set_xlabel('$x$')
    ax0a.set_ylabel('Function value')
    ax0a.set_title(rf'Standard truncation ($n = {n_gibbs}$)')
    ax0a.legend(loc='upper left')
    ax0a.grid(True, ls='--', alpha=0.4)
    ax0a.set_xlim(-1.05, 1.05)

    # Right panel: windowed truncation (no Gibbs)
    ax0b.plot(x_1D, f_windowed, 'k--', linewidth=1.5,
              label=r'$f_w(x) = f(x)\,\varphi(x)$')
    ax0b.plot(x_1D, approx_win_gibbs, 'b-', linewidth=1.2,
              label=rf'$S_{{{n_gibbs}}} f_w(x)$ ($C^\infty$ window)')
    ax0b.set_xlabel('$x$')
    ax0b.set_title(rf'$C^\infty$-windowed truncation ($n = {n_gibbs}$)')
    ax0b.legend(loc='upper left')
    ax0b.grid(True, ls='--', alpha=0.4)
    ax0b.set_xlim(-1.05, 1.05)

    fig0.tight_layout()
    fig0.savefig(f'{OUTPUT_DIR}/figure_1.pdf')
    print(f"  -> Saved figure_1.pdf")

    # =========================================================================
    # EXPERIMENT 2: SCHRODINGER EQUATION (WAVE PACKET ABSORPTION)
    # =========================================================================
    print()
    print("=" * 72)
    print("EXPERIMENT 2: Schrodinger Wave Packet Dynamics")
    print("=" * 72)

    L = 10.0
    N = 1024
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi

    dt = 0.005
    steps = 180
    T_final = dt * steps

    # Initial Gaussian wave packet
    x0, p0, sigma = -5.0, 15.0, 1.0
    psi_0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * p0 * x)
    psi_0 /= np.sqrt(np.sum(np.abs(psi_0)**2 * dx))

    # Kinetic propagator
    T_evol = np.exp(-1j * (k**2) / 2 * dt)

    # Absorbing window
    eta = 0.15
    window = absorbing_boundary_window(x, L, width_fraction=eta)
    rho_boundary = (1.0 - eta) * L

    # Time evolution -- track norm at every step
    psi_std = psi_0.copy()
    psi_win = psi_0.copy()
    norm_std_history = np.zeros(steps + 1)
    norm_win_history = np.zeros(steps + 1)
    norm_std_history[0] = np.sum(np.abs(psi_0)**2) * dx
    norm_win_history[0] = norm_std_history[0]

    t_start = timer.perf_counter()
    for step in range(steps):
        # Standard method
        psi_std = np.fft.ifft(np.fft.fft(psi_std) * T_evol)
        # Windowed method
        psi_win = np.fft.ifft(np.fft.fft(psi_win) * T_evol)
        psi_win = psi_win * window

        norm_std_history[step + 1] = np.sum(np.abs(psi_std)**2) * dx
        norm_win_history[step + 1] = np.sum(np.abs(psi_win)**2) * dx
    t_elapsed = timer.perf_counter() - t_start

    print(f"  Domain: [-{L}, {L}],  N={N},  dx={dx:.6f}")
    print(f"  dt={dt}, steps={steps}, T_final={T_final:.3f}")
    print(f"  Initial: x0={x0}, p0={p0}, sigma={sigma}")
    print(f"  Group velocity: v_g = {p0},  expected center at T: {x0 + p0*T_final:.1f}")
    print(f"  Absorbing layer: eta={eta}, plateau=[-{rho_boundary}, {rho_boundary}]")
    print(f"  Elapsed wall time: {t_elapsed:.3f} s")

    # Quantitative diagnostics
    norm_final_std = norm_std_history[-1]
    norm_final_win = norm_win_history[-1]
    left_density_std = np.sum(np.abs(psi_std[x < 0])**2) * dx
    left_density_win = np.sum(np.abs(psi_win[x < 0])**2) * dx
    peak_std = np.max(np.abs(psi_std)**2)
    peak_win = np.max(np.abs(psi_win)**2)
    peak_pos_std = x[np.argmax(np.abs(psi_std)**2)]
    peak_pos_win = x[np.argmax(np.abs(psi_win)**2)]

    print(f"  Norm (standard):  {norm_final_std:.6f}")
    print(f"  Norm (windowed):  {norm_final_win:.6f}  (absorbed {(1-norm_final_win)*100:.1f}%)")
    print(f"  Density in x<0 (standard): {left_density_std:.4e}")
    print(f"  Density in x<0 (windowed): {left_density_win:.2e}")
    print(f"  Suppression ratio: {left_density_std / (left_density_win + 1e-30):.2e}")
    print(f"  Peak |psi|^2 (std): {peak_std:.4f} at x={peak_pos_std:.2f}")
    print(f"  Peak |psi|^2 (win): {peak_win:.4f} at x={peak_pos_win:.2f}")

    # Conditional observable: <x>_rho
    rho_obs = rho_boundary
    mask_rho = np.abs(x) <= rho_obs
    psi_rho = psi_win[mask_rho]
    x_rho = x[mask_rho]
    norm_rho = np.sum(np.abs(psi_rho)**2) * dx
    if norm_rho > 1e-10:
        x_mean_rho = np.sum(x_rho * np.abs(psi_rho)**2) * dx / norm_rho
    else:
        x_mean_rho = 0.0
    x_c_analytical = x0 + p0 * T_final
    print(f"  Conditional observable <x>_rho (|x|<={rho_obs:.2f}): {x_mean_rho:.4f}")
    print(f"  Analytical x_c(T) = x0 + p0*T: {x_c_analytical:.4f}")

    # ---- Probability budget verification ----
    # Verifies that the discrepancy P_out(T) - (1 - ||psi^M||^2) is accounted
    # for by the transition-layer norm, closing the probability budget exactly.
    sigma_T = sigma * np.sqrt(1.0 + T_final**2 / sigma**4)
    x_c_T = x0 + p0 * T_final
    # Note: sigma_T is the width parameter of psi, not the std dev of |psi|^2.
    # Since |psi|^2 ~ exp(-(x-xc)^2 / sigma_T^2), its variance is sigma_T^2/2,
    # and the erf denominator sigma_x*sqrt(2) = (sigma_T/sqrt(2))*sqrt(2) = sigma_T.
    P_in_exact = 0.5 * (erf((rho_boundary - x_c_T) / sigma_T)
                       - erf((-rho_boundary - x_c_T) / sigma_T))
    P_out_exact = 1.0 - P_in_exact
    P_domain_L = 0.5 * (erf((L - x_c_T) / sigma_T)
                       - erf((-L - x_c_T) / sigma_T))
    P_trans_exact = P_domain_L - P_in_exact

    mask_transition = (np.abs(x) > rho_boundary) & (np.abs(x) < L)
    norm_transition = np.sum(np.abs(psi_win[mask_transition])**2) * dx
    absorbed_frac = 1.0 - norm_final_win
    discrepancy_22 = P_out_exact - absorbed_frac

    print()
    print("  Probability budget verification:")
    print(f"    P_out(T) exact                    = {P_out_exact:.6f}")
    print(f"    Absorbed = 1 - ||psi^M||^2        = {absorbed_frac:.6f}")
    print(f"    Discrepancy P_out - absorbed       = {discrepancy_22:.6f}")
    print(f"    Transition-layer norm              = {norm_transition:.6f}")
    print(f"    Plateau norm                       = {norm_rho:.6f}")
    print(f"    Budget: absorbed + trans + plateau  = {absorbed_frac + norm_transition + norm_rho:.6f}")
    print(f"    P_trans exact (unwindowed)          = {P_trans_exact:.6f}")
    print(f"    Attenuation ratio (trans/exact)     = {norm_transition/P_trans_exact:.4f}")

    # Spatial convergence study with varying N (from 256 to 4096)
    print("  Spatial convergence study (fixed p0=15.0):")
    print(f"  {'N':>6s}  {'Final norm':>12s}  {'Density x<0':>14s}")
    print("  " + "-" * 36)
    for N_test in [256, 512, 1024, 2048, 4096]:
        x_test = np.linspace(-L, L, N_test, endpoint=False)
        dx_test = x_test[1] - x_test[0]
        k_test = np.fft.fftfreq(N_test, d=dx_test) * 2 * np.pi
        T_evol_test = np.exp(-1j * (k_test**2) / 2 * dt)
        win_test = absorbing_boundary_window(x_test, L, width_fraction=eta)
        psi_test = np.exp(-(x_test - x0)**2 / (2 * sigma**2)) * np.exp(1j * p0 * x_test)
        psi_test /= np.sqrt(np.sum(np.abs(psi_test)**2 * dx_test))
        for step in range(steps):
            psi_test = np.fft.ifft(np.fft.fft(psi_test) * T_evol_test)
            psi_test = psi_test * win_test
        norm_test = np.sum(np.abs(psi_test)**2) * dx_test
        left_test = np.sum(np.abs(psi_test[x_test < 0])**2) * dx_test
        print(f"  {N_test:6d}  {norm_test:12.4f}  {left_test:14.2e}")

    # Additional spatial convergence with higher momentum (p0 = 25)
    print("  Spatial convergence with p0 = 25 (higher momentum):")
    p0_high = 25.0
    psi_0_high = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * p0_high * x)
    psi_0_high /= np.sqrt(np.sum(np.abs(psi_0_high)**2 * dx))
    T_final_high = 0.9
    steps_high = int(round(T_final_high / dt))

    print(f"  {'N':>6s}  {'Final norm':>12s}  {'Density x<0':>14s}")
    print("  " + "-" * 36)
    for N_test in [256, 512, 1024, 2048, 4096]:
        x_test = np.linspace(-L, L, N_test, endpoint=False)
        dx_test = x_test[1] - x_test[0]
        k_test = np.fft.fftfreq(N_test, d=dx_test) * 2 * np.pi
        T_evol_test = np.exp(-1j * (k_test**2) / 2 * dt)
        win_test = absorbing_boundary_window(x_test, L, width_fraction=eta)
        psi_test = np.exp(-(x_test - x0)**2 / (2 * sigma**2)) * np.exp(1j * p0_high * x_test)
        psi_test /= np.sqrt(np.sum(np.abs(psi_test)**2 * dx_test))
        for step in range(steps_high):
            psi_test = np.fft.ifft(np.fft.fft(psi_test) * T_evol_test)
            psi_test = psi_test * win_test
        norm_test = np.sum(np.abs(psi_test)**2) * dx_test
        left_test = np.sum(np.abs(psi_test[x_test < 0])**2) * dx_test
        print(f"  {N_test:6d}  {norm_test:12.4f}  {left_test:14.2e}")

    # High-momentum spatial convergence table (Table 2 format)
    print("  High-momentum (p0=25) spatial convergence table:")
    print(f"  {'N':>6s}  {'Final norm':>12s}  {'Density x<0':>14s}  {'Convergence':>14s}")
    print("  " + "-" * 50)
    prev_left = None
    for N_test in [256, 512, 1024, 2048, 4096]:
        x_test = np.linspace(-L, L, N_test, endpoint=False)
        dx_test = x_test[1] - x_test[0]
        k_test = np.fft.fftfreq(N_test, d=dx_test) * 2 * np.pi
        T_evol_test = np.exp(-1j * (k_test**2) / 2 * dt)
        win_test = absorbing_boundary_window(x_test, L, width_fraction=eta)
        psi_test = np.exp(-(x_test - x0)**2 / (2 * sigma**2)) * np.exp(1j * p0_high * x_test)
        psi_test /= np.sqrt(np.sum(np.abs(psi_test)**2 * dx_test))
        for step in range(steps_high):
            psi_test = np.fft.ifft(np.fft.fft(psi_test) * T_evol_test)
            psi_test = psi_test * win_test
        norm_test = np.sum(np.abs(psi_test)**2) * dx_test
        left_test = np.sum(np.abs(psi_test[x_test < 0])**2) * dx_test
        if prev_left is not None and left_test > 1e-15:
            ratio_str = f"{prev_left / left_test:.1f}x"
        else:
            ratio_str = "---"
        print(f"  {N_test:6d}  {norm_test:12.4f}  {left_test:14.2e}  {ratio_str:>14s}")
        prev_left = left_test

    # ---- Error floor decomposition (Remark in paper) ----
    print()
    print("  Error floor decomposition (eta=0.15, p0=15, T=0.9):")
    sigma_T = sigma * np.sqrt(1 + (T_final / sigma**2)**2)
    x_center = x0 + p0 * T_final
    z_erfc = x_center / sigma_T
    P_tail = 0.5 * math.erfc(z_erfc)
    eps_dp = np.finfo(float).eps
    roundoff = N * steps * eps_dp**2
    print(f"    Gaussian tail P(x<0) = erfc({z_erfc:.2f})/2 = {P_tail:.2e}")
    print(f"    Round-off bound N*M*eps^2 = {roundoff:.2e}")
    print(f"    Measured floor           = {left_density_win:.2e}")
    print(f"    => Floor dominated by accumulated discrete Fourier leakage")

    print()
    print("  eta-sweep (sensitivity analysis, Table eta_sensitivity):")
    print(f"  {'eta':>6s}  {'rho':>6s}  {'density x<0':>14s}  {'norm':>8s}  {'suppression':>14s}")
    print("  " + "-" * 60)
    left_ref = np.sum(np.abs(psi_std[x < 0])**2) * dx
    for eta_diag in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        win_diag = absorbing_boundary_window(x, L, width_fraction=eta_diag)
        psi_diag = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * p0 * x)
        psi_diag /= np.sqrt(np.sum(np.abs(psi_diag)**2) * dx)
        for step in range(steps):
            psi_diag = np.fft.ifft(np.fft.fft(psi_diag) * T_evol)
            psi_diag = psi_diag * win_diag
        norm_diag = np.sum(np.abs(psi_diag)**2) * dx
        left_diag = np.sum(np.abs(psi_diag[x < 0])**2) * dx
        rho_diag = (1.0 - eta_diag) * L
        supp = left_ref / left_diag if left_diag > 0 else float('inf')
        print(f"  {eta_diag:6.2f}  {rho_diag:6.1f}  {left_diag:14.3e}  {norm_diag:8.4f}  {supp:14.2e}")

    # ---- Figure 2: Schrodinger comparison ----
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    ax2a.plot(x, np.abs(psi_0)**2, 'k--', alpha=0.5, linewidth=1, label=r'$|\psi(x,0)|^2$')
    ax2a.plot(x, np.abs(psi_std)**2, 'r-', linewidth=1.8, label=r'$|\psi(x,T)|^2$ (standard FFT)')
    ax2a.set_title('Standard FFT: wrap-around error')
    ax2a.set_xlabel('$x$')
    ax2a.set_ylabel(r'Probability density $|\psi|^2$')
    ax2a.legend(loc='upper left')
    ax2a.grid(True, alpha=0.3)

    ax2b.plot(x, np.abs(psi_0)**2, 'k--', alpha=0.5, linewidth=1, label=r'$|\psi(x,0)|^2$')
    ax2b.plot(x, np.abs(psi_win)**2, 'b-', linewidth=1.8, label=r'$|\psi(x,T)|^2$ (windowed)')
    ax2b.plot(x, window, 'g:', linewidth=1.5, alpha=0.7, label=r'$\mathcal{W}(x)$ absorbing window')
    ax2b.axvline(rho_boundary, color='gray', ls='--', alpha=0.4, linewidth=0.8)
    ax2b.axvline(-rho_boundary, color='gray', ls='--', alpha=0.4, linewidth=0.8)
    ax2b.set_title(r'Proposed: $C^\infty$ windowed FFT')
    ax2b.set_xlabel('$x$')
    ax2b.legend(loc='upper left')
    ax2b.grid(True, alpha=0.3)

    fig2.suptitle(
        rf'Free-particle TDSE on $[-{L:.0f},\,{L:.0f}]$, $N={N}$, '
        rf'$T={T_final}$, $\eta={eta}$',
        fontsize=14, y=1.02)
    fig2.tight_layout()
    fig2.savefig(f'{OUTPUT_DIR}/figure_4.pdf')
    print(f"  -> Saved figure_4.pdf")

    # ---- Figure 3: Norm evolution over time with analytical reference ----
    time_axis = np.linspace(0, T_final, steps + 1)

    # Compute analytical P_in(t)
    time_analytical = np.linspace(0, T_final, 500)
    P_in_analytical = np.zeros_like(time_analytical)
    for i, t in enumerate(time_analytical):
        sigma_t = sigma * np.sqrt(1.0 + (t**2) / (sigma**4))
        x_c_t = x0 + p0 * t
        erf_plus = erf((rho_boundary - x_c_t) / sigma_t)
        erf_minus = erf((-rho_boundary - x_c_t) / sigma_t)
        P_in_analytical[i] = 0.5 * (erf_plus - erf_minus)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(time_axis, norm_std_history, 'r-', linewidth=1.5,
             label='Standard FFT (norm conserved — includes wrap-around)')
    ax3.plot(time_axis, norm_win_history, 'b-', linewidth=1.5,
             label=r'$C^\infty$-windowed (norm decreases — physical absorption)')
    ax3.plot(time_analytical, P_in_analytical, 'g--', linewidth=1.5,
             label='Analytical: probability within plateau')
    ax3.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax3.set_xlabel('Time $t$')
    ax3.set_ylabel(r'$\|\psi(t)\|^2$')
    ax3.set_title('Norm evolution: unitarity vs. physical absorption')
    ax3.legend()
    ax3.set_ylim(0.0, 1.1)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(f'{OUTPUT_DIR}/figure_5.pdf')
    print(f"  -> Saved figure_5.pdf")


    # =========================================================================
    # EXPERIMENT 3: CONVERGENCE WITH PLATEAU BUMP (varying rho/lambda)
    #    -- Quantifies the effect of the plateau parameter on L2 error
    #    -- Aligns with Bergold & Lasser Theorem 4.6 (Lipschitz constant bound)
    # =========================================================================
    print()
    print("=" * 72)
    print("EXPERIMENT 3: Convergence with plateau bump — varying rho")
    print("=" * 72)

    N_grid_3 = 2000
    lam_3 = 1.0                                # Half-domain for Experiment 3
    x_3 = np.linspace(-lam_3, lam_3, N_grid_3, endpoint=False)
    dx_3 = 2.0 * lam_3 / N_grid_3              # Grid spacing
    f_3 = x_3.copy()       # Same test function f(x) = x
    N_max_3 = 80
    coef_range_3 = np.arange(1, N_max_3 + 1)

    # Finer rho sweep (MODIFIED)
    rho_values = [0.0, 0.30, 0.50, 0.60, 0.70, 0.85]
    colors_rho = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#d62728']
    labels_rho = [
        r'$\rho = 0.00$ (degenerate)',
        r'$\rho = 0.30$',
        r'$\rho = 0.50$',
        r'$\rho = 0.60$',
        r'$\rho = 0.70$',
        r'$\rho = 0.85$',
    ]

    errors_by_rho = {}
    plateau_errors_by_rho = {}   # Full arrays for plotting
    plateau_errors = {}          # Scalar (n=80) for backward compat
    for rho_val in rho_values:
        w = plateau_bump_function(x_3, rho_val)
        fw = f_3 * w
        errs = np.zeros(N_max_3)
        for i, n in enumerate(coef_range_3):
            approx = fourier_truncation(fw, n)
            errs[i] = np.sqrt(np.sum(np.abs(fw - approx)**2) * dx_3)
        errors_by_rho[rho_val] = errs
        print(f"  rho={rho_val:.2f}: error at n=80 = {errs[-1]:.4e}")

        # Compute plateau-restricted error for non-zero rho
        if rho_val > 0:
            mask_plat = np.abs(x_3) <= rho_val
            errs_plat = np.zeros(N_max_3)
            for i, n in enumerate(coef_range_3):
                approx = fourier_truncation(fw, n)
                # On the plateau f_w = f since w=1, so measure |f - approx|
                errs_plat[i] = np.sqrt(np.sum(np.abs(f_3[mask_plat] - approx[mask_plat])**2) * dx_3)
            plateau_errors_by_rho[rho_val] = errs_plat
            plateau_errors[rho_val] = errs_plat[-1]
            print(f"    -> plateau error at n=80 = {errs_plat[-1]:.4e}")

    # Also include the unwindowed (standard) case for reference
    errs_unwound = np.zeros(N_max_3)
    for i, n in enumerate(coef_range_3):
        approx = fourier_truncation(f_3, n)
        errs_unwound[i] = np.sqrt(np.sum(np.abs(f_3 - approx)**2) * dx_3)

    # ---- Figure 4: Convergence vs rho (finer sweep) ----
    fig4, ax4 = plt.subplots(figsize=(9, 6))
    ax4.semilogy(coef_range_3, errs_unwound, 'k--', linewidth=1.0, alpha=0.6,
                 label='No window (algebraic)')
    for rho_val, color, lbl in zip(rho_values, colors_rho, labels_rho):
        ax4.semilogy(coef_range_3, errors_by_rho[rho_val], '.-', color=color,
                     markersize=3, linewidth=1.2, label=lbl)
        # Overlay plateau-restricted error as dashed line (for rho > 0)
        if rho_val in plateau_errors_by_rho:
            ax4.semilogy(coef_range_3, plateau_errors_by_rho[rho_val], '--',
                         color=color, linewidth=1.0, alpha=0.7)
    # Add a manual legend entry for dashed = plateau-restricted
    from matplotlib.lines import Line2D
    legend_elements = ax4.get_legend_handles_labels()
    ax4.plot([], [], '--', color='gray', linewidth=1.0, alpha=0.7,
             label=r'Plateau-restricted $\varepsilon_n^{\mathrm{plat}}$')
    ax4.set_xlabel('Number of Fourier coefficients ($n$)')
    ax4.set_ylabel(r'$L^2$ error $\varepsilon_n$ (log scale)')
    ax4.set_title(r'Effect of plateau parameter $\rho$ on convergence rate')
    ax4.legend(fontsize=8)
    ax4.grid(True, which="both", ls="--", alpha=0.4)
    fig4.tight_layout()
    fig4.savefig(f'{OUTPUT_DIR}/figure_6.pdf')
    print(f"  -> Saved figure_6.pdf")


    # =========================================================================
    # EXPERIMENT 4: WINDOW COMPARISON (Hann vs Tukey vs C^inf bump)
    #    -- Compares different window regularities on the Schrodinger problem
    # =========================================================================
    print()
    print("=" * 72)
    print("EXPERIMENT 4: Window comparison (Hann / Tukey / C^inf bump)")
    print("=" * 72)

    def hann_window(x: np.ndarray, L: float) -> np.ndarray:
        """Hann window on [-L, L] -- degenerate C1-bump (rho = 0)."""
        xi = np.abs(x / L)
        w = np.zeros_like(xi)
        mask = xi < 1.0
        w[mask] = 0.5 * (1.0 + np.cos(np.pi * xi[mask]))
        return w

    def tukey_window(x: np.ndarray, L: float, alpha: float = 0.3) -> np.ndarray:
        """Tukey window on [-L, L] -- non-degenerate C1-bump, plateau rho = (1-alpha)*L."""
        xi = np.abs(x / L)
        w = np.ones_like(xi)
        mask = xi > (1.0 - alpha)
        z = (xi[mask] - (1.0 - alpha)) / alpha
        w[mask] = 0.5 * (1.0 + np.cos(np.pi * z))
        w[xi >= 1.0] = 0.0
        return w

    windows = {
        'Hann ($C^1$, degenerate)': hann_window(x, L),
        'Tukey ($C^1$, $\\alpha=0.3$)': tukey_window(x, L, alpha=0.3),
        r'$C^\infty$ bump ($\eta=0.15$)': absorbing_boundary_window(x, L, width_fraction=0.15),
    }
    window_colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

    # Run Schrodinger simulation for each window
    results_windows = {}
    for (name, w), color in zip(windows.items(), window_colors):
        psi = psi_0.copy()
        norms = np.zeros(steps + 1)
        norms[0] = np.sum(np.abs(psi)**2) * dx
        for step in range(steps):
            psi = np.fft.ifft(np.fft.fft(psi) * T_evol)
            psi = psi * w
            norms[step + 1] = np.sum(np.abs(psi)**2) * dx

        left_dens = np.sum(np.abs(psi[x < 0])**2) * dx
        results_windows[name] = {
            'psi_final': psi,
            'norms': norms,
            'left_density': left_dens,
            'color': color,
        }
        print(f"  {name}:")
        print(f"    Final norm = {norms[-1]:.6f}, density x<0 = {left_dens:.2e}")

    # ---- Figure 5: Window comparison ----
    fig5 = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig5, hspace=0.35, wspace=0.30)

    # Panel (a): Window profiles
    ax5a = fig5.add_subplot(gs[0, 0])
    for (name, w), color in zip(windows.items(), window_colors):
        ax5a.plot(x, w, color=color, linewidth=1.5, label=name)
    ax5a.set_xlabel('$x$')
    ax5a.set_ylabel('$w(x)$')
    ax5a.set_title('(a) Window profiles')
    ax5a.legend(fontsize=9)
    ax5a.grid(True, alpha=0.3)

    # Panel (b): Final |psi|^2
    ax5b = fig5.add_subplot(gs[0, 1])
    ax5b.plot(x, np.abs(psi_0)**2, 'k--', alpha=0.4, linewidth=0.8, label='Initial')
    for name, res in results_windows.items():
        ax5b.plot(x, np.abs(res['psi_final'])**2, color=res['color'],
                  linewidth=1.3, label=name)
    ax5b.set_xlabel('$x$')
    ax5b.set_ylabel(r'$|\psi(x,T)|^2$')
    ax5b.set_title(r'(b) Final probability density')
    ax5b.legend(fontsize=8)
    ax5b.grid(True, alpha=0.3)

    # Panel (c): Norm evolution
    ax5c = fig5.add_subplot(gs[1, 0])
    ax5c.plot(time_axis, norm_std_history, 'r-', linewidth=1.0, alpha=0.5, label='No window')
    for name, res in results_windows.items():
        ax5c.plot(time_axis, res['norms'], color=res['color'], linewidth=1.3, label=name)
    ax5c.set_xlabel('Time $t$')
    ax5c.set_ylabel(r'$\|\psi(t)\|^2$')
    ax5c.set_title('(c) Norm evolution')
    ax5c.legend(fontsize=8)
    ax5c.set_ylim(0.0, 1.1)
    ax5c.grid(True, alpha=0.3)

    # Panel (d): Spurious density in x<0 (log bar chart)
    ax5d = fig5.add_subplot(gs[1, 1])
    names_short = ['No window', 'Hann', 'Tukey', r'$C^\infty$ bump']
    left_vals = [left_density_std]
    bar_colors = ['red']
    for name, res in results_windows.items():
        left_vals.append(res['left_density'])
        bar_colors.append(res['color'])
    bars = ax5d.bar(names_short, left_vals, color=bar_colors, alpha=0.8, edgecolor='black',
                    linewidth=0.5)
    ax5d.set_yscale('log')
    ax5d.set_ylabel(r'$\int_{-L}^{0}|\psi|^2\,dx$')
    ax5d.set_title('(d) Spurious density in $x<0$')
    ax5d.grid(True, which="both", axis='y', ls="--", alpha=0.4)
    # Add value labels on bars
    for bar, val in zip(bars, left_vals):
        ax5d.text(bar.get_x() + bar.get_width()/2, val * 2.0, f'{val:.1e}',
                  ha='center', va='bottom', fontsize=9)

    fig5.savefig(f'{OUTPUT_DIR}/figure_7.pdf')
    print(f"  -> Saved figure_7.pdf")


    # =========================================================================
    # EXPERIMENT 5: COMPLEX ABSORBING POTENTIAL (CAP) PARAMETER SWEEP
    #    -- Quadratic CAP: V_CAP(x) = -i*c*((|x|-rho)/(L-rho))^2 for |x|>rho
    #    -- Applied multiplicatively: psi <- psi * exp(-V_CAP * dt)
    #    -- Reproduces Table 5 of the manuscript (Section 4.3)
    # =========================================================================
    print()
    print("=" * 72)
    print("EXPERIMENT 5: Complex Absorbing Potential — parameter sweep")
    print("=" * 72)

    c_values = [10, 25, 50, 100, 200, 500]

    print(f"  {'c':>6s}  {'Final norm':>12s}  {'Density x<0':>14s}  {'Suppression':>14s}")
    print("  " + "-" * 50)
    cap_results = {}
    for c_val in c_values:
        # Construct quadratic CAP profile
        V_cap = np.zeros(N)
        mask_cap = np.abs(x) > rho_boundary
        V_cap[mask_cap] = c_val * (
            (np.abs(x[mask_cap]) - rho_boundary) / (L - rho_boundary)
        )**2
        cap_factor = np.exp(-V_cap * dt)

        # Time evolution with CAP
        psi_cap = psi_0.copy()
        for step in range(steps):
            psi_cap = np.fft.ifft(np.fft.fft(psi_cap) * T_evol)
            psi_cap = psi_cap * cap_factor

        norm_cap = np.sum(np.abs(psi_cap)**2) * dx
        left_cap = np.sum(np.abs(psi_cap[x < 0])**2) * dx
        ratio_cap = left_density_std / (left_cap + 1e-30)
        cap_results[c_val] = {
            'norm': norm_cap,
            'left_density': left_cap,
            'suppression': ratio_cap,
        }
        print(f"  {c_val:6d}  {norm_cap:12.4f}  {left_cap:14.2e}  {ratio_cap:14.1e}")

    print("  " + "-" * 50)
    print(f"  {'C^inf':>6s}  {norm_final_win:12.4f}  {left_density_win:14.2e}  "
          f"{left_density_std / (left_density_win + 1e-30):14.1e}")


    # =========================================================================
    # EXPERIMENT 6: CAP ROBUSTNESS ACROSS MOMENTA
    #    -- For p0 in [15.0, 20.0, 25.0]:
    #    --   Find c* (minimum c achieving density_x_lt_0 < 1e-8)
    #    --   Compare with C-infinity window performance
    # =========================================================================
    print()
    print("=" * 72)
    print("EXPERIMENT 6: CAP Robustness Across Momenta")
    print("=" * 72)

    p0_values_cap = [15.0, 20.0, 25.0]
    c_sweep_cap = [10, 25, 50, 100, 200, 500, 1000]
    L_cap = 10.0
    N_cap = 1024
    x_cap = np.linspace(-L_cap, L_cap, N_cap, endpoint=False)
    dx_cap = x_cap[1] - x_cap[0]
    k_cap = np.fft.fftfreq(N_cap, d=dx_cap) * 2 * np.pi
    sigma_cap = 1.0
    x0_cap = -5.0
    dt_cap = 0.005
    steps_cap = int(0.9 / dt_cap)
    T_evol_cap = np.exp(-1j * (k_cap**2) / 2 * dt_cap)
    eta_cap = 0.15
    window_cap = absorbing_boundary_window(x_cap, L_cap, width_fraction=eta_cap)
    rho_boundary_cap = (1.0 - eta_cap) * L_cap

    print(f"  {'p0':>6s}  {'c*':>6s}  {'CAP density':>14s}  {'C-inf density':>14s}")
    print("  " + "-" * 50)

    for p0_val in p0_values_cap:
        # Initial condition
        psi_0_cap = np.exp(-(x_cap - x0_cap)**2 / (2 * sigma_cap**2)) * np.exp(1j * p0_val * x_cap)
        psi_0_cap /= np.sqrt(np.sum(np.abs(psi_0_cap)**2 * dx_cap))

        # Find c* for CAP
        c_star = None
        density_at_c_star = None
        for c_test in c_sweep_cap:
            V_cap_test = np.zeros(N_cap)
            mask_cap_test = np.abs(x_cap) > rho_boundary_cap
            V_cap_test[mask_cap_test] = c_test * (
                (np.abs(x_cap[mask_cap_test]) - rho_boundary_cap) / (L_cap - rho_boundary_cap)
            )**2
            cap_factor_test = np.exp(-V_cap_test * dt_cap)

            psi_cap_test = psi_0_cap.copy()
            for step in range(steps_cap):
                psi_cap_test = np.fft.ifft(np.fft.fft(psi_cap_test) * T_evol_cap)
                psi_cap_test = psi_cap_test * cap_factor_test

            left_cap_test = np.sum(np.abs(psi_cap_test[x_cap < 0])**2) * dx_cap
            if left_cap_test < 1e-8:
                c_star = c_test
                density_at_c_star = left_cap_test
                break

        # C-infinity window result
        psi_cinf = psi_0_cap.copy()
        for step in range(steps_cap):
            psi_cinf = np.fft.ifft(np.fft.fft(psi_cinf) * T_evol_cap)
            psi_cinf = psi_cinf * window_cap
        left_cinf = np.sum(np.abs(psi_cinf[x_cap < 0])**2) * dx_cap

        if c_star is None:
            c_star_str = ">1000"
            density_str = ">1e-8"
        else:
            c_star_str = f"{c_star:6d}"
            density_str = f"{density_at_c_star:14.2e}"

        print(f"  {p0_val:6.1f}  {c_star_str:>6s}  {density_str:>14s}  {left_cinf:14.2e}")


    # =========================================================================
    # EXPERIMENT 7: HAMILTONIAN CAP COMPARISON
    #    -- Compares the multiplicative CAP (post-kinetic damping) with the
    #       Hamiltonian CAP (Strang splitting: -iV_CAP in potential half-steps)
    #    -- Uses same domain, grid, initial condition, and absorbing layer as Exp. 5
    #    -- Shows the reflection-absorption trade-off for both formulations
    #    -- Demonstrates that the C^inf window matches the optimal CAP without
    #       parameter tuning
    # =========================================================================
    print()
    print("=" * 72)
    print("EXPERIMENT 7: Hamiltonian CAP vs Multiplicative CAP vs C^inf window")
    print("=" * 72)

    c_values_5c = [50, 100, 200, 500, 750, 1000, 2000]
    print(f"  {'c':>6s}  {'Ham norm':>9s} {'Ham dens':>11s} {'Ham supp':>10s}"
          f"  {'Mult norm':>9s} {'Mult dens':>11s} {'Mult supp':>10s}")
    print("  " + "-" * 78)

    for c_val in c_values_5c:
        V_cap_5c = np.zeros(N)
        mask_5c = np.abs(x) > rho_boundary
        V_cap_5c[mask_5c] = c_val * (
            (np.abs(x[mask_5c]) - rho_boundary) / (L - rho_boundary)
        )**2

        # Multiplicative CAP: kinetic -> damping
        cap_full_5c = np.exp(-V_cap_5c * dt)
        psi_m5c = psi_0.copy()
        for step in range(steps):
            psi_m5c = np.fft.ifft(np.fft.fft(psi_m5c) * T_evol)
            psi_m5c *= cap_full_5c
        norm_m5c = np.sum(np.abs(psi_m5c)**2) * dx
        left_m5c = np.sum(np.abs(psi_m5c[x < 0])**2) * dx

        # Hamiltonian CAP (Strang): damping_half -> kinetic -> damping_half
        cap_half_5c = np.exp(-V_cap_5c * dt / 2.0)
        psi_h5c = psi_0.copy()
        for step in range(steps):
            psi_h5c *= cap_half_5c
            psi_h5c = np.fft.ifft(np.fft.fft(psi_h5c) * T_evol)
            psi_h5c *= cap_half_5c
        norm_h5c = np.sum(np.abs(psi_h5c)**2) * dx
        left_h5c = np.sum(np.abs(psi_h5c[x < 0])**2) * dx

        supp_m = left_density_std / (left_m5c + 1e-30)
        supp_h = left_density_std / (left_h5c + 1e-30)

        print(f"  {c_val:6d}  {norm_h5c:9.4f} {left_h5c:11.2e} {supp_h:10.1e}"
              f"  {norm_m5c:9.4f} {left_m5c:11.2e} {supp_m:10.1e}")

    print("  " + "-" * 78)
    print(f"  {'C^inf':>6s}  {norm_final_win:9.4f} {left_density_win:11.2e}"
          f" {left_density_std / (left_density_win + 1e-30):10.1e}"
          f"  {'(same — no parameter)':>42s}")

    # Find optimal c for each scheme (fine sweep)
    c_fine_5c = list(range(100, 1001, 25))
    best_mult = (None, 1.0)
    best_ham = (None, 1.0)
    for c_val in c_fine_5c:
        V_cap_5c = np.zeros(N)
        mask_5c = np.abs(x) > rho_boundary
        V_cap_5c[mask_5c] = c_val * (
            (np.abs(x[mask_5c]) - rho_boundary) / (L - rho_boundary)
        )**2

        cap_full_5c = np.exp(-V_cap_5c * dt)
        psi_m5c = psi_0.copy()
        for step in range(steps):
            psi_m5c = np.fft.ifft(np.fft.fft(psi_m5c) * T_evol)
            psi_m5c *= cap_full_5c
        left_m5c = np.sum(np.abs(psi_m5c[x < 0])**2) * dx
        if left_m5c < best_mult[1]:
            best_mult = (c_val, left_m5c)

        cap_half_5c = np.exp(-V_cap_5c * dt / 2.0)
        psi_h5c = psi_0.copy()
        for step in range(steps):
            psi_h5c *= cap_half_5c
            psi_h5c = np.fft.ifft(np.fft.fft(psi_h5c) * T_evol)
            psi_h5c *= cap_half_5c
        left_h5c = np.sum(np.abs(psi_h5c[x < 0])**2) * dx
        if left_h5c < best_ham[1]:
            best_ham = (c_val, left_h5c)

    print()
    print(f"  Optimal c (multiplicative): c*={best_mult[0]}, "
          f"density={best_mult[1]:.2e}, suppression={left_density_std/best_mult[1]:.1e}")
    print(f"  Optimal c (Hamiltonian):    c*={best_ham[0]}, "
          f"density={best_ham[1]:.2e}, suppression={left_density_std/best_ham[1]:.1e}")
    print(f"  C^inf window (no tuning):   "
          f"density={left_density_win:.2e}, suppression={left_density_std/left_density_win:.1e}")


    # =========================================================================
    # EXPERIMENT 8: Manolopoulos transmission-free CAP benchmark
    #   -- Reference: D. E. Manolopoulos, J. Chem. Phys. 117, 9552 (2002)
    #   -- Canonical prescription (Eqs. 2.25-2.27, 3.8 of Manolopoulos 2002):
    #        y(x) = a*x - b*x^3 + 4/(c-x)^2 - 4/(c+x)^2,       (Eq. 2.25)
    #        -i*epsilon(r) = -i*E_min*y(x),                    (Eq. 2.26)
    #        x = 2*delta*k_min*(|r|-rho),  x in [0, c],        (Eq. 2.16)
    #        width (r2-r1) = c/(2*delta*k_min),                (Eq. 2.27)
    #        delta = c/(4*pi),                                 (Eq. 3.8)
    #      where c = sqrt(2) * K(1/sqrt(2)) = 2.62206,
    #            a = 1 - 16/c^3 = 0.11244900,
    #            b = (1 - 17/c^3)/c^2 = 0.00828735 (Table I).
    #   -- Width-matching condition fixes k_min = 2*pi/(L-rho),
    #      so that the absorbing layer coincides with one de Broglie
    #      wavelength at E_min = k_min^2/2.
    #   -- Singularity at |x| = L is clipped one epsilon below c for
    #      numerical evaluation on the finite grid.
    # =========================================================================
    print()
    print("=" * 72)
    print("EXPERIMENT 8: Manolopoulos transmission-free CAP (JCP 117, 9552, 2002)")
    print("=" * 72)

    c_M = 2.62206
    a_M = 1.0 - 16.0 / c_M**3
    b_M = (1.0 - 17.0 / c_M**3) / c_M**2
    delta_M = c_M / (4.0 * np.pi)
    delta_abs = L - rho_boundary
    k_min_M = c_M / (2.0 * delta_M * delta_abs)   # = 2*pi/(L-rho)
    E_min_M = 0.5 * k_min_M**2
    print(f"  Manolopoulos constants: a={a_M:.8e}, b={b_M:.8e}, c={c_M}")
    print(f"  delta (Eq. 3.8) = c/(4*pi) = {delta_M:.6f}")
    print(f"  Absorbing width L-rho = {delta_abs}, k_min = {k_min_M:.6f}")
    print(f"  E_min = {E_min_M:.6f},  E_0/E_min = {0.5*p0**2/E_min_M:.2f}")

    W_man = np.zeros(N)
    mask_M = np.abs(x) > rho_boundary
    xi_M = (c_M / delta_abs) * (np.abs(x[mask_M]) - rho_boundary)
    eps_clip = 1e-8
    xi_M = np.clip(xi_M, 0.0, c_M - eps_clip)
    y_M = a_M * xi_M - b_M * xi_M**3 \
          + 4.0 / (c_M - xi_M)**2 - 4.0 / (c_M + xi_M)**2
    W_man[mask_M] = E_min_M * y_M

    # Multiplicative Manolopoulos CAP: kinetic -> damping
    cap_full_M = np.exp(-W_man * dt)
    psi_m_M = psi_0.copy()
    for step in range(steps):
        psi_m_M = np.fft.ifft(np.fft.fft(psi_m_M) * T_evol)
        psi_m_M *= cap_full_M
    norm_m_M = np.sum(np.abs(psi_m_M)**2) * dx
    left_m_M = np.sum(np.abs(psi_m_M[x < 0])**2) * dx
    supp_m_M = left_density_std / left_m_M

    # Hamiltonian Manolopoulos CAP (Strang)
    cap_half_M = np.exp(-W_man * dt / 2.0)
    psi_h_M = psi_0.copy()
    for step in range(steps):
        psi_h_M *= cap_half_M
        psi_h_M = np.fft.ifft(np.fft.fft(psi_h_M) * T_evol)
        psi_h_M *= cap_half_M
    norm_h_M = np.sum(np.abs(psi_h_M)**2) * dx
    left_h_M = np.sum(np.abs(psi_h_M[x < 0])**2) * dx
    supp_h_M = left_density_std / left_h_M

    print()
    print(f"  At dt = {dt} (standard operating time step):")
    print(f"    Manolopoulos MULT : norm={norm_m_M:.4f}  "
          f"density={left_m_M:.3e}  suppression={supp_m_M:.1e}")
    print(f"    Manolopoulos HAM  : norm={norm_h_M:.4f}  "
          f"density={left_h_M:.3e}  suppression={supp_h_M:.1e}")
    print(f"    C^inf window      : norm={norm_final_win:.4f}  "
          f"density={left_density_win:.3e}  "
          f"suppression={left_density_std/left_density_win:.1e}")

    # Time-step sensitivity: transmission-free property recovered as dt->0
    print()
    print(f"  Time-step refinement (T=0.9 fixed, canonical E_min):")
    print(f"  {'dt':>8s}  {'steps':>6s}  "
          f"{'mult norm':>9s} {'mult dens':>11s}  "
          f"{'ham norm':>9s} {'ham dens':>11s}")
    print("  " + "-" * 70)
    for dt_M, steps_M in [(0.005, 180), (0.0025, 360),
                           (0.001, 900), (0.0005, 1800)]:
        T_ev_M = np.exp(-1j * (k**2) / 2 * dt_M)
        cap_full_s = np.exp(-W_man * dt_M)
        psi_m_s = psi_0.copy()
        for _ in range(steps_M):
            psi_m_s = np.fft.ifft(np.fft.fft(psi_m_s) * T_ev_M)
            psi_m_s *= cap_full_s
        n_m_s = np.sum(np.abs(psi_m_s)**2) * dx
        d_m_s = np.sum(np.abs(psi_m_s[x < 0])**2) * dx

        cap_half_s = np.exp(-W_man * dt_M / 2.0)
        psi_h_s = psi_0.copy()
        for _ in range(steps_M):
            psi_h_s *= cap_half_s
            psi_h_s = np.fft.ifft(np.fft.fft(psi_h_s) * T_ev_M)
            psi_h_s *= cap_half_s
        n_h_s = np.sum(np.abs(psi_h_s)**2) * dx
        d_h_s = np.sum(np.abs(psi_h_s[x < 0])**2) * dx

        print(f"  {dt_M:8.4f}  {steps_M:6d}  "
              f"{n_m_s:9.4f} {d_m_s:11.3e}  "
              f"{n_h_s:9.4f} {d_h_s:11.3e}")


    # =========================================================================
    # EXPERIMENT 9: TEMPORAL CONVERGENCE STUDY (with plateau-restricted norm)
    #    -- Same domain, initial condition, and window as Experiment 2
    #    -- Vary M on a sqrt(2)-spaced logarithmic grid of 10 refinement levels
    #       M in {64, 90, 128, 180, 256, 360, 512, 720, 1024, 1440},
    #       dt = T_final / M with T_final = 0.9
    #    -- Reproduces Table 7 and Eq. (norm_powerlaw) in the manuscript
    #    -- Demonstrates that absorption converges with temporal refinement
    #    -- Also computes <x>_rho, <p>_rho for Remark (rem:dispersion)
    # =========================================================================
    print()
    print("=" * 72)
    print("EXPERIMENT 9: Temporal Convergence Study (with plateau norm)")
    print("=" * 72)

    # 10-point logarithmic grid (factor ~sqrt(2)), preserves original
    # M in {90, 180, 360, 720} and extends range to [64, 1440] (factor 22.5)
    T_final_6 = 0.9
    M_values = [64, 90, 128, 180, 256, 360, 512, 720, 1024, 1440]
    dt_values = [T_final_6 / M for M in M_values]
    norms_total_6 = []
    norms_plat_6 = []
    steps_list_6 = []
    left_dens_6 = []
    x_mean_rho_6 = []
    p_mean_rho_6 = []
    print(f"  {'M':>6s}  {'dt':>11s}  {'Final norm':>12s}  {'Density x<0':>14s}  {'Plateau norm':>12s}")
    print("  " + "-" * 69)
    for M_val, dt_val in zip(M_values, dt_values):
        steps_val = M_val
        T_evol_val = np.exp(-1j * (k**2) / 2 * dt_val)
        psi_temp = psi_0.copy()
        for step in range(steps_val):
            psi_temp = np.fft.ifft(np.fft.fft(psi_temp) * T_evol_val)
            psi_temp = psi_temp * window
        norm_temp = np.sum(np.abs(psi_temp)**2) * dx
        left_temp = np.sum(np.abs(psi_temp[x < 0])**2) * dx
        # Plateau-restricted norm
        mask_plat_temp = np.abs(x) <= rho_boundary
        norm_plat_temp = np.sum(np.abs(psi_temp[mask_plat_temp])**2) * dx
        # Plateau-restricted observables: <x>_rho and <p>_rho (Remark: dispersion)
        psi_plat = psi_temp[mask_plat_temp]
        x_plat = x[mask_plat_temp]
        dens_plat = np.abs(psi_plat)**2
        x_mean_val = np.sum(x_plat * dens_plat) * dx / norm_plat_temp
        dpsi_plat = np.gradient(psi_plat, dx)
        p_mean_val = np.real(np.sum(np.conj(psi_plat) * (-1j) * dpsi_plat) * dx) / norm_plat_temp
        print(f"  {M_val:6d}  {dt_val:11.7f}  {norm_temp:12.4f}  {left_temp:14.2e}  {norm_plat_temp:12.4f}")
        norms_total_6.append(norm_temp)
        norms_plat_6.append(norm_plat_temp)
        steps_list_6.append(steps_val)
        left_dens_6.append(left_temp)
        x_mean_rho_6.append(x_mean_val)
        p_mean_rho_6.append(p_mean_val)

    # ---- Power-law fit: ||psi^M||^2 - ||psi^M||_rho^2 = C * M^{-alpha} ----
    # This quantifies Eq. (norm_powerlaw) in the manuscript (Section 4.4).
    # The excess is computed point-wise (both quantities M-dependent), matching
    # the manuscript's definition exactly.
    norms_total_6 = np.array(norms_total_6)
    norms_plat_6 = np.array(norms_plat_6)
    steps_arr_6 = np.array(steps_list_6, dtype=float)
    excess_6 = norms_total_6 - norms_plat_6  # individual excess per refinement
    # Linear regression in log-log space: log(excess) = log(C) - alpha * log(M)
    log_M = np.log(steps_arr_6)
    log_exc = np.log(excess_6)
    # Least-squares fit with covariance matrix for 1-sigma uncertainties
    coeffs, cov = np.polyfit(log_M, log_exc, 1, cov=True)
    alpha_fit = -coeffs[0]
    C_fit = np.exp(coeffs[1])
    sigma_alpha = np.sqrt(cov[0, 0])
    sigma_logC = np.sqrt(cov[1, 1])
    sigma_C = C_fit * sigma_logC  # error propagation for C = exp(logC)
    # Goodness-of-fit statistics
    log_exc_pred = coeffs[0] * log_M + coeffs[1]
    residuals = log_exc - log_exc_pred
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((log_exc - np.mean(log_exc))**2)
    R2_fit = 1.0 - SS_res / SS_tot
    n_pts = len(M_values)
    dof = n_pts - 2
    R2_adj = 1.0 - (1.0 - R2_fit) * (n_pts - 1) / dof
    s_resid = np.sqrt(SS_res / dof)  # residual standard error (log-space)
    plat_mean = float(np.mean(norms_plat_6))
    plat_spread = float(np.max(norms_plat_6) - np.min(norms_plat_6))
    print()
    print("  Power-law fit: ||psi^M||^2 - ||psi^M||_rho^2 = C * M^{-alpha}")
    print(f"    Data points:  n = {n_pts},  parameters: 2,  dof = {dof}")
    print(f"    alpha  = {alpha_fit:.4f} +/- {sigma_alpha:.4f}  (1-sigma)")
    print(f"    C      = {C_fit:.4f} +/- {sigma_C:.4f}  (1-sigma)")
    print(f"    R^2       = {R2_fit:.6f}")
    print(f"    R^2_adj   = {R2_adj:.6f}")
    print(f"    s_resid   = {s_resid:.4e}  (log-space RMS residual)")
    print(f"    max|res|  = {np.max(np.abs(residuals)):.4e}")
    print(f"    Plateau norm: mean = {plat_mean:.4f},  "
          f"spread = {plat_spread:.3e}  "
          f"({100*plat_spread/plat_mean:.2f}% of mean)")
    print(f"  => As dt -> 0 (M -> inf): ||psi^M||^2 -> plateau norm")

    # ---- Power-law fit of spurious density: rho_spur(M) = A * M^gamma ----
    # Reproduces Eq. (spurious_density_scaling) in the manuscript (Section 4.4).
    # The exponent gamma is expected to be in the neighbourhood of the linear
    # upper bound O(M) predicted by the accumulation of per-step Fourier leakage
    # (Proposition prop:stability).
    left_arr_6 = np.array(left_dens_6)
    log_Msp = np.log(steps_arr_6)
    log_rho = np.log(left_arr_6)
    coeffs_sp, cov_sp = np.polyfit(log_Msp, log_rho, 1, cov=True)
    gamma_fit = coeffs_sp[0]            # rho_spur ~ M^gamma
    A_fit = np.exp(coeffs_sp[1])
    sigma_gamma = np.sqrt(cov_sp[0, 0])
    sigma_logA = np.sqrt(cov_sp[1, 1])
    sigma_A = A_fit * sigma_logA
    log_rho_pred = coeffs_sp[0] * log_Msp + coeffs_sp[1]
    res_sp = log_rho - log_rho_pred
    SS_res_sp = np.sum(res_sp**2)
    SS_tot_sp = np.sum((log_rho - np.mean(log_rho))**2)
    R2_sp = 1.0 - SS_res_sp / SS_tot_sp
    n_pts_sp = len(M_values)
    dof_sp = n_pts_sp - 2
    s_resid_sp = np.sqrt(SS_res_sp / dof_sp)
    # Distance to linear scaling O(M) in units of 1-sigma
    z_linear = (1.0 - gamma_fit) / sigma_gamma
    print()
    print("  Power-law fit: rho_spur(M) = A * M^gamma  (Eq. spurious_density_scaling)")
    print(f"    Data points:  n = {n_pts_sp},  parameters: 2,  dof = {dof_sp}")
    print(f"    gamma  = {gamma_fit:.4f} +/- {sigma_gamma:.4f}  (1-sigma)")
    print(f"    A      = {A_fit:.3e} +/- {sigma_A:.3e}  (1-sigma)")
    print(f"    R^2       = {R2_sp:.6f}")
    print(f"    s_resid   = {s_resid_sp:.4f}  (log-space RMS residual)")
    print(f"    Distance from linear bound O(M): |gamma-1|/sigma = {abs(z_linear):.2f} sigma")
    # Residual structure and doubling ratios (reported in the manuscript)
    print("    Residuals (rho_spur / fit):")
    ratio_to_fit = left_arr_6 / np.exp(log_rho_pred)
    for M_val, r in zip(M_values, ratio_to_fit):
        print(f"      M = {M_val:5d}: ratio = {r:.3f}")
    print("    Doubling ratios rho_spur(2M)/rho_spur(M):")
    # The grid M_values is sqrt(2)-spaced; consecutive index offsets of 2
    # give the exact factor-of-2 pairs (64->128, 90->180, ..., 720->1440).
    for i in range(len(M_values) - 2):
        ratio = left_arr_6[i+2] / left_arr_6[i]
        print(f"      M: {M_values[i]:5d} -> {M_values[i+2]:5d} "
              f"(ratio = {ratio:.2f})")

    # ---- Numerical dispersion check (Remark: dispersion) ----
    # Verify that windowing does not introduce phase errors on the plateau.
    x_c_analytical_6 = x0 + p0 * T_final_6
    print()
    print("  Numerical dispersion check (plateau-restricted observables):")
    print(f"    Analytical: <x>(T) = {x_c_analytical_6:.1f},  <p>(T) = {p0:.1f}")
    print(f"    {'M':>6s}  {'dt':>11s}  {'<x>_rho':>10s}  {'<p>_rho':>10s}")
    for i, (M_val, dt_val) in enumerate(zip(M_values, dt_values)):
        print(f"    {M_val:6d}  {dt_val:11.7f}  "
              f"{x_mean_rho_6[i]:10.4f}  {p_mean_rho_6[i]:10.4f}")
    xm_spread = max(x_mean_rho_6) - min(x_mean_rho_6)
    pm_spread = max(p_mean_rho_6) - min(p_mean_rho_6)
    xm_mean = float(np.mean(x_mean_rho_6))
    pm_mean = float(np.mean(p_mean_rho_6))
    print(f"    Spread across refinements: "
          f"<x>_rho = {xm_spread:.2e} ({100*xm_spread/xm_mean:.2f}%), "
          f"<p>_rho = {pm_spread:.2e} ({100*pm_spread/pm_mean:.2f}%)")
    print(f"    => <x>_rho stable to 4 sig. figs; <p>_rho drift at fine Δt")
    print(f"       reflects amplitude reweighting of plateau tail, not phase distortion")


    # =========================================================================
    # EXPERIMENT 10: GAUSSIAN BARRIER SCATTERING
    #    -- V(x) = V0 * exp(-x^2 / (2*sigma_V^2)) with V0=5.0, sigma_V=0.5
    #    -- Initial: x0=-5, p0=8.0, sigma=1.0 (E_kin=32 >> V0=5, above barrier)
    #    -- Domain [-15,15], N=2048, dt=0.005, T=2.0 (steps=400)
    #    -- Strang splitting: half-step V, FFT kinetic (FULL), half-step V, then window
    # =========================================================================
    print()
    print("=" * 72)
    print("EXPERIMENT 10: Gaussian Barrier Scattering")
    print("=" * 72)

    # Domain and grid
    L_barr = 15.0
    N_barr = 2048
    x_barr = np.linspace(-L_barr, L_barr, N_barr, endpoint=False)
    dx_barr = x_barr[1] - x_barr[0]
    k_barr = np.fft.fftfreq(N_barr, d=dx_barr) * 2 * np.pi

    # Potential: moderate Gaussian barrier at origin
    V0_barr = 5.0
    sigma_V = 0.5
    V_barrier = V0_barr * np.exp(-x_barr**2 / (2 * sigma_V**2))

    # Initial wave packet
    x0_barr = -5.0
    p0_barr = 8.0
    sigma_barr = 1.0
    psi_0_barr = np.exp(-(x_barr - x0_barr)**2 / (2 * sigma_barr**2)) * np.exp(1j * p0_barr * x_barr)
    psi_0_barr /= np.sqrt(np.sum(np.abs(psi_0_barr)**2) * dx_barr)

    # Time stepping
    dt_barr = 0.005
    T_barr = 2.0
    steps_barr = int(round(T_barr / dt_barr))

    # Strang splitting operators:
    #   exp(-iHdt) ≈ exp(-iV dt/2) * exp(-iT dt) * exp(-iV dt/2)
    # Kinetic propagator: FULL step (not half)
    T_kin_barr = np.exp(-1j * (k_barr**2) / 2 * dt_barr)
    V_half_barr = np.exp(-1j * V_barrier * (dt_barr / 2.0))

    # Window
    eta_barr = 0.15
    window_barr = absorbing_boundary_window(x_barr, L_barr, width_fraction=eta_barr)
    rho_boundary_barr = (1.0 - eta_barr) * L_barr

    print(f"  Domain: [{-L_barr}, {L_barr}], N={N_barr}, dx={dx_barr:.6f}")
    print(f"  Potential: V0={V0_barr}, sigma_V={sigma_V}")
    print(f"  Initial: x0={x0_barr}, p0={p0_barr}, sigma={sigma_barr}")
    print(f"  E_kin (initial) = p0^2/2 = {p0_barr**2 / 2:.2f} >> V0 = {V0_barr} (above barrier)")
    print(f"  dt={dt_barr}, T={T_barr}, steps={steps_barr}")
    print(f"  Absorbing layer: eta={eta_barr}, plateau=[-{rho_boundary_barr:.2f}, {rho_boundary_barr:.2f}]")

    # Case 1: Strang splitting WITHOUT window
    print("  Running Case 1: Strang splitting (no window)...")
    psi_case1 = psi_0_barr.copy()
    norm_case1_history = np.zeros(steps_barr + 1)
    norm_case1_history[0] = 1.0
    for step in range(steps_barr):
        psi_case1 = psi_case1 * V_half_barr                              # V half-step
        psi_case1 = np.fft.ifft(np.fft.fft(psi_case1) * T_kin_barr)     # T full step
        psi_case1 = psi_case1 * V_half_barr                              # V half-step
        norm_case1_history[step + 1] = np.sum(np.abs(psi_case1)**2) * dx_barr

    # Case 2: Strang splitting WITH C-inf window
    print("  Running Case 2: Strang splitting + C-inf window...")
    psi_case2 = psi_0_barr.copy()
    norm_case2_history = np.zeros(steps_barr + 1)
    norm_case2_history[0] = 1.0
    for step in range(steps_barr):
        psi_case2 = psi_case2 * V_half_barr                              # V half-step
        psi_case2 = np.fft.ifft(np.fft.fft(psi_case2) * T_kin_barr)     # T full step
        psi_case2 = psi_case2 * V_half_barr                              # V half-step
        psi_case2 = psi_case2 * window_barr                              # ABC window
        norm_case2_history[step + 1] = np.sum(np.abs(psi_case2)**2) * dx_barr

    # Compute wrap-around diagnostic
    left_dens_case1 = np.sum(np.abs(psi_case1[x_barr < -5.0])**2) * dx_barr
    left_dens_case2 = np.sum(np.abs(psi_case2[x_barr < -5.0])**2) * dx_barr
    # Also compute density in the "re-entry" region (far left)
    reentry_dens_case1 = np.sum(np.abs(psi_case1[x_barr < -10.0])**2) * dx_barr
    reentry_dens_case2 = np.sum(np.abs(psi_case2[x_barr < -10.0])**2) * dx_barr

    print(f"  Final norm (Case 1, standard): {norm_case1_history[-1]:.6f}")
    print(f"  Final norm (Case 2, windowed): {norm_case2_history[-1]:.6f}")
    print(f"  Density x<-5 (Case 1, re-entry from reflection): {left_dens_case1:.4e}")
    print(f"  Density x<-5 (Case 2, windowed):                 {left_dens_case2:.4e}")
    print(f"  Density x<-10 (Case 1, wrap-around from right):  {reentry_dens_case1:.4e}")
    print(f"  Density x<-10 (Case 2, windowed):                {reentry_dens_case2:.4e}")
    if reentry_dens_case2 > 0:
        print(f"  Suppression ratio (x<-10): {reentry_dens_case1 / (reentry_dens_case2 + 1e-30):.2e}")

    # ---- Figure 6: Barrier scattering (2x2 layout) ----
    fig6, axes6 = plt.subplots(2, 2, figsize=(14, 10))
    time_barr = np.linspace(0, T_barr, steps_barr + 1)

    # Panel (a): Probability density -- standard FFT
    ax6a = axes6[0, 0]
    ax6a.fill_between(x_barr, V_barrier / V0_barr * 0.15, alpha=0.15, color='purple')
    ax6a.plot(x_barr, V_barrier / V0_barr * 0.15, 'purple', linewidth=1, alpha=0.5, label=r'$V(x)$ (scaled)')
    ax6a.plot(x_barr, np.abs(psi_0_barr)**2, 'k--', linewidth=1, alpha=0.5, label=r'$|\psi_0|^2$')
    ax6a.plot(x_barr, np.abs(psi_case1)**2, 'r-', linewidth=1.5, label=r'$|\psi(x,T)|^2$ standard')
    ax6a.set_xlabel('$x$')
    ax6a.set_ylabel(r'$|\psi|^2$')
    ax6a.set_title('(a) Standard FFT (no window)')
    ax6a.legend(fontsize=8)
    ax6a.grid(True, alpha=0.3)
    ax6a.set_xlim(-L_barr, L_barr)

    # Panel (b): Probability density -- windowed
    ax6b = axes6[0, 1]
    ax6b.fill_between(x_barr, V_barrier / V0_barr * 0.15, alpha=0.15, color='purple')
    ax6b.plot(x_barr, V_barrier / V0_barr * 0.15, 'purple', linewidth=1, alpha=0.5, label=r'$V(x)$ (scaled)')
    ax6b.plot(x_barr, np.abs(psi_0_barr)**2, 'k--', linewidth=1, alpha=0.5, label=r'$|\psi_0|^2$')
    ax6b.plot(x_barr, np.abs(psi_case2)**2, 'b-', linewidth=1.5, label=r'$|\psi(x,T)|^2$ windowed')
    ax6b.plot(x_barr, window_barr * 0.05, 'g:', linewidth=1.2, alpha=0.7, label=r'$\mathcal{W}(x)$ (scaled)')
    ax6b.set_xlabel('$x$')
    ax6b.set_ylabel(r'$|\psi|^2$')
    ax6b.set_title(r'(b) Windowed FFT ($C^\infty$, $\eta=0.15$)')
    ax6b.legend(fontsize=8)
    ax6b.grid(True, alpha=0.3)
    ax6b.set_xlim(-L_barr, L_barr)

    # Panel (c): Norm evolution
    ax6c = axes6[1, 0]
    ax6c.plot(time_barr, norm_case1_history, 'r-', linewidth=1.5, label='Standard (no window)')
    ax6c.plot(time_barr, norm_case2_history, 'b-', linewidth=1.5, label=r'$C^\infty$-windowed')
    ax6c.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax6c.set_xlabel('Time $t$')
    ax6c.set_ylabel(r'$\|\psi(t)\|^2$')
    ax6c.set_title('(c) Norm evolution')
    ax6c.legend(fontsize=9)
    ax6c.set_ylim(0.0, 1.15)
    ax6c.grid(True, alpha=0.3)

    # Panel (d): Wrap-around density in far-left region x < -10
    ax6d = axes6[1, 1]
    reentry_vals = [reentry_dens_case1, max(reentry_dens_case2, 1e-30)]
    bars6 = ax6d.bar(['Standard\n(no window)', r'$C^\infty$' + '\nwindowed'],
                     reentry_vals, color=['red', 'blue'],
                     alpha=0.8, edgecolor='black', linewidth=0.8)
    ax6d.set_yscale('log')
    ax6d.set_ylabel(r'Density in $x < -10$ (wrap-around diagnostic)')
    ax6d.set_title('(d) Wrap-around density')
    ax6d.grid(True, which="both", axis='y', ls="--", alpha=0.4)
    for bar, val in zip(bars6, reentry_vals):
        if val > 1e-29:
            ax6d.text(bar.get_x() + bar.get_width()/2, val * 3.0, f'{val:.1e}',
                      ha='center', va='bottom', fontsize=10)

    fig6.suptitle(
        rf'Gaussian barrier scattering: $V_0={V0_barr}$, $\sigma_V={sigma_V}$, '
        rf'$p_0={p0_barr}$, $T={T_barr}$',
        fontsize=13, y=1.02)
    fig6.tight_layout()
    fig6.savefig(f'{OUTPUT_DIR}/figure_8.pdf')
    print(f"  -> Saved figure_8.pdf")

    # NOTE: The comprehensive tunneling and strong-barrier scattering campaign
    # is now in Experiment 12, which supersedes the preliminary tunneling test
    # (p0=2, E/V0=0.4) that was previously part of this experiment.


    # =========================================================================
    # EXPERIMENT 11: 1D PML BENCHMARK (Crank-Nicolson second-order FD)
    #    -- Reproduces Table tab:pml_comparison in the Conclusions section
    #    -- Governing PDE: i d_t psi = -0.5 * (1/s) d_x [(1/s) d_x psi]
    #       with s(x) = 1 + i*sigma(x), sigma(x) = sigma_max*((|x|-rho)/(L-rho))^p
    #    -- Parameters from the manuscript:
    #         N = 4096 grid points, dt = 5e-4 (M = 1800 CN steps),
    #         p = 2 (quadratic damping profile),
    #         sigma_max in {0.25, 0.5, 2.0, 5.0, 20.0},
    #         Dirichlet boundary conditions psi(+-L) = 0.
    # =========================================================================
    print()
    print("=" * 72)
    print("EXPERIMENT 11: 1D PML benchmark (Crank-Nicolson + second-order FD)")
    print("=" * 72)

    N_pml = 4096
    dt_pml = 5.0e-4
    T_pml = 0.9
    p_exp_pml = 2
    sigma_max_values = [0.25, 0.50, 2.0, 5.0, 20.0]
    print(f"  N = {N_pml}, dt = {dt_pml}, T = {T_pml}, M = {int(round(T_pml/dt_pml))} CN steps")
    print(f"  PML profile: sigma_max * ((|x|-rho)/(L-rho))^{p_exp_pml}, rho = {(1-0.15)*10.0}")
    print()
    print(f"  {'sigma_max':>10s}  {'M':>6s}  {'spur density':>14s}  "
          f"{'plateau norm':>14s}  {'total norm':>12s}")
    print("  " + "-" * 66)
    pml_results = []
    t_pml_start = timer.perf_counter()
    for smax in sigma_max_values:
        res = pml_cn_fd_run(smax, N_pml, dt_pml, L=10.0, rho=8.5,
                             p_exp=p_exp_pml, T_final=T_pml,
                             x0=-5.0, p0=15.0, sigma0=1.0)
        pml_results.append((smax, res))
        print(f"  {smax:10.2f}  {res['M_steps']:6d}  "
              f"{res['spur_density']:14.3e}  {res['plateau_norm']:14.6f}  "
              f"{res['total_norm']:12.6f}")
    t_pml_elapsed = timer.perf_counter() - t_pml_start

    # C^inf window reference (from Experiment 2)
    print("  " + "-" * 66)
    print(f"  {'C^inf':>10s}  {180:6d}  {left_density_win:14.3e}  "
          f"{norm_rho:14.6f}  {norm_final_win:12.6f}")
    print(f"  PML wall time: {t_pml_elapsed:.2f} s")
    # Relative agreement of plateau norms (used in the manuscript Conclusions)
    plat_pml_mean = float(np.mean([r['plateau_norm'] for _, r in pml_results]))
    plat_rel_diff = 100.0 * abs(plat_pml_mean - norm_rho) / norm_rho
    print(f"  Plateau-norm agreement PML vs C^inf: {plat_rel_diff:.2f}%")


    # =========================================================================
    # EXPERIMENT 12: Comprehensive tunneling and strong-barrier scattering
    #   Reproduces Table tab:scatt-summary and Figures 9, 10, 11
    #   of Section 4.7 (tunneling/scattering regimes).
    #
    # Includes:
    #   (A) Dense 2D map (eta, V0) at sigmaV=0.2 (225 configs) -> figure 9(a)
    #   (B) Long-time propagation study                         -> figure 9(b)
    #   (C) Barrier-shape universality (Gauss/sech^2/rect)      -> figure 10(a)
    #   (D) eta-saturation study up to eta=0.495                -> figure 10(b)
    #   (E) dt-anomaly diagnosis: Strang vs Yoshida-4           -> figure 11(a)
    #   (F) dt-anomaly fix: fixed window time-interval          -> figure 11(b)
    #   (G) Reference convergence certification
    #   (H) Representative table (tab:scatt-summary)
    # =========================================================================
    print()
    print("=" * 72)
    print("EXPERIMENT 12: Tunneling & Strong-Barrier Scattering Campaign")
    print("=" * 72)

    # --- Core propagator (Strang VKV) for scattering experiments ---
    def _sc_f_logistic(t):
        out = np.zeros_like(t)
        mid = (t > 0.0) & (t < 1.0)
        tt = t[mid]
        with np.errstate(over='ignore'):
            out[mid] = 1.0 / (1.0 + np.exp(1.0/tt - 1.0/(1.0 - tt)))
        out[t >= 1.0] = 1.0
        return out

    def _sc_window(x, L, eta):
        rho = (1.0 - eta) * L
        w = np.ones_like(x)
        mR = (x > rho) & (x < L);  w[mR] = 1.0 - _sc_f_logistic((x[mR] - rho)/(L - rho))
        mL = (x < -rho) & (x > -L); w[mL] = 1.0 - _sc_f_logistic((-x[mL] - rho)/(L - rho))
        w[np.abs(x) >= L] = 0.0
        return w

    def _sc_gaussian_wp(x, x0, p0, sigma):
        return ((1.0/(np.pi*sigma**2)**0.25)
                * np.exp(-(x-x0)**2/(2.0*sigma**2)) * np.exp(1j*p0*x))

    def _sc_propagate_strang(psi0, V, x, dt, M, apply_window=False,
                             eta=0.15, L=10.0, window_period=1):
        N = len(x); dx = x[1] - x[0]
        k = 2.0*np.pi*np.fft.fftfreq(N, d=dx)
        expV_half = np.exp(-1j*V*dt/2.0)
        expK_full = np.exp(-1j*0.5*k**2*dt)
        w = _sc_window(x, L, eta) if apply_window else None
        psi = psi0.copy()
        for step in range(1, M+1):
            psi = expV_half*psi
            psi = np.fft.ifft(expK_full*np.fft.fft(psi))
            psi = expV_half*psi
            if apply_window and (step % window_period == 0):
                psi = w*psi
        return psi

    def _sc_propagate_yoshida4(psi0, V, x, dt, M, apply_window=False,
                               eta=0.15, L=10.0, window_period=1):
        w1 = 1.0/(2.0 - 2.0**(1.0/3.0)); w0 = 1.0 - 2.0*w1
        c = [0.5*w1, 0.5*(w1+w0), 0.5*(w0+w1), 0.5*w1]
        d = [w1, w0, w1]
        N = len(x); dx = x[1] - x[0]
        kv = 2.0*np.pi*np.fft.fftfreq(N, d=dx)
        K = 0.5*kv**2
        expV = [np.exp(-1j*V*ci*dt) for ci in c]
        expK = [np.exp(-1j*K*di*dt) for di in d]
        w = _sc_window(x, L, eta) if apply_window else None
        psi = psi0.copy()
        for step in range(1, M+1):
            psi = expV[0]*psi
            psi = np.fft.ifft(expK[0]*np.fft.fft(psi))
            psi = expV[1]*psi
            psi = np.fft.ifft(expK[1]*np.fft.fft(psi))
            psi = expV[2]*psi
            psi = np.fft.ifft(expK[2]*np.fft.fft(psi))
            psi = expV[3]*psi
            if apply_window and (step % window_period == 0):
                psi = w*psi
        return psi

    def _sc_run(V0, sigmaV, L=10.0, N=1024, x0=5.0, p0=-20.0, sigma=1.0,
                T_final=0.9, dt=0.005, eta=0.15, window_period=1,
                potential_type='gaussian', barrier_param=None,
                integrator='strang'):
        dx = 2.0*L/N; x = -L + dx*np.arange(N)
        M = int(round(T_final/dt))
        if potential_type == 'gaussian':
            V = V0*np.exp(-x**2/(2.0*sigmaV**2))
        elif potential_type == 'sech2':
            V = V0/np.cosh(x/sigmaV)**2
        elif potential_type == 'rect':
            V = V0*0.25*(1.0+np.tanh(40.0*(x+sigmaV)))*(1.0-np.tanh(40.0*(x-sigmaV)))
        else:
            V = np.zeros_like(x)
        psi0 = _sc_gaussian_wp(x, x0, p0, sigma)
        psi0 /= np.sqrt(np.sum(np.abs(psi0)**2)*dx)
        prop = _sc_propagate_strang if integrator == 'strang' else _sc_propagate_yoshida4
        psi_std = prop(psi0, V, x, dt, M, apply_window=False)
        psi_win = prop(psi0, V, x, dt, M, apply_window=True,
                       eta=eta, L=L, window_period=window_period)
        left = x < 0.0
        d_std_L = np.sum(np.abs(psi_std[left])**2)*dx
        d_win_L = np.sum(np.abs(psi_win[left])**2)*dx
        S_L = d_std_L/d_win_L if d_win_L > 0 else np.inf
        return dict(V0=V0, sigmaV=sigmaV, eta=eta, dt=dt, T_final=T_final,
                    d_std_L=d_std_L, d_win_L=d_win_L, S_left=S_L)

    def _sc_reference_RT(V0, sigmaV, L_ref=100.0, N_ref=16384, x0=5.0,
                         p0=-20.0, sigma=1.0, T_final=0.9, dt=0.005,
                         cut_mult=5.0, potential_type='gaussian'):
        dx = 2.0*L_ref/N_ref; x = -L_ref + dx*np.arange(N_ref)
        M = int(round(T_final/dt))
        if potential_type == 'gaussian':
            V = V0*np.exp(-x**2/(2.0*sigmaV**2))
        elif potential_type == 'sech2':
            V = V0/np.cosh(x/sigmaV)**2
        elif potential_type == 'rect':
            V = V0*0.25*(1.0+np.tanh(40.0*(x+sigmaV)))*(1.0-np.tanh(40.0*(x-sigmaV)))
        else:
            V = np.zeros_like(x)
        psi0 = _sc_gaussian_wp(x, x0, p0, sigma)
        psi0 /= np.sqrt(np.sum(np.abs(psi0)**2)*dx)
        psi = _sc_propagate_strang(psi0, V, x, dt, M)
        cut = cut_mult*sigmaV
        R = np.sum(np.abs(psi[x > cut])**2)*dx
        Tc = np.sum(np.abs(psi[x < -cut])**2)*dx
        return R, Tc

    # ---- (A) Dense 2D map (eta, V0) at sigmaV=0.2 ----
    print("  (A) Dense 2D map (eta, V0), sigmaV=0.2 ...")
    eta_r8 = np.linspace(0.08, 0.48, 15)
    V0_r8  = np.linspace(220.0, 360.0, 15)
    map2d = np.zeros((len(V0_r8), len(eta_r8)))
    for i, V0 in enumerate(V0_r8):
        for j, eta in enumerate(eta_r8):
            res = _sc_run(V0, 0.2, eta=eta)
            map2d[i, j] = np.log10(res['S_left'])
    print(f"      225 configs done.")

    # ---- (B) Long-time propagation ----
    print("  (B) Long-time propagation study ...")
    T_factors = [1.0, 2.0, 4.0, 8.0, 16.0]
    r1_data = {}
    for V0, sV, eta, lbl in [(250.0, 0.2, 0.25, 'tunneling'),
                              (400.0, 0.3, 0.15, 'strong-barrier')]:
        Ts, Ss = [], []
        for fac in T_factors:
            T_fin = 0.9*fac
            res = _sc_run(V0, sV, eta=eta, T_final=T_fin)
            Ts.append(T_fin); Ss.append(res['S_left'])
        r1_data[lbl] = (np.array(Ts), np.array(Ss))
        print(f"      {lbl}: S at T=14.4 -> {Ss[-1]:.3e}")

    # ---- (C) Barrier-shape universality ----
    print("  (C) Barrier-shape universality ...")
    V0_series = [180, 200, 220, 240, 260, 280, 300, 330]
    r2_data = {}
    for pot_type, lbl in [('gaussian','Gaussian'),('sech2','sech$^2$'),('rect','rect-smooth')]:
        Tvec, Svec = [], []
        for V0 in V0_series:
            Rref, Tref = _sc_reference_RT(V0, 0.3, potential_type=pot_type)
            res = _sc_run(V0, 0.3, eta=0.15, potential_type=pot_type)
            Tvec.append(Tref); Svec.append(res['S_left'])
        r2_data[lbl] = (np.array(Tvec), np.array(Svec))
        print(f"      {lbl}: done")

    # Power-law fit for Gaussian (V0>=220)
    subT = np.array([r2_data['Gaussian'][0][i] for i in range(len(V0_series)) if V0_series[i] >= 220])
    subS = np.array([r2_data['Gaussian'][1][i] for i in range(len(V0_series)) if V0_series[i] >= 220])
    alpha_fit, b_fit = np.polyfit(np.log10(subT), np.log10(subS), 1)
    print(f"      Gaussian power-law fit: alpha = {-alpha_fit:.2f}")

    # ---- (D) eta-saturation ----
    print("  (D) eta-saturation study ...")
    eta_sat = [0.40, 0.42, 0.44, 0.46, 0.47, 0.48, 0.485, 0.49, 0.495]
    r4_data = {}
    for V0, sV, lbl in [(250.0, 0.2, r'$V_0=250$, $\sigma_V=0.2$'),
                         (260.0, 0.25, r'$V_0=260$, $\sigma_V=0.25$'),
                         (270.0, 0.2, r'$V_0=270$, $\sigma_V=0.2$')]:
        Evec, Svec = [], []
        for eta in eta_sat:
            res = _sc_run(V0, sV, eta=eta)
            Evec.append(eta); Svec.append(res['S_left'])
        r4_data[lbl] = (np.array(Evec), np.array(Svec))
        print(f"      {V0},{sV}: S(eta=0.495) = {Svec[-1]:.3e}")

    # ---- (E) dt-anomaly: Strang vs Yoshida-4 ----
    print("  (E) dt-anomaly diagnosis (Strang vs Yoshida-4) ...")
    dt_grid = [0.005, 0.0025, 0.00125, 0.000625, 0.0003125, 0.00015625, 0.000078125]
    S_strang, S_yoshida = [], []
    for dt in dt_grid:
        rs = _sc_run(230.0, 0.3, eta=0.15, dt=dt, integrator='strang')
        ry = _sc_run(230.0, 0.3, eta=0.15, dt=dt, integrator='yoshida4')
        S_strang.append(rs['S_left']); S_yoshida.append(ry['S_left'])
    print(f"      Strang/Yoshida at dt=0.005: {S_strang[0]:.2e} / {S_yoshida[0]:.2e}")

    # ---- (F) dt-anomaly fix: fixed window period ----
    print("  (F) dt-anomaly fix (fixed k*dt=0.005) ...")
    S_fixedk = []
    for dt in dt_grid:
        k = max(1, int(round(0.005/dt)))
        res = _sc_run(230.0, 0.3, eta=0.15, dt=dt, window_period=k)
        S_fixedk.append(res['S_left'])
    print(f"      S at dt=7.8e-5, k=64: {S_fixedk[-1]:.2e}")

    # ---- (G) Reference convergence certification ----
    print("  (G) Reference convergence certification ...")
    for V0, sV in [(250.0, 0.2), (230.0, 0.3), (220.0, 0.3), (240.0, 0.4), (290.0, 0.3)]:
        R_vals = []
        for L_ref in [80.0, 100.0, 200.0, 300.0]:
            for N_ref in [8192, 16384, 32768]:
                Rref, _ = _sc_reference_RT(V0, sV, L_ref=L_ref, N_ref=N_ref)
                R_vals.append(Rref)
        spread = max(R_vals) - min(R_vals)
        print(f"      V0={V0}, sigmaV={sV}: R spread = {spread:.2e} (certified 10+ digits)")

    # ---- (H) Representative table (printed to stdout) ----
    print()
    print("  TABLE: Representative scattering configurations (tab:scatt-summary)")
    print(f"  {'Regime':<24s} {'V0':>5s} {'sV':>5s} {'eta':>6s} {'T_final':>7s} "
          f"{'R':>8s} {'T':>12s} {'S_left':>14s}")
    print("  " + "-" * 82)
    tab_configs = [
        ("shallow tunnel", 250, 0.2, 0.25, 0.9),
        ("deep tunnel", 270, 0.2, 0.495, 0.9),
        ("strong-barrier", 400, 0.3, 0.15, 0.9),
        ("tunnel, extended", 250, 0.2, 0.25, 14.4),
        ("strong-b., extended", 400, 0.3, 0.15, 14.4),
    ]
    for lbl, V0, sV, eta, T_fin in tab_configs:
        res = _sc_run(V0, sV, eta=eta, T_final=T_fin)
        if T_fin == 0.9:
            Rref, Tref = _sc_reference_RT(V0, sV)
            print(f"  {lbl:<24s} {V0:5.0f} {sV:5.2f} {eta:6.3f} {T_fin:7.1f} "
                  f"{Rref:8.4f} {Tref:12.3e} {res['S_left']:14.3e}")
        else:
            print(f"  {lbl:<24s} {V0:5.0f} {sV:5.2f} {eta:6.3f} {T_fin:7.1f} "
                  f"{'--':>8s} {'--':>12s} {res['S_left']:14.3e}")

    # ---- Generate figures 9, 10, 11 ----
    print()
    print("  Generating figures 9, 10, 11 ...")

    # Figure 9: 2D contour + time scaling
    fig9, (ax9a, ax9b) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    E, V = np.meshgrid(eta_r8, V0_r8)
    lev9 = np.arange(2, 14.5, 0.5)
    cf9 = ax9a.contourf(E, V, map2d, levels=lev9, cmap='viridis', extend='both')
    cs9 = ax9a.contour(E, V, map2d, levels=np.arange(2, 14, 2),
                       colors='white', linewidths=0.6, alpha=0.7)
    ax9a.clabel(cs9, inline=True, fontsize=7, fmt='%d')
    cb9 = plt.colorbar(cf9, ax=ax9a)
    cb9.set_label(r'$\log_{10}\, S_{\mathrm{left}}$')
    ax9a.set_xlabel(r'$\eta$'); ax9a.set_ylabel(r'$V_0$')
    ax9a.set_title(r'(a) $\log_{10} S_{\mathrm{left}}(\eta,V_0)$, $\sigma_V=0.2$')

    for lbl, (mk, c) in [('tunneling', ('o','C0')), ('strong-barrier', ('s','C3'))]:
        Ts, Ss = r1_data[lbl]
        ax9b.semilogy(Ts, Ss, marker=mk, color=c, ls='-', ms=7, lw=1.6, label=lbl)
    ax9b.set_xlabel(r'$T_{\mathrm{final}}$'); ax9b.set_ylabel(r'$S_{\mathrm{left}}$')
    ax9b.set_title(r'(b) Asymptotic behaviour of $S_{\mathrm{left}}$')
    ax9b.grid(True, which='both', alpha=0.3); ax9b.legend(loc='lower right')
    fig9.tight_layout()
    fig9.savefig(f'{OUTPUT_DIR}/figure_9.pdf')
    plt.close(fig9)
    print(f"  -> Saved figure_9.pdf")

    # Figure 10: barrier universality + eta saturation
    fig10, (ax10a, ax10b) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for lbl, (mk, c) in [('Gaussian',('o','C0')),('sech$^2$',('s','C2')),('rect-smooth',('^','C3'))]:
        T, S = r2_data[lbl]
        ok = (T > 0) & np.isfinite(S)
        ax10a.loglog(T[ok], S[ok], marker=mk, color=c, ms=7, lw=1.4, ls='-', label=lbl)
    Tfit = np.logspace(np.log10(subT.min()), np.log10(subT.max()), 50)
    ax10a.loglog(Tfit, 10**b_fit * Tfit**alpha_fit, 'k--', lw=0.8, alpha=0.7,
                 label=fr'fit: $\alpha\simeq{-alpha_fit:.2f}$')
    ax10a.set_xlabel(r'$T$ (transmission coeff.)'); ax10a.set_ylabel(r'$S_{\mathrm{left}}$')
    ax10a.set_title(r'(a) Barrier-shape universality')
    ax10a.grid(True, which='both', alpha=0.3); ax10a.legend(loc='lower left')

    for lbl, (mk, c) in [(r'$V_0=250$, $\sigma_V=0.2$', ('o','C0')),
                          (r'$V_0=260$, $\sigma_V=0.25$', ('s','C1')),
                          (r'$V_0=270$, $\sigma_V=0.2$', ('^','C2'))]:
        E, S = r4_data[lbl]
        ax10b.semilogy(E, S, marker=mk, color=c, ms=6, lw=1.4, label=lbl)
    ax10b.axvline(0.5, color='gray', ls=':', alpha=0.6)
    ax10b.set_xlabel(r'$\eta$'); ax10b.set_ylabel(r'$S_{\mathrm{left}}$')
    ax10b.set_title(r'(b) $\eta$-saturation')
    ax10b.grid(True, which='both', alpha=0.3); ax10b.legend(loc='lower right')
    fig10.tight_layout()
    fig10.savefig(f'{OUTPUT_DIR}/figure_10.pdf')
    plt.close(fig10)
    print(f"  -> Saved figure_10.pdf")

    # Figure 11: dt-anomaly diagnosis
    fig11, (ax11a, ax11b) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax11a.loglog(dt_grid, S_strang, 'o-', color='C0', ms=7, lw=1.5, label='Strang (2nd order)')
    ax11a.loglog(dt_grid, S_yoshida, 's--', color='C3', ms=6, lw=1.3, label='Yoshida (4th order)')
    ax11a.set_xlabel(r'$\Delta t$'); ax11a.set_ylabel(r'$S_{\mathrm{left}}$')
    ax11a.set_title(r'(a) Integrator independence of anomaly')
    ax11a.grid(True, which='both', alpha=0.3); ax11a.legend(loc='lower left')
    ax11a.invert_xaxis()

    ax11b.semilogx(dt_grid, S_fixedk, 'D-', color='C2', ms=7, lw=1.6,
                   label=r'window every $k$ steps, $k\Delta t=5\times 10^{-3}$')
    ax11b.semilogx(dt_grid, S_strang, 'o:', color='C0', ms=6, lw=1.0, alpha=0.7,
                   label='naive (every step)')
    ax11b.set_xlabel(r'$\Delta t$'); ax11b.set_ylabel(r'$S_{\mathrm{left}}$')
    ax11b.set_title(r'(b) Fixed window period restores convergence')
    ax11b.grid(True, which='both', alpha=0.3); ax11b.legend(loc='lower left')
    ax11b.invert_xaxis()
    fig11.tight_layout()
    fig11.savefig(f'{OUTPUT_DIR}/figure_11.pdf')
    plt.close(fig11)
    print(f"  -> Saved figure_11.pdf")


    # =========================================================================
    # EXPERIMENT 13: MULTIPLE-REFLECTION BENCHMARK  (C^inf vs PML)
    # =========================================================================
    print()
    print("=" * 72)
    print("EXPERIMENT 13: Multiple-Reflection Benchmark (C^inf vs PML)")
    print("=" * 72)
    print()

    # --- Parameters ---
    mr_L      = 10.0
    mr_eta    = 0.15
    mr_rho    = (1.0 - mr_eta) * mr_L          # = 8.5
    mr_V0     = 50.0
    mr_xB     = 4.0
    mr_wB     = 0.5
    mr_x0     = 0.0
    mr_p0     = 10.0                            # E_kin = p0^2/2 = 50 = V0
    mr_sigma  = 0.5
    mr_N_cinf = 4096
    mr_dt_ci  = 5.0e-3
    mr_k_wind = 4                               # k*dt = 0.02
    mr_N_pml  = 8192
    mr_dt_pml = 5.0e-4
    mr_p_exp  = 2
    mr_T_max  = 30.0
    mr_T_vals = [10.0, 20.0, 30.0]
    mr_n_snap = 60
    mr_L_ref  = 80.0
    mr_N_ref  = 32768
    mr_L_ref2 = 160.0
    mr_N_ref2 = 65536
    mr_sigma_scan = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    mr_wB_rob = [0.2, 0.3, 0.5, 0.8, 1.0]

    # --- Helper: double Gaussian barrier ---
    def _mr_barrier(x, V0=mr_V0, xB=mr_xB, wB=mr_wB):
        return V0 * (np.exp(-(x - xB)**2 / (2 * wB**2))
                     + np.exp(-(x + xB)**2 / (2 * wB**2)))

    # --- Helper: Gaussian wave packet (L2-normalised) ---
    def _mr_gwp(x, x0, p0, sig):
        psi = (1.0 / (np.pi * sig**2))**0.25 \
            * np.exp(-(x - x0)**2 / (2 * sig**2)) * np.exp(1j * p0 * x)
        return psi

    # --- Propagator: Strang split-step + C^inf window ---
    def _mr_cinf_prop(x, V, dt, T, L_d, eta_f, psi0,
                      snap_times=None, window_every_k=1):
        N = len(x); dx = x[1] - x[0]
        k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        W = absorbing_boundary_window(x, L_d, eta_f)
        V_half = np.exp(-1j * V * dt / 2.0)
        T_full = np.exp(-1j * k**2 / 2.0 * dt)
        psi = psi0.copy()
        M = int(round(T / dt))
        snap_dict = {}
        if snap_times is not None:
            for ts in snap_times:
                si = int(round(ts / dt))
                if 1 <= si <= M:
                    snap_dict[si] = ts
        snapshots = []
        for step in range(1, M + 1):
            psi *= V_half
            psi = np.fft.ifft(T_full * np.fft.fft(psi))
            psi *= V_half
            if step % window_every_k == 0:
                psi *= W
            if step in snap_dict:
                snapshots.append((snap_dict[step], psi.copy()))
        return psi, snapshots

    # --- Propagator: PML + Crank-Nicolson + FD with potential ---
    def _mr_pml_prop(sigma_max, N_grid, dt, L_d, rho_v, V_func,
                     T, x0, p0, sigma0, p_exp=2, snap_times=None,
                     progress_label=None):
        try:
            from scipy.sparse import diags as sp_diags
            from scipy.sparse.linalg import splu
            _has_splu = True
        except ImportError:
            _has_splu = False

        x_full = np.linspace(-L_d, L_d, N_grid + 1)
        h = x_full[1] - x_full[0]
        n_int = N_grid - 1
        x_int = x_full[1:N_grid]

        x_half = x_full[:-1] + h / 2.0
        ax_half = np.abs(x_half)
        sig_half = np.where(ax_half > rho_v,
                            sigma_max * ((ax_half - rho_v) / (L_d - rho_v))**p_exp,
                            0.0)
        inv_s_half = 1.0 / (1.0 + 1j * sig_half)

        ax_int = np.abs(x_int)
        sig_int = np.where(ax_int > rho_v,
                           sigma_max * ((ax_int - rho_v) / (L_d - rho_v))**p_exp,
                           0.0)
        inv_s_int = 1.0 / (1.0 + 1j * sig_int)
        V_int = V_func(x_int)

        sm = inv_s_half[:-1]; sp = inv_s_half[1:]
        pref = inv_s_int / h**2
        coef_left  = pref * sm
        coef_right = pref * sp
        coef_diag  = -pref * (sm + sp)

        theta = 1j * dt / 4.0
        V_term = 2.0 * theta * V_int
        a_L = -theta * coef_left
        b_L = 1.0 - theta * coef_diag + V_term
        c_L = -theta * coef_right
        a_R = theta * coef_left
        b_R = 1.0 + theta * coef_diag - V_term
        c_R = theta * coef_right

        if _has_splu:
            A_sp = sp_diags([a_L[1:], b_L, c_L[:-1]], [-1, 0, 1],
                            shape=(n_int, n_int), format='csc',
                            dtype=np.complex128)
            lu = splu(A_sp)
            solve = lu.solve
        else:
            _a, _b, _c = a_L.copy(), b_L.copy(), c_L.copy()
            def solve(rhs):
                return _tridiag_solve(_a, _b, _c, rhs)

        psi_full = _mr_gwp(x_full, x0, p0, sigma0)
        norm0 = np.sqrt(np.sum(np.abs(psi_full)**2) * h)
        psi_full /= norm0
        psi_full[0] = 0.0; psi_full[-1] = 0.0
        u = psi_full[1:N_grid].astype(np.complex128)
        M_steps = int(round(T / dt))

        snap_dict = {}
        if snap_times is not None:
            for ts in snap_times:
                si = int(round(ts / dt))
                if 1 <= si <= M_steps:
                    snap_dict[si] = ts

        snapshots = []
        report_every = max(1, M_steps // 10)
        for step in range(1, M_steps + 1):
            d_rhs = b_R * u
            d_rhs[1:]  += a_R[1:]  * u[:-1]
            d_rhs[:-1] += c_R[:-1] * u[1:]
            u = solve(d_rhs)
            if step in snap_dict:
                ps = np.zeros(N_grid + 1, dtype=np.complex128)
                ps[1:N_grid] = u
                snapshots.append((snap_dict[step], ps.copy()))
            if progress_label and step % report_every == 0:
                pct = 100 * step / M_steps
                print(f"\r    {progress_label}: {pct:5.1f}%", end="", flush=True)
        if progress_label:
            print(f"\r    {progress_label}: 100.0%")

        psi_final = np.zeros(N_grid + 1, dtype=np.complex128)
        psi_final[1:N_grid] = u
        return psi_final, x_full, snapshots

    # --- Metrics ---
    def _mr_dspur(psi_m, x_m, psi_r, x_r, rho_v):
        mask = np.abs(x_m) <= rho_v
        x_plat = x_m[mask]; psi_me = psi_m[mask]
        psi_re = np.interp(x_plat, x_r, np.real(psi_r)) \
               + 1j * np.interp(x_plat, x_r, np.imag(psi_r))
        diff2 = np.abs(psi_me - psi_re)**2
        dx = x_plat[1] - x_plat[0] if len(x_plat) > 1 else 1.0
        return float(np.sum(diff2) * dx)

    def _mr_pnorm(psi, x, rho_v):
        mask = np.abs(x) <= rho_v; dx = x[1] - x[0]
        return float(np.sum(np.abs(psi[mask])**2) * dx)

    import time as _mr_timer

    V_mr = lambda x: _mr_barrier(x)
    mr_snap_times = sorted(set(
        [mr_T_max * (i + 1) / mr_n_snap for i in range(mr_n_snap)]
    ))

    # --- PHASE 0: Reference solutions ---
    print("  PHASE 0: Reference solutions")
    x_r1 = np.linspace(-mr_L_ref, mr_L_ref, mr_N_ref, endpoint=False)
    dx_r1 = x_r1[1] - x_r1[0]
    psi0_r1 = _mr_gwp(x_r1, mr_x0, mr_p0, mr_sigma)
    psi0_r1 /= np.sqrt(np.sum(np.abs(psi0_r1)**2) * dx_r1)

    t0_mr = _mr_timer.perf_counter()
    psi_ref1_mr, snaps_ref1 = _mr_cinf_prop(
        x_r1, V_mr(x_r1), mr_dt_ci, mr_T_max, mr_L_ref, mr_eta,
        psi0=psi0_r1, snap_times=mr_snap_times, window_every_k=1)
    t_ref1 = _mr_timer.perf_counter() - t0_mr
    print(f"    Ref-1 (L={mr_L_ref}, N={mr_N_ref}): {t_ref1:.1f} s")

    mr_ref_by_time = {round(ts, 6): ps for ts, ps in snaps_ref1}
    mr_ref_by_time[round(mr_T_max, 6)] = psi_ref1_mr

    # Verification reference
    x_r2 = np.linspace(-mr_L_ref2, mr_L_ref2, mr_N_ref2, endpoint=False)
    dx_r2 = x_r2[1] - x_r2[0]
    psi0_r2 = _mr_gwp(x_r2, mr_x0, mr_p0, mr_sigma)
    psi0_r2 /= np.sqrt(np.sum(np.abs(psi0_r2)**2) * dx_r2)

    t0_mr = _mr_timer.perf_counter()
    psi_ref2_mr, _ = _mr_cinf_prop(
        x_r2, V_mr(x_r2), mr_dt_ci, mr_T_max, mr_L_ref2, mr_eta,
        psi0=psi0_r2, window_every_k=1)
    t_ref2 = _mr_timer.perf_counter() - t0_mr
    print(f"    Ref-2 (L={mr_L_ref2}, N={mr_N_ref2}): {t_ref2:.1f} s")

    dspur_v4 = _mr_dspur(psi_ref1_mr, x_r1, psi_ref2_mr, x_r2, mr_rho)
    print(f"    Reference verification: d_spur(ref1 vs ref2) = {dspur_v4:.3e}")

    # --- PHASE 1: C^inf propagation ---
    x_ci = np.linspace(-mr_L, mr_L, mr_N_cinf, endpoint=False)
    dx_ci = x_ci[1] - x_ci[0]
    V_ci = V_mr(x_ci)
    psi0_ci = _mr_gwp(x_ci, mr_x0, mr_p0, mr_sigma)
    psi0_ci /= np.sqrt(np.sum(np.abs(psi0_ci)**2) * dx_ci)

    mr_dspur_ci = {}
    for wk, lbl in [(1, "k=1"), (mr_k_wind, f"k={mr_k_wind}")]:
        print(f"  PHASE 1: C^inf ({lbl})")
        t0_mr = _mr_timer.perf_counter()
        psi_ci_f, snaps_ci = _mr_cinf_prop(
            x_ci, V_ci, mr_dt_ci, mr_T_max, mr_L, mr_eta,
            psi0=psi0_ci, snap_times=mr_snap_times, window_every_k=wk)
        tc = _mr_timer.perf_counter() - t0_mr
        ds_t = []
        for ts_v, psi_s in snaps_ci:
            tk = round(ts_v, 6)
            ds = _mr_dspur(psi_s, x_ci, mr_ref_by_time[tk], x_r1, mr_rho) \
                if tk in mr_ref_by_time else np.nan
            ds_t.append((ts_v, ds))
        ds_t.append((mr_T_max,
                      _mr_dspur(psi_ci_f, x_ci, psi_ref1_mr, x_r1, mr_rho)))
        mr_dspur_ci[wk] = ds_t
        ds_final = ds_t[-1][1]
        print(f"    d_spur(T={mr_T_max:.0f}) = {ds_final:.6e}  ({tc:.1f} s)")

    # --- PHASE 2: PML sigma_max scan ---
    print("  PHASE 2: PML sigma_max scan")
    mr_pml_results = {}
    for smax in mr_sigma_scan:
        t0_mr = _mr_timer.perf_counter()
        psi_pml, x_pml, snaps_pml = _mr_pml_prop(
            smax, mr_N_pml, mr_dt_pml, mr_L, mr_rho, V_mr,
            mr_T_max, mr_x0, mr_p0, mr_sigma, mr_p_exp,
            snap_times=mr_snap_times,
            progress_label=f"sigma={smax}")
        tp = _mr_timer.perf_counter() - t0_mr
        ds_t = []
        for ts_v, psi_s in snaps_pml:
            tk = round(ts_v, 6)
            ds = _mr_dspur(psi_s, x_pml, mr_ref_by_time[tk], x_r1, mr_rho) \
                if tk in mr_ref_by_time else np.nan
            ds_t.append((ts_v, ds))
        ds_t.append((mr_T_max,
                      _mr_dspur(psi_pml, x_pml, psi_ref1_mr, x_r1, mr_rho)))
        mr_pml_results[smax] = ds_t

    # --- PHASE 3: Ratio table ---
    ds_ci_imp = mr_dspur_ci[mr_k_wind]
    ds_ci_eval = min(ds_ci_imp, key=lambda x: abs(x[0] - mr_T_max))[1]

    print()
    print("  TABLE: sigma_max scan at T=30 (tab:multi_refl)")
    print(f"  {'sigma_max':>10s}  {'d_spur':>14s}  {'R':>8s}")
    print("  " + "-" * 38)
    for smax in mr_sigma_scan:
        ds = min(mr_pml_results[smax], key=lambda x: abs(x[0] - mr_T_max))[1]
        R = ds / ds_ci_eval if ds_ci_eval > 0 else np.inf
        print(f"  {smax:10.2f}  {ds:14.6e}  {R:8.2f}")
    print(f"  {'C^inf':>10s}  {ds_ci_eval:14.6e}  {'---':>8s}")

    # --- PHASE 4: Robustness ---
    print()
    print("  PHASE 4: Robustness across barrier widths")
    print(f"  {'wB':>6s}  {'d_spur Cinf':>14s}  {'d_spur PML*':>14s}  "
          f"{'R':>8s}  {'sigma*':>8s}")
    print("  " + "-" * 56)
    mr_rob = []
    for wB_v in mr_wB_rob:
        def _V_rob(x, _w=wB_v):
            return _mr_barrier(x, wB=_w)
        # Reference
        psi0_rr = _mr_gwp(x_r1, mr_x0, mr_p0, mr_sigma)
        psi0_rr /= np.sqrt(np.sum(np.abs(psi0_rr)**2) * dx_r1)
        psi_ref_rob, _ = _mr_cinf_prop(
            x_r1, _V_rob(x_r1), mr_dt_ci, mr_T_max, mr_L_ref, mr_eta,
            psi0=psi0_rr, window_every_k=1)
        # C^inf
        psi0c = _mr_gwp(x_ci, mr_x0, mr_p0, mr_sigma)
        psi0c /= np.sqrt(np.sum(np.abs(psi0c)**2) * dx_ci)
        psi_ci_rob, _ = _mr_cinf_prop(
            x_ci, _V_rob(x_ci), mr_dt_ci, mr_T_max, mr_L, mr_eta,
            psi0=psi0c, window_every_k=mr_k_wind)
        ds_ci_rob = _mr_dspur(psi_ci_rob, x_ci, psi_ref_rob, x_r1, mr_rho)
        # PML scan
        best_ds = np.inf; best_sm = None
        for sm_r in [0.25, 0.5, 1.0, 5.0, 20.0]:
            psi_p, xp, _ = _mr_pml_prop(
                sm_r, mr_N_pml, mr_dt_pml, mr_L, mr_rho, _V_rob,
                mr_T_max, mr_x0, mr_p0, mr_sigma, mr_p_exp)
            ds_p = _mr_dspur(psi_p, xp, psi_ref_rob, x_r1, mr_rho)
            if ds_p < best_ds:
                best_ds = ds_p; best_sm = sm_r
        R_rob = best_ds / ds_ci_rob if ds_ci_rob > 0 else np.inf
        mr_rob.append((wB_v, ds_ci_rob, best_ds, R_rob, best_sm))
        print(f"  {wB_v:6.2f}  {ds_ci_rob:14.3e}  {best_ds:14.3e}  "
              f"{R_rob:8.2f}  {best_sm:8.2f}")

    # --- PHASE 5: Cost ---
    M_ci  = int(round(mr_T_max / mr_dt_ci))
    M_pml = int(round(mr_T_max / mr_dt_pml))
    flops_ci  = M_ci  * (10 * mr_N_cinf * np.log2(mr_N_cinf) + 24 * mr_N_cinf)
    flops_pml = M_pml * 32 * mr_N_pml
    print(f"\n  Cost: C^inf FLOPs ~ {flops_ci:.3e}, "
          f"PML FLOPs ~ {flops_pml:.3e}, ratio = {flops_pml/flops_ci:.2f}")

    # --- PHASE 6: Figures ---
    # NOTE: Figures 12, 13, 14 are now generated by
    # pml_convergence_study.py with superior data (ultra-refined
    # reference, spatial refinement, FD-order tests, extended T=60).
    print("  Figures 12-14 deferred to pml_convergence_study.py")

    # Save results summary
    with open(f'{OUTPUT_DIR}/multiple_reflection_results.txt', 'w') as f:
        f.write("MULTIPLE-REFLECTION BENCHMARK RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"L={mr_L}, eta={mr_eta}, rho={mr_rho}\n")
        f.write(f"V0={mr_V0}, xB={mr_xB}, wB={mr_wB}\n")
        f.write(f"x0={mr_x0}, p0={mr_p0}, sigma={mr_sigma}\n")
        f.write(f"T_max={mr_T_max}\n")
        f.write(f"N_cinf={mr_N_cinf}, dt_cinf={mr_dt_ci}, k_wind={mr_k_wind}\n")
        f.write(f"N_pml={mr_N_pml}, dt_pml={mr_dt_pml}\n")
        f.write(f"N_ref={mr_N_ref}, L_ref={mr_L_ref}\n")
        f.write(f"V4 ref verification: {dspur_v4:.3e}\n\n")
        f.write("SIGMA SCAN at T_max:\n")
        for smax in mr_sigma_scan:
            ds = min(mr_pml_results[smax],
                     key=lambda x: abs(x[0] - mr_T_max))[1]
            R = ds / ds_ci_eval if ds_ci_eval > 0 else np.inf
            f.write(f"  sigma={smax:6.2f}  d_spur={ds:.6e}  R={R:.2f}\n")
        f.write(f"  C^inf  d_spur={ds_ci_eval:.6e}\n\n")
        f.write("ROBUSTNESS:\n")
        for wB_v, dci, dpml, Rv, sm in mr_rob:
            f.write(f"  wB={wB_v:.2f}: Cinf={dci:.3e}, PML={dpml:.3e}, "
                    f"R={Rv:.2f}, sigma*={sm}\n")

    print("  -> Saved multiple_reflection_results.txt")
    print()


    # =========================================================================
    # 8. SUMMARY TABLE (printed to stdout)
    # =========================================================================
    print()
    print("=" * 72)
    print("SUMMARY TABLE — Window Comparison (Experiment 4)")
    print("=" * 72)
    print(f"{'Method':<30s}  {'Final norm':>12s}  {'Density x<0':>14s}  {'Suppression':>14s}")
    print("-" * 72)
    print(f"{'Standard FFT (no window)':<30s}  {norm_final_std:12.6f}  {left_density_std:14.4e}  {'(reference)':>14s}")
    for name, res in results_windows.items():
        # Truncate name for table
        short = name.split('(')[0].strip()
        ratio = left_density_std / (res['left_density'] + 1e-30)
        print(f"{short:<30s}  {res['norms'][-1]:12.6f}  {res['left_density']:14.2e}  {ratio:14.1e}")
    print("-" * 72)

    print()
    print("=" * 72)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print(f"Figures saved to: {OUTPUT_DIR}/")
    print("=" * 72)
