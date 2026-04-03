#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smooth Windowing via C-infinity Bump Functions for Pseudo-Spectral Quantum Dynamics.

Companion code for:
    D. Ariza-Ruiz, "Eliminating Wrap-Around Errors in Pseudo-Spectral
    Quantum Dynamics via C-infinity Windowing",
    Computer Physics Communications (2026).

Theoretical foundation:
    P. Bergold & C. Lasser, "Fourier Series Windowed by a Bump Function",
    J. Fourier Anal. Appl. 26 (2020) 65.
    https://doi.org/10.1007/s00041-020-09773-3

This script runs four numerical experiments and generates five publication-
quality figures (300 dpi PNG) plus a summary table printed to stdout:

    Experiment 1 — Gibbs suppression & L2 convergence (standard / Hann / C-inf)
    Experiment 2 — Schrodinger wave packet absorption (standard vs. windowed)
    Experiment 3 — Convergence vs. plateau parameter rho
    Experiment 4 — Window regularity comparison (Hann / Tukey / C-inf bump)

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

import numpy as np
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

    Equals 1 on the plateau |x| <= (1 - eta)*L and decays smoothly to 0
    at |x| = L via a reparametrized bump phi(z) = exp(-1/(1 - z^2)).

    Parameters
    ----------
    x : ndarray
        Spatial grid points.
    L : float
        Half-width of the computational domain.
    width_fraction : float
        Fraction eta of each half-domain used for absorption (default 0.15).

    Returns
    -------
    ndarray
        Window values in [0, 1].
    """
    eta = width_fraction
    xi = np.abs(x / L)               # normalized coordinate in [0, 1]
    result = np.ones_like(xi)

    mask = xi > (1.0 - eta)
    z = (xi[mask] - (1.0 - eta)) / eta
    z = np.clip(z, 0.0, 1.0 - 1e-15)
    result[mask] = np.exp(1.0) * np.exp(-1.0 / (1.0 - z**2))

    result[xi >= 1.0] = 0.0
    return result


def fourier_truncation(signal: np.ndarray, num_coefs: int) -> np.ndarray:
    """Truncate the DFT of *signal* to the lowest *num_coefs* modes (symmetric)."""
    coeffs = np.fft.fft(signal)
    truncated = np.zeros_like(coeffs)
    truncated[:num_coefs] = coeffs[:num_coefs]
    truncated[-num_coefs:] = coeffs[-num_coefs:]
    return np.real(np.fft.ifft(truncated))


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

    N_grid = 1000
    x_1D = np.linspace(-1, 1, N_grid, endpoint=False)
    f_x = x_1D.copy()                        # Test function f(x) = x
    bump_degen = standard_bump_function(x_1D) # Degenerate C^inf bump
    f_windowed = f_x * bump_degen

    N_max = 50
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
        err_std[i] = np.linalg.norm(f_x - approx_s) / np.sqrt(N_grid)
        err_win[i] = np.linalg.norm(f_windowed - approx_w) / np.sqrt(N_grid)
        err_hann[i] = np.linalg.norm(f_hann - approx_h) / np.sqrt(N_grid)

    # ---- Figure 1: Convergence plot (3 curves) ----
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.semilogy(coef_range, err_std, 'r.-', markersize=5, linewidth=1.2,
                 label='Standard truncation (no window)')
    ax1.semilogy(coef_range, err_hann, 's-', color='#ff7f0e', markersize=4, linewidth=1.2,
                 label=r'Hann window ($C^1$, algebraic $\mathcal{O}(n^{-2})$)')
    ax1.semilogy(coef_range, err_win, 'b.-', markersize=5, linewidth=1.2,
                 label=r'$C^\infty$ bump (super-algebraic)')
    ax1.set_xlabel('Number of Fourier coefficients ($n$)')
    ax1.set_ylabel(r'$L^2$ error $\varepsilon_n$ (log scale)')
    ax1.set_title(r'Convergence analysis: $f(x) = x$ on $[-1,\,1]$')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    fig1.tight_layout()
    fig1.savefig(f'{OUTPUT_DIR}/figure1_convergence.png')
    print(f"  -> Saved figure1_convergence.png")

    # Print key values
    print(f"  Standard  error at n=50:  {err_std[-1]:.4e}")
    print(f"  Hann      error at n=50:  {err_hann[-1]:.4e}")
    print(f"  C^inf     error at n=50:  {err_win[-1]:.4e}")
    print(f"  Improvement C^inf vs Std at n=50: {err_std[-1]/err_win[-1]:.0f}x")
    print(f"  Improvement C^inf vs Hann at n=50: {err_hann[-1]/err_win[-1]:.0f}x")


    # =========================================================================
    # EXPERIMENT 2: SCHRÖDINGER EQUATION (WAVE PACKET ABSORPTION)
    # =========================================================================
    print()
    print("=" * 72)
    print("EXPERIMENT 2: Schrödinger Wave Packet Dynamics")
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

    # Time evolution — track norm at every step
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

    # ---- Figure 2: Schrödinger comparison ----
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
    fig2.savefig(f'{OUTPUT_DIR}/figure2_schrodinger.png')
    print(f"  -> Saved figure2_schrodinger.png")

    # ---- Figure 3: Norm evolution over time ----
    time_axis = np.linspace(0, T_final, steps + 1)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(time_axis, norm_std_history, 'r-', linewidth=1.5,
             label='Standard FFT (norm conserved — includes wrap-around)')
    ax3.plot(time_axis, norm_win_history, 'b-', linewidth=1.5,
             label=r'$C^\infty$-windowed (norm decreases — physical absorption)')
    ax3.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax3.set_xlabel('Time $t$')
    ax3.set_ylabel(r'$\|\psi(t)\|^2$')
    ax3.set_title('Norm evolution: unitarity vs. physical absorption')
    ax3.legend()
    ax3.set_ylim(0.0, 1.1)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(f'{OUTPUT_DIR}/figure3_norm_evolution.png')
    print(f"  -> Saved figure3_norm_evolution.png")


    # =========================================================================
    # EXPERIMENT 3: CONVERGENCE WITH PLATEAU BUMP (varying rho/lambda)
    #    — Quantifies the effect of the plateau parameter on L2 error
    #    — Aligns with Bergold & Lasser Theorem 4.6 (Lipschitz constant bound)
    # =========================================================================
    print()
    print("=" * 72)
    print("EXPERIMENT 3: Convergence with plateau bump — varying rho")
    print("=" * 72)

    N_grid_3 = 2000
    x_3 = np.linspace(-1, 1, N_grid_3, endpoint=False)
    f_3 = x_3.copy()       # Same test function f(x) = x
    N_max_3 = 80
    coef_range_3 = np.arange(1, N_max_3 + 1)

    rho_values = [0.0, 0.50, 0.70, 0.85]
    colors_rho = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels_rho = [
        r'$\rho = 0.00$ (degenerate)',
        r'$\rho = 0.50$',
        r'$\rho = 0.70$',
        r'$\rho = 0.85$',
    ]

    errors_by_rho = {}
    for rho_val in rho_values:
        if rho_val == 0.0:
            w = standard_bump_function(x_3)
        else:
            w = plateau_bump_function(x_3, rho_val)
        fw = f_3 * w
        errs = np.zeros(N_max_3)
        for i, n in enumerate(coef_range_3):
            approx = fourier_truncation(fw, n)
            errs[i] = np.linalg.norm(fw - approx) / np.sqrt(N_grid_3)
        errors_by_rho[rho_val] = errs
        print(f"  rho={rho_val:.2f}: error at n=80 = {errs[-1]:.4e}")

    # Also include the unwindowed (standard) case for reference
    errs_unwound = np.zeros(N_max_3)
    for i, n in enumerate(coef_range_3):
        approx = fourier_truncation(f_3, n)
        errs_unwound[i] = np.linalg.norm(f_3 - approx) / np.sqrt(N_grid_3)

    # ---- Figure 4: Convergence vs rho ----
    fig4, ax4 = plt.subplots(figsize=(9, 6))
    ax4.semilogy(coef_range_3, errs_unwound, 'k--', linewidth=1.0, alpha=0.6,
                 label='No window (algebraic)')
    for rho_val, color, lbl in zip(rho_values, colors_rho, labels_rho):
        ax4.semilogy(coef_range_3, errors_by_rho[rho_val], '.-', color=color,
                     markersize=3, linewidth=1.2, label=lbl)
    ax4.set_xlabel('Number of Fourier coefficients ($n$)')
    ax4.set_ylabel(r'$L^2$ error $\varepsilon_n$ (log scale)')
    ax4.set_title(r'Effect of plateau parameter $\rho$ on convergence rate')
    ax4.legend()
    ax4.grid(True, which="both", ls="--", alpha=0.4)
    fig4.tight_layout()
    fig4.savefig(f'{OUTPUT_DIR}/figure4_convergence_rho.png')
    print(f"  -> Saved figure4_convergence_rho.png")


    # =========================================================================
    # EXPERIMENT 4: WINDOW COMPARISON (Hann vs Tukey vs C^inf bump)
    #    — Compares different window regularities on the Schrödinger problem
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

    # Run Schrödinger simulation for each window
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

    fig5.savefig(f'{OUTPUT_DIR}/figure5_window_comparison.png')
    print(f"  -> Saved figure5_window_comparison.png")


    # =========================================================================
    # 6. SUMMARY TABLE (printed to stdout)
    # =========================================================================
    print()
    print("=" * 72)
    print("SUMMARY TABLE")
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

    plt.show()
