#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended high-precision Fourier convergence experiment.

Tracks the super-algebraic convergence of C∞-windowed Fourier series
far beyond the float64 machine-epsilon floor, using mpmath arbitrary-
precision arithmetic.

Companion script for:
    D. Ariza-Ruiz, "Parameter-Free Absorbing Boundaries for Pseudo-Spectral
    Quantum Dynamics via C∞ Windowing", CPC (2026).

Theoretical foundation:
    P. Bergold & C. Lasser, "Fourier Series Windowed by a Bump Function",
    J. Fourier Anal. Appl. 26 (2020) 65.

Strategy:
    Phase 1 (numpy, seconds):  Standard and Hann errors — algebraic rates,
            no need for extended precision.
    Phase 2 (mpmath, minutes): C∞ bump errors — super-algebraic decay that
            breaks through the float64 floor.

    The DFT power spectrum |C_k|^2 is computed via Horner's method in
    real arithmetic (no complex type needed).  L2 errors are recovered
    via Parseval's identity, so all n values come from a single DFT pass.
    Early termination when |C_k|^2 drops below 10^{-(2·DPS+10)}.

Usage:
    python3 convergence_extended.py

Output:
    convergence_extended.csv          — full table (n, eps_cinf, eps_hann, eps_std)
    figure_convergence_extended.png   — publication-quality convergence plot
    stdout                            — key values, alpha_eff, verification

Requirements:
    pip3 install mpmath numpy matplotlib
    pip3 install gmpy2               # optional, ~5x speedup for mpmath

Author:  David Ariza-Ruiz
License: MIT
"""

from __future__ import annotations
import sys
import time
import csv
import os

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these parameters to adjust the experiment
# ═══════════════════════════════════════════════════════════════════════════
DPS           = 80       # mpmath decimal digits of precision
N_GRID        = 20000    # number of grid points (must be even)
N_FOURIER     = 4000     # max Fourier modes to analyze (keep ≤ N_GRID // 4)
LAMBDA_VAL    = 1        # half-domain parameter λ
TEST_FUNCTIONS = True    # also compute exp(-x²), sin(5πx), 1/(1+25x²)
MAKE_PLOT     = True     # generate PNG figure

# ═══════════════════════════════════════════════════════════════════════════
#  Derived constants (do not edit)
# ═══════════════════════════════════════════════════════════════════════════
assert N_GRID % 2 == 0, "N_GRID must be even"
assert N_FOURIER <= N_GRID // 4, (
    f"N_FOURIER={N_FOURIER} > N_GRID//4={N_GRID//4}: DFT aliasing "
    f"would contaminate the results.  Increase N_GRID or reduce N_FOURIER."
)
N_HALF = N_GRID // 2

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1 — Standard and Hann errors (numpy, float64)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  EXTENDED FOURIER CONVERGENCE — ARBITRARY PRECISION")
print(f"  Precision : {DPS} decimal digits")
print(f"  Grid      : N = {N_GRID}")
print(f"  Modes     : n = 1 … {N_FOURIER}")
print(f"  Domain    : [−{LAMBDA_VAL}, {LAMBDA_VAL}]")
print("=" * 72)
print()

import numpy as np

print("[Phase 1] Standard & Hann errors (numpy float64) ...")
t_phase1 = time.time()

dx_np  = 2.0 * LAMBDA_VAL / N_GRID
x_np   = np.linspace(-LAMBDA_VAL, LAMBDA_VAL, N_GRID, endpoint=False)
f_np   = x_np.copy()

# Hann window on [-1, 1]
hann_np = np.zeros_like(x_np)
mask_h  = np.abs(x_np) < 1.0
hann_np[mask_h] = 0.5 * (1.0 + np.cos(np.pi * np.abs(x_np[mask_h])))
fh_np = f_np * hann_np


def parseval_errors_numpy(signal: np.ndarray, n_max: int) -> np.ndarray:
    """L2 errors for n = 1 … n_max via Parseval (numpy)."""
    N  = len(signal)
    Nh = N // 2
    dx = 2.0 * LAMBDA_VAL / N

    C  = np.fft.fft(signal)
    ps = np.abs(C[:Nh + 1]) ** 2            # |C_k|^2  for k = 0 … N/2

    # Cumulative tail: tail[n] = Σ_{k=n}^{N/2} ps[k]
    tail = np.zeros(Nh + 2)
    for k in range(Nh, -1, -1):
        tail[k] = tail[k + 1] + ps[k]

    # ε_n² = (dx/N) · (tail[n] + tail[n+1] − ps[N/2])
    errors = np.zeros(n_max)
    for n in range(1, n_max + 1):
        if n <= Nh:
            err_sq = (dx / N) * (tail[n] + tail[n + 1] - ps[Nh])
            errors[n - 1] = np.sqrt(max(err_sq, 0.0))
    return errors


err_std_np  = parseval_errors_numpy(f_np,  N_FOURIER)
err_hann_np = parseval_errors_numpy(fh_np, N_FOURIER)

t1 = time.time()
print(f"  Done in {t1 - t_phase1:.1f} s")
print(f"  Std   n=80: {err_std_np[79]:.4e}   n=160: {err_std_np[159]:.4e}")
print(f"  Hann  n=80: {err_hann_np[79]:.4e}   n=160: {err_hann_np[159]:.4e}")
print()

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2 — C∞ bump errors (mpmath, arbitrary precision)
# ═══════════════════════════════════════════════════════════════════════════
from mpmath import (mp, mpf, cos as mpcos, sin as mpsin, exp as mpexp,
                    pi as mppi, sqrt as mpsqrt, log as mplog, fsum)

mp.dps = DPS

print(f"[Phase 2] C∞ bump errors (mpmath, {DPS}-digit precision) ...")
try:
    import gmpy2                                   # noqa: F401
    print("  Backend: gmpy2 (fast C library)")
except ImportError:
    print("  Backend: pure Python  — install gmpy2 for ~5× speedup:")
    print("           pip3 install gmpy2")
print()

# ── 2a. Build the windowed signal f_w = f · w_{0,λ} ─────────────────────
# Bergold & Lasser Eq. (4.1) with ρ = 0, λ = 1 (degenerate bump):
#   w(x) = 1 / (exp(1/(1−|x|) − 1/|x|) + 1)   for 0 < |x| < 1
#   w(0) = 1,  w(x) = 0 for |x| ≥ 1

print("  Building windowed signal on the grid ...")
t2 = time.time()

N     = N_GRID
mp_dx = mpf(2) * LAMBDA_VAL / N
mp_x  = [mpf(-LAMBDA_VAL) + mp_dx * j for j in range(N)]

fw_vals = []
for j in range(N):
    xj  = mp_x[j]
    axj = abs(xj)
    if axj >= mpf(1):
        fw_vals.append(mpf(0))
    elif axj == mpf(0):
        # f(0) · w(0) = 0 · 1 = 0
        fw_vals.append(mpf(0))
    else:
        exponent = mpf(1) / (1 - axj) - mpf(1) / axj
        wj = mpf(1) / (mpexp(exponent) + 1)
        fw_vals.append(xj * wj)

t3 = time.time()
print(f"  Signal built in {t3 - t2:.1f} s")

# ── 2b. DFT power spectrum via Horner (real arithmetic) ──────────────────
# C_k = Σ_j f_j · exp(−2πi jk/N)
# Horner from j = N−1 down to 0:
#   (re, im) ← f_j + ω_k · (re, im)
# with ω_k = (cos θ_k, −sin θ_k),  θ_k = 2πk/N.
#
# For real f:  |C_{N−k}|² = |C_k|²  →  only k = 0 … N/2 needed.

TAIL_CUTOFF = mpf(10) ** (-(2 * DPS + 10))     # ≈ 10^{-110}

print(f"  Computing DFT |C_k|² for k = 0 … {N_HALF}")
print(f"  (early stop when |C_k|² < {mp.nstr(TAIL_CUTOFF, 3)})")
print()

power_spec = [mpf(0)] * (N_HALF + 1)   # will be filled up to k_stop
k_stop     = N_HALF                     # actual last k computed

t_dft_start = time.time()

for k in range(N_HALF + 1):
    theta_k = 2 * mppi * k / N
    ck = mpcos(theta_k)
    sk = mpsin(theta_k)

    re = mpf(0)
    im = mpf(0)
    for j in range(N - 1, -1, -1):
        new_re = fw_vals[j] + ck * re + sk * im
        new_im = ck * im - sk * re
        re = new_re
        im = new_im

    power_spec[k] = re * re + im * im

    # Progress
    if k % 25 == 0 or k == N_HALF:
        elapsed = time.time() - t_dft_start
        rate = (k + 1) / elapsed if elapsed > 0 else 1
        eta  = (N_HALF - k) / rate if rate > 0 else 0
        print(f"\r  k = {k:5d}/{N_HALF}   |C_k|² = {mp.nstr(power_spec[k], 6):>16s}"
              f"   [{elapsed:6.0f}s elapsed, ~{eta:5.0f}s left]    ",
              end="", flush=True)

    # Early termination
    if k > N_FOURIER and power_spec[k] < TAIL_CUTOFF:
        k_stop = k
        print(f"\n\n  *** Early stop at k = {k}:  |C_k|² = {mp.nstr(power_spec[k], 4)}"
              f"  < threshold ***")
        # Zero out the rest
        for kk in range(k + 1, N_HALF + 1):
            power_spec[kk] = mpf(0)
        break

t_dft_end = time.time()
print(f"\n  DFT done in {t_dft_end - t_dft_start:.1f} s"
      f"  ({k_stop + 1} coefficients computed)")
print()

# ── 2c. L2 errors via Parseval ──────────────────────────────────────────
# ε_n² = (dx/N) · [tail(n) + tail(n+1) − ps(N/2)]
# where tail(n) = Σ_{k=n}^{N/2} power_spec[k]

print("  Computing L2 errors via Parseval ...")

# Build cumulative tail sums
tail = [mpf(0)] * (N_HALF + 2)   # tail[N/2 + 1] = 0
for k in range(N_HALF, -1, -1):
    tail[k] = tail[k + 1] + power_spec[k]

ps_nyquist = power_spec[N_HALF]

err_cinf = []
for n in range(1, N_FOURIER + 1):
    if n <= N_HALF:
        err_sq = (mp_dx / N) * (tail[n] + tail[n + 1] - ps_nyquist)
        err_cinf.append(mpsqrt(err_sq) if err_sq > 0 else mpf(0))
    else:
        err_cinf.append(mpf(0))

t_parse = time.time()
print(f"  Done in {t_parse - t_dft_end:.1f} s")
print()

# ── 2d. (Optional) Additional test functions ─────────────────────────────
extra_results = {}
if TEST_FUNCTIONS:
    print("  Computing additional test functions ...")

    test_fns = {
        'exp(-x²)':      lambda x: mpexp(-x * x),
        'sin(5πx)':       lambda x: mpsin(5 * mppi * x),
        '1/(1+25x²)':    lambda x: mpf(1) / (1 + 25 * x * x),
    }

    for name, fn in test_fns.items():
        print(f"    {name} ... ", end="", flush=True)
        t_fn = time.time()

        # Build windowed signal
        fw_test = []
        for j in range(N):
            xj  = mp_x[j]
            axj = abs(xj)
            if axj >= mpf(1):
                fw_test.append(mpf(0))
            elif axj == mpf(0):
                fw_test.append(fn(xj))  # w(0) = 1
            else:
                exponent = mpf(1) / (1 - axj) - mpf(1) / axj
                wj = mpf(1) / (mpexp(exponent) + 1)
                fw_test.append(fn(xj) * wj)

        # DFT power spectrum (with early stop)
        ps_test = [mpf(0)] * (N_HALF + 1)
        for kk in range(N_HALF + 1):
            theta_k = 2 * mppi * kk / N
            ckk = mpcos(theta_k)
            skk = mpsin(theta_k)
            re = mpf(0)
            im = mpf(0)
            for j in range(N - 1, -1, -1):
                new_re = fw_test[j] + ckk * re + skk * im
                new_im = ckk * im - skk * re
                re = new_re
                im = new_im
            ps_test[kk] = re * re + im * im
            if kk > N_FOURIER and ps_test[kk] < TAIL_CUTOFF:
                break

        # Tail sums
        tail_t = [mpf(0)] * (N_HALF + 2)
        for kk in range(N_HALF, -1, -1):
            tail_t[kk] = tail_t[kk + 1] + ps_test[kk]

        # Errors at key n values
        milestones = [80, 160, 250, 500, 1000, 2000]
        result = {}
        for nn in milestones:
            if nn <= N_FOURIER and nn <= N_HALF:
                esq = (mp_dx / N) * (tail_t[nn] + tail_t[nn + 1] - ps_test[N_HALF])
                result[nn] = mpsqrt(esq) if esq > 0 else mpf(0)
        extra_results[name] = result
        print(f"done ({time.time() - t_fn:.0f} s)")

    print()

# ═══════════════════════════════════════════════════════════════════════════
#  VERIFICATION — cross-check with numpy at n ≤ 160
# ═══════════════════════════════════════════════════════════════════════════
print("[Verification] Cross-checking mpmath vs numpy at n = 80, 160:")

# Also compute the C∞ error in numpy for comparison
bump_np = np.zeros_like(x_np)
mask_b  = np.abs(x_np) < 1.0
ax_b    = np.abs(x_np[mask_b])
exp_arg = 1.0 / (1.0 - ax_b) - 1.0 / ax_b
with np.errstate(over='ignore'):
    bump_np[mask_b] = 1.0 / (np.exp(exp_arg) + 1.0)
# Handle x=0 (exponent = 1 - inf = -inf → w=1, but f(0)=0 so fw=0)
fw_np = f_np * bump_np
err_cinf_np = parseval_errors_numpy(fw_np, N_FOURIER)

for n_check in [80, 160]:
    e_mp  = float(err_cinf[n_check - 1])
    e_np  = err_cinf_np[n_check - 1]
    e_np2 = float(err_cinf_np[n_check - 1])
    rel   = abs(e_mp - e_np2) / e_np2 if e_np2 > 0 else 0
    print(f"  n = {n_check}:  mpmath = {e_mp:.6e},  numpy = {e_np2:.6e},"
          f"  rel diff = {rel:.2e}")

print()

# ═══════════════════════════════════════════════════════════════════════════
#  RESULTS — Key values
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  KEY RESULTS — C∞ bump (degenerate, ρ = 0, λ = 1)")
print("=" * 72)

milestones = [10, 20, 40, 60, 80, 100, 120, 160, 200, 250, 300, 400,
              500, 600, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000]
print(f"  {'n':>6s}   {'ε_n (C∞)':>22s}   {'ε_n (Hann)':>16s}   {'ε_n (Std)':>16s}")
print("  " + "─" * 68)

for n in milestones:
    if n <= N_FOURIER:
        e_c = err_cinf[n - 1]
        e_h = err_hann_np[n - 1] if n <= len(err_hann_np) else 0
        e_s = err_std_np[n - 1]  if n <= len(err_std_np)  else 0
        if e_c > 0:
            print(f"  {n:6d}   {mp.nstr(e_c, 12):>22s}   {e_h:16.6e}   {e_s:16.6e}")
        else:
            s = f"< 10^{{-{DPS}}}"
            print(f"  {n:6d}   {s:>22s}   {e_h:16.6e}   {e_s:16.6e}")

# ── Alpha_eff (wide-ratio doubling) ──────────────────────────────────────
print()
print("  α_eff (wide-ratio doubling estimator):")
print(f"  {'n₁→n₂':>12s}   {'α_eff':>10s}")
print("  " + "─" * 28)

doublings = [(5, 10), (10, 20), (15, 30), (20, 40), (25, 50), (30, 60),
             (40, 80), (50, 100), (60, 120), (80, 160), (100, 200),
             (125, 250), (150, 300), (200, 400), (250, 500), (300, 600),
             (400, 800), (500, 1000), (750, 1500), (1000, 2000),
             (1250, 2500), (1500, 3000), (2000, 4000)]

for n1, n2 in doublings:
    if n2 <= N_FOURIER:
        e1 = err_cinf[n1 - 1]
        e2 = err_cinf[n2 - 1]
        if e1 > 0 and e2 > 0 and e2 < e1:
            alpha = -mplog(e2 / e1) / mplog(2)
            print(f"  {n1:5d} → {n2:5d}   {mp.nstr(alpha, 6):>10s}")

# ── Additional test functions summary ────────────────────────────────────
if TEST_FUNCTIONS and extra_results:
    print()
    print("  Additional test functions (C∞ degenerate bump):")
    print(f"  {'Function':>18s}   {'ε_80':>14s}   {'ε_160':>14s}   {'ε_500':>14s}"
          f"   {'ε_1000':>14s}")
    print("  " + "─" * 72)
    for name, res in extra_results.items():
        cols = []
        for nn in [80, 160, 500, 1000]:
            if nn in res and res[nn] > 0:
                cols.append(f"{mp.nstr(res[nn], 6):>14s}")
            else:
                cols.append(f"{'—':>14s}")
        print(f"  {name:>18s}   {'   '.join(cols)}")

# ═══════════════════════════════════════════════════════════════════════════
#  CSV OUTPUT
# ═══════════════════════════════════════════════════════════════════════════
csv_path = os.path.join(OUTPUT_DIR, "convergence_extended.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['n', 'eps_cinf', 'eps_hann', 'eps_std'])
    for n in range(1, N_FOURIER + 1):
        e_c = mp.nstr(err_cinf[n - 1], 20, strip_zeros=False) if err_cinf[n - 1] > 0 else "0"
        e_h = f"{err_hann_np[n - 1]:.15e}"
        e_s = f"{err_std_np[n - 1]:.15e}"
        writer.writerow([n, e_c, e_h, e_s])

print()
print(f"  CSV saved to: {csv_path}")

# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE (optional)
# ═══════════════════════════════════════════════════════════════════════════
if MAKE_PLOT:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            'font.size': 12, 'axes.labelsize': 13, 'legend.fontsize': 10,
            'xtick.labelsize': 11, 'ytick.labelsize': 11,
            'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
            'font.family': 'serif',
        })

        ns = np.arange(1, N_FOURIER + 1)
        eps_c = np.array([float(e) if float(e) > 0 else np.nan
                          for e in err_cinf])

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.semilogy(ns, err_std_np[:N_FOURIER], 'r-', linewidth=1.0, alpha=0.8,
                     label=r'Standard truncation ($\sim n^{-1/2}$)')
        ax.semilogy(ns, err_hann_np[:N_FOURIER], '-', color='#ff7f0e',
                     linewidth=1.0, alpha=0.8,
                     label=r'Hann window $C^1$ ($\sim n^{-2.5}$)')
        ax.semilogy(ns, eps_c, 'b-', linewidth=1.5,
                     label=r'$C^\infty$ bump (super-algebraic)')

        # Reference lines
        n_ref = 5
        ref_val = eps_c[n_ref - 1]
        for s, ls in [(4, ':'), (8, '--'), (16, '-.')]:
            ref = ref_val * (n_ref / ns) ** s
            ax.semilogy(ns, ref, color='gray', linestyle=ls, alpha=0.3,
                         linewidth=0.7, label=f'$O(n^{{-{s}}})$ ref.')

        # Machine epsilon line
        ax.axhline(y=2.2e-16, color='red', linestyle=':', alpha=0.5, linewidth=0.8)
        ax.text(N_FOURIER * 0.75, 5e-16, 'float64 floor',
                color='red', fontsize=9, alpha=0.7)

        ax.set_xlabel('Number of retained Fourier coefficients $n$')
        ax.set_ylabel(r'$L^2$ error $\varepsilon_n$')
        ax.set_title(f'Extended convergence: $f(x)=x$ on $[-1,1]$, '
                     f'$N_{{\\mathrm{{grid}}}}={N_GRID}$, {DPS}-digit precision')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, which='both', ls='--', alpha=0.3)
        ax.set_xlim(1, N_FOURIER)

        fig.tight_layout()
        fig_path = os.path.join(OUTPUT_DIR, "figure_convergence_extended.png")
        fig.savefig(fig_path)
        print(f"  Figure saved to: {fig_path}")
    except Exception as e:
        print(f"  (Plot skipped: {e})")

# ═══════════════════════════════════════════════════════════════════════════
#  TIMING SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
total = time.time() - t_phase1
print()
print("=" * 72)
print(f"  Total execution time: {total:.0f} s  ({total / 60:.1f} min)")
print("=" * 72)
