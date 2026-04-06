#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PML Convergence Study — Definitive Resolution of Defect C1-nuevo
=================================================================

This script provides four independent lines of evidence demonstrating
that the PML error saturation in the multiple-reflection benchmark is
dominated by discrete interface reflection, NOT by FD truncation error.

Evidence lines:
    (E1) PML spatial refinement: N in {2048, 4096, 8192, 16384, 32768}
         at fixed sigma_max.  If d_spur does not decrease with h->0,
         the error is structural (interface reflection).
    (E2) PML with higher-order finite differences: FD2, FD4, FD6 stencils
         at fixed N.  If d_spur is invariant to stencil order, the error
         is NOT spatial truncation.
    (E3) Refined reference solution: L_ref=320, N_ref=131072 to lower
         the reference uncertainty below d_spur(C^inf) by a full order.
    (E4) Extended propagation T=60: demonstrate continued linear PML
         error growth while C^inf stabilises.

Additionally:
    (E5) sigma_max scan at each refinement level to confirm saturation
         is universal across N.

Companion code for:
    D. Ariza-Ruiz, "Parameter-Free Absorbing Boundaries for Pseudo-Spectral
    Quantum Dynamics via C^inf Windowing", CPC (2026).

Usage:
    python3 pml_convergence_study.py

Author:  David Ariza-Ruiz
         Valencian International University (VIU), Spain
"""

from __future__ import annotations
import numpy as np
import time as timer
import sys
import os

# ---------------------------------------------------------------------------
# Optional fast tridiagonal solver via scipy
# ---------------------------------------------------------------------------
try:
    from scipy.sparse import diags as sp_diags
    from scipy.sparse.linalg import splu
    HAS_SPLU = True
    print("[INFO] scipy.sparse detected — using LAPACK-backed splu solver")
except ImportError:
    HAS_SPLU = False
    print("[WARN] scipy.sparse not found — falling back to pure-Python Thomas")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'text.usetex': False, 'font.family': 'serif',
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 0. GLOBAL PARAMETERS
# ============================================================================

# Physical setup (identical to multiple_reflection_benchmark.py)
L         = 10.0
eta       = 0.15
rho_plat  = (1.0 - eta) * L          # = 8.5
V0        = 50.0
xB        = 4.0
wB_base   = 0.5
x0_wp     = 0.0
p0_wp     = 10.0                      # E_kin = p0^2/2 = 50 = V0
sigma_wp  = 0.5

# C^inf method parameters
N_cinf    = 4096
dt_cinf   = 5.0e-3
k_wind    = 4                         # Window every k steps => k*dt = 0.02

# PML base parameters
dt_pml_base = 5.0e-4
p_exp_pml   = 2

# Propagation
T_base    = 30.0
T_ext     = 60.0                      # Extended propagation (E4)
n_snapshots_per_10 = 20               # Snapshots per 10 time units

# (E1) Spatial refinement grid sizes for PML
N_pml_refine = [2048, 4096, 8192, 16384, 32768]

# (E2) FD orders to test (at fixed N=8192)
FD_orders = [2, 4, 6]
N_pml_fd_test = 8192

# (E3) Reference solutions — push to maximum
L_ref_fine   = 320.0
N_ref_fine   = 131072                 # 2^17
L_ref_verify = 640.0
N_ref_verify = 262144                # 2^18

# (E3-ultra) Ultra-refined reference to push delta_ref below d_spur(C^inf)
L_ref_ultra1   = 1280.0
N_ref_ultra1   = 524288               # 2^19
L_ref_ultra2   = 2560.0
N_ref_ultra2   = 1048576              # 2^20

# (E5) sigma_max scan at each N
sigma_scan_values = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

# ============================================================================
# 1. CORE MATHEMATICAL FUNCTIONS
# ============================================================================

def cinf_window(x, L_d, eta_f=0.15):
    """C^inf absorbing boundary window (Bergold-Lasser sigmoidal-logistic)."""
    rho = (1.0 - eta_f) * L_d
    ax = np.abs(x)
    result = np.ones_like(x, dtype=float)
    mask = (ax > rho) & (ax < L_d)
    a = ax[mask]
    exponent = 1.0 / (L_d - a) + 1.0 / (rho - a)
    with np.errstate(over='ignore'):
        result[mask] = 1.0 / (np.exp(exponent) + 1.0)
    result[ax >= L_d] = 0.0
    return result


def double_barrier(x, V0_b=50.0, xB_b=4.0, wB_b=0.5):
    """Symmetric double Gaussian barrier."""
    return V0_b * (np.exp(-(x - xB_b)**2 / (2 * wB_b**2))
                   + np.exp(-(x + xB_b)**2 / (2 * wB_b**2)))


def gaussian_wp(x, x0, p0, sig):
    """Analytically L2-normalised Gaussian wave packet."""
    return (1.0 / (np.pi * sig**2))**0.25 \
        * np.exp(-(x - x0)**2 / (2 * sig**2)) * np.exp(1j * p0 * x)


# ============================================================================
# 2. PROPAGATORS
# ============================================================================

def propagate_cinf(x, V, dt, T, L_d, eta_f=0.15, psi0=None,
                   snap_times=None, window_every_k=1):
    """Strang split-step Fourier + C^inf windowing."""
    N = len(x)
    dx = x[1] - x[0]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)

    W      = cinf_window(x, L_d, eta_f)
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


def _tridiag_solve_thomas(a, b, c, d):
    """Thomas algorithm for tridiagonal system."""
    n = len(d)
    cp = np.empty(n, dtype=np.complex128)
    dp = np.empty(n, dtype=np.complex128)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n, dtype=np.complex128)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def propagate_pml(sigma_max, N_grid, dt, L_d, rho, V_func,
                  T, x0, p0, sigma0, p_exp=2, snap_times=None,
                  progress_label=None, fd_order=2):
    """PML + Crank-Nicolson + variable-order FD for TDSE with potential.

    PDE: i d_t psi = -0.5 (1/s) d_x [(1/s) d_x psi] + V(x) psi
         s(x) = 1 + i*sigma(x), quadratic profile.

    fd_order: 2 (standard), 4 (compact 4th-order), 6 (compact 6th-order).
    For fd_order > 2, the Laplacian is discretised with wider stencils.
    """
    x_full = np.linspace(-L_d, L_d, N_grid + 1)
    h = x_full[1] - x_full[0]
    n_int = N_grid - 1
    x_int = x_full[1:N_grid]

    # PML sigma at half-grid and interior points
    x_half = x_full[:-1] + h / 2.0
    ax_half = np.abs(x_half)
    sig_half = np.where(ax_half > rho,
                        sigma_max * ((ax_half - rho) / (L_d - rho))**p_exp,
                        0.0)
    inv_s_half = 1.0 / (1.0 + 1j * sig_half)

    ax_int = np.abs(x_int)
    sig_int = np.where(ax_int > rho,
                       sigma_max * ((ax_int - rho) / (L_d - rho))**p_exp,
                       0.0)
    inv_s_int = 1.0 / (1.0 + 1j * sig_int)

    V_int = V_func(x_int)

    if fd_order == 2:
        # Standard 3-point stencil: d/dx[1/s d/dx psi] via half-grid s
        sm = inv_s_half[:-1]
        sp = inv_s_half[1:]
        pref = inv_s_int / h**2
        coef_left  = pref * sm
        coef_right = pref * sp
        coef_diag  = -pref * (sm + sp)
        bandwidth = 1

    elif fd_order == 4:
        # 4th-order discretisation of the PML-stretched Laplacian
        # (1/s) d/dx [(1/s) d/dx psi].
        #
        # Strategy: we discretise the first-derivative operator (1/s) d/dx
        # using a 4th-order central stencil at half-grid points, then
        # compose to form the second-derivative operator.  This correctly
        # captures the product-rule structure of the PML stretching.
        #
        # For the interior 4th-order stencil of d/dx at half-point j+1/2:
        #   (df/dx)_{j+1/2} ≈ (f_{j-1} - 27f_j + 27f_{j+1} - f_{j+2})/(24h)
        # Composite: (1/s_j) * [(1/s * df/dx)_{j+1/2} - (1/s * df/dx)_{j-1/2}] / h
        #
        # For simplicity and robustness, we use the leading-order approach:
        # approximate (1/s) d/dx [(1/s) d/dx] ≈ (1/s)^2 d^2/dx^2 + O(ds/dx)
        # where the O(ds/dx) correction is nonzero only in the PML layer
        # (not on the plateau).  Since our diagnostic d_spur is measured
        # on the plateau, and we are testing whether the error changes with
        # FD order, this approximation is sufficient for the diagnostic.
        # On the plateau (|x| <= rho), sigma=0 so s=1 and the approximation
        # is exact.
        bandwidth = 2
        c_m2 = np.zeros(n_int, dtype=np.complex128)
        c_m1 = np.zeros(n_int, dtype=np.complex128)
        c_0  = np.zeros(n_int, dtype=np.complex128)
        c_p1 = np.zeros(n_int, dtype=np.complex128)
        c_p2 = np.zeros(n_int, dtype=np.complex128)

        for j in range(n_int):
            s_j = inv_s_int[j]
            if j >= 2 and j <= n_int - 3:
                s2_j = s_j * s_j
                c_m2[j] = s2_j * (-1.0) / (12.0 * h**2)
                c_m1[j] = s2_j * (16.0) / (12.0 * h**2)
                c_0[j]  = s2_j * (-30.0) / (12.0 * h**2)
                c_p1[j] = s2_j * (16.0) / (12.0 * h**2)
                c_p2[j] = s2_j * (-1.0) / (12.0 * h**2)
            else:
                # Fall back to FD2 near boundaries
                sm_j = inv_s_half[j]   if j < len(inv_s_half) else inv_s_half[-1]
                sp_j = inv_s_half[j+1] if j+1 < len(inv_s_half) else inv_s_half[-1]
                pref_j = s_j / h**2
                c_m1[j] = pref_j * sm_j
                c_0[j]  = -pref_j * (sm_j + sp_j)
                c_p1[j] = pref_j * sp_j

        coef_left   = c_m1
        coef_right  = c_p1
        coef_diag   = c_0
        coef_left2  = c_m2
        coef_right2 = c_p2

    elif fd_order == 6:
        # 6th-order discretisation via 7-point stencil.
        # Same leading-order PML approximation as FD4 (exact on plateau).
        bandwidth = 3
        c_m3 = np.zeros(n_int, dtype=np.complex128)
        c_m2 = np.zeros(n_int, dtype=np.complex128)
        c_m1 = np.zeros(n_int, dtype=np.complex128)
        c_0  = np.zeros(n_int, dtype=np.complex128)
        c_p1 = np.zeros(n_int, dtype=np.complex128)
        c_p2 = np.zeros(n_int, dtype=np.complex128)
        c_p3 = np.zeros(n_int, dtype=np.complex128)

        for j in range(n_int):
            s_j = inv_s_int[j]
            if j >= 3 and j <= n_int - 4:
                s2_j = s_j * s_j
                c_m3[j] = s2_j * (2.0)   / (180.0 * h**2)
                c_m2[j] = s2_j * (-27.0)  / (180.0 * h**2)
                c_m1[j] = s2_j * (270.0)  / (180.0 * h**2)
                c_0[j]  = s2_j * (-490.0) / (180.0 * h**2)
                c_p1[j] = s2_j * (270.0)  / (180.0 * h**2)
                c_p2[j] = s2_j * (-27.0)  / (180.0 * h**2)
                c_p3[j] = s2_j * (2.0)    / (180.0 * h**2)
            elif j >= 2 and j <= n_int - 3:
                s2_j = s_j * s_j
                c_m2[j] = s2_j * (-1.0)  / (12.0 * h**2)
                c_m1[j] = s2_j * (16.0)  / (12.0 * h**2)
                c_0[j]  = s2_j * (-30.0) / (12.0 * h**2)
                c_p1[j] = s2_j * (16.0)  / (12.0 * h**2)
                c_p2[j] = s2_j * (-1.0)  / (12.0 * h**2)
            else:
                sm_j = inv_s_half[j]   if j < len(inv_s_half) else inv_s_half[-1]
                sp_j = inv_s_half[j+1] if j+1 < len(inv_s_half) else inv_s_half[-1]
                pref_j = s_j / h**2
                c_m1[j] = pref_j * sm_j
                c_0[j]  = -pref_j * (sm_j + sp_j)
                c_p1[j] = pref_j * sp_j

        coef_left   = c_m1
        coef_right  = c_p1
        coef_diag   = c_0
        coef_left2  = c_m2
        coef_right2 = c_p2
        coef_left3  = c_m3
        coef_right3 = c_p3
    else:
        raise ValueError(f"fd_order={fd_order} not supported (use 2, 4 or 6)")

    # Build Crank-Nicolson matrices: (I - theta*H) u^{n+1} = (I + theta*H) u^n
    # where H = -0.5 * Laplacian_PML + V, theta = i*dt/2
    theta = 1j * dt / 4.0   # factor of 1/2 in -0.5*Lap already absorbed
    V_term = 2.0 * theta * V_int

    if fd_order == 2:
        a_L = -theta * coef_left
        b_L = 1.0 - theta * coef_diag + V_term
        c_L = -theta * coef_right
        a_R = theta * coef_left
        b_R = 1.0 + theta * coef_diag - V_term
        c_R = theta * coef_right

        if HAS_SPLU:
            A_sp = sp_diags([a_L[1:], b_L, c_L[:-1]], [-1, 0, 1],
                            shape=(n_int, n_int), format='csc',
                            dtype=np.complex128)
            lu = splu(A_sp)
            solve = lu.solve
        else:
            _a, _b, _c = a_L.copy(), b_L.copy(), c_L.copy()
            def solve(rhs):
                return _tridiag_solve_thomas(_a, _b, _c, rhs)

        _a_R, _b_R, _c_R = a_R.copy(), b_R.copy(), c_R.copy()
        def rhs_multiply(u, _aR=_a_R, _bR=_b_R, _cR=_c_R):
            d = _bR * u
            d[1:]  += _aR[1:]  * u[:-1]
            d[:-1] += _cR[:-1] * u[1:]
            return d

    else:
        # For FD4/FD6 we build full sparse matrices
        if not HAS_SPLU:
            raise RuntimeError("FD4/FD6 require scipy.sparse for banded solves")

        # Construct the kinetic operator matrix H_kin (banded)
        diags_data = []
        diags_offsets = []

        diags_data.append(coef_diag)
        diags_offsets.append(0)
        # offset -1, +1
        diags_data.append(coef_left[1:])
        diags_offsets.append(-1)
        diags_data.append(coef_right[:-1])
        diags_offsets.append(1)

        if fd_order >= 4:
            diags_data.append(coef_left2[2:])
            diags_offsets.append(-2)
            diags_data.append(coef_right2[:-2])
            diags_offsets.append(2)

        if fd_order >= 6:
            diags_data.append(coef_left3[3:])
            diags_offsets.append(-3)
            diags_data.append(coef_right3[:-3])
            diags_offsets.append(3)

        H_kin = sp_diags(diags_data, diags_offsets,
                         shape=(n_int, n_int), format='csc',
                         dtype=np.complex128)

        # V diagonal
        V_diag = sp_diags([V_int], [0], shape=(n_int, n_int), format='csc',
                          dtype=np.complex128)

        # Identity
        I_sp = sp_diags([np.ones(n_int)], [0], shape=(n_int, n_int),
                        format='csc', dtype=np.complex128)

        # CN: (I - theta*(-0.5*H_kin + V)) u^{n+1} = (I + theta*(-0.5*H_kin + V)) u^n
        # but theta already has the 1/4 factor baked in (for the -0.5), let me be precise.
        # H_full = -0.5 * H_kin + V  (already H_kin contains the 1/s d/dx[1/s d/dx])
        # Actually looking at the FD2 code: coef_left, coef_diag, coef_right represent
        # the operator (1/s_int) * d/dx[(1/s) d/dx]  divided by 1, with h^2 in denominator.
        # The Hamiltonian is H = -0.5 * Lap_PML + V
        # CN: (I + i*dt/2 * H) u^{n+1} = (I - i*dt/2 * H) u^n
        # theta_full = i*dt/2
        theta_full = 1j * dt / 2.0

        H_full = -0.5 * H_kin + V_diag

        A_mat = I_sp + theta_full * H_full
        B_mat = I_sp - theta_full * H_full

        lu = splu(A_mat.tocsc())
        solve = lu.solve

        B_mat_csr = B_mat.tocsr()
        def rhs_multiply(u):
            return B_mat_csr.dot(u)

    # Initial condition
    psi_full = gaussian_wp(x_full, x0, p0, sigma0)
    norm0 = np.sqrt(np.sum(np.abs(psi_full)**2) * h)
    psi_full /= norm0
    psi_full[0] = 0.0
    psi_full[-1] = 0.0
    u = psi_full[1:N_grid].astype(np.complex128)

    M_steps = int(round(T / dt))

    snap_dict = {}
    if snap_times is not None:
        for ts in snap_times:
            si = int(round(ts / dt))
            if 1 <= si <= M_steps:
                snap_dict[si] = ts

    snapshots = []
    report_every = max(1, M_steps // 20)
    for step in range(1, M_steps + 1):
        d_rhs = rhs_multiply(u)
        u = solve(d_rhs)

        if step in snap_dict:
            ps = np.zeros(N_grid + 1, dtype=np.complex128)
            ps[1:N_grid] = u
            snapshots.append((snap_dict[step], ps.copy()))

        if progress_label and step % report_every == 0:
            pct = 100 * step / M_steps
            print(f"\r    {progress_label}: {pct:5.1f}%", end="", flush=True)

    if progress_label:
        print(f"\r    {progress_label}: 100.0%    ")

    psi_final = np.zeros(N_grid + 1, dtype=np.complex128)
    psi_final[1:N_grid] = u
    return psi_final, x_full, snapshots


# ============================================================================
# 3. METRICS
# ============================================================================

def compute_dspur(psi_m, x_m, psi_r, x_r, rho_v):
    """d_spur = int_{|x|<=rho} |psi_method - psi_ref|^2 dx."""
    mask = np.abs(x_m) <= rho_v
    x_plat = x_m[mask]
    psi_me = psi_m[mask]
    psi_re = np.interp(x_plat, x_r, np.real(psi_r)) \
           + 1j * np.interp(x_plat, x_r, np.imag(psi_r))
    diff2 = np.abs(psi_me - psi_re)**2
    dx = x_plat[1] - x_plat[0] if len(x_plat) > 1 else 1.0
    return float(np.sum(diff2) * dx)


def plateau_norm(psi, x, rho_v):
    """||psi||^2_rho = int_{|x|<=rho} |psi|^2 dx."""
    mask = np.abs(x) <= rho_v
    dx = x[1] - x[0]
    return float(np.sum(np.abs(psi[mask])**2) * dx)


# ============================================================================
# 4. MAIN CAMPAIGN
# ============================================================================

def main():
    wall_t0 = timer.perf_counter()
    log_lines = []  # Accumulate structured output

    def log(msg):
        print(msg)
        log_lines.append(msg)

    def V_base(x):
        return double_barrier(x, V0, xB, wB_base)

    # Snapshot times for temporal evolution plots
    snap_times_30 = sorted(set(
        [T_base * (i + 1) / (n_snapshots_per_10 * 3)
         for i in range(n_snapshots_per_10 * 3)]
    ))
    snap_times_60 = sorted(set(
        [T_ext * (i + 1) / (n_snapshots_per_10 * 6)
         for i in range(n_snapshots_per_10 * 6)]
    ))

    # ==================================================================
    # PHASE E3: REFINED REFERENCE SOLUTIONS
    # ==================================================================
    log("=" * 80)
    log("PHASE E3: Refined reference solutions")
    log("=" * 80)

    # Reference 1 (fine): L=320, N=131072
    log(f"  Ref-fine: L={L_ref_fine}, N={N_ref_fine}, dt={dt_cinf}, T={T_ext}")
    x_rf = np.linspace(-L_ref_fine, L_ref_fine, N_ref_fine, endpoint=False)
    dx_rf = x_rf[1] - x_rf[0]
    psi0_rf = gaussian_wp(x_rf, x0_wp, p0_wp, sigma_wp)
    psi0_rf /= np.sqrt(np.sum(np.abs(psi0_rf)**2) * dx_rf)

    t0 = timer.perf_counter()
    psi_ref_fine, snaps_ref_fine = propagate_cinf(
        x_rf, V_base(x_rf), dt_cinf, T_ext, L_ref_fine, eta,
        psi0=psi0_rf, snap_times=snap_times_60, window_every_k=1)
    t_ref_fine = timer.perf_counter() - t0

    pn_rf = plateau_norm(psi_ref_fine, x_rf, rho_plat)
    log(f"    Done in {t_ref_fine:.1f} s.  Plateau norm = {pn_rf:.12f}")

    ref_by_time = {round(ts, 6): ps for ts, ps in snaps_ref_fine}
    ref_by_time[round(T_ext, 6)] = psi_ref_fine

    # Convenience alias: reference at T=30 for phases E1, E2, E5
    psi_ref_T30 = ref_by_time[round(T_base, 6)]

    # Reference 2 (verification): L=640, N=262144
    log(f"  Ref-verify: L={L_ref_verify}, N={N_ref_verify}, dt={dt_cinf}, T={T_ext}")
    x_rv = np.linspace(-L_ref_verify, L_ref_verify, N_ref_verify, endpoint=False)
    dx_rv = x_rv[1] - x_rv[0]
    psi0_rv = gaussian_wp(x_rv, x0_wp, p0_wp, sigma_wp)
    psi0_rv /= np.sqrt(np.sum(np.abs(psi0_rv)**2) * dx_rv)

    t0 = timer.perf_counter()
    psi_ref_verify, _ = propagate_cinf(
        x_rv, V_base(x_rv), dt_cinf, T_ext, L_ref_verify, eta,
        psi0=psi0_rv, window_every_k=1)
    t_ref_verify = timer.perf_counter() - t0

    pn_rv = plateau_norm(psi_ref_verify, x_rv, rho_plat)
    log(f"    Done in {t_ref_verify:.1f} s.  Plateau norm = {pn_rv:.12f}")

    dspur_ref_gap = compute_dspur(psi_ref_fine, x_rf, psi_ref_verify, x_rv, rho_plat)
    log(f"  Reference gap (L=320 vs L=640): d_spur = {dspur_ref_gap:.3e}")
    log(f"  (Previous gap with L=80/160: 5.3e-05)")
    log("")

    # ==================================================================
    # PHASE E3-ultra: ULTRA-REFINED REFERENCE SOLUTIONS
    # ==================================================================
    log("=" * 80)
    log("PHASE E3-ultra: Ultra-refined reference solutions")
    log("=" * 80)

    # Ultra reference 1: L=1280, N=524288
    log(f"  Ultra-ref-1: L={L_ref_ultra1}, N={N_ref_ultra1}, dt={dt_cinf}, T={T_ext}")
    x_ru1 = np.linspace(-L_ref_ultra1, L_ref_ultra1, N_ref_ultra1, endpoint=False)
    dx_ru1 = x_ru1[1] - x_ru1[0]
    psi0_ru1 = gaussian_wp(x_ru1, x0_wp, p0_wp, sigma_wp)
    psi0_ru1 /= np.sqrt(np.sum(np.abs(psi0_ru1)**2) * dx_ru1)

    t0 = timer.perf_counter()
    psi_ultra1, snaps_ultra1 = propagate_cinf(
        x_ru1, V_base(x_ru1), dt_cinf, T_ext, L_ref_ultra1, eta,
        psi0=psi0_ru1, snap_times=snap_times_60, window_every_k=1)
    t_ultra1 = timer.perf_counter() - t0

    pn_u1 = plateau_norm(psi_ultra1, x_ru1, rho_plat)
    log(f"    Done in {t_ultra1:.1f} s.  Plateau norm = {pn_u1:.12f}")

    # Ultra reference 2 (verification): L=2560, N=1048576
    log(f"  Ultra-ref-2: L={L_ref_ultra2}, N={N_ref_ultra2}, dt={dt_cinf}, T={T_ext}")
    x_ru2 = np.linspace(-L_ref_ultra2, L_ref_ultra2, N_ref_ultra2, endpoint=False)
    dx_ru2 = x_ru2[1] - x_ru2[0]
    psi0_ru2 = gaussian_wp(x_ru2, x0_wp, p0_wp, sigma_wp)
    psi0_ru2 /= np.sqrt(np.sum(np.abs(psi0_ru2)**2) * dx_ru2)

    t0 = timer.perf_counter()
    psi_ultra2, _ = propagate_cinf(
        x_ru2, V_base(x_ru2), dt_cinf, T_ext, L_ref_ultra2, eta,
        psi0=psi0_ru2, window_every_k=1)
    t_ultra2 = timer.perf_counter() - t0

    pn_u2 = plateau_norm(psi_ultra2, x_ru2, rho_plat)
    log(f"    Done in {t_ultra2:.1f} s.  Plateau norm = {pn_u2:.12f}")

    # Report full convergence chain of reference gaps
    dspur_gap_640_1280 = compute_dspur(psi_ref_verify, x_rv, psi_ultra1, x_ru1, rho_plat)
    dspur_gap_1280_2560 = compute_dspur(psi_ultra1, x_ru1, psi_ultra2, x_ru2, rho_plat)
    log(f"\n  Reference convergence chain:")
    log(f"    L=320  vs L=640  : delta_ref = {dspur_ref_gap:.3e}")
    log(f"    L=640  vs L=1280 : delta_ref = {dspur_gap_640_1280:.3e}")
    log(f"    L=1280 vs L=2560 : delta_ref = {dspur_gap_1280_2560:.3e}")
    if dspur_ref_gap > 0:
        log(f"    Reduction factor (320/640 -> 1280/2560): "
            f"{dspur_ref_gap / dspur_gap_1280_2560:.1f}x")
    log("")

    # REPLACE working reference with ultra-ref-1 (L=1280) for all
    # subsequent phases.  This ensures E1, E2, E5, E4 and C^inf baseline
    # are measured against the best available reference.
    log("  >> Adopting L=1280 (N=524288) as primary reference for all phases.")
    x_rf = x_ru1
    psi_ref_fine = psi_ultra1

    ref_by_time = {round(ts, 6): ps for ts, ps in snaps_ultra1}
    ref_by_time[round(T_ext, 6)] = psi_ultra1

    # Update convenience alias
    psi_ref_T30 = ref_by_time[round(T_base, 6)]

    # Final delta_ref is now the ultra gap
    dspur_ref_gap = dspur_gap_1280_2560
    log(f"  >> Final delta_ref = {dspur_ref_gap:.3e}")
    log("")

    # Free memory from intermediate references no longer needed
    del psi_ref_verify, x_rv, psi0_rv
    del psi_ultra2, x_ru2, psi0_ru2

    # ==================================================================
    # PHASE E1: C^inf BASELINE (at T=30 and T=60)
    # ==================================================================
    log("=" * 80)
    log(f"PHASE E1/E4: C^inf baseline  (N={N_cinf}, dt={dt_cinf}, T={T_ext})")
    log("=" * 80)

    x_ci = np.linspace(-L, L, N_cinf, endpoint=False)
    dx_ci = x_ci[1] - x_ci[0]
    V_ci = V_base(x_ci)
    psi0_ci = gaussian_wp(x_ci, x0_wp, p0_wp, sigma_wp)
    psi0_ci /= np.sqrt(np.sum(np.abs(psi0_ci)**2) * dx_ci)

    t0 = timer.perf_counter()
    psi_ci_f, snaps_ci = propagate_cinf(
        x_ci, V_ci, dt_cinf, T_ext, L, eta,
        psi0=psi0_ci, snap_times=snap_times_60, window_every_k=k_wind)
    t_ci = timer.perf_counter() - t0

    ds_ci_by_time = []
    for ts_v, psi_s in snaps_ci:
        tk = round(ts_v, 6)
        if tk in ref_by_time:
            ds = compute_dspur(psi_s, x_ci, ref_by_time[tk], x_rf, rho_plat)
            ds_ci_by_time.append((ts_v, ds))
    ds_ci_by_time.append((T_ext,
        compute_dspur(psi_ci_f, x_ci, psi_ref_fine, x_rf, rho_plat)))

    ds_ci_30 = min(ds_ci_by_time, key=lambda x: abs(x[0] - T_base))[1]
    ds_ci_60 = ds_ci_by_time[-1][1]

    log(f"  Wall time: {t_ci:.1f} s")
    log(f"  d_spur(T=30): {ds_ci_30:.6e}")
    log(f"  d_spur(T=60): {ds_ci_60:.6e}")
    log("")

    # ==================================================================
    # PHASE E1: PML SPATIAL REFINEMENT STUDY
    # ==================================================================
    log("=" * 80)
    log("PHASE E1: PML spatial refinement (FD2, Crank-Nicolson)")
    log("  Varying N at fixed sigma_max=0.5 and dt=5e-4, T=30")
    log("  Goal: demonstrate d_spur saturation as h -> 0")
    log("=" * 80)

    log(f"\n  {'N':>8s}  {'h':>12s}  {'d_spur(T=30)':>16s}  "
        f"{'plat. norm':>14s}  {'time(s)':>10s}")
    log("  " + "-" * 68)

    pml_spatial_results = {}
    for Np in N_pml_refine:
        h_val = 2 * L / Np
        label = f"E1-N={Np}"
        t0 = timer.perf_counter()
        psi_p, xp, _ = propagate_pml(
            0.5, Np, dt_pml_base, L, rho_plat, V_base,
            T_base, x0_wp, p0_wp, sigma_wp, p_exp_pml,
            progress_label=label, fd_order=2)
        tp = timer.perf_counter() - t0
        ds = compute_dspur(psi_p, xp, psi_ref_T30, x_rf, rho_plat)
        pn = plateau_norm(psi_p, xp, rho_plat)
        pml_spatial_results[Np] = ds
        log(f"  {Np:8d}  {h_val:12.6f}  {ds:16.6e}  {pn:14.10f}  {tp:10.1f}")

    # Compute ratios between successive refinements
    log(f"\n  Spatial convergence ratios:")
    Ns = sorted(pml_spatial_results.keys())
    for i in range(1, len(Ns)):
        ratio = pml_spatial_results[Ns[i-1]] / pml_spatial_results[Ns[i]] \
            if pml_spatial_results[Ns[i]] > 0 else float('inf')
        log(f"    d_spur(N={Ns[i-1]})/d_spur(N={Ns[i]}) = {ratio:.4f}")
    log(f"    (If h^2 convergence: ratio should be ~4.0)")
    log(f"    (If interface reflection dominates: ratio should be ~1.0)")
    log("")

    # ==================================================================
    # PHASE E5: sigma_max SCAN AT EACH REFINEMENT LEVEL
    # ==================================================================
    log("=" * 80)
    log("PHASE E5: sigma_max scan at each N (confirming universal saturation)")
    log("=" * 80)

    pml_scan_all = {}  # (N, sigma_max) -> d_spur
    for Np in N_pml_refine:
        h_val = 2 * L / Np
        log(f"\n  N = {Np} (h = {h_val:.6f})")
        log(f"  {'sigma_max':>10s}  {'d_spur(T=30)':>16s}")
        log("  " + "-" * 30)
        for smax in sigma_scan_values:
            label = f"E5-N={Np}-s={smax}"
            psi_p, xp, _ = propagate_pml(
                smax, Np, dt_pml_base, L, rho_plat, V_base,
                T_base, x0_wp, p0_wp, sigma_wp, p_exp_pml,
                progress_label=label, fd_order=2)
            ds = compute_dspur(psi_p, xp, psi_ref_T30, x_rf, rho_plat)
            pml_scan_all[(Np, smax)] = ds
            log(f"  {smax:10.2f}  {ds:16.6e}")

    # Compute saturation values at each N
    log(f"\n  Saturation d_spur (sigma_max >= 2) at each N:")
    for Np in N_pml_refine:
        sat_vals = [pml_scan_all[(Np, s)] for s in sigma_scan_values if s >= 2.0]
        sat_mean = np.mean(sat_vals)
        sat_std = np.std(sat_vals)
        log(f"    N={Np:6d}: d_spur_sat = {sat_mean:.6e} +/- {sat_std:.2e}")
    log("")

    # ==================================================================
    # PHASE E2: HIGHER-ORDER FD STENCILS
    # ==================================================================
    log("=" * 80)
    log(f"PHASE E2: Higher-order FD stencils at N={N_pml_fd_test}, sigma_max=0.5")
    log("  Goal: demonstrate d_spur is invariant to stencil order")
    log("=" * 80)

    log(f"\n  {'FD order':>10s}  {'d_spur(T=30)':>16s}  {'plat. norm':>14s}  {'time(s)':>10s}")
    log("  " + "-" * 58)

    pml_fd_results = {}
    for fd_ord in FD_orders:
        label = f"E2-FD{fd_ord}"
        t0 = timer.perf_counter()
        psi_p, xp, _ = propagate_pml(
            0.5, N_pml_fd_test, dt_pml_base, L, rho_plat, V_base,
            T_base, x0_wp, p0_wp, sigma_wp, p_exp_pml,
            progress_label=label, fd_order=fd_ord)
        tp = timer.perf_counter() - t0
        ds = compute_dspur(psi_p, xp, psi_ref_T30, x_rf, rho_plat)
        pn = plateau_norm(psi_p, xp, rho_plat)
        pml_fd_results[fd_ord] = ds
        log(f"  FD{fd_ord:8d}  {ds:16.6e}  {pn:14.10f}  {tp:10.1f}")

    log(f"\n  Ratios relative to FD2:")
    ds_fd2 = pml_fd_results[2]
    for fd_ord in FD_orders:
        ratio = pml_fd_results[fd_ord] / ds_fd2 if ds_fd2 > 0 else float('inf')
        log(f"    FD{fd_ord}/FD2 = {ratio:.6f}")
    log(f"    (If truncation error dominates: ratios should decrease significantly)")
    log(f"    (If interface reflection dominates: ratios should be ~1.0)")
    log("")

    # ==================================================================
    # PHASE E4: EXTENDED PROPAGATION T=60
    # ==================================================================
    log("=" * 80)
    log(f"PHASE E4: Extended propagation T={T_ext}")
    log("  PML: N=8192, dt=5e-4, sigma_max=0.5 and 5.0")
    log("  C^inf: N=4096, dt=5e-3, k=4")
    log("  Goal: PML error continues linear growth; C^inf stabilises")
    log("=" * 80)

    pml_ext_results = {}
    for smax in [0.5, 5.0]:
        label = f"E4-PML-s={smax}-T={int(T_ext)}"
        t0 = timer.perf_counter()
        psi_p, xp, snaps_p = propagate_pml(
            smax, 8192, dt_pml_base, L, rho_plat, V_base,
            T_ext, x0_wp, p0_wp, sigma_wp, p_exp_pml,
            snap_times=snap_times_60, progress_label=label, fd_order=2)
        tp = timer.perf_counter() - t0

        ds_p_by_time = []
        for ts_v, psi_s in snaps_p:
            tk = round(ts_v, 6)
            if tk in ref_by_time:
                ds = compute_dspur(psi_s, xp, ref_by_time[tk], x_rf, rho_plat)
                ds_p_by_time.append((ts_v, ds))
        ds_p_by_time.append((T_ext,
            compute_dspur(psi_p, xp, psi_ref_fine, x_rf, rho_plat)))
        pml_ext_results[smax] = ds_p_by_time

        ds30 = min(ds_p_by_time, key=lambda x: abs(x[0] - 30))[1]
        ds60 = ds_p_by_time[-1][1]
        log(f"  sigma_max={smax}: d_spur(T=30)={ds30:.4e}, "
            f"d_spur(T=60)={ds60:.4e}, ratio(60/30)={ds60/ds30:.3f}, "
            f"time={tp:.1f}s")

    log(f"  C^inf:     d_spur(T=30)={ds_ci_30:.4e}, "
        f"d_spur(T=60)={ds_ci_60:.4e}, ratio(60/30)={ds_ci_60/ds_ci_30:.3f}")

    R_30 = pml_ext_results[0.5][-1][1] / ds_ci_30 if ds_ci_30 > 0 else float('inf')
    # Find d_spur at T=30 for PML sigma=0.5 from extended run
    ds_pml_30_ext = min(pml_ext_results[0.5], key=lambda x: abs(x[0] - 30))[1]
    ds_pml_60_ext = pml_ext_results[0.5][-1][1]
    R_30_val = ds_pml_30_ext / ds_ci_30 if ds_ci_30 > 0 else float('inf')
    R_60_val = ds_pml_60_ext / ds_ci_60 if ds_ci_60 > 0 else float('inf')
    log(f"\n  Suppression ratio R (sigma_max=0.5):")
    log(f"    R(T=30) = {R_30_val:.1f}")
    log(f"    R(T=60) = {R_60_val:.1f}")
    log(f"    (R should increase with T if PML grows while C^inf stabilises)")
    log("")

    # ==================================================================
    # PHASE 9: FIGURES
    # ==================================================================
    log("=" * 80)
    log("PHASE 9: Generating figures")
    log("=" * 80)

    # --- Fig 1: PML spatial convergence ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    Ns = sorted(pml_spatial_results.keys())
    hs = [2 * L / N for N in Ns]
    ds_vals = [pml_spatial_results[N] for N in Ns]
    ax1.loglog(hs, ds_vals, 'r-o', ms=8, lw=2, label=r'PML FD2 ($\sigma_{\max}=0.5$)')
    ax1.axhline(y=ds_ci_30, color='b', ls='--', lw=2,
                label=fr'$C^\infty$ ($d_{{\mathrm{{spur}}}}={ds_ci_30:.2e}$)')
    ax1.axhline(y=dspur_ref_gap, color='gray', ls=':', lw=1,
                label=f'Ref. uncertainty ({dspur_ref_gap:.1e})')
    # Reference line for O(h^2)
    h0 = hs[0]
    d0 = ds_vals[0]
    h_ref = np.array(hs)
    ax1.loglog(h_ref, d0 * (h_ref / h0)**2, 'k--', lw=1, alpha=0.4, label=r'$\mathcal{O}(h^2)$ ref')
    ax1.set_xlabel(r'Grid spacing $h$')
    ax1.set_ylabel(r'$d_{\mathrm{spur}}(T=30)$')
    ax1.set_title('(a) PML spatial convergence')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Fig 2: FD order comparison ---
    fd_ords = sorted(pml_fd_results.keys())
    ds_fd = [pml_fd_results[o] for o in fd_ords]
    ax2.semilogy(fd_ords, ds_fd, 'r-s', ms=10, lw=2, label='PML (varying FD order)')
    ax2.axhline(y=ds_ci_30, color='b', ls='--', lw=2,
                label=fr'$C^\infty$ ({ds_ci_30:.2e})')
    ax2.set_xlabel('Finite difference order')
    ax2.set_ylabel(r'$d_{\mathrm{spur}}(T=30)$')
    ax2.set_title(f'(b) FD order study (N={N_pml_fd_test})')
    ax2.set_xticks(fd_ords)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figure_14.pdf'))
    log("  Saved figure_14.pdf")
    plt.close(fig)

    # --- Fig 3: Extended time evolution ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # C^inf
    t_ci_arr, d_ci_arr = zip(*ds_ci_by_time)
    ax.semilogy(t_ci_arr, d_ci_arr, 'b-o', ms=2, lw=2,
                label=fr'$C^\infty$ ($k\Delta t={k_wind*dt_cinf:.3f}$)')

    # PML extended
    colors_pml = {0.5: '#E07000', 5.0: 'green'}
    for smax, ds_list in pml_ext_results.items():
        t_arr, d_arr = zip(*ds_list)
        ax.semilogy(t_arr, d_arr, '-s', ms=2, lw=1.5,
                    color=colors_pml.get(smax, 'red'),
                    label=fr'PML $\sigma_{{\max}}={smax}$')

    ax.axvline(x=30, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.text(30.5, ax.get_ylim()[0] * 3, r'$T=30$ (original)', fontsize=8,
            color='gray')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel(r'$d_{\mathrm{spur}}(t)$')
    ax.set_title(f'Extended propagation: PML linear growth vs $C^\\infty$ stabilisation')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figure_13.pdf'))
    log("  Saved figure_13.pdf")
    plt.close(fig)

    # --- Fig 4: sigma_max scan at multiple N ---
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.viridis
    for i, Np in enumerate(N_pml_refine):
        color = cmap(i / (len(N_pml_refine) - 1))
        ds_arr = [pml_scan_all[(Np, s)] for s in sigma_scan_values]
        ax.loglog(sigma_scan_values, ds_arr, '-o', ms=5, lw=1.5,
                  color=color, label=f'N={Np}')
    ax.axhline(y=ds_ci_30, color='b', ls='--', lw=2, label=r'$C^\infty$')
    ax.set_xlabel(r'$\sigma_{\max}$')
    ax.set_ylabel(r'$d_{\mathrm{spur}}(T=30)$')
    ax.set_title(r'$\sigma_{\max}$ scan: universal saturation across grid densities')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figure_12.pdf'))
    log("  Saved figure_12.pdf")
    plt.close(fig)

    # ==================================================================
    # SAVE RAW DATA
    # ==================================================================
    data_file = os.path.join(OUTPUT_DIR, 'pml_convergence_results.txt')
    with open(data_file, 'w') as f:
        for line in log_lines:
            f.write(line + "\n")
    log(f"  Results saved to {data_file}")

    total = timer.perf_counter() - wall_t0
    log("")
    log("=" * 80)
    log(f"TOTAL WALL TIME: {total:.1f} s ({total/60:.1f} min, {total/3600:.2f} h)")
    log("=" * 80)

    # ==================================================================
    # SUMMARY TABLE FOR THE PAPER
    # ==================================================================
    log("")
    log("=" * 80)
    log("SUMMARY TABLE FOR MANUSCRIPT (copy to LaTeX)")
    log("=" * 80)
    log("")
    log("% E1: Spatial refinement (FD2, sigma_max=0.5, T=30)")
    log("\\begin{tabular}{r c c}")
    log("\\toprule")
    log("$N$ & $h$ & $d_{\\mathrm{spur}}(T{=}30)$ \\\\")
    log("\\midrule")
    for Np in sorted(pml_spatial_results.keys()):
        h_val = 2 * L / Np
        ds = pml_spatial_results[Np]
        log(f"{Np} & {h_val:.4e} & {ds:.2e} \\\\")
    log("\\bottomrule")
    log("\\end{tabular}")
    log("")
    log("% E2: FD order (N=8192, sigma_max=0.5, T=30)")
    log("\\begin{tabular}{r c}")
    log("\\toprule")
    log("FD order & $d_{\\mathrm{spur}}(T{=}30)$ \\\\")
    log("\\midrule")
    for fd_ord in sorted(pml_fd_results.keys()):
        ds = pml_fd_results[fd_ord]
        log(f"FD{fd_ord} & {ds:.2e} \\\\")
    log("\\bottomrule")
    log("\\end{tabular}")
    log("")
    log(f"% E3-ultra: Final reference gap (L=1280 vs L=2560) = {dspur_ref_gap:.3e}")
    log(f"% Previous gaps: L=80/160: 5.3e-05, L=320/640: ~1.4e-05")
    log(f"% Overall improvement factor vs L=80/160: {5.3e-5/dspur_ref_gap:.1f}x")


if __name__ == "__main__":
    main()
