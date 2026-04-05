#!/usr/bin/env python3
"""
pml_benchmark_1d.py
===================
Benchmarks a 1D PML absorber for the free-particle Schrödinger equation
against the C∞ windowed split-step method.

Equation: i∂ₜψ = -(1/2)∂²ₓψ  (ℏ = 1, m = 1)

PML formulation (Nissen & Kreiss 2011, Zheng 2007):
  i∂ₜψ + (1/2)(1/s)∂ₓ[(1/s)∂ₓψ] = 0
  s(x) = 1 + iσ(x),  σ(x) = σ_max·((|x|−ρ)/(L−ρ))^p  for |x| > ρ

Time stepping: Crank-Nicolson (unconditionally stable).
Spatial discretization: 2nd-order central finite differences.
Boundary conditions: Dirichlet ψ(±L) = 0.

The PML solver uses dt_pml = 0.0005 to suppress Crank-Nicolson
temporal dispersion (at dt = 0.005, the CN group velocity error
at k = p₀ = 15 is ~8%).
"""

import numpy as np
import time as timer


# ============================================================
#  Thomas algorithm (tridiagonal solver)
# ============================================================
def thomas_factor(lower, diag, upper):
    """LU-factor a tridiagonal matrix (Thomas algorithm)."""
    n = len(diag)
    d = diag.astype(complex).copy()
    l = lower.astype(complex).copy()
    for i in range(1, n):
        l[i-1] = l[i-1] / d[i-1]
        d[i]  -= l[i-1] * upper[i-1]
    return l, d, upper


def thomas_solve(l_fac, d_fac, u_fac, rhs):
    """Solve using pre-factored Thomas LU."""
    n = len(d_fac)
    b = rhs.copy()
    for i in range(1, n):
        b[i] -= l_fac[i-1] * b[i-1]
    b[-1] /= d_fac[-1]
    for i in range(n-2, -1, -1):
        b[i] = (b[i] - u_fac[i] * b[i+1]) / d_fac[i]
    return b


# ============================================================
#  Physical parameters (matching the paper exactly)
# ============================================================
L       = 10.0
rho     = 8.5
x0      = -5.0
p0      = 15.0
sigma0  = 1.0
T_final = 0.9

# Time steps
dt_fft  = 0.005      # for split-step FFT (exact kinetic propagator → no temporal dispersion)
dt_pml  = 0.0005     # for Crank-Nicolson PML (CN group velocity error < 0.1% at this dt)

PML_POW = 2

# σ_max values for the sweep
SIGMA_MAX_LIST = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


# ============================================================
#  Bergold & Lasser bump (Eq. 4.1) — matching paper exactly
# ============================================================
def bl_bump(x, rho_w, L_w):
    """Bergold & Lasser Eq. (4.1): w(x) = 1/(exp(1/(L-|x|) + 1/(ρ-|x|)) + 1)."""
    ax = np.abs(x)
    result = np.ones_like(ax, dtype=float)
    mask = (ax > rho_w) & (ax < L_w)
    a = ax[mask]
    exponent = 1.0 / (L_w - a) + 1.0 / (rho_w - a)
    with np.errstate(over='ignore'):
        result[mask] = 1.0 / (np.exp(exponent) + 1.0)
    result[ax >= L_w] = 0.0
    return result


# ============================================================
#  PML solver  (Crank-Nicolson + 2nd-order FD)
# ============================================================
def pml_solve(N, sigma_max):
    """
    Solve i∂ₜψ = -(1/2)∂²ₓψ on [−L, L] with PML.
    Returns (spur_neg, norm_total, norm_plateau).
    """
    dt  = dt_pml
    M   = int(round(T_final / dt))
    dx  = 2.0 * L / N
    N_int = N - 1

    x_int = -L + np.arange(1, N) * dx
    x_all = -L + np.arange(N + 1) * dx

    # PML damping
    d_pml = L - rho
    sig = np.zeros(N + 1)
    mr = x_all > rho;  ml = x_all < -rho
    sig[mr] = sigma_max * ((x_all[mr] - rho) / d_pml) ** PML_POW
    sig[ml] = sigma_max * ((-x_all[ml] - rho) / d_pml) ** PML_POW

    s_all = 1.0 + 1j * sig
    s   = s_all[1:-1]
    shp = 0.5 * (s_all[1:-1] + s_all[2:])
    shm = 0.5 * (s_all[:-2]  + s_all[1:-1])

    # FD operator for (1/2)(1/s)d/dx[(1/s)du/dx]
    idx2    = 0.5 / dx**2
    D_diag  = -(1.0/s) * (1.0/shp + 1.0/shm) * idx2
    D_upper = (1.0 / (s[:-1] * shp[:-1])) * idx2
    D_lower = (1.0 / (s[1:]  * shm[1:]))  * idx2

    # CN: (I - r D)ψ^{n+1} = (I + r D)ψ^n,  r = i dt / 2
    r = 0.5j * dt

    L_diag  = 1.0 - r * D_diag
    L_upper = -r * D_upper
    L_lower = -r * D_lower
    R_diag  = 1.0 + r * D_diag
    R_upper = r * D_upper
    R_lower = r * D_lower

    l_fac, d_fac, u_fac = thomas_factor(L_lower, L_diag, L_upper)

    # Initial condition (matching paper exactly)
    psi = np.exp(-(x_int - x0)**2 / (2*sigma0**2)) * np.exp(1j * p0 * x_int)
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

    # Time loop
    rhs_vec = np.empty(N_int, dtype=complex)
    for _ in range(M):
        rhs_vec[:]   = R_diag * psi
        rhs_vec[:-1] += R_upper * psi[1:]
        rhs_vec[1:]  += R_lower * psi[:-1]
        psi = thomas_solve(l_fac, d_fac, u_fac, rhs_vec)

    # Observables
    dens = np.abs(psi)**2
    norm_total   = np.sum(dens) * dx
    spur_neg     = np.sum(dens[x_int < 0]) * dx
    norm_plateau = np.sum(dens[np.abs(x_int) <= rho]) * dx

    return spur_neg, norm_total, norm_plateau


# ============================================================
#  C∞ windowed split-step solver (cross-verification)
# ============================================================
def window_solve(N_fft):
    """Split-step FFT with BL bump. Propagator: exp(-ik²dt/2)."""
    dt = dt_fft
    M  = int(round(T_final / dt))
    dx = 2.0 * L / N_fft
    x  = np.linspace(-L, L, N_fft, endpoint=False)
    k  = 2.0 * np.pi * np.fft.fftfreq(N_fft, d=dx)

    W   = bl_bump(x, rho, L)
    kin = np.exp(-1j * k**2 * dt / 2.0)

    psi = np.exp(-(x - x0)**2 / (2*sigma0**2)) * np.exp(1j * p0 * x)
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

    for _ in range(M):
        psi = np.fft.ifft(kin * np.fft.fft(psi))
        psi *= W

    dens = np.abs(psi)**2
    return (np.sum(dens[x < 0]) * dx,
            np.sum(dens) * dx,
            np.sum(dens[np.abs(x) <= rho]) * dx)


# ============================================================
#  Main
# ============================================================
def main():
    print("=" * 78)
    print("  PML Benchmark — 1D Schrödinger: i∂ₜψ = -(1/2)∂²ₓψ")
    print(f"  Domain [−{L}, {L}],  ρ = {rho},  η = {(L-rho)/L:.2f}")
    print(f"  Gaussian: x₀ = {x0}, p₀ = {p0}, σ = {sigma0}")
    print(f"  Δt_FFT = {dt_fft},  Δt_PML = {dt_pml},  T = {T_final}")
    print(f"  PML: σ(x) = σ_max·((|x|−ρ)/(L−ρ))², Crank-Nicolson + 2nd-order FD")
    print("=" * 78)

    # --- C∞ window reference ------------------------------------
    print("\n  C∞ window reference (BL bump, split-step FFT, Δt = 0.005):")
    for N_w in [1024, 2048, 4096]:
        t0 = timer.time()
        cs, cn, cp = window_solve(N_w)
        el = timer.time() - t0
        print(f"    N={N_w:>5d}:  spur = {cs:.3e},  ‖ψ‖² = {cn:.6f},"
              f"  ‖ψ‖²_ρ = {cp:.6f}  ({el:.3f}s)")

    ref_spur = window_solve(2048)[0]
    ref_plat = window_solve(2048)[2]

    # --- PML sweep (N = 4096, dt = 0.0005) ----------------------
    N_PML = 4096
    print(f"\n  PML sweep (N = {N_PML}, Δt = {dt_pml}):")
    print(f"  {'σ_max':>8s} | {'Spur':>12s} | {'‖ψ‖²':>10s} |"
          f" {'‖ψ‖²_ρ':>10s} | {'Spur ratio':>12s} | {'Time':>6s}")
    print("  " + "-" * 70)

    results = []
    for smax in SIGMA_MAX_LIST:
        t0 = timer.time()
        spur, norm_t, norm_p = pml_solve(N_PML, smax)
        elapsed = timer.time() - t0
        ratio = spur / ref_spur if ref_spur > 0 else 0
        print(f"  {smax:8.2f} | {spur:12.3e} | {norm_t:10.6f} |"
              f" {norm_p:10.6f} | {ratio:12.2e} | {elapsed:5.1f}s")
        results.append((smax, spur, norm_t, norm_p, elapsed))

    best_idx = np.argmin([r[1] for r in results])
    best = results[best_idx]
    print(f"\n  ── Summary ──")
    print(f"  Optimal PML:  σ_max = {best[0]},  spur = {best[1]:.3e}")
    print(f"  C∞ window:    spur = {ref_spur:.3e}")
    print(f"  PML advantage (spur): {ref_spur/best[1]:.0f}×")
    print(f"  PML plateau norm: {best[3]:.6f}  (C∞: {ref_plat:.6f}, diff: {abs(best[3]-ref_plat)/ref_plat*100:.1f}%)")

    # --- Spatial convergence ------------------------------------
    print(f"\n  Spatial convergence (PML, σ_max = {best[0]}, Δt = {dt_pml}):")
    for N_test in [512, 1024, 2048, 4096, 8192]:
        t0 = timer.time()
        s_t, n_t, p_t = pml_solve(N_test, best[0])
        el = timer.time() - t0
        print(f"    N = {N_test:>5d}:  spur = {s_t:.3e},  ‖ψ‖²_ρ = {p_t:.6f}  ({el:.1f}s)")

    # --- Cost comparison ----------------------------------------
    M_pml = int(round(T_final / dt_pml))
    M_fft = int(round(T_final / dt_fft))
    print(f"\n  Computational cost:")
    print(f"    C∞ window: N=2048, M={M_fft}, cost ~ N·log₂(N)·M = {2048*11*M_fft:.2e}")
    print(f"    PML:       N={N_PML}, M={M_pml}, cost ~ N·M         = {N_PML*M_pml:.2e}")
    print(f"    Ratio (PML/FFT): {N_PML*M_pml/(2048*11*M_fft):.1f}×")

    # --- Save results -------------------------------------------
    fname = "pml_results.txt"
    with open(fname, "w") as f:
        f.write("PML Benchmark Results — 1D Schrodinger\n")
        f.write(f"Equation: i dt psi = -(1/2) dxx psi\n")
        f.write(f"Domain [-{L}, {L}], rho={rho}, T={T_final}\n")
        f.write(f"PML: N={N_PML}, dt={dt_pml}, profile=quadratic\n")
        f.write(f"FFT: N=2048, dt={dt_fft}, window=Bergold-Lasser bump\n\n")
        f.write(f"C_inf window (N=2048): spur = {ref_spur:.6e}, plat = {ref_plat:.6f}\n\n")
        f.write(f"{'sigma_max':>10s} {'spur':>15s} {'norm':>12s} {'norm_plat':>12s}\n")
        for smax, spur, nt, np_, _ in results:
            f.write(f"{smax:10.2f} {spur:15.6e} {nt:12.6f} {np_:12.6f}\n")
        f.write(f"\nOptimal: sigma_max = {best[0]}, spur = {best[1]:.6e}\n")
    print(f"\n  Results saved to {fname}")


if __name__ == "__main__":
    main()
