"""
Test whether the coherence loss from a β-mismatch follows sinc² or a Fresnel factor.

For a uniform-amplitude signal with a CONSTANT frequency offset Δf, the NUFFT
integral gives sinc(Δf·T), so power ~ sinc².  But in the PBH case Δf(t) grows
from 0 to 1/T_obs (linearly, to first order in β), so δφ(t) is quadratic — a
Fresnel integral, not sinc.  This script checks which hypothesis fits the data.

The key quantity is N_cycles = ∫₀^{t_cross} Δf dt (accumulated phase cycles in
the coherent window [0, t_cross]).  The correction factor is plotted against N
for three models:
  • sinc²(N)               — standard bin-dephasing formula (constant Δf)
  • Fresnel, uniform amp   — numerically integrated e^{iδφ} with w=1
  • Fresnel, chirp amp     — numerically integrated e^{iδφ} weighted by h₀(t)
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as pl
import numpy as np

NOV2025_DIR = Path(__file__).resolve().parents[1]
if str(NOV2025_DIR) not in sys.path:
    sys.path.insert(0, str(NOV2025_DIR))

from fiveVec_resampler_utils import (
    _create_PBH_signal,
    _estimator,
    _resample_and_extract_5vec,
    _time_domain_5vec,
)

SEED = 1
N_DELTA_BETA = 200
DELTA_BETA_MIN = 1e-12
DELTA_BETA_MAX = 1e-8
OUTPUT_PATH = Path(__file__).resolve().parent / "sinc_correction_test.png"


# ---------- physics helpers ----------

def _infer_beta(t, h0):
    return (3.0 / (8.0 * t[-1])) * (1.0 - (h0[0] / h0[-1]) ** 4)


def _analysis_tau(t, beta):
    tau = -(3.0 / (5.0 * beta)) * (1.0 - (8.0 / 3.0) * beta * t) ** (5.0 / 8.0)
    return tau - tau[0]


def _delta_f(t, f0, beta, db):
    p = 1.0 - (8.0 / 3.0) * beta * t
    temp0 = (3.0 * f0 * db) / (5.0 * beta * p ** (3.0 / 8.0))
    temp1 = 1.0 + (-1.0 + beta * t) / p
    return np.abs(temp0 * temp1)


def _crossover_idx(t, f0, beta, db):
    hit = np.flatnonzero(_delta_f(t, f0, beta, db) >= 1.0 / t[-1])
    return int(hit[0]) if hit.size else len(t) - 1


def _tail_power(beta, tcross, T):
    if tcross >= T:
        return 1.0
    a = np.sqrt(1.0 - (8.0 / 3.0) * beta * tcross)
    b = np.sqrt(1.0 - (8.0 / 3.0) * beta * T)
    return 1.0 - (a - b) / (1.0 - b)


def _n_cycles(t, f0, beta, db, idx):
    """Total phase cycles accumulated in [0, t_cross]: N = ∫ Δf dt."""
    t_c = t[: idx + 1]
    return float(np.trapz(_delta_f(t_c, f0, beta, db), t_c)) if len(t_c) > 1 else 0.0


def _accumulated_phase(t_c, f0, beta, db):
    """δφ(t) = 2π ∫₀^t Δf dt' — cumulative phase at each sample."""
    df = _delta_f(t_c, f0, beta, db)
    # trapezoidal cumulative integral, length = len(t_c), starts at 0
    pieces = np.diff(t_c) * 0.5 * (df[:-1] + df[1:])
    return 2.0 * np.pi * np.concatenate(([0.0], np.cumsum(pieces)))


def _coherence_factor(t, f0, beta, db, idx, weights=None):
    """
    |∫ w(t) e^{iδφ(t)} dt|² / |∫ w(t) dt|²  over [0, t_cross].

    weights=None → uniform (w=1); otherwise pass an array of length len(t).
    """
    t_c = t[: idx + 1]
    if len(t_c) < 2:
        return 1.0
    dphi = _accumulated_phase(t_c, f0, beta, db)
    w = np.ones(len(t_c)) if weights is None else weights[: idx + 1]
    num = np.trapz(w * np.exp(1j * dphi), t_c)
    den = np.trapz(w, t_c)
    return float(abs(num) ** 2 / abs(den) ** 2) if abs(den) > 0 else 1.0


def _sinc2(N):
    N = np.asarray(N, float)
    out = np.ones_like(N)
    m = np.abs(N) > 1e-12
    x = np.pi * N[m]
    out[m] = (np.sin(x) / x) ** 2
    return out


def _recovered_power_norm(signal, sidereal, t_ast, tau, omega0):
    X, _ = _resample_and_extract_5vec(signal, tau, omega0)
    tmpl, _, _ = _time_domain_5vec(sidereal, t_ast, tau)
    return float(abs(_estimator(X, tmpl)) ** 2)


# ---------- scan ----------

def run_scan(dbs):
    np.random.seed(SEED)
    signal, _, omega0, sidereal, h0, _, t_ast = _create_PBH_signal(
        f0_setting="midpoint", delta_beta=0.0
    )
    t = t_ast.gps - t_ast.gps[0]
    beta = _infer_beta(t, h0)
    f0 = omega0 / (2.0 * np.pi)
    T = t[-1]

    tau0 = _analysis_tau(t, beta)
    base = _recovered_power_norm(signal, sidereal, t_ast, tau0, omega0)

    rows = []
    for db in dbs:
        tau = _analysis_tau(t, beta + db)
        rec = _recovered_power_norm(signal, sidereal, t_ast, tau, omega0) / base

        idx = _crossover_idx(t, f0, beta, db)
        tcross = t[idx]
        tp = _tail_power(beta, tcross, T)
        N = _n_cycles(t, f0, beta, db, idx)
        cf_unif = _coherence_factor(t, f0, beta, db, idx, weights=None)
        cf_amp = _coherence_factor(t, f0, beta, db, idx, weights=h0)
        rows.append((rec, tp, N, cf_unif, cf_amp))

    recovered, tail_power, n_cycles, cf_u, cf_a = map(np.array, zip(*rows))
    return dict(
        recovered=recovered,
        tail_power=tail_power,
        n_cycles=n_cycles,
        cf_uniform=cf_u,
        cf_amp=cf_a,
        T=T, f0=f0, beta=beta,
    )


# ---------- plot ----------

def make_plot(dbs, R):
    rec, tp, N = R["recovered"], R["tail_power"], R["n_cycles"]
    cfu, cfa = R["cf_uniform"], R["cf_amp"]
    T, f0 = R["T"], R["f0"]

    corr_sinc2 = tp * _sinc2(N)
    corr_fresnel_amp = tp * cfa

    # Ratio actual/tail-power (only where tail-power is well-defined)
    valid = tp > 0.02
    ratio = np.where(valid, rec / tp, np.nan)

    N_dense = np.linspace(0, max(N[valid]) * 1.1, 600)

    fig, axes = pl.subplots(2, 1, figsize=(9, 10), constrained_layout=True)

    # ---- top: power vs Δβ ----
    ax = axes[0]
    ax.plot(dbs, rec, "o-", ms=3, lw=1.2, zorder=4, label="Recovered (actual)")
    ax.plot(dbs, tp, lw=2, color="C1", label="Tail-power model")
    ax.plot(dbs, corr_sinc2, lw=1.5, ls="--", color="C2",
            label=r"Tail $\times$ sinc²($N$)")
    ax.plot(dbs, corr_fresnel_amp, lw=1.5, ls="-.", color="C3",
            label=r"Tail $\times$ Fresnel (chirp-amp weighted)")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel(r"$\Delta\beta$")
    ax.set_ylabel("Recovered power fraction")
    ax.set_title(
        r"PBH power recovery: sinc² vs Fresnel coherence correction"
        + f"\nf0={f0:.6f} Hz,  Tobs={T / 86164.090:.2f} d"
    )
    ax.legend()

    # ---- bottom: correction factor vs N_cycles ----
    ax2 = axes[1]
    ax2.plot(N[valid], ratio[valid], "o", ms=4, color="C0", zorder=5,
             label="Actual / tail-power (data)")
    ax2.plot(N_dense, _sinc2(N_dense), lw=2, color="C2", ls="--",
             label=r"sinc²($N$)  [constant $\Delta f$ limit]")
    ax2.plot(N[valid], cfu[valid], lw=1.5, color="C4",
             label=r"Fresnel, uniform amp  $\left|\int e^{i\delta\phi}\right|^2$")
    ax2.plot(N[valid], cfa[valid], lw=1.5, ls="-.", color="C3",
             label=r"Fresnel, chirp amp  $\left|\int h_0 e^{i\delta\phi}\right|^2$")
    ax2.axvline(0.5, color="0.5", lw=1.0, ls=":",
                label=r"$N=0.5$ (β-cutoff, $t_{\rm cross}=T_{\rm obs}$)")
    ax2.annotate(
        r"sinc²(0.5) $\approx$ 0.41" + "\nFresnel(0.5) $\approx$ ?",
        xy=(0.5, _sinc2(np.array([0.5]))[0]),
        xytext=(0.3, 0.6),
        arrowprops=dict(arrowstyle="->", lw=1.0),
        fontsize=9,
    )
    ax2.set_xlabel(
        r"$N_{\rm cycles} = \int_0^{t_{\rm cross}} \Delta f\, dt$  "
        r"(accumulated phase cycles in coherent window)"
    )
    ax2.set_ylabel("Correction factor")
    ax2.set_title(
        r"Is the coherence loss sinc² or Fresnel?"
        "\n"
        r"(sinc² assumes constant $\Delta f$; Fresnel uses actual $\delta\phi(t) \propto t^2$)"
    )
    ax2.set_xlim(left=-0.02)
    ax2.set_ylim(-0.05, 1.25)
    ax2.legend(fontsize=9)

    return fig


def main():
    dbs = np.logspace(np.log10(DELTA_BETA_MIN), np.log10(DELTA_BETA_MAX), N_DELTA_BETA)
    R = run_scan(dbs)

    # N_cycles at β_cutoff (where tcross ≈ T_obs, i.e. tail_power ≈ 1.0)
    cut_idx = np.argmin(np.abs(R["tail_power"] - 1.0) + np.abs(R["n_cycles"] - 0.5))
    print(f"At β-cutoff:")
    print(f"  N_cycles     = {R['n_cycles'][cut_idx]:.4f}  (theory: 0.5 for linear Δf ramp)")
    print(f"  sinc²(N)     = {_sinc2(np.array([R['n_cycles'][cut_idx]]))[0]:.4f}")
    print(f"  Fresnel(unif)= {R['cf_uniform'][cut_idx]:.4f}")
    print(f"  Fresnel(amp) = {R['cf_amp'][cut_idx]:.4f}")
    print(f"  actual/model = {R['recovered'][cut_idx] / R['tail_power'][cut_idx]:.4f}")

    fig = make_plot(dbs, R)
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
