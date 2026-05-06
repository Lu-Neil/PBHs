"""Analytic template count over (f0, beta) as a function of the Mc lower cutoff.

For each (f0, beta) the beta spacing is set so that the tail-power model of
delta_beta_behaviour.py predicts a fixed retained power P_target at the
template edge.  Delta_f0 is the NUFFT tau-bin resolution (no extra mismatch).
Cheap: no NUFFTs run here.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib.pyplot as pl
from scipy import integrate

C_LIGHT = 3e8
G_NEWT = 6.67e-11
M_SUN = 2e30
CONST = 96.0 / 5.0 * np.pi ** (8.0 / 3.0) * (G_NEWT / C_LIGHT**3) ** (5.0 / 3.0)

F0_MIN, F0_MAX = 20.0, 200.0
T_OBS_CAP = 1000  # 2 * 86400  # hard cap on analysis duration [s]  (2 days)
MC_UPPER = 1e-1  # upper Mc bound in solar masses
P_TARGETS = (0.9, 0.5)  # retained power fractions to use for bank spacing

OUTPUT_DIR = Path(__file__).resolve().parent / "figs"
OUTPUT_DIR.mkdir(exist_ok=True)


def beta_of(f0, Mc_solar):
    return CONST * f0 ** (8.0 / 3.0) * (Mc_solar * M_SUN) ** (5.0 / 3.0)


def t_coalesce(f0, beta, f_max=F0_MAX):
    """Time at which f(t) reaches f_max (signal leaves band)."""
    return (3.0 / (8.0 * beta)) * (1.0 - (f0 / f_max) ** (8.0 / 3.0))


def T_obs(f0, beta):
    return min(T_OBS_CAP, max(0.0, t_coalesce(f0, beta)))


def delta_beta_cut(f0, beta):
    """Analytic crossover: tau-space frequency mismatch at t=T equals one NUFFT bin = 1/tau_duration.

    delta_f_tau(T) = f0 * delta_beta * T / (1 - 8*beta*T/3)  [linearised]
    Setting = 1/tau_duration gives:
        Delta_beta_cut = (1 - 8 beta T / 3) / (f0 * T * tau_duration)
    """
    T = T_obs(f0, beta)
    return (1.0 - (8.0 / 3.0) * beta * T) / (f0 * T * tau_duration(beta, T))


def tau_duration(beta, T):
    """Total tau span: tau(T) - tau(0)."""
    return (3.0 / (5.0 * beta)) * (1.0 - (1.0 - (8.0 / 3.0) * beta * T) ** (5.0 / 8.0))


def delta_f0(f0, beta):
    """NUFFT bin width in tau-space (Hz)."""
    T = T_obs(f0, beta)
    return 1.0 / tau_duration(beta, T)


def mu_for_retained_power(f0, beta, P_target):
    """Multiplier on Delta_beta_cut that yields retained power = P_target
    under the tail-power model of delta_beta_behaviour.py.

    Derivation: with x = 8 beta T / 3, a = sqrt(1 - x),
        retained = 1 - (sqrt(1 - x s_cross) - a) / (1 - a)
    set retained = P, solve:  sqrt(1 - x s_cross) = 1 - P (1 - a) = L
    then crossover condition mu s_cross (1-x)/(1 - x s_cross) = 1 gives
        mu = (x / (1 - x)) * L^2 / (1 - L^2).
    Limits: x -> 0   => mu -> 1 / P        (no-chirp case)
            x -> 1   => mu diverges, but mu * Delta_beta_cut stays finite.
    """
    T = T_obs(f0, beta)
    if T <= 0.0:
        return np.inf
    x = (8.0 / 3.0) * beta * T
    if x <= 0.0:
        return 1.0 / P_target
    x = min(x, 1.0 - 1e-12)  # avoid 1/(1-x) blow-up at strict coalescence
    a = np.sqrt(1.0 - x)
    L = 1.0 - P_target * (1.0 - a)
    L2 = L * L
    return (x / (1.0 - x)) * L2 / (1.0 - L2)


def density_f0_Mc(f0, Mc_solar, P_target):
    """dN / (df0 dMc) -- number of templates per (f0, Mc) area element,
    with beta spacing set to hit retained power = P_target at the edge,
    and f0 spacing at the NUFFT tau-bin resolution."""
    beta = beta_of(f0, Mc_solar)
    T = T_obs(f0, beta)
    if T <= 0.0:
        return 0.0
    mu_b = mu_for_retained_power(f0, beta, P_target)
    dbeta_dMc = CONST * f0 ** (8.0 / 3.0) * (5.0 / 3.0) * (Mc_solar * M_SUN) ** (2.0 / 3.0) * M_SUN
    d_beta = mu_b * delta_beta_cut(f0, beta)
    d_f0 = delta_f0(f0, beta)
    return dbeta_dMc / (d_beta * d_f0)


def N_total(Mc_lower, P_target):
    val, _ = integrate.dblquad(
        lambda Mc, f0: density_f0_Mc(f0, Mc, P_target),
        F0_MIN,
        F0_MAX,
        lambda f0: Mc_lower,
        lambda f0: MC_UPPER,
        epsabs=1.0,
        epsrel=1e-3,
    )
    return val


def _sanity_check():
    """Verify mu_for_retained_power inverts the forward tail-power formula."""

    def forward_retained(mu, x):
        # replicates delta_beta_behaviour._retained_power_from_tcross algebra
        # crossover condition: mu s (1-x)/(1 - x s) = 1
        # => s = 1 / (mu (1-x) + x)
        s = 1.0 / (mu * (1.0 - x) + x)
        if s >= 1.0:
            return 1.0
        edge_cross = np.sqrt(1.0 - x * s)
        edge_T = np.sqrt(1.0 - x)
        return 1.0 - (edge_cross - edge_T) / (1.0 - edge_T)

    print("mu_for_retained_power sanity check:")
    print(f"  {'f0':>6} {'Mc':>8} {'x':>8} {'P_target':>8} {'mu':>10} {'P_fwd':>8}")
    for f0 in (20.0, 50.0, 150.0):
        for Mc in (1e-1, 1e-3, 1e-5):
            beta = beta_of(f0, Mc)
            T = T_obs(f0, beta)
            if T <= 0:
                continue
            x = (8.0 / 3.0) * beta * T
            for P in P_TARGETS:
                mu = mu_for_retained_power(f0, beta, P)
                P_fwd = forward_retained(mu, min(x, 1 - 1e-12))
                print(f"  {f0:6.1f} {Mc:8.0e} {x:8.3e} {P:8.2f} {mu:10.4f} {P_fwd:8.4f}")


def main(n_points=40, mc_lower_min=1e-5):
    import time

    _sanity_check()

    # timing probe
    t0 = time.time()
    n_ref = N_total(1e-2, P_target=0.9)
    print(f"\ndummy N_total(Mc_low=1e-2, P=0.9) = {n_ref:.3e}   [{time.time() - t0:.2f} s]")

    mc_lowers = np.logspace(np.log10(mc_lower_min), np.log10(MC_UPPER) - 0.05, n_points)

    results = {}
    for P in P_TARGETS:
        t0 = time.time()
        Ns = np.array([N_total(mc, P_target=P) for mc in mc_lowers])
        print(
            f"P_target={P:>4}: {n_points} pts in {time.time() - t0:.2f} s,  "
            f"N(Mc_low={mc_lowers[0]:.1e}) = {Ns[0]:.3e},  "
            f"N(Mc_low={mc_lowers[-1]:.1e}) = {Ns[-1]:.3e}"
        )
        results[P] = Ns

    fig, ax = pl.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    for P, Ns in results.items():
        ax.loglog(mc_lowers, Ns, "o-", ms=3, lw=1.4, label=rf"retained $= {P}$")
    ax.set_xlabel(r"$M_c^{\rm lower}\ /\ M_\odot$")
    ax.set_ylabel("Analytic template count")
    ax.set_title(
        rf"Templates over $f_0 \in [{F0_MIN:.0f}, {F0_MAX:.0f}]$ Hz, "
        rf"$M_c \in [M_c^{{\rm lower}}, {MC_UPPER}]\,M_\odot$"
        f"\n$T_{{\\rm obs}}$ cap = {T_OBS_CAP:.1e} s,  "
        r"$\Delta\beta$ set to fixed edge retained power,  "
        r"$\Delta f_0 = 1/T_\tau$"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(OUTPUT_DIR / "template_count_vs_mc_cutoff.png", dpi=180)
    print(f"Saved {OUTPUT_DIR / 'template_count_vs_mc_cutoff.png'}")


if __name__ == "__main__":
    main()
