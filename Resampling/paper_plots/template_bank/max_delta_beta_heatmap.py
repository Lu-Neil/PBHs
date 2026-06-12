from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import lal


G = lal.G_SI
C = lal.C_SI
MSUN = lal.MSUN_SI
PI = np.pi

ETA = 0.25
CHUNK_DURATION = 1.0
F_MIN = 20.0
F_MAX = 128.0
MCHIRP_MIN_MSUN = 1e-4
MCHIRP_MAX_MSUN = 1e-1
N_F0 = 70
N_MCHIRP = 80
OUTPUT_PATH = (
    Path(__file__).resolve().parent / "figs" / "max_delta_beta_heatmap.png"
)


def beta_0pn(f_start_hz, mchirp_msun):
    mchirp_sec = G * (mchirp_msun * MSUN) / C**3
    return (
        (96.0 / 5.0)
        * PI ** (8.0 / 3.0)
        * mchirp_sec ** (5.0 / 3.0)
        * f_start_hz ** (8.0 / 3.0)
    )


def total_mass_seconds_from_beta(f_start_hz, beta, eta):
    mchirp_sec = (
        beta / ((96.0 / 5.0) * PI ** (8.0 / 3.0) * f_start_hz ** (8.0 / 3.0))
    ) ** (3.0 / 5.0)
    return mchirp_sec / eta ** (3.0 / 5.0)


def taylor_t4_factor_35pn(v, eta):
    """Nonspinning point-particle TaylorT4 3.5PN velocity factor."""
    a2 = -(743.0 + 924.0 * eta) / 336.0
    a3 = 4.0 * PI
    a4 = (34103.0 + 122949.0 * eta + 59472.0 * eta**2) / 18144.0
    a5 = -PI * (4159.0 + 15876.0 * eta) / 672.0
    a6 = (
        16447322263.0 / 139708800.0
        - 1712.0 * np.euler_gamma / 105.0
        - 856.0 * np.log(16.0) / 105.0
        - 56198689.0 * eta / 217728.0
        + PI**2 * (16.0 / 3.0 + 451.0 * eta / 48.0)
        + 541.0 * eta**2 / 896.0
        - 5605.0 * eta**3 / 2592.0
    )
    a6_log = -1712.0 / 105.0
    a7 = PI * (-13245.0 + 717350.0 * eta + 731960.0 * eta**2) / 12096.0

    return (
        1.0
        + a2 * v**2
        + a3 * v**3
        + a4 * v**4
        + a5 * v**5
        + (a6 + a6_log * np.log(v)) * v**6
        + a7 * v**7
    )


def taylor_t4_phase_after(f_start_hz, beta, duration, eta):
    """Return 3.5PN TaylorT4 phase accumulated over duration seconds."""
    total_mass_sec = total_mass_seconds_from_beta(f_start_hz, beta, eta)
    v_start = (PI * total_mass_sec * f_start_hz) ** (1.0 / 3.0)
    v_isco = 1.0 / np.sqrt(6.0)

    if not 0.0 < v_start < v_isco:
        return np.nan

    def rhs(_, y):
        v, phase = y
        del phase
        factor = taylor_t4_factor_35pn(v, eta)
        dv_dt = (32.0 * eta / (5.0 * total_mass_sec)) * v**9 * factor
        dphase_dt = 2.0 * v**3 / total_mass_sec
        return [dv_dt, dphase_dt]

    def reaches_isco(_, y):
        return y[0] - v_isco

    reaches_isco.terminal = True
    reaches_isco.direction = 1

    sol = solve_ivp(
        rhs,
        (0.0, duration),
        [v_start, 0.0],
        events=reaches_isco,
        rtol=1e-10,
        atol=[1e-12, 1e-10],
        method="DOP853",
    )

    if sol.status == 1 or not sol.success:
        return np.nan

    return float(sol.y[1, -1])


def phase_0pn_after(f_start_hz, beta, duration):
    bracket = 1.0 - (8.0 / 3.0) * beta * duration
    if bracket <= 0.0:
        return np.nan

    return (
        2.0
        * PI
        * f_start_hz
        * (3.0 / (5.0 * beta))
        * (1.0 - bracket ** (5.0 / 8.0))
    )


def max_delta_beta_fraction(f_start_hz, beta, duration, eta):
    phi_35pn = taylor_t4_phase_after(f_start_hz, beta, duration, eta)
    if not np.isfinite(phi_35pn):
        return np.nan

    def residual(delta_beta_fraction):
        beta_trial = beta * (1.0 + delta_beta_fraction)
        return phi_35pn - phase_0pn_after(f_start_hz, beta_trial, duration)

    beta_limit = 3.0 / (8.0 * duration)
    upper_limit = beta_limit / beta - 1.0
    if upper_limit <= 0.0:
        return np.nan

    residual_at_zero = residual(0.0)
    if residual_at_zero <= -PI:
        return np.nan

    # The plus-beta residual decreases monotonically because the 0PN phase
    # accumulated in the chunk increases with beta.
    upper = min(1.0, upper_limit * (1.0 - 1e-12))
    while upper < upper_limit * (1.0 - 1e-12) and residual(upper) > -PI:
        upper = min(2.0 * upper, upper_limit * (1.0 - 1e-12))

    if residual(upper) > -PI:
        return upper

    return brentq(lambda fraction: residual(fraction) + PI, 0.0, upper, rtol=1e-11)


def log_edges(values):
    edges = np.empty(values.size + 1)
    edges[1:-1] = np.sqrt(values[:-1] * values[1:])
    edges[0] = values[0] ** 2 / edges[1]
    edges[-1] = values[-1] ** 2 / edges[-2]
    return edges


def linear_edges(values):
    edges = np.empty(values.size + 1)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def compute_grid(f0_values, mchirp_values, duration, eta):
    allowed = np.empty((mchirp_values.size, f0_values.size))
    for i, mchirp in enumerate(mchirp_values):
        for j, f0 in enumerate(f0_values):
            beta = beta_0pn(f0, mchirp)
            allowed[i, j] = max_delta_beta_fraction(f0, beta, duration, eta)
    return allowed


def make_plot(f0_values, mchirp_values, allowed, args):
    masked_allowed = np.ma.masked_invalid(allowed)
    positive_values = masked_allowed[masked_allowed > 0.0]
    if positive_values.size == 0:
        raise RuntimeError("No positive deltaBeta/beta values were found in the grid.")

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    mesh = ax.pcolormesh(
        linear_edges(f0_values),
        log_edges(mchirp_values),
        masked_allowed,
        norm=LogNorm(vmin=positive_values.min(), vmax=positive_values.max()),
        shading="auto",
        cmap="viridis",
    )

    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(r"max allowed $\Delta\beta/\beta$")

    ax.set_yscale("log")
    ax.set_xlabel(r"$f_0$ [Hz]")
    ax.set_ylabel(r"$M_c$ [$M_\odot$]")
    ax.set_title(
        rf"1 s 3.5PN/0PN dephasing bound, $\eta={args.eta:g}$",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the maximum positive deltaBeta/beta such that a 3.5PN "
            "TaylorT4 signal corrected with a 0PN beta+deltaBeta model has "
            "less than pi radians residual phase after one chunk."
        )
    )
    parser.add_argument("--f-min", type=float, default=F_MIN)
    parser.add_argument("--f-max", type=float, default=F_MAX)
    parser.add_argument("--n-f0", type=int, default=N_F0)
    parser.add_argument("--mchirp-min", type=float, default=MCHIRP_MIN_MSUN)
    parser.add_argument("--mchirp-max", type=float, default=MCHIRP_MAX_MSUN)
    parser.add_argument("--n-mchirp", type=int, default=N_MCHIRP)
    parser.add_argument("--duration", type=float, default=CHUNK_DURATION)
    parser.add_argument("--eta", type=float, default=ETA)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()

    f0_values = np.linspace(args.f_min, args.f_max, args.n_f0)
    mchirp_values = np.geomspace(args.mchirp_min, args.mchirp_max, args.n_mchirp)

    allowed = compute_grid(f0_values, mchirp_values, args.duration, args.eta)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = make_plot(f0_values, mchirp_values, allowed, args)
    fig.savefig(args.output, dpi=220)

    finite = allowed[np.isfinite(allowed)]
    print(
        f"Computed {allowed.shape[0]} x {allowed.shape[1]} grid for "
        f"duration = {args.duration:g} s"
    )
    print(f"f0 range: {args.f_min:g} Hz to {args.f_max:g} Hz")
    print(
        f"Mc range: {args.mchirp_min:.6e} Msun to "
        f"{args.mchirp_max:.6e} Msun"
    )
    print(
        "max deltaBeta/beta range: "
        f"{np.nanmin(finite):.6e} to {np.nanmax(finite):.6e}"
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
