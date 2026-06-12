"""Compare coherent phase models against a 3.5PN TaylorT4 chirp.

The default case is the edge point used in the manuscript template-bank
discussion: f0 = 64 Hz, Mc = 0.1 Msun, eta = 1/4.

The script reports the first time at which each approximate coherent model
accumulates a phase residual of pi/2 and pi relative to the 3.5PN track.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import lal
import lalsimulation as lalsim


@dataclass(frozen=True)
class ModelResult:
    name: str
    t_pi_over_2: float | None
    t_pi: float | None
    residual_at_stop: float


def component_mass_seconds_from_mchirp(mchirp_msun: float, eta: float) -> float:
    mchirp = mchirp_msun * lal.MSUN_SI
    total_mass = mchirp / eta ** (3.0 / 5.0)
    return lal.G_SI * total_mass / lal.C_SI**3


def taylor_t4_coefficients(eta: float) -> dict[str, float]:
    """Nonspinning point-particle TaylorT4 coefficients."""
    return {
        "a2": -(743.0 + 924.0 * eta) / 336.0,
        "a3": 4.0 * np.pi,
        "a4": (34103.0 + 122949.0 * eta + 59472.0 * eta**2) / 18144.0,
        "a5": -np.pi * (4159.0 + 15876.0 * eta) / 672.0,
        "a6": (
            16447322263.0 / 139708800.0
            - 1712.0 * np.euler_gamma / 105.0
            - 856.0 * np.log(16.0) / 105.0
            - 56198689.0 * eta / 217728.0
            + np.pi**2 * (16.0 / 3.0 + 451.0 * eta / 48.0)
            + 541.0 * eta**2 / 896.0
            - 5605.0 * eta**3 / 2592.0
        ),
        "a6_log": -1712.0 / 105.0,
        "a7": np.pi * (-13245.0 + 717350.0 * eta + 731960.0 * eta**2) / 12096.0,
    }


def taylor_t4_factor(v: float | np.ndarray, eta: float, pn_order: int) -> float | np.ndarray:
    coeff = taylor_t4_coefficients(eta)
    factor = np.ones_like(v, dtype=float)

    if pn_order >= lalsim.PNORDER_ONE:
        factor += coeff["a2"] * v**2
    if pn_order >= lalsim.PNORDER_ONE_POINT_FIVE:
        factor += coeff["a3"] * v**3
    if pn_order >= lalsim.PNORDER_TWO:
        factor += coeff["a4"] * v**4
    if pn_order >= lalsim.PNORDER_TWO_POINT_FIVE:
        factor += coeff["a5"] * v**5
    if pn_order >= lalsim.PNORDER_THREE:
        factor += (coeff["a6"] + coeff["a6_log"] * np.log(v)) * v**6
    if pn_order >= lalsim.PNORDER_THREE_POINT_FIVE:
        factor += coeff["a7"] * v**7

    return factor


def beta_0pn(f0_hz: float, mchirp_msun: float) -> float:
    mchirp_sec = lal.G_SI * (mchirp_msun * lal.MSUN_SI) / lal.C_SI**3
    return (
        (96.0 / 5.0)
        * np.pi ** (8.0 / 3.0)
        * mchirp_sec ** (5.0 / 3.0)
        * f0_hz ** (8.0 / 3.0)
    )


def phase_0pn(t: np.ndarray, f0_hz: float, beta: float) -> np.ndarray:
    factor = 1.0 - (8.0 / 3.0) * beta * t
    if np.any(factor <= 0.0):
        raise ValueError("0PN phase requested after formal coalescence.")
    tau = (3.0 / (5.0 * beta)) * (1.0 - factor ** (5.0 / 8.0))
    return 2.0 * np.pi * f0_hz * tau


def make_35pn_track(
    f0_hz: float,
    mchirp_msun: float,
    eta: float,
    t_stop: float,
    n_eval: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m_sec = component_mass_seconds_from_mchirp(mchirp_msun, eta)
    v0 = (np.pi * m_sec * f0_hz) ** (1.0 / 3.0)
    v_isco = 1.0 / np.sqrt(6.0)

    def rhs(t: float, y: np.ndarray) -> list[float]:
        v, phi = y
        factor = taylor_t4_factor(v, eta, lalsim.PNORDER_THREE_POINT_FIVE)
        dvdt = (32.0 * eta / (5.0 * m_sec)) * v**9 * factor
        dphidt = 2.0 * v**3 / m_sec
        return [dvdt, dphidt]

    def reaches_isco(_t: float, y: np.ndarray) -> float:
        return y[0] - v_isco

    reaches_isco.terminal = True
    reaches_isco.direction = 1

    sol = solve_ivp(
        rhs,
        (0.0, t_stop),
        [v0, 0.0],
        events=reaches_isco,
        dense_output=True,
        rtol=1e-11,
        atol=[1e-13, 1e-10],
        max_step=max(t_stop / 5000.0, 1e-3),
    )

    t_end = sol.t_events[0][0] if sol.status == 1 else t_stop
    t = np.linspace(0.0, t_end, n_eval)
    v, phi = sol.sol(t)
    f = v**3 / (np.pi * m_sec)
    return t, f, phi


def first_crossing(t: np.ndarray, residual: np.ndarray, threshold: float) -> float | None:
    y = np.abs(residual) - threshold
    crossed = np.flatnonzero(y >= 0.0)
    if len(crossed) == 0:
        return None
    idx = int(crossed[0])
    if idx == 0:
        return float(t[0])
    return float(brentq(lambda x: np.interp(x, t, y), t[idx - 1], t[idx]))


def compare_models(
    f0_hz: float,
    mchirp_msun: float,
    eta: float,
    t_stop: float,
    n_eval: int,
) -> list[ModelResult]:
    t, f_35pn, phi_35pn = make_35pn_track(f0_hz, mchirp_msun, eta, t_stop, n_eval)

    beta = beta_0pn(f0_hz, mchirp_msun)
    fdot0_35pn = np.gradient(f_35pn, t, edge_order=2)[0]

    phases = {
        "monochromatic": 2.0 * np.pi * f0_hz * t,
        "linear-frequency Taylor": 2.0 * np.pi * (f0_hz * t + 0.5 * fdot0_35pn * t**2),
        "0PN": phase_0pn(t, f0_hz, beta),
    }

    results = []
    for name, phi_model in phases.items():
        residual = phi_35pn - phi_model
        results.append(
            ModelResult(
                name=name,
                t_pi_over_2=first_crossing(t, residual, np.pi / 2.0),
                t_pi=first_crossing(t, residual, np.pi),
                residual_at_stop=float(residual[-1]),
            )
        )
    return results


def format_time(value: float | None) -> str:
    return "not reached" if value is None else f"{value:.6g} s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare monochromatic, linear-frequency, and 0PN coherence times."
    )
    parser.add_argument("--f0", type=float, default=64.0, help="initial GW frequency in Hz")
    parser.add_argument("--mchirp", type=float, default=1e-1, help="chirp mass in solar masses")
    parser.add_argument("--eta", type=float, default=0.25, help="symmetric mass ratio")
    parser.add_argument("--t-stop", type=float, default=120.0, help="maximum integration time in seconds")
    parser.add_argument("--n-eval", type=int, default=20000, help="number of time samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = compare_models(args.f0, args.mchirp, args.eta, args.t_stop, args.n_eval)

    print(f"Truth model: 3.5PN TaylorT4")
    print(f"f0 = {args.f0:g} Hz, Mc = {args.mchirp:g} Msun, eta = {args.eta:g}")
    print("Coherence time is first |Delta phi| crossing.")
    print(f"{'model':<24} {'pi/2':>16} {'pi':>16} {'Delta phi at stop [rad]':>26}")
    for row in results:
        print(
            f"{row.name:<24} "
            f"{format_time(row.t_pi_over_2):>16} "
            f"{format_time(row.t_pi):>16} "
            f"{row.residual_at_stop:>26.6g}"
        )


if __name__ == "__main__":
    main()
