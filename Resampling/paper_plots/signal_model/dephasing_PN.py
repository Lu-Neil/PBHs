import argparse

import numpy as np
from scipy.integrate import solve_ivp

import lal
import lalsimulation as lalsim


PN_ORDERS = [
    ("1PN", lalsim.PNORDER_ONE),
    ("1.5PN", lalsim.PNORDER_ONE_POINT_FIVE),
    ("2PN", lalsim.PNORDER_TWO),
    ("2.5PN", lalsim.PNORDER_TWO_POINT_FIVE),
    ("3PN", lalsim.PNORDER_THREE),
    ("3.5PN", lalsim.PNORDER_THREE_POINT_FIVE),
]


def component_mass_seconds_from_mchirp(Mc_msun, eta):
    Mc = Mc_msun * lal.MSUN_SI
    M = Mc / eta ** (3.0 / 5.0)
    return lal.G_SI * M / lal.C_SI**3


def taylor_t4_coefficients(eta):
    """Nonspinning point-particle TaylorT4 coefficients used by LAL."""
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


def taylor_t4_factor(v, eta, pn_order):
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


def v_0pn_at_time(t, v_start, M_sec, eta):
    denominator = v_start ** -8.0 - (256.0 * eta / (5.0 * M_sec)) * t
    if denominator <= 0.0:
        return np.inf
    return denominator ** (-1.0 / 8.0)


def t_coh_for_order(f0, Mc_msun, pn_order, eta=0.25, threshold=np.pi):
    M_sec = component_mass_seconds_from_mchirp(Mc_msun, eta)
    v_start = (np.pi * M_sec * f0) ** (1.0 / 3.0)
    v_stop = 1.0 / np.sqrt(6.0)

    if not 0.0 < v_start < v_stop:
        raise ValueError(
            f"Initial frequency gives v0 = {v_start:.6g}; expected 0 < v0 < 1/sqrt(6)."
        )

    def rhs(v, y):
        elapsed_time, delta_phi = y
        del delta_phi

        factor = taylor_t4_factor(v, eta, pn_order)
        if factor <= 0.0:
            raise ValueError(f"TaylorT4 dv/dt factor became non-positive at v = {v:.6g}.")

        dT_dv = (5.0 * M_sec / (32.0 * eta)) * v ** -9.0 / factor
        v_0pn = v_0pn_at_time(elapsed_time, v_start, M_sec, eta)
        ddelta_dv = (2.0 / M_sec) * (v**3 - v_0pn**3) * dT_dv
        return [dT_dv, ddelta_dv]

    def dephasing_reaches_threshold(v, y):
        return abs(y[1]) - threshold

    dephasing_reaches_threshold.terminal = True
    dephasing_reaches_threshold.direction = 1

    sol = solve_ivp(
        rhs,
        (v_start, v_stop),
        [0.0, 0.0],
        events=dephasing_reaches_threshold,
        rtol=1e-10,
        atol=[1e-6, 1e-10],
        max_step=(v_stop - v_start) / 2000.0,
    )

    if sol.status == 1:
        elapsed_time, delta_phi = sol.y_events[0][0]
        return {
            "T_coh": elapsed_time,
            "delta_phi": delta_phi,
            "f_end": sol.t_events[0][0] ** 3 / (np.pi * M_sec),
        }

    return {
        "T_coh": None,
        "delta_phi": sol.y[1, -1],
        "f_end": sol.t[-1] ** 3 / (np.pi * M_sec),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the coherent time where same-time TaylorT4 PN-minus-0PN "
            "dephasing first exceeds pi."
        )
    )
    parser.add_argument("f0", type=float, help="starting GW frequency in Hz")
    parser.add_argument("Mc", type=float, help="chirp mass in solar masses")
    parser.add_argument(
        "--eta",
        type=float,
        default=0.25,
        help="symmetric mass ratio; default is 0.25 for an equal-mass binary",
    )
    parser.add_argument(
        "--order",
        choices=[label for label, _ in PN_ORDERS],
        help="only compute one PN truncation; default computes 1PN through 3.5PN",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    orders = PN_ORDERS
    if args.order is not None:
        orders = [entry for entry in PN_ORDERS if entry[0] == args.order]

    print(f"f0 = {args.f0:g} Hz, Mc = {args.Mc:g} Msun, eta = {args.eta:g}")
    for label, pn_order in orders:
        result = t_coh_for_order(args.f0, args.Mc, pn_order, eta=args.eta)
        if result["T_coh"] is None:
            print(
                f"{label}: |Delta phi| did not reach pi before f = {result['f_end']:.6g} Hz "
                f"(final Delta phi = {result['delta_phi']:.6g} rad)"
            )
        else:
            print(
                f"{label}: T_coh = {result['T_coh']:.9g} s "
                f"(Delta phi = {result['delta_phi']:.6g} rad, f_end = {result['f_end']:.6g} Hz)"
            )


if __name__ == "__main__":
    main()
