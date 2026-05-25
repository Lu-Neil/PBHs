import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from pathlib import Path
import sys

import lal
import lalsimulation as lalsim

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Nov2025"))
from resampler import Resampler


# --- Physical constants and binary parameters ---
G = lal.G_SI
c = lal.C_SI

Mc_values_msun = [1e-1, 1e-2]
q = 1.0
eta = q / (1.0 + q) ** 2

f0 = 20.0
f_stop = 200.0


def component_masses_from_mchirp_eta(Mc_msun, eta):
    Mc = Mc_msun * lal.MSUN_SI
    M = Mc / eta ** (3.0 / 5.0)
    sqrt_term = np.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * M * (1.0 + sqrt_term)
    m2 = 0.5 * M * (1.0 - sqrt_term)
    return Mc, M, m1, m2


def taylorf2_track(Mc_msun, n_grid=80_000):
    """Return a 3.5PN TaylorF2 time-domain frequency and phase track."""
    _, M, m1, m2 = component_masses_from_mchirp_eta(Mc_msun, eta)
    mtot_sec = G * M / c**3

    params = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(
        params, lalsim.PNORDER_THREE_POINT_FIVE
    )
    phasing = lalsim.SimInspiralTaylorF2AlignedPhasing(m1, m2, 0.0, 0.0, params)

    f_grid = np.geomspace(f0, f_stop, n_grid)
    t_of_f = np.array(
        [
            lalsim.PNPhaseDerivative(f, 2, phasing, mtot_sec) / (2.0 * np.pi)
            for f in f_grid
        ]
    )
    t = t_of_f[0] - t_of_f

    valid = np.isfinite(t) & np.isfinite(f_grid)
    t = t[valid]
    f_grid = f_grid[valid]
    if np.any(np.diff(t) <= 0.0):
        raise ValueError(
            f"TaylorF2 chirp time is not monotonic for Mc = {Mc_msun:.0e} Msun."
        )

    phi = cumulative_trapezoid(2.0 * np.pi * f_grid, t, initial=0.0)
    return {
        "t": t,
        "f": f_grid,
        "phi": phi,
        "t_end": t[-1],
    }


def beta_0pn(f_start, Mc_msun):
    Mc_sec = G * (Mc_msun * lal.MSUN_SI) / c**3
    return (
        (96.0 / 5.0)
        * np.pi ** (8.0 / 3.0)
        * Mc_sec ** (5.0 / 3.0)
        * f_start ** (8.0 / 3.0)
    )


def tau_0pn_from_beta(t, beta):
    bracket = 1.0 - (8.0 / 3.0) * beta * t
    valid = bracket > 0.0
    tau = np.full_like(t, np.nan, dtype=float)
    tau[valid] = -(3.0 / (5.0 * beta)) * bracket[valid] ** (5.0 / 8.0)
    tau[valid] -= tau[valid][0]
    return tau, valid


def recovered_power_with_resampler(track, beta):
    """Resample the TaylorF2 signal using the 0PN tau and return the peak power."""
    tau, valid = tau_0pn_from_beta(track["t"], beta)
    if np.count_nonzero(valid) < 2:
        raise RuntimeError("The requested beta has fewer than two valid samples.")

    tau_valid = tau[valid]
    signal = np.exp(1j * (track["phi"][valid] - 2.0 * np.pi * f0 * tau_valid))

    resampler = Resampler(nthreads=4, eps=1e-9, upsampfac=2.0)
    resampler.timeseries = signal
    resampler.resampled_time = tau_valid
    resampler.nufft()

    delta_f_spectrum = resampler.freq_in_hz
    f0_spectrum = f0 + delta_f_spectrum
    power_spectrum = resampler.power_normalized
    max_index = int(np.argmax(power_spectrum))
    nominal_index = int(np.argmin(np.abs(delta_f_spectrum)))

    return {
        "tau_span": tau_valid[-1] - tau_valid[0],
        "t_valid_end": track["t"][valid][-1],
        "f0_spectrum": f0_spectrum,
        "power_spectrum": power_spectrum,
        "max_power": power_spectrum[max_index],
        "f0_at_max": f0_spectrum[max_index],
        "nominal_power": power_spectrum[nominal_index],
        "f0_nominal_bin": f0_spectrum[nominal_index],
        "delta_f_at_max": delta_f_spectrum[max_index],
    }


def main():
    results = []
    for Mc_msun in Mc_values_msun:
        track = taylorf2_track(Mc_msun)
        beta_true = beta_0pn(f0, Mc_msun)

        recovered = recovered_power_with_resampler(track, beta_true)

        results.append(
            {
                "Mc_msun": Mc_msun,
                "t_end": track["t_end"],
                "beta_true": beta_true,
                **recovered,
                "loss": 1.0 - recovered["max_power"],
            }
        )

    print("TaylorF2 3.5PN unit-amplitude signal resampled with the true 0PN beta")
    print(
        "Power is Resampler.power_normalized; the reported value is the maximum "
        "over the heterodyned recovered-f0 spectrum."
    )
    for result in results:
        print(
            f"Mc = {result['Mc_msun']:.0e} Msun, "
            f"TaylorF2 reaches {f_stop:.1f} Hz in {result['t_end']:.4f} s"
        )
        print(
            f"  beta = {result['beta_true']:.6e}; valid 0PN-beta track ends at "
            f"t = {result['t_valid_end']:.4f} s; tau span = {result['tau_span']:.4f} s"
        )
        print(
            f"  true recovered power = {result['nominal_power']:.6e} "
            f"at {result['f0_nominal_bin']:.9f} Hz"
        )
        print(
            f"  max recovered power = {result['max_power']:.6e} "
            f"at f0 = {result['f0_at_max']:.9f} Hz "
            f"(delta f = {result['delta_f_at_max']:.6e} Hz); "
            f"loss = {result['loss']:.6e}"
        )

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for result in results:
        peak = result["max_power"]
        window = result["power_spectrum"] > max(peak * 1e-4, np.finfo(float).tiny)
        ax.plot(
            result["f0_spectrum"][window],
            result["power_spectrum"][window],
            label=rf"$M_c={result['Mc_msun']:.0e}M_\odot$",
        )
        ax.plot(
            result["f0_at_max"],
            result["max_power"],
            "o",
            color=ax.lines[-1].get_color(),
        )
    ax.axvline(f0, color="black", linestyle=":", alpha=0.7, label=rf"input $f_0={f0:g}$ Hz")
    ax.set_yscale("log")
    ax.set_xlabel("Recovered $f_0$ [Hz]")
    ax.set_ylabel("Resampler normalized power")
    ax.set_title("TaylorF2 3.5PN signal resampled with the true 0PN beta")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
