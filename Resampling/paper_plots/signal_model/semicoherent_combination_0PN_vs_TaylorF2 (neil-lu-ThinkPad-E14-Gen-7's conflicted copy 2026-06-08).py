import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq

import lal
import lalsimulation as lalsim


# --- Physical Constants (SI Units) ---
G = lal.G_SI
c = lal.C_SI

# --- Binary System Parameters ---
Mc_values_msun = [1e-2, 1e-1]
q = 1
eta = q / (1 + q) ** 2

# --- Band Limits ---
f_start_global = 20.0
f_stop = 200.0


def component_masses_from_mchirp_eta(mchirp_msun, eta):
    Mc = mchirp_msun * lal.MSUN_SI
    M = Mc / (eta ** (3 / 5))
    sqrt_term = np.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * M * (1.0 + sqrt_term)
    m2 = 0.5 * M * (1.0 - sqrt_term)
    return Mc, M, m1, m2


def time_to_frequency_0pn(f_start, f_end, M_sec):
    omega_start = np.pi * f_start
    omega_end = np.pi * f_end
    chirp_coeff = (24.0 / 5.0) * M_sec ** (5 / 3)
    return (omega_start ** (-8 / 3) - omega_end ** (-8 / 3)) / (
        (8.0 / 3.0) * chirp_coeff
    )


def frequency_after_0pn(f_start, t, M_sec):
    omega_start = np.pi * f_start
    chirp_coeff = (24.0 / 5.0) * M_sec ** (5 / 3)
    omega = (omega_start ** (-8 / 3) - (8.0 / 3.0) * chirp_coeff * t) ** (
        -3 / 8
    )
    return omega / np.pi


class TaylorF2TimeMap:
    def __init__(self, m1, m2, f_min, f_max, n_grid=50000):
        self.f_min = f_min
        self.f_max = f_max
        self.f_grid = np.geomspace(f_min, f_max, n_grid)
        mtot_sec = G * (m1 + m2) / c**3
        phasing = lalsim.SimInspiralTaylorF2AlignedPhasing(
            m1, m2, 0.0, 0.0, lal.CreateDict()
        )
        t_of_f = np.array(
            [
                lalsim.PNPhaseDerivative(f, 2, phasing, mtot_sec) / (2.0 * np.pi)
                for f in self.f_grid
            ]
        )

        valid = np.isfinite(t_of_f) & np.isfinite(self.f_grid)
        self.f_grid = self.f_grid[valid]
        t_of_f = t_of_f[valid]
        if np.any(np.diff(t_of_f) >= 0.0):
            raise ValueError("TaylorF2 chirp time must decrease monotonically with frequency.")

        self.t_of_f = interp1d(
            self.f_grid, t_of_f, kind="cubic", bounds_error=True
        )
        self.f_of_t = interp1d(
            t_of_f[::-1], self.f_grid[::-1], kind="cubic", bounds_error=True
        )

    def elapsed_time_to(self, f_start, f_end):
        return float(self.t_of_f(f_start) - self.t_of_f(f_end))

    def frequency_after(self, f_start, t_elapsed):
        target_t = float(self.t_of_f(f_start)) - t_elapsed
        return float(self.f_of_t(target_t))


def first_bin_crossover(f_start, M_sec, taylorf2_map):
    t_0pn_stop = time_to_frequency_0pn(f_start, f_stop, M_sec)
    t_taylor_stop = taylorf2_map.elapsed_time_to(f_start, f_stop)
    t_max = min(t_0pn_stop, t_taylor_stop)

    def mismatch_minus_bin_width(t):
        f_0pn = frequency_after_0pn(f_start, t, M_sec)
        f_taylorf2 = taylorf2_map.frequency_after(f_start, t)
        return abs(f_taylorf2 - f_0pn) - 1.0 / t

    t_min = max(1e-6, t_max * 1e-12)
    sample_times = np.geomspace(t_min, t_max, 1000)
    sample_values = np.array([mismatch_minus_bin_width(t) for t in sample_times])
    crossing_idx = np.flatnonzero(sample_values >= 0.0)

    if len(crossing_idx) == 0:
        return None

    idx = crossing_idx[0]
    t_left = sample_times[max(idx - 1, 0)]
    t_right = sample_times[idx]
    t_cross = brentq(mismatch_minus_bin_width, t_left, t_right, rtol=1e-12)

    f_0pn = frequency_after_0pn(f_start, t_cross, M_sec)
    f_taylorf2 = taylorf2_map.frequency_after(f_start, t_cross)
    return {
        "t_cross": t_cross,
        "f_end_0pn": f_0pn,
        "f_end_taylorf2": f_taylorf2,
        "bin_width": 1.0 / t_cross,
        "mismatch": abs(f_taylorf2 - f_0pn),
    }


def compute_chunks(mchirp_msun):
    _, M, m1, m2 = component_masses_from_mchirp_eta(mchirp_msun, eta)
    M_sec = G * M / c**3
    taylorf2_map = TaylorF2TimeMap(m1, m2, f_start_global, f_stop)
    rows = []

    print(f"Semicoherent 0PN vs TaylorF2 chunks for Mc = {mchirp_msun:.0e} Msun")
    print(
        "chunk  t_cross_s    f_start_Hz  f_end_TaylorF2_Hz  "
        "delta_f_Hz  mismatch_Hz  bin_width_Hz"
    )

    f_start = f_start_global
    chunk = 1
    while f_start < f_stop:
        crossover = first_bin_crossover(f_start, M_sec, taylorf2_map)

        if crossover is None:
            t_to_stop = taylorf2_map.elapsed_time_to(f_start, f_stop)
            rows.append((chunk, f_start, f_stop, t_to_stop, False))
            print(
                f"{chunk:5d}  {t_to_stop:9.3f}  {f_start:10.3f}  "
                f"{f_stop:17.3f}  {f_stop - f_start:10.3f}  "
                "no crossover before f_stop"
            )
            break

        f_end = crossover["f_end_taylorf2"]
        rows.append((chunk, f_start, f_end, crossover["t_cross"], True))
        # print(
        #     f"{chunk:5d}  {crossover['t_cross']:9.3f}  {f_start:10.3f}  "
        #     f"{f_end:17.3f}  {f_end - f_start:10.3f}  "
        #     f"{crossover['mismatch']:11.3e}  {crossover['bin_width']:12.3e}"
        # )

        if f_end >= f_stop:
            break
        if f_end <= f_start:
            raise RuntimeError("Chunk endpoint did not advance in frequency.")

        f_start = f_end
        chunk += 1

    print()
    return np.array(rows, dtype=object)


def main():
    chunk_rows_by_mass = {
        mchirp_msun: compute_chunks(mchirp_msun) for mchirp_msun in Mc_values_msun
    }

    colors = ["dodgerblue", "darkorange", "crimson"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for color, mchirp_msun in zip(colors, Mc_values_msun):
        rows = chunk_rows_by_mass[mchirp_msun]
        f_starts = rows[:, 1].astype(float)
        t_crosses = rows[:, 3].astype(float)
        has_crossing = rows[:, 4].astype(bool)

        label = rf"$M_c={mchirp_msun:.0e}M_\odot$, chunks={len(rows)}"
        ax.semilogy(
            f_starts[has_crossing],
            t_crosses[has_crossing],
            color=color,
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=label,
        )
        if np.any(~has_crossing):
            ax.scatter(
                f_starts[~has_crossing],
                t_crosses[~has_crossing],
                marker="x",
                color=color,
                zorder=3,
            )

    ax.set_xlabel("Starting GW Frequency [Hz]", fontsize=11)
    ax.set_ylabel("Chunk Duration [s]", fontsize=11)
    ax.set_title("Semicoherent chunk duration", fontsize=13, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

if __name__ == "__main__":
    main()
