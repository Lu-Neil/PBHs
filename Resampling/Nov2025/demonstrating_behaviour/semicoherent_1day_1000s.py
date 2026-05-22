"""
Semicoherent recovery of a one-day sidereally modulated PBH chirp.

The injected signal includes the detector sidereal amplitude modulation, but
the recovery intentionally ignores that model: each 1000 s chunk contributes
only the demodulated carrier-bin power. This is the short-coherence limit where
the sidereal sidebands are unresolved and a power sum over chunks is enough to
recover the chirp track.

Defaults target a 20 -> 200 Hz chirp over 86 x 1000 s chunks, using F_S=512 Hz.
That covers the first 86000 s of one sidereal day and leaves the final 164.09 s
unused so every chunk has identical duration.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

NOV2025_DIR = Path(__file__).resolve().parent.parent
if str(NOV2025_DIR) not in sys.path:
    sys.path.insert(0, str(NOV2025_DIR))

from five_vec import five_vec
from resampler import Resampler


C_LIGHT = 3e8
G_NEWT = 6.67e-11
M_SUN = 2e30
KPC = 3.086e19
DISTANCE = 8.0 * KPC
SIDE_DAY = 86164.09053083288
CHIRP_CONST = 96.0 / 5.0 * np.pi ** (8.0 / 3.0) * (G_NEWT / C_LIGHT**3) ** (5.0 / 3.0)
REF_MJD = 58583.52425925926  # Time("2019-04-10T12:34:56.000").mjd
WHITE_PSD_TWO_SIDED = 6.0e-46
DEFAULT_SIGNAL_SCALE = float(os.getenv("PBH_SEMICOHERENT_SIGNAL_SCALE", "0.03"))
OUTPUT_DIR = Path(__file__).resolve().parent / "figs"


@dataclass(frozen=True)
class DemoConfig:
    noise: str
    mode: str
    fs: float
    f0: float
    f_end: float
    chunk_duration: float
    n_chunks: int
    signal_scale: float
    seed: int
    scan_fracs: np.ndarray
    precision: str
    nthreads: int
    no_plot: bool

    @property
    def t_analyzed(self) -> float:
        return self.chunk_duration * self.n_chunks

    @property
    def n_samples_per_chunk(self) -> int:
        n_samples = self.fs * self.chunk_duration
        if not np.isclose(n_samples, round(n_samples), rtol=0.0, atol=1e-9):
            raise ValueError("fs * chunk_duration must be an integer")
        return int(round(n_samples))


def beta_for_band(f0: float, f_end: float, duration: float) -> float:
    return (3.0 / (8.0 * duration)) * (1.0 - (f0 / f_end) ** (8.0 / 3.0))


def chirp_mass_from_beta(f0: float, beta: float) -> float:
    return (beta / (CHIRP_CONST * f0 ** (8.0 / 3.0))) ** (3.0 / 5.0)


def beta_from_chirp_mass(f0: float, chirp_mass_solar: float) -> float:
    return CHIRP_CONST * f0 ** (8.0 / 3.0) * (chirp_mass_solar * M_SUN) ** (5.0 / 3.0)


def instantaneous_frequency(f0: float, beta: float, t_offset: np.ndarray) -> np.ndarray:
    return f0 * (1.0 - (8.0 / 3.0) * beta * t_offset) ** (-3.0 / 8.0)


def tau_absolute(beta: float, t_offset: np.ndarray) -> np.ndarray:
    return -(3.0 / (5.0 * beta)) * (1.0 - (8.0 / 3.0) * beta * t_offset) ** (5.0 / 8.0)


def tau_from_beta(beta: float, t_offset: np.ndarray) -> np.ndarray:
    tau = tau_absolute(beta, t_offset)
    return tau - tau[0]


def physical_amplitude(chirp_mass: float, freq_hz: np.ndarray) -> np.ndarray:
    return 4.0 / DISTANCE * (G_NEWT * chirp_mass / C_LIGHT**2) ** (5.0 / 3.0) * (
        np.pi * freq_hz / C_LIGHT
    ) ** (2.0 / 3.0)


def build_sidereal(seed: int) -> five_vec:
    rng = np.random.default_rng(seed)
    sidereal = five_vec(
        ra=rng.uniform(0.0, 2.0 * np.pi),
        dec=rng.uniform(-np.pi / 2.0, np.pi / 2.0),
        eta=rng.uniform(-1.0, 1.0),
        psi=rng.uniform(0.0, 2.0 * np.pi),
        lat=rng.uniform(-np.pi / 2.0, np.pi / 2.0),
        lng=rng.uniform(-np.pi, np.pi),
        az=rng.uniform(0.0, 2.0 * np.pi),
    )
    sidereal.compute_H()
    return sidereal


def parse_scan_fracs(raw: str) -> np.ndarray:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        values = [0.0]
    if not any(np.isclose(values, 0.0, rtol=0.0, atol=1e-15)):
        values.append(0.0)
    return np.array(sorted(set(values)), dtype=float)


def parse_mc_grid(raw: str | None, mc_min: float, mc_max: float, n_mc: int) -> np.ndarray:
    if raw:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
        return np.array(sorted(set(values)), dtype=float)
    return np.geomspace(mc_min, mc_max, n_mc)


def make_white_noise_drawer(cfg: DemoConfig):
    sigma = np.sqrt(WHITE_PSD_TWO_SIDED * cfg.fs)

    def draw(chunk_index: int) -> np.ndarray:
        rng = np.random.default_rng(cfg.seed + 1009 * (chunk_index + 1))
        real = rng.normal(0.0, sigma / np.sqrt(2.0), cfg.n_samples_per_chunk)
        imag = rng.normal(0.0, sigma / np.sqrt(2.0), cfg.n_samples_per_chunk)
        return real + 1j * imag

    return draw


def make_white_psd():
    def Sn(freq_hz):
        return np.full_like(np.asarray(freq_hz, dtype=float), WHITE_PSD_TWO_SIDED)

    return Sn


def make_bilby_noise_and_psd(cfg: DemoConfig):
    import bilby
    from scipy.interpolate import interp1d

    bilby.core.utils.logger.setLevel("WARNING")
    ifo = bilby.gw.detector.InterferometerList(["H1"])[0]
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=cfg.fs,
        duration=cfg.chunk_duration,
        start_time=-cfg.chunk_duration / 2.0,
    )
    f_design = ifo.strain_data.frequency_array
    psd_design = ifo.power_spectral_density_array
    finite = np.isfinite(psd_design) & (psd_design > 0.0) & (f_design > 0.0)
    log_Sn = interp1d(
        np.log(f_design[finite]),
        np.log(psd_design[finite] / 2.0),
        kind="linear",
        bounds_error=False,
        fill_value=-np.inf,
    )

    def Sn(freq_hz):
        freq_hz = np.asarray(freq_hz, dtype=float)
        log_val = log_Sn(np.where(freq_hz > 0.0, np.log(np.maximum(freq_hz, 1e-30)), -np.inf))
        out = np.exp(log_val)
        out[freq_hz <= 0.0] = 0.0
        return out

    def draw(chunk_index: int) -> np.ndarray:
        np.random.seed(cfg.seed + 1009 * (chunk_index + 1))
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=cfg.fs,
            duration=cfg.chunk_duration,
            start_time=-cfg.chunk_duration / 2.0,
        )
        return ifo.strain_data.time_domain_strain.astype(complex)

    return Sn, draw


def analytical_S_eff(freq_out_hz: float, t_start: float, t_end: float, beta: float, Sn, n_t: int = 1000) -> float:
    t_quad = np.linspace(t_start, t_end, n_t, endpoint=False)
    dtau_dt = (1.0 - (8.0 / 3.0) * beta * t_quad) ** (-3.0 / 8.0)
    return float(np.mean(Sn(freq_out_hz * dtau_dt)))


def carrier_bin(data: np.ndarray, tau_local: np.ndarray, omega0: float, cfg: DemoConfig) -> tuple[complex, float]:
    resampler = Resampler(nthreads=cfg.nthreads, precision=cfg.precision)
    resampler.timeseries = data
    resampler.resampled_time = tau_local
    resampler.nufft()
    idx = int(np.abs(resampler.freqs - omega0).argmin())
    return complex(resampler.weights_normalized[idx]), float(resampler.freq_in_hz[idx])


def make_signal_chunk(
    cfg: DemoConfig,
    sidereal: five_vec,
    chirp_mass: float,
    beta_true: float,
    gamma: float,
    chunk_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = cfg.n_samples_per_chunk
    t_abs = chunk_index * cfg.chunk_duration + np.arange(n, dtype=float) / cfg.fs
    freq = instantaneous_frequency(cfg.f0, beta_true, t_abs)
    tau_abs = tau_absolute(beta_true, t_abs)
    mjd = REF_MJD + t_abs / 86400.0
    sidereal.compute_A(sidereal.gmst(mjd))
    amp = cfg.signal_scale * physical_amplitude(chirp_mass, freq)
    tau_ref = tau_absolute(beta_true, np.array([0.0]))[0]
    phase = 2.0 * np.pi * cfg.f0 * (tau_abs - tau_ref) + gamma
    signal = amp * sidereal.amp_modulation * np.exp(1j * phase)
    return t_abs, signal


def make_truncated_signal_chunk(
    cfg: DemoConfig,
    sidereal: five_vec,
    chirp_mass_solar: float,
    beta_true: float,
    gamma: float,
    chunk_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = cfg.n_samples_per_chunk
    t_abs = chunk_index * cfg.chunk_duration + np.arange(n, dtype=float) / cfg.fs
    base = 1.0 - (8.0 / 3.0) * beta_true * t_abs
    valid = base > 0.0
    freq = np.full_like(t_abs, np.nan)
    freq[valid] = cfg.f0 * base[valid] ** (-3.0 / 8.0)
    active = valid & (freq <= cfg.f_end)
    if np.count_nonzero(active) < 2:
        return t_abs[active], np.zeros(0, dtype=complex), freq[active]

    t_use = t_abs[active]
    freq_use = freq[active]
    mjd = REF_MJD + t_use / 86400.0
    sidereal.compute_A(sidereal.gmst(mjd))
    tau_abs = tau_absolute(beta_true, t_use)
    tau_ref = tau_absolute(beta_true, np.array([0.0]))[0]
    amp = cfg.signal_scale * physical_amplitude(chirp_mass_solar * M_SUN, freq_use)
    phase = 2.0 * np.pi * cfg.f0 * (tau_abs - tau_ref) + gamma
    signal = amp * sidereal.amp_modulation * np.exp(1j * phase)
    return t_use, signal, freq_use


def analyze(cfg: DemoConfig) -> dict:
    beta_true = beta_for_band(cfg.f0, cfg.f_end, cfg.t_analyzed)
    chirp_mass = chirp_mass_from_beta(cfg.f0, beta_true)
    omega0 = 2.0 * np.pi * cfg.f0
    sidereal = build_sidereal(cfg.seed)
    gamma = float(np.random.default_rng(cfg.seed + 1).uniform(0.0, 2.0 * np.pi))

    if cfg.noise == "white":
        Sn = make_white_psd()
        draw_noise = make_white_noise_drawer(cfg)
    else:
        Sn, draw_noise = make_bilby_noise_and_psd(cfg)

    beta_grid = beta_true * (1.0 + cfg.scan_fracs)
    beta_valid = np.array([(1.0 - (8.0 / 3.0) * beta * cfg.t_analyzed) > 0.0 for beta in beta_grid])
    stat_by_beta = np.zeros(beta_grid.size, dtype=float)
    stat_by_beta[~beta_valid] = np.nan
    true_idx = int(np.argmin(np.abs(cfg.scan_fracs)))
    per_chunk_true = np.zeros(cfg.n_chunks, dtype=float)
    per_chunk_freq = np.zeros(cfg.n_chunks, dtype=float)

    for chunk_index in range(cfg.n_chunks):
        if chunk_index % max(1, cfg.n_chunks // 10) == 0:
            print(f"  chunk {chunk_index + 1}/{cfg.n_chunks}")

        t_abs, signal = make_signal_chunk(cfg, sidereal, chirp_mass, beta_true, gamma, chunk_index)
        if cfg.mode == "noisy":
            data = signal + draw_noise(chunk_index)
        else:
            data = signal

        t_start = float(t_abs[0])
        t_end = float(t_abs[-1] + 1.0 / cfg.fs)
        for i, beta_assumed in enumerate(beta_grid):
            if not beta_valid[i]:
                continue
            tau_local = tau_from_beta(beta_assumed, t_abs)
            if not np.all(np.isfinite(tau_local)):
                stat_by_beta[i] = np.nan
                beta_valid[i] = False
                continue
            coeff, bin_freq_hz = carrier_bin(data, tau_local, omega0, cfg)
            S_eff = analytical_S_eff(abs(bin_freq_hz), t_start, t_end, beta_assumed, Sn)
            sigma2 = S_eff / cfg.chunk_duration
            if sigma2 <= 0.0 or not np.isfinite(sigma2):
                contribution = np.nan
            else:
                contribution = abs(coeff) ** 2 / sigma2
            stat_by_beta[i] += contribution
            if i == true_idx:
                per_chunk_true[chunk_index] = contribution
                per_chunk_freq[chunk_index] = bin_freq_hz

    return {
        "beta_true": beta_true,
        "chirp_mass": chirp_mass,
        "beta_grid": beta_grid,
        "beta_valid": beta_valid,
        "scan_fracs": cfg.scan_fracs,
        "stat_by_beta": stat_by_beta,
        "true_idx": true_idx,
        "per_chunk_true": per_chunk_true,
        "per_chunk_freq": per_chunk_freq,
        "noise_mean": float(cfg.n_chunks),
        "noise_std": float(np.sqrt(cfg.n_chunks)),
    }


def analyze_frequency_distribution(cfg: DemoConfig, mc_values: np.ndarray, n_freq_bins: int) -> dict:
    omega0 = 2.0 * np.pi * cfg.f0
    sidereal = build_sidereal(cfg.seed)
    gamma = float(np.random.default_rng(cfg.seed + 1).uniform(0.0, 2.0 * np.pi))

    if cfg.noise == "white":
        Sn = make_white_psd()
    else:
        Sn, _ = make_bilby_noise_and_psd(cfg)

    freq_edges = np.linspace(cfg.f0, cfg.f_end, n_freq_bins + 1)
    freq_centers = 0.5 * (freq_edges[:-1] + freq_edges[1:])
    curves = []

    for j, mc_solar in enumerate(mc_values):
        beta_true = beta_from_chirp_mass(cfg.f0, mc_solar)
        hist = np.zeros(n_freq_bins, dtype=float)
        chunk_freqs = []
        chunk_powers = []
        chunks_used = 0
        print(f"  Mc={mc_solar:.6g} Msun ({j + 1}/{len(mc_values)})")

        for chunk_index in range(cfg.n_chunks):
            t_use, signal, freq_use = make_truncated_signal_chunk(
                cfg, sidereal, mc_solar, beta_true, gamma, chunk_index
            )
            if signal.size == 0:
                continue
            tau_local = tau_from_beta(beta_true, t_use)
            if not np.all(np.isfinite(tau_local)):
                continue
            coeff, bin_freq_hz = carrier_bin(signal, tau_local, omega0, cfg)
            duration = signal.size / cfg.fs
            S_eff = analytical_S_eff(abs(bin_freq_hz), float(t_use[0]), float(t_use[-1] + 1.0 / cfg.fs), beta_true, Sn)
            sigma2 = S_eff / duration
            if sigma2 <= 0.0 or not np.isfinite(sigma2):
                continue

            power = abs(coeff) ** 2 / sigma2
            freq_mid = float(np.mean(freq_use))
            bin_idx = np.searchsorted(freq_edges, freq_mid, side="right") - 1
            if 0 <= bin_idx < n_freq_bins:
                hist[bin_idx] += power
            chunks_used += 1
            chunk_freqs.append(freq_mid)
            chunk_powers.append(power)

        total = float(np.sum(hist))
        density = hist / total / np.diff(freq_edges) if total > 0.0 else hist
        curves.append(
            {
                "mc_solar": float(mc_solar),
                "beta": float(beta_true),
                "hist": hist,
                "density": density,
                "total": total,
                "chunks_used": chunks_used,
                "chunk_freqs": np.array(chunk_freqs),
                "chunk_powers": np.array(chunk_powers),
            }
        )

    return {"freq_edges": freq_edges, "freq_centers": freq_centers, "curves": curves}


def make_plot(cfg: DemoConfig, result: dict) -> Path:
    beta_true = result["beta_true"]
    stat_by_beta = result["stat_by_beta"]
    true_idx = result["true_idx"]
    per_chunk = result["per_chunk_true"]
    noise_mean = result["noise_mean"]
    noise_std = result["noise_std"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)

    t_plot = np.linspace(0.0, cfg.t_analyzed, 800)
    axes[0].plot(t_plot / 3600.0, instantaneous_frequency(cfg.f0, beta_true, t_plot), lw=1.6)
    axes[0].set_xlabel("time [h]")
    axes[0].set_ylabel("instantaneous frequency [Hz]")
    axes[0].set_title("Injected chirp track")

    axes[1].plot(cfg.scan_fracs, stat_by_beta, marker="o", lw=1.4)
    axes[1].axvline(0.0, color="k", ls="--", lw=1.0, label="true beta")
    axes[1].axhline(noise_mean, color="C3", ls=":", lw=1.0, label="H0 mean")
    axes[1].fill_between(
        [cfg.scan_fracs.min(), cfg.scan_fracs.max()],
        noise_mean - noise_std,
        noise_mean + noise_std,
        color="C3",
        alpha=0.15,
        label="H0 +/- 1 std",
    )
    axes[1].set_xlabel("(beta - beta_true) / beta_true")
    axes[1].set_ylabel("summed carrier statistic")
    axes[1].set_title("Semicoherent beta scan")
    axes[1].legend(fontsize=8)

    axes[2].plot(np.arange(cfg.n_chunks), per_chunk, lw=1.1)
    axes[2].set_xlabel("1000 s chunk")
    axes[2].set_ylabel("carrier statistic")
    axes[2].set_title("True-template chunk powers")

    peak_idx = int(np.nanargmax(stat_by_beta))
    fig.suptitle(
        (
            f"{cfg.noise} / {cfg.mode}: sidereal modulation injected, carrier-only recovery. "
            f"peak frac={cfg.scan_fracs[peak_idx]:+.4g}, true stat={stat_by_beta[true_idx]:.3g}"
        ),
        fontsize=10,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"semicoherent_1day_1000s_{cfg.noise}_{cfg.mode}.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def make_distribution_plot(cfg: DemoConfig, result: dict) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(result["curves"])))

    for color, curve in zip(colors, result["curves"]):
        label = (
            f"Mc={curve['mc_solar']:.3g} Msun, "
            f"N={curve['chunks_used']}, stat={curve['total']:.2g}"
        )
        ax.step(result["freq_centers"], curve["density"], where="mid", color=color, lw=1.7, label=label)

    ax.set_xlabel("chunk mean instantaneous frequency [Hz]")
    ax.set_ylabel("fraction of semicoherent statistic per Hz")
    ax.set_title(
        (
            "True-template carrier-power distribution\n"
            f"{cfg.noise} PSD, no noise injection, signal truncated above {cfg.f_end:g} Hz"
        )
    )
    ax.set_xlim(cfg.f0, cfg.f_end)
    ax.legend(fontsize=8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"semicoherent_1day_1000s_{cfg.noise}_mc_power_distribution.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", choices=("white", "bilby"), default="white")
    parser.add_argument("--mode", choices=("signal_only", "noisy"), default="signal_only")
    parser.add_argument("--fs", type=float, default=512.0)
    parser.add_argument("--f0", type=float, default=20.0)
    parser.add_argument("--f-end", type=float, default=200.0)
    parser.add_argument("--chunk-duration", type=float, default=1000.0)
    parser.add_argument("--n-chunks", type=int, default=86)
    parser.add_argument("--signal-scale", type=float, default=DEFAULT_SIGNAL_SCALE)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scan-fracs", default="-0.004,-0.002,0,0.001,0.002")
    parser.add_argument("--precision", choices=("single", "double"), default="single")
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--power-distribution", action="store_true")
    parser.add_argument("--mc-values", default=None)
    parser.add_argument("--mc-min", type=float, default=1e-2)
    parser.add_argument("--mc-max", type=float, default=1e-1)
    parser.add_argument("--n-mc", type=int, default=6)
    parser.add_argument("--freq-bins", type=int, default=45)
    return parser


def validate_config(cfg: DemoConfig) -> None:
    if cfg.n_chunks <= 0:
        raise ValueError("n_chunks must be positive")
    if cfg.chunk_duration <= 0.0:
        raise ValueError("chunk_duration must be positive")
    if cfg.fs <= 0.0:
        raise ValueError("fs must be positive")
    if cfg.f0 <= 0.0 or cfg.f_end <= cfg.f0:
        raise ValueError("Require 0 < f0 < f_end")
    if cfg.f_end >= cfg.fs / 2.0:
        raise ValueError("f_end must be below Nyquist; increase --fs")
    if cfg.t_analyzed > SIDE_DAY:
        raise ValueError("Default demonstration should not exceed one sidereal day")
    _ = cfg.n_samples_per_chunk


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = DemoConfig(
        noise=args.noise,
        mode=args.mode,
        fs=args.fs,
        f0=args.f0,
        f_end=args.f_end,
        chunk_duration=args.chunk_duration,
        n_chunks=args.n_chunks,
        signal_scale=args.signal_scale,
        seed=args.seed,
        scan_fracs=parse_scan_fracs(args.scan_fracs),
        precision=args.precision,
        nthreads=args.nthreads,
        no_plot=args.no_plot,
    )
    validate_config(cfg)

    if args.power_distribution:
        mc_values = parse_mc_grid(args.mc_values, args.mc_min, args.mc_max, args.n_mc)
        print("=" * 72)
        print("Frequency distribution of semicoherent carrier power")
        print("=" * 72)
        print(f"noise={cfg.noise}, mode=signal_only, seed={cfg.seed}")
        print(f"chunks={cfg.n_chunks} x {cfg.chunk_duration:.3f} s = {cfg.t_analyzed:.3f} s")
        print(f"F_S={cfg.fs:.3f} Hz, samples/chunk={cfg.n_samples_per_chunk}")
        print(f"f0={cfg.f0:.6g} Hz, truncate signal above {cfg.f_end:.6g} Hz")
        print(f"Mc grid={mc_values}")
        print("No noise is injected; the selected PSD only normalizes the chunk powers.")
        result = analyze_frequency_distribution(cfg, mc_values, args.freq_bins)
        for curve in result["curves"]:
            print(
                f"  Mc={curve['mc_solar']:.6g} Msun: "
                f"chunks_used={curve['chunks_used']}, summed_stat={curve['total']:.6g}"
            )
        if not cfg.no_plot:
            output = make_distribution_plot(cfg, result)
            print(f"\nSaved plot -> {output}")
        return

    unused = SIDE_DAY - cfg.t_analyzed
    beta_true = beta_for_band(cfg.f0, cfg.f_end, cfg.t_analyzed)
    chirp_mass = chirp_mass_from_beta(cfg.f0, beta_true)

    print("=" * 72)
    print("1-day semicoherent carrier-only recovery")
    print("=" * 72)
    print(f"noise={cfg.noise}, mode={cfg.mode}, seed={cfg.seed}")
    print(f"chunks={cfg.n_chunks} x {cfg.chunk_duration:.3f} s = {cfg.t_analyzed:.3f} s")
    print(f"sidereal-day tail left unused = {unused:.3f} s")
    print(f"F_S={cfg.fs:.3f} Hz, samples/chunk={cfg.n_samples_per_chunk}")
    print(f"chirp track={cfg.f0:.6g} -> {cfg.f_end:.6g} Hz")
    print(f"beta={beta_true:.6e}  Mc={chirp_mass / M_SUN:.6g} Msun")
    print(f"signal_scale={cfg.signal_scale:.6g}")
    print("Recovery uses carrier-bin power only; no sidereal template is used.")

    result = analyze(cfg)
    stat_by_beta = result["stat_by_beta"]
    true_idx = result["true_idx"]
    peak_idx = int(np.nanargmax(stat_by_beta))

    print("\nSemicoherent statistic:")
    for frac, value in zip(result["scan_fracs"], stat_by_beta):
        marker = "  <-- true" if np.isclose(frac, 0.0, rtol=0.0, atol=1e-15) else ""
        print(f"  beta offset {frac:+.6g}: {value:.6g}{marker}")
    print(f"\npeak beta offset = {result['scan_fracs'][peak_idx]:+.6g}")
    print(f"true-template statistic = {stat_by_beta[true_idx]:.6g}")
    print(f"H0 mean +/- std = {result['noise_mean']:.6g} +/- {result['noise_std']:.6g}")

    if not cfg.no_plot:
        output = make_plot(cfg, result)
        print(f"\nSaved plot -> {output}")


if __name__ == "__main__":
    main()
