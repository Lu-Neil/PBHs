"""Build a 0PN beta bank that covers finite-duration 3.5PN waveforms.

The carrier frequency is treated as a separately searched coordinate.  At
each physical ``(f0, Mc)`` point this script compares a TaylorT4 3.5PN target
with 0PN templates having the same ``f0`` and varying beta.  The acceptable
template-beta interval is the connected interval around the best-fitting beta
for which the ASD-weighted mismatch is no larger than the requested limit.

The narrowest acceptable distance from a best-fitting beta to either interval
edge sets a conservative regular-bank spacing.  Interleaved points in ``f0``
and physical beta are then used to validate the bank and add any constraints
missed by the construction grid.  ``grid_safety_factor`` leaves margin for
the continuum between the sampled parameter-space points.
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize_scalar


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_PLOTS_DIR = SCRIPT_DIR.parents[1]
ASD_PATH = PAPER_PLOTS_DIR / "asd.txt"
OUTPUT_PATH = SCRIPT_DIR / "figs" / "beta_3.5PN_mismatch.png"
BANK_OUTPUT_PATH = SCRIPT_DIR / "figs" / "beta_3.5PN_mismatch_bank.csv"

if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from signal_generators import (  # noqa: E402
    NoiseCurve,
    SignalTrack,
    beta_0pn,
    make_0pn_track,
    taylor_t4_factor_35pn,
)
from template_bank.coherent_chunk.pn_mismatch import (  # noqa: E402
    PNComparisonWaveformModel,
    time_samples,
    track_to_real_strain,
)


@dataclass
class Beta35PNMismatchConfig:
    f_min: float = 40.0
    f_max: float = 60.0
    mchirp_min: float = 5.0e-4
    mchirp_max: float = 1.0e-1
    eta: float = 0.25
    t_chunk: float = 30.0
    max_mismatch: float = 0.05
    sample_rate: float = 512.0
    zero_pad_factor: int = 4
    amplitude_frequency_power: float = 2.0 / 3.0
    n_f0_bank: int = 11
    n_beta_bank: int = 65
    grid_safety_factor: float = 0.90
    max_refinement_rounds: int = 3

    @property
    def beta_min(self):
        return beta_0pn(self.f_min, self.mchirp_min)

    @property
    def beta_max(self):
        return beta_0pn(self.f_max, self.mchirp_max)

    @property
    def placement_mismatch(self):
        return self.grid_safety_factor * self.max_mismatch


@dataclass(frozen=True)
class AcceptableBetaInterval:
    f0_hz: float
    mchirp_msun: float
    physical_beta: float
    best_beta: float
    fitting_mismatch: float
    beta_low: float
    beta_high: float


@dataclass
class PreparedTarget:
    """A cached 3.5PN target and its frequency-domain inner-product data."""

    f0_hz: float
    mchirp_msun: float
    physical_beta: float
    effective_beta_guess: float
    t: np.ndarray
    dt: float
    amplitude_frequency_power: float
    n_fft: int
    band: np.ndarray
    weights: np.ndarray
    target_tilde: np.ndarray
    target_norm: float

    def template_strain(self, beta):
        if beta == 0.0:
            phase = 2.0 * np.pi * self.f0_hz * self.t
            track = SignalTrack(
                t=self.t,
                frequency=np.full_like(self.t, self.f0_hz),
                phase=phase,
                signal=np.exp(1j * phase),
            )
        else:
            track = make_0pn_track(
                self.t,
                f0_hz=self.f0_hz,
                mchirp_msun=None,
                beta=beta,
            )
        return track_to_real_strain(track, self.amplitude_frequency_power)

    def mismatch(self, beta):
        if beta < 0.0:
            return 1.0
        template = self.template_strain(float(beta))
        template_tilde = self.dt * np.fft.rfft(template, n=self.n_fft)
        template_tilde = template_tilde[self.band]
        template_norm = np.sum(self.weights * np.abs(template_tilde) ** 2).real
        if template_norm <= 0.0:
            raise ValueError("0PN template norm is not positive.")
        cross_term = np.sum(
            self.weights * np.conj(self.target_tilde) * template_tilde
        )
        match = np.abs(cross_term) / np.sqrt(self.target_norm * template_norm)
        return 1.0 - float(np.clip(match, 0.0, 1.0))


def validate_config(config):
    if config.f_min <= 0.0 or config.f_max <= config.f_min:
        raise ValueError("frequency range must satisfy 0 < f_min < f_max")
    if config.mchirp_min <= 0.0 or config.mchirp_max <= config.mchirp_min:
        raise ValueError("chirp-mass range must satisfy 0 < Mc_min < Mc_max")
    if not 0.0 < config.eta <= 0.25:
        raise ValueError("eta must be in the range (0, 0.25]")
    if config.t_chunk <= 0.0:
        raise ValueError("t_chunk must be positive")
    if not 0.0 < config.max_mismatch < 1.0:
        raise ValueError("max_mismatch must be in the range (0, 1)")
    if config.sample_rate <= 2.0 * config.f_max:
        raise ValueError("sample_rate must exceed twice f_max")
    if config.zero_pad_factor < 1:
        raise ValueError("zero_pad_factor must be at least 1")
    if config.n_f0_bank < 2 or config.n_beta_bank < 2:
        raise ValueError("bank grids must contain at least two points per axis")
    if not 0.0 < config.grid_safety_factor <= 1.0:
        raise ValueError("grid_safety_factor must be in the range (0, 1]")
    if config.max_refinement_rounds < 1:
        raise ValueError("max_refinement_rounds must be positive")
    singular_beta = 3.0 / (8.0 * config.t_chunk)
    if config.beta_max >= singular_beta:
        raise ValueError("the 0PN beta range reaches coalescence within t_chunk")


def mchirp_from_beta(beta, f0_hz):
    """Invert the Newtonian beta relation, returning chirp mass in Msun."""

    reference_beta = beta_0pn(f0_hz, 1.0)
    return float((beta / reference_beta) ** (3.0 / 5.0))


def physical_parameter_grid(config):
    """Yield the construction grid, uniform in f0 and physical beta."""

    for f0_hz in np.linspace(config.f_min, config.f_max, config.n_f0_bank):
        beta_values = np.linspace(
            beta_0pn(f0_hz, config.mchirp_min),
            beta_0pn(f0_hz, config.mchirp_max),
            config.n_beta_bank,
        )
        for beta in beta_values:
            yield float(f0_hz), mchirp_from_beta(beta, f0_hz)


def validation_parameter_grid(config):
    """Yield cell centres and boundary midpoints of the construction grid."""

    f0_nodes = np.linspace(config.f_min, config.f_max, config.n_f0_bank)
    f0_midpoints = 0.5 * (f0_nodes[:-1] + f0_nodes[1:])
    beta_fractions = (np.arange(config.n_beta_bank - 1) + 0.5) / (
        config.n_beta_bank - 1
    )

    # Cell centres check simultaneous interpolation in both coordinates.
    for f0_hz in f0_midpoints:
        beta_low = beta_0pn(f0_hz, config.mchirp_min)
        beta_high = beta_0pn(f0_hz, config.mchirp_max)
        for fraction in beta_fractions:
            beta = beta_low + fraction * (beta_high - beta_low)
            yield float(f0_hz), mchirp_from_beta(beta, f0_hz)

    # Boundary midpoints make sure the four edges are not missed.
    for f0_hz in (config.f_min, config.f_max):
        beta_low = beta_0pn(f0_hz, config.mchirp_min)
        beta_high = beta_0pn(f0_hz, config.mchirp_max)
        for fraction in beta_fractions:
            beta = beta_low + fraction * (beta_high - beta_low)
            yield float(f0_hz), mchirp_from_beta(beta, f0_hz)

    for f0_hz in f0_midpoints:
        yield float(f0_hz), config.mchirp_min
        yield float(f0_hz), config.mchirp_max


def prepare_target(f0_hz, mchirp_msun, config, noise, t, dt):
    model = PNComparisonWaveformModel(
        f0_hz,
        mchirp_msun,
        config.eta,
        config.t_chunk,
    )
    _, target_track = model.tracks(t)
    if np.max(target_track.frequency) >= 0.5 * config.sample_rate:
        raise ValueError(
            "a 3.5PN target reaches the Nyquist frequency within t_chunk"
        )
    target = track_to_real_strain(
        target_track,
        config.amplitude_frequency_power,
    )

    n_fft = max(target.size, config.zero_pad_factor * target.size)
    frequency = np.fft.rfftfreq(n_fft, dt)
    band = (
        (frequency > 0.0)
        & (frequency >= noise.frequency[0])
        & (frequency <= noise.frequency[-1])
    )
    if not np.any(band):
        raise ValueError("no FFT bins overlap the ASD frequency range")

    df = frequency[1] - frequency[0]
    weights = 4.0 * df / noise.psd_at(frequency[band])
    target_tilde = (dt * np.fft.rfft(target, n=n_fft))[band]
    target_norm = np.sum(weights * np.abs(target_tilde) ** 2).real
    if target_norm <= 0.0:
        raise ValueError("3.5PN target norm is not positive")

    effective_beta_guess = model.beta * float(
        taylor_t4_factor_35pn(model.phase_model.v_start, config.eta)
    )
    return PreparedTarget(
        f0_hz=float(f0_hz),
        mchirp_msun=float(mchirp_msun),
        physical_beta=float(model.beta),
        effective_beta_guess=effective_beta_guess,
        t=t,
        dt=dt,
        amplitude_frequency_power=config.amplitude_frequency_power,
        n_fft=n_fft,
        band=band,
        weights=weights,
        target_tilde=target_tilde,
        target_norm=float(target_norm),
    )


def best_fitting_beta(target, config):
    """Return the local best-fitting effective beta and its mismatch."""

    singular_beta = 3.0 / (8.0 * config.t_chunk)
    phase_scale = 4.0 / (
        np.pi * target.f0_hz * config.t_chunk**2
    )
    search_half_width = max(
        0.03 * target.physical_beta,
        phase_scale,
        2.0 * abs(target.physical_beta - target.effective_beta_guess),
    )
    lower = max(0.0, target.effective_beta_guess - search_half_width)
    upper = min(
        0.999999 * singular_beta,
        target.effective_beta_guess + search_half_width,
    )
    result = minimize_scalar(
        target.mismatch,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1.0e-15},
    )
    candidates = (
        (float(result.x), float(result.fun)),
        (lower, target.mismatch(lower)),
        (upper, target.mismatch(upper)),
        (
            target.effective_beta_guess,
            target.mismatch(target.effective_beta_guess),
        ),
    )
    return min(candidates, key=lambda item: item[1])


def mismatch_boundary(target, best_beta, fitting_mismatch, limit, direction, config):
    """Find one edge of the connected acceptable interval around best_beta."""

    if direction not in (-1.0, 1.0):
        raise ValueError("direction must be -1 or +1")
    singular_beta = 3.0 / (8.0 * config.t_chunk)
    domain_edge = 0.0 if direction < 0.0 else 0.999999 * singular_beta
    remaining_mismatch = max(limit - fitting_mismatch, np.finfo(float).eps)
    mismatch_scale = np.sqrt(45.0 * remaining_mismatch / 2.0) / (
        np.pi * target.f0_hz * config.t_chunk**2
    )
    step = max(mismatch_scale / 4.0, 1.0e-12)
    inside = best_beta

    for _ in range(128):
        candidate = inside + direction * step
        if direction < 0.0:
            candidate = max(candidate, domain_edge)
        else:
            candidate = min(candidate, domain_edge)
        residual = target.mismatch(candidate) - limit
        if residual > 0.0:
            a, b = sorted((inside, candidate))
            return float(
                brentq(
                    lambda beta: target.mismatch(beta) - limit,
                    a,
                    b,
                    rtol=1.0e-11,
                    xtol=1.0e-16,
                )
            )
        if candidate == domain_edge:
            return float(domain_edge)
        inside = candidate
        step *= 1.5

    raise RuntimeError("could not find an acceptable-beta interval boundary")


def acceptable_beta_interval(target, config, mismatch_limit=None):
    if mismatch_limit is None:
        mismatch_limit = config.placement_mismatch
    best_beta, fitting_mismatch = best_fitting_beta(target, config)
    if fitting_mismatch >= mismatch_limit:
        raise RuntimeError(
            "the 0PN family cannot cover a 3.5PN waveform within the requested "
            f"mismatch: best mismatch={fitting_mismatch:.12e} at "
            f"f0={target.f0_hz:g} Hz, Mc={target.mchirp_msun:.12e} Msun"
        )
    beta_low = mismatch_boundary(
        target,
        best_beta,
        fitting_mismatch,
        mismatch_limit,
        -1.0,
        config,
    )
    beta_high = mismatch_boundary(
        target,
        best_beta,
        fitting_mismatch,
        mismatch_limit,
        1.0,
        config,
    )
    return AcceptableBetaInterval(
        f0_hz=target.f0_hz,
        mchirp_msun=target.mchirp_msun,
        physical_beta=target.physical_beta,
        best_beta=best_beta,
        fitting_mismatch=fitting_mismatch,
        beta_low=beta_low,
        beta_high=beta_high,
    )


def conservative_interval_cover(intervals, config):
    """Return a regular bank covered by every sampled interval radius.

    Interval edges at beta=0 or at the formal 0PN singularity are truncated
    parameter-domain boundaries, rather than mismatch contours, and therefore
    do not constrain the interior template spacing.
    """

    singular_beta = 3.0 / (8.0 * config.t_chunk)
    lower_radii = [
        interval.best_beta - interval.beta_low
        for interval in intervals
        if interval.beta_low > 0.0
    ]
    upper_radii = [
        interval.beta_high - interval.best_beta
        for interval in intervals
        if interval.beta_high < 0.999998 * singular_beta
    ]
    radii = np.asarray(lower_radii + upper_radii)
    if radii.size == 0 or np.any(radii <= 0.0):
        raise RuntimeError("could not determine a positive beta covering radius")

    covering_radius = float(np.min(radii))
    effective_beta_min = min(interval.best_beta for interval in intervals)
    effective_beta_max = max(interval.best_beta for interval in intervals)
    beta_span = effective_beta_max - effective_beta_min
    if beta_span == 0.0:
        return np.array([effective_beta_min])

    n_templates = max(1, int(np.ceil(beta_span / (2.0 * covering_radius))))
    spacing = beta_span / n_templates
    return effective_beta_min + spacing * (np.arange(n_templates) + 0.5)


def nearest_template_mismatch(target, betas):
    """Evaluate the templates nearest the target's effective-beta estimate."""

    insertion = int(np.searchsorted(betas, target.effective_beta_guess))
    indices = range(max(0, insertion - 2), min(betas.size, insertion + 3))
    values = [(target.mismatch(betas[index]), index) for index in indices]
    return min(values)


def build_beta_bank(config, noise):
    validate_config(config)
    t, dt = time_samples(config.t_chunk, config.sample_rate)
    intervals = []
    for f0_hz, mchirp_msun in physical_parameter_grid(config):
        target = prepare_target(f0_hz, mchirp_msun, config, noise, t, dt)
        intervals.append(acceptable_beta_interval(target, config))

    betas = conservative_interval_cover(intervals, config)
    return betas, intervals


def validate_and_refine_bank(betas, intervals, config, noise):
    """Validate at interleaved points and add uncovered intervals if needed."""

    t, dt = time_samples(config.t_chunk, config.sample_rate)
    validation_points = tuple(validation_parameter_grid(config))
    last_results = None

    for refinement_round in range(config.max_refinement_rounds):
        results = []
        missed_targets = []
        for f0_hz, mchirp_msun in validation_points:
            target = prepare_target(f0_hz, mchirp_msun, config, noise, t, dt)
            mismatch, template_index = nearest_template_mismatch(target, betas)
            results.append(
                (
                    target.physical_beta,
                    mismatch,
                    f0_hz,
                    mchirp_msun,
                    template_index,
                )
            )
            if mismatch > config.max_mismatch:
                missed_targets.append(target)

        last_results = np.asarray(results, dtype=float)
        if not missed_targets:
            return betas, intervals, last_results, refinement_round

        for target in missed_targets:
            intervals.append(acceptable_beta_interval(target, config))
        betas = conservative_interval_cover(intervals, config)

    worst = last_results[np.argmax(last_results[:, 1])]
    raise RuntimeError(
        "bank refinement did not satisfy the validation grid: "
        f"mismatch={worst[1]:.12e} at f0={worst[2]:g} Hz, "
        f"Mc={worst[3]:.12e} Msun"
    )


def write_bank_csv(path, betas):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["template_number", "beta"])
        writer.writerows(enumerate(betas))


def plot_bank(betas, validation_results, config):
    fig, (ax_bank, ax_mismatch) = plt.subplots(
        2,
        1,
        figsize=(7.6, 6.4),
        constrained_layout=True,
    )
    ax_bank.plot(np.arange(betas.size), betas, marker="o", ms=2.8, lw=1.0)
    ax_bank.set_yscale("log")
    ax_bank.set_xlabel("Template number")
    ax_bank.set_ylabel(r"Effective template $\beta$")
    ax_bank.set_title(
        rf"3.5PN $\rightarrow$ 0PN bank, $T_{{\rm coh}}={config.t_chunk:g}$ s, "
        rf"maximum mismatch $={config.max_mismatch:g}$"
    )
    ax_bank.grid(True, which="both", alpha=0.25)

    points = ax_mismatch.scatter(
        validation_results[:, 0],
        validation_results[:, 1],
        c=validation_results[:, 2],
        s=8,
        cmap="viridis",
        rasterized=True,
    )
    ax_mismatch.axhline(config.max_mismatch, color="black", ls="--", lw=1.0)
    ax_mismatch.set_xscale("log")
    ax_mismatch.set_xlabel(r"Physical Newtonian $\beta(f_0,M_c)$")
    ax_mismatch.set_ylabel("Nearest-template mismatch")
    ax_mismatch.grid(True, which="both", alpha=0.25)
    colorbar = fig.colorbar(points, ax=ax_mismatch)
    colorbar.set_label(r"$f_0$ [Hz]")
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build an effective-beta 0PN template bank that covers TaylorT4 "
            "3.5PN waveforms at a fixed coherent duration."
        )
    )
    parser.add_argument("--f-min", type=float, default=Beta35PNMismatchConfig.f_min)
    parser.add_argument("--f-max", type=float, default=Beta35PNMismatchConfig.f_max)
    parser.add_argument(
        "--mchirp-min",
        "--mc-min",
        dest="mchirp_min",
        type=float,
        default=Beta35PNMismatchConfig.mchirp_min,
    )
    parser.add_argument(
        "--mchirp-max",
        "--mc-max",
        dest="mchirp_max",
        type=float,
        default=Beta35PNMismatchConfig.mchirp_max,
    )
    parser.add_argument("--eta", type=float, default=Beta35PNMismatchConfig.eta)
    parser.add_argument(
        "--t-chunk",
        "--t-coh",
        dest="t_chunk",
        type=float,
        default=Beta35PNMismatchConfig.t_chunk,
    )
    parser.add_argument(
        "--max-mismatch",
        type=float,
        default=Beta35PNMismatchConfig.max_mismatch,
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=Beta35PNMismatchConfig.sample_rate,
    )
    parser.add_argument(
        "--zero-pad-factor",
        type=int,
        default=Beta35PNMismatchConfig.zero_pad_factor,
    )
    parser.add_argument(
        "--amplitude-frequency-power",
        type=float,
        default=Beta35PNMismatchConfig.amplitude_frequency_power,
    )
    parser.add_argument(
        "--n-f0-bank",
        type=int,
        default=Beta35PNMismatchConfig.n_f0_bank,
    )
    parser.add_argument(
        "--n-beta-bank",
        type=int,
        default=Beta35PNMismatchConfig.n_beta_bank,
    )
    parser.add_argument(
        "--grid-safety-factor",
        type=float,
        default=Beta35PNMismatchConfig.grid_safety_factor,
    )
    parser.add_argument(
        "--max-refinement-rounds",
        type=int,
        default=Beta35PNMismatchConfig.max_refinement_rounds,
    )
    parser.add_argument("--asd", type=Path, default=ASD_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--bank-output", type=Path, default=BANK_OUTPUT_PATH)
    return parser.parse_args()


def config_from_args(args):
    return Beta35PNMismatchConfig(
        f_min=args.f_min,
        f_max=args.f_max,
        mchirp_min=args.mchirp_min,
        mchirp_max=args.mchirp_max,
        eta=args.eta,
        t_chunk=args.t_chunk,
        max_mismatch=args.max_mismatch,
        sample_rate=args.sample_rate,
        zero_pad_factor=args.zero_pad_factor,
        amplitude_frequency_power=args.amplitude_frequency_power,
        n_f0_bank=args.n_f0_bank,
        n_beta_bank=args.n_beta_bank,
        grid_safety_factor=args.grid_safety_factor,
        max_refinement_rounds=args.max_refinement_rounds,
    )


def print_summary(
    config,
    asd_path,
    output_path,
    bank_output_path,
    betas,
    intervals,
    validation_results,
    refinement_rounds,
):
    worst_index = int(np.argmax(validation_results[:, 1]))
    worst = validation_results[worst_index]
    worst_fitting = max(intervals, key=lambda item: item.fitting_mismatch)
    spacings = np.diff(betas)
    lines = [
        "3.5PN-covered effective-beta bank",
        f"f0 range: {config.f_min:g} Hz to {config.f_max:g} Hz",
        (
            f"Mc range: {config.mchirp_min:.6e} Msun to "
            f"{config.mchirp_max:.6e} Msun"
        ),
        f"eta: {config.eta:g}",
        f"T_chunk: {config.t_chunk:g} s",
        f"max_mismatch: {config.max_mismatch:.12g}",
        f"placement mismatch: {config.placement_mismatch:.12g}",
        f"construction grid: {config.n_f0_bank} x {config.n_beta_bank}",
        f"validation points: {validation_results.shape[0]}",
        f"refinement rounds used: {refinement_rounds}",
        f"templates required: {betas.size}",
        f"template beta range: {betas[0]:.12e} to {betas[-1]:.12e}",
        (
            "maximum fitting mismatch: "
            f"{worst_fitting.fitting_mismatch:.12e} at "
            f"f0={worst_fitting.f0_hz:g} Hz, "
            f"Mc={worst_fitting.mchirp_msun:.6e} Msun"
        ),
        (
            "maximum validation mismatch: "
            f"{worst[1]:.12e} at f0={worst[2]:g} Hz, "
            f"Mc={worst[3]:.6e} Msun"
        ),
        f"asd: {asd_path}",
        f"bank: {bank_output_path}",
        f"plot: {output_path}",
    ]
    if spacings.size:
        lines.insert(
            12,
            f"template spacing range: {spacings.min():.12e} to "
            f"{spacings.max():.12e}",
        )
    print("\n".join(lines))


def main():
    args = parse_args()
    config = config_from_args(args)
    noise = NoiseCurve.from_asd_file(args.asd)
    betas, intervals = build_beta_bank(config, noise)
    betas, intervals, validation_results, refinement_rounds = (
        validate_and_refine_bank(betas, intervals, config, noise)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_bank(betas, validation_results, config)
    fig.savefig(args.output, dpi=220)
    write_bank_csv(args.bank_output, betas)
    print_summary(
        config,
        args.asd,
        args.output,
        args.bank_output,
        betas,
        intervals,
        validation_results,
        refinement_rounds,
    )


if __name__ == "__main__":
    main()
