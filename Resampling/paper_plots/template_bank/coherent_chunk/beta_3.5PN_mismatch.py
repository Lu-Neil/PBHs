"""Build a 0PN beta bank that covers finite-duration 3.5PN waveforms.

Edit the experiment settings below, then run

    conda run -n PBH python template_bank/coherent_chunk/beta_3.5PN_mismatch.py

At each physical (f0, Mc) point, a TaylorT4 3.5PN signal is compared with 0PN
templates at the same f0.  The narrowest beta interval satisfying the mismatch
limit determines a conservative, regularly spaced template bank.  Interleaved
points validate the final bank.
"""

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
REPOSITORY_DIR = SCRIPT_DIR.parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

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


# Experiment settings -------------------------------------------------------
# These replace the former command-line arguments.  Paths are relative to this
# file, so the calculation is independent of the directory it is launched from.
F_MIN_HZ = 40.0
F_MAX_HZ = 60.0
MCHIRP_MIN_MSUN = 5.0e-4
MCHIRP_MAX_MSUN = 1.0e-1
ETA = 0.25
T_CHUNK_S = 30.0
MAX_MISMATCH = 0.05

SAMPLE_RATE_HZ = 512.0
ZERO_PAD_FACTOR = 4
AMPLITUDE_FREQUENCY_POWER = 2.0 / 3.0

# Construction nodes are uniform in f0 and physical Newtonian beta.  The bank
# is checked halfway between these nodes and refined if a point is uncovered.
N_F0_NODES = 11
N_BETA_NODES = 65
GRID_SAFETY_FACTOR = 0.90
MAX_REFINEMENT_ROUNDS = 3
PLACEMENT_MISMATCH = GRID_SAFETY_FACTOR * MAX_MISMATCH

ASD_PATH = REPOSITORY_DIR / "asd.txt"
PLOT_PATH = SCRIPT_DIR / "figs" / "beta_3.5PN_mismatch.png"
BANK_PATH = SCRIPT_DIR / "figs" / "beta_3.5PN_mismatch_bank.csv"


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
        return track_to_real_strain(track, AMPLITUDE_FREQUENCY_POWER)

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


def validate_settings():
    """Catch inconsistent experiment settings before the expensive loop."""

    if F_MIN_HZ <= 0.0 or F_MAX_HZ <= F_MIN_HZ:
        raise ValueError("frequency range must satisfy 0 < F_MIN_HZ < F_MAX_HZ")
    if MCHIRP_MIN_MSUN <= 0.0 or MCHIRP_MAX_MSUN <= MCHIRP_MIN_MSUN:
        raise ValueError("chirp-mass range must be positive and increasing")
    if not 0.0 < ETA <= 0.25:
        raise ValueError("ETA must be in the range (0, 0.25]")
    if T_CHUNK_S <= 0.0:
        raise ValueError("T_CHUNK_S must be positive")
    if not 0.0 < MAX_MISMATCH < 1.0:
        raise ValueError("MAX_MISMATCH must be in the range (0, 1)")
    if SAMPLE_RATE_HZ <= 2.0 * F_MAX_HZ:
        raise ValueError("SAMPLE_RATE_HZ must exceed twice F_MAX_HZ")
    if ZERO_PAD_FACTOR < 1:
        raise ValueError("ZERO_PAD_FACTOR must be at least 1")
    if N_F0_NODES < 2 or N_BETA_NODES < 2:
        raise ValueError("construction grids need at least two nodes per axis")
    if not 0.0 < GRID_SAFETY_FACTOR <= 1.0:
        raise ValueError("GRID_SAFETY_FACTOR must be in the range (0, 1]")
    if MAX_REFINEMENT_ROUNDS < 1:
        raise ValueError("MAX_REFINEMENT_ROUNDS must be positive")

    beta_max = beta_0pn(F_MAX_HZ, MCHIRP_MAX_MSUN)
    singular_beta = 3.0 / (8.0 * T_CHUNK_S)
    if beta_max >= singular_beta:
        raise ValueError("the 0PN beta range reaches coalescence within t_chunk")


def mchirp_from_beta(beta, f0_hz):
    """Invert the Newtonian beta relation, returning chirp mass in Msun."""

    reference_beta = beta_0pn(f0_hz, 1.0)
    return float((beta / reference_beta) ** (3.0 / 5.0))


def physical_parameter_grid():
    """Yield the construction grid, uniform in f0 and physical beta."""

    for f0_hz in np.linspace(F_MIN_HZ, F_MAX_HZ, N_F0_NODES):
        beta_values = np.linspace(
            beta_0pn(f0_hz, MCHIRP_MIN_MSUN),
            beta_0pn(f0_hz, MCHIRP_MAX_MSUN),
            N_BETA_NODES,
        )
        for beta in beta_values:
            yield float(f0_hz), mchirp_from_beta(beta, f0_hz)


def validation_parameter_grid():
    """Yield cell centres and boundary midpoints of the construction grid."""

    f0_nodes = np.linspace(F_MIN_HZ, F_MAX_HZ, N_F0_NODES)
    f0_midpoints = 0.5 * (f0_nodes[:-1] + f0_nodes[1:])
    beta_fractions = (np.arange(N_BETA_NODES - 1) + 0.5) / (N_BETA_NODES - 1)

    # Cell centres check simultaneous interpolation in both coordinates.
    for f0_hz in f0_midpoints:
        beta_low = beta_0pn(f0_hz, MCHIRP_MIN_MSUN)
        beta_high = beta_0pn(f0_hz, MCHIRP_MAX_MSUN)
        for fraction in beta_fractions:
            beta = beta_low + fraction * (beta_high - beta_low)
            yield float(f0_hz), mchirp_from_beta(beta, f0_hz)

    # Boundary midpoints make sure the four edges are not missed.
    for f0_hz in (F_MIN_HZ, F_MAX_HZ):
        beta_low = beta_0pn(f0_hz, MCHIRP_MIN_MSUN)
        beta_high = beta_0pn(f0_hz, MCHIRP_MAX_MSUN)
        for fraction in beta_fractions:
            beta = beta_low + fraction * (beta_high - beta_low)
            yield float(f0_hz), mchirp_from_beta(beta, f0_hz)

    for f0_hz in f0_midpoints:
        yield float(f0_hz), MCHIRP_MIN_MSUN
        yield float(f0_hz), MCHIRP_MAX_MSUN


def prepare_target(f0_hz, mchirp_msun, noise, t, dt):
    model = PNComparisonWaveformModel(
        f0_hz,
        mchirp_msun,
        ETA,
        T_CHUNK_S,
    )
    _, target_track = model.tracks(t)
    if np.max(target_track.frequency) >= 0.5 * SAMPLE_RATE_HZ:
        raise ValueError(
            "a 3.5PN target reaches the Nyquist frequency within t_chunk"
        )
    target = track_to_real_strain(target_track, AMPLITUDE_FREQUENCY_POWER)

    n_fft = ZERO_PAD_FACTOR * target.size
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
        taylor_t4_factor_35pn(model.phase_model.v_start, ETA)
    )
    return PreparedTarget(
        f0_hz=float(f0_hz),
        mchirp_msun=float(mchirp_msun),
        physical_beta=float(model.beta),
        effective_beta_guess=effective_beta_guess,
        t=t,
        dt=dt,
        n_fft=n_fft,
        band=band,
        weights=weights,
        target_tilde=target_tilde,
        target_norm=float(target_norm),
    )


def best_fitting_beta(target):
    """Return the local best-fitting effective beta and its mismatch."""

    singular_beta = 3.0 / (8.0 * T_CHUNK_S)
    phase_scale = 4.0 / (np.pi * target.f0_hz * T_CHUNK_S**2)
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


def mismatch_boundary(target, best_beta, fitting_mismatch, limit, direction):
    """Find one edge of the connected acceptable interval around best_beta."""

    if direction not in (-1.0, 1.0):
        raise ValueError("direction must be -1 or +1")
    singular_beta = 3.0 / (8.0 * T_CHUNK_S)
    domain_edge = 0.0 if direction < 0.0 else 0.999999 * singular_beta
    remaining_mismatch = max(limit - fitting_mismatch, np.finfo(float).eps)
    mismatch_scale = np.sqrt(45.0 * remaining_mismatch / 2.0) / (
        np.pi * target.f0_hz * T_CHUNK_S**2
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


def acceptable_beta_interval(target, mismatch_limit=None):
    if mismatch_limit is None:
        mismatch_limit = PLACEMENT_MISMATCH
    best_beta, fitting_mismatch = best_fitting_beta(target)
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
    )
    beta_high = mismatch_boundary(
        target,
        best_beta,
        fitting_mismatch,
        mismatch_limit,
        1.0,
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


def conservative_interval_cover(intervals):
    """Return a regular bank covered by every sampled interval radius.

    Interval edges at beta=0 or at the formal 0PN singularity are truncated
    parameter-domain boundaries, rather than mismatch contours, and therefore
    do not constrain the interior template spacing.
    """

    singular_beta = 3.0 / (8.0 * T_CHUNK_S)
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


def build_beta_bank(noise):
    t, dt = time_samples(T_CHUNK_S, SAMPLE_RATE_HZ)
    intervals = []
    for f0_hz, mchirp_msun in physical_parameter_grid():
        target = prepare_target(f0_hz, mchirp_msun, noise, t, dt)
        intervals.append(acceptable_beta_interval(target))

    betas = conservative_interval_cover(intervals)
    return betas, intervals


def validate_and_refine_bank(betas, intervals, noise):
    """Validate at interleaved points and add uncovered intervals if needed."""

    t, dt = time_samples(T_CHUNK_S, SAMPLE_RATE_HZ)
    validation_points = tuple(validation_parameter_grid())
    last_results = None

    for refinement_round in range(MAX_REFINEMENT_ROUNDS):
        results = []
        missed_targets = []
        for f0_hz, mchirp_msun in validation_points:
            target = prepare_target(f0_hz, mchirp_msun, noise, t, dt)
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
            if mismatch > MAX_MISMATCH:
                missed_targets.append(target)

        last_results = np.asarray(results, dtype=float)
        if not missed_targets:
            return betas, intervals, last_results, refinement_round

        for target in missed_targets:
            intervals.append(acceptable_beta_interval(target))
        betas = conservative_interval_cover(intervals)

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


def plot_bank(betas, validation_results):
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
        rf"3.5PN $\rightarrow$ 0PN bank, $T_{{\rm coh}}={T_CHUNK_S:g}$ s, "
        rf"maximum mismatch $={MAX_MISMATCH:g}$"
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
    ax_mismatch.axhline(MAX_MISMATCH, color="black", ls="--", lw=1.0)
    ax_mismatch.set_xscale("log")
    ax_mismatch.set_xlabel(r"Physical Newtonian $\beta(f_0,M_c)$")
    ax_mismatch.set_ylabel("Nearest-template mismatch")
    ax_mismatch.grid(True, which="both", alpha=0.25)
    colorbar = fig.colorbar(points, ax=ax_mismatch)
    colorbar.set_label(r"$f_0$ [Hz]")
    return fig


def print_summary(
    betas,
    intervals,
    validation_results,
    refinement_rounds,
):
    worst_index = int(np.argmax(validation_results[:, 1]))
    worst = validation_results[worst_index]
    worst_fitting = max(intervals, key=lambda item: item.fitting_mismatch)
    spacings = np.diff(betas)
    print("3.5PN-covered effective-beta bank")
    print(f"f0 range: {F_MIN_HZ:g} Hz to {F_MAX_HZ:g} Hz")
    print(
        f"Mc range: {MCHIRP_MIN_MSUN:.6e} Msun to "
        f"{MCHIRP_MAX_MSUN:.6e} Msun"
    )
    print(f"eta: {ETA:g}")
    print(f"T_chunk: {T_CHUNK_S:g} s")
    print(f"max_mismatch: {MAX_MISMATCH:.12g}")
    print(f"placement mismatch: {PLACEMENT_MISMATCH:.12g}")
    print(f"construction grid: {N_F0_NODES} x {N_BETA_NODES}")
    print(f"validation points: {validation_results.shape[0]}")
    print(f"refinement rounds used: {refinement_rounds}")
    print(f"templates required: {betas.size}")
    print(f"template beta range: {betas[0]:.12e} to {betas[-1]:.12e}")
    if spacings.size:
        print(
            f"template spacing range: {spacings.min():.12e} to "
            f"{spacings.max():.12e}"
        )
    print(
        "maximum fitting mismatch: "
        f"{worst_fitting.fitting_mismatch:.12e} at "
        f"f0={worst_fitting.f0_hz:g} Hz, "
        f"Mc={worst_fitting.mchirp_msun:.6e} Msun"
    )
    print(
        "maximum validation mismatch: "
        f"{worst[1]:.12e} at f0={worst[2]:g} Hz, "
        f"Mc={worst[3]:.6e} Msun"
    )
    print(f"asd: {ASD_PATH}")
    print(f"bank: {BANK_PATH}")
    print(f"plot: {PLOT_PATH}")


def main():
    validate_settings()
    noise = NoiseCurve.from_asd_file(ASD_PATH)
    betas, intervals = build_beta_bank(noise)
    betas, intervals, validation_results, refinement_rounds = (
        validate_and_refine_bank(betas, intervals, noise)
    )

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_bank(betas, validation_results)
    fig.savefig(PLOT_PATH, dpi=220)
    write_bank_csv(BANK_PATH, betas)
    print_summary(
        betas,
        intervals,
        validation_results,
        refinement_rounds,
    )


if __name__ == "__main__":
    main()
