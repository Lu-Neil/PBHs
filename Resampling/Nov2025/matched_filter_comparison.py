"""Compare the resampling + 5-vector pipeline against exact frequency-domain matched filtering.

This script uses the same class structure as the current `Resampling/Nov2025`
pipeline, but compares it to a conventional matched filter built from the exact
time-domain template and evaluated in the frequency domain with FFTs.

The comparison is set up so both methods should agree up to numerical error:

- the phase evolution is either PBH-like or quadratic-in-time (`fdot`)
- the intrinsic complex amplitude is constant
- the observation spans an integer number of sidereal days
- the carrier is injected at the midpoint of a Fourier bin by default

Because the intrinsic amplitude is constant, both methods are solving the same
estimation problem. The script reports:

- numerical agreement of the recovered complex amplitudes
- runtime measurements as the signal duration grows

The default `f0_setting="midpoint"` is the regime where the current
resampling implementation and the exact matched filter should agree most
tightly. The optional `uniform` mode is still useful for experimentation, but
it is no longer a strict apples-to-apples numerical check because the present
5-vector extraction keeps only the nearest Fourier bins.

Usage examples:

    python Resampling/Nov2025/matched_filter_comparison.py
    python Resampling/Nov2025/matched_filter_comparison.py --phase-model fdot
    python Resampling/Nov2025/matched_filter_comparison.py --days 1 2 4 8 --repeats 5
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
from astropy.time import Time

try:
    from .five_vec import five_vec
    from .resampler import Resampler
except ImportError:
    from five_vec import five_vec
    from resampler import Resampler


PBH_CONST = 96 / 5 * np.pi ** (8 / 3) * (6.67e-11 / 3e8**3) ** (5 / 3)


@dataclass
class SimulationData:
    signal: np.ndarray
    tau: np.ndarray
    omega0: float
    t: Time
    target_h: complex
    template: np.ndarray
    template_p: np.ndarray
    template_c: np.ndarray
    n_samples: int
    number_of_days: float


@dataclass
class FilterOutputs:
    h_est: complex
    hp_est: complex
    hc_est: complex


@dataclass
class ComparisonRow:
    number_of_days: float
    n_samples: int
    h_rel_err: float
    hp_rel_err: float
    hc_rel_err: float
    h_phase_err: float
    hp_phase_err: float
    hc_phase_err: float
    resampling_time_s: float
    matched_filter_time_s: float


def _estimator(data_vec: np.ndarray, template_vec: np.ndarray) -> complex:
    return np.vdot(template_vec, data_vec) / np.vdot(template_vec, template_vec)


def _joint_estimator(
    data_vec: np.ndarray,
    template_p_vec: np.ndarray,
    template_c_vec: np.ndarray,
) -> tuple[complex, complex]:
    design = np.column_stack([template_p_vec, template_c_vec])
    hp_est, hc_est = np.linalg.lstsq(design, data_vec, rcond=None)[0]
    return hp_est, hc_est


def _wrapped_phase_diff(phi_a: float, phi_b: float) -> float:
    return float(np.angle(np.exp(1j * (phi_a - phi_b))))


def _relative_error(estimate: complex, target: complex) -> float:
    return float(abs(estimate - target) / abs(target))


def _resample_and_extract_5vec(signal: np.ndarray, tau: np.ndarray, omega0: float) -> np.ndarray:
    resampler = Resampler()
    resampler.timeseries = signal
    resampler.resampled_time = tau
    resampler.nufft()
    return resampler.extract_5vec(omega0)


def _sidereal_templates(
    sidereal: five_vec,
    t: Time,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sidereal_t = sidereal.gmst(t.mjd)
    sidereal_t -= sidereal_t[0]
    exp_terms = np.exp(1j * (np.arange(5) - 2)[:, np.newaxis] * sidereal_t)

    template_p = np.dot(sidereal.A_p, exp_terms)
    template_c = np.dot(sidereal.A_c, exp_terms)
    template = np.dot(sidereal.A, exp_terms)
    return template, template_p, template_c


def _random_params(rng: np.random.Generator) -> dict[str, float]:
    return dict(
        ra=rng.uniform(0, 2 * np.pi),
        dec=rng.uniform(-np.pi / 2, np.pi / 2),
        eta=rng.uniform(-1, 1),
        psi=rng.uniform(0, 2 * np.pi),
        lat=rng.uniform(-np.pi / 2, np.pi / 2),
        lng=rng.uniform(-np.pi, np.pi),
        az=rng.uniform(0, 2 * np.pi),
    )


def _pbh_tau(t_offset: np.ndarray, f0: float, chirp_mass: float) -> tuple[np.ndarray, float]:
    beta = PBH_CONST * f0 ** (8 / 3) * chirp_mass ** (5 / 3)
    tau = -(3 / (5 * beta)) * (1 - 8 / 3 * beta * t_offset) ** (5 / 8)
    tau -= tau[0]
    return tau, beta


def _midpoint_pbh_frequency(t_last: float, chirp_mass: float, carrier_bin: int) -> float:
    def _tau_span(f0_local: float) -> float:
        beta_local = PBH_CONST * f0_local ** (8 / 3) * chirp_mass ** (5 / 3)
        return (3 / (5 * beta_local)) * (1 - (1 - 8 / 3 * beta_local * t_last) ** (5 / 8))

    f0 = carrier_bin / t_last
    for _ in range(20):
        next_f0 = carrier_bin / _tau_span(f0)
        if np.isclose(next_f0, f0, rtol=0.0, atol=1e-14):
            break
        f0 = next_f0
    return f0


def _create_simulation(
    *,
    number_of_days: float,
    sample_rate_hz: float,
    phase_model: str,
    f0_setting: str,
    rng: np.random.Generator,
) -> SimulationData:
    sidereal = five_vec(**_random_params(rng))
    t_obs = number_of_days * sidereal.side_day
    n_samples = round(sample_rate_hz * t_obs)
    t_offset = np.arange(n_samples, dtype=float) / sample_rate_hz
    t_last = t_offset[-1]
    carrier_bin = 20000

    if phase_model == "pbh":
        chirp_mass = 10 ** rng.uniform(-3, -1) * 2e30
        if f0_setting == "midpoint":
            f0 = _midpoint_pbh_frequency(t_last, chirp_mass, carrier_bin)
        elif f0_setting == "uniform":
            f0 = rng.uniform(0.1, 0.2)
        else:
            raise ValueError(f"Unknown f0_setting: {f0_setting}")
        tau, _ = _pbh_tau(t_offset, f0, chirp_mass)
    elif phase_model == "fdot":
        fdot = 1e-9
        if f0_setting == "midpoint":
            f0 = (carrier_bin - 0.5 * fdot * t_last**2) / t_last
        elif f0_setting == "uniform":
            f0 = rng.uniform(0.1, 0.2)
        else:
            raise ValueError(f"Unknown f0_setting: {f0_setting}")
        tau = t_offset + 0.5 * (fdot / f0) * t_offset**2
    else:
        raise ValueError(f"Unknown phase_model: {phase_model}")

    if not (0 < f0 < sample_rate_hz / 2):
        raise ValueError(
            f"Injected carrier f0={f0:.6f} Hz is outside the Nyquist range for "
            f"sample_rate_hz={sample_rate_hz:.6f} Hz."
        )

    ref_time = Time("2019-04-10T12:34:56.000")
    t = Time(ref_time.gps + t_offset, format="gps", scale="utc")
    omega0 = 2 * np.pi * f0

    sidereal.compute_H()
    sidereal.compute_A(sidereal.gmst(t.mjd))
    sidereal.compute_5vec()

    modulation, modulation_p, modulation_c = _sidereal_templates(sidereal, t)
    target_h = rng.uniform(1, 5) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    carrier = np.exp(1j * omega0 * tau)

    template = modulation * carrier
    template_p = modulation_p * carrier
    template_c = modulation_c * carrier
    signal = target_h * template

    return SimulationData(
        signal=signal,
        tau=tau,
        omega0=omega0,
        t=t,
        target_h=target_h,
        template=template,
        template_p=template_p,
        template_c=template_c,
        n_samples=n_samples,
        number_of_days=number_of_days,
    )


def resampling_pipeline_outputs(sim: SimulationData) -> FilterOutputs:
    data_x = _resample_and_extract_5vec(sim.signal, sim.tau, sim.omega0)
    template_x = _resample_and_extract_5vec(sim.template, sim.tau, sim.omega0)
    template_xp = _resample_and_extract_5vec(sim.template_p, sim.tau, sim.omega0)
    template_xc = _resample_and_extract_5vec(sim.template_c, sim.tau, sim.omega0)

    h_est = _estimator(data_x, template_x)
    hp_est, hc_est = _joint_estimator(data_x, template_xp, template_xc)
    return FilterOutputs(h_est=h_est, hp_est=hp_est, hc_est=hc_est)


def frequency_domain_outputs(sim: SimulationData) -> FilterOutputs:
    signal_fft = np.fft.fft(sim.signal)
    template_fft = np.fft.fft(sim.template)
    template_p_fft = np.fft.fft(sim.template_p)
    template_c_fft = np.fft.fft(sim.template_c)

    h_est = _estimator(signal_fft, template_fft)
    hp_est, hc_est = _joint_estimator(signal_fft, template_p_fft, template_c_fft)
    return FilterOutputs(h_est=h_est, hp_est=hp_est, hc_est=hc_est)


def _time_call(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return out, elapsed


def compare_single_case(
    *,
    number_of_days: float,
    sample_rate_hz: float,
    phase_model: str,
    f0_setting: str,
    seed: int,
) -> ComparisonRow:
    rng = np.random.default_rng(seed)
    sim = _create_simulation(
        number_of_days=number_of_days,
        sample_rate_hz=sample_rate_hz,
        phase_model=phase_model,
        f0_setting=f0_setting,
        rng=rng,
    )

    resampling_out, resampling_time = _time_call(resampling_pipeline_outputs, sim)
    matched_out, matched_time = _time_call(frequency_domain_outputs, sim)

    return ComparisonRow(
        number_of_days=number_of_days,
        n_samples=sim.n_samples,
        h_rel_err=_relative_error(resampling_out.h_est, matched_out.h_est),
        hp_rel_err=_relative_error(resampling_out.hp_est, matched_out.hp_est),
        hc_rel_err=_relative_error(resampling_out.hc_est, matched_out.hc_est),
        h_phase_err=abs(_wrapped_phase_diff(np.angle(resampling_out.h_est), np.angle(matched_out.h_est))),
        hp_phase_err=abs(_wrapped_phase_diff(np.angle(resampling_out.hp_est), np.angle(matched_out.hp_est))),
        hc_phase_err=abs(_wrapped_phase_diff(np.angle(resampling_out.hc_est), np.angle(matched_out.hc_est))),
        resampling_time_s=resampling_time,
        matched_filter_time_s=matched_time,
    )


def _benchmark_method(fn, sim: SimulationData, repeats: int) -> tuple[FilterOutputs, float]:
    timings = []
    last_out = None
    for _ in range(repeats):
        last_out, elapsed = _time_call(fn, sim)
        timings.append(elapsed)
    return last_out, float(np.median(timings))


def benchmark_lengths(
    *,
    days_list: list[float],
    sample_rate_hz: float,
    phase_model: str,
    f0_setting: str,
    repeats: int,
    seed: int,
) -> list[ComparisonRow]:
    rows = []
    for idx, number_of_days in enumerate(days_list):
        rng = np.random.default_rng(seed + idx)
        sim = _create_simulation(
            number_of_days=number_of_days,
            sample_rate_hz=sample_rate_hz,
            phase_model=phase_model,
            f0_setting=f0_setting,
            rng=rng,
        )

        _ = resampling_pipeline_outputs(sim)
        _ = frequency_domain_outputs(sim)

        resampling_out, resampling_time = _benchmark_method(resampling_pipeline_outputs, sim, repeats)
        matched_out, matched_time = _benchmark_method(frequency_domain_outputs, sim, repeats)

        rows.append(
            ComparisonRow(
                number_of_days=number_of_days,
                n_samples=sim.n_samples,
                h_rel_err=_relative_error(resampling_out.h_est, matched_out.h_est),
                hp_rel_err=_relative_error(resampling_out.hp_est, matched_out.hp_est),
                hc_rel_err=_relative_error(resampling_out.hc_est, matched_out.hc_est),
                h_phase_err=abs(
                    _wrapped_phase_diff(np.angle(resampling_out.h_est), np.angle(matched_out.h_est))
                ),
                hp_phase_err=abs(
                    _wrapped_phase_diff(np.angle(resampling_out.hp_est), np.angle(matched_out.hp_est))
                ),
                hc_phase_err=abs(
                    _wrapped_phase_diff(np.angle(resampling_out.hc_est), np.angle(matched_out.hc_est))
                ),
                resampling_time_s=resampling_time,
                matched_filter_time_s=matched_time,
            )
        )
    return rows


def assert_numerical_agreement(rows: list[ComparisonRow], *, atol: float = 1e-9) -> None:
    max_err = max(
        max(row.h_rel_err, row.hp_rel_err, row.hc_rel_err, row.h_phase_err, row.hp_phase_err, row.hc_phase_err)
        for row in rows
    )
    if max_err > atol:
        raise AssertionError(
            "Resampling and frequency-domain matched filtering disagree beyond tolerance. "
            f"Maximum observed error was {max_err:.3e}, tolerance is {atol:.3e}."
        )


def _print_rows(rows: list[ComparisonRow]) -> None:
    header = (
        "days    samples      h_rel_err    hp_rel_err   hc_rel_err   "
        "h_phase_err  hp_phase_err hc_phase_err  resamp_ms  matched_ms  speedup"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        speedup = row.matched_filter_time_s / row.resampling_time_s
        print(
            f"{row.number_of_days:4.1f}  "
            f"{row.n_samples:9d}  "
            f"{row.h_rel_err:11.3e}  "
            f"{row.hp_rel_err:11.3e}  "
            f"{row.hc_rel_err:11.3e}  "
            f"{row.h_phase_err:11.3e}  "
            f"{row.hp_phase_err:11.3e}  "
            f"{row.hc_phase_err:11.3e}  "
            f"{1e3 * row.resampling_time_s:9.2f}  "
            f"{1e3 * row.matched_filter_time_s:10.2f}  "
            f"{speedup:7.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-model",
        choices=["pbh", "fdot"],
        default="pbh",
        help="Phase evolution used to build the synthetic signal.",
    )
    parser.add_argument(
        "--f0-setting",
        choices=["midpoint", "uniform"],
        default="midpoint",
        help="How the carrier frequency is injected relative to Fourier bins.",
    )
    parser.add_argument(
        "--days",
        nargs="+",
        type=float,
        default=[1, 2, 4, 8],
        help="Observation lengths, in sidereal days, for the runtime benchmark.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=1.0,
        help="Uniform input sampling rate in Hz.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timing repeats per benchmark point.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for the random source geometry and signal parameters.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Maximum allowed numerical mismatch in the agreement check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = benchmark_lengths(
        days_list=args.days,
        sample_rate_hz=args.sample_rate_hz,
        phase_model=args.phase_model,
        f0_setting=args.f0_setting,
        repeats=args.repeats,
        seed=args.seed,
    )

    assert_numerical_agreement(rows, atol=args.tolerance)
    print(
        f"Agreement check passed for phase_model={args.phase_model!r}, "
        f"f0_setting={args.f0_setting!r}, tolerance={args.tolerance:.1e}"
    )
    _print_rows(rows)
    print("speedup = matched_filter_time / resampling_time")


if __name__ == "__main__":
    main()
