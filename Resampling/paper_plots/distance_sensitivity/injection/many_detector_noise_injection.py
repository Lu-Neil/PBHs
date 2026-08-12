"""Recover randomized injections in successive H1 O4a science windows."""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

PAPER_PLOTS_DIR = Path(__file__).resolve().parents[2]
if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from distance_sensitivity.injection import (  # noqa: E402
    detector_noise_injection as single,
)
from distance_sensitivity.injection import nonoise_injection as base  # noqa: E402


NUM_INJ = 100
RANDOM_SEED = 12345
INJECTION_DISTANCES_PC = np.geomspace(3e5, 3e8, 20)
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "detector_noise_injections.h5"


def science_blocks(duration, count, DataQualityFlag, to_gps):
    """Return successive non-overlapping science-mode blocks in O4a."""
    flag = DataQualityFlag.fetch_open_data(
        single.DATA_QUALITY_FLAG,
        float(to_gps(single.O4A_START)),
        float(to_gps(single.O4A_END)),
    )
    blocks = []
    for segment in flag.active:
        start, end = map(float, segment)
        while start + duration <= end and len(blocks) < count:
            blocks.append((start, start + duration))
            start += duration
        if len(blocks) == count:
            return blocks
    raise RuntimeError(f"O4a contains only {len(blocks)} complete science blocks")


def random_source(rng):
    """Draw an isotropic sky position and uniform polarization parameters."""
    ra = rng.uniform(0, 2 * np.pi)
    dec = np.arcsin(rng.uniform(-1, 1))
    psi = rng.uniform(0, np.pi)
    eta = rng.uniform(-1, 1)
    return ra, dec, psi, eta


def resample_carriers(noise_history, analyses, frequency_track, args):
    """Use the single-injection rolling-background resampling implementation."""
    return single.resample_carriers(
        noise_history, analyses, frequency_track, args
    )


def source_weights(args, frequency_track, effective_psds, gps_start, source):
    """Return the normalized Eq. (19) weights for one source."""
    ra, dec, psi, eta = source
    starts, chunk_samples = single.analysis_chunk_starts(
        frequency_track.size, args
    )
    t = np.arange(frequency_track.size) / args.sample_rate
    window = np.hanning(chunk_samples)
    window_power = np.mean(window**2)
    response = single.detector_response(
        single.DETECTOR,
        gps_start,
        t,
        eta,
        psi,
        ra=ra,
        dec=dec,
    )
    weights = []

    for start, psd in zip(starts, effective_psds):
        stop = start + chunk_samples
        frequency = float(frequency_track[start])
        if not args.f_min <= frequency <= args.f_max:
            continue

        amplitude = base.h0_amplitude(
            base.PARSEC_M,
            frequency_track[start:stop],
            args.mchirp,
        )
        coherent_amplitude = np.mean(
            window * amplitude * response[start:stop]
        )
        weights.append(
            np.abs(coherent_amplitude) ** 2 / (window_power * psd)
        )

    weights = np.asarray(weights)
    return weights / np.linalg.norm(weights)


def recover_statistics(
    background,
    noise,
    gps_start,
    source,
    args,
    frequency_track,
    duration,
):
    """Recover this noise realization at every configured injection distance."""
    ra, dec, psi, eta = source
    reference_distance_pc = INJECTION_DISTANCES_PC[0]
    _, signal, *_ = single.make_detector_injection(
        args,
        reference_distance_pc * base.PARSEC_M,
        duration,
        gps_start,
        ra=ra,
        dec=dec,
        psi=psi,
        eta=eta,
    )
    noise_history = np.concatenate((background.ravel(), noise))
    carriers = resample_carriers(
        noise_history, noise, frequency_track, args
    )
    background_carriers = carriers[:-1]
    noise_carrier = carriers[-1]
    injected_carrier = resample_carriers(
        noise_history, noise + signal, frequency_track, args
    )[-1]
    signal_carrier = injected_carrier - noise_carrier

    effective_psds, background_powers, means, standard_deviations = (
        single.background_statistics(
            background_carriers, args.chunk_duration
        )
    )
    weights = source_weights(
        args, frequency_track, effective_psds, gps_start, source
    )

    distance_scaling = reference_distance_pc / INJECTION_DISTANCES_PC
    scan_carriers = noise_carrier + distance_scaling[:, None] * signal_carrier
    powers = single.normalized_powers(
        scan_carriers, effective_psds, args.chunk_duration
    )
    statistics = ((powers - means) / standard_deviations) @ weights
    return statistics, effective_psds, means, standard_deviations


def fetch_block(index, args, TimeSeries, duration):
    """Read one cached block through the single-injection cache reader."""
    data, path = single.read_cached_strain(
        args, duration, TimeSeries, inj_num=index
    )
    return np.asarray(data.value), float(data.t0.value), path


def main():
    args = single.analysis_config()
    _, TimeSeries, _ = single.import_gwpy()
    frequency_model = base.semicoherent_frequency_model(args)
    duration, _, n_segments = base.analysis_span(args, frequency_model)
    chunk_samples, _ = base.chunk_config(args)
    background_duration = single.BACKGROUND_SPECTRA * args.chunk_duration
    block_duration = background_duration + duration
    frequency_track = np.asarray(
        frequency_model.frequency(base.sample_times(args.sample_rate, duration))
    )
    rng = np.random.default_rng(RANDOM_SEED)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUTPUT_PATH, "w") as output, ThreadPoolExecutor(
        max_workers=1
    ) as loader:
        output.create_dataset("distances_pc", data=INJECTION_DISTANCES_PC)
        output.create_dataset(
            "segment_frequencies_hz", data=frequency_track[::chunk_samples]
        )
        recovered = output.create_dataset(
            "recovered_statistics",
            (NUM_INJ, INJECTION_DISTANCES_PC.size),
            dtype="f8",
        )
        effective_psds = output.create_dataset(
            "effective_psds", (NUM_INJ, n_segments), dtype="f8"
        )
        background_means = output.create_dataset(
            "background_power_means", (NUM_INJ, n_segments), dtype="f8"
        )
        background_stds = output.create_dataset(
            "background_power_standard_deviations",
            (NUM_INJ, n_segments),
            dtype="f8",
        )
        start_times = output.create_dataset("start_gps", (NUM_INJ,))
        end_times = output.create_dataset("end_gps", (NUM_INJ,))
        strain_files = output.create_dataset(
            "strain_files",
            (NUM_INJ,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        ra_values = output.create_dataset("ra", (NUM_INJ,))
        dec_values = output.create_dataset("dec", (NUM_INJ,))
        psi_values = output.create_dataset("psi", (NUM_INJ,))
        eta_values = output.create_dataset("eta", (NUM_INJ,))

        output.attrs["detector"] = single.DETECTOR
        output.attrs["data_quality_flag"] = single.DATA_QUALITY_FLAG
        output.attrs["random_seed"] = RANDOM_SEED
        output.attrs["sample_rate_hz"] = args.sample_rate
        output.attrs["initial_frequency_hz"] = args.f0
        output.attrs["maximum_frequency_hz"] = args.f_max
        output.attrs["chirp_mass_msun"] = args.mchirp
        output.attrs["injection_duration_s"] = duration
        output.attrs["coherent_duration_s"] = args.chunk_duration
        output.attrs["coherent_segments"] = n_segments
        output.attrs["background_segments"] = single.BACKGROUND_SPECTRA
        output.attrs["coherent_pn_mismatch"] = single.COHERENT_PN_MISMATCH
        output.attrs["semicoherent_bank_mismatch"] = (
            single.SEMICOHERENT_BANK_MISMATCH
        )
        output.attrs["psd_convention"] = "one-sided"
        output.attrs["time_units"] = "GPS seconds"
        output.attrs["angle_units"] = "radians"

        pending_data = loader.submit(
            fetch_block, 0, args, TimeSeries, block_duration
        )
        for index in range(NUM_INJ):
            wait_start = time.perf_counter()
            strain, block_start, strain_path = pending_data.result()
            read_wait = time.perf_counter() - wait_start
            if index + 1 < NUM_INJ:
                pending_data = loader.submit(
                    fetch_block,
                    index + 1,
                    args,
                    TimeSeries,
                    block_duration,
                )

            background_samples = single.BACKGROUND_SPECTRA * chunk_samples
            background = strain[:background_samples].reshape(
                single.BACKGROUND_SPECTRA, chunk_samples
            )
            noise = strain[background_samples:]
            injection_start = block_start + background_duration
            injection_end = injection_start + duration
            source = random_source(rng)

            analysis_start = time.perf_counter()
            (
                recovered[index],
                effective_psds[index],
                background_means[index],
                background_stds[index],
            ) = recover_statistics(
                background,
                noise,
                injection_start,
                source,
                args,
                frequency_track,
                duration,
            )
            analysis_time = time.perf_counter() - analysis_start
            ra, dec, psi, eta = source
            start_times[index] = injection_start
            end_times[index] = injection_end
            strain_files[index] = strain_path.name
            ra_values[index] = ra
            dec_values[index] = dec
            psi_values[index] = psi
            eta_values[index] = eta
            output.attrs["completed_injections"] = index + 1
            output.flush()
            print(
                f"{index + 1:3d}/{NUM_INJ}: GPS "
                f"{injection_start:.0f}-{injection_end:.0f}, "
                f"read wait={read_wait:.1f} s, "
                f"analysis={analysis_time:.1f} s"
            )

    print(f"Saved {NUM_INJ} injections to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
