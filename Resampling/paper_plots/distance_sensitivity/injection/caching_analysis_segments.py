"""Cache O4a science-mode analysis segments as 512 Hz GWpy time series.

The segment selection and segment duration are the same as those used by
``many_detector_noise_injection.py``.  Each output file contains the 15
background chunks followed by the strain used for one injection analysis.

Run with::

    conda run -n PBH python distance_sensitivity/injection/caching_analysis_segments.py
"""

import sys
from pathlib import Path

import numpy as np


PAPER_PLOTS_DIR = Path(__file__).resolve().parents[2]
if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from distance_sensitivity.injection import (  # noqa: E402
    detector_noise_injection as single,
)
from distance_sensitivity.injection import (  # noqa: E402
    many_detector_noise_injection as many,
)
from distance_sensitivity.injection import nonoise_injection as base  # noqa: E402


NUM_SEGMENTS = 300
OUTPUT_DIR = Path(__file__).resolve().parent / "strain_data"
HDF5_DATASET = "strain"


def output_path(start, end, sample_rate):
    """Return a GWOSC-style path encoding this segment's GPS interval."""
    duration = int(round(end - start))
    return OUTPUT_DIR / (
        f"{single.DETECTOR}-{int(round(start))}-{duration}-"
        f"{int(round(sample_rate))}Hz.hdf5"
    )


def cached_series_is_valid(path, TimeSeries, start, required_samples, sample_rate):
    """Check that an existing cache file describes the requested time series."""
    try:
        data = TimeSeries.read(path, format="hdf5", path=HDF5_DATASET)
    except (OSError, ValueError, KeyError):
        return False

    return (
        len(data) == required_samples
        and np.isclose(float(data.t0.value), start)
        and np.isclose(float(data.sample_rate.value), sample_rate)
    )


def cache_segment(bounds, args, TimeSeries, required_samples):
    """Download and atomically save one detrended, resampled strain segment."""
    start, end = bounds
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = output_path(start, end, args.sample_rate)
    if path.exists():
        if cached_series_is_valid(
            path, TimeSeries, start, required_samples, args.sample_rate
        ):
            return path, False
        raise RuntimeError(
            f"Existing cache file has incompatible metadata: {path}"
        )

    data = single.fetch_open_strain(args, start, end, TimeSeries)
    if len(data) < required_samples:
        raise RuntimeError(
            f"Fetched only {len(data)} of {required_samples} samples for "
            f"GPS {start:g}-{end:g}"
        )
    data = data[:required_samples]

    temporary_path = path.with_suffix(path.suffix + ".part")
    temporary_path.unlink(missing_ok=True)
    data.write(
        temporary_path,
        format="hdf5",
        path=HDF5_DATASET,
        compression="gzip",
    )
    temporary_path.replace(path)
    return path, True


def main():
    args = single.analysis_config()
    if not np.isclose(args.sample_rate, 512.0):
        raise ValueError(
            f"Expected a 512 Hz analysis sample rate, got {args.sample_rate:g} Hz"
        )

    DataQualityFlag, TimeSeries, to_gps = single.import_gwpy()
    frequency_model = base.semicoherent_frequency_model(args)
    injection_duration, _, _ = base.analysis_span(args, frequency_model)
    chunk_samples, _ = base.chunk_config(args)
    background_duration = single.BACKGROUND_SPECTRA * args.chunk_duration
    block_duration = background_duration + injection_duration
    required_samples = single.BACKGROUND_SPECTRA * chunk_samples + round(
        injection_duration * args.sample_rate
    )
    blocks = many.science_blocks(
        block_duration, NUM_SEGMENTS, DataQualityFlag, to_gps
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, bounds in enumerate(blocks, start=1):
        path, downloaded = cache_segment(
            bounds, args, TimeSeries, required_samples
        )
        action = "saved" if downloaded else "cached"
        print(f"{index:3d}/{NUM_SEGMENTS}: {action} {path.name}", flush=True)

    print(f"Cached {NUM_SEGMENTS} analysis segments in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
