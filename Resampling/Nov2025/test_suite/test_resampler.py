from ..resampler import Resampler
import numpy as np


def _wrapped_phase_diff(phi_a, phi_b):
    return np.angle(np.exp(1j * (phi_a - phi_b)))


def test_nufft_matches_fft_for_monochromatic_signal():
    """Check amplitude and phase at the monochromatic peak against np.fft."""
    n_samples = 2**14
    t_end = 2 * np.pi
    time = np.linspace(0, t_end, n_samples, endpoint=False, dtype=float)
    amp = np.random.uniform(1, 5)
    phase0 = np.random.uniform(0, 2 * np.pi)
    k0 = np.random.uniform(1, 5)
    omega0 = 2 * np.pi * k0

    signal = amp * np.exp(1j * (omega0 * time + phase0))

    fft_weights = np.fft.fftshift(np.fft.fft(signal)) / n_samples
    fft_freqs = np.fft.fftshift(np.fft.fftfreq(n_samples, np.diff(time)[0]))
    fft_idx = abs(fft_freqs - k0).argmin()
    fft_peak = fft_weights[fft_idx]

    resampler = Resampler()
    resampler.timeseries = signal
    resampler.resampled_time = time
    resampler.nufft()

    nufft_weights = resampler.weights_normalized
    nufft_idx = np.abs(resampler.freqs - omega0).argmin()
    nufft_peak = nufft_weights[nufft_idx]

    assert np.isclose(np.abs(nufft_peak), np.abs(fft_peak), rtol=5e-2, atol=0.0)
    assert np.isclose(
        _wrapped_phase_diff(np.angle(nufft_peak), np.angle(fft_peak)),
        0.0,
        atol=5e-2,
    )
