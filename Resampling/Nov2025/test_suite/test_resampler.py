from ..resampler import Resampler
import numpy as np


def _wrapped_phase_diff(phi_a, phi_b):
    return np.angle(np.exp(1j * (phi_a - phi_b)))


def _expected_nearest_bin(signal, tau, delta_omega):
    """Return the exact nearest-bin response for a pure mode on nonuniform tau."""
    tau_offset = tau - tau[0]
    return signal[0] * np.mean(np.exp(1j * delta_omega * tau_offset))


def test_monochromatic_signal():
    """Check amplitude and phase at the monochromatic peak against np.fft."""
    # Create signal
    n_samples = 2**14
    t_end = 2 * np.pi
    time = np.linspace(0, t_end, n_samples, endpoint=False, dtype=float)
    amp = np.random.uniform(1, 5)
    phase0 = np.random.uniform(0, 2 * np.pi)
    f0 = np.random.uniform(1, 5)
    omega0 = 2 * np.pi * f0
    signal = amp * np.exp(1j * (omega0 * time + phase0))

    # Compute FFT
    fft_weights = np.fft.fftshift(np.fft.fft(signal)) / n_samples
    fft_freqs = np.fft.fftshift(np.fft.fftfreq(n_samples, np.diff(time)[0]))
    fft_idx = abs(fft_freqs - f0).argmin()
    fft_peak = fft_weights[fft_idx]

    # Compute NUFFT
    resampler = Resampler()
    resampler.timeseries = signal
    resampler.resampled_time = time
    resampler.nufft()

    nufft_weights = resampler.weights_normalized
    nufft_idx = np.abs(resampler.freqs - omega0).argmin()
    nufft_peak = nufft_weights[nufft_idx]

    # The exact nearest-bin response reduces to the Dirichlet kernel for
    # uniform tau, so use the same exact-kernel helper as the chirping tests.
    delta_omega = omega0 - resampler.freqs[nufft_idx]
    expected = _expected_nearest_bin(signal, time, delta_omega)

    # Check NUFFT and FFT consistent
    assert np.isclose(np.abs(nufft_peak), np.abs(fft_peak), rtol=1e-2, atol=0.0)
    assert np.isclose(
        _wrapped_phase_diff(np.angle(nufft_peak), np.angle(fft_peak)),
        0.0,
        atol=1e-2,
    )

    # Check NUFFT consistent with Dirchlet kernel
    assert np.isclose(
        _wrapped_phase_diff(np.angle(nufft_peak), np.angle(expected)),
        0.0,
        atol=5e-3,
    )
    assert np.isclose(
        np.abs(nufft_peak),
        np.abs(expected),
        0.0,
        atol=1e-4,
    )


def test_fDot_signal():
    """Inject a small fdot term, absorb it in tau, and recover injected complex amplitude."""
    # Create signal
    n_samples = 2**14
    t_end = 2 * np.pi
    time = np.linspace(0, t_end, n_samples, endpoint=False, dtype=float)
    amp = np.random.uniform(1, 5)
    phase0 = np.random.uniform(0, 2 * np.pi)
    f0 = np.random.uniform(1, 5)
    omega0 = 2 * np.pi * f0
    fdot = 1e-2  # cycles / time^2 (small spin-down)
    phase = 2 * np.pi * (f0 * time + 0.5 * fdot * time**2) + phase0
    signal = amp * np.exp(1j * phase)
    tau = time + 0.5 * (fdot / f0) * time**2

    resampler = Resampler()
    resampler.timeseries = signal
    resampler.resampled_time = tau
    resampler.nufft()

    nufft_weights = resampler.weights_normalized
    nufft_idx = np.abs(resampler.freqs - omega0).argmin()
    recovered = nufft_weights[nufft_idx]

    # For nonuniform tau, the nearest-bin leakage is set by the exact complex
    # kernel sampled at the tau locations rather than a simple Dirichlet factor.
    delta_omega = omega0 - resampler.freqs[nufft_idx]
    expected = _expected_nearest_bin(signal, tau, delta_omega)

    assert np.isclose(np.abs(recovered), np.abs(expected), rtol=1e-4, atol=0.0)
    assert np.isclose(
        _wrapped_phase_diff(np.angle(recovered), np.angle(expected)),
        0.0,
        atol=5e-3,
    )


def test_PBH_signal():
    """Inject a PBH-inspired chirp phase and recover the monochromatic mode after time remapping."""
    n_samples = 2**14
    t_end = 2 * np.pi
    t_offset = np.linspace(0, t_end, n_samples, endpoint=False, dtype=float)

    c, G, pi = 3e8, 6.67e-11, np.pi
    const = 96 / 5 * pi ** (8 / 3) * (G / c**3) ** (5 / 3)
    Mc = 10 ** np.random.uniform(-3, -1) * 2e30
    f0_injected = np.random.uniform(0.1, 0.2)
    omega0 = 2 * np.pi * f0_injected
    amp = np.random.uniform(1, 5)
    phase0 = np.random.uniform(0, 2 * np.pi)
    beta = const * f0_injected ** (8 / 3) * Mc ** (5 / 3)
    phi = -6 * pi / 5 * f0_injected * (1 - 8 / 3 * beta * t_offset) ** (5 / 8) / beta
    signal_source = amp * np.exp(1j * (phi + phase0))

    # Define tau so that phi = omega0 * tau, then remove the constant offset.
    tau = -(3 / (5 * beta)) * (1 - 8 / 3 * beta * t_offset) ** (5 / 8)
    tau_shifted = tau - tau[0]

    resampler = Resampler()
    resampler.timeseries = signal_source
    resampler.resampled_time = tau_shifted
    resampler.nufft()

    nufft_weights = resampler.weights_normalized
    nufft_idx = np.abs(resampler.freqs - omega0).argmin()
    recovered = nufft_weights[nufft_idx]

    delta_omega = omega0 - resampler.freqs[nufft_idx]
    expected = _expected_nearest_bin(signal_source, tau_shifted, delta_omega)

    assert np.isclose(np.abs(recovered), np.abs(expected), rtol=1e-4, atol=0.0)
    assert np.isclose(
        _wrapped_phase_diff(np.angle(recovered), np.angle(expected)),
        0.0,
        atol=5e-3,
    )
