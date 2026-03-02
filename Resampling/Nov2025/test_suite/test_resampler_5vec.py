from ..five_vec import five_vec
from ..resampler import Resampler
import numpy as np
from astropy.time import Time


def _estimator(X, A_template):
    return np.dot(X, np.conj(A_template)) / np.sum(np.abs(A_template) ** 2)


def _wrapped_phase_diff(phi_a, phi_b):
    return np.angle(np.exp(1j * (phi_a - phi_b)))


def test_resampler_then_5vec_demodulation():
    """Inject fDot+sidereal modulation, then demodulate with resampler followed by 5-vector."""
    h0 = np.random.uniform(1, 5)
    gamma = 0.7
    fdot = 1e-9  # cycles / s^2, intentionally small

    params = dict(
        ra=1.1,
        dec=0.3,
        eta=0.2,
        psi=0.4,
        lat=0.8,
        lng=0.5,
        az=1.2,
    )
    sidereal = five_vec(**params)

    number_of_days = 2  # keep integer days for clean 1/day sideband spacing
    T_obs = number_of_days * sidereal.side_day
    f0 = 10000 / sidereal.side_day
    omega0 = 2 * np.pi * f0
    f_sample = 8 * f0
    n_samples = round(f_sample * T_obs)

    ref_time = Time("2019-04-10T12:34:56.000")
    t_gps = ref_time.gps + np.arange(n_samples) / f_sample
    t = Time(t_gps, format="gps", scale="utc")
    t_offset = t.gps - t.gps[0]

    sidereal.compute_H()
    sidereal.compute_A(sidereal.gmst(t.mjd))
    sidereal.compute_5vec()
    amp_modulation = h0 * sidereal.amp_modulation

    phase = 2 * np.pi * (f0 * t_offset + 0.5 * fdot * t_offset**2) + gamma
    signal = amp_modulation * np.exp(1j * phase)

    # First-stage demodulation: remove fDot by time reparameterization.
    tau = t_offset + 0.5 * (fdot / f0) * t_offset**2
    resampler = Resampler()
    resampler.timeseries = signal
    resampler.resampled_time = tau
    resampler.nufft()

    # Second-stage demodulation: remove sidereal modulation with 5-vector templates.
    exp_terms = np.exp(1j * (np.arange(5) - 2)[:, np.newaxis] * (t.mjd - t.mjd[0]))
    template_p = np.dot(sidereal.A_p, exp_terms)
    template_c = np.dot(sidereal.A_c, exp_terms)
    template_comb = np.dot(sidereal.A, exp_terms)

    X = resampler.extract_5vec(omega0)
    h_est = _estimator(X, sidereal.A)
    hp_est = _estimator(X, sidereal.A_p)
    hc_est = _estimator(X, sidereal.A_c)

    # Correct expected values for finite FFT-bin mismatch (Dirichlet response).
    idx0 = np.abs(resampler.freqs - omega0).argmin()
    delta_omega = omega0 - resampler.freqs[idx0]
    tau_span = tau[-1] - tau[0]
    bin_amp_loss = np.sinc(delta_omega * tau_span / (2 * np.pi))
    bin_dephasing = delta_omega * 0.5 * tau_span
    bin_factor = bin_amp_loss * np.exp(1j * bin_dephasing)

    target = h0 * np.exp(1j * gamma)
    expected_h = target * bin_factor
    expected_hp = sidereal.H_p * target * bin_factor
    expected_hc = sidereal.H_c * target * bin_factor

    assert np.isclose(np.abs(h_est), np.abs(expected_h), rtol=1e-2, atol=0.0)
    assert np.isclose(np.abs(hp_est), np.abs(expected_hp), rtol=1e-2, atol=0.0)
    assert np.isclose(np.abs(hc_est), np.abs(expected_hc), rtol=1e-2, atol=0.0)

    assert np.isclose(
        _wrapped_phase_diff(np.angle(h_est), np.angle(expected_h)),
        0.0,
        atol=1e-2,
    )
    assert np.isclose(
        _wrapped_phase_diff(np.angle(hp_est), np.angle(expected_hp)),
        0.0,
        atol=1e-2,
    )
    assert np.isclose(
        _wrapped_phase_diff(np.angle(hc_est), np.angle(expected_hc)),
        0.0,
        atol=1e-2,
    )
