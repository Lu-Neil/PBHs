from ..five_vec import five_vec
from ..resampler import Resampler
import numpy as np
from astropy.time import Time


def _estimator(X, A_template):
    return np.dot(X, np.conj(A_template)) / np.sum(np.abs(A_template) ** 2)


def _wrapped_phase_diff(phi_a, phi_b):
    return np.angle(np.exp(1j * (phi_a - phi_b)))


def _resample_and_extract_5vec(signal, tau, omega0):
    resampler = Resampler()
    resampler.timeseries = signal
    resampler.resampled_time = tau
    resampler.nufft()
    return resampler.extract_5vec(omega0), resampler


def _random_params():
    params = dict(
        ra=np.random.uniform(0, 2 * np.pi),
        dec=np.random.uniform(-np.pi / 2, np.pi / 2),
        eta=np.random.uniform(-1, 1),
        psi=np.random.uniform(0, 2 * np.pi),
        lat=np.random.uniform(-np.pi / 2, np.pi / 2),
        lng=np.random.uniform(-np.pi, np.pi),
        az=np.random.uniform(0, 2 * np.pi),
    )
    return params


def _time_domain_5vec(sidereal, t, tau):
    sidereal_t = sidereal.gmst(t.mjd)
    sidereal_t -= sidereal_t[0]
    exp_terms = np.exp(1j * (np.arange(5) - 2)[:, np.newaxis] * sidereal_t)
    template_p = np.dot(sidereal.A_p, exp_terms)
    template_c = np.dot(sidereal.A_c, exp_terms)
    template_comb = np.dot(sidereal.A, exp_terms)
    template_Xp, _ = _resample_and_extract_5vec(template_p, tau, 0)
    template_Xc, _ = _resample_and_extract_5vec(template_c, tau, 0)
    template_X, _ = _resample_and_extract_5vec(template_comb, tau, 0)
    return template_X, template_Xp, template_Xc


def _Dirchlet_corrections(resampler, omega0, tau, h0, gamma):
    idx0 = np.abs(resampler.freqs - omega0).argmin()
    delta_omega = omega0 - resampler.freqs[idx0]
    tau_span = tau[-1] - tau[0]
    bin_amp_loss = np.sinc(delta_omega * tau_span / (2 * np.pi))
    bin_dephasing = delta_omega * 0.5 * tau_span
    bin_factor = bin_amp_loss * np.exp(1j * bin_dephasing)

    if np.isscalar(h0):
        target = h0 * np.exp(1j * gamma)
    else:
        target = np.mean(h0) * np.exp(1j * gamma)
    expected_h = target * bin_factor
    return expected_h, bin_factor, delta_omega


def _create_fDot_signal(f0_setting="midpoint"):
    params = _random_params()
    sidereal = five_vec(**params)

    number_of_days = 2  # keep integer days for clean 1/day sideband spacing
    T_obs = number_of_days * sidereal.side_day
    f_signal = 1
    n_samples = round(f_signal * T_obs)
    t_offset = np.linspace(0, T_obs, n_samples, endpoint=False, dtype=float)
    t_last = t_offset[-1]

    # Pick f0 so the demodulated carrier lands on a NUFFT bin in tau.
    h0 = np.random.uniform(1, 5)
    gamma = np.random.uniform(0, 2 * np.pi)
    fdot = 1e-9  # cycles / s^2, intentionally small
    carrier_bin = 20000
    if f0_setting == "midpoint":
        f0 = (carrier_bin - 0.5 * fdot * t_last**2) / t_last
    elif f0_setting == "uniform":
        f0 = np.random.uniform(0.1, 0.2)
    else:
        raise Exception("f0_setting error")
    omega0 = 2 * np.pi * f0
    assert f0 < f_signal / 2

    ref_time = Time("2019-04-10T12:34:56.000")
    t_gps = ref_time.gps + t_offset
    t = Time(t_gps, format="gps", scale="utc")

    sidereal.compute_H()
    sidereal.compute_A(sidereal.gmst(t.mjd))
    sidereal.compute_5vec()
    amp_modulation = h0 * sidereal.amp_modulation

    phase = 2 * np.pi * (f0 * t_offset + 0.5 * fdot * t_offset**2) + gamma
    signal = amp_modulation * np.exp(1j * phase)
    tau = t_offset + 0.5 * (fdot / f0) * t_offset**2
    return signal, tau, omega0, sidereal, h0, gamma, t


def _check_fDot_signal(err, f0_setting):
    signal, tau, omega0, sidereal, h0, gamma, t = _create_fDot_signal(f0_setting=f0_setting)
    data_X, resampler = _resample_and_extract_5vec(signal, tau, omega0)

    # Second-stage demodulation: remove sidereal modulation with 5-vector templates.
    template_X, template_Xp, template_Xc = _time_domain_5vec(sidereal, t, tau)
    h_est = _estimator(data_X, template_X)
    hp_est = _estimator(data_X, template_Xp)
    hc_est = _estimator(data_X, template_Xc)
    hp_ratio = hp_est / h_est
    hc_ratio = hc_est / h_est

    # Correct expected values for finite FFT-bin mismatch (Dirichlet response).
    expected_h, _, _ = _Dirchlet_corrections(resampler, omega0, tau, h0, gamma)

    assert np.isclose(np.abs(h_est), np.abs(expected_h), rtol=err, atol=0.0)
    assert np.isclose(
        _wrapped_phase_diff(np.angle(h_est), np.angle(expected_h)),
        0.0,
        atol=err,
    )

    assert np.isclose(np.abs(hp_ratio), np.abs(sidereal.H_p), rtol=err, atol=0.0)
    assert np.isclose(np.abs(hc_ratio), np.abs(sidereal.H_c), rtol=err, atol=0.0)
    assert np.isclose(
        _wrapped_phase_diff(np.angle(hp_ratio), np.angle(sidereal.H_p)),
        0.0,
        atol=err,
    )
    assert np.isclose(
        _wrapped_phase_diff(np.angle(hc_ratio), np.angle(sidereal.H_c)),
        0.0,
        atol=err,
    )


def test_midpoint_fDot_signal():
    """Signals with a fdot term signal where the f0 is injected at the bin centre."""
    _check_fDot_signal(err=1e-3, f0_setting="midpoint")


def test_uniform_fDot_signal():
    """Signals with a fdot term signal where the f0 is injected uniformly (instead of at bin centres).
    Much larger errors because doesn't scale exactly according
    to the Dirchlet kernel. Not sure why but within 20% in over 95% of simulations.
    Verified in ..PBH-5vec.ipynb"""
    _check_fDot_signal(err=2e-1, f0_setting="uniform")


# --------------- PBH signal -----------------
from importlib import import_module

_utils = import_module("..fiveVec_resampler_utils", package=__package__)
check_PBH_signal = _utils.check_PBH_signal


def test_midpoint_PBH_signal():
    """PBH chirp with midpoint-injected f0 and a contiguous 15% data gap."""
    check_PBH_signal(err=1e-3, f0_setting="midpoint", gap_fraction=0.0)


def test_uniform_PBH_signal():
    """PBH chirp with uniformly injected f0 and a contiguous 15% data gap."""
    check_PBH_signal(err=2e-1, f0_setting="uniform", gap_fraction=0.0)
