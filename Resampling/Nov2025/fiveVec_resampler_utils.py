from __future__ import annotations

try:
    from .five_vec import five_vec
    from .resampler import Resampler
except ImportError:
    from five_vec import five_vec
    from resampler import Resampler
import numpy as np
from astropy.time import Time


def _estimator(X, A_template):
    return np.dot(X, np.conj(A_template)) / np.sum(np.abs(A_template) ** 2)


def _joint_estimator(data_X, template_Xp, template_Xc):
    A = np.column_stack([template_Xp, template_Xc])  # shape (5,2), complex
    hp_est, hc_est = np.linalg.lstsq(A, data_X, rcond=None)[0]
    return hp_est, hc_est


def _wrapped_phase_diff(phi_a, phi_b):
    return np.angle(np.exp(1j * (phi_a - phi_b)))


def _resample_and_extract_5vec(signal, tau, omega0):
    resampler = Resampler()
    resampler.timeseries = signal
    resampler.resampled_time = tau
    resampler.nufft()
    return resampler.extract_5vec(omega0), resampler


def _random_params():
    return dict(
        ra=np.random.uniform(0, 2 * np.pi),
        dec=np.random.uniform(-np.pi / 2, np.pi / 2),
        eta=np.random.uniform(-1, 1),
        psi=np.random.uniform(0, 2 * np.pi),
        lat=np.random.uniform(-np.pi / 2, np.pi / 2),
        lng=np.random.uniform(-np.pi, np.pi),
        az=np.random.uniform(0, 2 * np.pi),
    )


def _resolve_tolerances(
    err,
    h_mag_err,
    h_phase_err,
    ratio_mag_err,
    ratio_phase_err,
    detection_stat_err,
):
    if h_mag_err is None:
        h_mag_err = err
    if h_phase_err is None:
        h_phase_err = err
    if ratio_mag_err is None:
        ratio_mag_err = err
    if ratio_phase_err is None:
        ratio_phase_err = err
    if detection_stat_err is None:
        detection_stat_err = ratio_mag_err
    return h_mag_err, h_phase_err, ratio_mag_err, ratio_phase_err, detection_stat_err


def _build_gap_mask(n_samples, gap_fraction=0.15):
    """Create a boolean mask with one contiguous missing-data segment."""
    mask = np.ones(n_samples, dtype=bool)
    gap_size = max(0, int(gap_fraction * n_samples))
    gap_start = np.random.randint(0, n_samples - gap_size + 1)
    gap_end = gap_start + gap_size
    mask[gap_start:gap_end] = False
    return mask, slice(gap_start, gap_end)


def _build_time_domain_templates(sidereal, t, gap_mask=None):
    """Construct the τ-independent time-domain 5-vector templates (comb, plus, cross)."""
    sidereal_t = sidereal.gmst(t.mjd)
    sidereal_t -= sidereal_t[0]
    exp_terms = np.exp(1j * (np.arange(5) - 2)[:, np.newaxis] * sidereal_t)
    template_p = np.dot(sidereal.A_p, exp_terms)
    template_c = np.dot(sidereal.A_c, exp_terms)
    template_comb = np.dot(sidereal.A, exp_terms)

    if gap_mask is not None:
        template_p = np.where(gap_mask, template_p, 0.0)
        template_c = np.where(gap_mask, template_c, 0.0)
        template_comb = np.where(gap_mask, template_comb, 0.0)

    return template_comb, template_p, template_c


def _time_domain_5vec(sidereal, t, tau, gap_mask=None):
    template_comb, template_p, template_c = _build_time_domain_templates(sidereal, t, gap_mask=gap_mask)
    template_Xp, _ = _resample_and_extract_5vec(template_p, tau, 0)
    template_Xc, _ = _resample_and_extract_5vec(template_c, tau, 0)
    template_X, _ = _resample_and_extract_5vec(template_comb, tau, 0)
    return template_X, template_Xp, template_Xc


def _detection_stat(template_Xp, template_Xc, hp_est, hc_est):
    return np.sum(np.abs(template_Xp) ** 4) * abs(hp_est) ** 2 + np.sum(np.abs(template_Xc) ** 4) * abs(hc_est) ** 2


def _sampled_bin_factor(resampler, omega0, tau):
    idx0 = np.abs(resampler.freqs - omega0).argmin()
    delta_omega = omega0 - resampler.freqs[idx0]
    tau_offset = tau - tau[0]
    bin_factor = np.mean(np.exp(1j * delta_omega * tau_offset))
    return bin_factor, delta_omega


def _expected_5vec_bins(signal, tau, resampler, omega0):
    """Return the exact finite-sample NUDFT values at the extracted 5 bins."""
    tau_offset = tau - tau[0]
    side_day = 86164.09053083288
    offsets = 2 * np.pi / side_day * np.arange(-2, 3)
    expected_X = np.empty(5, dtype=complex)
    for i, offset in enumerate(offsets):
        idx = np.abs(resampler.freqs - (omega0 + offset)).argmin()
        bin_freq = resampler.freqs[idx]
        expected_X[i] = np.mean(signal * np.exp(-1j * bin_freq * tau_offset))
    return expected_X


def _expected_carrier_response(resampler, omega0, tau, h0, gamma, gap_mask=None):
    bin_factor, delta_omega = _sampled_bin_factor(resampler, omega0, tau)
    if np.isscalar(h0):
        target = h0 * np.exp(1j * gamma)
    elif gap_mask is None:
        target = np.mean(h0) * np.exp(1j * gamma)
    else:
        target = np.mean(h0[gap_mask]) * np.exp(1j * gamma)

    expected_h = target * bin_factor
    return expected_h, bin_factor, delta_omega


def _Dirichlet_corrections(resampler, omega0, tau, h0, gamma, gap_mask=None):
    """Backward-compatible alias for the sampled-bin carrier response helper."""
    return _expected_carrier_response(resampler, omega0, tau, h0, gamma, gap_mask=gap_mask)


def _create_PBH_signal(f0_setting="midpoint", delta_beta=0, Mc=None, f_signal=1, n_days=2):
    c, G, pi = 3e8, 6.67e-11, np.pi
    const = 96 / 5 * pi ** (8 / 3) * (G / c**3) ** (5 / 3)
    kpc = 3.086e19
    dist = 8 * kpc
    sidereal = five_vec(**_random_params())

    number_of_days = n_days  # keep integer days for clean 1/day sideband spacing
    T_obs = number_of_days * sidereal.side_day
    f_signal = f_signal
    n_samples = round(f_signal * T_obs)
    t_offset = np.linspace(0, T_obs, n_samples, endpoint=False, dtype=float)
    t_last = t_offset[-1]

    if Mc is not None:
        Mc = Mc * 2e30
    else:
        Mc = 10 ** np.random.uniform(-3, -1) * 2e30

    # Pick f0 so the demodulated carrier lands on a NUFFT bin in tau.
    if f0_setting == "midpoint":
        carrier_bin = 20000

        def _tau_span(f0_local):
            beta_local = const * f0_local ** (8 / 3) * Mc ** (5 / 3)
            return (3 / (5 * beta_local)) * (1 - (1 - 8 / 3 * beta_local * t_last) ** (5 / 8))

        f0 = carrier_bin / t_last
        for _ in range(10):
            f0_next = carrier_bin / _tau_span(f0)
            if np.isclose(f0_next, f0, rtol=0.0, atol=1e-14):
                break
            f0 = f0_next
    elif f0_setting == "uniform":
        f0 = np.random.uniform(0.1, 0.2)
    elif type(f0_setting) == float:
        f0 = f0_setting
    else:
        raise Exception("f0_setting error")

    omega0 = 2 * np.pi * f0
    assert f0 < f_signal / 2

    beta = const * f0 ** (8 / 3) * Mc ** (5 / 3)
    f = f0 * (1 - 8 / 3 * beta * t_offset) ** (-3 / 8)
    if any(f > f_signal):
        raise Exception("signal frequency goes above Nyquist")
    gamma = np.random.uniform(0, 2 * np.pi)
    h0 = 4 / dist * (G * Mc / (c**2)) ** (5 / 3) * (np.pi * f / c) ** (2 / 3)
    phi = -6 * pi / 5 * f0 * (1 - 8 / 3 * beta * t_offset) ** (5 / 8) / beta

    ref_time = Time("2019-04-10T12:34:56.000")
    t_gps = ref_time.gps + t_offset
    t = Time(t_gps, format="gps", scale="utc")

    sidereal.compute_H()
    sidereal.compute_A(sidereal.gmst(t.mjd))
    sidereal.compute_5vec()

    signal = h0 * sidereal.amp_modulation * np.exp(1j * (phi - phi[0] + gamma))
    beta_analysis = beta + delta_beta
    tau = -(3 / (5 * beta_analysis)) * (1 - 8 / 3 * beta_analysis * t_offset) ** (5 / 8)
    tau -= tau[0]
    return signal, tau, omega0, sidereal, h0, gamma, t


def check_PBH_signal(
    err=None,
    f0_setting="midpoint",
    gap_fraction=0.15,
    delta_beta=0,
    h_mag_err=None,
    h_phase_err=None,
    ratio_mag_err=None,
    ratio_phase_err=None,
    detection_stat_err=None,
    check_ratios=True,
):
    (
        h_mag_err,
        h_phase_err,
        ratio_mag_err,
        ratio_phase_err,
        detection_stat_err,
    ) = _resolve_tolerances(
        err,
        h_mag_err,
        h_phase_err,
        ratio_mag_err,
        ratio_phase_err,
        detection_stat_err,
    )

    signal, tau, omega0, sidereal, h0, gamma, t = _create_PBH_signal(f0_setting=f0_setting, delta_beta=delta_beta)
    gap_mask, gap_slice = _build_gap_mask(signal.size, gap_fraction=gap_fraction)

    gap_size = gap_slice.stop - gap_slice.start
    expected_gap_size = max(0, int(gap_fraction * signal.size))
    assert gap_size == expected_gap_size

    signal = np.where(gap_mask, signal, 0.0)
    data_X, resampler = _resample_and_extract_5vec(signal, tau, omega0)

    template_X, template_Xp, template_Xc = _time_domain_5vec(sidereal, t, tau, gap_mask=gap_mask)
    h_est = _estimator(data_X, template_X)
    hp_est, hc_est = _joint_estimator(data_X, template_Xp, template_Xc)
    hp_ratio = hp_est / h_est
    hc_ratio = hc_est / h_est
    h_reconstruct = np.sqrt(abs(hp_est) ** 2 + abs(hc_est) ** 2)

    expected_X = _expected_5vec_bins(signal, tau, resampler, omega0)
    expected_h = _estimator(expected_X, template_X)
    expected_hp, expected_hc = _joint_estimator(expected_X, template_Xp, template_Xc)
    expected_hp_ratio = expected_hp / expected_h
    expected_hc_ratio = expected_hc / expected_h
    expected_h_reconstruct = np.sqrt(abs(expected_hp) ** 2 + abs(expected_hc) ** 2)
    _, delta_omega = _sampled_bin_factor(resampler, omega0, tau)
    detected_stat = _detection_stat(template_Xp, template_Xc, hp_est, hc_est)
    injected_stat = _detection_stat(template_Xp, template_Xc, expected_hp, expected_hc)

    assert np.isclose(h_reconstruct, expected_h_reconstruct, rtol=h_mag_err, atol=0.0)
    assert np.isclose(np.abs(h_est), np.abs(expected_h), rtol=h_mag_err, atol=0.0)
    assert np.isclose(
        _wrapped_phase_diff(np.angle(h_est), np.angle(expected_h)),
        0.0,
        atol=h_phase_err,
    )
    assert np.isclose(detected_stat, injected_stat, rtol=detection_stat_err, atol=0.0)

    if check_ratios:
        assert np.isclose(np.abs(hp_ratio), np.abs(expected_hp_ratio), rtol=ratio_mag_err, atol=0.0)
        assert np.isclose(np.abs(hc_ratio), np.abs(expected_hc_ratio), rtol=ratio_mag_err, atol=0.0)
        if ratio_phase_err is not None:
            assert np.isclose(
                _wrapped_phase_diff(np.angle(hp_ratio), np.angle(expected_hp_ratio)),
                0.0,
                atol=ratio_phase_err,
            )
            assert np.isclose(
                _wrapped_phase_diff(np.angle(hc_ratio), np.angle(expected_hc_ratio)),
                0.0,
                atol=ratio_phase_err,
            )
    return np.array(
        [
            delta_omega / np.diff(resampler.freqs)[0],
            np.abs(h_est) / np.abs(expected_h) - 1,
            _wrapped_phase_diff(np.angle(h_est), np.angle(expected_h)),
            np.abs(hp_ratio) / np.abs(expected_hp_ratio) - 1,
            _wrapped_phase_diff(np.angle(hp_ratio), np.angle(expected_hp_ratio)),
            np.abs(hc_ratio) / np.abs(expected_hc_ratio) - 1,
            _wrapped_phase_diff(np.angle(hc_ratio), np.angle(expected_hc_ratio)),
        ]
    )
