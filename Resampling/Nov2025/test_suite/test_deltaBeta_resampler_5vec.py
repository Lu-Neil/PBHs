from functools import lru_cache
from importlib import import_module

import numpy as np
import pytest

_utils = import_module("..fiveVec_resampler_utils", package=__package__)
_create_PBH_signal = _utils._create_PBH_signal
_estimator = _utils._estimator
_resample_and_extract_5vec = _utils._resample_and_extract_5vec
_time_domain_5vec = _utils._time_domain_5vec


@lru_cache(maxsize=None)
def _pbh_case(seed, delta_beta):
    np.random.seed(seed)
    return _create_PBH_signal(f0_setting="midpoint", delta_beta=delta_beta)


@lru_cache(maxsize=None)
def _recovered_h_power(seed, delta_beta):
    signal, tau, omega0, sidereal, _, _, t = _pbh_case(seed, delta_beta)
    data_X, _ = _resample_and_extract_5vec(signal, tau, omega0)
    template_X, _, _ = _time_domain_5vec(sidereal, t, tau)
    h_est = _estimator(data_X, template_X)
    return float(np.abs(h_est) ** 2)


def _infer_beta_from_amplitude(t_offset, h0):
    t_last = t_offset[-1]
    return (3.0 / (8.0 * t_last)) * (1.0 - (h0[0] / h0[-1]) ** 4)


def _delta_f(t_offset, f0, beta, delta_beta):
    temp0 = (3.0 * f0 * delta_beta) / (5.0 * beta * (1.0 - (8.0 / 3.0) * beta * t_offset) ** (3.0 / 8.0))
    temp1 = 1.0 + (-1.0 + beta * t_offset) / (1.0 - (8.0 / 3.0) * beta * t_offset)
    return np.abs(temp0 * temp1)


def _crossover_time(t_offset, f0, beta, delta_beta):
    T_obs = t_offset[-1]
    delta_f = _delta_f(t_offset, f0, beta, delta_beta)
    crossed = np.flatnonzero(delta_f >= 1.0 / T_obs)
    if crossed.size == 0:
        return T_obs
    return t_offset[crossed[0]]


def _theoretical_loss_fraction(beta, tcross, T_obs):
    if tcross >= T_obs:
        return 0.0

    edge_tcross = np.sqrt(1.0 - (8.0 / 3.0) * beta * tcross)
    edge_T = np.sqrt(1.0 - (8.0 / 3.0) * beta * T_obs)
    return float((edge_tcross - edge_T) / (1.0 - edge_T))


def _discrete_tail_loss(h0, t_offset, tcross):
    if tcross >= t_offset[-1]:
        return 0.0

    tail = t_offset >= tcross
    return float(np.sum(np.abs(h0[tail]) ** 2) / np.sum(np.abs(h0) ** 2))


def _theoretical_retained_power(seed, delta_beta):
    _, _, omega0, _, h0, _, t = _pbh_case(seed, 0.0)
    t_offset = t.gps - t.gps[0]
    beta = _infer_beta_from_amplitude(t_offset, h0)
    f0 = omega0 / (2.0 * np.pi)
    tcross = _crossover_time(t_offset, f0, beta, delta_beta)
    return 1.0 - _theoretical_loss_fraction(beta, tcross, t_offset[-1])


def _recovered_power_fraction(seed, delta_beta):
    return _recovered_h_power(seed, delta_beta) / _recovered_h_power(seed, 0.0)


@pytest.mark.parametrize("delta_beta", [0.0, 3e-10, 5e-10, 1e-9])
def test_delta_beta_tail_power_formula(delta_beta):
    """Eq. (31) matches the discrete PBH chirp power beyond tcross."""
    _, _, omega0, _, h0, _, t = _pbh_case(0, 0.0)
    t_offset = t.gps - t.gps[0]
    beta = _infer_beta_from_amplitude(t_offset, h0)
    f0 = omega0 / (2.0 * np.pi)
    tcross = _crossover_time(t_offset, f0, beta, delta_beta)

    numerical_loss = _discrete_tail_loss(h0, t_offset, tcross)
    theoretical_loss = _theoretical_loss_fraction(beta, tcross, t_offset[-1])
    assert np.isclose(numerical_loss, theoretical_loss, rtol=3e-4, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_delta_beta_small_mismatch_keeps_5vec_power(seed):
    """Before tcross enters the run, the coherent recovery stays near unity."""
    delta_beta = 2e-10
    retained = _theoretical_retained_power(seed, delta_beta)
    recovered = _recovered_power_fraction(seed, delta_beta)

    assert retained == pytest.approx(1.0, abs=0.0)
    assert recovered > 0.95


@pytest.mark.parametrize("delta_beta", [3e-10, 5e-10, 1e-9])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_delta_beta_5vec_power_is_bounded_by_tail_power_prediction(seed, delta_beta):
    """The tail-power model is a conservative upper bound for the coherent estimator."""
    retained = _theoretical_retained_power(seed, delta_beta)
    recovered = _recovered_power_fraction(seed, delta_beta)

    assert retained < 1.0
    assert recovered < retained
