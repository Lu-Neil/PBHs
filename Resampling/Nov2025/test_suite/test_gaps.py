from importlib import import_module

_utils = import_module("..fiveVec_resampler_utils", package=__package__)
check_PBH_signal = _utils.check_PBH_signal


def test_midpoint_PBH_signal_w_gap():
    """PBH chirp with midpoint-injected f0 and a contiguous 15% data gap."""
    check_PBH_signal(err=1e-3, f0_setting="midpoint", gap_fraction=0.15)


def test_uniform_PBH_signal_w_gap():
    """PBH chirp with uniformly injected f0 and a contiguous 15% data gap."""
    check_PBH_signal(err=2e-1, f0_setting="uniform", gap_fraction=0.15)
