"""Importable 0PN and 3.5PN signal-model generators.

The conventions match the paper-plot scripts:

- frequencies are gravitational-wave frequencies in Hz;
- phases are accumulated gravitational-wave phases in radians with
  ``phi(t=0) = 0``;
- chirp masses are supplied in solar masses;
- the 0PN model uses the beta convention from the resampling/template-bank
  scripts;
- the 3.5PN phase model is TaylorT4 in the time domain;
- the 3.5PN frequency track is obtained by inverting the TaylorF2
  chirp-time relation used in ``signal_model/PN_error.py``.
"""

import lal
import lalsimulation as lalsim
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


G = lal.G_SI
C = lal.C_SI
MSUN = lal.MSUN_SI
PI = np.pi

DEFAULT_ETA = 0.25
TAYLORF2_35PN_ORDER = lalsim.PNORDER_THREE_POINT_FIVE

__all__ = [
    "DEFAULT_ETA",
    "NoiseCurve",
    "SignalTrack",
    "TaylorF2FrequencyModel",
    "TaylorT4PhaseModel",
    "beta_0pn",
    "complex_signal_from_phase",
    "masses_from_mchirp_eta",
    "frequency_0pn",
    "frequency_35pn",
    "frequency_35pn_taylor_f2",
    "make_0pn_track",
    "make_35pn_track",
    "phase_0pn",
    "phase_35pn",
    "phase_35pn_taylor_t4",
    "tau_0pn",
    "taylor_t4_factor_35pn",
    # "total_mass_seconds_from_beta",
    "total_mass_seconds_from_mchirp_eta",
    "truncate_taylorf2_phasing",
]


def _as_float_array(t):
    return np.asarray(t, dtype=float)


def _return_scalar_if_scalar_input(value, original):
    if np.ndim(original) == 0:
        return float(np.asarray(value))
    return value


class NoiseCurve:
    """Interpolated one-sided PSD loaded from a two-column ASD file."""

    def __init__(self, frequency, psd):
        self.frequency = np.asarray(frequency, dtype=float)
        self.psd = np.asarray(psd, dtype=float)

    @classmethod
    def from_asd_file(cls, path):
        data = np.loadtxt(path)
        frequency = np.asarray(data[:, 0], dtype=float)
        asd = np.asarray(data[:, 1], dtype=float)
        valid = (
            np.isfinite(frequency)
            & np.isfinite(asd)
            & (frequency > 0.0)
            & (asd > 0.0)
        )
        frequency = frequency[valid]
        psd = asd[valid] ** 2

        order = np.argsort(frequency)
        frequency = frequency[order]
        psd = psd[order]
        if frequency.size < 2:
            raise ValueError(f"ASD file {path} does not contain a usable band.")
        return cls(frequency, psd)

    def psd_at(self, frequency):
        frequency = np.asarray(frequency, dtype=float)
        if np.any((frequency < self.frequency[0]) | (frequency > self.frequency[-1])):
            raise ValueError("PSD requested outside ASD frequency range.")
        return np.exp(
            np.interp(
                np.log(frequency),
                np.log(self.frequency),
                np.log(self.psd),
            )
        )


def masses_from_mchirp_eta(
    mchirp_msun, eta=DEFAULT_ETA
):
    """Return chirp, total, and component masses in SI units.
    """

    mchirp = mchirp_msun * MSUN
    total_mass = mchirp / eta ** (3.0 / 5.0)
    sqrt_term = np.sqrt(max(0.0, 1.0 - 4.0 * eta))
    m1 = 0.5 * total_mass * (1.0 + sqrt_term)
    m2 = 0.5 * total_mass * (1.0 - sqrt_term)
    return mchirp, total_mass, m1, m2


def total_mass_seconds_from_mchirp_eta(
    mchirp_msun, eta=DEFAULT_ETA
):
    """Return total mass in seconds from chirp mass in solar masses."""

    _, total_mass, _, _ = masses_from_mchirp_eta(mchirp_msun, eta)
    return G * total_mass / C**3


def beta_0pn(f0_hz, mchirp_msun):
    """Return the 0PN beta used by the resampling scripts."""

    return (
        (96.0 / 5.0)
        * PI ** (8.0 / 3.0)
        * (G * mchirp_msun * MSUN / C**3) ** (5.0 / 3.0)
        * f0_hz ** (8.0 / 3.0)
    )


# def total_mass_seconds_from_beta(f0_hz, beta, eta):
#     """Invert the 0PN beta convention to a total mass in seconds."""

#     mchirp_sec = (
#         beta
#         / ((96.0 / 5.0) * PI ** (8.0 / 3.0) * f0_hz ** (8.0 / 3.0))
#     ) ** (3.0 / 5.0)
#     return mchirp_sec / eta ** (3.0 / 5.0)


def tau_0pn(
    t,
    beta,
):
    """Return accumulated 0PN resampled time ``tau = phi / (2*pi*f0)``.

    This is the zero-at-start convention used by the phase comparisons:

    ``tau(t) = 3/(5 beta) * (1 - (1 - 8 beta t / 3)**(5/8))``.
    """

    t_array = _as_float_array(t)
    chirp_term = 1.0 - (8.0 / 3.0) * beta * t_array
    if np.any(chirp_term <= 0.0):
        raise ValueError("0PN frequency has diverged.")

    tau = (3.0 / (5.0 * beta)) * (1.0 - chirp_term ** (5.0 / 8.0))
    return _return_scalar_if_scalar_input(tau, t)


def frequency_0pn(
    t,
    f0_hz,
    mchirp_msun=None,
    beta=None,
):
    """Return the 0PN gravitational-wave frequency in Hz."""

    if beta is None:
        if mchirp_msun is None:
            raise ValueError("Provide either mchirp_msun or beta.")
        beta = beta_0pn(f0_hz, mchirp_msun)

    t_array = _as_float_array(t)
    chirp_term = 1.0 - (8.0 / 3.0) * beta * t_array
    if np.any(chirp_term <= 0.0):
        raise ValueError("0PN frequency requested after formal coalescence.")

    frequency = f0_hz * chirp_term ** (-3.0 / 8.0)
    return _return_scalar_if_scalar_input(frequency, t)


def phase_0pn(
    t,
    f0_hz,
    mchirp_msun=None,
    beta=None,
):
    """Return the accumulated 0PN gravitational-wave phase in radians."""

    if beta is None:
        if mchirp_msun is None:
            raise ValueError("Provide either mchirp_msun or beta.")
        beta = beta_0pn(f0_hz, mchirp_msun)

    phase = 2.0 * PI * f0_hz * tau_0pn(t, beta)
    return _return_scalar_if_scalar_input(np.asarray(phase), t)


def complex_signal_from_phase(
    phase,
    amplitude=1.0,
):
    """Return a restricted complex signal ``amplitude * exp(i phase)``."""

    signal = amplitude * np.exp(1j * np.asarray(phase, dtype=float))
    if np.ndim(phase) == 0:
        return complex(signal)
    return signal


def taylor_t4_factor_35pn(v, eta=DEFAULT_ETA):
    """Nonspinning point-particle TaylorT4 3.5PN velocity factor."""

    v = np.asarray(v, dtype=float)
    a2 = -(743.0 + 924.0 * eta) / 336.0
    a3 = 4.0 * PI
    a4 = (34103.0 + 122949.0 * eta + 59472.0 * eta**2) / 18144.0
    a5 = -PI * (4159.0 + 15876.0 * eta) / 672.0
    a6 = (
        16447322263.0 / 139708800.0
        - 1712.0 * np.euler_gamma / 105.0
        - 856.0 * np.log(16.0) / 105.0
        - 56198689.0 * eta / 217728.0
        + PI**2 * (16.0 / 3.0 + 451.0 * eta / 48.0)
        + 541.0 * eta**2 / 896.0
        - 5605.0 * eta**3 / 2592.0
    )
    a6_log = -1712.0 / 105.0
    a7 = PI * (-13245.0 + 717350.0 * eta + 731960.0 * eta**2) / 12096.0

    return (
        1.0
        + a2 * v**2
        + a3 * v**3
        + a4 * v**4
        + a5 * v**5
        + (a6 + a6_log * np.log(v)) * v**6
        + a7 * v**7
    )


class TaylorT4PhaseModel:
    """3.5PN TaylorT4 time-domain phase model.

    ``phase(t)`` returns the accumulated gravitational-wave phase in radians,
    aligned so that ``phase(0) = 0``. ``frequency(t)`` is provided as the
    instantaneous TaylorT4 frequency for diagnostics; use
    ``TaylorF2FrequencyModel`` for the 3.5PN frequency convention used by
    ``PN_error.py``.
    """

    def __init__(
        self,
        f0_hz,
        mchirp_msun,
        eta=DEFAULT_ETA,
        t_max=1.0,
        rtol=1.0e-10,
        atol_v=1.0e-12,
        atol_phase=1.0e-10,
    ):
        self.f0_hz = f0_hz
        self.mchirp_msun = mchirp_msun
        self.eta = eta
        self.t_max = t_max
        self.rtol = rtol
        self.atol_v = atol_v
        self.atol_phase = atol_phase
        self.total_mass_sec = total_mass_seconds_from_mchirp_eta(
            self.mchirp_msun, self.eta
        )
        v_start = (PI * self.total_mass_sec * self.f0_hz) ** (1.0 / 3.0)
        v_isco = 1.0 / np.sqrt(6.0)
        if not 0.0 < v_start < v_isco:
            raise ValueError("starting frequency is outside the TaylorT4 range")

        self.v_start = v_start
        self.v_isco = v_isco
        self.solution = self._solve()

    def _solve(self):
        def rhs(_t, y):
            v, _phase = y
            factor = taylor_t4_factor_35pn(v, self.eta)
            dv_dt = (32.0 * self.eta / (5.0 * self.total_mass_sec)) * v**9 * factor
            dphase_dt = 2.0 * v**3 / self.total_mass_sec
            return [float(dv_dt), float(dphase_dt)]

        def reaches_isco(_t, y):
            return y[0] - self.v_isco

        reaches_isco.terminal = True
        reaches_isco.direction = 1

        solution = solve_ivp(
            rhs,
            (0.0, float(self.t_max)),
            [self.v_start, 0.0],
            events=reaches_isco,
            dense_output=True,
            rtol=self.rtol,
            atol=[self.atol_v, self.atol_phase],
            method="DOP853",
        )

        if solution.status == 1:
            raise ValueError("TaylorT4 track reaches ISCO before t_max.")
        if not solution.success:
            raise RuntimeError(solution.message)
        return solution

    def phase(self, t):
        """Return accumulated 3.5PN TaylorT4 phase in radians."""

        t_array = _as_float_array(t)
        if np.any((t_array < 0.0) | (t_array > self.t_max)):
            raise ValueError("TaylorT4 phase requested outside [0, t_max].")
        phase = self.solution.sol(t_array)[1]
        return _return_scalar_if_scalar_input(phase, t)

    def frequency(self, t):
        """Return instantaneous TaylorT4 GW frequency in Hz."""

        t_array = _as_float_array(t)
        if np.any((t_array < 0.0) | (t_array > self.t_max)):
            raise ValueError("TaylorT4 frequency requested outside [0, t_max].")
        v = self.solution.sol(t_array)[0]
        frequency = v**3 / (PI * self.total_mass_sec)
        return _return_scalar_if_scalar_input(frequency, t)


def phase_35pn_taylor_t4(
    t,
    f0_hz,
    mchirp_msun,
    eta=DEFAULT_ETA,
):
    """Convenience wrapper for 3.5PN TaylorT4 accumulated phase."""

    t_array = _as_float_array(t)
    t_max = float(np.max(t_array)) if t_array.size else 0.0
    model = TaylorT4PhaseModel(f0_hz, mchirp_msun, eta=eta, t_max=t_max)
    return model.phase(t)


def truncate_taylorf2_phasing(phasing, pn_phase_order=TAYLORF2_35PN_ORDER):
    """Zero TaylorF2 phasing coefficients above the requested PN order."""

    max_order = lalsim.PNORDER_THREE_POINT_FIVE
    for order in range(pn_phase_order + 1, max_order + 1):
        phasing.v[order] = 0.0
        phasing.vlogv[order] = 0.0
        phasing.vlogvsq[order] = 0.0
    return phasing


class TaylorF2FrequencyModel:
    """3.5PN TaylorF2 elapsed-time frequency model.

    The model follows ``signal_model/PN_error.py``: compute LAL's TaylorF2
    chirp-time derivative on a frequency grid, subtract the value at
    ``f0_hz`` to get elapsed time, and invert the monotonic relation.
    """

    def __init__(
        self,
        f0_hz,
        f_stop_hz,
        mchirp_msun,
        eta=DEFAULT_ETA,
        pn_phase_order=TAYLORF2_35PN_ORDER,
        n_grid=20000,
        interpolator_kind="cubic",
    ):
        self.f0_hz = f0_hz
        self.f_stop_hz = f_stop_hz
        self.mchirp_msun = mchirp_msun
        self.eta = eta
        self.pn_phase_order = pn_phase_order
        self.n_grid = n_grid
        self.interpolator_kind = interpolator_kind

        if self.f_stop_hz <= self.f0_hz:
            raise ValueError("f_stop_hz must be greater than f0_hz.")
        if self.n_grid < 4:
            raise ValueError("n_grid must be at least 4 for cubic interpolation.")

        _, total_mass, m1, m2 = masses_from_mchirp_eta(
            self.mchirp_msun, self.eta
        )
        mtot_sec = G * total_mass / C**3
        params = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(params, self.pn_phase_order)
        phasing = lalsim.SimInspiralTaylorF2AlignedPhasing(m1, m2, 0.0, 0.0, params)
        phasing = truncate_taylorf2_phasing(phasing, self.pn_phase_order)

        f_grid = np.geomspace(self.f0_hz, self.f_stop_hz, self.n_grid)
        t_of_f = np.array(
            [
                lalsim.PNPhaseDerivative(float(f), 2, phasing, mtot_sec)
                / (2.0 * PI)
                for f in f_grid
            ]
        )
        t_elapsed = t_of_f[0] - t_of_f

        valid = np.isfinite(t_elapsed) & np.isfinite(f_grid)
        t_elapsed = t_elapsed[valid]
        f_grid = f_grid[valid]
        if np.any(np.diff(t_elapsed) <= 0.0):
            raise ValueError("TaylorF2 chirp time must increase monotonically.")

        interpolator = interp1d(
            t_elapsed,
            f_grid,
            kind=self.interpolator_kind,
            bounds_error=True,
        )

        self.frequency_grid = f_grid
        self.elapsed_time_grid = t_elapsed
        self.t_end = float(t_elapsed[-1])
        self._frequency_of_time = interpolator

    def frequency(self, t):
        """Return 3.5PN TaylorF2 frequency in Hz at elapsed time ``t``."""

        frequency = self._frequency_of_time(t)
        return _return_scalar_if_scalar_input(np.asarray(frequency), t)


def frequency_35pn_taylor_f2(
    t,
    f0_hz,
    f_stop_hz,
    mchirp_msun,
    eta=DEFAULT_ETA,
    n_grid=20000,
):
    """Convenience wrapper for the 3.5PN TaylorF2 frequency track."""

    model = TaylorF2FrequencyModel(
        f0_hz=f0_hz,
        f_stop_hz=f_stop_hz,
        mchirp_msun=mchirp_msun,
        eta=eta,
        n_grid=n_grid,
    )
    return model.frequency(t)


phase_35pn = phase_35pn_taylor_t4
frequency_35pn = frequency_35pn_taylor_f2


class SignalTrack:
    """Container for sampled time, frequency, phase, and complex signal."""

    def __init__(self, t, frequency, phase, signal):
        self.t = t
        self.frequency = frequency
        self.phase = phase
        self.signal = signal


def make_0pn_track(
    t,
    f0_hz,
    mchirp_msun,
    beta=None,
):
    """Return sampled 0PN frequency, phase, and unit-amplitude signal."""

    t = _as_float_array(t)
    if beta is None:
        beta = beta_0pn(f0_hz, mchirp_msun)
    frequency = np.asarray(frequency_0pn(t, f0_hz, beta=beta), dtype=float)
    phase = np.asarray(phase_0pn(t, f0_hz, beta=beta), dtype=float)
    return SignalTrack(t=t, frequency=frequency, phase=phase, signal=np.exp(1j * phase))


def make_35pn_track(
    t,
    f0_hz,
    f_stop_hz,
    mchirp_msun,
    eta=DEFAULT_ETA,
    n_grid=20000,
):
    """Return sampled 3.5PN track using TaylorF2 frequency and TaylorT4 phase."""

    t = _as_float_array(t)
    phase_model = TaylorT4PhaseModel(
        f0_hz=f0_hz,
        mchirp_msun=mchirp_msun,
        eta=eta,
        t_max=float(np.max(t)) if t.size else 0.0,
    )
    frequency_model = TaylorF2FrequencyModel(
        f0_hz=f0_hz,
        f_stop_hz=f_stop_hz,
        mchirp_msun=mchirp_msun,
        eta=eta,
        n_grid=n_grid,
    )
    phase = np.asarray(phase_model.phase(t), dtype=float)
    frequency = np.asarray(frequency_model.frequency(t), dtype=float)
    return SignalTrack(t=t, frequency=frequency, phase=phase, signal=np.exp(1j * phase))
