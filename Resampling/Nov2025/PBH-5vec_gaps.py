# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: PBH
#     language: python
#     name: python3
# ---

# %%
from five_vec import five_vec
from resampler import Resampler
import numpy as np
from astropy.time import Time
import matplotlib.pyplot as pl

c, G, pi = 3e8, 6.67e-11, np.pi
const = 96 / 5 * pi ** (8 / 3) * (G / c**3) ** (5 / 3)


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


def _build_gap_mask(n_samples, gap_fraction=0.15):
    """Create a boolean mask with one contiguous missing-data segment at a random time."""
    mask = np.ones(n_samples, dtype=bool)
    gap_size = max(1, int(gap_fraction * n_samples))
    gap_start = np.random.randint(0, n_samples - gap_size + 1)
    gap_end = gap_start + gap_size
    mask[gap_start:gap_end] = False
    return mask, slice(gap_start, gap_end)


# %%
def create_signal():

    params = dict(
        ra=np.random.uniform(0, 2 * np.pi),
        dec=np.random.uniform(-np.pi / 2, np.pi / 2),
        eta=np.random.uniform(-1, 1),
        psi=np.random.uniform(0, 2 * np.pi),
        lat=np.random.uniform(-np.pi / 2, np.pi / 2),
        lng=np.random.uniform(-np.pi, np.pi),
        az=np.random.uniform(0, 2 * np.pi),
    )
    sidereal = five_vec(**params)
    h0 = np.random.uniform(1, 5)

    number_of_days = 2  # keep integer days for clean 1/day sideband spacing
    T_obs = number_of_days * sidereal.side_day
    f0 = np.random.uniform(0.1, 0.2)  # 10000 / sidereal.side_day
    omega0 = 2 * np.pi * f0
    f_sample = 1  # 8 * f0
    n_samples = round(f_sample * T_obs)

    ref_time = Time("2019-04-10T12:34:56.000")
    t_gps = ref_time.gps + np.arange(n_samples) / f_sample
    t = Time(t_gps, format="gps", scale="utc")
    t_offset = t.gps - t.gps[0]

    sidereal.compute_H()
    sidereal.compute_A(sidereal.gmst(t.mjd))
    sidereal.compute_5vec()
    amp_modulation = h0 * sidereal.amp_modulation

    Mc = 10 ** np.random.uniform(-3, -1) * 2e30
    beta = const * f0 ** (8 / 3) * Mc ** (5 / 3)
    phi = -6 * pi / 5 * f0 * (1 - 8 / 3 * beta * t_offset) ** (5 / 8) / beta
    gamma = np.random.uniform(0, 2 * np.pi)
    signal = amp_modulation * np.exp(1j * (phi + gamma))
    gap_mask, gap_slice = _build_gap_mask(signal.size, gap_fraction=0.15)
    signal = np.where(gap_mask, signal, 0.0)

    # Define tau so that phi = omega0 * tau, then remove the constant offset.
    tau = -(3 / (5 * beta)) * (1 - 8 / 3 * beta * t_offset) ** (5 / 8)
    tau_shifted = tau - tau[0]
    return signal, tau, omega0, sidereal, h0, gamma, t, gap_mask, gap_slice


def time_domain_5vec(t, sidereal, tau, gap_mask):
    sidereal_t = sidereal.gmst(t.mjd)
    sidereal_t -= sidereal_t[0]
    exp_terms = np.exp(1j * (np.arange(5) - 2)[:, np.newaxis] * sidereal_t)
    template_p = np.dot(sidereal.A_p, exp_terms)
    template_c = np.dot(sidereal.A_c, exp_terms)
    template_comb = np.dot(sidereal.A, exp_terms)
    template_p = np.where(gap_mask, template_p, 0.0)
    template_c = np.where(gap_mask, template_c, 0.0)
    template_comb = np.where(gap_mask, template_comb, 0.0)

    template_Xp, _ = _resample_and_extract_5vec(template_p, tau, 0)
    template_Xc, _ = _resample_and_extract_5vec(template_c, tau, 0)
    template_X, _ = _resample_and_extract_5vec(template_comb, tau, 0)
    return template_X, template_Xp, template_Xc


def estimate_parameters(hp_est, hc_est):
    reconstruct_h = np.sqrt(abs(hp_est) ** 2 + abs(hc_est) ** 2)
    # A = np.real(hp_est / reconstruct_h * np.conj(hc_est / reconstruct_h))
    # B = np.imag(hp_est / reconstruct_h * np.conj(hc_est / reconstruct_h))
    # C = abs(hp_est / reconstruct_h) ** 2 - abs(hc_est / reconstruct_h) ** 2
    # eta_est = (-1 + np.sqrt(1 - 4 * B**2)) / (2 * B)
    # psi_est = 1 / 4 * np.arccos(C / ((2 * A) ** 2 + C**2))
    # Hp_est = np.sqrt(1 / (1 + eta_est**2)) * (np.cos(2 * psi_est) - 1j * eta_est * np.sin(2 * psi_est))
    # gamma_est = np.angle(hp_est / Hp_est)
    return reconstruct_h  # , eta_est, psi_est, gamma_est


# %%
signal, tau, omega0, sidereal, h0, gamma, t, gap_mask, gap_slice = create_signal()

data_X, resampler = _resample_and_extract_5vec(signal, tau, omega0)
template_X, template_Xp, template_Xc = time_domain_5vec(t, sidereal, tau, gap_mask)

hp_est = _estimator(data_X, template_Xp)
hc_est = _estimator(data_X, template_Xc)
reconstruct_h = estimate_parameters(hp_est, hc_est)

idx0 = np.abs(resampler.freqs - omega0).argmin()
delta_omega = omega0 - resampler.freqs[idx0]
tau_span = tau[-1] - tau[0]
bin_amp_loss = np.sinc(delta_omega * tau_span / (2 * np.pi))
bin_dephasing = delta_omega * 0.5 * tau_span
bin_factor = bin_amp_loss * np.exp(1j * bin_dephasing)

target = h0 * np.exp(1j * gamma)
expected_h = target * bin_factor
print(
    np.array(
        [
            gap_slice.stop - gap_slice.start,
            delta_omega / np.diff(resampler.freqs)[0],
            reconstruct_h / abs(expected_h),
            # _wrapped_phase_diff(gamma, gamma_est),
            # eta_est - sidereal.eta,
            # _wrapped_phase_diff(psi_est, sidereal.psi),
        ]
    )
)


# %%
# Coverage test
N = 20
result_arr = []

for i in range(N):
    signal, tau, omega0, sidereal, h0, gamma, t, gap_mask, gap_slice = create_signal()

    data_X, resampler = _resample_and_extract_5vec(signal, tau, omega0)
    template_X, template_Xp, template_Xc = time_domain_5vec(t, sidereal, tau, gap_mask)

    hp_est = _estimator(data_X, template_Xp)
    hc_est = _estimator(data_X, template_Xc)
    reconstruct_h = estimate_parameters(hp_est, hc_est)

    idx0 = np.abs(resampler.freqs - omega0).argmin()
    delta_omega = omega0 - resampler.freqs[idx0]
    tau_span = tau[-1] - tau[0]
    bin_amp_loss = np.sinc(delta_omega * tau_span / (2 * np.pi))
    bin_dephasing = delta_omega * 0.5 * tau_span
    bin_factor = bin_amp_loss * np.exp(1j * bin_dephasing)

    target = h0 * np.exp(1j * gamma)
    expected_h = target * bin_factor
    result_arr.append(
        np.array(
            [
                gap_slice.stop - gap_slice.start,
                delta_omega / np.diff(resampler.freqs)[0],
                reconstruct_h / abs(expected_h),
                # _wrapped_phase_diff(gamma, gamma_est),
                # eta_est - sidereal.eta,
                # _wrapped_phase_diff(psi_est, sidereal.psi),
            ]
        )
    )

result_arr = np.array(result_arr)

# %%
pl.plot(result_arr[:, 1], result_arr[:, 2], "o", label="Reconstructed / Expected")
pl.ylabel("Amplitude ratio")
pl.legend()
pl.savefig("plots/PBH-5vec_gaps.png")

# %%
