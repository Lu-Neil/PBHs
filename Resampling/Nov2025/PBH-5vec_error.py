# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as pl
from five_vec import five_vec
import finufft
from astropy.time import Time
from resampler import Resampler
from pathlib import Path

# %%
c = 3e8
G = 6.67e-11
pi = np.pi
const = 96/5*pi**(8/3)*(G/c**3)**(5/3)


# %%
def t_max_calc(f0, beta, f_max):
    temp0 = 0.375/beta
    temp1 = 1-(f0/f_max)**(8/3)
    return temp0*temp1

def f_calc(f0, beta, t):
    return f0*(1-8/3*beta*t)**(-3/8)



# %%
side_day = 86164.09053083288
n_days = 3
T_obs = n_days*side_day

f0 = 1/side_day * 10000
Mc = 3e-2 * 2e30
f_max = 4
beta = const*f0**(8/3)*Mc**(5/3)
f_signal = 4*f_max
nt = round(f_signal*T_obs)+1

ref_time = Time('2019-04-02')
t = ref_time.gps + np.arange(nt)/f_signal
t = Time(t, format="gps", scale="utc")
t_offset = t.gps-t.gps[0]

phi = -6*pi/5*f0*(1-8./3.*(beta)*t_offset)**(5/8)/beta
tau = phi/f0
signal_source = 1*np.exp(1j*phi)

# %%
resampler = Resampler()
resampler.timeseries = signal_source
resampler.resampled_time = tau
resampler.nufft()
pl.plot(resampler.freqs, resampler.power_normalized, 'o')
pl.xlim(f0 - 20/side_day, f0 + 20/side_day)

# %%
# This uses five_vec.py to compute the doppler modulation of the signal
h0 = np.random.uniform(1, 5)
params = dict(ra = np.random.uniform(0, 2*np.pi), 
              dec = np.random.uniform(-np.pi/2, np.pi/2), 
              eta = np.random.uniform(-1, 1), 
              psi = np.random.uniform(0, 2*np.pi), 
              lat = np.random.uniform(-np.pi/2, np.pi/2), 
              lng = np.random.uniform(-np.pi, np.pi),
              az = np.random.uniform(0, 2*np.pi), 
              side_day=side_day
             )
sidereal = five_vec(**params)
sidereal.compute_H()
sidereal.compute_A(sidereal.gmst(t.mjd))
sidereal.compute_5vec()
amp_modulation = h0 * sidereal.amp_modulation
signal_det = amp_modulation * signal_source

# %%
resampler = Resampler()
resampler.timeseries = signal_det
resampler.resampled_time = tau
resampler.nufft()
pl.plot(resampler.freqs, resampler.power_normalized, 'o')
pl.xlim(f0 - 10/side_day, f0 + 10/side_day)

# %%
X = np.empty((5), dtype=complex)

X[0] = resampler.weights_normalized[abs(resampler.freqs-(f0-2/side_day)).argmin()]
X[1] = resampler.weights_normalized[abs(resampler.freqs-(f0-1/side_day)).argmin()]
X[2] = resampler.weights_normalized[abs(resampler.freqs-f0).argmin()]
X[3] = resampler.weights_normalized[abs(resampler.freqs-(f0+1/side_day)).argmin()]
X[4] = resampler.weights_normalized[abs(resampler.freqs-(f0+2/side_day)).argmin()]

print(abs(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2)))
print(h0)

# %%
phi = -6*pi/5*f0*(1-8./3.*(beta)*t_offset)**(5/8)/beta
tau = phi/f0
signal_source = 1*np.exp(1j*phi)
signal_det = amp_modulation * signal_source

resampler = Resampler()
resampler.timeseries = signal_det
resampler.resampled_time = tau
resampler.nufft()

X = np.empty((5), dtype=complex)
X[0] = resampler.weights_normalized[abs(resampler.freqs-(f0-2/side_day)).argmin()]
X[1] = resampler.weights_normalized[abs(resampler.freqs-(f0-1/side_day)).argmin()]
X[2] = resampler.weights_normalized[abs(resampler.freqs-f0).argmin()]
X[3] = resampler.weights_normalized[abs(resampler.freqs-(f0+1/side_day)).argmin()]
X[4] = resampler.weights_normalized[abs(resampler.freqs-(f0+2/side_day)).argmin()]

print(abs(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2))/h0)
print(min(abs(resampler.freqs-f0))/np.diff(resampler.freqs)[0])

# %% [markdown]
# ## Dependence on offset from bin

# %%
# from scipy.signal.windows import hann
# window = hann(len(t_offset))
# window_power = np.sum(window**2)/len(window)

# %%
def extract_5vec_nearest(resampler, f0):
    X = np.empty((5), dtype=complex)
    for i, offset in enumerate(np.arange(-2, 3) / side_day):
        X[i] = resampler.weights_normalized[abs(resampler.freqs - (f0 + offset)).argmin()]
    return X


def sampled_carrier_factor(resampler, f0, tau):
    idx = abs(resampler.freqs - f0).argmin()
    delta = f0 - resampler.freqs[idx]
    tau_offset = tau - tau[0]
    return np.mean(np.exp(1j * delta * tau_offset))


def expected_projected_5vec(signal, tau, resampler, f0, A_template):
    """Exact finite-sample expectation for the 5-vector estimator.

    FINUFFT approximates this NUDFT sum.  This includes the off-bin carrier
    response and the small non-orthogonality of the sidereal sidebands when the
    sidereal modulation is periodic in t but the transform coordinate is tau.
    """
    tau_offset = tau - tau[0]
    X_expected = np.empty((5), dtype=complex)
    for i, offset in enumerate(np.arange(-2, 3) / side_day):
        idx = abs(resampler.freqs - (f0 + offset)).argmin()
        bin_freq = resampler.freqs[idx]
        X_expected[i] = np.mean(signal * np.exp(-1j * bin_freq * tau_offset))
    return np.dot(X_expected, np.conj(A_template)) / np.sum(np.abs(A_template) ** 2)


# %%
power_arr = []
expected_power_arr = []
carrier_kernel_power_arr = []
sinc_power_arr = []
distance_arr = []

f0_arr = np.linspace(0.01, 0.02, 35)

for f0 in f0_arr:
    beta = const*f0**(8/3)*Mc**(5/3)
    
    phi = -6*pi/5*f0*(1-8./3.*(beta)*t_offset)**(5/8)/beta
    tau = phi/f0
    signal_source = 1*np.exp(1j*phi)
    signal_det = amp_modulation * signal_source

    resampler = Resampler()
    resampler.timeseries = signal_det #window
    resampler.resampled_time = tau
    resampler.nufft()

    X = extract_5vec_nearest(resampler, f0)
    h_est = np.dot(X, np.conj(sidereal.A)) / np.sum(np.abs(sidereal.A)**2)
    expected_h = expected_projected_5vec(signal_det, tau, resampler, f0, sidereal.A)

    distance = min(abs(resampler.freqs-f0))/np.diff((resampler.freqs))[0]
    power_arr.append(abs(h_est/h0)**2)
    expected_power_arr.append(abs(expected_h/h0)**2)
    carrier_kernel_power_arr.append(abs(sampled_carrier_factor(resampler, f0, tau))**2)
    sinc_power_arr.append(abs(np.sinc(distance))**2)
    distance_arr.append(distance)

# %%
distance_arr = np.array(distance_arr)
power_arr = np.array(power_arr)
expected_power_arr = np.array(expected_power_arr)
carrier_kernel_power_arr = np.array(carrier_kernel_power_arr)
sinc_power_arr = np.array(sinc_power_arr)
order = np.argsort(distance_arr)

pl.figure(figsize=(7.0, 4.5))
pl.plot(distance_arr[order], power_arr[order], 'o', label='NUFFT recovered 5-vector power')
pl.plot(distance_arr[order], expected_power_arr[order], '-', label='Exact sampled 5-vector expectation')
pl.plot(distance_arr[order], carrier_kernel_power_arr[order], '--', label='Sampled carrier-kernel power')
pl.plot(distance_arr[order], sinc_power_arr[order], ':', label='Uniform-tau sinc$^2$ limit')
pl.ylabel(r"Recovered power $|\hat h/h_0|^2$")
pl.xlabel("Offset from nearest bin centre [bins]")
pl.legend()
pl.grid(True)
pl.tight_layout()

fig_path = Path("Resampling/Nov2025/demonstrating_behaviour/figs/pbh_5vec_bin_offset_power.png")
fig_path.parent.mkdir(parents=True, exist_ok=True)
pl.savefig(fig_path, dpi=200)
print(f"Saved {fig_path}")

# %%
pl.figure(figsize=(7.0, 3.5))
pl.plot(distance_arr[order], power_arr[order] / expected_power_arr[order], 'o')
pl.axhline(1, color='k', linewidth=1)
pl.ylabel("Recovered / expected power")
pl.xlabel("Offset from nearest bin centre [bins]")
pl.grid(True)
pl.tight_layout()

# %%
