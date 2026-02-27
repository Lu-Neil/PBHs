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

# %%
c = 3e8
G = 6.67e-11
pi = np.pi
kpc = 3.086e+19
const = 96/5*pi**(8/3)*(G/c**3)**(5/3)


# %%
def f_calc(f0, beta, t):
    return f0*(1-8/3*beta*t)**(-3/8)


# %%
side_day = 86164.09053083288
n_days = 3
T_obs = n_days*side_day

f0 = 1/side_day * 10000
Mc = 1e-1 * 2e30
f_max = 4
beta = const*f0**(8/3)*Mc**(5/3)
f_signal = 4*f_max
nt = round(f_signal*T_obs)+1

ref_time = Time('2019-04-03')
t = ref_time.gps + np.arange(nt)/f_signal
t = Time(t, format="gps", scale="utc")
t_offset = t.gps-t.gps[0]

phi = -6*pi/5*f0*(1-8./3.*(beta)*t_offset)**(5/8)/beta
f = f_calc(f0, beta, t_offset)
signal_source = 1*np.exp(1j*phi)
tau = phi / (2*np.pi*f0)

# %%
resampler = Resampler()
resampler.timeseries = signal_source
resampler.resampled_time = tau
resampler.nufft()
pl.plot(resampler.freqs, resampler.power_normalized, 'o')
pl.xlim(f0*(2*np.pi) - 20/side_day, f0*(2*np.pi) + 20/side_day)

# %%
# This uses five_vec.py to compute the doppler modulation of the signal

d = 8 * kpc
h0 = 4/d * (G*Mc/(c**2))**(5/3)*(np.pi*f/c)**(2/3)

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
pl.plot(resampler.freqs/(2*np.pi), resampler.power_normalized, 'o')
pl.xlim(f0 - 10/side_day, f0 + 10/side_day)

# %%
X = np.empty((5), dtype=complex)

X[0] = resampler.weights_normalized[abs(resampler.freqs/(2*np.pi)-(f0-2/side_day)).argmin()]
X[1] = resampler.weights_normalized[abs(resampler.freqs/(2*np.pi)-(f0-1/side_day)).argmin()]
X[2] = resampler.weights_normalized[abs(resampler.freqs/(2*np.pi)-f0).argmin()]
X[3] = resampler.weights_normalized[abs(resampler.freqs/(2*np.pi)-(f0+1/side_day)).argmin()]
X[4] = resampler.weights_normalized[abs(resampler.freqs/(2*np.pi)-(f0+2/side_day)).argmin()]

print(abs(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2)))
print(h0[0])

# %%
sidereal_t = sidereal.gmst(t.mjd)
sidereal_t -= sidereal_t[0]

exp_terms = np.exp(1j*(np.arange(5)-2)[:, np.newaxis]*sidereal_t)
template_p = np.dot(sidereal.A_p, exp_terms)
template_c = np.dot(sidereal.A_c, exp_terms)
template_comb = np.dot(sidereal.A, exp_terms)

# %%
resampler_sidereal = Resampler()
resampler_sidereal.timeseries = template_c
resampler_sidereal.resampled_time = tau
resampler_sidereal.nufft()

# %%
pl.plot(resampler_sidereal.freqs/(2*np.pi), resampler_sidereal.power_normalized, 'o')
pl.xlim(-10/side_day,10/side_day)

# %%
template_X = np.empty((5), dtype=complex)
template_weights = resampler_sidereal.weights_normalized
template_freqs = resampler_sidereal.freqs/(2*np.pi)

template_X[0] = template_weights[abs(template_freqs-(-2/side_day)).argmin()]
template_X[1] = template_weights[abs(template_freqs-(-1/side_day)).argmin()]
template_X[2] = template_weights[abs(template_freqs).argmin()]
template_X[3] = template_weights[abs(template_freqs-(1/side_day)).argmin()]
template_X[4] = template_weights[abs(template_freqs-(2/side_day)).argmin()]

print(abs(np.dot(X, np.conj(template_X))/np.sum(np.abs(template_X)**2)))
print(h0[0])

# %%
resampler_sidereal = Resampler()
resampler_sidereal.timeseries = template_p
resampler_sidereal.resampled_time = tau
resampler_sidereal.nufft()

template_X = np.empty((5), dtype=complex)
template_weights = resampler_sidereal.weights_normalized
template_freqs = resampler_sidereal.freqs/(2*np.pi)

template_X[0] = template_weights[abs(template_freqs-(-2/side_day)).argmin()]
template_X[1] = template_weights[abs(template_freqs-(-1/side_day)).argmin()]
template_X[2] = template_weights[abs(template_freqs).argmin()]
template_X[3] = template_weights[abs(template_freqs-(1/side_day)).argmin()]
template_X[4] = template_weights[abs(template_freqs-(2/side_day)).argmin()]

print("hp --------")
print(abs(np.dot(X, np.conj(template_X))/np.sum(np.abs(template_X)**2)))
print(h0[0] * sidereal.H_p)

# %%
resampler_sidereal = Resampler()
resampler_sidereal.timeseries = template_c
resampler_sidereal.resampled_time = tau
resampler_sidereal.nufft()

template_X = np.empty((5), dtype=complex)
template_weights = resampler_sidereal.weights_normalized
template_freqs = resampler_sidereal.freqs/(2*np.pi)

template_X[0] = template_weights[abs(template_freqs-(-2/side_day)).argmin()]
template_X[1] = template_weights[abs(template_freqs-(-1/side_day)).argmin()]
template_X[2] = template_weights[abs(template_freqs).argmin()]
template_X[3] = template_weights[abs(template_freqs-(1/side_day)).argmin()]
template_X[4] = template_weights[abs(template_freqs-(2/side_day)).argmin()]

print("hc --------")
print(abs(np.dot(X, np.conj(template_X))/np.sum(np.abs(template_X)**2)))
print(h0[0] * sidereal.H_c)

# %%
