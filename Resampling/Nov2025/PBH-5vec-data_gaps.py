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
Mc = 3e-2 * 2e30
f_max = 4
beta = const*f0**(8/3)*Mc**(5/3)
f_signal = 4*f_max
nt = round(f_signal*T_obs)+1

ref_time = Time('2019-04-03')
t = ref_time.gps + np.arange(nt)/f_signal
t = Time(t, format="gps", scale="utc")
t_offset = t.gps-t.gps[0]
f = f_calc(f0, beta, t_offset)

phi = -6*pi/5*f0*(1-8./3.*(beta)*t_offset)**(5/8)/beta
signal_source = 1*np.exp(1j*phi)
tau = phi / (f0)
gamma = np.mod(phi[0], 2*np.pi)

# %%
resampler = Resampler()
resampler.timeseries = signal_source
resampler.resampled_time = tau/(2*np.pi)
resampler.nufft()
pl.plot(resampler.freqs, resampler.power_normalized, 'o')
pl.xlim(f0*(2*np.pi) - 10/side_day, f0*(2*np.pi) + 10/side_day)

# %%
# This uses five_vec.py to compute the doppler modulation of the signal

d = 8 * kpc
h0 = 1 #4/d * (G*Mc/(c**2))**(5/3)*(np.pi*f/c)**(2/3)

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
sidereal_t = sidereal.gmst(t.mjd)
sidereal_t -= sidereal_t[0]

exp_terms = np.exp(1j*(np.arange(5)-2)[:, np.newaxis]*sidereal_t)
template_p = np.dot(sidereal.A_p, exp_terms)
template_c = np.dot(sidereal.A_c, exp_terms)
template_comb = np.dot(sidereal.A, exp_terms)

# %%
test = Resampler()
test.timeseries = template_comb
test.resampled_time = tau/(2*np.pi)
test.nufft()
print(test.extract_5vec(0))
print(sidereal.A)

# %%
resampler_data = Resampler()
resampler_data.timeseries = signal_det
resampler_data.resampled_time = tau/(2*np.pi)
resampler_data.nufft()

data_X = resampler_data.extract_5vec(f0*(2*np.pi))
# data_X = resampler_data.extract_5vec(resampler_data.freqs, resampler_data.weights_normalized, f0)

pl.plot(resampler_data.freqs, resampler_data.power_normalized, 'o')
pl.xlim(f0*(2*np.pi) - 10*(2*np.pi)/side_day, f0*(2*np.pi) + 10*(2*np.pi)/side_day)

# %%
resampler_sidereal = Resampler()
resampler_sidereal.timeseries = template_comb
resampler_sidereal.resampled_time = tau/(2*np.pi)
resampler_sidereal.nufft()

template_X = resampler_sidereal.extract_5vec(0)

pl.plot(resampler_sidereal.freqs, resampler_sidereal.power_normalized, 'o')
pl.xlim(-10*(2*np.pi)/side_day,10*(2*np.pi)/side_day)

# %%
print("h0 --------")
h_est = np.dot(data_X, np.conj(template_X))/np.sum(np.abs(template_X)**2)
print(h_est)
print(h0 * np.exp(1j*gamma))

# %%
resampler_sidereal = Resampler()
resampler_sidereal.timeseries = template_comb
resampler_sidereal.resampled_time = tau/(2*np.pi)
resampler_sidereal.nufft()

template_X = resampler_sidereal.extract_5vec(0)
print(template_X)

resampler_sidereal = Resampler()
resampler_sidereal.timeseries = template_comb
resampler_sidereal.resampled_time = t_offset
resampler_sidereal.nufft()

template_X = resampler_sidereal.extract_5vec(0)
print(template_X)

# %%
resampler_sidereal = Resampler()
resampler_sidereal.timeseries = template_p
resampler_sidereal.resampled_time = tau/(2*np.pi)
resampler_sidereal.nufft()

template_freqs = resampler_sidereal.freqs
template_weights = resampler_sidereal.weights_normalized
template_X = resampler_sidereal.extract_5vec(0)

print("hp --------")
hp_est = np.dot(data_X, np.conj(template_X))/np.sum(np.abs(template_X)**2)
print(hp_est)
print(h0*sidereal.H_p*np.exp(1j*gamma))

# %%
resampler_sidereal = Resampler()
resampler_sidereal.timeseries = template_c
resampler_sidereal.resampled_time = tau/(2*np.pi)
resampler_sidereal.nufft()

template_freqs = resampler_sidereal.freqs
template_weights = resampler_sidereal.weights_normalized
template_X = resampler_sidereal.extract_5vec(0)

print("hc --------")
hc_est = np.dot(data_X, np.conj(template_X))/np.sum(np.abs(template_X)**2)
print(hc_est)
print(h0*sidereal.H_c*np.exp(1j*gamma))

# %%
A = np.real(hp_est * np.conj(hc_est))
B = np.imag(hp_est * np.conj(hc_est))
C = np.abs(hp_est)**2 - np.abs(hc_est)**2

# %%
eta_est = (-1 + np.sqrt(1-4*B**2)) / (2*B)
psi_est = 1/4*np.arcsin(2*A / ((2*A)**2 + C**2))

# %%
print(eta_est)
print(params['eta'])

# %%
print(psi_est)
print(params['psi'])

# %%
Hp_est=np.sqrt(1/(1+eta_est**2))*(np.cos(2*psi_est)-1j*eta_est*np.sin(2*psi_est))
Hc_est=np.sqrt(1/(1+eta_est**2))*(np.sin(2*psi_est)+1j*eta_est*np.cos(2*psi_est))
print(np.angle(hp_est / Hp_est))
print(np.angle(hc_est / Hc_est))

# %%
