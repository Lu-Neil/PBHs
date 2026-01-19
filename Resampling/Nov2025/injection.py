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
import finufft
import matplotlib.pyplot as pl
from resampler import Resampler

pi = np.pi

# %% [markdown]
# ## beta

# %%
c = 3e8
G = 6.67e-11
pi = np.pi
const = 96/5*pi**(8/3)*(G/c**3)**(5/3)

T_obs = 3*86400 #2**15
f0 = 60000/T_obs
Mc = 1e-1* 2e30
f_max = 10
beta = const*f0**(8/3)*Mc**(5/3)
f_signal = 4*f_max
nt = round(f_signal*T_obs)
t = (np.arange(nt)/f_signal)

phi = 6*np.pi/5*f0*(1-8./3.*(beta)*t)**(5/8)/beta
tau = 6*np.pi/5*(1-8/3*beta*t)**(5/8)/beta

d = 10 * 3.086e+19 #10 kpc
f = f0*(1-8/3*beta*t)**(-3/8)
h = 4/d * (G*Mc/c**2)**(5/3)*(pi*f/c)**(2/3)

signal = h * np.real(np.exp(-1j*phi))

# %%
# 0.5 factor from np.real and 0.5 factor from nufft_real
0.25 * np.sum(abs(h)**2)

# %%
resampler = Resampler()
resampler.timeseries = signal
resampler.resampled_time = tau
resampler.nufft_real()
np.sum(resampler.power)/len(signal)

# %%
print(max(resampler.power_normalized))
window_offset_power = np.sinc(min(abs(resampler.freqs-f0))/np.diff(resampler.freqs)[0])**2
print(window_offset_power * np.sum(resampler.power_normalized))

# %%
min(abs(resampler.freqs-f0))

# %%
window_offset_power

# %%
pl.plot(resampler.freqs, resampler.power, 'o')
pl.xlim(f0-0.00005, f0+0.00005)

# %% [markdown]
# Here we have shown that: 
# 1. the total power of the signal is as we expect
# 2. The maximum power is the one we expected (due to distance from the bin central frequency)

# %%
t_prime = np.linspace(0, tau[-1], len(t))
resamp_test = Resampler()
resamp_test.timeseries = h*np.real(np.exp(-1j*f0*t_prime))
resamp_test.resampled_time = t_prime
resamp_test.nufft_real()
np.sum(resamp_test.power)/len(t_prime)

# %% [markdown]
# ## Antenna pattern

# %%
from five_vec import five_vec

# This uses five_vec.py to compute the doppler modulation of the signal
# params = dict(ra = 0.5, dec = 0.5, eta = 0.2, psi = 0.3, lat = 0.5, 
#               lng = 0.5, az = 0.5, side_day=side_day)
params = dict(ra = 198.783/180*np.pi, #np.random.uniform(0, 2*np.pi), 
              dec = 27.153/180*np.pi, #np.random.uniform(-np.pi/2, np.pi/2), 
              eta = 0.1, #np.random.uniform(-1, 1), 
              psi = 0.1, #np.random.uniform(0, 2*np.pi), 
              lat = 46.4550/180*np.pi, #np.random.uniform(-np.pi/2, np.pi/2), 
              lng = 240.5920/180*np.pi, #np.random.uniform(-np.pi, np.pi) 
              az = 144.0006/180*np.pi, #np.random.uniform(0, 2*np.pi),
             )
sidereal = five_vec(**params)
sidereal.compute_H()
sidereal.compute_A(t+1e6)
sidereal.compute_5vec()
amp_modulation = sidereal.amp_modulation

# %%
signal = np.real(amp_modulation * h * 1*np.exp(-1j*phi))
0.25*np.sum(abs(amp_modulation * h)**2)

# %%
resampler = Resampler()
resampler.timeseries = np.real(amp_modulation * h * 1*np.exp(-1j*phi))
resampler.resampled_time = tau
resampler.nufft_real()
np.sum(resampler.power)/len(signal)

# %%
P_expected = 0.25*np.sum(abs(amp_modulation * h)**2)
P_resampling = np.sum(resampler.power)/len(signal)
print(P_resampling / P_expected)

# %%
sidereal_fft_freqs = np.fft.fftshift(np.fft.fftfreq(len(amp_modulation), d=np.diff(t)[0]))
sidereal_fft_amps = np.fft.fftshift(np.fft.fft(amp_modulation))

# %%
# 1/2 factor from rfft
print(abs(sidereal_fft_amps[len(amp_modulation)//2])**2 / np.sum(abs(sidereal_fft_amps)**2))
print(resampler.power[np.argmin(abs(resampler.freqs-f0))] / (np.sum(resampler.power)))
print("The spectra are not the same")

# %%
print(max(resampler.power) / (np.sum(resampler.power)))

# %%
pl.plot(resampler.freqs, resampler.power, 'o')
pl.axvline(f0, ls='--', c='r')
pl.axvline(f0-2/86400, ls='--', c='r')
pl.axvline(f0-1/86400, ls='--', c='r')
pl.axvline(f0+1/86400, ls='--', c='r')
pl.axvline(f0+2/86400, ls='--', c='r')
pl.axvline(f0, ls='--', c='r')
pl.xlim(f0 - 0.001, f0+0.001)

# %%
pl.plot(sidereal_fft_freqs, abs(sidereal_fft_amps)**2, 'o')
pl.xlim(-30/sidereal.side_day, 30/sidereal.side_day)

# %%
resampler0 = Resampler()
resampler0.timeseries = np.real(h * 1*np.exp(-1j*phi))
resampler0.resampled_time = tau
resampler0.nufft_real()
resampler0.power[np.argmin(abs(resampler0.freqs-f0))]
print(f"Normalized power: {max(resampler0.power) / (np.sum(resampler0.power))}")

# %%
resampler1 = Resampler()
resampler1.timeseries = np.real(amp_modulation * h * 1*np.exp(-1j*phi))
resampler1.resampled_time = tau
resampler1.nufft_real()
print(f"Normalized power: {max(resampler1.power) / (np.sum(resampler1.power))}")

# %%
resampler2 = Resampler()
resampler2.timeseries = np.real(amp_modulation * h * 1*np.exp(-1j*phi))
resampler2.resampled_time = tau - tau[0] + np.unwrap(np.angle(amp_modulation))
resampler2.nufft_real()
print(f"Normalized power: {max(resampler2.power) / (np.sum(resampler2.power))}")

# %%
pl.plot(t, tau-tau[0])

# %%
pl.plot(t, np.unwrap(np.angle(amp_modulation)))

# %%
max(resampler1.power)

# %%
max(resampler2.power)

# %%
day = 86400

# %%
pl.plot(resampler1.freqs, resampler1.power, 'o', label='PBH')
pl.plot(resampler2.freqs, resampler2.power, 'o', label='PBH + sidereal')
pl.xlim(f0 - 5/day, f0+5/day)
pl.legend()

# %%
np.angle(amp_modulation)

# %%
pl.plot(amp_modulation)

# %%
pl.plot(np.exp(-1j*np.angle(amp_modulation)))

# %%
