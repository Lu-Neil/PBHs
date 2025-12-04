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
from resampler import Resampler

# %% [markdown]
# ## 5 vector for a monochromatic signal

# %%
pi = np.pi
f0 = 2
# We define a shorter sidereal day to verify the code without requiring large memory usage
side_day = 2**10 #86164.09053083288
number_of_days = 2 # needs to be integer
T_obs = number_of_days*side_day #need to ensure this is integer of t_side

f_signal = 8*f0
nt = round(f_signal*T_obs)
t = np.arange(nt)/f_signal

#Monochromatic signal
mono = np.exp(1j*2*pi*f0*t)

# %%
mono_freqs = np.fft.fftshift(np.fft.fftfreq(len(mono), np.diff(t)[0]))
mono_amps = np.fft.fftshift(np.fft.fft(mono))/len(mono)

# %%
pl.plot(mono_freqs, abs(mono_amps)**2, 'o')
pl.xlim(f0-0.1, f0+0.1)

# %%
# This uses five_vec.py to compute the doppler modulation of the signal
# params = dict(ra = 0.5, dec = 0.5, eta = 0.2, psi = 0.3, lat = 0.5, 
#               lng = 0.5, az = 0.5, side_day=side_day)
params = dict(ra = 0.5, #np.random.uniform(0, 2*np.pi), 
              dec = 0.5, #np.random.uniform(-np.pi/2, np.pi/2), 
              eta = 0.1, #np.random.uniform(-1, 1), 
              psi = 0.1, #np.random.uniform(0, 2*np.pi), 
              lat = 46.4550/180*np.pi, #np.random.uniform(-np.pi/2, np.pi/2), 
              lng = 240.5920/180*np.pi, #np.random.uniform(-np.pi, np.pi) 
              az = 144.0006/180*np.pi, #np.random.uniform(0, 2*np.pi), 
              side_day=side_day
             )
sidereal = five_vec(**params)
sidereal.compute_H()
sidereal.compute_A(t)
sidereal.compute_5vec()
amp_modulation = sidereal.amp_modulation

# %%
params['az']

# %%
signal = mono*amp_modulation
fft_freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), np.diff(t)[0]))
fft_amps = np.fft.fftshift(np.fft.fft(signal))/len(signal)

# %%
pl.plot(signal)

# %%
# We verify that the modulated waveform has the distinctive five peaks expected
expected_f = fft_freqs[abs(fft_freqs-f0).argmin()]
pl.plot(fft_freqs, abs(fft_amps)**2, 'o')
pl.axvline(expected_f-2/side_day, c='r', ls='--')
pl.axvline(expected_f-1/side_day, c='r', ls='--')
pl.axvline(expected_f, c='r', ls='--')
pl.axvline(expected_f+1/side_day, c='r', ls='--')
pl.axvline(expected_f+2/side_day, c='r', ls='--')
pl.xlim(expected_f-10/side_day, expected_f+10/side_day)

# %%
X = np.empty((5), dtype=complex)

f0_idx = abs(fft_freqs-f0).argmin()
X[0] = fft_amps[f0_idx-2*number_of_days]
X[1] = fft_amps[f0_idx-number_of_days]
X[2] = fft_amps[f0_idx]
X[3] = fft_amps[f0_idx+number_of_days]
X[4] = fft_amps[f0_idx+2*number_of_days]

# %%
# Here we compute the five vector statistic and check how much power it recovers
estimator = abs(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2))
print(estimator)

# %%
results_arr = []
distance_arr = []

for i in range(100):
    f0 = np.random.uniform(1, 2)    
    mono = np.exp(1j*2*pi*f0*t)
    params = dict(ra = 0.5, #np.random.uniform(0, 2*np.pi), 
                  dec = 0.5, #np.random.uniform(-np.pi/2, np.pi/2), 
                  eta = 0.5, #np.random.uniform(-1, 1), 
                  psi = 0.5, #np.random.uniform(0, 2*np.pi), 
                  lat = 46.4550/180*np.pi, #np.random.uniform(-np.pi/2, np.pi/2), 
                  lng = 240.5920/180*np.pi, #np.random.uniform(-np.pi, np.pi)
                  az = 144.0006/180*np.pi, #np.random.uniform(0, 2*np.pi), 
                  side_day = side_day
                 )
    """       
    name: 'ligoh'
    lat: 46.4550
    long: 240.5920
    azim: 144.0006
    height: 142.5000"""
    sidereal = five_vec(**params)
    sidereal.compute_H()
    sidereal.compute_A(t)
    sidereal.compute_5vec()
    amp_modulation = sidereal.amp_modulation

    
    signal = mono*amp_modulation
    
    fft_freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), np.diff(t)[0]))
    fft_amps = np.fft.fftshift(np.fft.fft(signal))/len(signal)
    f0_idx = abs(fft_freqs-f0).argmin()
    
    X = np.empty((5), dtype=complex)
    X[0] = fft_amps[f0_idx-2*number_of_days]
    X[1] = fft_amps[f0_idx-number_of_days]
    X[2] = fft_amps[f0_idx]
    X[3] = fft_amps[f0_idx+number_of_days]
    X[4] = fft_amps[f0_idx+2*number_of_days]
    
    estimator = abs(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2))
    distance = min(abs(fft_freqs-f0))/np.diff(fft_freqs)[0]
    results_arr.append(estimator)
    distance_arr.append(distance)

# %%
pl.plot(distance_arr, results_arr, 'o')
pl.plot(distance_arr, np.sinc(distance_arr)**2, 'o')

# %%
f0 = np.random.uniform(1, 2)    
mono = np.exp(1j*2*pi*f0*t)
params = dict(ra = 0.5, #np.random.uniform(0, 2*np.pi), 
              dec = 0.5, #np.random.uniform(-np.pi/2, np.pi/2), 
              eta = 0.5, #np.random.uniform(-1, 1), 
              psi = 0.5, #np.random.uniform(0, 2*np.pi), 
              lat = 46.4550/180*np.pi, #np.random.uniform(-np.pi/2, np.pi/2), 
              lng = 240.5920/180*np.pi, #np.random.uniform(-np.pi, np.pi)
              az = 144.0006/180*np.pi, #np.random.uniform(0, 2*np.pi), 
             )
"""       
name: 'ligoh'
lat: 46.4550
long: 240.5920
azim: 144.0006
height: 142.5000"""
sidereal = five_vec(**params)
sidereal.compute_H()
sidereal.compute_A(t)
sidereal.compute_5vec()
amp_modulation = sidereal.amp_modulation
eta = params['eta']
cos_iota = (-1+np.sqrt(1-eta**2))/(eta)
H0 = np.sqrt((1+6*cos_iota**2+cos_iota**4)/4)

signal = H0*mono*amp_modulation

fft_freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), np.diff(t)[0]))
fft_amps = np.fft.fftshift(np.fft.fft(signal))/len(signal)
f0_idx = abs(fft_freqs-f0).argmin()

X = np.empty((5), dtype=complex)
X[0] = fft_amps[f0_idx-2*number_of_days]
X[1] = fft_amps[f0_idx-number_of_days]
X[2] = fft_amps[f0_idx]
X[3] = fft_amps[f0_idx+number_of_days]
X[4] = fft_amps[f0_idx+2*number_of_days]

estimator = abs(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2))
distance = min(abs(fft_freqs-f0))/np.diff(fft_freqs)[0]

# %%
estimator

# %%
pl.plot(abs(sidereal.A_c)**2)

# %%
print(resampler.freqs[f0_idx-2*number_of_days])
print(expected_f-2*np.pi*2/side_day)

# %%
expected_f = resampler.freqs[f0_idx]
pl.plot(resampler.freqs, resampler.power_normalized, 'o')
pl.axvline(expected_f-2*np.pi*2/side_day, c='r', ls='--')
pl.axvline(expected_f-2*np.pi*1/side_day, c='r', ls='--')
pl.axvline(expected_f, c='r', ls='--')
pl.axvline(expected_f+2*np.pi*1/side_day, c='r', ls='--')
pl.axvline(expected_f+2*np.pi*2/side_day, c='r', ls='--')
pl.xlim(expected_f-50/side_day, expected_f+50/side_day)

# %%
recovery_arr = []
distance_arr = []

for i in range(100):
    f0 = np.random.uniform(1, 2) # rad/s
    phi = f0*t
    signal = np.exp(-1j*phi) # amp_modulation * 
    
    resampler = Resampler()
    resampler.timeseries = signal
    resampler.resampled_time = t
    resampler.nufft()

    X = np.empty((5), dtype=complex)
    f0_idx = abs(resampler.freqs-f0).argmin()
    X[0] = resampler.weights_normalized[f0_idx-2*number_of_days]
    X[1] = resampler.weights_normalized[f0_idx-number_of_days]
    X[2] = resampler.weights_normalized[f0_idx]
    X[3] = resampler.weights_normalized[f0_idx+number_of_days]
    X[4] = resampler.weights_normalized[f0_idx+2*number_of_days]

    estimator = abs(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2))
    
    recovery_arr.append(max(resampler.power_normalized))
    distance_arr.append(min(abs(resampler.freqs-f0))/np.diff(resampler.freqs)[0])

# %%
pl.plot(distance_arr, recovery_arr, 'o')
pl.plot(distance_arr, np.sinc(distance_arr)**2, 'o')

# %% [markdown]
# ## PBH signal + amplitude modulation

# %%
from resampler import Resampler

# %%
c = 3e8
G = 6.67e-11
pi = np.pi
const = 96/5*pi**(8/3)*(G/c**3)**(5/3)

f0 = 2**3
Mc = 3e-2* 2e30
f_max = 2**6
T_obs = 2**10
beta = const*f0**(8/3)*Mc**(5/3)
f_signal = 4*f_max
side_day = 2**10 #86164.09053083288
number_of_days = 2 # needs to be integer
T_obs = number_of_days*side_day #need to ensure this is integer of t_side

f_signal = 8*f0
nt = round(f_signal*T_obs)
t = np.arange(nt)/f_signal

phi = 6*np.pi/5*f0*(1-8./3.*(beta)*t)**(5/8)/beta
tau = 6*np.pi/5*(1-8/3*beta*t)**(5/8)/beta
signal = 1*np.exp(-1j*phi)

# This uses five_vec.py to compute the doppler modulation of the signal
params = dict(ra = 0.5, dec = 0.5, eta = 0.5, psi = 0.5, lat = 0.5, 
              lng = 0.5, az = 0.5, side_day=side_day)
sidereal = five_vec(**params)
sidereal.compute_H()
sidereal.compute_A(t)
sidereal.compute_5vec()
amp_modulation = sidereal.amp_modulation
signal *= amp_modulation

# %%
resampler = Resampler()
resampler.timeseries = signal
resampler.resampled_time = tau
resampler.nufft()
pl.plot(resampler.freqs, resampler.power_normalized, 'o')
pl.xlim(f0 - 0.05, f0 + 0.05)

# %%
X = np.empty((5), dtype=complex)

f0_idx = abs(resampler.freqs-f0).argmin()
X[0] = resampler.weights_normalized[f0_idx-2*number_of_days]
X[1] = resampler.weights_normalized[f0_idx-number_of_days]
X[2] = resampler.weights_normalized[f0_idx]
X[3] = resampler.weights_normalized[f0_idx+number_of_days]
X[4] = resampler.weights_normalized[f0_idx+2*number_of_days]

# %%
estimator = abs(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2))
print(estimator)

# %%
