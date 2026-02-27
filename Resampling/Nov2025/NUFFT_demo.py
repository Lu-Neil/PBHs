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
# ## Monochromatic signal

# %%
t = np.linspace(4*pi,6*pi, int(2**16))
f0 = 8 # rad/s
phi = f0*t
signal = 1*np.exp(-1j*phi)

# %%
fft_freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), d=np.diff(t)[0]))
fft_amps = np.fft.fftshift(np.fft.fft(signal))/len(signal)

# %%
resampler = Resampler()
resampler.timeseries = signal
resampler.resampled_time = t
resampler.nufft()
pl.plot(resampler.freqs, np.imag(resampler.weights_normalized - fft_amps), 'o')
# pl.plot(fft_freqs*2*np.pi, abs(fft_amps)**2, 'o')
# pl.xlim(f0 - 10, f0 + 10)

# %%
pl.plot(resampler.freqs, np.imag(resampler.weights_normalized), 'o')
# pl.plot(resampler.freqs, np.imag(fft_amps), 'o')

# %%
np.allclose(resampler.power_normalized, abs(fft_amps)**2, atol=1e-5)

# %%
resampler.nufft()
pl.plot(resampler.freqs, resampler.power_normalized, 'o')
pl.xlim(f0 - 10, f0 + 10)

# %% [markdown]
# ## With $\dot{f}$

# %%
t = np.linspace(0*pi,16*pi, 2**14)
f0 = int(2*t[-1])/t[-1] # rad/s
fdot = 0.1 # rad/s^2 
phi = f0*t+fdot*t**2
tau = phi/f0
signal = 1*np.exp(1j*phi)

# %%
resampler = Resampler()
resampler.timeseries = signal
resampler.resampled_time = tau
resampler.nufft()
pl.plot(resampler.freqs, resampler.power_normalized, 'o')
pl.xlim(f0 - 1, f0 + 1)

# %% [markdown]
# ## beta

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
nt = round(f_signal*T_obs)
t = (np.arange(nt)/f_signal)

phi = 6*np.pi/5*f0*(1-8./3.*(beta)*t)**(5/8)/beta
tau = phi/f0
signal = 1*np.exp(1j*phi)

# %%
resampler = Resampler()
resampler.timeseries = signal
resampler.resampled_time = tau
resampler.nufft()
pl.plot(resampler.freqs, resampler.power_normalized, 'o')
pl.xlim(f0 - 0.05, f0 + 0.05)

# %%
# # %timeit resampler.nufft()

# %%
# # %timeit resampler.nufft()

# %% [markdown]
# ## Coverage test

# %%
import scipy.signal as ss

# %%
recovery_arr = []
distance_arr = []
t = np.linspace(0, 2**8, int(2**14), endpoint=False)

for i in range(10):
    f0 = np.random.uniform(1, 2) # rad/s
    phi = -f0*t
    signal = 1*np.exp(1j*phi)
    
    resampler = Resampler()
    resampler.timeseries = signal
    resampler.resampled_time = t
    resampler.nufft()
    recovery_arr.append(max(resampler.power_normalized))
    distance_arr.append(min(abs(resampler.freqs-f0))/np.diff(resampler.freqs)[0])

# %%
pl.plot(distance_arr, recovery_arr, 'o')
pl.plot(distance_arr, np.sinc(distance_arr)**2, 'o')
pl.title("Monochromatic signal")

# %%
recovery_arr = []
distance_arr = []
t = np.linspace(0, 2**8, int(2**14), endpoint=False)

for i in range(10):
    f0 = np.random.uniform(1, 2) 
    df0 = 10**np.random.uniform(-3, -1)
    phi = f0*t+df0*t**2
    tau = t*(1+df0/f0*t)
    signal = 1*np.exp(-1j*phi)
    
    resampler = Resampler()
    resampler.timeseries = signal
    resampler.resampled_time = tau
    resampler.nufft()
    recovery_arr.append(max(resampler.power_normalized))
    distance_arr.append(min(abs(resampler.freqs-f0))/np.diff(resampler.freqs)[0])

# %%
pl.plot(distance_arr, recovery_arr, 'o', label="Numerically recovered power")
pl.plot(distance_arr, np.sinc(distance_arr)**2, 'o', label="Theoretically predicted recovery")
pl.xlabel("Distance from bin centre")
pl.ylabel("Normalized power")
pl.legend()
pl.title("$\dot{f}$ signal")

# %%
f0 = np.random.uniform(1, 4)
Mc = 3e-2* 2e30
f_max = 2**6
T_obs = 2**10
beta = const*f0**(8/3)*Mc**(5/3)
f_signal = 4*f_max
nt = round(f_signal*T_obs)
t = (np.arange(nt)/f_signal)

phi = -6*np.pi/5*f0*(1-8./3.*(beta)*t)**(5/8)/beta
tau = -6*np.pi/5*(1-8/3*beta*t)**(5/8)/beta
signal = np.real(1*np.exp(1j*phi))

resampler = Resampler()
resampler.timeseries = signal
resampler.resampled_time = tau
resampler.nufft()
pl.plot(resampler.freqs, resampler.power_normalized, 'o')
resampler.nufft()
pl.plot(resampler.freqs, resampler.power_normalized, 'o')
pl.xlim(f0 - 0.05, f0 + 0.05)

print(max(resampler.power_normalized))
distance = min(abs(resampler.freqs-f0))/np.diff(resampler.freqs)[0]
print(np.sinc(distance)**2)

# %%
recovery_arr = []
distance_arr = []

for i in range(10):
    f0 = np.random.uniform(1, 4)
    Mc = 10**np.random.uniform(-3, -2) * 2e30
    beta = const*f0**(8/3)*Mc**(5/3)
    
    phi = -6*np.pi/5*f0*(1-8./3.*(beta)*t)**(5/8)/beta
    tau = -6*np.pi/5*(1-8/3*beta*t)**(5/8)/beta
    signal = 1*np.exp(1j*phi)

    resampler = Resampler()
    resampler.timeseries = signal
    resampler.resampled_time = tau
    resampler.nufft()
    recovery_arr.append(max(resampler.power_normalized))
    distance_arr.append(min(abs(resampler.freqs-f0))/np.diff(resampler.freqs)[0])
# print(np.sinc(distance)**2)

# %%
pl.plot(distance_arr, recovery_arr, 'o', label="Numerically recovered power")
pl.plot(distance_arr, np.sinc(distance_arr)**2, 'o', label="Theoretically predicted recovery")
pl.xlabel("Distance from bin centre")
pl.ylabel("Normalized power")
pl.legend()
pl.title("PBH signal")

# %%
