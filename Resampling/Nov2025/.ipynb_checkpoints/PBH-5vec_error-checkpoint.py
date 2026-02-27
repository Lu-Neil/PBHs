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
recovery_arr = []
distance_arr = []

for i in range(20):
    f0 = np.random.uniform(0.01, 0.02)
    beta = const*f0**(8/3)*Mc**(5/3)
    
    phi = -6*pi/5*f0*(1-8./3.*(beta)*t_offset)**(5/8)/beta
    tau = phi/f0
    signal_source = 1*np.exp(1j*phi)
    signal_det = amp_modulation * signal_source

    resampler = Resampler()
    resampler.timeseries = signal_det #window
    resampler.resampled_time = tau
    resampler.nufft()

    X = np.empty((5), dtype=complex)
    X[0] = resampler.weights_normalized[abs(resampler.freqs-(f0-2/side_day)).argmin()]
    X[1] = resampler.weights_normalized[abs(resampler.freqs-(f0-1/side_day)).argmin()]
    X[2] = resampler.weights_normalized[abs(resampler.freqs-f0).argmin()]
    X[3] = resampler.weights_normalized[abs(resampler.freqs-(f0+1/side_day)).argmin()]
    X[4] = resampler.weights_normalized[abs(resampler.freqs-(f0+2/side_day)).argmin()]

    recovery_arr.append(abs(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2))/h0)
    distance_arr.append(min(abs(resampler.freqs-f0))/np.diff((resampler.freqs))[0])

# %%
pl.plot(distance_arr, recovery_arr, 'o', label='Numerical')
pl.plot(distance_arr, np.sinc(distance_arr)**2, 'o', label='Expected')
pl.ylabel("Power recovered")
pl.xlabel("Offset from bin center")
pl.legend()
pl.grid(True)

# %%
pl.plot(distance_arr, recovery_arr / np.sinc(distance_arr)**2, 'o')

# %%
