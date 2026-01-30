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
from astropy.time import Time

pi = np.pi


# %% [markdown]
# ## 5 vector for a monochromatic signal

# %%
def gmst(t):
    """
    %GMST  Greenwich mean sidereal time (in rad)
    %
    %   t   time (in JD or mjd)
    %
    % add longitude in hours (deg/15) to have local sidereal time 
    Adapted from Snag v2.0 by Sergio Frasca
    """
    t = np.asarray(t)
    jd = np.where(t > 1000000, t + 2400000.5, t)
    
    jd0=np.floor(jd-0.5)+0.5;
    h=(jd-jd0)*24;
    
    d=jd-2451545;
    d0=jd0-2451545;
    T=d/36525;
    
    st=np.mod(6.697374558+0.06570982441908*d0+1.00273790935*h+0.000026*T**2,24)
    return st/12*np.pi


# %%
# We can define a shorter sidereal day to verify the code without requiring large memory usage
side_day = 86164.09053083288
number_of_days = 3 # needs to be integer
T_obs = number_of_days*side_day #need to ensure this is integer of t_side

f0 = 1/side_day * 10000
omega0 = 2*np.pi*f0
gamma = 0

ref_time = Time('2019-04-03')
f_signal = 8*f0
nt = round(f_signal*T_obs)
t = ref_time.gps + np.arange(nt)/f_signal
t = Time(t, format="gps", scale="utc")

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
sidereal.compute_A(gmst(t.mjd))
sidereal.compute_5vec()
amp_modulation = h0 * sidereal.amp_modulation

# %%
phi = gmst(t.mjd[0]) + params['ra'] - params['lng']
A_recovery = sidereal.A.copy()
Ap_recovery = sidereal.A_p.copy()
Ac_recovery = sidereal.A_c.copy()

for i in range(5):
    A_recovery[i] = sidereal.A[i] * np.exp(1j*(i-2)*phi)
    Ap_recovery[i] = sidereal.A_p[i] * np.exp(1j*(i-2)*phi)
    Ac_recovery[i] = sidereal.A_c[i] * np.exp(1j*(i-2)*phi)

# %%
fft_freqs = np.fft.fftshift(np.fft.fftfreq(len(amp_modulation), d=np.diff(t.gps)[0]))
fft_amps = np.fft.fftshift(np.fft.fft(amp_modulation))/len(amp_modulation)

X = np.empty((5), dtype=complex)

X[0] = fft_amps[abs(fft_freqs-(-2/side_day)).argmin()]
X[1] = fft_amps[abs(fft_freqs-(-1/side_day)).argmin()]
X[2] = fft_amps[abs(fft_freqs).argmin()]
X[3] = fft_amps[abs(fft_freqs-(+1/side_day)).argmin()]
X[4] = fft_amps[abs(fft_freqs-(+2/side_day)).argmin()]

h_est = np.dot(X, np.conj(A_recovery))/np.sum(np.abs(A_recovery)**2)
hp_est = np.dot(X, np.conj(Ap_recovery))/np.sum(np.abs(Ap_recovery)**2)
hc_est = np.dot(X, np.conj(Ac_recovery))/np.sum(np.abs(Ac_recovery)**2)
print(h_est)
print(h0)

# %%
print(hp_est)
print(h0 * sidereal.H_p)
print(hc_est)
print(h0 * sidereal.H_c)

# %%
A = np.real(hp_est/h_est * np.conj(hc_est/h_est))
B = np.imag(hp_est/h_est * np.conj(hc_est/h_est))
C = np.abs(hp_est/h_est)**2 - np.abs(hc_est/h_est)**2

# %%
eta_est = (-1 + np.sqrt(1-4*B**2)) / (2*B)
psi_est = np.arcsin(np.real(hc_est * np.sqrt(1+eta_est**2)))/2 # annoying to check because of quadrature

# %%
print(eta_est)
print(params['eta'])

# %%
t_space = t.gps - t.gps[0]
mono = 4*np.exp(1j*(omega0*t_space+gamma))
detector = sidereal.amp_modulation * mono

# %%
# fft_freqs = np.fft.fftshift(np.fft.fftfreq(len(mono), d=np.diff(t.gps)[0]))
# fft_amps = np.fft.fftshift(np.fft.fft(mono))/len(mono)

# pl.semilogy(fft_freqs, abs(fft_amps)**2, 'o')
# pl.xlim(f0 - 20/side_day, f0 + 20/side_day)

# %%
fft_freqs = np.fft.fftshift(np.fft.fftfreq(len(detector), d=np.diff(t.gps)[0]))
fft_amps = np.fft.fftshift(np.fft.fft(detector))/len(detector)

X = np.empty((5), dtype=complex)

X[0] = fft_amps[abs(fft_freqs-(f0-2/side_day)).argmin()]
X[1] = fft_amps[abs(fft_freqs-(f0-1/side_day)).argmin()]
X[2] = fft_amps[abs(fft_freqs-f0).argmin()]
X[3] = fft_amps[abs(fft_freqs-(f0+1/side_day)).argmin()]
X[4] = fft_amps[abs(fft_freqs-(f0+2/side_day)).argmin()]

print(abs(np.dot(X, np.conj(A_recovery))/np.sum(np.abs(A_recovery)**2)))

# %%

# %% [markdown]
# ## Recovering injection

# %%
