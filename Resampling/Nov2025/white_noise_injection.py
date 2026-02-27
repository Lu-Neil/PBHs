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
import scipy.signal as ss

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

ref_time = Time('2019-04-03')
t = ref_time.gps + np.arange(nt)/f_signal
t = Time(t, format="gps", scale="utc")
t_offset = t.gps-t.gps[0]

phi = -6*pi/5*f0*(1-8./3.*(beta)*t_offset)**(5/8)/beta
signal_source = np.real(1*np.exp(1j*phi))

# %%
normalized_power = 0.01 * len(t_offset)

noise = np.random.normal(0, np.sqrt(normalized_power), len(t_offset))
noise_amps = np.fft.fftshift(np.fft.fft(noise))
noise_freqs = np.fft.fftshift(np.fft.fftfreq(len(noise), np.diff(t_offset)[0]))
pl.plot(noise_freqs, abs(noise_amps/len(noise))**2)
pl.xlim(-0.001, 0.001)

# %%
resampler = Resampler()
resampler.timeseries = signal_source+noise
resampler.resampled_time = t_offset
resampler.nufft()
pl.plot(resampler.freqs, resampler.power_normalized, 'o')
pl.xlim(f0*(2*np.pi) - 20/side_day, f0*(2*np.pi) + 20/side_day)


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
signal_det = amp_modulation * signal_source + noise

# %%
# sidereal_phase = gmst(t.mjd[0]) + params['ra'] - params['lng']
# A_recovery = sidereal.A.copy()
# Ap_recovery = sidereal.A_p.copy()
# Ac_recovery = sidereal.A_c.copy()

# for i in range(5):
#     A_recovery[i] = sidereal.A[i] * np.exp(1j*(i-2)*sidereal_phase)
#     Ap_recovery[i] = sidereal.A_p[i] * np.exp(1j*(i-2)*sidereal_phase)
#     Ac_recovery[i] = sidereal.A_c[i] * np.exp(1j*(i-2)*sidereal_phase)

# %%
resampler = Resampler()
resampler.timeseries = amp_modulation
resampler.resampled_time = t.gps
resampler.nufft()
pl.plot(resampler.freqs/(2*np.pi), resampler.power_normalized, 'o')
# pl.plot(fft_freqs, abs(fft_amps)**2, 'o')
pl.xlim(- 10/side_day, 10/side_day)

# %%
nufft_freqs = resampler.freqs/(2*np.pi) #np.fft.fftshift(np.fft.fftfreq(len(amp_modulation), d=np.diff(t.gps)[0]))
nufft_amps = resampler.weights_normalized #np.fft.fftshift(np.fft.fft(amp_modulation))/len(amp_modulation)
fft_freqs = np.fft.fftshift(np.fft.fftfreq(len(amp_modulation), d=np.diff(t.gps)[0]))
fft_amps = np.fft.fftshift(np.fft.fft(amp_modulation))/len(amp_modulation)

nufft_X = np.empty((5), dtype=complex)
nufft_X[0] = nufft_amps[abs(nufft_freqs-(-2/side_day)).argmin()]
nufft_X[1] = nufft_amps[abs(nufft_freqs-(-1/side_day)).argmin()]
nufft_X[2] = nufft_amps[abs(nufft_freqs).argmin()]
nufft_X[3] = nufft_amps[abs(nufft_freqs-(+1/side_day)).argmin()]
nufft_X[4] = nufft_amps[abs(nufft_freqs-(+2/side_day)).argmin()]
nufft_h_est = np.dot(nufft_X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2)

fft_X = np.empty((5), dtype=complex)
fft_X[0] = fft_amps[abs(fft_freqs-(-2/side_day)).argmin()]
fft_X[1] = fft_amps[abs(fft_freqs-(-1/side_day)).argmin()]
fft_X[2] = fft_amps[abs(fft_freqs).argmin()]
fft_X[3] = fft_amps[abs(fft_freqs-(+1/side_day)).argmin()]
fft_X[4] = fft_amps[abs(fft_freqs-(+2/side_day)).argmin()]
fft_h_est = np.dot(fft_X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2)

print(f"NUFFT h_est={nufft_h_est}")
print(f"FFT h_est={fft_h_est}")
print(f"True h0={h0}")

# %%
nufft_hp_est = np.dot(nufft_X, np.conj(sidereal.A_p))/np.sum(np.abs(sidereal.A_p)**2)
nufft_hc_est = np.dot(nufft_X, np.conj(sidereal.A_c))/np.sum(np.abs(sidereal.A_c)**2)

print(f"NUFFT hp_est={nufft_hp_est}")
print(f"True hp={h0 * sidereal.H_p}")
print(f"NUFFT hc_est={nufft_hc_est}")
print(f"True hc={h0 * sidereal.H_c}")

# %%
resampler = Resampler()
resampler.timeseries = signal_det
resampler.resampled_time = t_offset
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
print(h0)

# %%
