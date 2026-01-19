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
number_of_days = 2 # needs to be integer
T_obs = number_of_days*side_day #need to ensure this is integer of t_side

f0 = 1/side_day * 100000
omega0 = 2*np.pi*f0
gamma = 0

ref_time = Time('2019-04-03')
f_signal = 8*f0
nt = round(f_signal*T_obs)
t = ref_time.gps + np.arange(nt)/f_signal
t = Time(t, format="gps", scale="utc")
t_space = t.gps - t.gps[0]
monochromatic = 2*np.exp(1j*(omega0*t_space+gamma))

# %%
# This uses five_vec.py to compute the doppler modulation of the signal
params = dict(ra = 1.783725740253688e+02/180*np.pi, #np.random.uniform(0, 2*np.pi), 
              dec = -33.436602425196504/180*np.pi, #np.random.uniform(-np.pi/2, np.pi/2), 
              eta = 0.16, #np.random.uniform(-1, 1), 
              psi = 25.4390/180*np.pi, #np.random.uniform(0, 2*np.pi), 
              lat = 46.4550/180*np.pi, #np.random.uniform(-np.pi/2, np.pi/2), 
              lng = 3, #240.5920/180*np.pi, #np.random.uniform(-np.pi, np.pi) 
              az = 144.0006/180*np.pi, #np.random.uniform(0, 2*np.pi), 
              side_day=side_day
             )
sidereal = five_vec(**params)
sidereal.compute_H()
sidereal.compute_A(gmst(t.mjd))
sidereal.compute_5vec()
amp_modulation = sidereal.amp_modulation

# %%
nsid = 50000
N = nt
st = np.arange(0,nsid)*2*pi/nsid

stsub=gmst(t.mjd)
stsub = stsub  + params['ra'] - params['lng']
isub = np.mod(np.round(stsub*(nsid-1)/24),nsid-1)
isub = isub.astype(int)
temp_isub = np.searchsorted(st, gmst(t.mjd), side='left')

# %%
side_p = 0
side_c = 0
for i in range(5):
    side_p += sidereal.A_p[i]*np.exp(1j*(i-2)*(st + params['ra'] - params['lng']))
    side_c += sidereal.A_c[i]*np.exp(1j*(i-2)*(st + params['ra'] - params['lng']))

reconstruct_amp_mod = sidereal.H_p * side_p + sidereal.H_c * side_c

# %%
test_Ap = sidereal.A_p.copy()
test_Ac = sidereal.A_c.copy()
for i in range(5):
    test_Ap[i] = sidereal.A_p[i]*np.exp(1j*(i-2)*(params['ra'] - params['lng']))
    test_Ac[i] = sidereal.A_c[i]*np.exp(1j*(i-2)*(params['ra'] - params['lng']))

test_A = sidereal.H_p*test_Ap + sidereal.H_c*test_Ac

# %%
pl.plot(np.real(reconstruct_amp_mod[temp_isub-1]))
pl.plot(sidereal.amp_modulation)

# %%
fft_freqs = np.fft.fftshift(np.fft.fftfreq(len(sidereal.amp_modulation), d=np.diff(t.gps)[0]))
fft_amps = np.fft.fftshift(np.fft.fft(sidereal.amp_modulation))/len(sidereal.amp_modulation)

X = np.empty((5), dtype=complex)

X[0] = fft_amps[abs(fft_freqs-(-2/side_day)).argmin()]
X[1] = fft_amps[abs(fft_freqs-(-1/side_day)).argmin()]
X[2] = fft_amps[abs(fft_freqs).argmin()]
X[3] = fft_amps[abs(fft_freqs-(+1/side_day)).argmin()]
X[4] = fft_amps[abs(fft_freqs-(+2/side_day)).argmin()]

print(abs(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2)))

# pl.plot(fft_freqs, abs(fft_amps)**2, 'o')
# pl.axvline(2/side_day, c='r', ls='--')
# pl.axvline(1/side_day, c='r', ls='--')
# pl.axvline(0, c='r', ls='--')
# pl.axvline(-1/side_day, c='r', ls='--')
# pl.axvline(-2/side_day, c='r', ls='--')
# pl.xlim(-10/side_day, 10/side_day)

# %%
def compute_5comp_num(times, data, f_ref):
    f_side = 1/side_day
    freqs = f_ref + np.arange(-2, 3)*f_side
    dt = np.diff(times)[0]
    A = np.zeros(5, dtype=complex)
    
    for i in range(5):
        A[i] = np.sum(data * np.exp(-1j*2*np.pi*freqs[i]*times))*dt
    return A


# %%
X

# %%
np.sum(sidereal.amp_modulation * np.exp(-1j*2*np.pi*0*t.gps))*np.diff(t.gps)[0] / len(sidereal.amp_modulation)

# %%
compute_5comp_num(t.gps, sidereal.amp_modulation, 0) / len(sidereal.amp_modulation)

# %%
sidereal.A

# %% [markdown]
# ## With signal

# %%
gmst(t.mjd)

# %%
print(monochromatic)

# %%
mono_freqs = np.fft.fftshift(np.fft.fftfreq(len(t_space), np.diff(t_space)[0]))
mono_power = abs(np.fft.fftshift(np.fft.fft(monochromatic)/len(monochromatic)))**2
pl.semilogy(mono_freqs, mono_power, 'o')
pl.axvline(f0, c='r', ls='--')
pl.xlim(f0-0.003, f0+0.003)

# %%
gmst(t.mjd)

# %%
params = dict(ra = 5, #rad
            dec = 0.5, #rad
            eta = 0.5, #[-1,1]
            psi = 0.5, #rad
            
            lat = 0.5, #rad
            lng = 5, #rad
            az = 0.5,  #rad
              side_day=side_day
             )

sidereal = five_vec(**params)
sidereal.compute_H()
sidereal.compute_A(gmst(t.mjd))
sidereal.compute_5vec()

detector = sidereal.amp_modulation*monochromatic
det_freqs = np.fft.fftshift(np.fft.fftfreq(len(t_space), np.diff(t_space)[0]))
det_power = abs(np.fft.fftshift(np.fft.fft(detector)/len(detector)))**2
det_weights = np.fft.fftshift(np.fft.fft(detector)/len(detector))
pl.plot(det_freqs, det_power, 'o')
pl.axvline(f0-2/side_day, c='r', ls='--')
pl.axvline(f0-1/side_day, c='r', ls='--')
pl.axvline(f0, c='r', ls='--')
pl.axvline(f0+1/side_day, c='r', ls='--')
pl.axvline(f0+2/side_day, c='r', ls='--')
pl.xlim(f0-10/side_day, f0+10/side_day)

# %%
1/side_day/3 - np.diff(det_freqs)[0]

# %%

# %%
f0 - det_freqs[abs(det_freqs-f0).argmin()]

# %%
X = np.empty((5), dtype=complex)

X[0] = det_weights[abs(det_freqs-(f0-2/side_day)).argmin()]
X[1] = det_weights[abs(det_freqs-(f0-1/side_day)).argmin()]
X[2] = det_weights[abs(det_freqs-f0).argmin()]
X[3] = det_weights[abs(det_freqs-(f0+1/side_day)).argmin()]
X[4] = det_weights[abs(det_freqs-(f0+2/side_day)).argmin()]
abs(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2))

# %%
X

# %%
sidereal.A

# %%
side_day= 86164.09053083288
f0 = 1/side_day * 100000
omega0 = 2*np.pi*f0
gamma = 0
side_omega = 2*np.pi/side_day
t_space = np.arange(0, 2*side_day, 1/(8*f0))
monochromatic = 2*np.exp(1j*(omega0*t_space+gamma))

mono_freqs = np.fft.fftshift(np.fft.fftfreq(len(t_space), np.diff(t_space)[0]))
mono_power = abs(np.fft.fftshift(np.fft.fft(monochromatic)/len(monochromatic)))**2
pl.semilogy(mono_freqs, mono_power, 'o')
pl.axvline(f0, c='r', ls='--')
pl.xlim(f0-0.003, f0+0.003)

# %%
