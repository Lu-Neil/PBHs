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
import copy

pi = np.pi

# %% [markdown]
# ## 5 vector for a monochromatic signal

# %%
# We can define a shorter sidereal day to verify the code without requiring large memory usage
side_day = 86164.09053083288
number_of_days = 0.5 # needs to be integer
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
h0 = 1 #np.random.uniform(1, 5)
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

# %%
fft_freqs = np.fft.fftshift(np.fft.fftfreq(len(amp_modulation), d=np.diff(t.gps)[0]))
fft_amps = np.fft.fftshift(np.fft.fft(amp_modulation))/len(amp_modulation)

X = np.empty((5), dtype=complex)

X[0] = fft_amps[abs(fft_freqs-(-2/side_day)).argmin()]
X[1] = fft_amps[abs(fft_freqs-(-1/side_day)).argmin()]
X[2] = fft_amps[abs(fft_freqs).argmin()]
X[3] = fft_amps[abs(fft_freqs-(+1/side_day)).argmin()]
X[4] = fft_amps[abs(fft_freqs-(+2/side_day)).argmin()]

h_est = np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2)
hp_est = np.dot(X, np.conj(sidereal.A_p))/np.sum(np.abs(sidereal.A_p)**2)
hc_est = np.dot(X, np.conj(sidereal.A_c))/np.sum(np.abs(sidereal.A_c)**2)
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
mono = np.exp(1j*(omega0*t_space+gamma))
detector = amp_modulation * mono

# %%
fft_freqs = np.fft.fftshift(np.fft.fftfreq(len(detector), d=np.diff(t.gps)[0]))
fft_amps = np.fft.fftshift(np.fft.fft(detector))/len(detector)

X = np.empty((5), dtype=complex)

X[0] = fft_amps[abs(fft_freqs-(f0-2/side_day)).argmin()]
X[1] = fft_amps[abs(fft_freqs-(f0-1/side_day)).argmin()]
X[2] = fft_amps[abs(fft_freqs-f0).argmin()]
X[3] = fft_amps[abs(fft_freqs-(f0+1/side_day)).argmin()]
X[4] = fft_amps[abs(fft_freqs-(f0+2/side_day)).argmin()]

print(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2))
print(h0)
# print(abs(np.dot(X, np.conj(A_recovery))/np.sum(np.abs(A_recovery)**2)))

# %%
pl.plot(fft_freqs, abs(fft_amps)**2, 'o')
pl.xlim(f0 - 20/side_day, f0 + 20/side_day)

# %% [markdown]
# ## Using time domain MF

# %% [markdown]
# Solved, when duration is not an integer number of days, the time domain templates are not orthogonal and so there is mixing. 
#
# Should MCMC over eta and psi to maximise the computed h (which works correctly), need to take a bunch of vectors to account for the spreading, not just the 5

# %%
# template_p = np.zeros(len(t.mjd))
# template_c = np.zeros(len(t.mjd))
# for i in range(5):
#     template_p = template_p + sidereal.A_p[i] * np.exp(1j*(i-2)*sidereal.gmst(t.mjd))
#     template_c = template_c + sidereal.A_c[i] * np.exp(1j*(i-2)*sidereal.gmst(t.mjd))

# %%
not_sidereal_t = copy.deepcopy(sidereal.gmst(t.mjd))
not_sidereal_t -= not_sidereal_t[0]

exp_terms = np.exp(1j*(np.arange(5)-2)[:, np.newaxis]*not_sidereal_t)
template_p = np.dot(sidereal.A_p, exp_terms)
template_c = np.dot(sidereal.A_c, exp_terms)
template_comb = np.dot(sidereal.A, exp_terms)

# %%
template_freqs = np.fft.fftshift(np.fft.fftfreq(len(template_comb), d=np.diff(t.gps)[0]))
template_amps = np.fft.fftshift(np.fft.fft(template_comb))/len(template_p)
template_p_amps = np.fft.fftshift(np.fft.fft(template_p))/len(template_p)
template_c_amps = np.fft.fftshift(np.fft.fft(template_c))/len(template_c)

# %%
pl.plot(fft_freqs, abs(fft_amps)**2, 'o')
pl.xlim(f0 - 20/side_day, f0 + 20/side_day)

# %%
pl.plot(template_freqs, abs(template_amps)**2, 'o')
pl.xlim( - 20/side_day, + 20/side_day)

# %%
leakage_pad = 3 # maximum number of bins the power is expected to leak into
edge_offset = int(number_of_days * 2 + leakage_pad)

data_idx = int(abs(fft_freqs-f0).argmin())
data_manyVec = fft_amps[data_idx-edge_offset:data_idx+edge_offset]

template_idx = int(abs(template_freqs).argmin()) # == len(template_freqs)//2 + 1
template_manyVec = template_amps[template_idx-edge_offset:template_idx+edge_offset]

template_p_manyVec = template_p_amps[template_idx-edge_offset:template_idx+edge_offset]
template_c_manyVec = template_c_amps[template_idx-edge_offset:template_idx+edge_offset]

# %%
print(np.dot(data_manyVec, np.conj(template_manyVec))/np.sum(np.abs(template_manyVec)**2))
print(h0)

# %%
hc_est = np.dot(data_manyVec, np.conj(template_c_manyVec))/np.sum(np.abs(template_c_manyVec)**2)
print(hc_est)
print(h0 * sidereal.H_c)

# %%
hp_est = np.dot(data_manyVec, np.conj(template_p_manyVec))/np.sum(np.abs(template_p_manyVec)**2)
print(hp_est)
print(h0 * sidereal.H_p)

# %%
np.sqrt(hp_est**2 + hc_est**2)

# %%
print(np.sqrt(abs(hp_est)**2 + abs(hc_est)**2))
print(h0)

# %%
H_p=np.sqrt(1/(1+params['eta']**2))*(np.cos(2*params['psi'])-1j*params['eta']*np.sin(2*params['psi']))
H_c=np.sqrt(1/(1+params['eta']**2))*(np.sin(2*params['psi'])+1j*params['eta']*np.cos(2*params['psi']))
H0 = np.sqrt(H_p**2 + H_c**2)
H0


# %% [markdown]
# ## Sample over eta and psi

# %%
def check_eta_psi(eta, psi):
    H_p=np.sqrt(1/(1+eta**2))*(np.cos(2*psi)-1j*eta*np.sin(2*psi))
    H_c=np.sqrt(1/(1+eta**2))*(np.sin(2*psi)+1j*eta*np.cos(2*psi))
    test_template = H_p * template_p_manyVec + H_c * template_c_manyVec
    computed_h0 = np.dot(data_manyVec, np.conj(test_template))/np.sum(np.abs(test_template)**2)
    return computed_h0


# %%
eta_space = np.linspace(-1, 1, 200)
psi_space = np.linspace(0, 2*np.pi, 200)
param_grid = np.array([np.array([(i, j) for i in eta_space]) for j in psi_space])
h0_grid = np.array([np.array([check_eta_psi(i, j) for i in eta_space]) for j in psi_space])

temp_grid = np.real(h0_grid) / np.abs(h0_grid)
max_idx = np.unravel_index(np.argmax(temp_grid), np.shape(h0_grid))
psi_idx, eta_idx = max_idx
eta_max, psi_max = eta_space[eta_idx], psi_space[psi_idx]

print(f"Estimated eta, psi = {eta_max:.2f}, {psi_max:.2f}")
print(f"True eta, psi = {params['eta']:.2f}, {params['psi']:.2f}")

# %%
check_eta_psi(eta_max, psi_max)

# %%
check_eta_psi(params['eta'], params['psi'])

# %%
# Convert to array (important for plotting)
im = pl.imshow(
    np.abs(h0_grid),
    extent=[eta_space[0], eta_space[-1], psi_space[0], psi_space[-1]],
    origin='lower',
    aspect='auto'
)

pl.colorbar(im, label='h0 value')

pl.xlabel(r'$\eta$')
pl.ylabel(r'$\psi$')

# Mark maximum
# pl.scatter(eta_max, psi_max, color='red', marker='x', s=100, label='Maximum')
pl.scatter(params['eta'], params['psi'], color='red', marker='+', s=100, label='True')
pl.legend()

pl.tight_layout()

# %%
h0

# %%
print(eta_max)
print(params['eta'])

# %%
print(psi_max)
print(params['psi'])

# %%
print(psi_max - params['psi'])

# %%
np.diff(psi_space)[0]

# %%
check_eta_psi(params['eta'], params['psi'])

# %%
np.max(h0_grid)

# %%

# %%

# %%

# %% [markdown]
# ## Coverage test

# %%
# This uses five_vec.py to compute the doppler modulation of the signal
h0 = 1
params = dict(ra = np.random.uniform(0, 2*np.pi), 
              dec = np.random.uniform(-np.pi/2, np.pi/2), 
              eta = np.random.uniform(-1, 1), 
              psi = np.random.uniform(0, 2*np.pi), 
              lat = np.random.uniform(-np.pi/2, np.pi/2), 
              lng = np.random.uniform(-np.pi, np.pi),
              az = np.random.uniform(0, 2*np.pi), 
              side_day=side_day
             )

# %%
# We can define a shorter sidereal day to verify the code without requiring large memory usage
side_day = 86164.09053083288
number_of_days = 3 # needs to be integer
T_obs = number_of_days*side_day #need to ensure this is integer of t_side

ref_time = Time('2019-04-03')
f_signal = 8*f0
nt = round(f_signal*T_obs)
t = ref_time.gps + np.arange(nt)/f_signal
t = Time(t, format="gps", scale="utc")

sidereal = five_vec(**params)
sidereal.compute_H()
sidereal.compute_A(sidereal.gmst(t.mjd))
sidereal.compute_5vec()
amp_modulation = h0 * sidereal.amp_modulation

# %%
recovery_arr = []
distance_arr = []

for i in range(50):
    f0 = np.random.uniform(0.1, 0.2)
    omega0 = 2*np.pi*f0
    gamma = 0
        
    t_space = t.gps - t.gps[0]
    mono = np.exp(1j*(omega0*t_space+gamma))
    detector = amp_modulation * mono

    fft_freqs = np.fft.fftshift(np.fft.fftfreq(len(detector), d=np.diff(t.gps)[0]))
    fft_amps = np.fft.fftshift(np.fft.fft(detector))/len(detector)
    
    X = np.empty((5), dtype=complex)
    
    X[0] = fft_amps[abs(fft_freqs-(f0-2/side_day)).argmin()]
    X[1] = fft_amps[abs(fft_freqs-(f0-1/side_day)).argmin()]
    X[2] = fft_amps[abs(fft_freqs-f0).argmin()]
    X[3] = fft_amps[abs(fft_freqs-(f0+1/side_day)).argmin()]
    X[4] = fft_amps[abs(fft_freqs-(f0+2/side_day)).argmin()]

    recovery_arr.append(abs(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2)))
    distance_arr.append(min(abs(fft_freqs-f0))/np.diff(fft_freqs)[0])
# print(np.sinc(distance)**2)

# %%
pl.plot(distance_arr, recovery_arr, 'o', label="Numerically recovered power")
pl.plot(distance_arr, np.sinc(distance_arr)**2, 'o', label="Theoretically predicted recovery")
pl.xlabel("Distance from bin centre")
pl.ylabel("Normalized power")
pl.legend()
pl.title("PBH signal")
pl.grid(True)

# %%
pl.plot(t.mjd, sidereal.amp_modulation)

# %%
pl.plot(distance_arr, recovery_arr / np.sinc(distance_arr)**2, 'o')

# %%
