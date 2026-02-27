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

def extract_5vec(freqs, weights, f0):
    X = np.empty((5), dtype=complex)
    X[0] = weights[abs(freqs-(f0-2/side_day)).argmin()]
    X[1] = weights[abs(freqs-(f0-1/side_day)).argmin()]
    X[2] = weights[abs(freqs-f0).argmin()]
    X[3] = weights[abs(freqs-(f0+1/side_day)).argmin()]
    X[4] = weights[abs(freqs-(f0+2/side_day)).argmin()]
    return X


# %%
import numpy as np
import finufft
import matplotlib.pyplot as plt

def test_phase_warping():
    # 1. Setup Parameters
    M = 50000             # Number of samples
    t = np.sort(np.random.uniform(0, 10, M))  # Non-uniform time samples
    phi_0 = np.pi / 3     # Constant phase offset we want to recover (60 deg)
    
    # 2. Define a Non-Monochromatic Signal (Quadratic Chirp)
    # Phase = a*t^2 + b*t
    a, b = 0.5, 2.0
    inst_phase = a * t**2 + b * t 
    signal = np.exp(-1j * (inst_phase + phi_0))

    # 3. Resample using FINUFFT
    # We treat the 'inst_phase' itself as our non-uniform spatial coordinate x.
    # Because signal = exp(i * (x + phi_0)), it should peak at mode k=1.
    
    # FINUFFT expects x in [-pi, pi]. We normalize our phase to this range.
    # Note: This mapping is what "makes it monochromatic"
    x_coords = (inst_phase % (2 * np.pi)) - np.pi
    
    N = 100  # Number of modes to look at
    f = finufft.nufft1d1(x_coords, signal, N, isign=-1)
    
    # 4. Analyze results
    # The signal exp(i*x) corresponds to mode k=1.
    # Since FINUFFT centers modes at [-N/2, N/2-1], index for k=1 is:
    idx_k1 = N // 2 + 1
    
    recovered_val = f[idx_k1]
    recovered_phase = np.angle(recovered_val)
    
    print(f"--- Phase Recovery via Warping ---")
    print(f"Initial Phase offset (phi_0): {phi_0:.5f}")
    print(f"Recovered Phase at k=1:       {recovered_phase:.5f}")
    print(f"Error:                        {np.abs(phi_0 - recovered_phase):.2e}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Time domain (Real part shows the chirp)
    ax1.plot(t[:500], signal.real[:500])
    ax1.set_title("Original Signal (Time Domain)")
    ax1.set_xlabel("Time (t)")

    # Fourier domain (Phase-warped)
    modes = np.arange(-N//2, N//2)
    ax2.stem(modes, np.abs(f))
    ax2.set_title("Warped Spectrum (Phase Domain)")
    ax2.set_xlabel("Mode (k)")
    plt.show()

test_phase_warping()

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
gamma = np.mod(phi[0], 2*np.pi)
f = f_calc(f0, beta, t_offset)
signal_source = 1*np.exp(1j*phi)
tau = phi 

# %%
import numpy as np
import finufft
import matplotlib.pyplot as plt
from astropy.time import Time

# 1. Constants & Signal Generation (Same as your snippet)
c, G, pi = 3e8, 6.67e-11, np.pi
const = 96/5*pi**(8/3)*(G/c**3)**(5/3)
side_day = 86164.09053083288
T_obs = 0.5 * side_day # Half day for example

f0_injected = 1/side_day * 10000  # The value we want to recover
Mc = 1e-1 * 2e30
beta = const * f0_injected**(8/3) * Mc**(5/3)

f_signal = 16.0 # Sampling rate
nt = round(f_signal * T_obs) + 1
t_offset = np.arange(nt) / f_signal

# The modulated phase
phi = -6*pi/5*f0_injected*(1-8./3.*(beta)*t_offset)**(5/8)/beta
signal_source = np.exp(1j * phi)

# 2. Define Warped Time Coordinate
# We assume we know the "shape" of the evolution but want to verify f0.
# We define tau such that if f0 is correct, the signal is exp(i * 2pi * f0 * tau)
tau = phi / (2 * pi * f0_injected)

# 3. Resample using FINUFFT
# We map tau to the range [-pi, pi] for the NUFFT
tau_min, tau_max = tau.min(), tau.max()
tau_range = tau_max - tau_min
x_coords = 2 * pi * (tau - tau_min) / tau_range - pi

# Number of modes to check. We look for a peak near f0 * tau_range
N = int(f0_injected * tau_range * 2) 
f_nufft = finufft.nufft1d1(x_coords, signal_source, N, isign=-1)

# 4. Recover f0 from the spectrum
modes = np.arange(-N//2, N//2)
magnitudes = np.abs(f_nufft)
peak_idx = np.argmax(magnitudes)
peak_mode = modes[peak_idx]

# Convert the mode back to a frequency in Hz
# The mode 'k' corresponds to frequency: f = k / tau_range
f0_recovered = peak_mode / tau_range

print(f"--- Frequency Recovery Test ---")
print(f"Injected f0:  {f0_injected:.10f} Hz")
print(f"Recovered f0: {f0_recovered:.10f} Hz")
print(f"Difference:   {np.abs(f0_injected - f0_recovered):.2e} Hz")

# 5. Check Phase at the Peak
recovered_phase = np.angle(f_nufft[peak_idx])
print(f"Phase at Peak: {recovered_phase:.5f} rad")

# Visualization

plt.figure(figsize=(10, 4))
plt.plot(modes / tau_range, magnitudes, label="NUFFT Magnitude")
plt.axvline(f0_injected, color='r', linestyle='--', alpha=0.5, label="Injected f0")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Strength")
plt.title("Recovering Injected f0 in Warped Time Space")
plt.legend()
plt.show()

# %%
# 1. Define tau correctly
tau = phi / (2 * np.pi * f0_injected)
tau_0 = tau[0]
tau_relative = tau - tau_0
T_span = tau_relative[-1] - tau_relative[0]

# 2. Perform NUFFT on tau_relative mapped to [-pi, pi]
x_coords = 2 * np.pi * (tau_relative / T_span)
f_nufft = finufft.nufft1d1(x_coords, signal_source, N, isign=-1)

# 3. Calculate Expected Phase
# The NUFFT is essentially: Sum( exp(i*phi) * exp(-i*k*x) )
# At the peak (k_peak), the phase is approximately:
# Phase_measured = -(phi_at_center_of_tau) 

# Find the peak
peak_idx = np.argmax(np.abs(f_nufft))
k_peak = modes[peak_idx]

# Map the phase back:
# Since x = 0 is the middle of our signal, we check phi at that midpoint
mid_idx = len(phi) // 2
expected_phase = phi[mid_idx] # Negative because isign=-1

# Normalize to [-pi, pi]
expected_phase = np.angle(np.exp(1j * expected_phase))
measured_phase = np.angle(f_nufft[peak_idx])

print(f"Residual Error: {np.abs(np.angle(np.exp(1j*(measured_phase - expected_phase)))):.4e}")

# %%
x_coords

# %%
resampler = Resampler()
resampler.timeseries = signal_source
resampler.resampled_time = tau
resampler.nufft()
pl.plot(resampler.freqs, resampler.power_normalized, 'o')
# pl.xlim(1*(2*np.pi) - 20/side_day, 1*(2*np.pi) + 20/side_day)

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
resampler_data = Resampler()
resampler_data.timeseries = signal_det
resampler_data.resampled_time = tau
resampler_data.nufft()
pl.plot(resampler_data.freqs/(2*np.pi), resampler_data.power_normalized, 'o')
pl.xlim(f0 - 10/side_day, f0 + 10/side_day)

# %%
# resampler_data = Resampler()
# resampler_data.timeseries = h0*sidereal.amp_modulation
# resampler_data.resampled_time = tau
# resampler_data.nufft()
# pl.plot(resampler_data.freqs/(2*np.pi), resampler_data.power_normalized, 'o')
# pl.xlim(f0 - 10/side_day, f0 + 10/side_day)

# %%
data_freqs = resampler_data.freqs/(2*np.pi)
data_weights = resampler_data.weights_normalized
data_X = extract_5vec(data_freqs, data_weights, 0)

# %%
sidereal_t = sidereal.gmst(t.mjd)
sidereal_t -= sidereal_t[0]

exp_terms = np.exp(1j*(np.arange(5)-2)[:, np.newaxis]*sidereal_t)
template_p = np.dot(sidereal.A_p, exp_terms)
template_c = np.dot(sidereal.A_c, exp_terms)
template_comb = np.dot(sidereal.A, exp_terms)

# %%
resampler_sidereal = Resampler()
resampler_sidereal.timeseries = template_comb
resampler_sidereal.resampled_time = tau
resampler_sidereal.nufft()

# %%
pl.plot(resampler_sidereal.freqs/(2*np.pi), resampler_sidereal.power_normalized, 'o')
pl.xlim(-10/side_day,10/side_day)

# %%
template_freqs = resampler_sidereal.freqs/(2*np.pi)
template_weights = resampler_sidereal.weights_normalized
template_X = extract_5vec(template_freqs, template_weights, 0)

print("h0 --------")
h_est = np.dot(data_X, np.conj(template_X))/np.sum(np.abs(template_X)**2)
print(h_est)
print(h0[0])

# %%
resampler_sidereal = Resampler()
resampler_sidereal.timeseries = template_p
resampler_sidereal.resampled_time = tau
resampler_sidereal.nufft()

template_freqs = resampler_sidereal.freqs/(2*np.pi)
template_weights = resampler_sidereal.weights_normalized

template_X = extract_5vec(template_freqs, template_weights, 0)

print("hp --------")
hp_est = np.dot(data_X, np.conj(template_X))/np.sum(np.abs(template_X)**2)
print(hp_est)
print(h0[0]*sidereal.H_p)

# %%

X[0] = template_weights[abs(template_freqs-(f0-2/self.side_day)).argmin()]
X[1] = template_weights[abs(template_freqs-(f0-1/self.side_day)).argmin()]
X[2] = template_weights[abs(template_freqs-f0).argmin()]
X[3] = template_weights[abs(template_freqs-(f0+1/self.side_day)).argmin()]
X[4] = template_weights[abs(template_freqs-(f0+2/self.side_day)).argmin()]

# %%
resampler_sidereal = Resampler()
resampler_sidereal.timeseries = template_c
resampler_sidereal.resampled_time = tau
resampler_sidereal.nufft()

template_freqs = resampler_sidereal.freqs/(2*np.pi)
template_weights = resampler_sidereal.weights_normalized
template_idx = int(abs(template_freqs).argmin())
template_manyVec = template_weights[template_idx-edge_offset:template_idx+edge_offset]

print("hc --------")
hc_est = abs(np.dot(data_manyVec, np.conj(template_manyVec))/np.sum(np.abs(template_manyVec)**2))
print(hc_est)
print(h0[0]*sidereal.H_c)

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
