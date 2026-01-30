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
def nufft(beta, data, t, bin_no):
    bins = np.arange(bin_no) - bin_no//2

    tau = -3/5*(1-8/3*beta*t)**(5/8)/beta
    scale = (2*pi)/(tau[-1]-tau[0])
    tau *= scale
    start_diff = -pi-tau[0]
    tau += start_diff
    nufft_amp = finufft.nufft1d1(
        tau.astype(np.float64), data.astype(complex), bin_no, eps=1e-12, upsampfac=2.0)/len(data)
    nufft_freq = -bins*scale/(2*pi)
    nufft_power = np.abs(nufft_amp)**2
    return nufft_freq, nufft_amp


# %%
side_day = 86164.09053083288
n_days = 3
T_obs = n_days*side_day

f0 = 1/side_day * 10000
Mc = 3e-2 * 2e30
f_max = 4
beta = const*f0**(8/3)*Mc**(5/3)
f_signal = 4*f_max
nt = round(f_signal*T_obs)

ref_time = Time('2019-04-03')
t = ref_time.gps + np.arange(nt)/f_signal
t = Time(t, format="gps", scale="utc")
t_offset = t.gps-t.gps[0]

phi = -6*pi/5*f0*(1-8./3.*(beta)*t_offset)**(5/8)/beta
signal_source = 1*np.exp(-1j*phi)

# %%
resampler = Resampler()
resampler.timeseries = signal_source
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
signal_det = sidereal.amp_modulation * signal_source

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
resampler = Resampler()
resampler.timeseries = amp_modulation
resampler.resampled_time = -t.gps
resampler.nufft()
pl.plot(resampler.freqs/(2*np.pi), resampler.power_normalized, 'o')
pl.plot(fft_freqs, abs(fft_amps)**2, 'o')
pl.xlim(- 10/side_day, 10/side_day)

# %%
signal = amp_modulation
tau = t.gps
bin_no = len(tau) # make it an input

scale = abs((2*np.pi) / (tau[-1] - tau[0]))
tau_scaled = scale * (tau-tau[0]) - np.pi
bins = np.arange(bin_no) - bin_no//2
freqs = bins * scale

weights = finufft.nufft1d1(tau_scaled, signal.astype(complex), bin_no, isign=1)
test_freqs0 = freqs
test_weights0 = weights

# %%
scale

# %%
tau_scaled

# %%
signal = amp_modulation
tau = -t.gps
bin_no = len(tau) # make it an input

scale = abs((2*np.pi) / (tau[-1] - tau[0]))
tau_scaled = scale * abs((tau-tau[0])) - np.pi
bins = np.arange(bin_no) - bin_no//2
freqs = bins * scale

weights = finufft.nufft1d1(-tau_scaled, signal.astype(complex), bin_no)
test_freqs1 = freqs
test_weights1 = weights

# %%
scale

# %%
tau_scaled

# %%
pl.plot(test_freqs0, abs(test_weights0)**2)
pl.plot(test_freqs1, abs(test_weights1)**2)
pl.plot(fft_freqs*2*np.pi, abs(fft_amps*len(amp_modulation))**2)
pl.xlim(-50/side_day, 50/side_day)

# %%
test_weights0

# %%
test_weights1

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
nufft_h_est = np.dot(nufft_X, np.conj(A_recovery))/np.sum(np.abs(A_recovery)**2)

fft_X = np.empty((5), dtype=complex)
fft_X[0] = fft_amps[abs(fft_freqs-(-2/side_day)).argmin()]
fft_X[1] = fft_amps[abs(fft_freqs-(-1/side_day)).argmin()]
fft_X[2] = fft_amps[abs(fft_freqs).argmin()]
fft_X[3] = fft_amps[abs(fft_freqs-(+1/side_day)).argmin()]
fft_X[4] = fft_amps[abs(fft_freqs-(+2/side_day)).argmin()]
fft_h_est = np.dot(fft_X, np.conj(A_recovery))/np.sum(np.abs(A_recovery)**2)

print(f"NUFFT h_est={nufft_h_est}")
print(f"FFT h_est={fft_h_est}")
print(f"True h0={h0}")

# %%
fft_freqs[abs(fft_freqs-(-1/side_day)).argmin()]

# %%


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
fft_freqs[abs(fft_freqs-(-1/side_day)).argmin()]

# %%
resampler = Resampler()
resampler.timeseries = signal_det
resampler.resampled_time = t_offset
resampler.nufft()
pl.plot(resampler.freqs/(2*np.pi), resampler.power_normalized, 'o')
pl.xlim(f0 - 10/side_day, f0 + 10/side_day)

# %%
A_recovery

# %%
X = np.empty((5), dtype=complex)

X[0] = resampler.weights_normalized[abs(resampler.freqs/(2*np.pi)-(f0-2/side_day)).argmin()]
X[1] = resampler.weights_normalized[abs(resampler.freqs/(2*np.pi)-(f0-1/side_day)).argmin()]
X[2] = resampler.weights_normalized[abs(resampler.freqs/(2*np.pi)-f0).argmin()]
X[3] = resampler.weights_normalized[abs(resampler.freqs/(2*np.pi)-(f0+1/side_day)).argmin()]
X[4] = resampler.weights_normalized[abs(resampler.freqs/(2*np.pi)-(f0+2/side_day)).argmin()]

print(abs(np.dot(X, np.conj(A_recovery))/np.sum(np.abs(A_recovery)**2)))

# %%
h0

# %%
f = f_calc(f0, beta, t)
d = 8e3*3.086e+16 # 8kpc
h0 = 4/d*(G*Mc/c**2)**(5/3)*(pi*f/c)**(2/3)

# %%
params = dict(ra = 0.5, dec = 0.5, eta = 0.5, psi = 0.5, lat = 0.5, 
              lng = 0.5, az = 0.5, side_day=side_day)
sidereal = five_vec(**params)
sidereal.compute_H()
sidereal.compute_A(t)
sidereal.compute_5vec()
amp_modulation = sidereal.amp_modulation

# %%
signal = amp_modulation*freq_signal # * h0

fft_freqs, fft_amps = nufft(beta, signal, t, len(signal))
f0_idx = abs(fft_freqs-f0).argmin()

# %%
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

# %%
pl.plot(fft_freqs, abs(fft_amps)**2, 'o')
pl.xlim(f0-0.005, f0+0.005)
pl.axvline(fft_freqs[f0_idx-2*n_days], c='r', ls='--')
pl.axvline(fft_freqs[f0_idx-1*n_days], c='r', ls='--')
pl.axvline(fft_freqs[f0_idx], c='r', ls='--')
pl.axvline(fft_freqs[f0_idx+1*n_days], c='r', ls='--')
pl.axvline(fft_freqs[f0_idx+2*n_days], c='r', ls='--')
pl.xlabel("Frequency")
pl.ylabel("Power")
pl.title("Demonstration of the 5-vector statistic")
pl.tight_layout()

# %%
X = np.empty((5), dtype=complex)

f0_idx = abs(fft_freqs-f0).argmin()
X[0] = fft_amps[f0_idx-6]
X[1] = fft_amps[f0_idx-3]
X[2] = fft_amps[f0_idx]
X[3] = fft_amps[f0_idx+3]
X[4] = fft_amps[f0_idx+6]

# %%
estimator = abs(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2))
estimator

# %%
estimator

# %%
# pl.plot(t[::100], h0[::100])
pl.axhline(estimator, c='r', ls='--', label = 'Estimator')
pl.axhline(2.4306/pi, c='tab:orange', label="Theoretical avg FFT loss")
pl.legend()
pl.xlabel("Time")
pl.ylabel("h0")

# %%
2.4306/pi

# %% [markdown]
# ## Coverage test

# %%
c = 3e8
G = 6.67e-11
pi = np.pi
const = 96/5*pi**(8/3)*(G/c**3)**(5/3)

f_max = 200
side_day = 1e4/3 #86164.09053083288
n_days = 3
T_obs = n_days*side_day
f_signal = 3*f_max
nt = round(f_signal*T_obs)
t = (np.arange(nt)/f_signal).astype(np.longdouble)
d = 8e3*3.086e+16 # 8kpc

params = dict(ra = 0.5, dec = 0.5, eta = 0.5, psi = 0.5, lat = 0.5, 
              lng = 0.5, az = 0.5, side_day=side_day)
sidereal = five_vec(**params)
sidereal.compute_H()
sidereal.compute_A(t)
sidereal.compute_5vec()
amp_modulation = sidereal.amp_modulation

# %%
nufft_results = []
fivevec_results=[]
for i in range(5):
    print(i)
    f0 = np.linspace(20, 40, 10, endpoint=False)[i]
    Mc = 3e-2 * 2e30
    beta = const*f0**(8/3)*Mc**(5/3)

    phi = -6*pi/5*f0*(1-8./3.*(beta)*t)**(5/8)/beta
    freq_signal = 1*np.exp(1j*phi)
    temp_freqs, temp_amps = nufft(beta, freq_signal, t, len(freq_signal))
    nufft_results.append(max(abs(temp_amps)**2))
    
    f = f_calc(f0, beta, t)
    h0 = 4/d*(G*Mc/c**2)**(5/3)*(pi*f/c)**(2/3)
    signal = h0[0]*amp_modulation*freq_signal

    fft_freqs, fft_amps = nufft(beta, signal, t, len(signal))
    f0_idx = abs(fft_freqs-f0).argmin()
    X = np.empty((5), dtype=complex)
    X[0] = fft_amps[f0_idx-n_days*2]
    X[1] = fft_amps[f0_idx-n_days*1]
    X[2] = fft_amps[f0_idx]
    X[3] = fft_amps[f0_idx+n_days*1]
    X[4] = fft_amps[f0_idx+n_days*2]
    
    estimator = abs(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2))
    fivevec_results.append(estimator/h0[0])

# %%
pl.hist(nufft_results)
pl.axvline(2.4306/pi, c='r', ls='--')

# %%
pl.hist(np.multiply(2, fivevec_results))
pl.axvline(2.4306/pi, c='r', ls='--')

# %%
pl.hist(np.divide(fivevec_results,nufft_results))

# %% [markdown]
# ## Additional loss from amp-modulation

# %%
f0 = 20
Mc = 4e-2 * 2e30
beta = const*f0**(8/3)*Mc**(5/3)

f_max = 200
side_day = 1e4/3 #86164.09053083288
n_days = 3
T_obs = n_days*side_day
f_signal = 3*f_max
nt = round(f_signal*T_obs)
t = (np.arange(nt)/f_signal).astype(np.longdouble)
d = 8e3*3.086e+16 # 8kpc

# Amp modulation
params = dict(ra = 0.5, dec = 0.5, eta = 0.5, psi = 0.5, lat = 0.5, 
              lng = 0.5, az = 0.5, side_day=side_day)
sidereal = five_vec(**params)
sidereal.compute_H()
sidereal.compute_A(t)
sidereal.compute_5vec()
amp_modulation = sidereal.amp_modulation

# %%
phi = -6*pi/5*f0*(1-8./3.*(beta)*t)**(5/8)/beta
freq_signal = 1*np.exp(1j*phi)

#NUFFT
bin_no = len(freq_signal)
data=freq_signal
bins = np.arange(bin_no) - bin_no//2

tau = -3/5*(1-8/3*beta*t)**(5/8)/beta
scale = (2*pi)/(tau[-1]-tau[0])
tau *= scale
start_diff = -pi-tau[0]
tau += start_diff
temp_amp = finufft.nufft1d1(
    tau.astype(np.float64), data.astype(complex), bin_no, eps=1e-10, upsampfac=2.0)/len(data)
temp_freq = -bins*scale/(2*pi)

# amp modulation
f = f_calc(f0, beta, t)
h0 = 4/d*(G*Mc/c**2)**(5/3)*(pi*f/c)**(2/3)
signal = h0[0]*amp_modulation*freq_signal

# 5-vector
fft_freqs, fft_amps = nufft(beta, signal, t, len(signal))
f0_idx = abs(fft_freqs-f0).argmin()
X = np.empty((5), dtype=complex)
X[0] = fft_amps[f0_idx-n_days*2]
X[1] = fft_amps[f0_idx-n_days*1]
X[2] = fft_amps[f0_idx]
X[3] = fft_amps[f0_idx+n_days*1]
X[4] = fft_amps[f0_idx+n_days*2]

estimator = abs(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2))

mono = 1*np.exp(1j*(2*pi*f0*t+phi[0]))
modulated_mono = h0[0]*amp_modulation*mono

# %%
inverse = finufft.nufft1d2(tau.astype(np.float64), fft_amps, eps=1e-12, upsampfac=2.0)

# %%
# test = np.fft.ifft(fft_amps*len(signal))

# %%
pl.plot(t, np.abs(inverse)-np.abs(signal))
pl.xlim(t[100], t[-100])
pl.ylim(-0.25e-22, 0.25e-22)


# %%
def strobo(beta, data, f_ratio):
    new_t = -3/5*(1-8/3*beta*t)**(5/8)/beta
    f_new = f_signal/f_ratio
    
    new_t *= f_new #changes the frequency to the new downsampled frequency (approximately, not exactly)
    floor_t = np.floor(new_t) #For some reason floor works better than round. Take it to the nearest time index which are integers
    idx = np.nonzero(np.diff(floor_t)) #The step that downsamples
    resampled = data[idx]
    t_out = (new_t[idx])/f_new
    return (resampled, t_out)


# %%
f0 = 20
Mc = 4e-2 * 2e30
beta = const*f0**(8/3)*Mc**(5/3)

f_max = 200
n_days = 3
side_day = 1e4/n_days #86164.09053083288
T_obs = n_days*side_day
f_signal = 20*f_max
nt = round(f_signal*T_obs)
t = (np.arange(nt)/f_signal).astype(np.longdouble)
d = 8e3*3.086e+16 # 8kpc

# Amp modulation
params = dict(ra = 0.5, dec = 0.5, eta = 0.5, psi = 0.5, lat = 0.5, 
              lng = 0.5, az = 0.5, side_day=side_day)
sidereal = five_vec(**params)
sidereal.compute_H()
sidereal.compute_A(t)
sidereal.compute_5vec()
amp_modulation = sidereal.amp_modulation

# %%
phi = -6*pi/5*f0*(1-8./3.*(beta)*t)**(5/8)/beta
freq_signal = 1*np.exp(1j*phi)
# strobo_data, strobo_time = strobo(beta, freq_signal, f_ratio = 8)
# strobo_amp = np.fft.fftshift(np.fft.fft(strobo_data))
# strobo_freq = np.fft.fftshift(np.fft.fftfreq(len(strobo_data), np.diff(strobo_time)[0]))

# amp modulation
f = f_calc(f0, beta, t)
h0 = 4/d*(G*Mc/c**2)**(5/3)*(pi*f/c)**(2/3)
signal = h0[0]*amp_modulation*freq_signal

# # 5-vector
strobo_data, strobo_time = strobo(beta, signal, f_ratio = 8)
strobo_amp = np.fft.fftshift(np.fft.fft(strobo_data))/len(strobo_data)
strobo_freq = np.fft.fftshift(np.fft.fftfreq(len(strobo_data), np.diff(strobo_time)[0]))

f0_idx = abs(strobo_freq-f0).argmin()
X = np.empty((5), dtype=complex)
X[0] = strobo_amp[f0_idx-n_days*2]
X[1] = strobo_amp[f0_idx-n_days*1]
X[2] = strobo_amp[f0_idx]
X[3] = strobo_amp[f0_idx+n_days*1]
X[4] = strobo_amp[f0_idx+n_days*2]

estimator = abs(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2))

# mono = 1*np.exp(1j*(2*pi*f0*t+phi[0]))
# modulated_mono = h0[0]*amp_modulation*mono

# %%
# pl.plot(strobo_freq, abs(strobo_amp/len(strobo_data))**2, 'o')
# pl.axhline(2.4306/pi, c='r', ls='--')
# pl.xlim(f0-0.003, f0+0.003)

# %%
pl.plot(strobo_freq, abs(strobo_amp)**2, 'o')
pl.xlim(f0-0.005, f0+0.005)
pl.axvline(strobo_freq[f0_idx-2*n_days], c='r', ls='--')
pl.axvline(strobo_freq[f0_idx-1*n_days], c='r', ls='--')
pl.axvline(strobo_freq[f0_idx], c='r', ls='--')
pl.axvline(strobo_freq[f0_idx+1*n_days], c='r', ls='--')
pl.axvline(strobo_freq[f0_idx+2*n_days], c='r', ls='--')

# %%
estimator/h0[0]

# %%
