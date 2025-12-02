import resource
import psutil
import numpy as np
import matplotlib.pyplot as pl
import matplotlib
import finufft
import time
import scipy.signal as ss
from scipy.interpolate import CubicSpline
from five_vec import five_vec
from scipy import integrate
tic = time.time()

red = '#D60606'
blue = '#0083DE'
green = '#00BA75'
yellow = '#FFC61E'
purple = '#A433B3'
orange = '#FD882E'
color_arr = [red, blue, green, yellow, purple, orange]

mplparams = {
    'text.usetex': True,
    'lines.linewidth': 1.5,
    'lines.markersize': 10,
    'axes.grid': False,
    'axes.labelweight': 'normal',
    'font.family': 'DejaVu Sans',
    'font.size': 36,
    'figure.figsize': (15, 10),
    'legend.fontsize': 26,
    'legend.handlelength': 2,
    'legend.numpoints': 1,
    'axes.grid': True,
    'grid.alpha': 0.9,
    'axes.prop_cycle': matplotlib.cycler(color=color_arr)
}
matplotlib.rcParams.update(mplparams)

free_mem = psutil.virtual_memory()[1]
resource.setrlimit(resource.RLIMIT_AS, (0.8*free_mem, 0.8*free_mem))


def f_calc(t, beta, f0):
    return f0*(1-8/3*beta*t)**(-3/8)


def df_calc(t, beta, f0):
    return f0*beta*(1-8/3*beta*t)**(-11/8)


def phi_calc(t, beta, f0):
    phi = -6*pi/5*f0*(1-8./3.*beta*t)**(5/8)/beta
#     phi = np.mod(phi,2*pi)
    return phi


def nufft(data, tau, bin_no):
    bins = np.arange(bin_no) - bin_no//2
    temp_tau = tau.copy()

    scale = (2*pi)/(tau[-1]-tau[0])
    temp_tau *= scale
    start_diff = -pi-tau[0]
    temp_tau += start_diff
#     plan.setpts(x=np.array(tau))
    nufft_freq = -bins*scale/(2*pi)
#     nufft_amp = plan.execute(np.array(data.astype(complex)))
    nufft_amp = finufft.nufft1d1(np.array(temp_tau), np.array(
        data.astype(complex)), bin_no)/len(data)
    nufft_power = np.abs(nufft_amp)**2

    sort_idx = np.argsort(nufft_power)
    thresh_idx = np.searchsorted(
        nufft_power, np.array([0.01]), sorter=sort_idx)[0]
    return 2*np.pi*nufft_freq, nufft_amp, nufft_power


c = 3e8
G = 6.67e-11
pi = np.pi
const = 96/5*pi**(8/3)*(G/c**3)**(5/3)

# pl.plot(t_space, f)
ephemeris = np.loadtxt("../doppler/earth00-40-DE440.dat",
                       delimiter='\t', skiprows=18)
[t_eph, pos_x, pos_y, pos_z, vel_x, vel_y,
    vel_z, acc_x, acc_y, acc_z] = ephemeris.T

pos_arr = np.array([pos_x, pos_y, pos_z])
interp_pos = CubicSpline(t_eph, pos_arr.T)
v_arr = np.array([vel_x, vel_y, vel_z])
interp_v = CubicSpline(t_eph, v_arr.T)
acc_arr = np.array([acc_x, acc_y, acc_z])
interp_acc = CubicSpline(t_eph, acc_arr.T)

H_lat = 119.41 * np.pi/180  # rad
H_lng = 46.45 * np.pi/180  # rad

params = dict(ra=0.5, dec=0.5, eta=0.5, psi=0.5, lat=H_lat,
              lng=H_lng, az=0.5)

n_hat = (np.cos(params['ra'])*np.cos(params['dec']),
         np.sin(params['ra'])*np.cos(params['dec']),
         np.sin(params['dec']))

# Using t=30 days for long enough to see Doppler effects
day = 86400
n_days = 30
duration = n_days*day
f_signal = 5
# very approximately 1st Jan 2024
start_time = t_eph[abs(t_eph-(t_eph[0]+24*365*day)).argmin()]
t_offset = np.arange(0, duration, 1/f_signal)
t_doppler = start_time + t_offset
samp_freq = 1/(np.diff(t_offset)[0])
print(f"Sampling frequency = {samp_freq}")

sidereal = five_vec(**params)
sidereal.compute_H()
sidereal.compute_A(t_offset)
sidereal.compute_5vec()
amp_modulation = sidereal.amp_modulation

f0 = 1.5
Mc = 1e-9 * 2e30
beta = const*f0**(8/3)*Mc**(5/3)
f_source = f_calc(t_offset, beta, f0)
phi_source = 2*np.pi * \
    integrate.cumulative_trapezoid(f_source, x=t_offset, initial=0)
signal_source = np.exp(1j*phi_source)
tau_source = 2*np.pi * \
    integrate.cumulative_trapezoid(f_source/f0, x=t_offset, initial=0)

f_doppler = f_source*(1+np.dot(interp_v(t_doppler), n_hat))
phi_doppler = 2*np.pi * \
    integrate.cumulative_trapezoid(f_doppler, x=t_offset, initial=0)
signal_doppler = np.exp(1j*phi_doppler)
tau_doppler = 2*np.pi * \
    integrate.cumulative_trapezoid(f_doppler/f0, x=t_offset, initial=0)
nufft_freq, nufft_amp, nufft_power = nufft(
    signal_doppler, tau_doppler, len(signal_doppler))

signal_sidereal = amp_modulation*signal_doppler
nufft_freq_sidereal, nufft_amp_sidereal, nufft_sidereal_power = nufft(
    signal_sidereal, tau_doppler, len(signal_doppler))

pl.semilogy(nufft_freq, nufft_power, label='PBH evolution + Doppler')
pl.semilogy(nufft_freq_sidereal, nufft_sidereal_power,
            label='PBH evolution + Doppler + Sidereal')
pl.legend()

f0_idx = abs(nufft_freq_sidereal-f0).argmin()
pl.axvline(nufft_freq_sidereal[f0_idx-2*n_days], c=green, ls='--')
pl.axvline(nufft_freq_sidereal[f0_idx-1*n_days], c=green, ls='--')
pl.axvline(nufft_freq_sidereal[f0_idx], c=green, ls='--')
pl.axvline(nufft_freq_sidereal[f0_idx+1*n_days], c=green, ls='--')
pl.axvline(nufft_freq_sidereal[f0_idx+2*n_days], c=green, ls='--')
pl.xlim(f0-0.0001, f0+0.0001)
pl.savefig("temp.png")

X = np.empty((5), dtype=complex)

f0_idx = abs(nufft_freq_sidereal-f0).argmin()
X[0] = nufft_amp_sidereal[f0_idx-2*n_days]
X[1] = nufft_amp_sidereal[f0_idx-1*n_days]
X[2] = nufft_amp_sidereal[f0_idx]
X[3] = nufft_amp_sidereal[f0_idx+1*n_days]
X[4] = nufft_amp_sidereal[f0_idx+2*n_days]
estimator = abs(np.dot(X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2))

breakpoint()


fft_amp_sidereal = np.fft.fftshift(np.fft.fft(
    amp_modulation*signal_source))/len(signal_source)
fft_power_sidereal = abs(fft_amp_sidereal)**2
pl.semilogy(fft_freq, fft_power)
pl.semilogy(fft_freq, fft_power_sidereal)


pl.xlim(0.9999, 1.0001)
pl.savefig("sidereal_modulation.png")


breakpoint()


print(f"Signal created in {(time.time()-tic)}")

tau = 2*np.pi*integrate.cumulative_trapezoid(f_doppler/f0, x=t_offset)
print(f"Integration in {(time.time()-tic)}")
nufft_freq, nufft_power = nufft(signal_doppler, tau, len(signal_doppler))
f0_found = nufft_freq[np.argmax(nufft_power)]*2*np.pi
print(f"Numerical doppler in {(time.time()-tic)}")

# If same beta but different f0, does resampling still work?


def beta_to_f0(beta, Mc):
    return const**(-3/8)*beta**(3/8)*Mc**(-5/8)


Mc_new = 2e-2*2e30
f0_new = beta_to_f0(beta, Mc_new)
f_source_new = f_calc(t_offset, beta, f0_new)
f_doppler_new = f_source_new*(1+np.dot(interp_v(t_doppler), n_hat))
phi_doppler_new = 2*np.pi * \
    integrate.cumulative_trapezoid(f_doppler_new, x=t_offset)
signal_doppler_new = np.exp(1j*phi_doppler_new)
nufft_freq_new, nufft_power_new = nufft(
    signal_doppler_new, tau, len(signal_doppler))
f0_found_new = nufft_freq_new[np.argmax(nufft_power_new)]*2*np.pi

tau_new = 2*np.pi * \
    integrate.cumulative_trapezoid(f_doppler_new/f0_new, x=t_offset)
nufft_freq_test, nufft_power_test = nufft(
    signal_doppler_new, tau_new, len(signal_doppler))
f0_found_test = nufft_freq_test[np.argmax(nufft_power_test)]*2*np.pi

print(f"Numerical doppler in {(time.time()-tic)}")
breakpoint()
pl.plot(nufft_freq, nufft_power, color=red)
pl.plot(nufft_freq_new, nufft_power_new, color=blue)
pl.axvline(f0/(2*np.pi), color=red, ls='--')
pl.axvline(f0_new/(2*np.pi), color=blue, ls='--')
pl.savefig("different_f0_same_beta.png")
pl.close()
print(f"Plotting {(time.time()-tic)}")
