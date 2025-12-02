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
    nufft_freq = 2*np.pi*bins/bin_no

    temp_tau = tau.copy()
    scale = (nufft_freq[-1]-nufft_freq[0])/(temp_tau[-1]-temp_tau[0])
    temp_tau *= scale
    start_diff = -pi-temp_tau[0]
    temp_tau += start_diff
#     plan.setpts(x=np.array(tau))
#     nufft_amp = plan.execute(np.array(data.astype(complex)))
    nufft_freq = bins*scale
    nufft_amp = finufft.nufft1d1(np.array(temp_tau), np.array(
        data.astype(complex)), bin_no, eps=1e-12, upsampfac=2.0)/len(data)
    nufft_amp = np.flip(nufft_amp)
    nufft_amp = np.roll(nufft_amp, 1)
    nufft_power = np.abs(nufft_amp)**2
    return nufft_freq, nufft_amp, nufft_power


def naive_fft(signal, time):
    fft_freq = np.fft.fftshift(np.fft.fftfreq(len(signal), np.diff(time)[0]))
    fft_amp = np.fft.fftshift(np.fft.fft(signal))/len(signal)
    fft_power = abs(fft_amp)**2
    return fft_freq, fft_amp, fft_power


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
n_days = 20
duration = n_days*day
f_signal = 5
# very approximately 1st Jan 2024
start_time = t_eph[abs(t_eph-(t_eph[0]+24*365*day)).argmin()]
t_offset = np.arange(0, duration, 1/f_signal)
t_doppler = start_time + t_offset
samp_freq = 1/(np.diff(t_offset)[0])
print(f"Sampling frequency = {samp_freq}")

f0 = 0.45
Mc = 1e-1 * 2e30
beta = const*f0**(8/3)*Mc**(5/3)
f_source = f_calc(t_offset, beta, f0)
# np.full(len(t_offset), f0, dtype=np.float64)
phi_source = 2*np.pi * \
    integrate.cumulative_trapezoid(f_source, x=t_offset, initial=0)
signal_source = np.exp(1j*phi_source)
tau_source = 2*np.pi * \
    integrate.cumulative_trapezoid(f_source/f0, x=t_offset, initial=0)

nufft_freq_source, nufft_amp_source, nufft_power_source = nufft(
    signal_source, tau_source, len(signal_source))
max_freq_source = nufft_freq_source[np.argmax(nufft_power_source)]
max_power_source = np.max(nufft_power_source)
pl.semilogy(nufft_freq_source, nufft_power_source,
            label=f'Mono \n Max f0 = {max_freq_source:.2f} \n Max power = {max_power_source:.2f}')
pl.legend()
pl.savefig("PBH_only.png")


f_doppler = f_source * (1+np.dot(interp_v(t_doppler), n_hat))
phi_doppler = 2*np.pi * \
    integrate.cumulative_trapezoid(f_doppler, x=t_offset, initial=0)
signal_doppler = np.exp(1j*phi_doppler)
tau_doppler = 2*np.pi * \
    integrate.cumulative_trapezoid(f_doppler/f0, x=t_offset, initial=0)

nufft_freq_doppler, nufft_amp_doppler, nufft_power_doppler = nufft(
    signal_doppler, tau_doppler, len(signal_doppler))
max_freq_doppler = nufft_freq_doppler[np.argmax(nufft_power_doppler)]
max_power_doppler = np.max(nufft_power_doppler)
pl.semilogy(nufft_freq_doppler, nufft_power_doppler,
            label=f'Mono + Doppler \n Max f0 = {max_freq_doppler:.2f} \n Max power = {max_power_source:.2f}')
pl.legend()
pl.savefig("PBH+doppler.png")
pl.close()

sidereal = five_vec(**params)
sidereal.compute_H()
sidereal.compute_A(t_offset)
sidereal.compute_5vec()
amp_modulation = sidereal.amp_modulation

signal_sidereal = amp_modulation*signal_doppler
nufft_freq_sidereal, nufft_amp_sidereal, nufft_power_sidereal = nufft(
    signal_sidereal, tau_doppler, len(signal_doppler))

nufft_f0_idx = abs(nufft_freq_sidereal-f0).argmin()
nufft_X = np.empty((5), dtype=complex)
nufft_X[0] = nufft_amp_sidereal[nufft_f0_idx-2*n_days]
nufft_X[1] = nufft_amp_sidereal[nufft_f0_idx-1*n_days]
nufft_X[2] = nufft_amp_sidereal[nufft_f0_idx]
nufft_X[3] = nufft_amp_sidereal[nufft_f0_idx+1*n_days]
nufft_X[4] = nufft_amp_sidereal[nufft_f0_idx+2*n_days]
estimator_sidereal = abs(
    np.dot(nufft_X, np.conj(sidereal.A))/np.sum(np.abs(sidereal.A)**2))

pl.semilogy(nufft_freq_sidereal, nufft_power_sidereal,
            label=f'Mono + Doppler + Sidereal \n 5 vector estimator = {estimator_sidereal:.2f}')
pl.legend(loc='lower center')
pl.axvline(nufft_freq_sidereal[nufft_f0_idx-2*n_days], c=orange)
pl.axvline(nufft_freq_sidereal[nufft_f0_idx-1*n_days], c=orange)
pl.axvline(nufft_freq_sidereal[nufft_f0_idx], c=orange)
pl.axvline(nufft_freq_sidereal[nufft_f0_idx+1*n_days], c=orange)
pl.axvline(nufft_freq_sidereal[nufft_f0_idx+2*n_days], c=orange)
pl.xlim(f0-0.0001, f0+0.0001)
pl.savefig("PBH+Dopppler+sidereal.png")
pl.close()


# fft_freqs = np.fft.fftshift(np.fft.fftfreq(
#     len(signal_sidereal), np.diff(t_offset)[0]))
# fft_amps = np.fft.fftshift(np.fft.fft(signal_sidereal))/len(signal_sidereal)
# fft_f0_idx = abs(fft_freqs-f0).argmin()
# expected_f = fft_freqs[fft_f0_idx]
# fft_X = np.empty((5), dtype=complex)
# fft_X[0] = fft_amps[fft_f0_idx-2*n_days]
# fft_X[1] = fft_amps[fft_f0_idx-n_days]
# fft_X[2] = fft_amps[fft_f0_idx]
# fft_X[3] = fft_amps[fft_f0_idx+n_days]
# fft_X[4] = fft_amps[fft_f0_idx+2*n_days]
# estimator = abs(np.dot(fft_X, np.conj(sidereal.A)) /
#                 np.sum(np.abs(sidereal.A)**2))

# pl.semilogy(fft_freqs, abs(fft_amps)**2, 'o',
#             label=f'FFT, estimator={estimator:.2f}')
# pl.xlim(f0-0.00003, f0+0.00003)
# pl.axvline(expected_f-2*n_days, c='r', ls='--')
# pl.axvline(expected_f-1*n_days, c='r', ls='--')
# pl.axvline(expected_f, c='r', ls='--')
# pl.axvline(expected_f+1*n_days, c='r', ls='--')
# pl.axvline(expected_f+2*n_days, c='r', ls='--')

# pl.semilogy(nufft_freq_sidereal, nufft_power_sidereal, 'o',
#             label=f'Mono + Doppler + Sidereal \n 5 vector estimator = {estimator_sidereal:.2f}')
# pl.legend(loc='lower center')
# pl.savefig("naive_5vector.png")


# print(estimator)

breakpoint()
