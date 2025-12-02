from scipy.signal.windows import tukey
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
import random
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
    nufft_freq = None
    temp_tau *= scale
    start_diff = -pi-temp_tau[0]
    temp_tau += start_diff
#     plan.setpts(x=np.array(tau))
#     nufft_amp = plan.execute(np.array(data.astype(complex)))
    nufft_amp = finufft.nufft1d1(np.array(temp_tau), np.array(
        data.astype(complex)), bin_no)/len(data)
    nufft_amp = np.flip(nufft_amp)
    nufft_amp = np.roll(nufft_amp, 1)
    nufft_power = np.abs(nufft_amp)**2
    nufft_freq = bins*scale
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
day = 86400

# ------------------------------------------------------------------------------------------
# NUFFT vs FFT
# n_days = 20
# duration = n_days*day
# f_signal = 5
# # very approximately 1st Jan 2024
# t_offset = np.arange(0, duration, 1/f_signal)
# samp_freq = 1/(np.diff(t_offset)[0])
# print(f"Sampling frequency = {samp_freq}")

# f0 = 0.45
# f_source = np.full(len(t_offset), f0, dtype=np.float64)
# phi_source = 2*np.pi * \
#     integrate.cumulative_trapezoid(f_source, x=t_offset, initial=0)
# signal_source = np.exp(1j*phi_source)
# tau_source = 2*np.pi * \
#     integrate.cumulative_trapezoid(f_source/f0, x=t_offset, initial=0)

# nufft_freq_source, nufft_amp_source, nufft_power_source = nufft(
#     signal_source, tau_source, len(signal_source))
# max_freq_nufft = nufft_freq_source[np.argmax(nufft_power_source)]
# max_power_nufft = np.max(nufft_power_source)
# pl.semilogy(nufft_freq_source, nufft_power_source,
#             label=f'NUFFT \n Max f0 = {max_freq_nufft:.2f} \n Max power = {max_power_nufft:.2f}')

# fft_freq, fft_amp, fft_power = naive_fft(signal_source, t_offset)
# max_freq_fft = nufft_freq_source[np.argmax(nufft_power_source)]
# max_power_fft = np.max(nufft_power_source)
# pl.semilogy(fft_freq, fft_power,
#             label=f'FFT \n Max f0 = {max_freq_fft:.2f} \n Max power = {max_power_fft:.2f}')
# pl.legend()
# pl.xlabel("Frequency [Hz]")
# pl.ylabel("Normalised power")
# pl.savefig("figs/NUFFT_vs_FFT.png")
# pl.close()

# # Coverage
# N = 20
# recovery_arr = []
# f0_arr = np.linspace(0.1, 2, N)
# for i in range(N):
#     print(i)
#     f0 = f0_arr[i]
#     f_source = np.full(len(t_offset), f0, dtype=np.float64)
#     phi_source = 2*np.pi * \
#         integrate.cumulative_trapezoid(f_source, x=t_offset, initial=0)
#     signal_source = np.exp(1j*phi_source)
#     tau_source = 2*np.pi * \
#         integrate.cumulative_trapezoid(f_source/f0, x=t_offset, initial=0)

#     nufft_freq_source, nufft_amp_source, nufft_power_source = nufft(
#         signal_source, tau_source, len(signal_source))
#     max_freq_nufft = nufft_freq_source[np.argmax(nufft_power_source)]
#     max_power_nufft = np.max(nufft_power_source)

#     fft_freq, fft_amp, fft_power = naive_fft(signal_source, t_offset)
#     max_freq_fft = nufft_freq_source[np.argmax(nufft_power_source)]
#     max_power_fft = np.max(nufft_power_source)

#     recovery = max_power_nufft/max_power_fft
#     recovery_arr.append(recovery)
# pl.plot(f0_arr, recovery_arr, '-o', label=f'Number of points = {N}')
# pl.xlabel("Injected frequency [Hz]")
# pl.ylabel("Recovery \% = P$_{NUFFT}$/P$_{FFT}$")
# pl.savefig("figs/NUFFT_vs_FFT_coverage.png")
# pl.close()
# breakpoint()

# ------------------------------------------------------------------------------------------
# PBH evolution

# n_days = 3
# duration = n_days*day
# f_signal = 100
# # very approximately 1st Jan 2024
# t_offset = np.arange(0, duration, 1/f_signal)
# samp_freq = 1/(np.diff(t_offset)[0])
# print(f"Sampling frequency = {samp_freq}")

# N = 20
# f0_arr = np.linspace(20, 30, N)
# f0 = f0_arr[3]
# Mc = 1e-3 * 2e30
# beta = const*f0**(8/3)*Mc**(5/3)
# f_source = f_calc(t_offset, beta, f0)
# phi_source = 2*np.pi * \
#     integrate.cumulative_trapezoid(f_source, x=t_offset, initial=0)
# signal_source = np.exp(1j*phi_source)
# tau_source = 2*np.pi * \
#     integrate.cumulative_trapezoid(f_source/f0, x=t_offset, initial=0)
# f_source = None
# phi_source = None
# alpha = 0.2
# window = tukey(len(signal_source), alpha)
# window_power = np.sum(window**2/len(signal_source))
# windowed_signal = window*signal_source

# nufft_freq_source, nufft_amp_source, nufft_power_source = nufft(
#     windowed_signal, tau_source, len(signal_source))
# max_freq_nufft = nufft_freq_source[np.argmax(nufft_power_source)]
# max_power_nufft = np.max(nufft_power_source)
# pl.semilogy(nufft_freq_source, nufft_power_source, '-o',
#             label=f'Windowed NUFFT \n Max f0 = {max_freq_nufft:.2f} \n Max power = {max_power_nufft/window_power:.2f}')

# nufft_freq_source, nufft_amp_source, nufft_power_source = nufft(
#     signal_source, tau_source, len(signal_source))
# max_freq_nufft = nufft_freq_source[np.argmax(nufft_power_source)]
# max_power_nufft = np.max(nufft_power_source)
# pl.semilogy(nufft_freq_source, nufft_power_source, '-o',
#             label=f'Non-windowed NUFFT \n Max f0 = {max_freq_nufft:.2f} \n Max power = {max_power_nufft:.2f}')
# pl.axvline(f0, c=yellow, ls='--', label='Injected f0')
# pl.xlim(f0-0.00003, f0+0.00003)
# pl.xlabel("Frequency [Hz]")
# pl.ylabel("Normalised power")
# pl.legend()
# pl.savefig("figs/PBH.png")
# pl.close()

# # Coverage
# N = 20
# recovery_arr = []
# windowed_arr = []
# f0_arr = np.linspace(20, 30, N)
# alpha = 0.2
# window = tukey(len(signal_source), alpha)
# window_power = np.sum(window**2/len(signal_source))

# for i in range(N):
#     print(i)
#     f0 = f0_arr[i]
#     Mc = 1e-3 * 2e30
#     beta = const*f0**(8/3)*Mc**(5/3)
#     f_source = f_calc(t_offset, beta, f0)
#     print(f_source[-1])
#     phi_source = 2*np.pi * \
#         integrate.cumulative_trapezoid(f_source, x=t_offset, initial=0)
#     signal_source = np.exp(1j*phi_source)
#     tau_source = 2*np.pi * \
#         integrate.cumulative_trapezoid(f_source/f0, x=t_offset, initial=0)
#     f_source = None
#     phi_source = None

#     nufft_freq_source, nufft_amp_source, nufft_power_source = nufft(
#         signal_source, tau_source, len(signal_source))
#     max_freq_nufft = nufft_freq_source[np.argmax(nufft_power_source)]
#     max_power_nufft = np.max(nufft_power_source)
#     nufft_freq_source, nufft_amp_source, nufft_power_source = [
#         None, None, None]
#     recovery_arr.append(max_power_nufft)

#     windowed_signal = window*signal_source
#     nufft_freq_source, nufft_amp_source, nufft_power_source = nufft(
#         windowed_signal, tau_source, len(signal_source))
#     max_freq_nufft = nufft_freq_source[np.argmax(nufft_power_source)]
#     max_power_nufft = np.max(nufft_power_source)
#     nufft_freq_source, nufft_amp_source, nufft_power_source = [
#         None, None, None]
#     windowed_arr.append(max_power_nufft/window_power)

# pl.plot(f0_arr, windowed_arr/window_power, '-o',
#         label=f'Windowed-NUFFT \n Number of points = {N}')
# pl.plot(f0_arr, recovery_arr, '-o',
#         label=f'Non-windowed NUFFT')
# pl.xlabel("Injected frequency [Hz]")
# pl.ylabel("P$_\mathrm{NUFFT}$/P$_\mathrm{window}$")
# pl.legend()
# pl.title("Low power recovery occurs when the injected f0 falls between two bins and splits the power. \n \
#     This is demonstrated in PBH.png", fontsize=24)
# pl.tight_layout()
# pl.savefig("figs/PBH_coverage.png")
# pl.close()

# ------------------------------------------------------------------------------------------
# # Doppler
# ephemeris = np.loadtxt("../doppler/earth00-40-DE440.dat",
#                        delimiter='\t', skiprows=18)
# [t_eph, pos_x, pos_y, pos_z, vel_x, vel_y,
#     vel_z, acc_x, acc_y, acc_z] = ephemeris.T

# pos_arr = np.array([pos_x, pos_y, pos_z])
# interp_pos = CubicSpline(t_eph, pos_arr.T)
# v_arr = np.array([vel_x, vel_y, vel_z])
# interp_v = CubicSpline(t_eph, v_arr.T)
# acc_arr = np.array([acc_x, acc_y, acc_z])
# interp_acc = CubicSpline(t_eph, acc_arr.T)

# H_lat = 119.41 * np.pi/180  # rad
# H_lng = 46.45 * np.pi/180  # rad
# loaded = np.loadtxt("figs/doppler_coverage_params.txt")[0]

# params = dict(ra=loaded[1], dec=loaded[2], eta=0.5, psi=0.5, lat=H_lat,
#               lng=H_lng, az=0.5)

# n_hat = (np.cos(params['ra'])*np.cos(params['dec']),
#          np.sin(params['ra'])*np.cos(params['dec']),
#          np.sin(params['dec']))

# n_days = 3
# duration = n_days*day
# f_signal = 100
# # very approximately 1st Jan 2024
# start_time = t_eph[abs(t_eph-(t_eph[0]+24*365*day)).argmin()]
# t_offset = np.arange(0, duration, 1/f_signal)
# t_doppler = start_time + t_offset
# samp_freq = 1/(np.diff(t_offset)[0])
# print(f"Sampling frequency = {samp_freq}")

# f0 = loaded[0]
# Mc = 1e-2 * 2e30
# beta = const*f0**(8/3)*Mc**(5/3)
# f_source = np.full(len(t_offset), f0)
# phi_source = 2*np.pi * \
#     integrate.cumulative_trapezoid(f_source, x=t_offset, initial=0)
# signal_source = np.exp(1j*phi_source)
# tau_source = 2*np.pi * \
#     integrate.cumulative_trapezoid(f_source/f0, x=t_offset, initial=0)
# phi_source = None

# f_doppler = f_source * (1+np.dot(interp_v(t_doppler), n_hat))
# phi_doppler = 2*np.pi * \
#     integrate.cumulative_trapezoid(f_doppler, x=t_offset, initial=0)
# signal_doppler = np.exp(1j*phi_doppler)
# tau_doppler = 2*np.pi * \
#     integrate.cumulative_trapezoid(f_doppler/f0, x=t_offset, initial=0)

# nufft_freq_doppler, nufft_amp_doppler, nufft_power_doppler = nufft(
#     signal_doppler, tau_doppler, len(signal_doppler))
# max_freq_nufft = nufft_freq_doppler[np.argmax(nufft_power_doppler)]
# max_power_nufft = np.max(nufft_power_doppler)

# pl.semilogy(nufft_freq_doppler, nufft_power_doppler, '-o',
#             label=f'NUFFT. Max f0 = {max_freq_nufft:.2f} \n Max power = {max_power_nufft:.2f}')
# pl.axvline(f0, color=yellow, ls='--', label='Injected f0')
# pl.legend()
# pl.xlabel("f0")
# pl.ylabel("P$_\mathrm{NUFFT}$")
# pl.title("Signal = Mono + Doppler. Power split between two bins. ", fontsize=24)
# pl.xlim(f0-0.00003, f0+0.00003)
# pl.tight_layout()
# pl.savefig("figs/doppler.png")
# pl.close()
# breakpoint()

# # Coverage
# N = 20
# recovery_arr = []
# params_arr = []

# for i in range(N):
#     print(i)
#     f0 = random.uniform(20, 35)
#     f_source = np.full(len(t_offset), f0)
#     params = dict(ra=random.uniform(0, 2*np.pi), dec=random.uniform(-np.pi/2, np.pi/2), eta=random.uniform(0, 2*np.pi),
#                   psi=random.uniform(0, 2*np.pi), lat=H_lat, lng=H_lng, az=random.uniform(0, 2*np.pi))
#     n_hat = (np.cos(params['ra'])*np.cos(params['dec']),
#              np.sin(params['ra'])*np.cos(params['dec']),
#              np.sin(params['dec']))

#     f_doppler = f_source * (1+np.dot(interp_v(t_doppler), n_hat))
#     phi_doppler = 2*np.pi * \
#         integrate.cumulative_trapezoid(f_doppler, x=t_offset, initial=0)
#     signal_doppler = np.exp(1j*phi_doppler)
#     tau_doppler = 2*np.pi * \
#         integrate.cumulative_trapezoid(f_doppler/f0, x=t_offset, initial=0)

#     nufft_freq_doppler, nufft_amp_doppler, nufft_power_doppler = nufft(
#         signal_doppler, tau_doppler, len(signal_doppler))
#     max_freq_nufft = nufft_freq_doppler[np.argmax(nufft_power_doppler)]
#     max_power_nufft = np.max(nufft_power_doppler)
#     nufft_freq_source, nufft_amp_source, nufft_power_source = [
#         None, None, None]
#     recovery_arr.append(max_power_nufft)
#     params_arr.append((f0, params['ra'], params['dec']))

# pl.plot(recovery_arr, '-o',
#         label=f'Number of points = {N}')
# pl.xlabel("Trial number")
# pl.ylabel("P$_\mathrm{NUFFT}$")
# pl.legend()
# pl.title("Injected monochromatic signal with random f0 from random sky-location \n \
#     Low power recovery occurs when the injected f0 falls between two bins and splits the power", fontsize=24)
# pl.tight_layout()
# pl.savefig("figs/doppler_coverage.png")
# breakpoint()
# pl.close()

# ------------------------------------------------------------------------------------------
# Sidereal modulation

print("Sidereal")
loaded = np.loadtxt("figs/sidereal_coverage_params.txt")[0]
H_lat = 46.455 * np.pi/180  # rad
H_lng = 240.592 * np.pi/180  # rad
H_az = 144.0006 * np.pi/180  # rad

pulsar3_params = dict(ra=178.3726/180*np.pi,
                      dec=-33.4366/180*np.pi,
                      eta=0.160,
                      psi=25.4390/180*np.pi,
                      lat=H_lat,
                      lng=H_lng,
                      az=H_az)

params = pulsar3_params
# dict(ra=loaded[1], dec=loaded[2], eta=loaded[3], psi=loaded[4], lat=H_lat,
#   lng=H_lng, az=loaded[5])

n_hat = (np.cos(params['ra'])*np.cos(params['dec']),
         np.sin(params['ra'])*np.cos(params['dec']),
         np.sin(params['dec']))

n_days = 3
duration = n_days*day
f_signal = 30
t0 = 1.238112018000000e+09
# very approximately 1st Jan 2024
t_offset = np.arange(t0, t0+duration, 1/f_signal)  # gps
gps_offset = H_lng/np.pi*180 / 15 * 3600 - 17806
t_offset += gps_offset
print(f"Sampling frequency = {f_signal}")

f0 = 10.673  # loaded[0]
f_source = np.full(len(t_offset), f0)
phi_source = 2*np.pi * \
    integrate.cumulative_trapezoid(f_source, x=t_offset, initial=0)
signal_source = np.exp(1j*phi_source)

# Reference recovery
ref_freq, ref_amp, ref_power = naive_fft(signal_source, t_offset)
ref_recovery = np.max(ref_power)
ref_freq_recovery = ref_freq[np.argmax(ref_power)]

# 5-vector recovery
sidereal = five_vec(**params)
sidereal.compute_H()
sidereal.compute_A(t_offset)
sidereal.compute_5vec()
amp_modulation = sidereal.amp_modulation
sidereal.check_ps_lf(t_offset)


def fft_5vec(signal, time, f0):
    fft_freq, fft_amp, _ = naive_fft(signal, time)
    idx = abs(fft_freq-f0).argmin()
    numerical_5vec = [fft_amp[idx + i * n_days]
                      for i in range(-2, 3)]
    return numerical_5vec


signal_sidereal = amp_modulation*signal_source
template_p = sidereal.scalar_Aplus * signal_source
template_c = sidereal.scalar_Across * signal_source

signal_5vec = fft_5vec(signal_sidereal, t_offset, f0)
plus_5vec = fft_5vec(template_p, t_offset, f0)
cross_5vec = fft_5vec(template_c, t_offset, f0)

hp_est = np.dot(signal_5vec, np.conj(plus_5vec))/np.sum(np.abs(plus_5vec)**2)
hc_est = np.dot(signal_5vec, np.conj(cross_5vec))/np.sum(np.abs(cross_5vec)**2)
h_est = np.sqrt(abs(hp_est)**2+abs(hc_est)**2)

fft_freq_sidereal, fft_amp_sidereal, fft_power_sidereal = naive_fft(
    signal_sidereal, t_offset)
fft_f0_idx = abs(fft_freq_sidereal-f0).argmin()

pl.semilogy(fft_freq_sidereal, fft_power_sidereal, '-o',
            label=f'Mono + Sidereal signal. 5 vector estimator = {h_est:.2f}')
# pl.semilogy(ref_freq, ref_power, '-o',
#             label=f'Mono-only signal. Power = {ref_recovery:.2f}')
pl.axvline(f0, c=orange, label='Injected')
pl.axvline(fft_freq_sidereal[fft_f0_idx-2*n_days], c=green, ls='--')
pl.axvline(fft_freq_sidereal[fft_f0_idx-1*n_days], c=green, ls='--')
pl.axvline(fft_freq_sidereal[fft_f0_idx], c=green, ls='--')
pl.axvline(fft_freq_sidereal[fft_f0_idx+1*n_days], c=green, ls='--')
pl.axvline(fft_freq_sidereal[fft_f0_idx+2*n_days], c=green, ls='--')
pl.xlim(f0-0.00005, f0+0.00005)
# pl.title("", fontsize=24)
pl.xlabel("Frequency [Hz]")
pl.ylabel("FFT power")
pl.legend(loc='lower center')
pl.tight_layout()
pl.savefig("figs/sidereal.png")
pl.close()
breakpoint()

# # Coverage
# N = 10
# recovery_arr = np.empty(N)
# ref_arr = np.empty(N)
# params_arr = []
# for i in range(N):
#     print(i)
#     params = dict(ra=random.uniform(0, 2*np.pi), dec=random.uniform(-np.pi/2, np.pi/2), eta=random.uniform(-1, 1),
#                   psi=random.uniform(0, np.pi), lat=H_lat, lng=H_lng, az=H_az)
#     n_hat = (np.cos(params['ra'])*np.cos(params['dec']),
#              np.sin(params['ra'])*np.cos(params['dec']),
#              np.sin(params['dec']))
#     f0 = random.uniform(5, 10)

#     f_source = np.full(len(t_offset), f0)
#     phi_source = 2*np.pi * \
#         integrate.cumulative_trapezoid(f_source, x=t_offset, initial=0)
#     signal_source = np.exp(1j*phi_source)
#     phi_source = None
#     ref_freq, ref_amp, ref_power = naive_fft(signal_source, t_offset)
#     ref_recovery = np.max(ref_power)
#     ref_freq, ref_amp, ref_power = [None, None, None]

#     sidereal = five_vec(**params)
#     sidereal.compute_H()
#     sidereal.compute_A(t_offset)
#     sidereal.compute_5vec()
#     amp_modulation = sidereal.amp_modulation

#     signal_sidereal = amp_modulation*signal_source
#     template_p = sidereal.scalar_Aplus * signal_source
#     template_c = sidereal.scalar_Across * signal_source

#     signal_5vec = fft_5vec(signal_sidereal, t_offset, f0)
#     plus_5vec = fft_5vec(template_p, t_offset, f0)
#     cross_5vec = fft_5vec(template_c, t_offset, f0)

#     hp_est = np.dot(signal_5vec, np.conj(plus_5vec)) / \
#         np.sum(np.abs(plus_5vec)**2)
#     hc_est = np.dot(signal_5vec, np.conj(cross_5vec)) / \
#         np.sum(np.abs(cross_5vec)**2)
#     h_est = np.sqrt(abs(hp_est)**2+abs(hc_est)**2)

#     recovery_arr[i] = h_est
#     ref_arr[i] = ref_recovery
#     params_arr.append(
#         (f0, params['ra'], params['dec'], params['eta'], params['psi'], params['az']))

# pl.plot(recovery_arr, '-o',
#         label=f'Number of points = {N}')
# pl.xlabel("Trial number")
# pl.ylabel("5-vector estimator/max(FFT power)")
# pl.legend()
# pl.title("Injected monochromatic signal + sidereal modulation (different randomised properties)", fontsize=24)
# pl.tight_layout()
# pl.savefig("figs/sidereal_coverage.png")
# breakpoint()
# pl.close()

# ------------------------------------------------------------------------------------------
# PBH + doppler
print("PBH + doppler")
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
H_az = 144.0006 * np.pi/180  # rad

params = dict(ra=0.5, dec=0.5, eta=0.5, psi=0.5, lat=H_lat,
              lng=H_lng, az=H_az)

n_hat = (np.cos(params['ra'])*np.cos(params['dec']),
         np.sin(params['ra'])*np.cos(params['dec']),
         np.sin(params['dec']))

n_days = 3
duration = n_days*day
f_signal = 20
# very approximately 1st Jan 2024
start_time = t_eph[abs(t_eph-(t_eph[0]+24*365*day)).argmin()]
t_offset = np.arange(0, duration, 1/f_signal)
t_doppler = start_time + t_offset
samp_freq = 1/(np.diff(t_offset)[0])
print(f"Sampling frequency = {samp_freq}")

f0 = 1.5
Mc = 1e-3 * 2e30
beta = const*f0**(8/3)*Mc**(5/3)
f_source = f_calc(t_offset, beta, f0)
f_doppler = f_source * (1+np.dot(interp_v(t_doppler), n_hat))
phi_doppler = 2*np.pi * \
    integrate.cumulative_trapezoid(f_doppler, x=t_offset, initial=0)
signal_doppler = np.exp(1j*phi_doppler)

ref_phi = 2*np.pi * \
    integrate.cumulative_trapezoid(f_doppler/f0, x=t_offset, initial=0)
ref_signal = np.exp(1j*ref_phi)

f_source = None
f_doppler = None
phi_doppler = None

nufft_freq, nufft_amp, nufft_power = nufft(
    signal_doppler, ref_phi, len(signal_doppler))
pl.plot(nufft_freq, nufft_power, '-o')
pl.xlim(f0 - 0.0001, f0 + 0.0001)
pl.title("Signal: PBH+doppler")
pl.savefig("figs/PBH+doppler.png")
pl.close()

# ------------------------------------------------------------------------------------------
# PBH + doppler + sidereal
print("PBH + doppler + sidereal")


def nufft_5vec(signal, time, f0):
    nufft_freq, nufft_amp, _ = nufft(signal, time, len(signal))
    idx = abs(nufft_freq-f0).argmin()
    numerical_5vec = [nufft_amp[idx + i * n_days]
                      for i in range(-2, 3)]
    return numerical_5vec


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
H_az = 144.0006 * np.pi/180  # rad

params = dict(ra=0.5, dec=0.5, eta=0.5, psi=0.5, lat=H_lat,
              lng=H_lng, az=H_az)

n_hat = (np.cos(params['ra'])*np.cos(params['dec']),
         np.sin(params['ra'])*np.cos(params['dec']),
         np.sin(params['dec']))

n_days = 3
duration = n_days*day
f_signal = 20
# very approximately 1st Jan 2024
start_time = t_eph[abs(t_eph-(t_eph[0]+24*365*day)).argmin()]
t_offset = np.arange(0, duration, 1/f_signal)
t_doppler = start_time + t_offset
samp_freq = 1/(np.diff(t_offset)[0])
print(f"Sampling frequency = {samp_freq}")

f0 = 1.5000012012457498
Mc = 1e-3 * 2e30
beta = const*f0**(8/3)*Mc**(5/3)
f_source = f_calc(t_offset, beta, f0)
f_doppler = f_source * (1+np.dot(interp_v(t_doppler), n_hat))
phi_doppler = 2*np.pi * \
    integrate.cumulative_trapezoid(f_doppler, x=t_offset, initial=0)
signal_doppler = np.exp(1j*phi_doppler)

ref_phi = 2*np.pi * \
    integrate.cumulative_trapezoid(f_doppler/f0, x=t_offset, initial=0)
ref_signal = np.exp(1j*ref_phi)

sidereal = five_vec(**params)
sidereal.compute_H()
sidereal.compute_A(t_doppler)
sidereal.compute_5vec()
amp_modulation = sidereal.amp_modulation

signal_sidereal = amp_modulation*signal_doppler
ref_sidereal = amp_modulation*ref_signal
template_p = sidereal.scalar_Aplus * ref_signal
template_c = sidereal.scalar_Across * ref_signal

f_source = None
f_doppler = None
phi_doppler = None
signal_doppler = None

signal_5vec = nufft_5vec(signal_sidereal, ref_phi, f0)
test_f = f0
test_freq, test_amp, test_power = naive_fft(
    amp_modulation, t_doppler)
test_idx = abs(test_freq).argmin()
test_5vec = np.array([test_amp[test_idx-2*n_days], test_amp[test_idx-1*n_days],
                     test_amp[test_idx], test_amp[test_idx+1*n_days], test_amp[test_idx+2*n_days]])
test_est = np.dot(signal_5vec, np.conj(test_5vec))/np.sum(np.abs(test_5vec)**2)
pl.plot(test_freq, test_power, '-o')
pl.xlim(0 - 0.0001, 0 + 0.0001)
pl.title("Signal: sidereal modulation only")
pl.savefig("figs/test.png")
pl.close()

nufft_freq, nufft_amp, nufft_power = nufft(
    signal_sidereal, ref_phi, len(signal_sidereal))
pl.plot(nufft_freq, nufft_power, '-o')
pl.xlim(f0 - 0.0001, f0 + 0.0001)
pl.title("Signal: PBH+doppler+sidereal")
pl.savefig("figs/test1.png")
pl.close()
print(abs(test_est))
breakpoint()

plus_5vec = nufft_5vec(template_p, ref_phi, 1)
cross_5vec = nufft_5vec(template_c, ref_phi, 1)

hp_est = np.dot(signal_5vec, np.conj(plus_5vec))/np.sum(np.abs(plus_5vec)**2)
hc_est = np.dot(signal_5vec, np.conj(cross_5vec))/np.sum(np.abs(cross_5vec)**2)
h_est = np.sqrt(abs(hp_est)**2+abs(hc_est)**2)
breakpoint()

nufft_freq, nufft_amp, nufft_power = nufft(
    signal_sidereal, ref_phi, len(signal_sidereal))
idx = abs(nufft_freq-f0).argmin()

pl.semilogy(nufft_freq, nufft_power, '-o',
            label=f'5 vector estimator = {h_est:.2f}')
pl.axvline(f0, c=orange, label='Injected')
pl.axvline(nufft_freq[idx-2*n_days], c=green, ls='--')
pl.axvline(nufft_freq[idx-1*n_days], c=green, ls='--')
pl.axvline(nufft_freq[idx], c=green, ls='--')
pl.axvline(nufft_freq[idx+1*n_days], c=green, ls='--')
pl.axvline(nufft_freq[idx+2*n_days], c=green, ls='--')
pl.legend(loc='lower center')
pl.axvline(f0, c=orange)
pl.xlim(f0-0.00005, f0+0.00005)
pl.title("PBH + Doppler + Sidereal signal", fontsize=24)
pl.xlabel("Frequency [Hz]")
pl.ylabel("FFT power")
pl.tight_layout()
pl.savefig("figs/PBH+Doppler+sidereal.png")
breakpoint()
pl.close()

# Coverage
N = 10
recovery_arr = np.empty(N)
ref_arr = np.empty(N)
params_arr = []
for i in range(N):
    print(i)
    params = dict(ra=random.uniform(0, 2*np.pi), dec=random.uniform(-np.pi/2, np.pi/2), eta=random.uniform(-1, 1),
                  psi=random.uniform(0, np.pi), lat=H_lat, lng=H_lng, az=H_az)
    n_hat = (np.cos(params['ra'])*np.cos(params['dec']),
             np.sin(params['ra'])*np.cos(params['dec']),
             np.sin(params['dec']))
    f0 = random.uniform(1, 5)
    Mc = 1e-2 * 2e30
    beta = const*f0**(8/3)*Mc**(5/3)

    f_source = f_calc(t_offset, beta, f0)
    f_doppler = f_source * (1+np.dot(interp_v(t_doppler), n_hat))
    phi_doppler = 2*np.pi * \
        integrate.cumulative_trapezoid(f_doppler, x=t_offset, initial=0)
    signal_doppler = np.exp(1j*phi_doppler)
    tau_doppler = 2*np.pi * \
        integrate.cumulative_trapezoid(f_doppler/f0, x=t_offset, initial=0)

    sidereal = five_vec(**params)
    sidereal.compute_H()
    sidereal.compute_A(t_doppler)
    sidereal.compute_5vec()
    amp_modulation = sidereal.amp_modulation

    signal_sidereal = amp_modulation*signal_doppler
    template_p = sidereal.scalar_Aplus * signal_doppler
    template_c = sidereal.scalar_Across * signal_doppler

    signal_5vec = nufft_5vec(signal_sidereal, tau_doppler, f0)
    plus_5vec = nufft_5vec(template_p, tau_doppler, f0)
    cross_5vec = nufft_5vec(template_c, tau_doppler, f0)

    hp_est = np.dot(signal_5vec, np.conj(plus_5vec)) / \
        np.sum(np.abs(plus_5vec)**2)
    hc_est = np.dot(signal_5vec, np.conj(cross_5vec)) / \
        np.sum(np.abs(cross_5vec)**2)
    h_est = np.sqrt(abs(hp_est)**2+abs(hc_est)**2)

    recovery_arr[i] = h_est
    params_arr.append(
        (f0, params['ra'], params['dec'], params['eta'], params['psi'], params['az']))

pl.plot(recovery_arr, '-o',
        label=f'Number of points = {N}')
pl.xlabel("Trial number")
pl.ylabel("Estimated amplitude")
pl.legend()
pl.title("Injected PBH + doppler + sidereal modulation (different randomised properties)", fontsize=24)
pl.tight_layout()
pl.savefig("figs/PBH+Doppler+sidereal_coverage.png")
breakpoint()
pl.close()

breakpoint()
