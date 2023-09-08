import numpy as np
import matplotlib.pyplot as pl
import time
from joblib import Parallel, delayed

tic = time.time()

def strobo(beta, data, f_ratio):
    new_t = -3/5*(1-8/3*beta*t)**(5/8)/beta
    f_new = f_signal/f_ratio
    
    new_t *= f_new 
    floor_t = np.floor(new_t) 
    idx = np.nonzero(np.diff(floor_t)) 
    resampled = data[idx]
    t_out = (new_t[idx]-new_t[0])/f_new
    return (resampled, t_out)

def pad_calc(beta, pad_len):
    out, t_out = strobo(beta, data, f_ratio)
    pad_frac = (pad_len - len(out)) / len(out)
    padded = np.full(pad_len, 0, dtype = complex)
    padded[:len(out)] = out
    pad_corrected = np.fft.fftshift(np.fft.fft(padded))
    pad_freq_corrected = np.fft.fftshift(np.fft.fftfreq(len(padded), d=t_out[1]-t_out[0]))
    pad_resampled_power = np.abs(pad_corrected/len(padded))**2
    
    arg_max = np.argmax(pad_resampled_power)
    peak_freq = pad_freq_corrected[arg_max]
    peak_power = pad_resampled_power[arg_max]
    return np.array([peak_freq, peak_power, pad_frac])

c = 3e8
G = 6.67e-11
pi = np.pi
const = 96/5*pi**(8/3)*(G/c**3)**(5/3)

# Signal constants
f0 = 120
Mc = 3e-4* 2e30
f_max = 200
T_obs = 1e4
pad_len = int(323.4*T_obs) #~0.01 added length to resampled data
beta = const*f0**(8/3)*Mc**(5/3)

f_signal = 40*f_max
nt = round(f_signal*T_obs)
t = np.arange(nt)/f_signal
f_ratio = 25

# Simulating the signal
phi = -6*pi/5*f0*(1-8./3.*beta*t)**(5/8)/beta
phi = np.mod(phi,2*pi)
signal = 1*np.exp(1j*phi)

nh = 0
noise = nh*np.random.normal(size = nt)
data = signal + noise

# Performing the calculations
pad_arr = [int(323.4*T_obs), int(502.4*T_obs)] # , int(650.4*T_obs)
offset_arr = np.logspace(-8, -1, 10)
result_arr = []
padfrac_arr = []

for i in pad_arr:
    temp = np.array([pad_calc(beta+j*beta, i) for j in offset_arr])
    result_arr.append(temp[:,1]/temp[0,1])
    padfrac_arr.append(100*temp[0,2])

for i,val in enumerate(result_arr):
    pl.semilogx(offset_arr, val, 'o', 
                label = 'pad_frac=%.1F %%' % padfrac_arr[i])
    
pl.xlabel(r'$\Delta \beta / \beta$')
pl.ylabel('Power ratio')
pl.title(r'$\beta_{exact} = %.1E$' % beta)
pl.legend()
# pl.axvline(offset_arr[-40])
pl.savefig('test.png')
    
toc = time.time()
print((toc-tic)/60)
