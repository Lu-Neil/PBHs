import numpy as np
import finufft
import matplotlib.pyplot as pl
import time
tic = time.time()

pi = np.pi
c = 3e8
G = 6.67e-11
pi = np.pi
const = 96/5*pi**(8/3)*(G/c**3)**(5/3)

f_max = 200
f_strobo = 40*f_max
f_nufft = 3*f_max

def t_max_calc(f0, beta, f_max):
    temp0 = 0.375/beta
    temp1 = 1-(f0/f_max)**(8/3)
    return temp0*temp1

def strobo(beta, data, t, f_new=f_strobo/20):
    new_t = -3/5*(1-8/3*beta*t)**(5/8)/beta
    new_t *= f_new 
    floor_t = np.floor(new_t) 
    idx = np.nonzero(np.diff(floor_t))
    resampled = data[idx]
    t_out = (new_t[idx]-new_t[0])/f_new
#     print(f_new)
#     print(t_out)
    corrected = np.fft.fftshift(np.fft.fft(resampled))
    strobo_freq = np.fft.fftshift(np.fft.fftfreq(len(t_out), d=t_out[1]-t_out[0]))
    strobo_power = np.abs(corrected/len(resampled))**2
    return (strobo_freq, strobo_power)

def nufft(beta, data, t, bin_no):
    bins = np.arange(bin_no) - bin_no//2
    
    tau = -3/5*(1-8/3*beta*t)**(5/8)/beta
    scale = (2*pi)/(tau[-1]-tau[0])
    tau *= scale
    start_diff = -pi-tau[0]
    tau += start_diff
    nufft_amp = finufft.nufft1d1(tau.astype(np.float64), data.astype(complex), bin_no)
    nufft_freq = bins*scale/(2*pi)
    nufft_power = np.abs(nufft_amp/len(data))**2
    return nufft_freq, nufft_power

def main_calc(f, M):
    beta = const*f**(8/3)*M**(5/3)
    T_obs = min(1e4, t_max_calc(f, beta, 200))
    
    #Creating the signal
    t_strobo = np.arange(0, T_obs, 1/f_strobo).astype(np.longdouble)
    phi = -6*pi/5*f*(1-8./3.*(beta)*t_strobo)**(5/8)/beta
    phi = np.mod(phi,2*pi)
    signal_strobo = 1*np.exp(1j*phi)
    signal_strobo[:int(1e3)] = np.zeros((int(1e3)))
    signal_strobo[-int(1e3):] = np.zeros((int(1e3)))
    
    t_nufft = np.arange(0, T_obs, 1/f_nufft).astype(np.longdouble)
    phi = -6*pi/5*f*(1-8./3.*(beta)*t_nufft)**(5/8)/beta
    phi = np.mod(phi,2*pi)
    signal_nufft = 1*np.exp(1j*phi)
    signal_nufft[:int(1e3)] = np.zeros((int(1e3)))
    signal_nufft[-int(1e3):] = np.zeros((int(1e3)))

    tic = time.time()
    strobo_freq_10, strobo_power_10 = strobo(beta, signal_strobo, t_strobo, f_new=f_strobo/10)
    strobo10_time = (time.time() - tic)/60
    
    tic = time.time()
    strobo_freq_20, strobo_power_20 = strobo(beta, signal_strobo, t_strobo, f_new=f_strobo/20)
    strobo20_time = (time.time() - tic)/60
    
    tic=time.time()
    nufft_freq, nufft_power = nufft(beta, signal_nufft, t_nufft, len(signal_nufft))
    nufft_time = (time.time() - tic)/60

    strobo10_peak = strobo_power_10>np.median(strobo_power_10)+6*np.std(strobo_power_10)
    strobo20_peak = strobo_power_20>np.median(strobo_power_20)+6*np.std(strobo_power_20)
    nufft_peak = nufft_power>np.median(nufft_power)+6*np.std(nufft_power)
    
    nufft_info={'name':'nufft',
                'power':np.sum(nufft_power[nufft_peak]),
                'peak_f':np.max(nufft_freq[nufft_peak]),
                'f_std': np.std(nufft_freq[nufft_peak]),
                'time': nufft_time}
    
    strobo10_info={'name':'strobo10',
                   'power':np.sum(nufft_power[strobo10_peak]),
                   'peak_f':np.max(nufft_freq[strobo10_peak]),
                   'f_std': np.std(nufft_freq[strobo10_peak]),
                   'time':strobo10_time}
    
    strobo20_info={'name':'strobo20',
                  'power':np.sum(nufft_power[strobo20_peak]),
                  'peak_f':np.max(nufft_freq[strobo20_peak]),
                  'f_std': np.std(nufft_freq[strobo20_peak]),
                  'time':strobo20_time}
    
    return np.array([nufft_info, strobo10_info, strobo20_info])

fspace = np.linspace(20, 190, 3)
Mspace = np.logspace(-5, -1, 3)*2e30

result = np.array([main_calc(f, M) for M in Mspace for f in fspace])
np.savetxt('nufft_vs_strobo_data.txt', result, header = '3 dictionaries')

toc=time.time()
print((toc-tic)/60)


# pl.plot(strobo_freq, strobo_power, 'o', label = 'Strobo')
# pl.plot(-nufft_freq, nufft_power, 'o', label = 'NUFFT')
# pl.legend()
# pl.title('Stroboscopic vs NUFFT resampling: \n \
#           Peak power diff = %.2f \n \
#           Recovered power diff = %.2f' % (peak_diff, recoveredP_diff))
# pl.savefig('NUFFT_vs_strobo.png')
                    
                    
