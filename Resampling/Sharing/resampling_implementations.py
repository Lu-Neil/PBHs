import numpy as np
import finufft

def strobo(new_t, data, f_ratio=32):
    """Python implementation of the stroboscopic resampling from Snag
    
    Parameters
    ----------
    new_t : ndarray
    	Resampled time
    data : ndarray
    	Data time series
    f_ratio : int
    	The ratio between the sampling frequency of data to the desired sampling frequency of the resampled time series. Higher values give more accurate results
    """
    f_new = f_signal/f_ratio
    
    new_t *= f_new 
    floor_t = np.floor(new_t) 
    idx = np.nonzero(np.diff(floor_t)) 
    resampled = data[idx]
    t_out = (new_t[idx]-new_t[0])/f_new
    return (resampled, t_out)

def nufft_cpu(new_t, data, bin_no=len(data):
    """Resampling implemented with a nonuniform fast-fourier transform
    
    Parameters
    ----------
    new_t : ndarray
    	Resampled time
    data : ndarray
    	Data time series
    bin_no : int
    	Desired number of frequency bins to calculate the NUFFT for
    """
    bins = np.arange(bin_no) - bin_no//2
          
    scale = (2*pi)/(new_t[-1]-new_t[0])
    new_t *= scale
    start_diff = -pi-new_t[0]
    new_t += start_diff
    nufft_amp = finufft.nufft1d1(tau.astype(np.float64), data.astype(complex), bin_no)
    nufft_freq = bins*scale/(2*pi)
    nufft_power = np.abs(nufft_amp/len(data))**2
    return nufft_freq, nufft_power
    
# GPU implementation to be added
