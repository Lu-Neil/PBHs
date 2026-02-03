"""Defining the core :class:`resampler` class.
"""
__all__ = ["Resampler"]

import numpy as np
import numpy.typing as npt
import finufft

class Resampler(object):
    """ Resample a timeseries and computes its FFT

    Example usage:

        TBD

    Attributes
    ----------
    TBD : 
    """
    def __init__(self, **kwws) -> None:
        self.timeseries = None # Replace with dict if multiple detectors
        self.resampled_time = None # Replace with dict if multiple detectors
    
    def nufft(self) -> None:
        signal = self.timeseries
        tau = self.resampled_time
        bin_no = len(tau)

        # Rescale time to [-pi, pi)
        scale = (2*np.pi) / (tau[-1] - tau[0]) 
        tau_scaled = scale * (tau-tau[0])
        bins = np.arange(bin_no) - bin_no//2
        freqs = bins * scale
        
        weights = finufft.nufft1d1(tau_scaled, signal.astype(complex), bin_no, isign = -1)
        self.freqs = freqs
        self.weights = weights

    # BE VERY CAREFUL WITH THIS, USES DIFFERENT TAU SIGN CONVENTION + MIGHT HAVE OTHER ISSUES
    # def nufft_real(self) -> None:
    #     signal = self.timeseries
    #     tau = self.resampled_time
    #     bin_no = len(tau)//2
    #     assert np.isrealobj(signal), "Timeseries is not real"

    #     scale = (2*np.pi) / (tau[-1] - tau[0])
    #     tau_scaled = scale * (tau-tau[0]) - np.pi
        
    #     bins = np.arange(bin_no) # no negative frequencies
    #     freqs = bins * scale
    #     normalisation = bin_no // 2
    #     signal_shifted = signal * np.exp(-1j * normalisation * tau_scaled) # might be -1j

    #     weights = finufft.nufft1d1(tau_scaled, signal_shifted, bin_no, isign = -1)
    #     self.freqs = freqs
    #     self.weights = weights

    @property
    def weights_normalized(self) -> npt.NDArray[np.float64]:
        return self.weights/len(self.weights)
    
    @property
    def power(self) -> npt.NDArray[np.float64]:
        return abs(self.weights)**2

    @property
    def power_normalized(self) -> npt.NDArray[np.float64]:
        # the normalization of a nufft and nufft_real are different because 
        # of their different lengths
        return abs(self.weights/len(self.weights))**2

    @property
    def freq_in_hz(self) -> npt.NDArray[np.float64]:
        return self.freqs / (2*np.pi)
