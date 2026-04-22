"""Defining the core :class:`resampler` class.
"""
__all__ = ["Resampler"]

import numpy as np
import numpy.typing as npt
import finufft

class Resampler(object):
    """ Resample a timeseries and computes its FFT

    Example usage:
        resampler = Resampler()
        resampler.timeseries = signal # signal should be generated with np.exp(1j*phi) [No negative sign]
        resampler.resampled_time = tau
        resampler.nufft()

    Attributes
    ----------
    timeseries : array
        Values of the timeseries
    resampled_time : array
        Times that the timeseries should be resampled and FFTed according to
        Use the convention tau = phi / (2*np.pi*f0). Remove the factor of 2*np.pi
    freqs : array
        Frequencies of the generated spectra. Units of radians
    weights : array (complex)
        Fourier weights of the spectra. 
    """
    def __init__(self, **kwws) -> None:
        """Constructor"""
        self.timeseries = None # Replace with dict if multiple detectors
        self.resampled_time = None # Replace with dict if multiple detectors
    
    def nufft(self) -> None:
        """Performs the NUFFT computing the Fourier frequncies and weights.
        Uses the (-) sign convention to match np.fft.fft
        """
        signal = self.timeseries
        tau = self.resampled_time
        bin_no = len(tau)

        # Rescale time to [-pi, pi)
        scale = (2*np.pi) / (tau[-1] - tau[0]) 
        tau_scaled = scale * (tau-tau[0])
        bins = np.arange(bin_no) - bin_no//2
        freqs = bins * scale
        
        weights = finufft.nufft1d1(tau_scaled, np.asarray(signal, dtype=complex), bin_no, isign = -1)
        self.freqs = freqs
        self.weights = weights

    # BE VERY CAREFUL WITH THIS, USES DIFFERENT TAU SIGN CONVENTION + MIGHT HAVE OTHER ISSUES
    # def nufft_real(self) -> None:
    #     """In principal this may use half the memory storage?. 
    #     Be careful about different normalization because of different length outputs
    #     """
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
        """Returns the normalized Fourier weights such that a pure trig function has power=1 regardless of length"""
        return self.weights/len(self.weights)
    
    @property
    def power(self) -> npt.NDArray[np.float64]:
        """Computes the Fourier power values"""
        return abs(self.weights)**2

    @property
    def power_normalized(self) -> npt.NDArray[np.float64]:
        """Returns the normalized Fourier powers such that a pure trig function has power=1 regardless of length"""
        # the normalization of a nufft and nufft_real are different because 
        # of their different lengths
        return abs(self.weights/len(self.weights))**2

    @property
    def freq_in_hz(self) -> npt.NDArray[np.float64]:
        """Returns the frequencies in units of hz"""
        return self.freqs / (2*np.pi)


    def extract_5vec(self, f0):
        """
        Extract the 5 normalized weights at the f0 frequency and the 2 sidebands from sidereal modulation.

        f0 may be a scalar (returns shape (5,)) or an array of carrier
        frequencies (returns shape (..., 5)).
        """
        side_day = 86164.09053083288
        f0 = np.asarray(f0)
        df = self.freqs[1] - self.freqs[0]
        offsets = 2 * np.pi / side_day * np.arange(-2, 3)  # (5,)
        targets = f0[..., None] + offsets                  # (..., 5)
        indices = np.round((targets - self.freqs[0]) / df).astype(int)
        indices = np.clip(indices, 0, self.freqs.size - 1)
        return self.weights_normalized[indices]