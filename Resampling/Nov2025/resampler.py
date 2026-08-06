"""Defining the core :class:`resampler` class.
"""
__all__ = ["Resampler", "clear_plan_cache"]

import numpy as np
import numpy.typing as npt
import finufft


# FFTW planner flags (from fftw3.h). finufft's ``fftw`` option takes these
# integer constants directly, so we hard-code them rather than depending on pyfftw.
_FFTW_MEASURE = 0
_FFTW_ESTIMATE = 64
_FFTW_PATIENT = 32

# Process-wide plan cache. finufft.Plan objects are reusable across calls --
# only setpts() and execute() depend on the data -- so reusing a plan
# amortises the FFTW planning cost. Mirrors the plan-caching pattern used in
# pycbc.fft.fftw. Not thread-safe: if multiple threads share a Resampler
# they will race on plan state, so call clear_plan_cache() and build per
# thread if you need that.
_PLAN_CACHE: dict = {}


def _get_or_make_plan(bin_no, *, eps, isign, nthreads, dtype, upsampfac, fftw_flag):
    """Return a cached finufft.Plan or build (and cache) a new one."""
    key = (bin_no, float(eps), int(isign), int(nthreads), str(dtype),
           float(upsampfac), int(fftw_flag))
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = finufft.Plan(
            nufft_type=1,
            n_modes_or_dim=(bin_no,),
            isign=isign,
            eps=eps,
            nthreads=nthreads,
            dtype=dtype,
            upsampfac=upsampfac,
            fftw=fftw_flag,
        )
        _PLAN_CACHE[key] = plan
    return plan


def clear_plan_cache() -> None:
    """Drop all cached finufft plans (releases their internal buffers)."""
    _PLAN_CACHE.clear()


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
    def __init__(self, nthreads: int = 4, eps: float = 1e-6, *,
                 precision: str = "double", upsampfac: float = 1.25,
                 fftw_measure: bool = False, n_modes: int | None = None,
                 **kwws) -> None:
        """Constructor

        Parameters
        ----------
        nthreads : int
            Number of threads passed to the finufft plan. Defaults to 4.
        eps : float
            Requested NUFFT relative tolerance. Defaults to 1e-6 (finufft default).
        precision : {"double", "single"}, optional
            "single" runs the NUFFT in complex64 -- roughly 2x faster and
            half the memory, but limited to eps >= 1e-6. Defaults to "double".
        upsampfac : float, optional
            finufft upsampling factor. 1.25 is the fast/low-precision path
            (valid for eps >= 1e-6) and is the default here -- the finufft
            internal default is 2.0. Use 2.0 if you need eps < 1e-9.
        fftw_measure : bool, optional
            If True, plans use FFTW_MEASURE instead of FFTW_ESTIMATE: slower
            to plan but faster to execute. Plans are cached process-wide,
            so the planning cost is paid once per (size, precision, ...) combo.
        n_modes : int, optional
            Number of output Fourier modes. Defaults to the number of input
            samples. Set this when the desired uniform-``tau`` output grid
            has a different length from the input time series.
        """
        self.timeseries = None # Replace with dict if multiple detectors
        self.resampled_time = None # Replace with dict if multiple detectors
        self.nthreads = nthreads
        self.eps = eps
        self.precision = precision
        self.upsampfac = upsampfac
        self.fftw_flag = _FFTW_MEASURE if fftw_measure else _FFTW_ESTIMATE
        if n_modes is not None and n_modes <= 0:
            raise ValueError(f"n_modes must be positive, got {n_modes}.")
        self.n_modes = n_modes

    @property
    def _complex_dtype(self):
        return np.complex64 if self.precision == "single" else np.complex128

    @property
    def _real_dtype(self):
        return np.float32 if self.precision == "single" else np.float64

    @property
    def _plan_dtype_str(self) -> str:
        return "complex64" if self.precision == "single" else "complex128"

    def nufft(self) -> None:
        """Performs the NUFFT computing the Fourier frequncies and weights.
        Uses the (-) sign convention to match np.fft.fft.

        Built on the finufft plan interface (``finufft.Plan``) with
        multithreading (``self.nthreads``) for maximum throughput. Plans are
        cached process-wide, so repeated calls with the same problem size
        and precision skip the planning cost entirely.
        """
        signal = self.timeseries
        tau = self.resampled_time
        bin_no = self.n_modes if self.n_modes is not None else len(tau)

        # Rescale time to [-pi, pi)
        scale = (2*np.pi) / (tau[-1] - tau[0])
        tau_scaled = scale * (tau - tau[0])
        bins = np.arange(bin_no) - bin_no//2
        freqs = bins * scale

        plan = _get_or_make_plan(
            bin_no,
            eps=self.eps,
            isign=-1,
            nthreads=self.nthreads,
            dtype=self._plan_dtype_str,
            upsampfac=self.upsampfac,
            fftw_flag=self.fftw_flag,
        )
        plan.setpts(np.ascontiguousarray(tau_scaled, dtype=self._real_dtype))
        weights = plan.execute(np.ascontiguousarray(signal, dtype=self._complex_dtype))

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
        return self.weights/len(self.timeseries)

    @property
    def power(self) -> npt.NDArray[np.float64]:
        """Computes the Fourier power values"""
        return abs(self.weights)**2

    @property
    def power_normalized(self) -> npt.NDArray[np.float64]:
        """Returns the normalized Fourier powers such that a pure trig function has power=1 regardless of length"""
        # the normalization of a nufft and nufft_real are different because
        # of their different lengths
        return abs(self.weights/len(self.timeseries))**2

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
