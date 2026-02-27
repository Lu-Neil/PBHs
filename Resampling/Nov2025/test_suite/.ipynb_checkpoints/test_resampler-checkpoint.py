from ..resampler import Resampler
import numpy as np

class Test_resampler:
    def generate_mono_signal(self):
        t = np.linspace(0*np.pi,2*np.pi, int(2**16))
        f0 = 8 # rad/s
        phi = f0*t
        signal = np.real(1*np.exp(-1j*phi))
        self.signal = signal
        self.t = t
        self.f0 = f0

    def fft(self, time, signal):
        """ Return normalized amplitude """
        fft_freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), d=np.diff(time)[0]))
        fft_amps = np.fft.fftshift(np.fft.fft(signal))/len(signal)
        return fft_freqs, fft_amps

    def test_mono(self):
        self.generate_mono_signal()
        fft_freqs, fft_amps = self.fft(self.t, self.signal)
        
        resampler = Resampler()
        resampler.timeseries = self.signal
        resampler.resampled_time = self.t
        resampler.nufft()
        # assert np.allclose(resampler.weights_normalized, fft_amps)