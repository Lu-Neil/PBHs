from ..five_vec import five_vec
import numpy as np
from astropy.time import Time


class Test_five_vec:
    def generate_test_signal(self):
        h0 = np.random.uniform(1, 5)
        params = dict(ra = np.random.uniform(0, 2*np.pi), 
                      dec = np.random.uniform(-np.pi/2, np.pi/2), 
                      eta = np.random.uniform(-1, 1), 
                      psi = np.random.uniform(0, 2*np.pi), 
                      lat = np.random.uniform(-np.pi/2, np.pi/2), 
                      lng = np.random.uniform(-np.pi, np.pi),
                      az = np.random.uniform(0, 2*np.pi),
                     )
        sidereal = five_vec(**params)
        
        number_of_days = 2 # needs to be integer
        T_obs = number_of_days*sidereal.side_day         
        f0 = 1/sidereal.side_day * 10000
        ref_time = Time(f'2019-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}T{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}:{np.random.uniform(0, 60):06.3f}')
        f_signal = 8*f0
        nt = round(f_signal*T_obs)
        t = ref_time.gps + np.arange(nt)/f_signal
        t = Time(t, format="gps", scale="utc")

        sidereal.compute_H()
        sidereal.compute_A(sidereal.gmst(t.mjd))
        sidereal.compute_5vec()
        amp_modulation = h0 * sidereal.amp_modulation

        omega0 = 2*np.pi*f0
        gamma = np.random.uniform(0, 2*np.pi)
        t_space = t.gps - t.gps[0]
        mono = 1*np.exp(1j*(omega0*t_space+gamma))
        detector = amp_modulation * mono

        self.t = t
        self.sidereal = sidereal
        self.amp_modulation = amp_modulation
        self.h0 = h0
        self.detector = detector
        self.f0 = f0
        self.gamma = gamma
        self.side_day = sidereal.side_day

    def fft(self, time, signal):
        """ Return normalized amplitude """
        fft_freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), d=np.diff(time)[0]))
        fft_amps = np.fft.fftshift(np.fft.fft(signal))/len(signal)
        return fft_freqs, fft_amps

    def extract_5vec(self, time, signal, f0):
        fft_freqs, fft_amps = self.fft(time, signal)
        
        X = np.empty((5), dtype=complex)
        X[0] = fft_amps[abs(fft_freqs-(f0-2/self.side_day)).argmin()]
        X[1] = fft_amps[abs(fft_freqs-(f0-1/self.side_day)).argmin()]
        X[2] = fft_amps[abs(fft_freqs-f0).argmin()]
        X[3] = fft_amps[abs(fft_freqs-(f0+1/self.side_day)).argmin()]
        X[4] = fft_amps[abs(fft_freqs-(f0+2/self.side_day)).argmin()]
        return X

    def estimator(self, X, A_template):
        return np.dot(X, np.conj(A_template))/np.sum(np.abs(A_template)**2)

    def test_amp_modulation(self):
        """ Checks that the 5 vector of the amplitude modulation works """
        self.generate_test_signal()
        X = self.extract_5vec(self.t.gps, self.amp_modulation, 0)
        h_est = self.estimator(X, self.sidereal.A)
        hp_est = self.estimator(X, self.sidereal.A_p)
        hc_est = self.estimator(X, self.sidereal.A_c)
        assert np.isclose(h_est, self.h0) # no gamma in amp_modulation
        assert np.isclose(hp_est, self.h0*self.sidereal.H_p)
        assert np.isclose(hc_est, self.h0*self.sidereal.H_c)

    def test_detector(self):
        """ Checks that the 5 vector of the detector = amp_modulation * monochromatic signal works """
        self.generate_test_signal()
        X = self.extract_5vec(self.t.gps, self.detector, self.f0)
        h_est = self.estimator(X, self.sidereal.A)
        hp_est = self.estimator(X, self.sidereal.A_p)
        hc_est = self.estimator(X, self.sidereal.A_c)
        assert np.isclose(h_est, self.h0*np.exp(1j*self.gamma))
        assert np.isclose(hp_est, self.h0*self.sidereal.H_p*np.exp(1j*self.gamma))
        assert np.isclose(hc_est, self.h0*self.sidereal.H_c*np.exp(1j*self.gamma))

    def test_time_domain(self):
        """ Checks that matched filtering with a time-domain template works """
        self.generate_test_signal()
        sidereal_t = self.sidereal.gmst(self.t.mjd)
        sidereal_t -= sidereal_t[0] # accounts for L77 of five_vec.py which removes the initial phase
        
        exp_terms = np.exp(1j*(np.arange(5)-2)[:, np.newaxis]*sidereal_t)
        template = np.dot(self.sidereal.A, exp_terms)
        template_p = np.dot(self.sidereal.A_p, exp_terms)
        template_c = np.dot(self.sidereal.A_c, exp_terms)
        
        template_X = self.extract_5vec(self.t.gps, template, 0)
        template_Xp = self.extract_5vec(self.t.gps, template_p, 0)
        template_Xc = self.extract_5vec(self.t.gps, template_c, 0)
        X = self.extract_5vec(self.t.gps, self.detector, self.f0)
        
        h_est = self.estimator(X, template_X)
        hp_est = self.estimator(X, template_Xp)
        hc_est = self.estimator(X, template_Xc)
        
        assert np.isclose(h_est, self.h0*np.exp(1j*self.gamma))
        assert np.isclose(hp_est, self.h0*self.sidereal.H_p*np.exp(1j*self.gamma))
        assert np.isclose(hc_est, self.h0*self.sidereal.H_c*np.exp(1j*self.gamma))
        breakpoint()