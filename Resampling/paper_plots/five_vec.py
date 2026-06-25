"""Defining the core :class:`five_vec` class.
"""
__all__ = ["five_vec"]

import numpy as np

class five_vec(object):
    def __init__(self, **kws) -> None:
        """Constructor"""
        self.ra = kws.get("ra", None) #rad
        self.dec = kws.get("dec", None) #rad
        self.eta = kws.get("eta", None) #[-1,1]
        self.psi = kws.get("psi", None) #rad

        self.lat = kws.get("lat", None) #rad
        self.lng = kws.get("lng", None) #rad
        self.az = kws.get("az", None) #rad
        self.side_day = kws.get("side_day", 86164.09053083288) #s
        
    def compute_A(self, omega_t) -> None:
        """
        omega_t is Greenwich mean sidereal time in rad
        """
        c_dec=np.cos(self.dec)
        c_lat=np.cos(self.lat)
        s_dec=np.sin(self.dec)
        s_lat=np.sin(self.lat)

        c_2az=np.cos(2*self.az)
        c_2dec=np.cos(2*self.dec)
        c_2lat=np.cos(2*self.lat)
        s_2az=np.sin(2*self.az)
        s_2dec=np.sin(2*self.dec)
        s_2lat=np.sin(2*self.lat)
        
        a0=-(3/16)*(1+c_2dec)*(1+c_2lat)*c_2az
        a1c=-(1/4)*s_2dec*s_2lat*c_2az
        a1s=-(1/2)*s_2dec*c_lat*s_2az
        a2c=(-1/16)*(3-c_2dec)*(3-c_2lat)*c_2az
        a2s=-(1/4)*(3-c_2dec)*s_lat*s_2az

        b1c=-c_dec*c_lat*s_2az
        b1s=(1/2)*c_dec*s_2lat*c_2az
        b2c=-s_dec*s_lat*s_2az
        b2s=(1/4)*s_dec*(3-c_2lat)*c_2az
        
        A_p = np.empty((5), dtype=complex)
        A_c = np.empty((5), dtype=complex)
        al=np.exp(-1j*(self.ra-self.lng)) #=e^{-j(\alpha-\beta)} in Eq.(17)-(18)
        
        scalar_Aplus = a0+a1c*np.cos(omega_t)+a1s*np.sin(omega_t)+\
                a2c*np.cos(2*omega_t)+a2s*np.sin(2*omega_t)
        scalar_Across = b1c*np.cos(omega_t)+b1s*np.sin(omega_t)+\
                b2c*np.cos(2*omega_t)+b2s*np.sin(2*omega_t)
        self.scalar_Aplus = scalar_Aplus
        self.scalar_Across = scalar_Across
        self.amp_modulation = scalar_Aplus*self.H_p+scalar_Across*self.H_c

        A_p[0]=(al**-2)*(a2c+1j*a2s)/2
        A_p[1]=(al**-1)*(a1c+1j*a1s)/2
        A_p[2]=a0
        A_p[3]=(al)*(a1c-1j*a1s)/2
        A_p[4]=(al**2)*(a2c-1j*a2s)/2

        A_c[0]=(al**-2)*(b2c+1j*b2s)/2
        A_c[1]=(al**-1)*(b1c+1j*b1s)/2
        A_c[2]=0
        A_c[3]=(al)*(b1c-1j*b1s)/2
        A_c[4]=(al**2)*(b2c-1j*b2s)/2
        
        self.A_p = A_p
        self.A_c = A_c
        
        # Correct for initial phase. 
        # This removes the al term and makes it independent of the initial omega_t value
        # A \cdot W in Eq. (15)
        phi = omega_t[0] + self.ra - self.lng # Eq. (13)-(14)
        for i in range(5):
            self.A_p[i] = self.A_p[i] * np.exp(1j*(i-2)*phi)
            self.A_c[i] = self.A_c[i] * np.exp(1j*(i-2)*phi)
        
    def compute_H(self, **kws) -> None:
        # Eq. (2)-(3)
        eta = kws.get("eta", self.eta)
        psi = kws.get("psi", self.psi)
        self.H_p=np.sqrt(1/(1+eta**2))*(np.cos(2*self.psi)-1j*eta*np.sin(2*self.psi))
        self.H_c=np.sqrt(1/(1+eta**2))*(np.sin(2*self.psi)+1j*eta*np.cos(2*self.psi))
        
    def compute_5vec(self) -> None:
        # Eq. (16)
        self.A = self.H_p*self.A_p + self.H_c*self.A_c

    def gmst(self, t) -> np.float64:
        """
        %GMST  Greenwich mean sidereal time (in rad)
        %
        %   t   time (in JD or mjd)
        %
        % add longitude in hours (deg/15) to have local sidereal time 
        Adapted from Snag v2.0 by Sergio Frasca
        """
        t = np.asarray(t)
        jd = np.where(t > 1000000, t + 2400000.5, t)
        
        jd0=np.floor(jd-0.5)+0.5;
        h=(jd-jd0)*24;
        
        d=jd-2451545;
        d0=jd0-2451545;
        T=d/36525;
        
        st=np.mod(6.697374558+0.06570982441908*d0+1.00273790935*h+0.000026*T**2,24)
        return st/12*np.pi