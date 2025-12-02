"""Defining the core :class:`five_vec` class.
"""
__all__ = ["five_vec"]

import numpy as np
from numpy.typing import NDArray


class five_vec(object):
    def __init__(self, **kws) -> None:
        """Constructor"""
        self.ra = kws.get("ra", None)  # rad
        self.dec = kws.get("dec", None)  # rad
        self.eta = kws.get("eta", None)  # [-1,1]
        self.psi = kws.get("psi", None)  # rad

        self.lat = kws.get("lat", None)  # rad
        self.lng = kws.get("lng", None)  # rad
        self.az = kws.get("az", None)  # rad
        self.side_day = kws.get("side_day", 86164.09053083288)  # s

    def compute_A(self, t) -> None:
        # Trig functions used in Eq. (11)-(12) of Astone2010
        # Lines 39-49 of sour_ant_2_5vec.m
        c_dec = np.cos(self.dec)
        c_lat = np.cos(self.lat)
        s_dec = np.sin(self.dec)
        s_lat = np.sin(self.lat)

        c_2az = np.cos(2*self.az)
        c_2dec = np.cos(2*self.dec)
        c_2lat = np.cos(2*self.lat)
        s_2az = np.sin(2*self.az)
        s_2dec = np.sin(2*self.dec)
        s_2lat = np.sin(2*self.lat)

        # Eq. (11)-(12) of Astone2010
        # Lines 51-60 of sour_ant_2_5vec.m
        a0 = -(3/16)*(1+c_2dec)*(1+c_2lat)*c_2az
        a1c = -(1/4)*s_2dec*s_2lat*c_2az
        a1s = -(1/2)*s_2dec*c_lat*s_2az
        a2c = (-1/16)*(3-c_2dec)*(3-c_2lat)*c_2az
        a2s = -(1/4)*(3-c_2dec)*s_lat*s_2az

        b1c = -c_dec*c_lat*s_2az
        b1s = (1/2)*c_dec*s_2lat*c_2az
        b2c = -s_dec*s_lat*s_2az
        b2s = (1/4)*s_dec*(3-c_2lat)*c_2az

        # Eq. (17)-(18) of Astone2010
        # Lines 62-72 of sour_ant_2_5vec.m
        A_p = np.empty((5), dtype=complex)
        A_c = np.empty((5), dtype=complex)
        al = np.exp(-1j*(self.ra-self.lng))
        # =e^{-j(\alpha-\beta)} in Eq.(17)-(18)

        A_p[0] = (al**-2)*(a2c+1j*a2s)/2
        A_p[1] = (al**-1)*(a1c+1j*a1s)/2
        A_p[2] = a0
        A_p[3] = (al)*(a1c-1j*a1s)/2
        A_p[4] = (al**2)*(a2c-1j*a2s)/2

        A_c[0] = (al**-2)*(b2c+1j*b2s)/2
        A_c[1] = (al**-1)*(b1c+1j*b1s)/2
        A_c[2] = 0
        A_c[3] = (al)*(b1c-1j*b1s)/2
        A_c[4] = (al**2)*(b2c-1j*b2s)/2

        self.A_p = A_p
        self.A_c = A_c

        side_day = self.side_day
        side_omega = 2*np.pi/side_day

        self.scalar_Aplus = a0+a1c*np.cos(side_omega*t)+a1s*np.sin(side_omega*t) +\
            a2c*np.cos(2*side_omega*t)+a2s*np.sin(2*side_omega*t)
        self.scalar_Across = b1c*np.cos(side_omega*t)+b1s*np.sin(side_omega*t) +\
            b2c*np.cos(2*side_omega*t)+b2s*np.sin(2*side_omega*t)
        self.amp_modulation = self.scalar_Aplus*self.H_p+self.scalar_Across*self.H_c
        # import matplotlib.pyplot as pl
        # pl.plot(t, np.real(self.scalar_Aplus))
        # pl.savefig("figs/test.png")
        # breakpoint()

    def compute_H(self, **kws) -> None:
        eta = kws.get("eta", self.eta)
        psi = kws.get("psi", self.psi)
        # Eq. (2)-(3) of Astone2010
        # Lines 83-84 of sour_ant_2_5vec.m
        self.H_p = np.sqrt(1/(1+eta**2)) * \
            (np.cos(2*psi)-1j*eta*np.sin(2*psi))
        self.H_c = np.sqrt(1/(1+eta**2)) * \
            (np.sin(2*psi)+1j*eta*np.cos(2*psi))

    def compute_5vec(self) -> None:
        # Eq. (16) of Astone2010
        # Line 85 of sour_ant_2_5vec
        self.A = self.H_p*self.A_p + self.H_c*self.A_c

    def H_calc(self, eta, psi) -> tuple:
        Hp = np.sqrt(1/(1+eta**2)) * \
            (np.cos(2*psi)-1j*eta*np.sin(2*psi))
        Hc = np.sqrt(1/(1+eta**2)) * \
            (np.sin(2*psi)+1j*eta*np.cos(2*psi))
        return (Hp, Hc)

    def check_ps_lf(self, t) -> NDArray:
        # Trig functions used in Eq. (11)-(12) of Astone2010
        # Lines 39-49 of sour_ant_2_5vec.m
        eta0 = 1
        psi0 = 0
        eta45 = -1
        psi45 = 0

        Hp0, Hc0 = self.H_calc(eta0, psi0)
        Hp45, Hc45 = self.H_calc(eta45, psi45)
        A0 = Hp0*self.A_p * Hc0*self.A_c
        A45 = Hp45*self.A_p * Hc45*self.A_c
        breakpoint()
