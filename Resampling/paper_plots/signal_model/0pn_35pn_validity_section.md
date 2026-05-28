# Validity of the 0PN phase approximation over a finite coherent window

Post-Newtonian (PN) waveforms expand the compact-binary inspiral in powers of the orbital velocity. In the TaylorT4 convention it is natural to use
$$
v=\left(\frac{G M_\mathrm{tot}\Omega}{c^3}\right)^{1/3},
$$
where $M_\mathrm{tot}=m_1+m_2$ is the total mass and $\Omega=d\Phi/dt$ is the orbital angular frequency. For the dominant $m=2$ quadrupole mode, the gravitational-wave phase and frequency are
$$
\phi_\mathrm{GW}(t)=2\Phi(t),
\qquad
f_\mathrm{GW}(t)=\frac{\Omega(t)}{\pi}
=\frac{c^3}{\pi G M_\mathrm{tot}}v(t)^3 .
$$
Thus, in TaylorT4, the frequency is obtained directly from the time-domain orbital evolution. The stationary-phase approximation is only needed if one wants to convert this time-domain chirp into a frequency-domain TaylorF2-like phase.

Higher-PN terms are formally small when $v\ll1$, but a coherent search is sensitive to accumulated phase error, not only to the local size of a PN correction. A small correction to $dv/dt$ can accumulate over many cycles into a large dephasing. We therefore judge the 0PN model by its accumulated time-domain phase error against a 3.5PN TaylorT4 track over a finite coherent window.

## TaylorT4 time-domain phase convention

TaylorT4 starts from the PN-expanded energy balance law and re-expands the velocity evolution as an ordinary differential equation. For a nonspinning, quasi-circular binary through 3.5PN,
$$
\frac{dv}{dt}
=
\frac{32\eta}{5M_s}v^9
F_\mathrm{T4}(v),
\qquad
M_s=\frac{G M_\mathrm{tot}}{c^3},
\qquad
\eta=\frac{m_1m_2}{M_\mathrm{tot}^2},
$$
with
$$
F_\mathrm{T4}(v)
=
1+a_2v^2+a_3v^3+a_4v^4+a_5v^5
+\left(a_6+a_{6\ell}\log v\right)v^6
+a_7v^7 .
$$
For the nonspinning point-particle terms used by LAL,
$$
a_2=-\frac{743+924\eta}{336},
\qquad
a_3=4\pi,
$$
$$
a_4=\frac{34103+122949\eta+59472\eta^2}{18144},
$$
$$
a_5=-\frac{\pi}{672}\left(4159+15876\eta\right),
$$
$$
a_6=
\frac{16447322263}{139708800}
-\frac{1712}{105}\gamma_E
-\frac{856}{105}\log 16
-\frac{56198689}{217728}\eta
+\pi^2\left(\frac{16}{3}+\frac{451}{48}\eta\right)
+\frac{541}{896}\eta^2
-\frac{5605}{2592}\eta^3,
$$
$$
a_{6\ell}=-\frac{1712}{105},
$$
$$
a_7=\frac{\pi}{12096}
\left(
-13245
+717350\eta
+731960\eta^2
\right).
$$
LAL's TaylorT4 source stores these as angular-acceleration coefficients and notes that $(d\Omega/dt)/\Omega=3(dv/dt)/v$, which gives the velocity equation above.

The GW phase is then evolved by
$$
\frac{d\phi_\mathrm{GW}}{dt}
=2\Omega
=\frac{2}{M_s}v^3 .
$$
Together, $dv/dt$ and $d\phi_\mathrm{GW}/dt$ define the time-domain 3.5PN TaylorT4 phase model.

Equivalently, the phase can be written as a PN expansion in the instantaneous velocity. Combining the two TaylorT4 evolution equations gives
$$
\frac{d\phi_\mathrm{GW}}{dv}
=
\frac{5}{16\eta}v^{-6}\frac{1}{F_\mathrm{T4}(v)} .
$$
Expanding the reciprocal of $F_\mathrm{T4}$ through 3.5PN,
$$
\frac{1}{F_\mathrm{T4}(v)}
=
1+b_2v^2+b_3v^3+b_4v^4+b_5v^5
+\left(b_6+b_{6\ell}\log v\right)v^6
+b_7v^7,
$$
where
$$
b_2=-a_2,
\qquad
b_3=-a_3,
$$
$$
b_4=a_2^2-a_4,
\qquad
b_5=2a_2a_3-a_5,
$$
$$
b_6=-a_6+2a_2a_4+a_3^2-a_2^3,
\qquad
b_{6\ell}=-a_{6\ell},
$$
$$
b_7=-a_7+2a_2a_5+2a_3a_4-3a_2^2a_3 .
$$
The integrated phase from $v_a$ to $v$ is therefore
$$
\phi_\mathrm{GW}(v)-\phi_\mathrm{GW}(v_a)
=
\frac{5}{16\eta}
\left[
-\frac{1}{5}v^{-5}
-\frac{b_2}{3}v^{-3}
-\frac{b_3}{2}v^{-2}
-b_4v^{-1}
+b_5\log v
+b_6v
+b_{6\ell}\left(v\log v-v\right)
+\frac{b_7}{2}v^2
\right]_{v_a}^{v}.
$$
The time-domain phase is then obtained by composing this expression with the TaylorT4 velocity track:
$$
\phi_\mathrm{GW}(t)=\phi_\mathrm{GW}\!\left(v_\mathrm{T4}(t)\right).
$$

## The 0PN limit

At 0PN, $F_\mathrm{T4}(v)=1$, so
$$
\frac{dv_0}{dt}
=
\frac{32\eta}{5M_s}v_0^9 .
$$
If $v_0(0)=v_a$, the solution is
$$
v_0(t)
=
v_a
\left[
1-\frac{256\eta}{5M_s}v_a^8t
\right]^{-1/8}.
$$
The corresponding GW frequency is
$$
f_0(t)
=
\frac{1}{\pi M_s}v_0(t)^3 .
$$
The 0PN GW phase accumulated from $v_a$ to $v$ is
$$
\phi_0(v)-\phi_0(v_a)
=
\frac{1}{16\eta}
\left(v_a^{-5}-v^{-5}\right).
$$
Equivalently, evaluating this at $v=v_0(t)$ gives $\phi_0(t)$.

## Accumulated 0PN--3.5PN dephasing

Let $v_\mathrm{T4}(t)$ be the solution of the 3.5PN TaylorT4 equation with the same initial frequency as the 0PN model:
$$
v_\mathrm{T4}(0)=v_0(0)=v_a,
\qquad
\phi_\mathrm{T4}(0)=\phi_0(0).
$$
The time-domain phase difference over a coherent window of duration $T$ is
$$
\Delta\phi_{0\mathrm{PN}\rightarrow3.5\mathrm{PN}}(T)
=
\phi_\mathrm{T4}(T)-\phi_0(T)
$$
with
$$
\Delta\phi_{0\mathrm{PN}\rightarrow3.5\mathrm{PN}}(T)
=
\frac{2}{M_s}
\int_0^T
\left[
v_\mathrm{T4}(t)^3-v_0(t)^3
\right]dt .
$$
This is the clean TaylorT4 convention: compare the two time-domain phases at the same physical time after aligning them at the start of the window.

It is also useful to write the 3.5PN TaylorT4 phase parametrically as a function of the final TaylorT4 velocity $v_b$:
$$
T_\mathrm{T4}(v_b;v_a)
=
\frac{5M_s}{32\eta}
\int_{v_a}^{v_b}
\frac{v^{-9}}{F_\mathrm{T4}(v)}\,dv,
$$
$$
\phi_\mathrm{T4}(v_b)-\phi_\mathrm{T4}(v_a)
=
\frac{5}{16\eta}
\int_{v_a}^{v_b}
\frac{v^{-6}}{F_\mathrm{T4}(v)}\,dv .
$$
The 0PN comparison at the same elapsed time is obtained by first setting
$$
v_0(T_\mathrm{T4})
=
v_a
\left[
1-\frac{256\eta}{5M_s}v_a^8T_\mathrm{T4}
\right]^{-1/8},
$$
and then using the 0PN phase formula above.

For a simpler same-frequency diagnostic, one may compare the accumulated phases between the same endpoints $v_a$ and $v_b$:
$$
\Delta\phi_\mathrm{same\ freq}(v_b;v_a)
=
\frac{5}{16\eta}
\int_{v_a}^{v_b}
v^{-6}
\left[
\frac{1}{F_\mathrm{T4}(v)}-1
\right]dv .
$$
This same-frequency expression is useful for estimating the PN contribution accumulated over a frequency band, but the coherent resampling loss should be evaluated with the same-time expression when the template and signal are both evolved from the same starting frequency.

## Power loss from phase dephasing

For a coherent matched filter or resampled Fourier bin, the recovered complex amplitude is reduced by the weighted phasor average of the residual phase error. If $w(t)$ denotes the relevant positive weight over the coherent window, for example $w=A^2/S_n$, then
$$
\frac{P_\mathrm{rec}}{P_\mathrm{opt}}
=
\left|
\frac{\int_0^T w(t)e^{i\Delta\phi(t)}\,dt}
{\int_0^T w(t)\,dt}
\right|^2,
\qquad
\mathcal{L}=1-\frac{P_\mathrm{rec}}{P_\mathrm{opt}} .
$$
After maximizing over an arbitrary constant phase, the small-dephasing expansion is
$$
\mathcal{L}\simeq
\left\langle
\left(\Delta\phi-\langle\Delta\phi\rangle_w\right)^2
\right\rangle_w ,
$$
where $\langle\cdot\rangle_w$ is the $w$-weighted average. If the residual phase is approximately a linear ramp from $0$ to $\Delta\phi_\mathrm{end}$ over the window and the weights are slowly varying, then
$$
\frac{P_\mathrm{rec}}{P_\mathrm{opt}}
\simeq
\operatorname{sinc}^2\left(\frac{\Delta\phi_\mathrm{end}}{2}\right),
\qquad
\mathcal{L}
\simeq
1-\operatorname{sinc}^2\left(\frac{\Delta\phi_\mathrm{end}}{2}\right),
$$
with $\operatorname{sinc}x=\sin x/x$. For small endpoint dephasing this gives
$$
\mathcal{L}\simeq \frac{\Delta\phi_\mathrm{end}^2}{12}.
$$

## Coherent-window criterion

The 0PN approximation is not globally valid merely because $v\ll1$. Over a long band the PN correction can accumulate many radians of phase, as shown by the 1PN example in `velcani_vs_pn_chirp.pdf`. The useful statement is local: choose the coherent duration $T$ so that
$$
\left|
\Delta\phi_{0\mathrm{PN}\rightarrow3.5\mathrm{PN}}(T)
\right|
<\pi .
$$
Any shorter window has less than $\pi$ radians of same-time dephasing between the 0PN and 3.5PN TaylorT4 tracks. The expected coherent power loss is then evaluated from the phasor formula above. Under the linear-ramp approximation, a window with endpoint dephasing $\Delta\phi_\mathrm{end}$ retains
$$
\operatorname{sinc}^2\left(\frac{\Delta\phi_\mathrm{end}}{2}\right)
$$
of the optimal coherent power. Thus the 0PN model can be justified for semicoherent analyses by choosing the segment length so that the TaylorT4 3.5PN-minus-0PN dephasing remains below the desired phase or power-loss budget.

## Implementation references

- Local note checked: `velcani_vs_pn_chirp.pdf` / `velcani_vs_pn_chirp.tex`.
- LALSuite TaylorT4 source checked: `LALSimInspiralTaylorT4.c`, which evolves the PN orbit using the TaylorT4 method and stores the angular-acceleration coefficients: <https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_inspiral_taylor_t4_8c.html>.
- LALSuite coefficient source checked: `LALSimInspiralPNCoefficients.c`, where the nonspinning TaylorT4 coefficients are defined and documented as the coefficients of the TaylorT4 frequency equation: <https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_inspiral_p_n_coefficients_8c_source.html>.
