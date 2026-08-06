# Semicoherent 0PN scaling notes

These are back-of-the-envelope scalings for the sensitivity envelopes in
`semicoherent_sensitivity_1d.py`.  They apply to a signal that traverses the
full interval \([f_{\rm start},f_{\rm end}]\), before the observation-time cap
and the maximisation over \(f_0\) are applied.

## Coherence time

A strictly 0PN signal has zero mismatch against a 0PN template, so 0PN alone
does **not** determine \(T_{\rm coh}\).  The coherence time in the plot is set
by the leading PN correction to the 0PN template.  At fixed mass ratio,

$$
\dot f_{\rm 0PN} \propto M_c^{5/3}f^{11/3},
\qquad
\frac{\Delta\dot f}{\dot f_{\rm 0PN}}\sim v^2\sim(Mf)^{2/3}.
$$

Aligning the models in phase and frequency at the start of a chunk gives

$$
\Delta\phi(T)\sim\Delta\dot f\,T^2,
\qquad
\mu\sim\left(\Delta\phi\right)^2.
$$

Thus a fixed waveform-mismatch allowance \(\mu_{\rm coh}\) gives

$$
T_{\rm coh}
\propto
\mu_{\rm coh}^{1/4}
M_{c,\max}^{-7/6}
f_{\rm end}^{-13/6}.
$$

The upper edge normally sets the worst mismatch, so the leading dependence on
\(f_{\rm start}\) is negligible.  For the present bank and mismatch criterion,
the numerical results are well described by

$$
T_{\rm coh}\simeq 32.25\,{\rm s}
\left(\frac{f_{\rm end}}{60\,{\rm Hz}}\right)^{-13/6}.
$$

This predicts \(10.7\,{\rm s}\) at \(100\,{\rm Hz}\) and
\(2.37\,{\rm s}\) at \(200\,{\rm Hz}\), close to the measured values
\(10.8\,{\rm s}\) and \(2.51\,{\rm s}\), respectively.

## Signal duration and number of chunks

The 0PN time to sweep across the band is

$$
T_{\rm sig}=
\frac{3}{8\beta}
\left[
1-\left(\frac{f_{\rm start}}{f_{\rm end}}\right)^{8/3}
\right]
\propto
M_c^{-5/3}
\left(
f_{\rm start}^{-8/3}-f_{\rm end}^{-8/3}
\right).
$$

Ignoring integer rounding and the maximum observation time,

$$
N_{\rm chunks}\sim\frac{T_{\rm sig}}{T_{\rm coh}}
\propto
M_c^{-5/3}
f_{\rm end}^{13/6}
\left(
f_{\rm start}^{-8/3}-f_{\rm end}^{-8/3}
\right).
$$

Writing \(r=f_{\rm start}/f_{\rm end}\), this becomes

$$
N_{\rm chunks}\propto
M_c^{-5/3}f_{\rm end}^{-1/2}
\left(r^{-8/3}-1\right).
$$

Hence, at fixed fractional bandwidth \(r\), moving a band to higher
frequencies mildly *reduces* its number of chunks: the chirp residence time
falls faster than the PN-limited coherence time.

## Noise-weighted chirp power

The quantity named `integrated_chirp_power` is matched-filter power, rather
than unweighted strain power.  For a full-band 0PN chirp it scales as

$$
P_{\rm chirp}\propto
M_c^{-5/3}f_{\rm start}^{-4/3}
\int_{f_{\rm start}}^{f_{\rm end}}
\frac{df}{f^{7/3}S_n(f)}.
$$

If the PSD is locally approximated by \(S_n(f)\propto f^\alpha\), then

$$
P_{\rm chirp}\propto
M_c^{-5/3}f_{\rm start}^{-4/3}
\frac{
f_{\rm start}^{-(4/3+\alpha)}
-f_{\rm end}^{-(4/3+\alpha)}
}{4/3+\alpha}.
$$

The detector PSD is not a single power law, so the integral should be
evaluated numerically for quantitative comparisons.

## Implication for the semicoherent envelope

For a large number of chunks, the noncentral-\(\chi^2\) threshold gives the
rough scaling

$$
d_{\rm sc}\propto \sqrt{P_{\rm chirp}}\,N_{\rm chunks}^{-1/4},
$$

apart from common mismatch factors.  A broad band gains power but usually has
a much shorter \(T_{\rm coh}\) and therefore many more chunks.  These effects
partially offset each other.  The plotted curves also maximise independently
over \(f_0\), so they do not necessarily accumulate power across the complete
labelled band.

## Example: 20--200 Hz versus 40--60 Hz

For a literal 0PN traversal from the lower edge to the upper edge, the present
coherence calculation gives

$$
T_{{\rm coh},20-200}=2.51\,{\rm s},
\qquad
T_{{\rm coh},40-60}=32.25\,{\rm s}.
$$

At \(M_c=10^{-3}\,M_\odot\), the full-band durations, before applying the
observation-time cap, are

$$
T_{{\rm sig},20-200}=2.191\times10^7\,{\rm s},
\qquad
T_{{\rm sig},40-60}=2.285\times10^6\,{\rm s}.
$$

Both durations scale as \(M_c^{-5/3}\).  The corresponding chunk counts are

$$
N_{20-200}=8.72\times10^6
\left(\frac{M_c}{10^{-3}M_\odot}\right)^{-5/3},
$$

$$
N_{40-60}=7.09\times10^4
\left(\frac{M_c}{10^{-3}M_\odot}\right)^{-5/3},
$$

so that

$$
\frac{N_{20-200}}{N_{40-60}}=123.
$$

The code's noise-weighted full-band powers are

$$
P_{20-200}=9.23\times10^{53}
\left(\frac{M_c}{10^{-3}M_\odot}\right)^{-5/3},
$$

$$
P_{40-60}=8.74\times10^{52}
\left(\frac{M_c}{10^{-3}M_\odot}\right)^{-5/3},
$$

and therefore

$$
\frac{P_{20-200}}{P_{40-60}}=10.55.
$$

Distance is not simply proportional to \(\sqrt{P_{\rm chirp}}\), since the
amplitude prefactor has a different \(f_{\rm start}\) dependence in the two
bands.  For coherent distance this dependence cancels the extra
\(f_{\rm start}^{-4/3}\) in \(P_{\rm chirp}\), leaving

$$
\frac{d_{{\rm coh},20-200}}{d_{{\rm coh},40-60}}
=
\left[
\frac{
\displaystyle\int_{20}^{200}df/(f^{7/3}S_n(f))
}{
\displaystyle\int_{40}^{60}df/(f^{7/3}S_n(f))
}
\right]^{1/2}
=2.05.
$$

For many chunks, the semicoherent approximation then predicts

$$
\frac{d_{{\rm sc},20-200}}{d_{{\rm sc},40-60}}
\simeq
2.05\,(123)^{-1/4}
=0.61.
$$

Thus the simple calculation supports the bands having comparable
order-of-magnitude distance reach, while predicting that 40--60 Hz should be
better by a factor of roughly \(1/0.61\simeq1.6\), rather than predicting
identical sensitivity.  Using the exact noncentral-\(\chi^2\) thresholds in
the code for full sweeps gives ratios

$$
\frac{d_{{\rm sc},20-200}}{d_{{\rm sc},40-60}}
\simeq
\begin{cases}
0.62, & M_c=10^{-3}M_\odot,\\
0.63, & M_c=10^{-2}M_\odot,\\
0.73, & M_c=10^{-1}M_\odot.
\end{cases}
$$

This comparison does not apply unchanged at \(M_c=10^{-4}M_\odot\): the
full-sweep times are \(1.02\times10^9\,{\rm s}\) and
\(1.06\times10^8\,{\rm s}\), respectively, so both exceed the
\(3\times10^7\,{\rm s}\) observation cap.  In addition, the plotted
envelopes maximise over \(f_0\).  In particular, the 20--200 Hz envelope can
select an \(f_0\) far above 20 Hz, so its plotted result need not follow this
fixed-lower-edge full-band comparison.

## Maximising over \(f_0\) within a band

The envelope is not tied to the lower frequency edge.  For a fixed upper edge
\(f_{\rm end}\), define

$$
I(f_0,f_{\rm end})
=
\int_{f_0}^{f_{\rm end}}
\frac{df}{f^{7/3}S_n(f)},
\qquad
D(f_0,f_{\rm end})
=f_0^{-8/3}-f_{\rm end}^{-8/3}.
$$

For a full, uncapped track, the 0PN expressions become

$$
P_{\rm chirp}(f_0,f_{\rm end})
\propto M_c^{-5/3}f_0^{-4/3}I(f_0,f_{\rm end}),
$$

$$
d_{\rm coh}(f_0,f_{\rm end})
\propto M_c^{5/6}\sqrt{I(f_0,f_{\rm end})},
$$

and

$$
N(f_0,f_{\rm end})
\propto
\frac{M_c^{-5/3}}{T_{\rm coh}}
D(f_0,f_{\rm end}).
$$

The explicit \(f_0\) factors cancel from the coherent distance.  Raising
\(f_0\) therefore loses coherent power only through the discarded part of the
noise-weighted integral \(I\), but it reduces the number of chunks much more
rapidly through \(D\).  In the large-\(N\) limit,

$$
d_{\rm sc}(f_0,f_{\rm end})
\propto
M_c^{5/4}T_{\rm coh}^{1/4}
\frac{\sqrt{I(f_0,f_{\rm end})}}
{D(f_0,f_{\rm end})^{1/4}}.
$$

Thus increasing \(f_0\) initially *increases* semicoherent range: the
\(N^{-1/4}\) gain can outweigh the loss of \(\sqrt{I}\).  Close to
\(f_{\rm end}\), both \(I\) and \(D\) vanish; the range scales as
\((f_{\rm end}-f_0)^{1/4}\) and falls to zero.  The maximum is consequently
at an interior \(f_0\), or at the permitted lower edge if the interior maximum
is below it.  Its condition is

$$
I(f_0,f_{\rm end})
=
\frac{3}{4}
\frac{f_0^{4/3}}{S_n(f_0)}
\left[f_0^{-8/3}-f_{\rm end}^{-8/3}\right].
$$

### Recovering the 20--200 Hz versus 40--60 Hz similarity

Around \(M_c\sim10^{-3}M_\odot\), the numerical 20--200 Hz envelope selects
\(f_0\simeq83\,{\rm Hz}\), whereas the 40--60 Hz envelope selects its lower
edge, \(f_0=40\,{\rm Hz}\).  The large-\(N\) scaling predicts the ratio

$$
\frac{d_{{\rm sc},20-200}(83,200)}
{d_{{\rm sc},40-60}(40,60)}
=
\left(\frac{2.513}{32.246}\right)^{1/4}
\left(\frac{1.044\times10^{44}}{6.932\times10^{43}}\right)^{1/2}
\left(\frac{6.898\times10^{-6}}{3.531\times10^{-5}}\right)^{-1/4}
=0.98.
$$

Here the second factor is the coherent-power contribution \(\sqrt{I}\), and
the third is the chunk-count contribution \(D^{-1/4}\).  Relative to starting
the wide band at \(20\,{\rm Hz}\), moving it to \(83\,{\rm Hz}\) discards
about \(64\%\) of \(I\), but reduces \(D\), and hence \(N\), by a factor
of about \(49\).  The resulting semicoherent gain is

$$
\sqrt{0.360}\,(49)^{1/4}\simeq1.59.
$$

It raises the fixed-lower-edge estimate of \(0.61\) to approximately unity.
This reproduces the order-unity agreement of the green and yellow envelopes.
For the higher-mass part of the plot, choosing \(f_0\) between roughly
\(64\) and \(83\,{\rm Hz}\) gives the same estimate, \(0.96\)--\(0.98\).
At low masses the observation-time cap changes these scalings, so the result
there requires the full numerical envelope rather than the full-track formula.
