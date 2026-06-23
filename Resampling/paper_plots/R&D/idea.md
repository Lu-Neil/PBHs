---
title: Optimal ways to semicoherently search for modelled signals
status: early
created: 2026-06-14
---

# Core idea:
Semicoherent analysis conventionally uses a frequency-time spectra and searches for templates which specify the frequencies of a modelled signal at various times. Rather than constructing a two-dimensional time-frequency map and summing power along template tracks, we can construct spectra in a higher-dimentsional space and create a semicoherent statistic obtained by accumulating power along the template track in this higher-dimensional space. 

I.e. if a signal phase model consists of a set of parameters $\phi(\theta)$, we can create spectra which depend on a different set of parameters $\zeta$ (which include $f$, $t$). What is the computationally optimal way to choose how many parameters there are in $\zeta$? Note $\zeta$ can consist of linear combinations of the parameters in $\theta$. This should depend on the size of the coherent chunks $T_{\rm coh}$. 

E.g. we consider the 3.5PN gravitational waveform. In the directory above we implement an example where $\zeta = \{f,t,\beta\}$ where $\beta$ is a 0PN parameter. But we could have chosen to use 1PN signal models in the coherent chunks and have more parameters in $\zeta$. Doing this would have increased the maximum possible coherence time, but possibly also increased the computational cost which comes from both the creating spectra and then semicoherently combining by summing over frequency tracks. 

This is a signals processing question which may already have solved answers.  

# Notes:
The file /home/neil-lu/Dropbox/PBHs/Codebase/CODEX.md has useful information. Make a note of any useful papers which are not in Zotero already. 

# Related techniques:
- StackSlide
- PowerFlux
- Weave
- Multi-bank template analysis (MBTA)
- Reduced order quadrature
- Principal component analysis

# Desired outputs

A research report which includes:
- Literature review
- Mathematical formalism
- Sensitivity analysis
- Computational cost estimation