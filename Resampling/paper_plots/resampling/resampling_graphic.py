# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: PBH
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
pl.style.use("../paper.mplstyle")

pi = np.pi

# %% [markdown]
# ## Constant f

# %%
f0 = 0.5*2*pi
t_space = np.linspace(0, 10, 2**10)
signal = np.sin(f0*t_space)
mark_idx = np.arange(0, len(signal), 30)
colorbar = mpl.colormaps['hsv']

# %% [markdown]
# ## Changing f

# %%
t_space = np.linspace(0, 8, 2**10)
f = 0.5*2*pi + 0.003*t_space**3
signal = np.sin(f*t_space)
mark_idx = np.arange(0, len(signal), 30)
tau = (1+0.003*t_space**3/(0.5*2*pi))*t_space
color_arr = np.multiply(0.7,[colorbar(int(np.mod(i/2, 1)*255)) for i in tau])

# %%
fig, axs = pl.subplots(2, figsize = (8, 6))

axs[0].plot(t_space, signal)
axs[0].grid(True)
for i in mark_idx:
    axs[0].plot([t_space[i]], [signal[i]], marker = 'o', markersize=8, c=color_arr[i])
    axs[0].plot([t_space[i], t_space[i]], [0, signal[i]], c=color_arr[i])
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Strain")

axs[1].plot(tau, signal)
for i in mark_idx:
    axs[1].plot([tau[i]], [signal[i]], marker = 'o', markersize=8, c=color_arr[i])
    axs[1].plot([tau[i], tau[i]], [0, signal[i]], c=color_arr[i])
axs[1].set_xlabel("Resampled time (s)")
axs[1].set_ylabel("Strain")
axs[1].grid(True)

fig.tight_layout()
pl.savefig("figs/resampling_graphic.png")

# %%

# %% [markdown]
# ## Interpolation onto a uniform resampled-time grid

# %%
# Reuse the existing samples, which are non-uniformly spaced in tau.
tau_samples = tau[mark_idx]
signal_samples = signal[mark_idx]

# Construct a uniform tau grid with the same number of samples and interpolate onto it.
tau_uniform = np.linspace(tau_samples[0], tau_samples[-1], len(tau_samples))
signal_uniform = np.interp(tau_uniform, tau_samples, signal_samples)
color_uniform = np.multiply(
    0.7, [colorbar(int(np.mod(tau_i/2, 1)*255)) for tau_i in tau_uniform]
)

# %%
fig, axs = pl.subplots(2, figsize=(8, 6), sharex=True)

# Existing samples in resampled time, with the target uniform grid behind them.
axs[0].plot(tau, signal, c='tab:blue')
for i, tau_i in enumerate(tau_uniform):
    axs[0].axvline(
        tau_i, color='0.75', linewidth=0.8, zorder=0,
        label='Uniform target grid' if i == 0 else None,
    )
for j, i in enumerate(mark_idx):
    axs[0].plot(
        tau[i], signal[i], linestyle='none', marker='o', markersize=8,
        c=color_arr[i], zorder=2,
        label='Non-uniform samples' if j == 0 else None,
    )
axs[0].set_ylabel('Strain')
axs[0].legend()
axs[0].grid(True)

# The same data after interpolation onto the uniform grid.
axs[1].plot(tau, signal, c='tab:blue')
for tau_i, signal_i, color_i in zip(tau_uniform, signal_uniform, color_uniform):
    axs[1].plot(tau_i, signal_i, linestyle='none', marker='o', markersize=8, c=color_i)
    axs[1].plot([tau_i, tau_i], [0, signal_i], c=color_i)
axs[1].set_xlabel('Resampled time (s)')
axs[1].set_ylabel('Strain')
axs[1].grid(True)

fig.tight_layout()
