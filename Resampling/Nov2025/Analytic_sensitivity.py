# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as pl
from scipy import interpolate, integrate
from astropy.constants import c, G
import matplotlib as mpl

# %%
c = c.value
G = G.value
pi = np.pi
const = 96/5*pi**(8/3)*(G/c**3)**(5/3)
day = 86400

# %%
asd = np.loadtxt('../asd.txt')
interp_asd = interpolate.interp1d(asd.T[0], asd.T[1]) # , bounds_error = False, fill_value="extrapolate"

xspace = np.logspace(1.01, 3.5, 1000)
pl.loglog(asd.T[0], asd.T[1], label = 'Imported asd')
pl.loglog(xspace, interp_asd(xspace), label = "Interpolated asd")
pl.legend()
pl.title("O4 design sensitivity")


# %%
def beta_calc(f0, Mc):
    M = np.multiply(Mc,2e30)
    return const*f0**(8/3)*M**(5/3)

def f_calc(t, f0, Mc):
    return f0*(1-8/3*beta_calc(f0, Mc)*t)**(-3/8)


# %%
fspace = np.linspace(20, 200, 31)
Mspace = np.logspace(-4.5, 0, 29)
[fgrid, Mgrid] = np.meshgrid(fspace, Mspace)

betaGrid = beta_calc(fgrid, Mgrid)
tMax_grid = 0.37/betaGrid
tMax_grid[tMax_grid>3e7] = 3e7

final_f = f_calc(tMax_grid, fgrid, Mgrid)

# %%
0.375/beta_calc(20, 0.3)

# %%
0.3*day

# %%
pl.contourf(fspace, np.log10(Mspace), np.log10(tMax_grid))
pl.colorbar()


# %% [markdown]
# ## Sensitivity

# %%
def integrated_func(t, f0, Mc):
    temp0 = np.power((1-8/3*beta_calc(f0, Mc)*t),-0.5)
    temp1 = interp_asd(f_calc(t, f0, Mc))**2
    return temp0/temp1

def distance_sensitivity(T, f0, Mc, l = 47, window = 0.3*day, powerLoss_beta = 0.05):
    #l := lambda. l=47 is the threshold for FAP=1e-6, detection probability = 0.95
    integration = integrate.quad(integrated_func, 0, T, args = (f0, Mc))[0]
    temp0 = 0.00757*c*beta_calc(f0,Mc)/(f0**2 * np.sqrt(l))
    if window < T:
        distLoss_t = (window / T)**(1/4)
    else:
        distLoss_t = (T / window)**(1/2)
    distLoss_beta = (1-powerLoss_beta)**0.5
    return distLoss_beta * distLoss_t * temp0*np.sqrt(integration)


# %%
distance_sensitivity(1200, 20, 0.353)/3e16/1e6

# %%
window = 0.3
powerLoss_beta = 0.05
sens_grid = [[distance_sensitivity(tMax_grid[j,i], fspace[i], Mspace[j], l=34, window=window*day, powerLoss_beta=powerLoss_beta) 
              for i in range(len(fspace))] for j in range(len(Mspace))]
sens_grid = np.divide(sens_grid, 3e16)
size = np.count_nonzero((sens_grid > 8000).flatten())
print(size)

# %%

# %%
pl.contourf(fspace, np.log10(Mspace), np.log10(sens_grid))
cbar = pl.colorbar()
cbar.ax.set_ylabel('log(Distance Sensitivity / pc)', rotation=270, labelpad=15)
pl.contour(fspace, np.log10(Mspace), np.log10(sens_grid), [np.log10(8000)], colors = 'red', linestyles = '--')
pl.title(f'Window = {window} days, powerLoss_beta = {powerLoss_beta*100}%')
pl.xlabel("Initial frequency (Hz)")
pl.ylabel(r'$log(M_c/M_\odot$)')

from matplotlib.lines import Line2D
line = Line2D([0], [0], label='Galactic center', color='r', ls = '--')
pl.legend(handles = [line])
pl.tight_layout()
print("Window = 0.3 day")
# pl.savefig("Distance_sensitivity.jpg", format = "jpeg")

# %%
window = 3
powerLoss_beta = 0.05
sens_grid = [[distance_sensitivity(tMax_grid[j,i], fspace[i], Mspace[j], l=34, window=window*day, powerLoss_beta=powerLoss_beta) 
              for i in range(len(fspace))] for j in range(len(Mspace))]
sens_grid = np.divide(sens_grid, 3e16)
size = np.count_nonzero((sens_grid > 8000).flatten())
print(size)

pl.contourf(fspace, np.log10(Mspace), np.log10(sens_grid))
cbar = pl.colorbar()
cbar.ax.set_ylabel('log(Distance Sensitivity / pc)', rotation=270, labelpad=15)
pl.contour(fspace, np.log10(Mspace), np.log10(sens_grid), [np.log10(8000)], colors = 'red', linestyles = '--')
pl.title(f'Window = {window} days, powerLoss_beta = {powerLoss_beta*100}%')
pl.xlabel("Initial frequency (Hz)")
pl.ylabel(r'$log(M_c/M_\odot$)')

from matplotlib.lines import Line2D
line = Line2D([0], [0], label='Galactic center', color='r', ls = '--')
pl.legend(handles = [line])
pl.tight_layout()
# pl.savefig("Distance_sensitivity.jpg", format = "jpeg")

# %%
window_space = np.linspace(0.01, 1, 20)
size_arr = []
for window in window_space:
    print(window)
    powerLoss_beta = 0.05
    sens_grid = np.array([np.array([distance_sensitivity(tMax_grid[j,i], fspace[i], Mspace[j], l=34, window=window*day, powerLoss_beta=powerLoss_beta) 
                  for i in range(len(fspace))]) for j in range(len(Mspace))])
    sens_grid = np.divide(sens_grid, 3e16)
    size = np.count_nonzero((sens_grid > 8000).flatten())
    size_arr.append(size)

# %%
print("The jump is due to discretisation effects")
pl.plot(window_space, size_arr)
pl.xlabel("Coherence time (days)")
pl.ylabel("Size")


# %%
def d_calc(h0, f, M):
    m_corr = M*2e30
    temp0 = 4/h0
    temp1 = (G*m_corr/(c**2))**(5/3)
    temp2 = (pi*f/c)**(2/3)
    return temp0 * temp1 * temp2


# %%
temp_dist = sens_grid[0,0] * 3e16
h0_calc(temp_dist, fspace[0], Mspace[0])

# %%
