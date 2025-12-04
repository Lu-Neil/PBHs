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
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import finufft
import matplotlib.pyplot as pl
from scipy import integrate
from scipy import optimize
from resampler import Resampler

pi = np.pi
c = 3e8
G = 6.67e-11
const = 96/5*pi**(8/3)*(G/c**3)**(5/3)
day = 86400


# %% [markdown]
# ## beta

# %%
def beta_calc(f0, M):
    """M in units of solar masses"""
    Mc = M*2e30
    return const*f0**(8/3)*Mc**(5/3)

def t_max_calc(f0, beta, f_max):
    temp0 = 0.375/beta
    temp1 = 1-(f0/f_max)**(8/3)
    return temp0*temp1

def freq_err(t, f0, M, delta_beta):
    """Error in the frequency from the dephasing
    M in units of solar masses
    """
    beta = beta_calc(f0, M)
    temp0 = (3*f0*delta_beta)/(5*beta*(1-8/3*beta*t)**(3/8))
    temp1 = 1+(-1+beta*t)/(1-8/3*beta*t)
    return temp0*temp1

def tau_calc(beta, t):
    return 6*np.pi/5*(1-8/3*beta*t)**(5/8)/beta


# %%
def integrated_func_without_asd(t, f0, M):
    temp0 = np.power((1-8/3*beta_calc(f0, M)*t),-0.5)
    return temp0

def analytical_loss(offset, f0, beta):
    temp_arr = []
    T = min(T_obs, t_max_calc(f0, beta, 200))
    t_space = np.logspace(0, np.log10(T), 10000)
    for i in np.atleast_1d(offset):
        err_arr = abs(freq_err(t_space, f0, M, i))
        idx = np.argmin(abs(err_arr - 2/(T)))
        crossover_t = t_space[idx]

        temp0 = integrate.quad(integrated_func_without_asd, crossover_t, T, args = (f0, M))[0] 
        temp1 = integrate.quad(integrated_func_without_asd, 0, T, args = (f0, M))[0] 
        temp_arr.append(temp0/temp1)
    return np.array(temp_arr)

def optimize_func(log_offset, f0, beta, target_power = 0.9):
    # 1-target_power necessary because we care about power remaining (not analytical loss)
    return analytical_loss(10**log_offset, f0, beta) - (1-target_power)


# %%
f0 = 20
M = 1e-1
Mc = M * 2e30
f_max = 50
beta = const*f0**(8/3)*Mc**(5/3)
f_signal = 4*f_max

T_temp = 0.5*day
T_obs = min(T_temp, t_max_calc(f0, beta, f_max))
nt = round(f_signal*T_obs)
t = (np.arange(nt)/f_signal)

phi = 6*np.pi/5*f0*(1-8./3.*(beta)*t)**(5/8)/beta
tau = 6*np.pi/5*(1-8/3*beta*t)**(5/8)/beta
signal = np.real(1*np.exp(-1j*phi))


# %%
def tcross_calc(beta, T_obs, power_loss=0.1):
    temp0 = 1-(power_loss+(1+power_loss)*(1-8/3*beta*T_obs)**(1/2))**2
    return 3/(8*beta)*temp0


# %%
crossover_t = tcross_calc(beta, T_obs)

# %%
temp0 = integrate.quad(integrated_func_without_asd, crossover_t, T_obs, args = (f0, M))[0] 
temp1 = integrate.quad(integrated_func_without_asd, 0, T_obs, args = (f0, M))[0] 
print(temp0/temp1)

# %%
offset_arr = np.logspace(-8, -4, 20)
analytical_arr = analytical_loss(offset_arr, f0, beta)
step_size = optimize.root_scalar(optimize_func, args=(f0, beta),
              bracket=[-12, -5], 
              method='brentq')

numerical_power = []
for i in offset_arr:
    resampler = Resampler()
    resampler.timeseries = signal
    resampler.resampled_time = tau_calc(beta+i, t)
    resampler.nufft_real()
    numerical_power.append(max(resampler.power_normalized))

# %%
pl.semilogx(offset_arr, 1-analytical_arr, 'o', label='Analytical power loss')
pl.semilogx(offset_arr, numerical_power/numerical_power[0], 'o', label='Numerical power loss')
pl.axvline(10**step_size.root, c='r', ls='--')
pl.legend()
pl.xlabel(r"$\Delta \beta$")
pl.ylabel("Power recovered")

# %%
# print(np.log10(beta))
# print(step_size.root)
temp_T = min(T_obs, t_max_calc(f0, beta, 200))
dephasing = phi_err(f0, beta, step_size.root, temp_T)
total_phase = phi_calc(f0, beta, temp_T)
print(f"{dephasing/total_phase:.2e}")

# %%
resampler = Resampler()
resampler.timeseries = signal
resampler.resampled_time = tau_calc(beta+10**step_size.root, t)
resampler.nufft_real()
pl.plot(resampler.freqs, resampler.power_normalized, 'o')
pl.xlim(f0 - 10, f0 + 10)


# %%
def phi_err(f0, beta, offset, t):
    temp0 = 6*np.pi/5*f0*offset/(beta**2)*(-1+beta*t)/((1-8/3*beta*t)**(3/8))
    temp1 = 6*np.pi*f0*offset/(5*beta**2)
    return -temp0 - temp1

def phi_calc(f0, beta, t):
    temp0 = 6*np.pi/5*f0*(1-8./3.*(beta)*t)**(5/8)/beta
    temp1 = 6*np.pi/5*f0*(1-8./3.*(beta)*0)**(5/8)/beta
    return temp0 - temp1


# %%
temp_phi0 = phi_calc(f0, beta, t)
temp_phi1 = phi_calc(f0, beta+10**step_size.root, t)
pl.plot(t, temp_phi0 - temp_phi1)
# pl.plot(t, tau_calc(beta+10**step_size.root, t) - tau_calc(beta+10**step_size.root, 0))
pl.plot(t, phi_err(f0, beta, np.longdouble(10**step_size.root), t))

# %%
pl.plot(t, tau_calc(beta, t) - tau_calc(beta, 0))
pl.plot(t, tau_calc(beta+10**step_size.root, t) - tau_calc(beta+10**step_size.root, 0))
pl.plot(t, phi_err(f0, beta, np.longdouble(10**step_size.root), t))

# %% [markdown]
# ## Steps through beta space

# %%

# %%

# %%
f0 = 20
T_obs = 300*day

M_start = 1e-1
beta_start = beta_calc(f0, M_start)
M_end = 1e-2
beta_end = beta_calc(f0, M_end)

# %%
beta_start

# %%
step_size = optimize.root_scalar(optimize_func, args=(f0, beta_start),
              bracket=[-12, -5], 
              method='brentq')
step_size

# %%
beta_arr = [beta_start]
while beta_arr[-1] > beta_end:
    step_size = optimize.root_scalar(optimize_func, args=(f0, beta_arr[-1]),
                  bracket=[-13, -5], 
                  method='brentq')
    beta_arr.append(beta_arr[-1] - 10**step_size.root)

# %%
pl.semilogy(beta_arr, 'o')

# %%
len(beta_arr)


# %%

# %% [markdown]
# ## Theory verification

# %%
def phi_calc(f0, beta, t):
    temp0 = 6*np.pi/5*f0*(1-8./3.*(beta)*t)**(5/8)/beta
    temp1 = 6*np.pi/5*f0*(1-8./3.*(beta)*0)**(5/8)/beta
    return temp0 - temp1

def phi_err(f0, beta, offset, t):
    temp0 = 6*np.pi/5*f0*offset/(beta**2)*(-1+beta*t)/((1-8/3*beta*t)**(3/8))
    temp1 = 6*np.pi*f0*offset/(5*beta**2)
    return -temp0 - temp1


# %%
f0 = 20
M = 1e-1
Mc = M * 2e30
f_max = 200
beta = const*f0**(8/3)*Mc**(5/3)
f_signal = 50 #4*f_max

T_temp = 0.5*day
T_obs = min(T_temp, t_max_calc(f0, beta, f_max))
nt = round(f_signal*T_obs)
t = (np.arange(nt)/f_signal)

phi = 6*np.pi/5*f0*(1-8./3.*(beta)*t)**(5/8)/beta
tau = 6*np.pi/5*(1-8/3*beta*t)**(5/8)/beta
signal = np.real(1*np.exp(-1j*phi))

# %%
beta

# %% [markdown]
# ### Delta phi

# %%
offset = 1e-10
exact_err = phi_calc(f0, beta, t) - phi_calc(f0, beta+offset, t)
analytical_err = phi_err(f0, beta, offset, t)

# %%
pl.semilogy(t/day, exact_err - exact_err[0], label="Exact")
pl.semilogy(t/day, analytical_err, label = "First order approximation")
pl.legend(loc='lower right')
pl.xlabel("Time [days]")
pl.ylabel(r"$\Delta \phi$")


# %% [markdown]
# ### Delta f

# %%
def f_err(f0, beta, offset, t):
    temp0 = 3*f0*offset
    temp1 = 5*beta*(1-8/3*beta*t)**(3/8)
    temp2 = 1+(-1+beta*t)/(1-8/3*beta*t)
    return -temp0 / temp1 * temp2


# %%
exact_err = 1/(2*np.pi) \
            * (
                (np.diff(phi_calc(f0, beta, t) - phi_calc(f0, beta+offset, t)))
                /np.diff(t) 
              )
analytical_err = f_err(f0, beta, offset, t)

# %%
# pl.semilogy(t[:-1]/day, exact_err - exact_err[0], label="Exact")
pl.semilogy(t/day, analytical_err, label = "First order approximation")
pl.legend(loc='lower right')
pl.xlabel("Time [days]")
pl.ylabel(r"$\Delta f$")

# %% [markdown]
# ### t crossover

# %%
numerical_time = t[np.argmin(abs(analytical_err - 2/T_obs))]

# %%
print(numerical_time)
print("Is there anything more to be done here???")


# %% [markdown]
# ### Power loss

# %%
def loss_frac(f0, beta, tcross, T_obs):
    temp0 = (1-8/3*beta*tcross)**(1/2)
    temp1 = (1-8/3*beta*T_obs)**(1/2)
    return (temp0 - temp1) / (1-temp1)


# %%
loss_frac(f0, beta, numerical_time, T_obs)


# %% [markdown]
# ### Inverting

# %%
def tcross_invert(beta, T_obs, loss_frac):
    temp0 = loss_frac + (1-loss_frac)*(1-8/3*beta*T_obs)**0.5
    temp1 = 1-temp0**2
    return 3/(8*beta)*temp1

def deltaBeta_invert(beta, f0, tcross, T_obs):
    temp0 = -2/T_obs*(5*beta)/(3*f0)*(1-8/3*beta*tcross)**(3/8)
    temp1 = 1+(-1+beta*tcross)/(1-8/3*beta*tcross)
    return temp0 / temp1


# %%
np.isclose(numerical_time, tcross_invert(beta, T_obs, loss_frac(f0, beta, numerical_time, T_obs)))

# %%
print(deltaBeta_invert(beta, f0, numerical_time, T_obs))
print(offset)

# %%
temp_tcross = tcross_invert(beta, T_obs, 0.1)
temp_deltaBeta = deltaBeta_invert(beta, f0, temp_tcross, T_obs)
temp_deltaBeta/beta

# %%
T = 10
(0.5/T)**(1/4) / (3/T)**(1/4)

# %% [markdown]
# ## Steps through beta space

# %%
f0 = 20
T_temp = 0.5*day
T_obs = min(T_temp, t_max_calc(f0, beta, f_max))

M_start = 1e-5
beta_start = beta_calc(f0, M_start)
M_end = 1e-1
beta_end = beta_calc(f0, M_end)

# %%
beta_arr = [beta_start]
while beta_arr[-1] < beta_end:
    temp_Tobs = min(T_temp, t_max_calc(f0, beta, f_max))
    temp_tcross = tcross_invert(beta, temp_Tobs, 0.1)
    temp_deltaBeta = deltaBeta_invert(beta, f0, temp_tcross, temp_Tobs)
    beta_arr.append(beta_arr[-1] + temp_deltaBeta)

# %%
mass_space = np.logspace(-5, -1, 50)
mass_grid = np.array(np.meshgrid(mass_space, mass_space))
beta_grid = beta_calc(f0, mass_grid)
idx_grid = np.searchsorted(beta_arr, beta_grid)
size_grid = np.subtract(idx_grid[0,:,], idx_grid[1,:,:])

# %%
print('Power loss due to template mismatch = 10%, f0=20')
pl.contourf(np.log10(mass_space), np.log10(mass_space), np.log10(np.triu(size_grid)))
pl.colorbar()
pl.title("log10(templates needed)")
pl.xlabel("Maximum mass")
pl.ylabel("Minimum mass")

# %%
M_start = 1e-5
beta_start = beta_calc(f0, M_start)
M_end = 1e-1
beta_end = beta_calc(f0, M_end)
beta_arr = [beta_start]
while beta_arr[-1] < beta_end:
    temp_Tobs = min(T_temp, t_max_calc(f0, beta, f_max))
    temp_tcross = tcross_invert(beta, temp_Tobs, 0.01)
    temp_deltaBeta = deltaBeta_invert(beta, f0, temp_tcross, temp_Tobs)
    beta_arr.append(beta_arr[-1] + temp_deltaBeta)

# %%
mass_space = np.logspace(-5, -1, 50)
mass_grid = np.array(np.meshgrid(mass_space, mass_space))
beta_grid = beta_calc(f0, mass_grid)
idx_grid = np.searchsorted(beta_arr, beta_grid)
size_grid = np.subtract(idx_grid[0,:,], idx_grid[1,:,:])

# %%
print('Power loss due to template mismatch = 1%, f0=20')
pl.contourf(np.log10(mass_space), np.log10(mass_space), np.log10(np.triu(size_grid)))
pl.colorbar()
pl.title("log10(templates needed)")
pl.xlabel("Maximum mass")
pl.ylabel("Minimum mass")


# %% [markdown]
# ### Intuition

# %%
def deltaBeta_dephasing(beta, f0, T_obs, dphi=np.pi/4):
    temp0 = 5*beta**2*dphi/(6*np.pi*f0)
    temp1 = 1+(-1+beta*T_obs)/((1-8/3*beta*T_obs)**(3/8))
    return temp0 / temp1


# %%
beta_space = np.logspace(np.log10(beta_start), np.log10(beta_end), 100)
results_arr0 = []
results_arr1 = []

for i in beta_space:
    temp_Tobs = min(T_temp, t_max_calc(f0, i, f_max))
    temp_tcross = tcross_invert(i, temp_Tobs, 0.1)
    results_arr0.append(deltaBeta_dephasing(i, f0, temp_Tobs))
    results_arr1.append(deltaBeta_invert(i, f0, temp_tcross, temp_Tobs))

results_arr0 = np.array(results_arr0)
results_arr1 = np.array(results_arr1)

# %%
pl.loglog(beta_space, abs(results_arr0), label="Dephasing condition")
pl.loglog(beta_space, abs(results_arr1), label="Power condition")
pl.xlabel("Beta")
pl.ylabel("Delta beta")
pl.legend()

# %%
mass_space = np.logspace(-5, -1, 50)
f0 = 20
results_arr0 = []
results_arr1 = []

for i in mass_space:
    temp_beta = beta_calc(f0, i)
    temp_Tobs = min(T_temp, t_max_calc(f0, temp_beta, f_max))
    temp_tcross = tcross_invert(temp_beta, temp_Tobs, 0.1)
    results_arr0.append(deltaBeta_dephasing(temp_beta, f0, temp_Tobs) / temp_beta)
    results_arr1.append(deltaBeta_invert(temp_beta, f0, temp_tcross, temp_Tobs) / temp_beta)

results_arr0 = np.array(results_arr0)
results_arr1 = np.array(results_arr1)

# %%
pl.loglog(mass_space, abs(results_arr0), label="Dephasing condition")
pl.loglog(mass_space, abs(results_arr1), label="Power condition")
pl.xlabel("Mass")
pl.ylabel(r"$\Delta \beta / \beta$")
pl.legend()

# %%

# %%

# %%
