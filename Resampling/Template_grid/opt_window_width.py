import numpy as np
import matplotlib.pyplot as pl
from scipy import interpolate, integrate
from joblib import Parallel, delayed
from matplotlib.lines import Line2D

def beta_calc(f0, Mc):
    M = np.multiply(Mc,2e30)
    return const*f0**(8/3)*M**(5/3)

def f_calc(t, f0, Mc):
    return f0*(1-8/3*beta_calc(f0, Mc)*t)**(-3/8)

def t_max_calc(f0, beta, f_max):
    temp0 = 0.375/beta
    temp1 = 1-(f0/f_max)**(8/3)
    return temp0*temp1
    
def integrated_func(t, f0, Mc):
    temp0 = np.power((1-8/3*beta_calc(f0, Mc)*t),-0.5)
    temp1 = interp_asd(f_calc(t, f0, Mc))**2
    return temp0/temp1

def opt_distance(f0, Mc, f_max=200, l = 47):
    #l := lambda. l=47 is the threshold for FAP=1e-6, detection probability = 0.95
    T = min(t_max_calc(f0, beta_calc(f0, Mc), f_max), 3e7)
    integration = integrate.quad(integrated_func, 0, T, args = (f0, Mc))[0]
    temp0 = 0.00757/np.sqrt(l)
    temp1 = 3*1e8*beta_calc(f0,Mc)/f0**2
    return temp0*temp1*np.sqrt(integration)/3e16

def distance_sensitivity(W, f0, Mc, l = 47, beta_loss = 0.95, f_max = 200):
    """Calculate the distance sensitivity reduced by the SNR loss from
    a fixed window size
    
    Arguments
    ----------
    W : float
        length of the analysis window [s]
    f0 : float
        initial frequency of the signal [Hz] - between 20-2000 Hz
    Mc : float
        chirp mass [solar masses] - between 1e-1 - 1e-5M_\odot
    l : float
        threshold spectral amplitude, default value corresponds to 
        FAP=1e-6 and detection_probability=95%
    f_max : float
        maximum frequency of the analysis
        
    Returns
    ----------
    distance : float
        distance sensitivity in units of pc
    """
    
    beta = beta_calc(f0, Mc)
    T = min(t_max_calc(f0, beta, f_max), 3e7)
    
    integration = integrate.quad(integrated_func, 0, T, args = (f0, Mc))[0]
    temp0 = 0.00757/np.sqrt(l)
    temp1 = c*beta_calc(f0,Mc)/f0**2
    
    opt_dist = temp0*temp1*np.sqrt(integration)/3e16

    if W>=T:
        return np.sqrt(T/W)*np.sqrt(beta_loss)*opt_dist
    else:
        return np.power(W/T, 1/4)*np.sqrt(beta_loss)*opt_dist

c = 3e8
G = 6.67e-11
pi = np.pi
const = 96/5*pi**(8/3)*(G/c**3)**(5/3)

asd = np.loadtxt('asd.txt')
interp_asd = interpolate.interp1d(asd.T[0], asd.T[1]) # , bounds_error = False, fill_value="extrapolate"

fspace = np.linspace(20, 200, 200)
Mspace = np.logspace(-5, -1, 200)
[fgrid, Mgrid] = np.meshgrid(fspace, Mspace)
flatten_array = [(i, j) for i in fspace for j in Mspace]

#Optimal distance sensitivity
opt_grid = np.array(Parallel(15)(
        delayed(opt_distance)(i, j) for i, j in flatten_array
    ))
optimal_area = np.count_nonzero(opt_grid>8000)

best_window = 0
best_area = 0
best_sens = 0

#Finding best window_width
window_space = np.logspace(3, 7, 5)
for w in window_space:
    sens_grid = np.array(Parallel(15)(
            delayed(distance_sensitivity)(w, i, j) for i, j in flatten_array
        ))
    temp_area = np.count_nonzero(sens_grid>8000)
    if temp_area > best_area:
        best_area = temp_area
        best_window = w
        best_sens = sens_grid
        
best_sens = np.reshape(best_sens, (len(fspace), len(Mspace))).T

print(best_window)
print(best_area/optimal_area)

#Plotting optimal window
pl.contourf(fspace, np.log10(Mspace), np.log10(best_sens))
pl.colorbar()
pl.contour(fspace, np.log10(Mspace), np.log10(best_sens), [np.log10(8000)], colors = 'red', linestyles = '--')
pl.title(r'log(Distance Sensitivity / pc) \n Window=%.1E, area_recovered=%.2f' % (best_window, best_area/optimal_area))
pl.xlabel("Initial frequency (Hz)")
pl.ylabel(r'$log(M_c/M_\odot$)')
# pl.scatter(fspace[f_idx], np.log10(Mspace[M_idx]), c='red')

line = Line2D([0], [0], label='Galactic center', color='r', ls = '--')
pl.legend(handles = [line])
pl.savefig("./opt_window_width.jpg")
