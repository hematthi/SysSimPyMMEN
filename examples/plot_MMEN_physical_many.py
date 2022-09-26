# To import required modules:
import numpy as np
import time
import os
import sys
import matplotlib
import matplotlib.cm as cm #for color maps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec #for specifying plot attributes
from matplotlib import ticker #for setting contour plots to log scale
import scipy.integrate #for numerical integration
import scipy.misc #for factorial function
from scipy.special import erf #error function, used in computing CDF of normal distribution
import scipy.interpolate #for interpolation functions
import scipy.optimize #for fitting functions
import corner #corner.py package for corner plots
#matplotlib.rc('text', usetex=True)

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *

from syssimpymmen.mmen import *





savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/MMEN/'
run_number = ''
model_name = 'Maximum_AMD_model' + run_number
model_label, model_color = 'Maximum AMD model', 'g' #'Maximum AMD model', 'g' #'Two-Rayleigh model', 'b'

##### To load the physical catalogs:

# To load and process the simulated physical catalog of stars and planets:
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)





# Plotting parameters:
n_bins = 50
lw = 1 # linewidth

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size

# Parameters for defining the MMEN:
a0 = 0.3 # normalization separation for fitting power-laws





##### To load and compute the same statistics for a large number of models:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
runs = 100

prescriptions = ['CL2013', 'S2014', 'nHill', 'RC2014']
fit_per_cat_dict = {pres:{'sigma0':[], 'beta':[]} for pres in prescriptions} # dictionary of dictionaries
for i in range(1,runs+1):
    run_number = i
    print('##### %s' % i)
    sssp_per_sys_i, sssp_i = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)

    for pres in prescriptions:
        fit_cat_dict = fit_power_law_MMEN_all_planets_physical(sssp_per_sys_i, sssp_i,. prescription=pres, a0=a0)
        fit_per_cat_dict[pres]['sigma0'].append(fit_cat_dict['sigma0'])
        fit_per_cat_dict[pres]['beta'].append(fit_cat_dict['beta'])

# Compute and print the quantiles:
for pres in prescriptions:
    fit_per_cat_dict[pres]['sigma0'] = np.array(fit_per_cat_dict[pres]['sigma0'])
    fit_per_cat_dict[pres]['beta'] = np.array(fit_per_cat_dict[pres]['beta'])
    sigma0_qtls = np.quantile(fit_per_cat_dict[pres]['sigma0'], [0.16,0.5,0.84])
    beta_qtls = np.quantile(fit_per_cat_dict[pres]['beta'], [0.16,0.5,0.84])
    print('# %s:' % pres)
    print(r'$\Sigma_0$: ${:0.1f}_{{-{:0.1f} }}^{{+{:0.1f} }}$'.format(sigma0_qtls[1], sigma0_qtls[1]-sigma0_qtls[0], sigma0_qtls[2]-sigma0_qtls[1]))
    print(r'$\beta$: ${:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(beta_qtls[1], beta_qtls[1]-beta_qtls[0], beta_qtls[2]-beta_qtls[1]))
