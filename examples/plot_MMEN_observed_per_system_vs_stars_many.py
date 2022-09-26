# To import required modules:
import numpy as np
import time
import os
import sys
import matplotlib
import matplotlib.cm as cm # for color maps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec # for specifying plot attributes
from matplotlib import ticker # for setting contour plots to log scale
import scipy.integrate # for numerical integration
import scipy.misc # for factorial function
from scipy.special import erf # error function, used in computing CDF of normal distribution
import scipy.interpolate # for interpolation functions
import scipy.optimize # for fitting functions
import corner # corner.py package for corner plots

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *

from syssimpymmen.mmen import *





savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_best_models/MMEN/'
run_number = ''
model_name = 'Maximum_AMD_model' + run_number
model_label, model_color = 'Maximum AMD model', 'g' #'Maximum AMD model', 'g' #'Two-Rayleigh model', 'b'

##### To load the physical and observed catalogs:

compute_ratios = compute_ratios_adjacent
AD_mod = 'true' # 'true' or 'false'
weights_all = load_split_stars_weights_only()
dists_include = ['delta_f',
                 'mult_CRPD_r',
                 'periods_KS',
                 'period_ratios_KS',
                 #'durations_KS',
                 #'durations_norm_circ_KS',
                 'durations_norm_circ_singles_KS',
                 'durations_norm_circ_multis_KS',
                 'duration_ratios_nonmmr_KS',
                 'duration_ratios_mmr_KS',
                 'depths_KS',
                 'radius_ratios_KS',
                 'radii_partitioning_KS',
                 'radii_monotonicity_KS',
                 'gap_complexity_KS',
                 ]

# To first read the number of simulated targets and bounds for the periods and radii:
N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

# To read the simulation parameters from the file:
param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)

# To load and process the simulated observed catalog of stars and planets:
sss_per_sys, sss = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)

# To load and process the observed Kepler catalog and compare with our simulated catalog:
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios)

dists, dists_w = compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod)





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

sss_per_sys_all = []
sss_all = []

prescriptions = ['CL2013', 'S2014', 'nHill', 'RC2014']
fit_per_sys_dict_all = {pres:[] for pres in prescriptions} # dictionary of list of dictionaries

for i in range(1,runs+1):
    run_number = i
    sss_per_sys_i, sss_i = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)
    dists_i, dists_w_i = compute_distances_sim_Kepler(sss_per_sys_i, sss_i, ssk_per_sys, ssk, weights_all['all'], dists_include, N_Kep, cos_factor=cos_factor, AD_mod=AD_mod)

    sss_per_sys_all.append(sss_per_sys_i)
    sss_all.append(sss_i)

    for pres in prescriptions:
        fit_per_sys_dict = fit_power_law_MMEN_per_system_observed(sss_per_sys_i, prescription=pres, a0=a0, scale_up=True)
        fit_per_sys_dict_all[pres].append(fit_per_sys_dict)

for pres in prescriptions:
    r_mstar_sigma0 = [Pearson_correlation_coefficient(fit_per_sys_dict['Mstar_obs'], fit_per_sys_dict['sigma0']) for fit_per_sys_dict in fit_per_sys_dict_all[pres]]
    r_mstar_beta = [Pearson_correlation_coefficient(fit_per_sys_dict['Mstar_obs'], fit_per_sys_dict['beta']) for fit_per_sys_dict in fit_per_sys_dict_all[pres]]
    r_mstar_sigma0_qtls = np.quantile(r_mstar_sigma0, [0.16,0.5,0.84])
    r_mstar_beta_qtls = np.quantile(r_mstar_beta, [0.16,0.5,0.84])
    print('# %s:' % pres)
    print(r'r_Mstar_sigma0: ${:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(r_mstar_sigma0_qtls[1], r_mstar_sigma0_qtls[1]-r_mstar_sigma0_qtls[0], r_mstar_sigma0_qtls[2]-r_mstar_sigma0_qtls[1]))
    print(r'r_Mstar_beta: ${:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(r_mstar_beta_qtls[1], r_mstar_beta_qtls[1]-r_mstar_beta_qtls[0], r_mstar_beta_qtls[2]-r_mstar_beta_qtls[1]))

# To repeat for the Kepler catalog (resampling masses many times):

N_cats = 100
for pres in prescriptions:
    r_mstar_sigma0_Kep = []
    r_mstar_beta_Kep = []
    for i in range(N_cats):
        fit_per_sys_dict_Kep = fit_power_law_MMEN_per_system_observed(ssk_per_sys, prescription=pres, a0=a0, scale_up=True)
        r_mstar_sigma0_Kep.append(Pearson_correlation_coefficient(fit_per_sys_dict_Kep['Mstar_obs'], fit_per_sys_dict_Kep['sigma0']))
        r_mstar_beta_Kep.append(Pearson_correlation_coefficient(fit_per_sys_dict_Kep['Mstar_obs'], fit_per_sys_dict_Kep['beta']))
    r_mstar_sigma0_Kep = np.array(r_mstar_sigma0_Kep)
    r_mstar_beta_Kep = np.array(r_mstar_beta_Kep)
    r_mstar_sigma0_qtls = np.quantile(r_mstar_sigma0_Kep, [0.16,0.5,0.84])
    r_mstar_beta_qtls = np.quantile(r_mstar_beta_Kep, [0.16,0.5,0.84])
    print('# %s:' % pres)
    print(r'r_Mstar_sigma0: ${:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(r_mstar_sigma0_qtls[1], r_mstar_sigma0_qtls[1]-r_mstar_sigma0_qtls[0], r_mstar_sigma0_qtls[2]-r_mstar_sigma0_qtls[1]))
    print(r'r_Mstar_beta: ${:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(r_mstar_beta_qtls[1], r_mstar_beta_qtls[1]-r_mstar_beta_qtls[0], r_mstar_beta_qtls[2]-r_mstar_beta_qtls[1]))
