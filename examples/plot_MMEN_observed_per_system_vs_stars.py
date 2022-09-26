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
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/MMEN/'
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
prescription_str = 'RC2014'
a0 = 0.3 # normalization separation for fitting power-laws





##### To fit a power-law to each observed system for the simulated observed and Kepler catalog:

fit_per_sys_dict = fit_power_law_MMEN_per_system_observed(sss_per_sys, prescription=prescription_str, a0=a0)
fit_per_sys_dict_Kep = fit_power_law_MMEN_per_system_observed(ssk_per_sys, prescription=prescription_str, a0=a0)

##### To test how the fitted observed MMEN may correlate with stellar mass:

# To plot sigma0 vs. stellar mass for the simulated and Kepler observed systems:
plot_2d_points_and_contours_with_histograms(fit_per_sys_dict['Mstar_obs'], fit_per_sys_dict['sigma0'], x_min=0.5, x_max=1.5, y_min=1e-2, y_max=1e8, log_y=True, bins_cont=20, points_only=True, xlabel_text=r'$M_\star$ ($M_\odot$)', ylabel_text=r'$\log_{10}(\Sigma_0/{\rm g cm^{-2}})$', extra_text='Simulated observed systems', plot_qtls=True, y_str_format='{:0.1f}', x_symbol=r'$M_\star$', y_symbol=r'$\Sigma_0$', save_name=savefigures_directory + model_name + '_obs_mmen_%s_sigma0_vs_starmass_per_system.pdf' % prescription_str, save_fig=savefigures)

plot_2d_points_and_contours_with_histograms(fit_per_sys_dict_Kep['Mstar_obs'], fit_per_sys_dict_Kep['sigma0'], x_min=0.5, x_max=1.5, y_min=1e-2, y_max=1e8, log_y=True, bins_cont=20, points_only=True, xlabel_text=r'$M_\star$ ($M_\odot$)', ylabel_text=r'$\log_{10}(\Sigma_0/{\rm g cm^{-2}})$', extra_text='Kepler observed systems', plot_qtls=True, y_str_format='{:0.1f}', x_symbol=r'$M_\star$', y_symbol=r'$\Sigma_0$', save_name=savefigures_directory + 'Kepler_mmen_%s_sigma0_vs_starmass_per_system.pdf' % prescription_str, save_fig=savefigures)

# To plot beta vs. stellar mass for the simulated and Kepler observed systems:
plot_2d_points_and_contours_with_histograms(fit_per_sys_dict['Mstar_obs'], fit_per_sys_dict['beta'], x_min=0.5, x_max=1.5, y_min=-8, y_max=4, bins_cont=20, points_only=True, xlabel_text=r'$M_\star$ ($M_\odot$)', ylabel_text=r'$\beta$', extra_text='Simulated observed systems', plot_qtls=True, x_symbol=r'$M_\star$', y_symbol=r'$\beta$', save_name=savefigures_directory + model_name + '_obs_mmen_%s_beta_vs_starmass_per_system.pdf' % prescription_str, save_fig=savefigures)

plot_2d_points_and_contours_with_histograms(fit_per_sys_dict_Kep['Mstar_obs'], fit_per_sys_dict_Kep['beta'], x_min=0.5, x_max=1.5, y_min=-8, y_max=4, bins_cont=20, points_only=True, xlabel_text=r'$M_\star$ ($M_\odot$)', ylabel_text=r'$\beta$', extra_text='Kepler observed systems', plot_qtls=True, x_symbol=r'$M_\star$', y_symbol=r'$\beta$', save_name=savefigures_directory + 'Kepler_mmen_%s_beta_vs_starmass_per_system.pdf' % prescription_str, save_fig=savefigures)

plt.show()
