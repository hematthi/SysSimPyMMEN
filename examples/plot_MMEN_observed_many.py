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

sys.path.append('/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/SysSimExClusters_plotting') # TODO: update when those files get made into a package

from functions_general import *
from functions_compare_kepler import *
from functions_load_sims import *
from functions_plot_catalogs import *
from functions_plot_params import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.MMEN_functions import *





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

# To load and process the simulated physical catalog of stars and planets:
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)

# To load and process the simulated observed catalog of stars and planets:
sss_per_sys, sss = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)

# To load and process the observed Kepler catalog and compare with our simulated catalog:
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios)

dists, dists_w = compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)





# Plotting parameters:
n_bins = 50
lw = 1 # linewidth

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size

# Parameters for defining the MMEN:
prescription_str = 'RC2014' # make sure this actually matches the prescription used!
solid_surface_density_prescription = solid_surface_density_system_RC2014
a0 = 0.3 # normalization separation for fitting power-laws





##### To fit a power-law to each observed system for the Kepler catalog and plot the distribution of fitted parameters (sigma0 vs. beta):

fit_per_sys_dict_Kep = fit_power_law_MMEN_per_system_observed(ssk_per_sys, solid_surface_density_prescription=solid_surface_density_prescription, a0=a0)

plot_2d_points_and_contours_with_histograms(fit_per_sys_dict_Kep['beta'], fit_per_sys_dict_Kep['sigma0'], x_min=-8., x_max=4., y_min=1e-2, y_max=1e8, log_y=True, xlabel_text=r'$\beta$', ylabel_text=r'$\log_{10}(\Sigma_0/{\rm g cm^{-2}})$', extra_text='Kepler observed systems', plot_qtls=True, y_str_format='{:0.1f}', x_symbol=r'$\beta$', y_symbol=r'$\Sigma_0$', save_name=savefigures_directory + 'Kepler_mmen_%s_sigma0_vs_beta_per_system.pdf' % prescription_str, save_fig=savefigures)
plt.show()





##### To load and compute the same statistics for a large number of models:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
runs = 100

sss_per_sys_all = []
sss_all = []
fit_per_sys_dict_all = []

for i in range(1,runs+1):
    run_number = i
    sss_per_sys_i, sss_i = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)
    dists_i, dists_w_i = compute_distances_sim_Kepler(sss_per_sys_i, sss_i, ssk_per_sys, ssk, weights_all['all'], dists_include, N_Kep, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)
    
    sss_per_sys_all.append(sss_per_sys_i)
    sss_all.append(sss_i)
    
    fit_per_sys_dict = fit_power_law_MMEN_per_system_observed(sss_per_sys_i, solid_surface_density_prescription=solid_surface_density_prescription, a0=a0)
    fit_per_sys_dict_all.append(fit_per_sys_dict)
    
    # To plot the distribution of fitted power-law parameters (sigma0 vs. beta) for the simulated observed systems:
    plot_2d_points_and_contours_with_histograms(fit_per_sys_dict['beta'], fit_per_sys_dict['sigma0'], x_min=-8., x_max=4., y_min=1e-2, y_max=1e8, log_y=True, xlabel_text=r'$\beta$', ylabel_text=r'$\log_{10}(\Sigma_0/{\rm g cm^{-2}})$', extra_text='Simulated observed systems', plot_qtls=True, y_str_format='{:0.1f}', x_symbol=r'$\beta$', y_symbol=r'$\Sigma_0$', save_name=savefigures_directory + model_name + '_obs_mmen_%s_sigma0_vs_beta_per_system_%s.pdf' % (prescription_str, run_number), save_fig=savefigures)
    #plt.show()
    plt.close()

plt.show()

# To plot the median sigma0 vs. beta for each simulated catalog:
sigma0_med_all = [np.median(fit_per_sys_dict['sigma0']) for fit_per_sys_dict in fit_per_sys_dict_all]
beta_med_all = [np.median(fit_per_sys_dict['beta']) for fit_per_sys_dict in fit_per_sys_dict_all]

ax_main = plot_2d_points_and_contours_with_histograms(beta_med_all, sigma0_med_all, x_min=-2.25, x_max=-1.5, y_min=250., y_max=750., log_y=True, n_bins=20, points_only=True, xlabel_text=r'Median $\beta$', ylabel_text=r'Median $\log_{10}(\Sigma_0/{\rm g cm^{-2}})$', plot_qtls=True, y_str_format='{:0.1f}', x_symbol=r'$\beta_{\rm med}$', y_symbol=r'$\Sigma_{0,\rm med}$')
ax_main.scatter(np.median(fit_per_sys_dict_Kep['beta']), np.log10(np.median(fit_per_sys_dict_Kep['sigma0'])), marker='x', color='r', label='Kepler')
ax_main.legend(loc='upper left', bbox_to_anchor=(0.,1.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_obs_mmen_%s_median_sigma0_vs_beta_per_system.pdf' % prescription_str)
    plt.close()
plt.show()
