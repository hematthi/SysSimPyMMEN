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





##### To fit a power-law to each observed system in a simulated observed catalog, and then for each underlying physical system (i.e. including all planets), to compare how the power-law fits change:
##### NOTE: we will use the 'true' planet masses for the simulated observed planets so their masses are consistent in both MMEN calculations of the physical systems and the observed systems

fit_per_sys_dict = fit_power_law_MMEN_per_system_observed_and_physical(sssp_per_sys, solid_surface_density_prescription=solid_surface_density_system_RC2014, a0=a0)

# To plot sigma0_obs vs sigma0_true for the simulated observed systems:
plot_2d_points_and_contours_with_histograms(fit_per_sys_dict['sigma0_true'], fit_per_sys_dict['sigma0_obs'], x_min=1e-2, x_max=1e7, y_min=1e-2, y_max=1e7, log_x=True, log_y=True, xlabel_text=r'$\log_{10}(\Sigma_{0,\rm true}/{\rm g cm^{-2}})$', ylabel_text=r'$\log_{10}(\Sigma_{0,\rm obs}/{\rm g cm^{-2}})$', extra_text='Simulated observed systems', plot_qtls=True, x_str_format='{:0.1f}', y_str_format='{:0.1f}', x_symbol=r'$\Sigma_{0,\rm true}$', y_symbol=r'$\Sigma_{0,\rm obs}$', save_name=savefigures_directory + model_name + '_true_vs_obs_mmen_%s_sigma0_per_system.pdf' % prescription_str, save_fig=savefigures)

# To plot beta_obs vs beta_true for the simulated observed systems:
plot_2d_points_and_contours_with_histograms(fit_per_sys_dict['beta_true'], fit_per_sys_dict['beta_obs'], x_min=-8., x_max=4., y_min=-8., y_max=4., xlabel_text=r'$\beta_{\rm true}$', ylabel_text=r'$\beta_{\rm obs}$', extra_text='Simulated observed systems', plot_qtls=True, x_symbol=r'$\beta_{\rm true}$', y_symbol=r'$\beta_{\rm obs}$', save_name=savefigures_directory + model_name + '_true_vs_obs_mmen_%s_beta_per_system.pdf' % prescription_str, save_fig=savefigures)

plt.show()
