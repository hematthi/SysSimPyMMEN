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
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/MMEN/RC2014/cap_core_mass_10Mearth_and_scale_up_sigma0/' #cap_core_mass_10Mearth_and_scale_up_sigma0/
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

dists, dists_w = compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod)





# Plotting parameters:
n_bins = 50
lw = 2 # linewidth

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size

# Parameters for defining the MMEN:
prescription_str = 'RC2014'
a0 = 0.3 # normalization separation for fitting power-laws

scale_up = True
y_sym_star = '*' if scale_up else ''





##### To fit a power-law to each observed system in a simulated observed catalog, and then for each underlying physical system (i.e. including all planets), to compare how the power-law fits change:
##### NOTE: we will use the 'true' planet masses for the simulated observed planets so their masses are consistent in both MMEN calculations of the physical systems and the observed systems

fit_per_sys_dict = fit_power_law_MMEN_per_system_observed_and_physical(sssp_per_sys, sssp, prescription=prescription_str, a0=a0, scale_up=scale_up)

# To plot sigma0_obs vs sigma0_true for the simulated observed systems:
ax_min, ax_max = 1e-1, 1e6
ax = plot_2d_points_and_contours_with_histograms(fit_per_sys_dict['sigma0_true'], fit_per_sys_dict['sigma0_obs'], x_min=ax_min, x_max=ax_max, y_min=ax_min, y_max=ax_max, log_x=True, log_y=True, xlabel_text=r'$\log_{10}(\Sigma_{0,\rm phys}^%s/{\rm g\, cm^{-2}})$' % y_sym_star, ylabel_text=r'$\log_{10}(\Sigma_{0,\rm obs}^%s/{\rm g\, cm^{-2}})$' % y_sym_star, extra_text='Simulated observed systems', plot_qtls=True, x_str_format='{:0.0f}', y_str_format='{:0.0f}', x_symbol=r'$\Sigma_{0,\rm phys}^%s$' % y_sym_star, y_symbol=r'$\Sigma_{0,\rm obs}^%s$' % y_sym_star)
ax.plot([np.log10(ax_min), np.log10(ax_max)], [np.log10(ax_min), np.log10(ax_max)], ls='--', lw=lw)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_phys_vs_obs_mmen_%s_sigma0_per_system.pdf' % prescription_str)
    plt.close()

# To plot beta_obs vs beta_true for the simulated observed systems:
ax_min, ax_max = -8., 4.
ax = plot_2d_points_and_contours_with_histograms(fit_per_sys_dict['beta_true'], fit_per_sys_dict['beta_obs'], x_min=ax_min, x_max=ax_max, y_min=ax_min, y_max=ax_max, xlabel_text=r'$\beta_{\rm phys}$', ylabel_text=r'$\beta_{\rm obs}$', extra_text='Simulated observed systems', plot_qtls=True, x_symbol=r'$\beta_{\rm phys}$', y_symbol=r'$\beta_{\rm obs}$')
ax.plot([ax_min, ax_max], [ax_min, ax_max], ls='--', lw=lw)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_phys_vs_obs_mmen_%s_beta_per_system.pdf' % prescription_str)
    plt.close()

plt.show()



# To plot sigma0_obs/sigma0_true vs. beta_obs/beta_true for the simulated observed systems:
beta_ratio_min, beta_ratio_max = -4., 6.
sigma0_ratio_min, sigma0_ratio_max = 1e-3, 1e3
ax = plot_2d_points_and_contours_with_histograms(fit_per_sys_dict['beta_obs']/fit_per_sys_dict['beta_true'], fit_per_sys_dict['sigma0_obs']/fit_per_sys_dict['sigma0_true'], x_min=beta_ratio_min, x_max=beta_ratio_max, y_min=sigma0_ratio_min, y_max=sigma0_ratio_max, log_x=False, log_y=True, xlabel_text=r'$\beta_{\rm obs}/\beta_{\rm phys}$', ylabel_text=r'$\log_{10}(\Sigma_{0,\rm obs}^%s/\Sigma_{0,\rm phys}^%s)$' % (y_sym_star, y_sym_star), extra_text='Simulated observed systems', plot_qtls=True, x_symbol=r'$\frac{\beta_{\rm obs}}{\beta_{\rm phys}}$', y_symbol=r'$\frac{\Sigma_{0,\rm obs}^%s}{\Sigma_{0,\rm phys}^%s}$' % (y_sym_star, y_sym_star))
ax.plot([beta_ratio_min, beta_ratio_max], [0., 0.], ls='--', lw=lw, color='b')
ax.plot([1., 1.], [np.log10(sigma0_ratio_min), np.log10(sigma0_ratio_max)], ls='--', lw=lw, color='b')
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_phys_vs_obs_mmen_%s_sigma0_ratio_vs_beta_ratio_per_system.pdf' % prescription_str)
    plt.close()

# Histograms only:
x = fit_per_sys_dict['beta_obs']/fit_per_sys_dict['beta_true']
x_qtls = np.quantile(x, q=[0.16,0.5,0.84])
ax = plot_fig_pdf_simple([x], [], x_min=-2., x_max=4., n_bins=50, normalize=False, lw=lw, xlabel_text=r'$\beta_{\rm obs}/\beta_{\rm phys}$', ylabel_text='Systems')
ax.text(x=0.98, y=0.95, s=r'$\frac{\beta_{\rm obs}}{\beta_{\rm phys}}$' + r'$ = {:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(x_qtls[1], x_qtls[1]-x_qtls[0], x_qtls[2]-x_qtls[1]), ha='right', va='top', fontsize=20, transform = ax.transAxes)
ax.axvline(x=1., lw=lw, linestyle='--')
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_phys_vs_obs_mmen_%s_beta_ratio_per_system.pdf' % prescription_str)
    plt.close()

x = fit_per_sys_dict['sigma0_obs']/fit_per_sys_dict['sigma0_true']
x_qtls = np.quantile(x, q=[0.16,0.5,0.84])
ax = plot_fig_pdf_simple([x], [], x_min=1e-3, x_max=1e3, n_bins=50, log_x=True, normalize=False, lw=lw, xlabel_text=r'$\Sigma_{0,\rm obs}^%s/\Sigma_{0,\rm phys}^%s$' % (y_sym_star, y_sym_star), ylabel_text='Systems')
ax.text(x=0.98, y=0.95, s=r'$\frac{\Sigma_{0,\rm obs}^%s}{\Sigma_{0,\rm phys}^%s}$' % (y_sym_star, y_sym_star) + r'$ = {:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(x_qtls[1], x_qtls[1]-x_qtls[0], x_qtls[2]-x_qtls[1]), ha='right', va='top', fontsize=20, transform = ax.transAxes)
ax.axvline(x=1., lw=lw, linestyle='--')
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_phys_vs_obs_mmen_%s_sigma0_ratio_per_system.pdf' % prescription_str)
    plt.close()



# To plot sigma0_obs/sigma0_true vs scale_factor_obs/scale_factor_true for the simulated observed systems:
'''
scale_ratio_min, scale_ratio_max = 0.1, 2.
ax = plot_2d_points_and_contours_with_histograms(fit_per_sys_dict['scale_factor_obs']/fit_per_sys_dict['scale_factor_true'], fit_per_sys_dict['sigma0_obs']/fit_per_sys_dict['sigma0_true'], x_min=scale_ratio_min, x_max=scale_ratio_max, y_min=sigma0_ratio_min, y_max=sigma0_ratio_max, log_x=True, log_y=True, xlabel_text=r'$\alpha_{\rm obs}/\alpha_{\rm phys}$', ylabel_text=r'$\log_{10}(\Sigma_{0,\rm obs}^%s/\Sigma_{0,\rm phys}^%s)$' % (y_sym_star, y_sym_star), extra_text='Simulated observed systems', plot_qtls=True, x_symbol=r'$\frac{\alpha_{\rm obs}}{\alpha_{\rm phys}}$', y_symbol=r'$\frac{\Sigma_{0,\rm obs}^%s}{\Sigma_{0,\rm phys}^%s}$' % (y_sym_star, y_sym_star))
ax.plot([np.log10(scale_ratio_min), np.log10(scale_ratio_max)], [0., 0.], ls='--', lw=lw, color='b')
ax.plot([0., 0.], [np.log10(sigma0_ratio_min), np.log10(sigma0_ratio_max)], ls='--', lw=lw, color='b')
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_phys_vs_obs_mmen_%s_sigma0_ratio_vs_scaleup_ratio_per_system.pdf' % prescription_str)
    plt.close()
'''

# To plot sigma0_obs/sigma0_true vs. Mp_tot_obs/Mp_tot_true (ratio of total core masses of observed vs. all planets) for the simulated observed systems:
'''
Mp_ratio_min, Mp_ratio_max = 1e-1, 1.
ax = plot_2d_points_and_contours_with_histograms(fit_per_sys_dict['Mp_tot_obs']/fit_per_sys_dict['Mp_tot_true'], fit_per_sys_dict['sigma0_obs']/fit_per_sys_dict['sigma0_true'], x_min=Mp_ratio_min, x_max=Mp_ratio_max, y_min=sigma0_ratio_min, y_max=sigma0_ratio_max, log_x=False, log_y=True, xlabel_text=r'$M_{p,\rm tot,obs}/M_{p,\rm tot,phys}$', ylabel_text=r'$\log_{10}(\Sigma_{0,\rm obs}^%s/\Sigma_{0,\rm phys}^%s)$' % (y_sym_star, y_sym_star), extra_text='Simulated observed systems', plot_qtls=True, x_symbol=r'$\frac{M_{p,\rm tot,obs}}{M_{p,\rm tot,phys}}$', y_symbol=r'$\frac{\Sigma_{0,\rm obs}^%s}{\Sigma_{0,\rm phys}^%s}$' % (y_sym_star, y_sym_star))
ax.plot([1., 1.], [np.log10(sigma0_ratio_min), np.log10(sigma0_ratio_max)], ls='--', lw=lw, color='b')
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_phys_vs_obs_mmen_%s_sigma0_ratio_vs_mass_ratio_per_system.pdf' % prescription_str)
    plt.close()
'''
plt.show()




##### To plot some systems, including various feeding zones for each planet and power-law fits:
#plot_feeding_zones_and_power_law_fit_MMEN_per_system_observed_and_physical(sssp_per_sys, sssp, n_mult_min=3, prescription=prescription_str, a0=a0, scale_up=scale_up, N_sys=10)
