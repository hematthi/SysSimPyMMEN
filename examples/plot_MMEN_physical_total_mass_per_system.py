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
prescription_str = 'CL2013' #'RC2014'
a0 = 0.3 # normalization separation for fitting power-laws

# Compute the MMSN for comparison:
a_array = np.linspace(1e-3,2,1001)
sigma_MMSN = MMSN(a_array)
MeVeEa_masses = np.array([0.0553, 0.815, 1.]) # masses of Mercury, Venus, and Earth, in Earth masses
MeVeEa_a = np.array([0.387, 0.723, 1.]) # semi-major axes of Mercury, Venus, and Earth, in AU
MeVeEa_sigmas = solid_surface_density_CL2013(MeVeEa_masses, MeVeEa_a) # WARNING: need to replace with any prescription





##### To integrate the total mass in solids for the fitted power-laws as a function of separation:

fit_per_sys_dict = fit_power_law_MMEN_per_system_physical(sssp_per_sys, sssp, prescription=prescription_str, a0=a0)

r0 = a_from_P(3., 1.) # inner truncation radius/limit of integration, in AU
r_array = np.logspace(np.log10(r0+1e-6), np.log10(1.1), 100)

sigma0_ex1, beta_ex1 = 50., -2. # example with sigma0 = 50 g/cm^2 and beta = -2 (should be linear with log(r))
Mr_array_ex1 = np.array([solid_mass_integrated_r0_to_r_given_power_law_profile(r, r0, sigma0_ex1, beta_ex1) for r in r_array])
sigma0_ex2, beta_ex2 = 50., -1.5
Mr_array_ex2 = np.array([solid_mass_integrated_r0_to_r_given_power_law_profile(r, r0, sigma0_ex2, beta_ex2) for r in r_array])

# To plot the quantiles of total mass in solids as a function of separation:
Mr_array_all = []
for i in range(len(fit_per_sys_dict['sigma0'])):
    sigma0, beta = fit_per_sys_dict['sigma0'][i], fit_per_sys_dict['beta'][i]
    Mr_array = [solid_mass_integrated_r0_to_r_given_power_law_profile(r, r0, sigma0, beta, a0=a0) for r in r_array]
    Mr_array_all.append(Mr_array)
Mr_array_all = np.array(Mr_array_all)
Mr_array_qtls = np.quantile(Mr_array_all, [0.16,0.5,0.84], axis=0)

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
#for Mr_array in Mr_array_all:
#    plt.plot(r_array, Mr_array, lw=0.1, color='k')
plt.plot(r_array, Mr_array_ex1, lw=2, color='b', label=r'$\Sigma_0 = {:0.0f}$ g/cm$^2$, $\beta = {:0.1f}$'.format(sigma0_ex1, beta_ex1))
plt.plot(r_array, Mr_array_ex2, lw=2, color='r', label=r'$\Sigma_0 = {:0.0f}$ g/cm$^2$, $\beta = {:0.1f}$'.format(sigma0_ex2, beta_ex2))
plt.plot(r_array, Mr_array_qtls[1], lw=1, ls='--', color='k', label='Median')
plt.fill_between(r_array, Mr_array_qtls[0], Mr_array_qtls[2], alpha=0.2, label=r'16%-84% of systems')
ax.tick_params(axis='both', labelsize=afs)
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
a_ticks = [0.05, 0.1, 0.2, 0.4, 1.]
plt.xticks(a_ticks)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
plt.xlim([0.05,1.1])
plt.ylim([1e-2,2e2])
plt.xlabel(r'Separation from star, $r$ (AU)', fontsize=20)
plt.ylabel(r'Total solid mass within $r$, $M_r$ ($M_\oplus$)', fontsize=20)
plt.legend(loc='lower right', bbox_to_anchor=(1.,0.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_total_mass_vs_separation_RC2014_per_system.pdf')
    plt.close()

# To plot the distributions (CDFs) of total masses in solids for several separations:
r_examples = [0.1, 0.2, 0.5, 1.]
r_linestyles = [':', '-.', '--', '-']
r_Mr_arrays = []
for r in r_examples:
    Mr_array = np.array([solid_mass_integrated_r0_to_r_given_power_law_profile(r, r0, fit_per_sys_dict['sigma0'][i], fit_per_sys_dict['beta'][i], a0=a0) for i in range(len(fit_per_sys_dict['sigma0']))])
    r_Mr_arrays.append(Mr_array)

plot_fig_cdf_simple((8,5), r_Mr_arrays, [], x_min=1e-2, x_max=200., log_x=True, c_sim=['k']*len(r_examples), ls_sim=r_linestyles, lw=1, labels_sim=[r'$r = {:0.1f}$ AU'.format(r) for r in r_examples], xticks_custom=[0.01, 0.1, 1, 10, 100], xlabel_text='Total solid mass within $r$, $M_r$ ($M_\oplus$)', ylabel_text=r'Cumulative fraction with $M_r$', one_minus=True, afs=afs, tfs=tfs, lfs=lfs, legend=True, save_name=savefigures_directory + model_name + '_total_mass_CDFs_per_separation_RC2014_per_system.pdf', save_fig=savefigures)
plt.show()

# To repeat the above using the planet masses directly (i.e. step-functions in enclosed mass):
'''
Mr_pl_array_all = []
for i,a_sys in enumerate(sssp_per_sys['a_all'][:100000]): #100000
    M_sys = sssp_per_sys['mass_all'][i][a_sys > 0]
    a_sys = a_sys[a_sys > 0]
    if len(a_sys) > 1: # using multis for a fair comparison with the power-law fits, but can also count singles with this method
        Mr_pl_array = [np.sum(M_sys[a_sys <= r]) for r in r_array]
        Mr_pl_array_all.append(Mr_pl_array)
Mr_pl_array_all = np.array(Mr_pl_array_all)
Mr_pl_array_qtls = np.quantile(Mr_pl_array_all, [0.16,0.5,0.84], axis=0)

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
plt.plot(r_array, Mr_array_ex1, lw=2, color='b', label=r'$\Sigma_0 = {:0.0f}$ g/cm$^2$, $\beta = {:0.1f}$'.format(sigma0_ex1, beta_ex1))
plt.plot(r_array, Mr_array_ex2, lw=2, color='r', label=r'$\Sigma_0 = {:0.0f}$ g/cm$^2$, $\beta = {:0.1f}$'.format(sigma0_ex2, beta_ex2))
plt.plot(r_array, Mr_pl_array_qtls[1], lw=1, ls='--', color='k', label='Median')
plt.fill_between(r_array, Mr_pl_array_qtls[0], Mr_pl_array_qtls[2], alpha=0.2, label=r'16%-84% of systems')
ax.tick_params(axis='both', labelsize=afs)
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.xticks(a_ticks)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
plt.xlim([0.05,1.1])
plt.ylim([1e-2,2e2])
plt.xlabel(r'Separation from star, $r$ (AU)', fontsize=20)
plt.ylabel(r'Total solid mass within $r$, $M_r$ ($M_\oplus$)', fontsize=20)
plt.legend(loc='lower right', bbox_to_anchor=(1.,0.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_total_mass_pl_vs_separation_RC2014_per_system.pdf')
    plt.close()

r_Mr_pl_arrays = []
for r in r_examples:
    Mr_pl_array = []
    for i,a_sys in enumerate(sssp_per_sys['a_all'][:100000]): #100000
        M_sys = sssp_per_sys['mass_all'][i][a_sys > 0]
        a_sys = a_sys[a_sys > 0]
        if len(a_sys) > 1: # using multis for a fair comparison with the power-law fits, but can also count singles with this method
            Mr_pl_array.append(np.sum(M_sys[a_sys <= r]))
    r_Mr_pl_arrays.append(np.array(Mr_pl_array))
    print(r)

plot_fig_cdf_simple((8,5), r_Mr_pl_arrays, [], x_min=1e-2, x_max=200., log_x=True, c_sim=['k']*len(r_examples), ls_sim=r_linestyles, lw=1, labels_sim=[r'$r = {:0.1f}$ AU'.format(r) for r in r_examples], xticks_custom=[0.01, 0.1, 1, 10, 100], xlabel_text='Total solid mass within $r$, $M_r$ ($M_\oplus$)', ylabel_text=r'Cumulative fraction with $M_r$', one_minus=True, afs=afs, tfs=tfs, lfs=lfs, legend=True, save_name=savefigures_directory + model_name + '_total_mass_pl_CDFs_per_separation_RC2014_per_system.pdf', save_fig=savefigures)
plt.show()
'''
