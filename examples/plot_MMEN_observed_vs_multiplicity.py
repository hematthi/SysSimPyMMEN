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





##### To plot the MMEN from planets in the observed catalogs only, compared to that of Kepler using the same M-R relation (re-drawn from our NWG18/Earth-like rocky composite model):

# Compute the MMSN for comparison:
a_array = np.linspace(1e-3,2,1001)
sigma_MMSN = MMSN(a_array)
MeVeEa_masses = np.array([0.0553, 0.815, 1.]) # masses of Mercury, Venus, and Earth, in Earth masses
MeVeEa_a = np.array([0.387, 0.723, 1.]) # semi-major axes of Mercury, Venus, and Earth, in AU
MeVeEa_sigmas = solid_surface_density_prescription(MeVeEa_masses, MeVeEa_a)

# Simulated observed planets:
sigma_obs_RC2014, mass_obs_2p, a_obs_2p, mult_obs_2p = solid_surface_density_RC2014_given_observed_catalog(sss_per_sys)

# Kepler observed planets:
sigma_obs_RC2014_Kep, mass_obs_2p_Kep, a_obs_2p_Kep, mult_obs_2p_Kep = solid_surface_density_RC2014_given_observed_catalog(ssk_per_sys)

sigma0_obs_RC2014, beta_obs_RC2014 = fit_power_law_MMEN(a_obs_2p, sigma_obs_RC2014, a0=a0)
sigma0_obs_RC2014_Kep, beta_obs_RC2014_Kep = fit_power_law_MMEN(a_obs_2p_Kep, sigma_obs_RC2014_Kep, a0=a0)
print('Sim. obs. (RC2014): sigma0 = {:0.4f}, beta = {:0.4f})'.format(sigma0_obs_RC2014, beta_obs_RC2014))
print('Kep. obs. (RC2014): sigma0 = {:0.4f}, beta = {:0.4f})'.format(sigma0_obs_RC2014_Kep, beta_obs_RC2014_Kep))

a_bins = np.logspace(np.log10(np.min(a_obs_2p)), np.log10(np.max(a_obs_2p)), n_bins+1)
a_bins_mid = (a_bins[1:]+a_bins[:-1])/2.
sigma_med_per_bin_RC2014 = [np.median(sigma_obs_RC2014[(a_obs_2p >= a_bins[i]) & (a_obs_2p < a_bins[i+1])]) for i in range(len(a_bins)-1)]
sigma_med_per_bin_RC2014_Kep = [np.median(sigma_obs_RC2014_Kep[(a_obs_2p_Kep >= a_bins[i]) & (a_obs_2p_Kep < a_bins[i+1])]) for i in range(len(a_bins)-1)]



##### To test the RC2014 prescription for delta_a (geometric means) as a function of observed multiplicity:

a_ticks = [0.05, 0.1, 0.2, 0.4, 0.8]

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
i_sample_plot = np.random.choice(np.arange(len(a_obs_2p)), 1000, replace=False)
plt.scatter(a_obs_2p[i_sample_plot], np.log10(sigma_obs_RC2014[i_sample_plot]), marker='.', s=1, color='k')
plt.scatter(a_obs_2p_Kep, np.log10(sigma_obs_RC2014_Kep), marker='.', s=1, color='b')
for m in range(2,5):
    a_obs_2p_m = a_obs_2p[mult_obs_2p == m]
    sigma_obs_RC2014_m = sigma_obs_RC2014[mult_obs_2p == m]
    sigma_med_per_bin_RC2014_m = [np.median(sigma_obs_RC2014_m[(a_obs_2p_m >= a_bins[i]) & (a_obs_2p_m < a_bins[i+1])]) for i in range(len(a_bins)-1)]
    a_obs_2p_Kep_m = a_obs_2p_Kep[mult_obs_2p_Kep == m]
    sigma_obs_RC2014_Kep_m = sigma_obs_RC2014_Kep[mult_obs_2p_Kep == m]
    sigma_med_per_bin_RC2014_Kep_m = [np.median(sigma_obs_RC2014_Kep_m[(a_obs_2p_Kep_m >= a_bins[i]) & (a_obs_2p_Kep_m < a_bins[i+1])]) for i in range(len(a_bins)-1)]

    sigma0_obs_RC2014_m, beta_obs_RC2014_m = fit_power_law_MMEN(a_obs_2p_m, sigma_obs_RC2014_m, a0=a0)
    sigma0_obs_RC2014_Kep_m, beta_obs_RC2014_Kep_m = fit_power_law_MMEN(a_obs_2p_Kep_m, sigma_obs_RC2014_Kep_m, a0=a0)
    print('Sim. obs. (m = %s): ' % m + 'sigma0 = {:0.4f}, beta = {:0.4f})'.format(sigma0_obs_RC2014_m, beta_obs_RC2014_m))
    print('Kep. obs. (m = %s): ' % m + 'sigma0 = {:0.4f}, beta = {:0.4f})'.format(sigma0_obs_RC2014_Kep_m, beta_obs_RC2014_Kep_m))

    plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_RC2014_m), drawstyle='steps-mid', lw=2, label=r'$m = %s$' % m)
    plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_RC2014_Kep_m), drawstyle='steps-mid', ls='--', lw=2, label=r'$m = %s (Kepler)$' % m)
plt.plot(a_array, np.log10(sigma_MMSN), lw=2, color='g')
plt.scatter(MeVeEa_a, np.log10(MeVeEa_sigmas), marker='o', s=100, color='g')
plt.gca().set_xscale("log")
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(a_ticks)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
plt.xlim([0.04,1.1])
plt.ylim([-0.5,5.5])
plt.xlabel(r'Semimajor axis $a$ (AU)', fontsize=20)
plt.ylabel(r'Surface density $\log_{10}(\sigma_{\rm solid})$ (g/cm$^2$)', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.,1.), ncol=2, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_obs_vs_Kepler_mmen_RC2014_per_mult.pdf')
    plt.close()

plt.show()
