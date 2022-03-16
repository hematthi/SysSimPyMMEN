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
prescription_str = 'CL2013' #'RC2014' # make sure this actually matches the prescription used!
solid_surface_density_prescription = solid_surface_density_CL2013 #solid_surface_density_system_RC2014
a0 = 0.3 # normalization separation for fitting power-laws

# Compute the MMSN for comparison:
a_array = np.linspace(1e-3,2,1001)
sigma_MMSN = MMSN(a_array)
MeVeEa_masses = np.array([0.0553, 0.815, 1.]) # masses of Mercury, Venus, and Earth, in Earth masses
MeVeEa_a = np.array([0.387, 0.723, 1.]) # semi-major axes of Mercury, Venus, and Earth, in AU
MeVeEa_sigmas = solid_surface_density_prescription(MeVeEa_masses, MeVeEa_a)





##### To test different prescriptions for the feeding zone width (delta_a):

mult_all = np.repeat(sssp_per_sys['Mtot_all'], sssp_per_sys['Mtot_all'])
sigma_all_CL2013, a_all = solid_surface_density_CL2013_given_physical_catalog(sssp_per_sys)
sigma_all_S2014, _ = solid_surface_density_S2014_given_physical_catalog(sssp_per_sys, sssp)
sigma_all_nHill10, _ = solid_surface_density_nHill_given_physical_catalog(sssp_per_sys, sssp, n=10.)
sigma_all_RC2014, a_all_2p, mult_all_2p = solid_surface_density_RC2014_given_physical_catalog(sssp_per_sys)

sigma0_CL2013, beta_CL2013 = fit_power_law_MMEN(a_all, sigma_all_CL2013, a0=a0)
sigma0_S2014, beta_S2014 = fit_power_law_MMEN(a_all, sigma_all_S2014, a0=a0)
sigma0_nHill10, beta_nHill10 = fit_power_law_MMEN(a_all, sigma_all_nHill10, a0=a0)
sigma0_RC2014, beta_RC2014 = fit_power_law_MMEN(a_all_2p, sigma_all_RC2014, a0=a0)

a_bins = np.logspace(np.log10(np.min(a_all)), np.log10(np.max(a_all)), n_bins+1)
a_bins_mid = (a_bins[1:]+a_bins[:-1])/2.
sigma_med_per_bin_CL2013 = [np.median(sigma_all_CL2013[(a_all >= a_bins[i]) & (a_all < a_bins[i+1])]) for i in range(len(a_bins)-1)]
sigma_med_per_bin_S2014 = [np.median(sigma_all_S2014[(a_all >= a_bins[i]) & (a_all < a_bins[i+1])]) for i in range(len(a_bins)-1)]
sigma_med_per_bin_nHill10 = [np.median(sigma_all_nHill10[(a_all >= a_bins[i]) & (a_all < a_bins[i+1])]) for i in range(len(a_bins)-1)]
sigma_med_per_bin_RC2014 = [np.median(sigma_all_RC2014[(a_all_2p >= a_bins[i]) & (a_all_2p < a_bins[i+1])]) for i in range(len(a_bins)-1)] # contains planets in multis (2+) only

##### To test the MMEN as a function of intrinsic multiplicity:

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
i_sample_plot = np.random.choice(np.arange(len(a_all_2p)), 1000, replace=False)
#plt.scatter(a_all_2p[i_sample_plot], np.log10(sigma_all_RC2014[i_sample_plot]), marker='.', s=1, color='m')
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all_CL2013[i_sample_plot]), marker='.', s=1, color='m')
for n in range(2,11):
    #a_all_n = a_all_2p[mult_all_2p == n]
    #sigma_all_n = sigma_all_RC2014[mult_all_2p == n]
    a_all_n = a_all[mult_all == n]
    sigma_all_n = sigma_all_CL2013[mult_all == n]
    sigma_med_per_bin_n = [np.median(sigma_all_n[(a_all_n >= a_bins[i]) & (a_all_n < a_bins[i+1])]) for i in range(len(a_bins)-1)]

    sigma0_n, beta_n = fit_power_law_MMEN(a_all_n, sigma_all_n, a0=a0)
    print('n = %s:' % n + 'sigma0 = {:0.4f}, beta = {:0.4f})'.format(sigma0_n, beta_n))

    plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_n), drawstyle='steps-mid', lw=2, label=r'$n = %s$' % n) #label=r'$n = %s$' % n
    #plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_n, beta_n, a0=a0)), lw=1, ls='--', color='k', label=r'$n = %s$' % n + r' ($\Sigma_0 = {:0.2f}$, $\beta = {:0.2f}$)'.format(sigma0_n, beta_n))
plt.plot(a_array, np.log10(sigma_MMSN), lw=2, color='g')
plt.scatter(MeVeEa_a, np.log10(MeVeEa_sigmas), marker='o', s=100, color='g')
ax.tick_params(axis='both', labelsize=afs)
plt.gca().set_xscale("log")
plt.xlim([0.04,1.1])
plt.ylim([-0.5,5.5])
plt.xlabel(r'Semimajor axis $a$ (AU)', fontsize=20)
plt.ylabel(r'Surface density $\log_{10}(\sigma_{\rm solid})$ (g/cm$^2$)', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.,1.), ncol=2, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_mmen_RC2014_per_mult.pdf')
    plt.close()
plt.show()
