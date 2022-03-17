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

##### To fit a line to each system:
a0 = 0.3

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
n_sys_all = []
sigma0_beta_RC2014_all = []
for i,a_sys in enumerate(sssp_per_sys['a_all'][:10000]): #100000
    M_sys = sssp_per_sys['mass_all'][i][a_sys > 0]
    a_sys = a_sys[a_sys > 0]
    if len(a_sys) > 1:
        #print(i)
        n_sys_all.append(len(a_sys))
        sigma_sys_RC2014 = solid_surface_density_system_RC2014(M_sys, a_sys)

        sigma0_RC2014, beta_RC2014 = fit_power_law_MMEN(a_sys, sigma_sys_RC2014, a0=a0)
        sigma0_beta_RC2014_all.append([sigma0_RC2014, beta_RC2014])

        plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_RC2014, beta_RC2014, a0=a0)), lw=0.1, ls='-', alpha=1, color='k')
n_sys_all = np.array(n_sys_all)
sigma0_beta_RC2014_all = np.array(sigma0_beta_RC2014_all)
sigma0_RC2014_qtls = np.quantile(sigma0_beta_RC2014_all[:,0], q=[0.16,0.5,0.84])
beta_RC2014_qtls = np.quantile(sigma0_beta_RC2014_all[:,1], q=[0.16,0.5,0.84])
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
    plt.savefig(savefigures_directory + model_name + '_mmen_RC2014_per_system.pdf')
    plt.close()

# To plot the distribution of fitted power-law parameters (sigma0 vs. beta):
x, y = sigma0_beta_RC2014_all[:,1], sigma0_beta_RC2014_all[:,0] # beta's, sigma0's
plot_2d_points_and_contours_with_histograms(x, y, x_min=-8., x_max=4., y_min=1., y_max=1e5, log_y=True, xlabel_text=r'$\beta$', ylabel_text=r'$\log_{10}(\Sigma_0/{\rm g cm^{-2}})$', extra_text='Simulated physical systems', plot_qtls=True, y_str_format='{:0.1f}', x_symbol=r'$\beta$', y_symbol=r'$\Sigma_0$', save_name=savefigures_directory + model_name + '_mmen_RC2014_sigma0_vs_beta_per_system.pdf', save_fig=savefigures)

# To repeat the above (plot sigma0 vs. beta) per intrinsic multiplicity:
for n in range(2,9):
    print(np.sum(n_sys_all == n))
    x = sigma0_beta_RC2014_all[:,1][n_sys_all == n] # beta's
    y = sigma0_beta_RC2014_all[:,0][n_sys_all == n] # sigma0's
    plot_2d_points_and_contours_with_histograms(x, y, x_min=-8., x_max=4., y_min=1., y_max=1e5, log_y=True, xlabel_text=r'$\beta$', ylabel_text=r'$\log_{10}(\Sigma_0/{\rm g cm^{-2}})$', extra_text=r'$n = %s$' % n, plot_qtls=True, y_str_format='{:0.1f}', x_symbol=r'$\beta$', y_symbol=r'$\Sigma_0$', save_name=savefigures_directory + model_name + '_mmen_RC2014_sigma0_vs_beta_per_system_n%s.pdf' % n, save_fig=savefigures)

plt.show()

# To change the normalization point a0 for sigma0:
'''
a0_array = np.logspace(np.log10(a_from_P(3.,1.)), np.log10(a_from_P(300.,1.)), 11)
for a0 in a0_array:
    fit_per_sys_dict = fit_power_law_MMEN_per_system_physical(sssp_per_sys, sssp, prescription=prescription_str, a0=a0)

    x, y = fit_per_sys_dict['beta'], fit_per_sys_dict['sigma0'] # beta's, sigma0's
    plot_2d_points_and_contours_with_histograms(x, y, x_min=-10., x_max=5., y_min=1e-4, y_max=1e6, log_y=True, xlabel_text=r'$\beta$', ylabel_text=r'$\log_{10}(\Sigma_0/{\rm g cm^{-2}})$', extra_text=r'$\Sigma_0 = \Sigma({:0.3f} AU)$'.format(a0), plot_qtls=True, y_str_format='{:0.1f}', x_symbol=r'$\beta$', y_symbol=r'$\Sigma_0$', save_name=savefigures_directory + model_name + '_mmen_RC2014_sigma0_vs_beta_per_system_a0_{:0.3f}.pdf'.format(a0), save_fig=savefigures)
plt.show()
'''
