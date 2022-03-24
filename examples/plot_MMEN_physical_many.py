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

    sigma_all_CL2013, a_all = solid_surface_density_CL2013_given_physical_catalog(sssp_per_sys_i)
    sigma_all_S2014, _ = solid_surface_density_S2014_given_physical_catalog(sssp_per_sys_i, sssp_i)
    sigma_all_nHill10, _ = solid_surface_density_nHill_given_physical_catalog(sssp_per_sys_i, sssp_i, n=10.)
    sigma_all_RC2014, a_all_2p, mult_all_2p = solid_surface_density_RC2014_given_physical_catalog(sssp_per_sys_i)

    sigma0_CL2013, beta_CL2013 = fit_power_law_MMEN(a_all, sigma_all_CL2013, a0=a0)
    sigma0_S2014, beta_S2014 = fit_power_law_MMEN(a_all, sigma_all_S2014, a0=a0)
    sigma0_nHill10, beta_nHill10 = fit_power_law_MMEN(a_all, sigma_all_nHill10, a0=a0)
    sigma0_RC2014, beta_RC2014 = fit_power_law_MMEN(a_all_2p, sigma_all_RC2014, a0=a0)

    fit_per_cat_dict['CL2013']['sigma0'].append(sigma0_CL2013)
    fit_per_cat_dict['CL2013']['beta'].append(beta_CL2013)

    fit_per_cat_dict['S2014']['sigma0'].append(sigma0_S2014)
    fit_per_cat_dict['S2014']['beta'].append(beta_S2014)

    fit_per_cat_dict['nHill']['sigma0'].append(sigma0_nHill10)
    fit_per_cat_dict['nHill']['beta'].append(beta_nHill10)

    fit_per_cat_dict['RC2014']['sigma0'].append(sigma0_RC2014)
    fit_per_cat_dict['RC2014']['beta'].append(beta_RC2014)

for pres in prescriptions:
    fit_per_cat_dict[pres]['sigma0'] = np.array(fit_per_cat_dict[pres]['sigma0'])
    fit_per_cat_dict[pres]['beta'] = np.array(fit_per_cat_dict[pres]['beta'])
    sigma0_qtls = np.quantile(fit_per_cat_dict[pres]['sigma0'], [0.16,0.5,0.84])
    beta_qtls = np.quantile(fit_per_cat_dict[pres]['beta'], [0.16,0.5,0.84])
    print('# %s:' % pres)
    print(r'$\Sigma_0$: ${:0.1f}_{{-{:0.1f} }}^{{+{:0.1f} }}$'.format(sigma0_qtls[1], sigma0_qtls[1]-sigma0_qtls[0], sigma0_qtls[2]-sigma0_qtls[1]))
    print(r'$\beta$: ${:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(beta_qtls[1], beta_qtls[1]-beta_qtls[0], beta_qtls[2]-beta_qtls[1]))





##### To plot the solid surface density vs. semi-major axis for each planet in the physical catalog:

# Compute the MMSN for comparison:
a_array = np.linspace(1e-3,2,1001)
sigma_MMSN = MMSN(a_array)
MeVeEa_masses = np.array([0.0553, 0.815, 1.]) # masses of Mercury, Venus, and Earth, in Earth masses
MeVeEa_radii = np.array([0.383, 0.949, 1.]) # radii of Mercury, Venus, and Earth, in Earth radii
MeVeEa_a = np.array([0.387, 0.723, 1.]) # semi-major axes of Mercury, Venus, and Earth, in AU
MeVeEa_sigmas = solid_surface_density_prescription(MeVeEa_masses, MeVeEa_radii, MeVeEa_a, prescription='CL2013')

##### To test different prescriptions for the feeding zone width (delta_a):

sigma_all_CL2013, a_all = solid_surface_density_CL2013_given_physical_catalog(sssp_per_sys)
sigma_all_S2014, _ = solid_surface_density_S2014_given_physical_catalog(sssp_per_sys, sssp)
sigma_all_nHill10, _ = solid_surface_density_nHill_given_physical_catalog(sssp_per_sys, sssp, n=10.)
sigma_all_RC2014, a_all_2p, mult_all_2p = solid_surface_density_RC2014_given_physical_catalog(sssp_per_sys)

sigma0_CL2013, beta_CL2013 = fit_power_law_MMEN(a_all, sigma_all_CL2013, a0=a0)
sigma0_S2014, beta_S2014 = fit_power_law_MMEN(a_all, sigma_all_S2014, a0=a0)
sigma0_nHill10, beta_nHill10 = fit_power_law_MMEN(a_all, sigma_all_nHill10, a0=a0)
sigma0_RC2014, beta_RC2014 = fit_power_law_MMEN(a_all_2p, sigma_all_RC2014, a0=a0)

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
i_sample_plot = np.random.choice(np.arange(len(a_all)), 1000, replace=False)
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all_CL2013[i_sample_plot]), marker='o', s=10, alpha=0.2, color='k')
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all_S2014[i_sample_plot]), marker='o', s=10, alpha=0.2, color='r')
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all_nHill10[i_sample_plot]), marker='o', s=10, alpha=0.2, color='b')
i_sample_plot = np.random.choice(np.arange(len(a_all_2p)), 1000, replace=False)
plt.scatter(a_all_2p[i_sample_plot], np.log10(sigma_all_RC2014[i_sample_plot]), marker='o', s=10, alpha=0.2, color='m')
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_CL2013, beta_CL2013, a0=a0)), lw=3, ls='--', color='k', label=r'$\Delta{a} = a$' + r' ($\Sigma_0 = {:0.2f}$, $\beta = {:0.2f}$)'.format(sigma0_CL2013, beta_CL2013))
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_S2014, beta_S2014, a0=a0)), lw=3, ls='--', color='r', label=r'$\Delta{a} = 2^{3/2}a(\frac{a M_p}{R_p M_\star})^{1/2}$' + r' ($\Sigma_0 = {:0.2f}$, $\beta = {:0.2f}$)'.format(sigma0_S2014, beta_S2014))
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_nHill10, beta_nHill10, a0=a0)), lw=3, ls='--', color='b', label=r'$\Delta{a} = 10 R_{\rm Hill}$' + r' ($\Sigma_0 = {:0.2f}$, $\beta = {:0.2f}$)'.format(sigma0_nHill10, beta_nHill10))
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_RC2014, beta_RC2014, a0=a0)), lw=3, ls='--', color='m', label=r'$\Delta{a} = \sqrt{a_{i+1} a_i} - \sqrt{a_i a_{i-1}}$, multis only' + r' ($\Sigma_0 = {:0.2f}$, $\beta = {:0.2f}$)'.format(sigma0_RC2014, beta_RC2014))
plt.plot(a_array, np.log10(sigma_MMSN), lw=3, color='g', label=r'MMSN ($\Sigma_0 = {:0.2f}$, $\beta = {:0.2f}$)'.format(MMSN(a0), -1.5)) #label=r'MMSN ($\sigma_{\rm solid} = 10.89(a/{\rm AU})^{-3/2}$ g/cm$^2$)'
plt.scatter(MeVeEa_a, np.log10(MeVeEa_sigmas), marker='o', s=100, color='g', label='') #label='Solar system planets (Mercury, Venus, Earth)'
ax.tick_params(axis='both', labelsize=afs)
plt.gca().set_xscale("log")
a_ticks = [0.05, 0.1, 0.2, 0.4, 0.8]
plt.xticks(a_ticks)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
plt.xlim([0.04,0.9])
plt.ylim([0.,5.5])
plt.xlabel(r'Semimajor axis, $a$ (AU)', fontsize=20)
plt.ylabel(r'Surface density, $\log_{10}(\Sigma_{\rm solid})$ (g/cm$^2$)', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.,1.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_mmen_deltaa_compare_credible.pdf')
    plt.close()
plt.show()
