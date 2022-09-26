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
lw = 2 # linewidth

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size

# Parameters for defining the MMEN:
prescription_str = 'RC2014'
a0 = 0.3 # normalization separation for fitting power-laws





##### To plot the MMEN from planets in the observed catalogs only, compared to that of Kepler using the same M-R relation (re-drawn from our NWG18/Earth-like rocky composite model):

# Compute the MMSN for comparison:
a_array = np.linspace(1e-3,2,1001)
sigma_MMSN = MMSN(a_array)
#MeVeEa_masses = np.array([0.0553, 0.815, 1.]) # masses of Mercury, Venus, and Earth, in Earth masses
#MeVeEa_radii = np.array([0.383, 0.949, 1.]) # radii of Mercury, Venus, and Earth, in Earth radii
#MeVeEa_a = np.array([0.387, 0.723, 1.]) # semi-major axes of Mercury, Venus, and Earth, in AU
MeVeEa_sigmas = solid_surface_density_prescription(MeVeEa_masses, MeVeEa_radii, MeVeEa_a, prescription=prescription_str)

# Simulated observed planets:
sigma_obs_CL2013, mass_obs, a_obs = solid_surface_density_CL2013_given_observed_catalog(sss_per_sys)
sigma_obs_RC2014, mass_obs_2p, a_obs_2p, mult_obs_2p = solid_surface_density_RC2014_given_observed_catalog(sss_per_sys)

# Kepler observed planets:
sigma_obs_CL2013_Kep, mass_obs_Kep, a_obs_Kep = solid_surface_density_CL2013_given_observed_catalog(ssk_per_sys)
sigma_obs_RC2014_Kep, mass_obs_2p_Kep, a_obs_2p_Kep, mult_obs_2p_Kep = solid_surface_density_RC2014_given_observed_catalog(ssk_per_sys)

sigma0_obs_CL2013, beta_obs_CL2013 = fit_power_law_MMEN(a_obs, sigma_obs_CL2013, a0=a0)
sigma0_obs_RC2014, beta_obs_RC2014 = fit_power_law_MMEN(a_obs_2p, sigma_obs_RC2014, a0=a0)
sigma0_obs_CL2013_Kep, beta_obs_CL2013_Kep = fit_power_law_MMEN(a_obs_Kep, sigma_obs_CL2013_Kep, a0=a0)
sigma0_obs_RC2014_Kep, beta_obs_RC2014_Kep = fit_power_law_MMEN(a_obs_2p_Kep, sigma_obs_RC2014_Kep, a0=a0)
print('Sim. obs. (CL2013): sigma0 = {:0.4f}, beta = {:0.4f})'.format(sigma0_obs_CL2013, beta_obs_CL2013))
print('Sim. obs. (RC2014): sigma0 = {:0.4f}, beta = {:0.4f})'.format(sigma0_obs_RC2014, beta_obs_RC2014))
print('Kep. obs. (CL2013): sigma0 = {:0.4f}, beta = {:0.4f})'.format(sigma0_obs_CL2013_Kep, beta_obs_CL2013_Kep))
print('Kep. obs. (RC2014): sigma0 = {:0.4f}, beta = {:0.4f})'.format(sigma0_obs_RC2014_Kep, beta_obs_RC2014_Kep))

a_bins = np.logspace(np.log10(np.min(a_obs)), np.log10(np.max(a_obs)), n_bins+1)
a_bins_mid = (a_bins[1:]+a_bins[:-1])/2.
sigma_med_per_bin_CL2013 = [np.median(sigma_obs_CL2013[(a_obs >= a_bins[i]) & (a_obs < a_bins[i+1])]) for i in range(len(a_bins)-1)]
sigma_med_per_bin_RC2014 = [np.median(sigma_obs_RC2014[(a_obs_2p >= a_bins[i]) & (a_obs_2p < a_bins[i+1])]) for i in range(len(a_bins)-1)]
sigma_med_per_bin_CL2013_Kep = [np.median(sigma_obs_CL2013_Kep[(a_obs_Kep >= a_bins[i]) & (a_obs_Kep < a_bins[i+1])]) for i in range(len(a_bins)-1)]
sigma_med_per_bin_RC2014_Kep = [np.median(sigma_obs_RC2014_Kep[(a_obs_2p_Kep >= a_bins[i]) & (a_obs_2p_Kep < a_bins[i+1])]) for i in range(len(a_bins)-1)]



a_ticks = [0.05, 0.1, 0.2, 0.4, 0.8]

# CL2013 (delta_a = a):
fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.15,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
#corner.hist2d(a_obs, np.log10(sigma_obs), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
i_sample_plot = np.random.choice(np.arange(len(a_obs)), len(a_obs_Kep), replace=False)
plt.scatter(a_obs[i_sample_plot], np.log10(sigma_obs_CL2013[i_sample_plot]), marker='o', s=10, alpha=0.2, color='b', label='Simulated observed planets (CL13)')
plt.scatter(a_obs_Kep, np.log10(sigma_obs_CL2013_Kep), marker='o', s=10, alpha=0.2, color='k', label='Kepler observed planets (CL13)')
#plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_CL2013), drawstyle='steps-mid', lw=2, color='r', label='MMEN (median per bin)')
#plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_CL2013_Kep), drawstyle='steps-mid', lw=2, color='m', label='MMEN (median per bin for Kepler)')
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_obs_CL2013, beta_obs_CL2013, a0=a0)), lw=lw, ls='--', color='b', label=r'Simulated ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0_obs_CL2013, beta_obs_CL2013))
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_obs_CL2013_Kep, beta_obs_CL2013_Kep, a0=a0)), lw=lw, ls='--', color='k', label=r'Kepler ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0_obs_CL2013_Kep, beta_obs_CL2013_Kep))
plt.plot(a_array, np.log10(sigma_MMSN), lw=lw, color='g', label=r'MMSN ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(MMSN(a0), -1.5)) #label=r'MMSN ($\sigma_{\rm solid} = 10.89(a/{\rm AU})^{-3/2}$ g/cm$^2$)'
#plt.scatter(MeVeEa_a, np.log10(MeVeEa_sigmas), marker='o', s=100, color='g', label='') #label='Solar system planets (Mercury, Venus, Earth)'
ax.tick_params(axis='both', labelsize=afs)
plt.gca().set_xscale("log")
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
plt.xticks(a_ticks)
plt.xlim([0.04,0.9])
plt.ylim([-0.5,5.5])
plt.xlabel(r'Semimajor axis, $a$ (AU)', fontsize=20)
plt.ylabel(r'Surface density, $\log_{10}(\Sigma/{\rm gcm}^{-2})$', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.,1.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_obs_vs_Kepler_mmen_CL2013.pdf')
    plt.close()



# RC2014 (delta_a = geometric means):
fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.15,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
#corner.hist2d(a_obs, np.log10(sigma_obs), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
i_sample_plot = np.random.choice(np.arange(len(a_obs_2p)), len(a_obs_Kep), replace=False)
plt.scatter(a_obs_2p[i_sample_plot], np.log10(sigma_obs_RC2014[i_sample_plot]), marker='o', s=10, alpha=0.2, color='b', label='Simulated planets') # 'RC14'
plt.scatter(a_obs_2p_Kep, np.log10(sigma_obs_RC2014_Kep), marker='o', s=10, alpha=0.2, color='k', label='Kepler planets') # 'RC14'
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_obs_RC2014, beta_obs_RC2014, a0=a0)), lw=lw, ls='--', color='b', label=r'Simulated ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0_obs_RC2014, beta_obs_RC2014))
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_obs_RC2014_Kep, beta_obs_RC2014_Kep, a0=a0)), lw=lw, ls='--', color='k', label=r'Kepler ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0_obs_RC2014_Kep, beta_obs_RC2014_Kep))
plt.plot(a_array, np.log10(sigma_MMSN), lw=lw, color='g', label=r'MMSN ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(MMSN(a0), -1.5)) #label=r'MMSN ($\sigma_{\rm solid} = 10.89(a/{\rm AU})^{-3/2}$ g/cm$^2$)'
#plt.scatter(MeVeEa_a, np.log10(MeVeEa_sigmas), marker='o', s=100, color='g', label='') #label='Solar system planets (Mercury, Venus, Earth)'
plt.text(x=0.98, y=0.98, s='Observed catalogs', ha='right', va='top', fontsize=lfs, transform = ax.transAxes)
ax.tick_params(axis='both', labelsize=afs)
plt.gca().set_xscale("log")
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
plt.xticks(a_ticks)
plt.xlim([0.04,0.9])
plt.ylim([-0.5,5.5])
plt.xlabel(r'Semimajor axis, $a$ (AU)', fontsize=20)
plt.ylabel(r'Surface density, $\log_{10}(\Sigma/{\rm g\,cm}^{-2})$', fontsize=20)
plt.legend(loc='lower left', bbox_to_anchor=(0.,0.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_obs_vs_Kepler_mmen_RC2014.pdf')
    plt.close()

plt.show()





##### Repeat for DDA poster:

fig = plt.figure(figsize=(8,5))
plot = GridSpec(1,1,left=0.15,bottom=0.15,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
i_sample_plot = np.random.choice(np.arange(len(a_obs_2p)), len(a_obs_Kep), replace=False)
#plt.scatter(a_obs_2p[i_sample_plot], np.log10(sigma_obs_RC2014[i_sample_plot]), marker='o', s=10, alpha=0.2, color='b', label='Simulated observed planets (RC14)')
plt.scatter(a_obs_2p_Kep, np.log10(sigma_obs_RC2014_Kep), marker='o', s=10, alpha=0.2, color='k', label='Kepler observed planets (RC14)')
#plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_obs_RC2014, beta_obs_RC2014, a0=a0)), lw=lw, ls='--', color='b', label=r'Simulated ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0_obs_RC2014, beta_obs_RC2014))
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_obs_RC2014_Kep, beta_obs_RC2014_Kep, a0=a0)), lw=lw, ls='--', color='k', label=r'Kepler ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0_obs_RC2014_Kep, beta_obs_RC2014_Kep))
ax.tick_params(axis='both', labelsize=afs)
plt.gca().set_xscale("log")
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
plt.xticks(a_ticks)
plt.xlim([0.04,0.9])
plt.ylim([0.5,5.5])
plt.xlabel(r'Semi-major axis, $a$ (AU)', fontsize=20)
plt.ylabel(r'Surface density, $\log_{10}(\Sigma/{\rm gcm}^{-2})$', fontsize=20)
#plt.legend(loc='lower left', bbox_to_anchor=(0.,0.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_obs_vs_Kepler_mmen_RC2014_simple.pdf')
    plt.close()

plt.show()
