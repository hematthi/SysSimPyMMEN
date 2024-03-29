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

##### To load the physical catalogs:

# To load and process the simulated physical catalog of stars and planets:
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)





# Plotting parameters:
n_bins = 50
lw = 2 # linewidth

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size

# Parameters for defining the MMEN:
prescription_str = 'CL2013' #'RC2014'
a0 = 0.3 # normalization separation for fitting power-laws





##### To plot the solid surface density vs. semi-major axis for each planet in the physical catalog:

# Compute the MMSN for comparison:
a_array = np.linspace(1e-3,2,1001)
sigma_MMSN = MMSN(a_array)
MeVeEa_masses = np.array([0.0553, 0.815, 1.]) # masses of Mercury, Venus, and Earth, in Earth masses
MeVeEa_radii = np.array([0.383, 0.949, 1.]) # radii of Mercury, Venus, and Earth, in Earth radii
MeVeEa_a = np.array([0.387, 0.723, 1.]) # semi-major axes of Mercury, Venus, and Earth, in AU
MeVeEa_sigmas = solid_surface_density_prescription(MeVeEa_masses, MeVeEa_radii, MeVeEa_a, prescription=prescription_str)

sigma_all = solid_surface_density_prescription(sssp_per_sys['mass_all'], sssp_per_sys['radii_all'], sssp_per_sys['a_all'], Mstar=sssp['Mstar_all'][:,None], prescription=prescription_str)[sssp_per_sys['a_all'] > 0]
a_all = sssp_per_sys['a_all'][sssp_per_sys['a_all'] > 0]

sigma0, beta = fit_power_law_MMEN(a_all, sigma_all, a0=a0)

# Plot MMEN with MMSN (linear x-axes):
a_bins = np.linspace(np.min(a_all), np.max(a_all), n_bins+1)
a_bins_mid = (a_bins[1:]+a_bins[:-1])/2.
sigma_med_per_bin = [np.median(sigma_all[(a_all >= a_bins[i]) & (a_all < a_bins[i+1])]) for i in range(len(a_bins)-1)]

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
#corner.hist2d(a_all, np.log10(sigma_all), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
i_sample_plot = np.random.choice(np.arange(len(a_all)), 10000, replace=False)
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all[i_sample_plot]), marker='o', s=1, color='k', label='Simulated planets')
plt.plot(a_bins_mid, np.log10(sigma_med_per_bin), drawstyle='steps-mid', lw=3, color='r', label='MMEN (median per bin)')
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0, beta, a0=a0)), lw=3, ls='--', color='r', label=r'MMEN (linear fit: $\Sigma_0 = {:0.2f}$, $\beta = {:0.2f}$)'.format(sigma0, beta))
plt.plot(a_array, np.log10(sigma_MMSN), lw=3, color='g', label=r'MMSN ($\sigma_{\rm solid} = 10.89(a/{\rm AU})^{-3/2}$ g/cm$^2$)')
plt.scatter(MeVeEa_a, np.log10(MeVeEa_sigmas), marker='o', s=100, color='g', label='Solar system planets (Mercury, Venus, Earth)')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0.,1.1]) #[0.,0.45]
plt.ylim([0.,6.]) #[1.5,6.]
plt.xlabel(r'Semimajor axis $a$ (AU)', fontsize=20)
plt.ylabel(r'Surface density, $\log_{10}(\Sigma)$ (g/cm$^2$)', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.,1.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_mmen_%s_unlogged.pdf' % prescription_str)
    plt.close()

# Plot MMEN with MMSN (log-log axes):
a_bins = np.logspace(np.log10(np.min(a_all)), np.log10(np.max(a_all)), n_bins+1)
a_bins_mid = (a_bins[1:]+a_bins[:-1])/2.
sigma_med_per_bin = [np.median(sigma_all[(a_all >= a_bins[i]) & (a_all < a_bins[i+1])]) for i in range(len(a_bins)-1)]

a_ticks = [0.05, 0.1, 0.2, 0.4, 0.8]

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
#corner.hist2d(a_all, np.log10(sigma_all), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
i_sample_plot = np.random.choice(np.arange(len(a_all)), 10000, replace=False)
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all[i_sample_plot]), marker='o', s=10, alpha=0.2, color='k', label='Simulated planets')
#plt.plot(a_bins_mid, np.log10(sigma_med_per_bin), drawstyle='steps-mid', lw=3, color='r', label='MMEN (median per bin)')
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0, beta, a0=a0)), lw=3, ls='--', color='r', label=r'MMEN (linear fit: $\Sigma_0 = {:0.2f}$, $\beta = {:0.2f}$)'.format(sigma0, beta))
plt.plot(a_array, np.log10(sigma_MMSN), lw=3, color='g', label=r'MMSN ($\sigma_{\rm solid} = 10.89(a/{\rm AU})^{-3/2}$ g/cm$^2$)')
plt.scatter(MeVeEa_a, np.log10(MeVeEa_sigmas), marker='o', s=100, color='g', label='Solar system planets (Mercury, Venus, Earth)')
ax.tick_params(axis='both', labelsize=afs)
plt.gca().set_xscale("log")
plt.xticks(a_ticks)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
plt.xlim([0.04,0.9])
plt.ylim([0.,5.5])
plt.xlabel(r'Semimajor axis $a$ (AU)', fontsize=20)
plt.ylabel(r'Surface density, $\log_{10}(\Sigma)$ (g/cm$^2$)', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.,1.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_mmen_%s.pdf' % prescription_str)
    plt.close()

plt.show()

##### Remake for defense talk:

fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.15,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
#corner.hist2d(a_all, np.log10(sigma_all), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
i_sample_plot = np.random.choice(np.arange(len(a_all)), 10000, replace=False)
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all[i_sample_plot]), marker='.', s=1, alpha=1, color='k', label='Simulated planets')
#plt.plot(a_bins_mid, np.log10(sigma_med_per_bin), drawstyle='steps-mid', lw=3, color='r', label='MMEN (median per bin)')
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0, beta, a0=a0)), lw=3, ls='--', color='r', label=r'MMEN (linear fit: $\Sigma_0 = {:0.2f}$, $\beta = {:0.2f}$)'.format(sigma0, beta))
plt.plot(a_array, np.log10(sigma_MMSN), lw=3, color='g', label=r'MMSN ($\sigma_{\rm solid} = 10.89(a/{\rm AU})^{-3/2}$ g/cm$^2$)')
#plt.scatter(MeVeEa_a, np.log10(MeVeEa_sigmas), marker='o', s=100, color='g', label='Solar system planets (Mercury, Venus, Earth)')
ax.tick_params(axis='both', labelsize=afs)
plt.gca().set_xscale("log")
plt.xticks(a_ticks)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
plt.xlim([0.04,0.9])
plt.ylim([0.,5.5])
plt.xlabel(r'Semimajor axis, $a$ (AU)', fontsize=20)
plt.ylabel(r'Surface density, $\log_{10}(\Sigma)$ (g/cm$^2$)', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.,1.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_mmen_%s.pdf' % prescription_str)
    plt.close()

plt.show()





##### To test different prescriptions for the feeding zone width (delta_a):

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

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
i_sample_plot = np.random.choice(np.arange(len(a_all)), 1000, replace=False)
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all_CL2013[i_sample_plot]), marker='o', s=10, alpha=0.2, color='k')
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all_S2014[i_sample_plot]), marker='o', s=10, alpha=0.2, color='r')
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all_nHill10[i_sample_plot]), marker='o', s=10, alpha=0.2, color='b')
i_sample_plot = np.random.choice(np.arange(len(a_all_2p)), 1000, replace=False)
plt.scatter(a_all_2p[i_sample_plot], np.log10(sigma_all_RC2014[i_sample_plot]), marker='o', s=10, alpha=0.2, color='m')
#plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_CL2013), drawstyle='steps-mid', lw=3, color='k') #label=r'$\Delta{a} = a$'
#plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_S2014), drawstyle='steps-mid', lw=3, color='r') #label=r'$\Delta{a} = 2^{3/2}a(\frac{a M_p}{R_p M_\star})^{1/2}$'
#plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_nHill10), drawstyle='steps-mid', lw=3, color='b') #label=r'$\Delta{a} = 10 R_{\rm Hill}$'
#plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_RC2014), drawstyle='steps-mid', lw=3, color='m') #label=r'$\Delta{a} = \sqrt{a_{i+1} a_i} - \sqrt{a_i a_{i-1}}$ (multis only)'
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_CL2013, beta_CL2013, a0=a0)), lw=lw, ls='--', color='k', label=r'CL13 ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0_CL2013, beta_CL2013))
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_RC2014, beta_RC2014, a0=a0)), lw=lw, ls='--', color='m', label=r'RC14 ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0_RC2014, beta_RC2014))
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_nHill10, beta_nHill10, a0=a0)), lw=lw, ls='--', color='b', label=r'10Hill ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0_nHill10, beta_nHill10))
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_S2014, beta_S2014, a0=a0)), lw=lw, ls='--', color='r', label=r'S14 ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0_S2014, beta_S2014))
plt.plot(a_array, np.log10(sigma_MMSN), lw=lw, color='g', label=r'MMSN ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(MMSN(a0), -1.5)) #label=r'MMSN ($\sigma_{\rm solid} = 10.89(a/{\rm AU})^{-3/2}$ g/cm$^2$)'
plt.scatter(MeVeEa_a, np.log10(MeVeEa_sigmas), marker='o', s=100, color='g', label='') #label='Solar system planets (Mercury, Venus, Earth)'
ax.tick_params(axis='both', labelsize=afs)
plt.gca().set_xscale("log")
plt.xticks(a_ticks)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
plt.xlim([0.04,0.9])
plt.ylim([0.,5.5])
plt.xlabel(r'Semimajor axis, $a$ (AU)', fontsize=20)
plt.ylabel(r'Surface density, $\log_{10}(\Sigma)$ (g/cm$^2$)', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.,1.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_mmen_deltaa_compare.pdf')
    plt.close()
plt.show()

##### Remake for a smaller version (square panel):
fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1,left=0.15,bottom=0.15,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
#corner.hist2d(a_all_2p, np.log10(sigma_all_RC2014), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
i_sample_plot = np.random.choice(np.arange(len(a_all)), 1000, replace=False)
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all_CL2013[i_sample_plot]), marker='o', s=10, alpha=0.1, color='k')
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all_S2014[i_sample_plot]), marker='o', s=10, alpha=0.1, color='r')
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all_nHill10[i_sample_plot]), marker='o', s=10, alpha=0.1, color='b')
i_sample_plot = np.random.choice(np.arange(len(a_all_2p)), 1000, replace=False)
plt.scatter(a_all_2p[i_sample_plot], np.log10(sigma_all_RC2014[i_sample_plot]), marker='o', s=10, alpha=0.1, color='m')
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_CL2013, beta_CL2013, a0=a0)), lw=lw, ls='--', color='k', label=r'CL13 ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0_CL2013, beta_CL2013))
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_RC2014, beta_RC2014, a0=a0)), lw=lw, ls='--', color='m', label=r'RC14 ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0_RC2014, beta_RC2014))
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_nHill10, beta_nHill10, a0=a0)), lw=lw, ls='--', color='b', label=r'10Hill ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0_nHill10, beta_nHill10))
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_S2014, beta_S2014, a0=a0)), lw=lw, ls='--', color='r', label=r'S14 ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0_S2014, beta_S2014))
plt.plot(a_array, np.log10(sigma_MMSN), lw=lw, color='g', label=r'MMSN ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(MMSN(a0), -1.5)) #label=r'MMSN ($\sigma_{\rm solid} = 10.89(a/{\rm AU})^{-3/2}$ g/cm$^2$)'
#plt.scatter(MeVeEa_a, np.log10(MeVeEa_sigmas), marker='o', s=100, color='g', label='') #label='Solar system planets (Mercury, Venus, Earth)'
plt.text(x=0.98, y=0.98, s='Physical catalog', ha='right', va='top', fontsize=lfs, transform = ax.transAxes)
ax.tick_params(axis='both', labelsize=afs)
plt.gca().set_xscale("log")
plt.xticks(a_ticks)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
plt.xlim([0.04,0.9])
plt.ylim([-0.5,5.5])
plt.xlabel(r'Semimajor axis, $a$ (AU)', fontsize=20)
plt.ylabel(r'Surface density, $\log_{10}(\Sigma/{\rm g\,cm}^{-2})$', fontsize=20)
plt.legend(loc='lower left', bbox_to_anchor=(0.,0.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_mmen_deltaa_compare_square.pdf')
    plt.close()
plt.show()
