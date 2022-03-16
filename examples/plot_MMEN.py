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





##### Functions to fit power-laws to MMEN (linear fits to log(sigma) vs. log(a)):

# sigma = sigma0 * (a/a0)^beta <==> log(sigma) = log(sigma0) + beta*log(a/a0)

f_linear = lambda x, p0, p1: p0 + p1*x
p0, p1 = [1., -1.5] # initial guesses for log(sigma0), beta
a0 = 1. # distance for normalization sigma0 = sigma(a0); fit lines to x=np.log10(a/a0)





##### To plot the solid surface density (MMEN) for the physical catalogs:

n_bins = 50
lw = 1 #linewidth

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size



#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=# Transferred below to "plot_MMEN_physical.py"

# Compute the MMSN:
a_array = np.linspace(1e-3,2,1001)
sigma_MMSN = MMSN(a_array)
MeVeEa_masses = np.array([0.0553, 0.815, 1.]) # masses of Mercury, Venus, and Earth, in Earth masses
MeVeEa_a = np.array([0.387, 0.723, 1.]) # semi-major axes of Mercury, Venus, and Earth, in AU
MeVeEa_sigmas = solid_surface_density_CL2013(MeVeEa_masses, MeVeEa_a)

a_all = sssp_per_sys['a_all'][sssp_per_sys['a_all'] > 0]
sigma_all = solid_surface_density_CL2013(sssp_per_sys['mass_all'], sssp_per_sys['a_all'])[sssp_per_sys['a_all'] > 0]

mmen_fit = scipy.optimize.curve_fit(f_linear, np.log10(a_all/a0), np.log10(sigma_all), [p0, p1])[0]
sigma0, beta = 10.**(mmen_fit[0]), mmen_fit[1]

# Plot MMEN with MMSN (linear x-axes):
a_bins = np.linspace(np.min(a_all), np.max(a_all), n_bins+1)
a_bins_mid = (a_bins[1:]+a_bins[:-1])/2.
sigma_med_per_bin = [np.median(sigma_all[(a_all >= a_bins[i]) & (a_all < a_bins[i+1])]) for i in range(len(a_bins)-1)]

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
#corner.hist2d(a_all, np.log10(sigma_all), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
i_sample_plot = np.random.choice(np.arange(len(a_all)), 10000, replace=False)
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all[i_sample_plot]), marker='.', s=1, color='k', label='Simulated planets')
plt.plot(a_bins_mid, np.log10(sigma_med_per_bin), drawstyle='steps-mid', lw=2, color='r', label='MMEN (median per bin)')
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0, beta, a0=a0)), lw=1, ls='--', color='r', label=r'MMEN (linear fit: $\Sigma_0 = {:0.2f}$, $\beta = {:0.2f}$)'.format(sigma0, beta))
plt.plot(a_array, np.log10(sigma_MMSN), lw=2, color='g', label=r'MMSN ($\sigma_{\rm solid} = 10.89(a/{\rm AU})^{-3/2}$ g/cm$^2$)')
plt.scatter(MeVeEa_a, np.log10(MeVeEa_sigmas), marker='o', s=100, color='g', label='Solar system planets (Mercury, Venus, Earth)')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0.,1.1]) #[0.,0.45]
plt.ylim([0.,6.]) #[1.5,6.]
plt.xlabel(r'Semimajor axis $a$ (AU)', fontsize=20)
plt.ylabel(r'Surface density $\log_{10}(\sigma_{\rm solid})$ (g/cm$^2$)', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.,1.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_mmen_CL2013_unlogged.pdf')
    plt.close()

# Plot MMEN with MMSN (log-log axes):
a_bins = np.logspace(np.log10(np.min(a_all)), np.log10(np.max(a_all)), n_bins+1)
a_bins_mid = (a_bins[1:]+a_bins[:-1])/2.
sigma_med_per_bin = [np.median(sigma_all[(a_all >= a_bins[i]) & (a_all < a_bins[i+1])]) for i in range(len(a_bins)-1)]

a_ticks = [0.05, 0.1, 0.2, 0.4, 1.]

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
#corner.hist2d(a_all, np.log10(sigma_all), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
i_sample_plot = np.random.choice(np.arange(len(a_all)), 10000, replace=False)
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all[i_sample_plot]), marker='.', s=1, color='k', label='Simulated planets')
plt.plot(a_bins_mid, np.log10(sigma_med_per_bin), drawstyle='steps-mid', lw=2, color='r', label='MMEN (median per bin)')
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0, beta, a0=a0)), lw=1, ls='--', color='r', label=r'MMEN (linear fit: $\Sigma_0 = {:0.2f}$, $\beta = {:0.2f}$)'.format(sigma0, beta))
plt.plot(a_array, np.log10(sigma_MMSN), lw=2, color='g', label=r'MMSN ($\sigma_{\rm solid} = 10.89(a/{\rm AU})^{-3/2}$ g/cm$^2$)')
plt.scatter(MeVeEa_a, np.log10(MeVeEa_sigmas), marker='o', s=100, color='g', label='Solar system planets (Mercury, Venus, Earth)')
ax.tick_params(axis='both', labelsize=afs)
plt.gca().set_xscale("log")
plt.xticks(a_ticks)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
plt.xlim([0.04,1.1])
plt.ylim([-0.5,5.5])
plt.xlabel(r'Semimajor axis $a$ (AU)', fontsize=20)
plt.ylabel(r'Surface density $\log_{10}(\sigma_{\rm solid})$ (g/cm$^2$)', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.,1.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_mmen_CL2013.pdf')
    plt.close()





##### To test different prescriptions for the feeding zone width (delta_a) in the MMEN, applied to the physical catalogs:

a_all = sssp_per_sys['a_all'][sssp_per_sys['a_all'] > 0]
sigma_all_CL2013 = solid_surface_density_CL2013(sssp_per_sys['mass_all'], sssp_per_sys['a_all'])[sssp_per_sys['a_all'] > 0] # delta_a = a
sigma_all_S2014 = solid_surface_density_S2014(sssp_per_sys['mass_all'], sssp_per_sys['radii_all'], sssp_per_sys['a_all'], Mstar=sssp['Mstar_all'][:,None])[sssp_per_sys['a_all'] > 0]
sigma_all_nHill10 = solid_surface_density_nHill(sssp_per_sys['mass_all'], sssp_per_sys['a_all'], Mstar=sssp['Mstar_all'][:,None], n=10.)[sssp_per_sys['a_all'] > 0]

a_all_2p = np.array([])
mult_all_2p = np.array([])
sigma_all_RC2014 = np.array([])
start = time.time()
for i,a_sys in enumerate(sssp_per_sys['a_all'][:100000]):
    M_sys = sssp_per_sys['mass_all'][i][a_sys > 0]
    a_sys = a_sys[a_sys > 0]
    if len(a_sys) > 1:
        #print(i)
        a_all_2p = np.append(a_all_2p, a_sys)
        mult_all_2p = np.append(mult_all_2p, len(a_sys)*np.ones(len(a_sys)))
        sigma_all_RC2014 = np.append(sigma_all_RC2014, solid_surface_density_system_RC2014(M_sys, a_sys))
stop = time.time()
print('Time: %s s' % (stop - start))

mmen_fit_CL2013 = scipy.optimize.curve_fit(f_linear, np.log10(a_all/a0), np.log10(sigma_all_CL2013), [p0, p1])[0]
mmen_fit_S2014 = scipy.optimize.curve_fit(f_linear, np.log10(a_all/a0), np.log10(sigma_all_S2014), [p0, p1])[0]
mmen_fit_nHill10 = scipy.optimize.curve_fit(f_linear, np.log10(a_all/a0), np.log10(sigma_all_nHill10), [p0, p1])[0]
mmen_fit_RC2014 = scipy.optimize.curve_fit(f_linear, np.log10(a_all_2p/a0), np.log10(sigma_all_RC2014), [p0, p1])[0]
sigma0_CL2013, beta_CL2013 = 10.**(mmen_fit_CL2013[0]), mmen_fit_CL2013[1]
sigma0_S2014, beta_S2014 = 10.**(mmen_fit_S2014[0]), mmen_fit_S2014[1]
sigma0_nHill10, beta_nHill10 = 10.**(mmen_fit_nHill10[0]), mmen_fit_nHill10[1]
sigma0_RC2014, beta_RC2014 = 10.**(mmen_fit_RC2014[0]), mmen_fit_RC2014[1]

a_bins = np.logspace(np.log10(np.min(a_all)), np.log10(np.max(a_all)), n_bins+1)
a_bins_mid = (a_bins[1:]+a_bins[:-1])/2.
sigma_med_per_bin_CL2013 = [np.median(sigma_all_CL2013[(a_all >= a_bins[i]) & (a_all < a_bins[i+1])]) for i in range(len(a_bins)-1)]
sigma_med_per_bin_S2014 = [np.median(sigma_all_S2014[(a_all >= a_bins[i]) & (a_all < a_bins[i+1])]) for i in range(len(a_bins)-1)]
sigma_med_per_bin_nHill10 = [np.median(sigma_all_nHill10[(a_all >= a_bins[i]) & (a_all < a_bins[i+1])]) for i in range(len(a_bins)-1)]
sigma_med_per_bin_RC2014 = [np.median(sigma_all_RC2014[(a_all_2p >= a_bins[i]) & (a_all_2p < a_bins[i+1])]) for i in range(len(a_bins)-1)] # contains planets in multis (2+) only

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
#corner.hist2d(a_all, np.log10(sigma_all), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
i_sample_plot = np.random.choice(np.arange(len(a_all)), 1000, replace=False)
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all_CL2013[i_sample_plot]), marker='.', s=1, color='k')
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all_S2014[i_sample_plot]), marker='.', s=1, color='r')
plt.scatter(a_all[i_sample_plot], np.log10(sigma_all_nHill10[i_sample_plot]), marker='.', s=1, color='b')
i_sample_plot = np.random.choice(np.arange(len(a_all_2p)), 1000, replace=False)
plt.scatter(a_all_2p[i_sample_plot], np.log10(sigma_all_RC2014[i_sample_plot]), marker='.', s=1, color='m')
plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_CL2013), drawstyle='steps-mid', lw=2, color='k') #label=r'$\Delta{a} = a$'
plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_S2014), drawstyle='steps-mid', lw=2, color='r') #label=r'$\Delta{a} = 2^{3/2}a(\frac{a M_p}{R_p M_\star})^{1/2}$'
plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_nHill10), drawstyle='steps-mid', lw=2, color='b') #label=r'$\Delta{a} = 10 R_{\rm Hill}$'
plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_RC2014), drawstyle='steps-mid', lw=2, color='m') #label=r'$\Delta{a} = \sqrt{a_{i+1} a_i} - \sqrt{a_i a_{i-1}}$ (multis only)'
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_CL2013, beta_CL2013, a0=a0)), lw=1, ls='--', color='k', label=r'$\Delta{a} = a$' + r' ($\Sigma_0 = {:0.2f}$, $\beta = {:0.2f}$)'.format(sigma0_CL2013, beta_CL2013))
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_S2014, beta_S2014, a0=a0)), lw=1, ls='--', color='r', label=r'$\Delta{a} = 2^{3/2}a(\frac{a M_p}{R_p M_\star})^{1/2}$' + r' ($\Sigma_0 = {:0.2f}$, $\beta = {:0.2f}$)'.format(sigma0_S2014, beta_S2014))
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_nHill10, beta_nHill10, a0=a0)), lw=1, ls='--', color='b', label=r'$\Delta{a} = 10 R_{\rm Hill}$' + r' ($\Sigma_0 = {:0.2f}$, $\beta = {:0.2f}$)'.format(sigma0_nHill10, beta_nHill10))
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_RC2014, beta_RC2014, a0=a0)), lw=1, ls='--', color='m', label=r'$\Delta{a} = \sqrt{a_{i+1} a_i} - \sqrt{a_i a_{i-1}}$, multis only' + r' ($\Sigma_0 = {:0.2f}$, $\beta = {:0.2f}$)'.format(sigma0_RC2014, beta_RC2014))
plt.plot(a_array, np.log10(sigma_MMSN), lw=2, color='g', label=r'MMSN ($\sigma_{\rm solid} = 10.89(a/{\rm AU})^{-3/2}$ g/cm$^2$)')
plt.scatter(MeVeEa_a, np.log10(MeVeEa_sigmas), marker='o', s=100, color='g', label='Solar system planets (Mercury, Venus, Earth)')
ax.tick_params(axis='both', labelsize=afs)
plt.gca().set_xscale("log")
plt.xlim([0.04,1.1])
plt.ylim([-0.5,5.5])
plt.xlabel(r'Semimajor axis $a$ (AU)', fontsize=20)
plt.ylabel(r'Surface density $\log_{10}(\sigma_{\rm solid})$ (g/cm$^2$)', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.,1.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_mmen_deltaa_compare.pdf')
    plt.close()

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#





#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=# Transferred below to "plot_MMEN_physical_vs_multiplicity.py"

##### To test the RC2014 prescription for delta_a (geometric means) as a function of intrinsic multiplicity:

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
i_sample_plot = np.random.choice(np.arange(len(a_all_2p)), 1000, replace=False)
plt.scatter(a_all_2p[i_sample_plot], np.log10(sigma_all_RC2014[i_sample_plot]), marker='.', s=1, color='m')
for n in range(2,11):
    a_all_2p_n = a_all_2p[mult_all_2p == n]
    sigma_all_RC2014_n = sigma_all_RC2014[mult_all_2p == n]
    sigma_med_per_bin_RC2014_n = [np.median(sigma_all_RC2014_n[(a_all_2p_n >= a_bins[i]) & (a_all_2p_n < a_bins[i+1])]) for i in range(len(a_bins)-1)]

    mmen_fit_RC2014_n = scipy.optimize.curve_fit(f_linear, np.log10(a_all_2p_n/a0), np.log10(sigma_all_RC2014_n), [p0, p1])[0]
    sigma0_RC2014_n, beta_RC2014_n = 10.**(mmen_fit_RC2014_n[0]), mmen_fit_RC2014_n[1]
    print('n = %s:' % n + 'sigma0 = {:0.4f}, beta = {:0.4f})'.format(sigma0_RC2014_n, beta_RC2014_n))

    plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_RC2014_n), drawstyle='steps-mid', lw=2, label=r'$n = %s$' % n) #label=r'$n = %s$' % n
    #plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_RC2014_n, beta_RC2014_n, a0=a0)), lw=1, ls='--', color='k', label=r'$n = %s$' % n + r' ($\Sigma_0 = {:0.2f}$, $\beta = {:0.2f}$)'.format(sigma0_RC2014_n, beta_RC2014_n))
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

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#





#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=# Transferred below to "plot_MMEN_physical_per_system.py"

##### To fit a line to each system (RC2014 prescription):
a0 = 0.3

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
n_sys_all = []
sigma0_beta_RC2014_all = []
for i,a_sys in enumerate(sssp_per_sys['a_all'][:100000]): #100000
    M_sys = sssp_per_sys['mass_all'][i][a_sys > 0]
    a_sys = a_sys[a_sys > 0]
    if len(a_sys) > 1:
        #print(i)
        n_sys_all.append(len(a_sys))
        sigma_sys_RC2014 = solid_surface_density_system_RC2014(M_sys, a_sys)

        mmen_fit_RC2014 = scipy.optimize.curve_fit(f_linear, np.log10(a_sys/a0), np.log10(sigma_sys_RC2014), [p0, p1])[0]
        sigma0_RC2014, beta_RC2014 = 10.**(mmen_fit_RC2014[0]), mmen_fit_RC2014[1]
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
    n_sys_all = []
    sigma0_beta_RC2014_all = []
    for i,a_sys in enumerate(sssp_per_sys['a_all'][:10000]): #100000
        M_sys = sssp_per_sys['mass_all'][i][a_sys > 0]
        a_sys = a_sys[a_sys > 0]
        if len(a_sys) > 1:
            #print(i)
            n_sys_all.append(len(a_sys))
            sigma_sys_RC2014 = solid_surface_density_system_RC2014(M_sys, a_sys)

            mmen_fit_RC2014 = scipy.optimize.curve_fit(f_linear, np.log10(a_sys/a0), np.log10(sigma_sys_RC2014), [p0, p1])[0]
            sigma0_RC2014, beta_RC2014 = 10.**(mmen_fit_RC2014[0]), mmen_fit_RC2014[1]
            sigma0_beta_RC2014_all.append([sigma0_RC2014, beta_RC2014])
    n_sys_all = np.array(n_sys_all)
    sigma0_beta_RC2014_all = np.array(sigma0_beta_RC2014_all)

    x, y = sigma0_beta_RC2014_all[:,1], sigma0_beta_RC2014_all[:,0] # beta's, sigma0's
    plot_2d_points_and_contours_with_histograms(x, y, x_min=-10., x_max=5., y_min=1e-4, y_max=1e6, log_y=True, xlabel_text=r'$\beta$', ylabel_text=r'$\log_{10}(\Sigma_0/{\rm g cm^{-2}})$', extra_text=r'$\Sigma_0 = \Sigma({:0.3f} AU)$'.format(a0), plot_qtls=True, y_str_format='{:0.1f}', x_symbol=r'$\beta$', y_symbol=r'$\Sigma_0$', save_name=savefigures_directory + model_name + '_mmen_RC2014_sigma0_vs_beta_per_system_a0_{:0.3f}.pdf'.format(a0), save_fig=savefigures)
plt.show()
a0 = 1.
'''

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#





#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=# Transferred below to "plot_MMEN_physical_total_mass_per_system.py"

##### To integrate the total mass in solids for the fitted power-laws as a function of separation:

r0 = a_from_P(3., 1.) # inner truncation radius/limit of integration, in AU
r_array = np.logspace(np.log10(r0+1e-6), np.log10(1.1), 100)

sigma0_ex1, beta_ex1 = 50., -2. # example with sigma0 = 50 g/cm^2 and beta = -2 (should be linear with log(r))
Mr_array_ex1 = np.array([solid_mass_integrated_r0_to_r_given_power_law_profile(r, r0, sigma0_ex1, beta_ex1) for r in r_array])
sigma0_ex2, beta_ex2 = 50., -1.5
Mr_array_ex2 = np.array([solid_mass_integrated_r0_to_r_given_power_law_profile(r, r0, sigma0_ex2, beta_ex2) for r in r_array])

# To plot the quantiles of total mass in solids as a function of separation:
Mr_array_all = []
for i,sigma0_beta in enumerate(sigma0_beta_RC2014_all):
    sigma0, beta = sigma0_beta
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
    Mr_array = np.array([solid_mass_integrated_r0_to_r_given_power_law_profile(r, r0, sigma0_beta_RC2014_all[i,0], sigma0_beta_RC2014_all[i,1], a0=a0) for i in range(len(sigma0_beta_RC2014_all))])
    r_Mr_arrays.append(Mr_array)

plot_fig_cdf_simple((8,5), r_Mr_arrays, [], x_min=1e-2, x_max=200., log_x=True, c_sim=['k']*len(r_examples), ls_sim=r_linestyles, lw=1, labels_sim=[r'$r = {:0.1f}$ AU'.format(r) for r in r_examples], xticks_custom=[0.01, 0.1, 1, 10, 100], xlabel_text='Total solid mass within $r$, $M_r$ ($M_\oplus$)', ylabel_text=r'Cumulative fraction with $M_r$', one_minus=True, afs=afs, tfs=tfs, lfs=lfs, legend=True, save_name=savefigures_directory + model_name + '_total_mass_CDFs_per_separation_RC2014_per_system.pdf', save_fig=savefigures)

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
'''
plt.show()

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#





#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=# Transferred below to "plot_MMEN_observed.py"

#####
##### To plot the MMEN from planets in the observed catalogs only, compared to that of Kepler using the same M-R relation (re-drawn from our NWG18/Earth-like rocky composite model):
#####

# Simulated observed planets:
a_obs_per_sys = a_from_P(sss_per_sys['P_obs'], sss_per_sys['Mstar_obs'][:,None])
a_obs = a_obs_per_sys[sss_per_sys['P_obs'] > 0]
sigma_obs_CL2013 = solid_surface_density_CL2013(generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec(sss_per_sys['radii_obs']), a_obs_per_sys)[sss_per_sys['P_obs'] > 0]

a_obs_2p = np.array([])
mult_obs_2p = np.array([])
sigma_obs_RC2014 = np.array([])
start = time.time()
for i,a_sys in enumerate(a_obs_per_sys):
    M_sys = generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec(sss_per_sys['radii_obs'][i][a_sys > 0])
    a_sys = a_sys[a_sys > 0]
    if len(a_sys) > 1:
        #print(i)
        a_obs_2p = np.append(a_obs_2p, a_sys)
        mult_obs_2p = np.append(mult_obs_2p, len(a_sys)*np.ones(len(a_sys)))
        sigma_obs_RC2014 = np.append(sigma_obs_RC2014, solid_surface_density_system_RC2014(M_sys, a_sys))
stop = time.time()
print('Time: %s s' % (stop - start))

# Kepler observed planets:
a_obs_per_sys_Kep = a_from_P(ssk_per_sys['P_obs'], ssk_per_sys['Mstar_obs'][:,None])
a_obs_Kep = a_obs_per_sys_Kep[ssk_per_sys['P_obs'] > 0]
sigma_obs_CL2013_Kep = solid_surface_density_CL2013(generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec(ssk_per_sys['radii_obs']), a_obs_per_sys_Kep)[ssk_per_sys['P_obs'] > 0]

a_obs_2p_Kep = np.array([])
mult_obs_2p_Kep = np.array([])
sigma_obs_RC2014_Kep = np.array([])
start = time.time()
for i,a_sys in enumerate(a_obs_per_sys_Kep):
    M_sys = generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec(ssk_per_sys['radii_obs'][i][a_sys > 0])
    a_sys = a_sys[a_sys > 0]
    if len(a_sys) > 1:
        #print(i)
        a_obs_2p_Kep = np.append(a_obs_2p_Kep, a_sys)
        mult_obs_2p_Kep = np.append(mult_obs_2p_Kep, len(a_sys)*np.ones(len(a_sys)))
        sigma_obs_RC2014_Kep = np.append(sigma_obs_RC2014_Kep, solid_surface_density_system_RC2014(M_sys, a_sys))
stop = time.time()
print('Time: %s s' % (stop - start))

mmen_obs_fit_CL2013 = scipy.optimize.curve_fit(f_linear, np.log10(a_obs/a0), np.log10(sigma_obs_CL2013), [p0, p1])[0]
mmen_obs_fit_RC2014 = scipy.optimize.curve_fit(f_linear, np.log10(a_obs_2p/a0), np.log10(sigma_obs_RC2014), [p0, p1])[0]
mmen_obs_fit_CL2013_Kep = scipy.optimize.curve_fit(f_linear, np.log10(a_obs_Kep/a0), np.log10(sigma_obs_CL2013_Kep), [p0, p1])[0]
mmen_obs_fit_RC2014_Kep = scipy.optimize.curve_fit(f_linear, np.log10(a_obs_2p_Kep/a0), np.log10(sigma_obs_RC2014_Kep), [p0, p1])[0]
sigma0_obs_CL2013, beta_obs_CL2013 = 10.**(mmen_obs_fit_CL2013[0]), mmen_obs_fit_CL2013[1]
sigma0_obs_RC2014, beta_obs_RC2014 = 10.**(mmen_obs_fit_RC2014[0]), mmen_obs_fit_RC2014[1]
sigma0_obs_CL2013_Kep, beta_obs_CL2013_Kep = 10.**(mmen_obs_fit_CL2013_Kep[0]), mmen_obs_fit_CL2013_Kep[1]
sigma0_obs_RC2014_Kep, beta_obs_RC2014_Kep = 10.**(mmen_obs_fit_RC2014_Kep[0]), mmen_obs_fit_RC2014_Kep[1]
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



# CL2013 (delta_a = a):
fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
#corner.hist2d(a_obs, np.log10(sigma_obs), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
i_sample_plot = np.random.choice(np.arange(len(a_obs)), 10000, replace=False)
plt.scatter(a_obs[i_sample_plot], np.log10(sigma_obs_CL2013[i_sample_plot]), marker='.', s=1, color='k', label='Simulated planets')
plt.scatter(a_obs_Kep, np.log10(sigma_obs_CL2013_Kep), marker='.', s=1, color='b', label='Kepler planets')
plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_CL2013), drawstyle='steps-mid', lw=2, color='r', label='MMEN (median per bin)')
plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_CL2013_Kep), drawstyle='steps-mid', lw=2, color='m', label='MMEN (median per bin for Kepler)')
plt.plot(a_array, np.log10(sigma_MMSN), lw=2, color='g', label=r'MMSN ($\sigma_{\rm solid} = 10.89(a/{\rm AU})^{-3/2}$ g/cm$^2$)')
plt.scatter(MeVeEa_a, np.log10(MeVeEa_sigmas), marker='o', s=100, color='g', label='Solar system planets (Mercury, Venus, Earth)')
ax.tick_params(axis='both', labelsize=afs)
plt.gca().set_xscale("log")
plt.xlim([0.04,1.1])
plt.ylim([-0.5,5.5])
plt.xlabel(r'Semimajor axis $a$ (AU)', fontsize=20)
plt.ylabel(r'Surface density $\log_{10}(\sigma_{\rm solid})$ (g/cm$^2$)', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.,1.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_obs_vs_Kepler_mmen_CL2013.pdf')
    plt.close()



# RC2014 (delta_a = geometric means):
fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
#corner.hist2d(a_obs, np.log10(sigma_obs), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
i_sample_plot = np.random.choice(np.arange(len(a_obs_2p)), 5000, replace=False)
plt.scatter(a_obs_2p[i_sample_plot], np.log10(sigma_obs_RC2014[i_sample_plot]), marker='.', s=1, color='k', label='Simulated planets')
plt.scatter(a_obs_2p_Kep, np.log10(sigma_obs_RC2014_Kep), marker='.', s=1, color='b', label='Kepler planets')
plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_RC2014), drawstyle='steps-mid', lw=2, color='r', label='MMEN (median per bin)')
plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_RC2014_Kep), drawstyle='steps-mid', lw=2, color='m', label='MMEN (median per bin for Kepler)')
plt.plot(a_array, np.log10(sigma_MMSN), lw=2, color='g', label=r'MMSN ($\sigma_{\rm solid} = 10.89(a/{\rm AU})^{-3/2}$ g/cm$^2$)')
plt.scatter(MeVeEa_a, np.log10(MeVeEa_sigmas), marker='o', s=100, color='g', label='Solar system planets (Mercury, Venus, Earth)')
ax.tick_params(axis='both', labelsize=afs)
plt.gca().set_xscale("log")
plt.xlim([0.04,1.1])
plt.ylim([-0.5,5.5])
plt.xlabel(r'Semimajor axis $a$ (AU)', fontsize=20)
plt.ylabel(r'Surface density $\log_{10}(\sigma_{\rm solid})$ (g/cm$^2$)', fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.,1.), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_obs_vs_Kepler_mmen_RC2014.pdf')
    plt.close()

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#





##### To test the RC2014 prescription for delta_a (geometric means) as a function of observed multiplicity:

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

    mmen_obs_fit_RC2014_m = scipy.optimize.curve_fit(f_linear, np.log10(a_obs_2p_m/a0), np.log10(sigma_obs_RC2014_m), [p0, p1])[0]
    mmen_obs_fit_RC2014_Kep_m = scipy.optimize.curve_fit(f_linear, np.log10(a_obs_2p_Kep_m/a0), np.log10(sigma_obs_RC2014_Kep_m), [p0, p1])[0]
    sigma0_obs_RC2014_m, beta_obs_RC2014_m = 10.**(mmen_obs_fit_RC2014_m[0]), mmen_obs_fit_RC2014_m[1]
    sigma0_obs_RC2014_Kep_m, beta_obs_RC2014_Kep_m = 10.**(mmen_obs_fit_RC2014_Kep_m[0]), mmen_obs_fit_RC2014_Kep_m[1]
    print('Sim. obs. (m = %s): ' % m + 'sigma0 = {:0.4f}, beta = {:0.4f})'.format(sigma0_obs_RC2014_m, beta_obs_RC2014_m))
    print('Kep. obs. (m = %s): ' % m + 'sigma0 = {:0.4f}, beta = {:0.4f})'.format(sigma0_obs_RC2014_Kep_m, beta_obs_RC2014_Kep_m))

    plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_RC2014_m), drawstyle='steps-mid', lw=2, label=r'$m = %s$' % m)
    plt.plot(a_bins_mid, np.log10(sigma_med_per_bin_RC2014_Kep_m), drawstyle='steps-mid', ls='--', lw=2, label=r'$m = %s (Kepler)$' % m)
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
    plt.savefig(savefigures_directory + model_name + '_obs_vs_Kepler_mmen_RC2014_per_mult.pdf')
    plt.close()

plt.show()





##### To fit a line to each observed system (RC2014 prescription), for simulated and Kepler catalogs:
a0 = 0.3

# Simulated observed systems with 2+ planets:
fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
m_sys_all = []
Mstar_obs_all = []
sigma0_beta_obs_RC2014_all = []
for i,a_sys in enumerate(a_obs_per_sys):
    Mstar_sys = sss_per_sys['Mstar_obs'][i]
    M_sys = generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec(sss_per_sys['radii_obs'][i][a_sys > 0])
    a_sys = a_sys[a_sys > 0]
    if len(a_sys) > 1:
        #print(i)
        m_sys_all.append(len(a_sys))
        Mstar_obs_all.append(Mstar_sys)
        sigma_obs_sys_RC2014 = solid_surface_density_system_RC2014(M_sys, a_sys)

        mmen_obs_fit_RC2014 = scipy.optimize.curve_fit(f_linear, np.log10(a_sys/a0), np.log10(sigma_obs_sys_RC2014), [p0, p1])[0]
        sigma0_obs_RC2014, beta_obs_RC2014 = 10.**(mmen_obs_fit_RC2014[0]), mmen_obs_fit_RC2014[1]
        sigma0_beta_obs_RC2014_all.append([sigma0_obs_RC2014, beta_obs_RC2014])

        plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_obs_RC2014, beta_obs_RC2014, a0=a0)), lw=0.1, ls='-', alpha=1, color='k')
m_sys_all = np.array(m_sys_all)
Mstar_obs_all = np.array(Mstar_obs_all)
sigma0_beta_obs_RC2014_all = np.array(sigma0_beta_obs_RC2014_all)
sigma0_obs_RC2014_qtls = np.quantile(sigma0_beta_obs_RC2014_all[:,0], q=[0.16,0.5,0.84])
beta_obs_RC2014_qtls = np.quantile(sigma0_beta_obs_RC2014_all[:,1], q=[0.16,0.5,0.84])
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
    plt.savefig(savefigures_directory + model_name + '_obs_mmen_RC2014_per_system.pdf')
    plt.close()

# Kepler observed systems with 2+ planets:
fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
m_sys_Kep_all = []
Mstar_obs_Kep_all = []
sigma0_beta_obs_RC2014_Kep_all = []
for i,a_sys in enumerate(a_obs_per_sys_Kep):
    Mstar_sys = ssk_per_sys['Mstar_obs'][i]
    M_sys = generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec(ssk_per_sys['radii_obs'][i][a_sys > 0])
    a_sys = a_sys[a_sys > 0]
    if len(a_sys) > 1:
        #print(i)
        m_sys_Kep_all.append(len(a_sys))
        Mstar_obs_Kep_all.append(Mstar_sys)
        sigma_obs_sys_RC2014_Kep = solid_surface_density_system_RC2014(M_sys, a_sys)

        mmen_obs_fit_RC2014_Kep = scipy.optimize.curve_fit(f_linear, np.log10(a_sys/a0), np.log10(sigma_obs_sys_RC2014_Kep), [p0, p1])[0]
        sigma0_obs_RC2014_Kep, beta_obs_RC2014_Kep = 10.**(mmen_obs_fit_RC2014_Kep[0]), mmen_obs_fit_RC2014_Kep[1]
        sigma0_beta_obs_RC2014_Kep_all.append([sigma0_obs_RC2014_Kep, beta_obs_RC2014_Kep])

        plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_obs_RC2014_Kep, beta_obs_RC2014_Kep, a0=a0)), lw=0.1, ls='-', alpha=1, color='k')
m_sys_Kep_all = np.array(m_sys_Kep_all)
Mstar_obs_Kep_all = np.array(Mstar_obs_Kep_all)
sigma0_beta_obs_RC2014_Kep_all = np.array(sigma0_beta_obs_RC2014_Kep_all)
sigma0_obs_RC2014_Kep_qtls = np.quantile(sigma0_beta_obs_RC2014_Kep_all[:,0], q=[0.16,0.5,0.84])
beta_obs_RC2014_Kep_qtls = np.quantile(sigma0_beta_obs_RC2014_Kep_all[:,1], q=[0.16,0.5,0.84])
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
    plt.savefig(savefigures_directory + 'Kepler_mmen_RC2014_per_system.pdf')
    plt.close()

# To plot the distribution of fitted power-law parameters (sigma0 vs. beta) for the simulated observed systems:
x, y = sigma0_beta_obs_RC2014_all[:,1], sigma0_beta_obs_RC2014_all[:,0] # beta's, sigma0's
plot_2d_points_and_contours_with_histograms(x, y, x_min=-8., x_max=4., y_min=1e-2, y_max=1e8, log_y=True, xlabel_text=r'$\beta$', ylabel_text=r'$\log_{10}(\Sigma_0/{\rm g cm^{-2}})$', extra_text='Simulated observed systems', plot_qtls=True, y_str_format='{:0.1f}', x_symbol=r'$\beta$', y_symbol=r'$\Sigma_0$', save_name=savefigures_directory + model_name + '_obs_mmen_RC2014_sigma0_vs_beta_per_system.pdf', save_fig=savefigures)

# To plot the distribution of fitted power-law parameters (sigma0 vs. beta) for the Kepler observed systems:
x, y = sigma0_beta_obs_RC2014_Kep_all[:,1], sigma0_beta_obs_RC2014_Kep_all[:,0] # beta's, sigma0's
plot_2d_points_and_contours_with_histograms(x, y, x_min=-8., x_max=4., y_min=1e-2, y_max=1e8, log_y=True, xlabel_text=r'$\beta$', ylabel_text=r'$\log_{10}(\Sigma_0/{\rm g cm^{-2}})$', extra_text='Kepler observed systems', plot_qtls=True, y_str_format='{:0.1f}', x_symbol=r'$\beta$', y_symbol=r'$\Sigma_0$', save_name=savefigures_directory + 'Kepler_mmen_RC2014_sigma0_vs_beta_per_system.pdf', save_fig=savefigures)

# To repeat the above (plot sigma0 vs beta) per observed multiplicity in the simulated systems: #NOTE: not enough systems to repeat these 2d contours for the higher multiplicity Kepler systems
for m in range(2,5):
    print(np.sum(m_sys_all == m))
    x = sigma0_beta_obs_RC2014_all[:,1][m_sys_all == m] # beta's
    y = sigma0_beta_obs_RC2014_all[:,0][m_sys_all == m] # sigma0's
    plot_2d_points_and_contours_with_histograms(x, y, x_min=-8., x_max=4., y_min=1e-2, y_max=1e8, log_y=True, xlabel_text=r'$\beta$', ylabel_text=r'$\log_{10}(\Sigma_0/{\rm g cm^{-2}})$', extra_text=r'$m = %s$' % m, plot_qtls=True, y_str_format='{:0.1f}', x_symbol=r'$\beta$', y_symbol=r'$\Sigma_0$', save_name=savefigures_directory + model_name + '_obs_mmen_RC2014_sigma0_vs_beta_per_system_m%s.pdf' % m, save_fig=savefigures)

plt.show()



##### To test how the fitted observed MMEN may correlate with stellar mass:

# To plot sigma0 vs. stellar mass for the simulated observed systems:
x, y = Mstar_obs_all, sigma0_beta_obs_RC2014_all[:,0] # stellar masses, sigma0's
plot_2d_points_and_contours_with_histograms(x, y, x_min=0.5, x_max=1.5, y_min=1e-2, y_max=1e8, log_y=True, xlabel_text=r'$M_\star$ ($M_\odot$)', ylabel_text=r'$\log_{10}(\Sigma_0/{\rm g cm^{-2}})$', extra_text='Simulated observed systems', plot_qtls=True, y_str_format='{:0.1f}', x_symbol=r'$M_\star$', y_symbol=r'$\Sigma_0$', save_name=savefigures_directory + model_name + '_obs_mmen_RC2014_sigma0_vs_starmass_per_system.pdf', save_fig=savefigures)

# To plot sigma0 vs. stellar mass for the Kepler observed systems:
x, y = Mstar_obs_Kep_all, sigma0_beta_obs_RC2014_Kep_all[:,0] # stellar masses, sigma0's
plot_2d_points_and_contours_with_histograms(x, y, x_min=0.5, x_max=1.5, y_min=1e-2, y_max=1e8, log_y=True, xlabel_text=r'$M_\star$ ($M_\odot$)', ylabel_text=r'$\log_{10}(\Sigma_0/{\rm g cm^{-2}})$', extra_text='Kepler observed systems', plot_qtls=True, y_str_format='{:0.1f}', x_symbol=r'$M_\star$', y_symbol=r'$\Sigma_0$', save_name=savefigures_directory + 'Kepler_mmen_RC2014_sigma0_vs_starmass_per_system.pdf', save_fig=savefigures)

# To plot beta vs. stellar mass for the simulated observed systems:
x, y = Mstar_obs_all, sigma0_beta_obs_RC2014_all[:,1] # stellar masses, beta's
plot_2d_points_and_contours_with_histograms(x, y, x_min=0.5, x_max=1.5, y_min=-8, y_max=4, xlabel_text=r'$M_\star$ ($M_\odot$)', ylabel_text=r'$\beta$', extra_text='Simulated observed systems', plot_qtls=True, x_symbol=r'$M_\star$', y_symbol=r'$\beta$', save_name=savefigures_directory + model_name + '_obs_mmen_RC2014_beta_vs_starmass_per_system.pdf', save_fig=savefigures)

# To plot beta vs. stellar mass for the Kepler observed systems:
x, y = Mstar_obs_Kep_all, sigma0_beta_obs_RC2014_Kep_all[:,1] # stellar masses, beta's
plot_2d_points_and_contours_with_histograms(x, y, x_min=0.5, x_max=1.5, y_min=-8, y_max=4, xlabel_text=r'$M_\star$ ($M_\odot$)', ylabel_text=r'$\beta$', extra_text='Kepler observed systems', plot_qtls=True, x_symbol=r'$M_\star$', y_symbol=r'$\beta$', save_name=savefigures_directory + 'Kepler_mmen_RC2014_beta_vs_starmass_per_system.pdf', save_fig=savefigures)

plt.show()





##### To compare the power-law fits of the observed systems to the corresponding true, physical systems:
##### NOTE: in the analyses above for the simulated observed catalogs, the planet masses are 're-drawn' using our M-R relationship (so they are treated in the same way as for the Kepler catalog). However, for the following comparisons, we will use the 'true' planet masses for the simulated observed planets so their masses are consistent in both MMEN calculations of the physical systems and the observed systems.
a0 = 0.3

n_m_sys_all = []
sigma0_beta_true_obs_RC2014_all = []
for i,det_sys in enumerate(sssp_per_sys['det_all']):
    if np.sum(det_sys) > 1:
        #print(i)
        a_sys = sssp_per_sys['a_all'][i] # all semimajor axes including padded zeros
        M_sys = sssp_per_sys['mass_all'][i] # all planet masses including padded zeros

        a_sys_obs = a_sys[det_sys == 1] # semimajor axes of observed planets
        M_sys_obs = M_sys[det_sys == 1] # masses of observed planets
        M_sys = M_sys[a_sys > 0] # masses of all planets
        a_sys = a_sys[a_sys > 0] # semimajor axes of all planets

        n_m_sys_all.append([len(a_sys), len(a_sys_obs)])
        sigma_sys_RC2014 = solid_surface_density_system_RC2014(M_sys, a_sys)
        sigma_sys_obs_RC2014 = solid_surface_density_system_RC2014(M_sys_obs, a_sys_obs)

        mmen_fit_RC2014 = scipy.optimize.curve_fit(f_linear, np.log10(a_sys/a0), np.log10(sigma_sys_RC2014), [p0, p1])[0]
        mmen_obs_fit_RC2014 = scipy.optimize.curve_fit(f_linear, np.log10(a_sys_obs/a0), np.log10(sigma_sys_obs_RC2014), [p0, p1])[0]
        sigma0_RC2014, beta_RC2014 = 10.**(mmen_fit_RC2014[0]), mmen_fit_RC2014[1]
        sigma0_obs_RC2014, beta_obs_RC2014 = 10.**(mmen_obs_fit_RC2014[0]), mmen_obs_fit_RC2014[1]
        sigma0_beta_true_obs_RC2014_all.append([sigma0_RC2014, sigma0_obs_RC2014, beta_RC2014, beta_obs_RC2014])
n_m_sys_all = np.array(n_m_sys_all)
sigma0_beta_true_obs_RC2014_all = np.array(sigma0_beta_true_obs_RC2014_all)
sigma0_true_RC2014_qtls = np.quantile(sigma0_beta_true_obs_RC2014_all[:,0], q=[0.16,0.5,0.84])
sigma0_obs_RC2014_qtls = np.quantile(sigma0_beta_true_obs_RC2014_all[:,1], q=[0.16,0.5,0.84])
beta_true_RC2014_qtls = np.quantile(sigma0_beta_true_obs_RC2014_all[:,2], q=[0.16,0.5,0.84])
beta_obs_RC2014_qtls = np.quantile(sigma0_beta_true_obs_RC2014_all[:,3], q=[0.16,0.5,0.84])

# To plot sigma0_obs vs sigma0_true for the simulated observed systems:
x, y = sigma0_beta_true_obs_RC2014_all[:,0], sigma0_beta_true_obs_RC2014_all[:,1] # sigma0_true's, sigma0_obs's
plot_2d_points_and_contours_with_histograms(x, y, x_min=1e-2, x_max=1e7, y_min=1e-2, y_max=1e7, log_x=True, log_y=True, xlabel_text=r'$\log_{10}(\Sigma_{0,\rm true}/{\rm g cm^{-2}})$', ylabel_text=r'$\log_{10}(\Sigma_{0,\rm obs}/{\rm g cm^{-2}})$', extra_text='Simulated observed systems', plot_qtls=True, x_str_format='{:0.1f}', y_str_format='{:0.1f}', x_symbol=r'$\Sigma_{0,\rm true}$', y_symbol=r'$\Sigma_{0,\rm obs}$', save_name=savefigures_directory + model_name + '_true_vs_obs_mmen_RC2014_sigma0_per_system.pdf', save_fig=savefigures)

# To plot beta_obs vs beta_true for the simulated observed systems:
x, y = sigma0_beta_true_obs_RC2014_all[:,2], sigma0_beta_true_obs_RC2014_all[:,3] # beta_true's, beta_obs's
plot_2d_points_and_contours_with_histograms(x, y, x_min=-8., x_max=4., y_min=-8., y_max=4., xlabel_text=r'$\beta_{\rm true}$', ylabel_text=r'$\beta_{\rm obs}$', extra_text='Simulated observed systems', plot_qtls=True, x_symbol=r'$\beta_{\rm true}$', y_symbol=r'$\beta_{\rm obs}$', save_name=savefigures_directory + model_name + '_true_vs_obs_mmen_RC2014_beta_per_system.pdf', save_fig=savefigures)

plt.show()
