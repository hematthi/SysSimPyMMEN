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

dists, dists_w = compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)





# Plotting parameters:
n_bins = 50
lw = 2 # linewidth

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size

# Parameters for defining the MMEN:
a0 = 0.3 # normalization separation for fitting power-laws

n_mult_min, n_mult_max = 2, 10
max_core_mass = 10.
prescription = 'RC2014'
n = 10.
p0, p1 = 1., -1.5
scale_up = True
y_sym_star = '*' if scale_up else ''





##### To plot an illustrative system depicting the ways in which the inferred MMEN can be distorted by undetected planets:
det_mult_all = np.sum(sssp_per_sys['det_all'], axis=1) # number of detected planets in each system
i_det_mult_in_range_all = np.arange(N_sim)[(n_mult_min <= det_mult_all) & (det_mult_all <= n_mult_max)]

i = 5677 #i_det_mult_in_range_all[16] = 5677
print('##### System %s:' % i)

det_sys = sssp_per_sys['det_all'][i]

Mstar = sssp['Mstar_all'][i]
Mp_sys = sssp_per_sys['mass_all'][i]
core_mass_sys = np.copy(Mp_sys) # all planet masses including padded zeros
core_mass_sys[core_mass_sys > max_core_mass] = max_core_mass
R_sys = sssp_per_sys['radii_all'][i] # all planet radii including padded zeros
a_sys = sssp_per_sys['a_all'][i] # all semimajor axes including padded zeros

core_mass_sys_obs = np.copy(Mp_sys)[det_sys == 1] # masses of observed planets
core_mass_sys_obs[core_mass_sys_obs > max_core_mass] = max_core_mass
R_sys_obs = R_sys[det_sys == 1] # radii of observed planets
a_sys_obs = a_sys[det_sys == 1] # semimajor axes of observed planets
core_mass_sys = core_mass_sys[a_sys > 0] # masses of all planets
R_sys = R_sys[a_sys > 0] # radii of all planets
a_sys = a_sys[a_sys > 0] # semimajor axes of all planets

sigma_sys = solid_surface_density_prescription(core_mass_sys, R_sys, a_sys, Mstar=Mstar, n=n, prescription=prescription) # using all planets
sigma_sys_obs = solid_surface_density_prescription(core_mass_sys_obs, R_sys_obs, a_sys_obs, Mstar=Mstar, n=n, prescription=prescription) # using observed planets only
sigma0, beta = fit_power_law_MMEN(a_sys, sigma_sys, a0=a0, p0=p0, p1=p1)
sigma0_obs, beta_obs = fit_power_law_MMEN(a_sys_obs, sigma_sys_obs, a0=a0, p0=p0, p1=p1)

sigma_fit_sys = MMEN_power_law(a_sys, sigma0, beta, a0=a0)
sigma_fit_sys_obs = MMEN_power_law(a_sys_obs, sigma0_obs, beta_obs, a0=a0)
sigma_fit_ratio_sys = sigma_sys/sigma_fit_sys
sigma_fit_ratio_sys_obs = sigma_sys_obs/sigma_fit_sys_obs
scale_factor_sigma0 = np.max(sigma_fit_ratio_sys)
scale_factor_sigma0_obs = np.max(sigma_fit_ratio_sys_obs)
sigma0 = sigma0*scale_factor_sigma0 if scale_up else sigma0
sigma0_obs = sigma0_obs*scale_factor_sigma0_obs if scale_up else sigma0_obs

# Plot the system:
a_array = np.linspace(1e-3,2,1001)
sigma_MMSN = MMSN(a_array)

delta_a_sys_RC2014, a_bounds_sys = feeding_zone_RC2014(a_sys)

fig = plt.figure(figsize=(16,6))
plot = GridSpec(1,1,left=0.075,bottom=0.15,right=0.975,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
plt.scatter(a_sys, np.log10(sigma_sys), marker='o', s=100.*R_sys**2., facecolors='none', edgecolors='k', label='All planets')
plt.scatter(a_sys_obs, np.log10(sigma_sys_obs), marker='o', s=100.*R_sys_obs**2., color='k', label='Observed planets')
for j,a in enumerate(a_sys):
    # Plot various feeding zones for each planet:
    #plt.plot([a_bounds_sys[j], a_bounds_sys[j+1]], [np.log10(sigma_sys[j])]*2, lw=2, color='k') # lines for feeding zones
    #plt.axvspan(a_bounds_sys[j], a_bounds_sys[j+1], alpha=0.2, ec=None, fc='k' if j%2==0 else 'b')
    plt.axvspan(a_bounds_sys[j], a_bounds_sys[j+1], alpha=0.2 if j%2==0 else 0.1, ec=None, fc='k')

    # To compare the planet masses to the integrated disk masses:
    Mp_core = core_mass_sys[j]
    text_ypos = np.log10(2*sigma_sys[j]) # + R_sys[j]/10.
    if det_sys[j]==1:
        sigma_sys_obs_j = sigma_sys_obs[np.where(np.isclose(a_sys_obs, a_sys[j]))[0][0]]
        text_ypos = np.log10(2*max(sigma_sys[j], sigma_sys_obs_j)) # + R_sys[j]/50.
    plt.annotate(r'${:0.1f} M_\oplus$'.format(Mp_core), (a, text_ypos), ha='center', fontsize=16)
    if prescription == 'RC2014':
        Mp_intdisk = solid_mass_integrated_r0_to_r_given_power_law_profile(a_bounds_sys[j+1], a_bounds_sys[j], sigma0, beta, a0=a0)
        print('Planet (core) mass: {:0.2f} M_earth --- Integrated disk mass (RC2014): {:0.2f} M_earth'.format(Mp_core, Mp_intdisk))

plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0, beta, a0=a0)), lw=3, ls='-', color='b', label=r'Fit to all planets ($\Sigma_0^%s = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0, beta) % y_sym_star)
plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_obs, beta_obs, a0=a0)), lw=3, ls='--', color='b', label=r'Fit to observed planets ($\Sigma_0^%s = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0_obs, beta_obs) % y_sym_star)
plt.plot(a_array, np.log10(sigma_MMSN), lw=3, color='g', label=r'MMSN') #label=r'MMSN ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(MMSN(a0), -1.5)
ax.tick_params(axis='both', labelsize=20)
plt.gca().set_xscale("log")
plt.xticks([0.05, 0.1, 0.2, 0.4, 0.8])
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
plt.xlim([0.04,0.9])
plt.ylim([-0.5,5.5])
plt.xlabel(r'Semimajor axis, $a$ (AU)', fontsize=20)
plt.ylabel(r'Surface density, $\log_{10}(\Sigma/{\rm g\,cm}^{-2})$', fontsize=20)
plt.legend(loc='lower left', bbox_to_anchor=(0.,0.), ncol=1, frameon=False, fontsize=16)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_example_fit_mmen_%s_labeled_system_%s.pdf' % (prescription, i))
    plt.close()
plt.show()



##### To make a multi-panel gallery of four illustrative, example systems:
i_beta_shallower_sigma_unaffected = 3637 # example where beta gets shallower due to missed innermost and outermost planets, but sigma0 is mildly affected
i_unaffected = 58654 # example where both sigma0 and beta are not strongly biased despite missing planets (outer planets, or small planets in between); 4308, 9084, 41572, 58654, 160816, 202823
i_beta_biased_flat = 83828 # example where beta is extremely biased to almost flat value due to missed outer small planets (sigma0 also affected); 5029, 13036, 20137, 23072, 24265, 83828, 123077
i_beta_biased_flat_miss_mid_out = 8274 # example where beta is strongly biased to almost a flat value due to missed middle and outer planets
i_beta_pos_extreme = 9037 # example of an extremely unphysical fit to just two observed planets; 9037, 12807, 14516, 17549, 117629
i_beta_biased_steep = 44302 # example where beta is strongly biased to a steeper value due to missed outer planets; also: 12691, 13536, 24443, 44302, 45286
i_beta_biased_steep_miss_one = 13812 # example where beta is biased to a steeper value due to just one missed outer planet; 13812, 26836
i_sigma0_biased_high_miss_one = 16059 # example where sigma0 is biased HIGHER due to missed outer planet (which affects feeding zone of middle planet; beta also biased)
i_weird = 130403 # example where physical system has a positive beta (due to two small inner planets), but those planets are missed so beta is biased to a steep negative value

#i_plot = [3637, 4308, 5029, 8274]
#i_plot = [23072, 24265, 83828, 123077]
#i_plot = [41572, 58654, 160816, 202823]
i_plot = [i_beta_biased_flat, i_beta_pos_extreme, i_beta_biased_steep, i_unaffected]
panel_labels = ['A', 'B', 'C', 'D']

fig = plt.figure(figsize=(16,6))
plot = GridSpec(2,2,left=0.075,bottom=0.15,right=0.975,top=0.95,wspace=0.05,hspace=0.1)
for idx_panel,i in enumerate(i_plot):
    print('##### System %s:' % i)

    det_sys = sssp_per_sys['det_all'][i]

    Mstar = sssp['Mstar_all'][i]
    Mp_sys = sssp_per_sys['mass_all'][i]
    core_mass_sys = np.copy(Mp_sys) # all planet masses including padded zeros
    core_mass_sys[core_mass_sys > max_core_mass] = max_core_mass
    R_sys = sssp_per_sys['radii_all'][i] # all planet radii including padded zeros
    a_sys = sssp_per_sys['a_all'][i] # all semimajor axes including padded zeros

    core_mass_sys_obs = np.copy(Mp_sys)[det_sys == 1] # masses of observed planets
    core_mass_sys_obs[core_mass_sys_obs > max_core_mass] = max_core_mass
    R_sys_obs = R_sys[det_sys == 1] # radii of observed planets
    a_sys_obs = a_sys[det_sys == 1] # semimajor axes of observed planets
    core_mass_sys = core_mass_sys[a_sys > 0] # masses of all planets
    R_sys = R_sys[a_sys > 0] # radii of all planets
    a_sys = a_sys[a_sys > 0] # semimajor axes of all planets

    sigma_sys = solid_surface_density_prescription(core_mass_sys, R_sys, a_sys, Mstar=Mstar, n=n, prescription=prescription) # using all planets
    sigma_sys_obs = solid_surface_density_prescription(core_mass_sys_obs, R_sys_obs, a_sys_obs, Mstar=Mstar, n=n, prescription=prescription) # using observed planets only
    sigma0, beta = fit_power_law_MMEN(a_sys, sigma_sys, a0=a0, p0=p0, p1=p1)
    sigma0_obs, beta_obs = fit_power_law_MMEN(a_sys_obs, sigma_sys_obs, a0=a0, p0=p0, p1=p1)

    sigma_fit_sys = MMEN_power_law(a_sys, sigma0, beta, a0=a0)
    sigma_fit_sys_obs = MMEN_power_law(a_sys_obs, sigma0_obs, beta_obs, a0=a0)
    sigma_fit_ratio_sys = sigma_sys/sigma_fit_sys
    sigma_fit_ratio_sys_obs = sigma_sys_obs/sigma_fit_sys_obs
    scale_factor_sigma0 = np.max(sigma_fit_ratio_sys)
    scale_factor_sigma0_obs = np.max(sigma_fit_ratio_sys_obs)
    sigma0 = sigma0*scale_factor_sigma0 if scale_up else sigma0
    sigma0_obs = sigma0_obs*scale_factor_sigma0_obs if scale_up else sigma0_obs

    delta_a_sys_RC2014, a_bounds_sys = feeding_zone_RC2014(a_sys)

    # Plot the system:
    row, col = int(idx_panel/2), idx_panel%2
    ax = plt.subplot(plot[row,col])
    plt.text(0.97, 0.9, s=panel_labels[idx_panel], ha='right', va='top', fontsize=20, transform=ax.transAxes)
    plt.scatter(a_sys, np.log10(sigma_sys), marker='o', s=100.*R_sys**2., facecolors='none', edgecolors='k', label='All planets')
    plt.scatter(a_sys_obs, np.log10(sigma_sys_obs), marker='o', s=100.*R_sys_obs**2., color='k', label='Observed planets')
    for j,a in enumerate(a_sys):
        # Plot various feeding zones for each planet:
        #plt.plot([a_bounds_sys[j], a_bounds_sys[j+1]], [np.log10(sigma_sys[j])]*2, lw=2, color='k') # lines for feeding zones
        #plt.axvspan(a_bounds_sys[j], a_bounds_sys[j+1], alpha=0.2, ec=None, fc='k' if j%2==0 else 'b')
        plt.axvspan(a_bounds_sys[j], a_bounds_sys[j+1], alpha=0.2 if j%2==0 else 0.1, ec=None, fc='k')

        # To compare the planet masses to the integrated disk masses:
        Mp_core = core_mass_sys[j]
        text_ypos = np.log10(2*sigma_sys[j])+R_sys[j]/10.
        if det_sys[j]==1:
            sigma_sys_obs_j = sigma_sys_obs[np.where(np.isclose(a_sys_obs, a_sys[j]))[0][0]]
            text_ypos = np.log10(2*max(sigma_sys[j], sigma_sys_obs_j))+R_sys[j]/10.
        plt.annotate(r'${:0.1f} M_\oplus$'.format(Mp_core), (a, text_ypos), ha='center', va='center', fontsize=14)
        if prescription == 'RC2014':
            Mp_intdisk = solid_mass_integrated_r0_to_r_given_power_law_profile(a_bounds_sys[j+1], a_bounds_sys[j], sigma0, beta, a0=a0)
            print('Planet (core) mass: {:0.2f} M_earth --- Integrated disk mass (RC2014): {:0.2f} M_earth'.format(Mp_core, Mp_intdisk))

    plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0, beta, a0=a0)), lw=3, ls='-', color='b', label=r'$\Sigma_0^%s = {:0.0f}$, $\beta = {:0.2f}$'.format(sigma0, beta) % y_sym_star)
    plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_obs, beta_obs, a0=a0)), lw=3, ls='--', color='b', label=r'$\Sigma_0^%s = {:0.0f}$, $\beta = {:0.2f}$'.format(sigma0_obs, beta_obs) % y_sym_star)
    #plt.plot(a_array, np.log10(sigma_MMSN), lw=3, color='g', label=r'MMSN') #label=r'MMSN ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(MMSN(a0), -1.5)
    ax.tick_params(axis='both', labelsize=20)
    plt.gca().set_xscale("log")
    if row==1:
        plt.xticks([0.05, 0.1, 0.2, 0.4, 0.8])
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    else:
        plt.xticks([])
    if col==1:
        plt.yticks([])
    plt.xlim([0.04,0.9])
    plt.ylim([-0.5,5.5])
    if row==1:
        plt.xlabel(r'$a$ (AU)', fontsize=20)
    if col==0:
        plt.ylabel(r'$\log_{10}(\Sigma/{\rm g\,cm}^{-2})$', fontsize=20)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], loc='lower left', bbox_to_anchor=(0.,0.), ncol=1, frameon=False, fontsize=16)

if savefigures:
    plt.savefig(savefigures_directory + model_name + '_example_fit_mmen_%s_systems_%s.pdf' % (prescription, '_'.join(str(i) for i in i_plot)))
    plt.close()
plt.show()



##### To loop through and plot more systems:
#plot_feeding_zones_and_power_law_fit_MMEN_per_system_observed_and_physical(sssp_per_sys, sssp, n_mult_min=4, prescription=prescription, a0=a0, scale_up=scale_up, N_sys=100)
