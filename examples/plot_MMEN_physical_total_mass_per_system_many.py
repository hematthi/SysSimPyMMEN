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
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_best_models/MMEN/RC2014/' #cap_core_mass_10Mearth_and_scale_up_sigma0/
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
#prescription_str = 'RC2014' #'RC2014'
a0 = 0.3 # normalization separation for fitting power-laws





##### To load and compute the same statistics for a large number of models:

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
runs = 100

prescriptions = ['CL2013', 'S2014', 'nHill', 'RC2014']
fit_per_sys_dict_per_cat = {pres:[] for pres in prescriptions} # dictionary of lists of dictionaries
for i in range(1,runs+1):
    run_number = i
    print('##### %s' % i)
    sssp_per_sys_i, sssp_i = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)

    for pres in prescriptions:
        fit_per_sys_dict_i = fit_power_law_MMEN_per_system_physical(sssp_per_sys_i, sssp_i, prescription=pres, a0=a0, scale_up=True, N_sys=len(sssp_i['Mstar_all']))
        fit_per_sys_dict_per_cat[pres].append(fit_per_sys_dict_i)





##### To integrate the total mass in solids for the fitted power-laws as a function of separation:

pres_use = 'RC2014'

fit_per_sys_dict = fit_power_law_MMEN_per_system_physical(sssp_per_sys, sssp, prescription=pres_use, a0=a0, scale_up=True, N_sys=N_sim)

r0 = a_from_P(3., 1.) # inner truncation radius/limit of integration, in AU

# To plot the distributions (CDFs) of total masses in solids for several separations:
r_examples = [0.1, 0.2, 0.5, 1.]
r_linestyles = [':', '-.', '--', '-']
Mr_array_eval = np.logspace(-3., 3., 1000) # array of masses to evaluate CDFs

#'''
plot_fig_cdf_simple((8,5), [], [], x_min=1e-2, x_max=500., log_x=True, xticks_custom=[0.01, 0.1, 1, 10, 100], xlabel_text='Total mass, $M$ ($M_\oplus$)', ylabel_text=r'Fraction with $M_r \geq M$', one_minus=True, afs=afs, tfs=tfs, lfs=lfs)
start = time.time()
for j,r in enumerate(r_examples):
    cdf_Mr_eval_all = []
    for fit_per_sys_dict_i in fit_per_sys_dict_per_cat[pres_use]:
        Mr_array = np.array([solid_mass_integrated_r0_to_r_given_power_law_profile(r, r0, fit_per_sys_dict_i['sigma0'][i], fit_per_sys_dict_i['beta'][i], a0=a0) for i in range(len(fit_per_sys_dict_i['sigma0']))])
        cdf_Mr_eval = [np.sum(Mr_array < Mr)/len(Mr_array) for Mr in Mr_array_eval]
        cdf_Mr_eval_all.append(cdf_Mr_eval)
    cdf_Mr_eval_all = np.array(cdf_Mr_eval_all)
    cdf_Mr_eval_qtls = np.quantile(cdf_Mr_eval_all, q=[0.16,0.5,0.84], axis=0)
    plt.plot(Mr_array_eval, 1.-cdf_Mr_eval_qtls[1], color='k', ls=r_linestyles[j], lw=2, label=r'$r = {:0.1f}$ AU'.format(r))
    plt.fill_between(Mr_array_eval, 1.-cdf_Mr_eval_qtls[0], 1.-cdf_Mr_eval_qtls[2], color='k', ec='none', alpha=0.2)
    print(r)
plt.legend(loc='lower left', bbox_to_anchor=(0,0), ncol=1, frameon=False, fontsize=lfs)
end = time.time()
print('Time elapsed: %s s' % (end-start))
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_total_mass_CDFs_per_separation_%s_per_system_credible.pdf' % pres_use)
    plt.close()
plt.show()
#'''
