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

# To first read the number of simulated targets and bounds for the periods and radii:
N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

# To load and process the observed Kepler catalog:
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios_adjacent)





# Plotting parameters:
n_bins = 50
lw = 2 # linewidth

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size

# Parameters for defining the MMEN:
a0 = 0.3 # normalization separation for fitting power-laws





##### To compute uncertainties in the MMEN power-law fits to the Kepler catalog by repeatedly resampling the same M-R relation (draws from our NWG18/Earth-like rocky composite model):

N_cats = 1000

sigma0_obs_CL13_Kep_all = []
sigma0_obs_RC14_Kep_all = []
sigma0_obs_10Hill_Kep_all = []
sigma0_obs_S14_Kep_all = []

beta_obs_CL13_Kep_all = []
beta_obs_RC14_Kep_all = []
beta_obs_10Hill_Kep_all = []
beta_obs_S14_Kep_all = []

start = time.time()
for i in range(N_cats):
    #print(i)
    sigma_obs_CL2013_Kep, mass_obs_Kep, a_obs_Kep = solid_surface_density_CL2013_given_observed_catalog(ssk_per_sys)
    sigma0_obs_CL2013_Kep, beta_obs_CL2013_Kep = fit_power_law_MMEN(a_obs_Kep, sigma_obs_CL2013_Kep, a0=a0)
    sigma0_obs_CL13_Kep_all.append(sigma0_obs_CL2013_Kep)
    beta_obs_CL13_Kep_all.append(beta_obs_CL2013_Kep)

    sigma_obs_RC2014_Kep, mass_obs_2p_Kep, a_obs_2p_Kep, mult_obs_2p_Kep = solid_surface_density_RC2014_given_observed_catalog(ssk_per_sys)
    sigma0_obs_RC2014_Kep, beta_obs_RC2014_Kep = fit_power_law_MMEN(a_obs_2p_Kep, sigma_obs_RC2014_Kep, a0=a0)
    sigma0_obs_RC14_Kep_all.append(sigma0_obs_RC2014_Kep)
    beta_obs_RC14_Kep_all.append(beta_obs_RC2014_Kep)

    sigma_obs_10Hill_Kep, mass_obs_Kep, a_obs_Kep = solid_surface_density_nHill_given_observed_catalog(ssk_per_sys)
    sigma0_obs_10Hill_Kep, beta_obs_10Hill_Kep = fit_power_law_MMEN(a_obs_Kep, sigma_obs_10Hill_Kep, a0=a0)
    sigma0_obs_10Hill_Kep_all.append(sigma0_obs_10Hill_Kep)
    beta_obs_10Hill_Kep_all.append(beta_obs_10Hill_Kep)

    sigma_obs_S2014_Kep, mass_obs_Kep, a_obs_Kep = solid_surface_density_S2014_given_observed_catalog(ssk_per_sys)
    sigma0_obs_S2014_Kep, beta_obs_S2014_Kep = fit_power_law_MMEN(a_obs_Kep, sigma_obs_S2014_Kep, a0=a0)
    sigma0_obs_S14_Kep_all.append(sigma0_obs_S2014_Kep)
    beta_obs_S14_Kep_all.append(beta_obs_S2014_Kep)
end = time.time()
print('Time elapsed: %s s' % (end-start))

sigma0_obs_CL13_Kep_all = np.array(sigma0_obs_CL13_Kep_all)
sigma0_obs_RC14_Kep_all = np.array(sigma0_obs_RC14_Kep_all)
sigma0_obs_10Hill_Kep_all = np.array(sigma0_obs_10Hill_Kep_all)
sigma0_obs_S14_Kep_all = np.array(sigma0_obs_S14_Kep_all)

beta_obs_CL13_Kep_all = np.array(beta_obs_CL13_Kep_all)
beta_obs_RC14_Kep_all = np.array(beta_obs_RC14_Kep_all)
beta_obs_10Hill_Kep_all = np.array(beta_obs_10Hill_Kep_all)
beta_obs_S14_Kep_all = np.array(beta_obs_S14_Kep_all)

qtls = [0.16,0.5,0.84]
sigma0_qtls_CL13 = np.quantile(sigma0_obs_CL13_Kep_all, qtls)
beta_qtls_CL13 = np.quantile(beta_obs_CL13_Kep_all, qtls)
print('# CL13:')
print(r'$\Sigma_0$: ${:0.0f}_{{-{:0.0f} }}^{{+{:0.0f} }}$'.format(sigma0_qtls_CL13[1], sigma0_qtls_CL13[1]-sigma0_qtls_CL13[0], sigma0_qtls_CL13[2]-sigma0_qtls_CL13[1]))
print(r'$\beta$: ${:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(beta_qtls_CL13[1], beta_qtls_CL13[1]-beta_qtls_CL13[0], beta_qtls_CL13[2]-beta_qtls_CL13[1]))

sigma0_qtls_RC14 = np.quantile(sigma0_obs_RC14_Kep_all, qtls)
beta_qtls_RC14 = np.quantile(beta_obs_RC14_Kep_all, qtls)
print('# RC14:')
print(r'$\Sigma_0$: ${:0.0f}_{{-{:0.0f} }}^{{+{:0.0f} }}$'.format(sigma0_qtls_RC14[1], sigma0_qtls_RC14[1]-sigma0_qtls_RC14[0], sigma0_qtls_RC14[2]-sigma0_qtls_RC14[1]))
print(r'$\beta$: ${:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(beta_qtls_RC14[1], beta_qtls_RC14[1]-beta_qtls_RC14[0], beta_qtls_RC14[2]-beta_qtls_RC14[1]))

sigma0_qtls_10Hill = np.quantile(sigma0_obs_10Hill_Kep_all, qtls)
beta_qtls_10Hill = np.quantile(beta_obs_10Hill_Kep_all, qtls)
print('# 10 Hill:')
print(r'$\Sigma_0$: ${:0.0f}_{{-{:0.0f} }}^{{+{:0.0f} }}$'.format(sigma0_qtls_10Hill[1], sigma0_qtls_10Hill[1]-sigma0_qtls_10Hill[0], sigma0_qtls_10Hill[2]-sigma0_qtls_10Hill[1]))
print(r'$\beta$: ${:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(beta_qtls_10Hill[1], beta_qtls_10Hill[1]-beta_qtls_10Hill[0], beta_qtls_10Hill[2]-beta_qtls_10Hill[1]))

sigma0_qtls_S14 = np.quantile(sigma0_obs_S14_Kep_all, qtls)
beta_qtls_S14 = np.quantile(beta_obs_S14_Kep_all, qtls)
print('# S14:')
print(r'$\Sigma_0$: ${:0.0f}_{{-{:0.0f} }}^{{+{:0.0f} }}$'.format(sigma0_qtls_S14[1], sigma0_qtls_S14[1]-sigma0_qtls_S14[0], sigma0_qtls_S14[2]-sigma0_qtls_S14[1]))
print(r'$\beta$: ${:0.2f}_{{-{:0.2f} }}^{{+{:0.2f} }}$'.format(beta_qtls_S14[1], beta_qtls_S14[1]-beta_qtls_S14[0], beta_qtls_S14[2]-beta_qtls_S14[1]))
