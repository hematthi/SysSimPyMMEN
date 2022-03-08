# To import required modules:
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),'SysSim_Plotting')) # TODO: update when those files get made into a package

from src.functions_general import *
from src.functions_compare_kepler import *
from src.functions_load_sims import *

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'src'))

from MMEN_functions import *





# Test functions in "src/MMEN_functions.py":

def test_mass_given_radius_density():
    assert np.isclose(mass_given_radius_density(1., 5.51), 1., atol=1e-3) # Earth mass/radius
    assert mass_given_radius_density(0., 1.) == 0
    assert mass_given_radius_density(1., 0.) == 0

def test_density_given_mass_radius():
    assert np.isclose(density_given_mass_radius(1., 1.), 5.513259, atol=1e-5) # Earth mean density

def test_generate_planet_mass_from_radius_Ning2018_table(seed=42):
    np.random.seed(seed)
    logR_min, logR_max = -1., 2. # log10(Earth radii)
    R_logunif = 10.**(logR_min + np.random.rand(10000)*(logR_max - logR_min)) # log-uniform
    M_array = np.array([generate_planet_mass_from_radius_Ning2018_table(R) for R in R_logunif])
    assert np.all(0. <= M_array)
    assert gen.Pearson_correlation_coefficient(R_logunif, M_array) > 0.

def test_σ_logM_linear_given_radius():
    assert np.isclose(σ_logM_linear_given_radius(0.5), σ_logM_at_radius_min)
    assert np.isclose(σ_logM_linear_given_radius(radius_switch), σ_logM_at_radius_switch)

def test_generate_planet_mass_from_radius_lognormal_mass_around_earthlike_rocky(seed=42):
    np.random.seed(seed)
    logR_min, logR_max = np.log10(0.5), np.log10(1.5) # log10(Earth radii)
    R_logunif = 10.**(logR_min + np.random.rand(10000)*(logR_max - logR_min)) # log-uniform
    M_array = np.array([generate_planet_mass_from_radius_lognormal_mass_around_earthlike_rocky(R) for R in R_logunif])
    assert np.all(0. <= M_array)
    assert gen.Pearson_correlation_coefficient(R_logunif, M_array) > 0.

def test_generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below(seed=42):
    np.random.seed(seed)
    logR_min, logR_max = np.log10(0.5), 2. # log10(Earth radii)
    R_logunif = 10.**(logR_min + np.random.rand(10000)*(logR_max - logR_min)) # log-uniform
    M_array = np.array([generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below(R) for R in R_logunif])
    assert np.all(0. <= M_array)
    assert gen.Pearson_correlation_coefficient(R_logunif, M_array) > 0.
    M_array2 = generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec(R_logunif)
    assert np.all(0. <= M_array2)
    assert gen.Pearson_correlation_coefficient(R_logunif, M_array2) > 0.

def test_solid_surface_density(seed=42):
    np.random.seed(seed)
    M, a, delta_a = np.array([100., 2., 2.])*np.random.rand(3)
    assert np.isclose(solid_surface_density(1., 1., 1.), 4.246946, atol=1e-5)
    assert np.isclose(solid_surface_density(10., 0.3, 0.3), 471.882894, atol=1e-5)
    assert solid_surface_density(2., a, delta_a) < solid_surface_density(5., a, delta_a)
    assert solid_surface_density(M, 0.8, delta_a) < solid_surface_density(M, 0.5, delta_a)
    assert solid_surface_density(M, a, 0.2) < solid_surface_density(M, a, 0.1)

def test_solid_surface_density_CL2013():
    assert np.isclose(solid_surface_density_CL2013(1., 1.,), 4.246946, atol=1e-5)
    assert np.isclose(solid_surface_density_CL2013(10., 0.3), 471.882894, atol=1e-5)

def test_solid_surface_density_S2014():
    assert np.isclose(solid_surface_density_S2014(1., 1., 1.), 5.654930, atol=1e-5)

def test_solid_surface_density_nHill():
    assert np.isclose(solid_surface_density_nHill(1., 1.), 42.457605, atol=1e-5)
    assert 0 < solid_surface_density_nHill(1., 1., n=10.) < solid_surface_density_nHill(1., 1., n=8.)

def test_solid_surface_density_system_RC2014(seed=42):
    np.random.seed(seed)
    M_sys = 10.*np.random.rand(3)
    a_sys = np.sort(10.**(-1. + 2.*np.random.rand(3))) # log-uniform between 0.1 and 10 AU
    sigma_sys_in_out = solid_surface_density_system_RC2014(M_sys, a_sys)[::2] # solid surface densities based on inner and outer planets
    sigma_sys_removed_mid = solid_surface_density_system_RC2014(M_sys[::2], a_sys[::2]) # same as above, but remove the middle planet
    assert 0 < sigma_sys_removed_mid[0] < sigma_sys_in_out[0]
    assert 0 < sigma_sys_removed_mid[1] < sigma_sys_in_out[1]

#loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
loadfiles_directory = 'C:/Users/HeYMa/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
run_number = ''
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True, match_observed=True)

def test_solid_surface_density_CL2013_given_physical_catalog(sssp_per_sys=sssp_per_sys):
    sigma_all, a_all = solid_surface_density_CL2013_given_physical_catalog(sssp_per_sys)
    assert len(sigma_all) == len(a_all)
    assert 0 < np.min(sigma_all)
    assert 0 < np.min(a_all)

def test_MMSN():
    assert np.isclose(MMSN(1.), 10.89)
    assert np.isclose(MMSN(0.5, F=2.), 2.*MMSN(0.5, F=1.))
    assert np.isclose(MMSN(0.3, Zrel=0.4), 4.*MMSN(0.3, Zrel=0.1))

def test_MMEN_power_law(seed=42):
    np.random.seed(seed)
    a0 = 2.*np.random.rand()
    sigma0 = 100.*np.random.rand()
    beta = -5. + 4.*np.random.rand()
    assert np.isclose(MMEN_power_law(a0, sigma0, beta, a0=a0), sigma0)
    assert MMEN_power_law(1., sigma0, beta) < MMEN_power_law(0.5, sigma0, beta)
    assert np.isclose(MMEN_power_law(1., 2.*sigma0, beta), 2.*MMEN_power_law(1., sigma0, beta))
