# To import required modules:
import numpy as np
import os
import sys

#sys.path.append('/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/SysSimExClusters_plotting') # TODO: update when those files get made into a package

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.MMEN_functions import *





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
