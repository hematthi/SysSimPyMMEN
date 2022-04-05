# To import required modules:
import numpy as np
import os
import time
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
from scipy.stats import truncnorm
from scipy.optimize import curve_fit # for fitting functions
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec # for specifying plot attributes
from matplotlib import ticker # for setting contour plots to log scale

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),'SysSim_Plotting/src')) # TODO: update when those files get made into a package

import functions_general as gen





MeVeEa_masses = np.array([0.0553, 0.815, 1.]) # masses of Mercury, Venus, and Earth, in Earth masses
MeVeEa_radii = np.array([0.383, 0.949, 1.]) # radii of Mercury, Venus, and Earth, in Earth radii
MeVeEa_a = np.array([0.387, 0.723, 1.]) # semi-major axes of Mercury, Venus, and Earth, in AU



##### Mass-radius functions:

def mass_given_radius_density(R, ρ):
    # Compute mass M (Earth masses) given radius R (Earth radii) and mean density ρ (g/cm^3)
    return ρ * (4.*np.pi/3.) * (R*gen.Rearth)**3. * (1./(gen.Mearth*1e3))

def density_given_mass_radius(M, R):
    # Compute mean density ρ (g/cm^3) given mass M (Earth masses) and radius R (Earth radii)
    return (M*gen.Mearth*1e3) / ((4.*np.pi/3.)*(R*gen.Rearth)**3.)



data_path = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/SysSimExClusters/src/mr_model/MRpredict_table_weights3025_R1001_Q1001.txt'
#data_path = 'C:/Users/HeYMa/Documents/GradSchool/Research/SysSimExClusters/src/mr_model/MRpredict_table_weights3025_R1001_Q1001.txt'
logMR_Ning2018_table = np.genfromtxt(data_path, delimiter=',', skip_header=2, names=True) # first column is array of log_R values

table_array = logMR_Ning2018_table.view(np.float64).reshape(logMR_Ning2018_table.shape + (-1,))[:,1:]
log_R_table = logMR_Ning2018_table['log_R']
qtls_table = np.linspace(0,1,1001)
logMR_table_interp = RectBivariateSpline(log_R_table, qtls_table, table_array)

def generate_planet_mass_from_radius_Ning2018_table(R):
    # Draw a planet mass from the Ning et al 2018 model interpolated on a precomputed table
    # Requires a globally defined interpolation object ('logMR_table_interp')
    # R is planet radius in Earth radii
    logR = np.log10(R)
    q = np.random.random() # drawn quantile
    logM = logMR_table_interp(logR, q)[0][0]
    return 10.**logM # planet mass in Earth masses

data_path = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/SysSimExClusters/src/mr_model/MR_earthlike_rocky.txt'
#data_path = 'C:/Users/HeYMa/Documents/GradSchool/Research/SysSimExClusters/src/mr_model/MR_earthlike_rocky.txt'
MR_earthlike_rocky = np.genfromtxt(data_path, delimiter='\t', skip_header=2, names=('mass','radius'))

M_earthlike_rocky_interp = interp1d(MR_earthlike_rocky['radius'], MR_earthlike_rocky['mass'])

radius_switch = 1.472
σ_logM_at_radius_switch = 0.3 # log10(M/M_earth); 0.3 corresponds to about a factor of 2, and appears to be close to the std of the distribution at R=1.472 R_earth for the NWG-2018 model
σ_logM_at_radius_min = 0.04 # log10(M/M_earth); 0.04 corresponds to about a factor of 10%
def σ_logM_linear_given_radius(R, σ_logM_r1=σ_logM_at_radius_min, σ_logM_r2=σ_logM_at_radius_switch, r1=0.5, r2=radius_switch):
    return ((σ_logM_at_radius_switch - σ_logM_at_radius_min) / (r2 - r1))*(R - r1) + σ_logM_at_radius_min

def generate_planet_mass_from_radius_lognormal_mass_around_earthlike_rocky(R, ρ_min=1., ρ_max=100.):
    # Draw a planet mass given a planet radius, from a lognormal distribution centered around the Earth-like rocky model
    # Requires a globally defined interpolation object ('M_earthlike_rocky_interp')
    # R is planet radius in Earth radii
    M = M_earthlike_rocky_interp(R)
    assert M > 0
    μ_logM = np.log10(M)
    σ_logM = σ_logM_linear_given_radius(R)
    logM_min = np.log10(mass_given_radius_density(R, ρ_min))
    logM_max = np.log10(mass_given_radius_density(R, ρ_max))
    a, b = (logM_min - μ_logM)/σ_logM, (logM_max - μ_logM)/σ_logM
    logM = truncnorm.rvs(a, b, loc=μ_logM, scale=σ_logM, size=1)[0] # draw from truncated normal distribution for log10(M/M_earth)
    return 10.**logM # planet mass in Earth masses

def generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below(R, R_switch=radius_switch, ρ_min=1., ρ_max=100.):
    #assert R > 0
    if R > R_switch:
        M = generate_planet_mass_from_radius_Ning2018_table(R)
    elif R <= 0:
        M = 0.
    else:
        M = generate_planet_mass_from_radius_lognormal_mass_around_earthlike_rocky(R, ρ_min=ρ_min, ρ_max=ρ_max)
    return M

generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec = np.vectorize(generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below) # vectorized form of function



##### Functions to compute various formulations of the solid surface density/minimum mass extrasolar nebular (MMEN):

def solid_surface_density(M, a, delta_a):
    # Compute the solid surface density (g/cm^2) given a planet mass M (M_earth), semi-major axis a (AU), and feeding zone width delta_a (AU)
    sigma_solid = (M*gen.Mearth*1e3)/(2.*np.pi*(a*gen.AU)*(delta_a*gen.AU))
    return sigma_solid

def solid_surface_density_CL2013(M, a):
    # Compute the solid surface density (g/cm^2) using the Chiang & Laughlin (2013) prescription (delta_a = a), given a planet mass M (M_earth) and semi-major axis a (AU)
    return solid_surface_density(M, a, a)

# TODO: write unit tests
def feeding_zone_S2014(M, R, a, Mstar=1.):
    # Compute the feeding zone width using the Schlicting (2014) prescription (delta_a = 2^(3/2)*a((a*M)/(R*Mstar))^(1/2), given a planet mass M (M_earth), planet radius R (R_earth), semi-major axis a (AU), and stellar mass Mstar (M_sun)
    delta_a = 2.**(3./2.)*a*np.sqrt(((a*gen.AU)/(R*gen.Rearth))*((M*gen.Mearth)/(Mstar*gen.Msun))) # AU
    return delta_a

def solid_surface_density_S2014(M, R, a, Mstar=1.):
    # Compute the solid surface density (g/cm^2) using the Schlichting (2014) prescription (delta_a = 2^(3/2)*a((a*M)/(R*Mstar))^(1/2), given a planet mass M (M_earth), planet radius R (R_earth), semi-major axis a (AU), and stellar mass Mstar (M_sun)
    delta_a = feeding_zone_S2014(M, R, a, Mstar=Mstar)
    return solid_surface_density(M, a, delta_a)

# TODO: write unit tests
def feeding_zone_nHill(M, a, Mstar=1., n=10.):
    # Compute the feeding zone width using a number of Hill radii (delta_a = n*R_Hill), given a planet mass M (M_earth), semi-major axis a (AU), and stellar mass Mstar (M_sun)
    delta_a = n*a*((M*gen.Mearth)/(3.*Mstar*gen.Msun))**(1./3.) # AU
    return delta_a

def solid_surface_density_nHill(M, a, Mstar=1., n=10.):
    # Compute the solid surface density (g/cm^2) using a number of Hill radii for the feeding zone width (delta_a = n*R_Hill), given a planet mass M (M_earth), semi-major axis a (AU), and stellar mass Mstar (M_sun)
    delta_a = feeding_zone_nHill(M, a, Mstar=Mstar, n=n)
    return solid_surface_density(M, a, delta_a)

# TODO: write unit tests
def feeding_zone_RC2014(a_sys):
    # Compute the feeding zone widths for planets in a multiplanet system using the Raymond & Cossou (2014) prescription (neighboring planets' feeding zones separated by the geometric means of their semi-major axes)
    a_bounds_sys = np.zeros(len(a_sys)+1) # bounds in semi-major axis for each planet
    a_bounds_sys[1:-1] = np.sqrt(a_sys[:-1]*a_sys[1:]) # geometric means between planets
    a_bounds_sys[0] = a_sys[0]*np.sqrt(a_sys[0]/a_sys[1]) # same ratio for upper bound to a_sys[1] as a_sys[1] to lower bound
    a_bounds_sys[-1] = a_sys[-1]*np.sqrt(a_sys[-1]/a_sys[-2]) # same ratio for upper bound to a_sys[-1] as a_sys[-1] to lower bound
    delta_a_sys = np.diff(a_bounds_sys)
    return delta_a_sys, a_bounds_sys

def solid_surface_density_system_RC2014(M_sys, a_sys):
    # Compute the solid surface density (g/cm^2) of planets in a multiplanet system using the Raymond & Cossou (2014) prescription (neighboring planets' feeding zones separated by the geometric means of their semi-major axes)
    n_pl = len(M_sys)
    assert n_pl == len(a_sys) > 1
    delta_a_sys, _ = feeding_zone_RC2014(a_sys)
    return solid_surface_density(M_sys, a_sys, delta_a_sys)

def solid_surface_density_prescription(M, R, a, Mstar=1., n=10., prescription='CL2013'):
    # Wrapper function to compute the solid surface density (g/cm^2) of planets given a prescription
    if prescription == 'CL2013':
        return solid_surface_density_CL2013(M, a)
    elif prescription == 'S2014':
        return solid_surface_density_S2014(M, R, a, Mstar=Mstar)
    elif prescription == 'nHill':
        return solid_surface_density_nHill(M, a, Mstar=Mstar, n=n)
    elif prescription == 'RC2014':
        # Warning: 'M' and 'a' must all be in the same system
        return solid_surface_density_system_RC2014(M, a)
    else:
        print('No matching prescription!')



def solid_surface_density_CL2013_given_physical_catalog(sssp_per_sys, max_core_mass=10.):
    # Compute the solid surface density (g/cm^2) using the Chiang & Laughlin (2013) prescription, for each planet in a given physical catalog
    # 'max_core_mass' is the maximum core mass (Earth masses)
    # Returns an array of solid surface densities and semi-major axes
    a_all = sssp_per_sys['a_all'][sssp_per_sys['a_all'] > 0]
    core_mass_all = np.copy(sssp_per_sys['mass_all'][sssp_per_sys['a_all'] > 0])
    core_mass_all[core_mass_all > max_core_mass] = max_core_mass
    sigma_all = solid_surface_density_CL2013(core_mass_all, a_all)
    return sigma_all, a_all

def solid_surface_density_S2014_given_physical_catalog(sssp_per_sys, sssp, max_core_mass=10.):
    # Compute the solid surface density (g/cm^2) using the Schlichting (2014) prescription, for each planet in a given physical catalog
    # 'max_core_mass' is the maximum core mass (Earth masses)
    # Returns an array of solid surface densities and semi-major axes
    a_all = sssp_per_sys['a_all'][sssp_per_sys['a_all'] > 0]
    core_mass_all = np.copy(sssp_per_sys['mass_all'])
    core_mass_all[core_mass_all > max_core_mass] = max_core_mass
    sigma_all = solid_surface_density_S2014(core_mass_all, sssp_per_sys['radii_all'], sssp_per_sys['a_all'], Mstar=sssp['Mstar_all'][:,None])[sssp_per_sys['a_all'] > 0]
    return sigma_all, a_all

def solid_surface_density_nHill_given_physical_catalog(sssp_per_sys, sssp, max_core_mass=10., n=10.):
    # Compute the solid surface density (g/cm^2) using a number of Hill radii for the feeding zone width, for each planet in a given physical catalog
    # 'max_core_mass' is the maximum core mass (Earth masses)
    # 'n' is the number of Hill radii for  the feeding zone width of each planet
    # Returns an array of solid surface densities and semi-major axes
    a_all = sssp_per_sys['a_all'][sssp_per_sys['a_all'] > 0]
    core_mass_all = np.copy(sssp_per_sys['mass_all'])
    core_mass_all[core_mass_all > max_core_mass] = max_core_mass
    sigma_all = solid_surface_density_nHill(core_mass_all, sssp_per_sys['a_all'], Mstar=sssp['Mstar_all'][:,None], n=n)[sssp_per_sys['a_all'] > 0]
    return sigma_all, a_all

def solid_surface_density_RC2014_given_physical_catalog(sssp_per_sys, max_core_mass=10.):
    # Compute the solid surface density (g/cm^2) using the Raymond & Cossou (2014) prescription, for each planet in each multi-planet system in a given physical catalog
    # 'max_core_mass' is the maximum core mass (Earth masses)
    # Returns an array of solid surface densities and semi-major axes
    mult_all = sssp_per_sys['Mtot_all']
    a_all_2p = []
    mult_all_2p = []
    sigma_all_2p = []
    for i in np.arange(len(mult_all))[mult_all > 1]: # only consider multi-planet systems
        a_sys = sssp_per_sys['a_all'][i]
        core_mass_sys = np.copy(sssp_per_sys['mass_all'][i][a_sys > 0])
        core_mass_sys[core_mass_sys > max_core_mass] = max_core_mass
        a_sys = a_sys[a_sys > 0]
        a_all_2p += list(a_sys)
        mult_all_2p += [len(a_sys)]*len(a_sys)
        sigma_all_2p += list(solid_surface_density_system_RC2014(core_mass_sys, a_sys))
    a_all_2p = np.array(a_all_2p)
    mult_all_2p = np.array(mult_all_2p)
    sigma_all_2p = np.array(sigma_all_2p)
    return sigma_all_2p, a_all_2p, mult_all_2p



def solid_surface_density_CL2013_given_observed_catalog(sss_per_sys, max_core_mass=10.):
    # Compute the solid surface density (g/cm^2) using the Chiang & Laughlin (2013) prescription, for each planet in a given observed catalog, using a mass-radius relation on the observed radii
    # 'max_core_mass' is the maximum core mass (Earth masses)
    # Returns an array of solid surface densities and semi-major axes
    a_obs_per_sys = gen.a_from_P(sss_per_sys['P_obs'], sss_per_sys['Mstar_obs'][:,None])
    a_obs = a_obs_per_sys[sss_per_sys['P_obs'] > 0]
    radii_obs = sss_per_sys['radii_obs'][sss_per_sys['P_obs'] > 0]
    core_mass_obs = generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec(radii_obs)
    core_mass_obs[core_mass_obs > max_core_mass] = max_core_mass
    sigma_obs = solid_surface_density_CL2013(core_mass_obs, a_obs)
    return sigma_obs, core_mass_obs, a_obs

def solid_surface_density_S2014_given_observed_catalog(sss_per_sys, max_core_mass=10.):
    # Compute the solid surface density (g/cm^2) using the Schlichting (2014) prescription, for each planet in a given observed catalog, using a mass-radius relation on the observed radii
    # 'max_core_mass' is the maximum core mass (Earth masses)
    # Returns an array of solid surface densities and semi-major axes
    Mstar_obs = np.repeat(sss_per_sys['Mstar_obs'][:,None], np.shape(sss_per_sys['P_obs'])[1], axis=1)[sss_per_sys['P_obs'] > 0] # flattened array of stellar masses repeated for each planet
    a_obs_per_sys = gen.a_from_P(sss_per_sys['P_obs'], sss_per_sys['Mstar_obs'][:,None])
    a_obs = a_obs_per_sys[sss_per_sys['P_obs'] > 0]
    radii_obs = sss_per_sys['radii_obs'][sss_per_sys['P_obs'] > 0]
    core_mass_obs = generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec(radii_obs)
    core_mass_obs[core_mass_obs > max_core_mass] = max_core_mass
    sigma_obs = solid_surface_density_S2014(core_mass_obs, radii_obs, a_obs, Mstar=Mstar_obs)
    return sigma_obs, core_mass_obs, a_obs

def solid_surface_density_nHill_given_observed_catalog(sss_per_sys, max_core_mass=10., n=10.):
    # Compute the solid surface density (g/cm^2) using a number of Hill radii for the feeding zone width, for each planet in a given observed catalog, using a mass-radius relation on the observed radii
    # 'max_core_mass' is the maximum core mass (Earth masses)
    # 'n' is the number of Hill radii for  the feeding zone width of each planet
    # Returns an array of solid surface densities and semi-major axes
    Mstar_obs = np.repeat(sss_per_sys['Mstar_obs'][:,None], np.shape(sss_per_sys['P_obs'])[1], axis=1)[sss_per_sys['P_obs'] > 0] # flattened array of stellar masses repeated for each planet
    a_obs_per_sys = gen.a_from_P(sss_per_sys['P_obs'], sss_per_sys['Mstar_obs'][:,None])
    a_obs = a_obs_per_sys[sss_per_sys['P_obs'] > 0]
    radii_obs = sss_per_sys['radii_obs'][sss_per_sys['P_obs'] > 0]
    core_mass_obs = generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec(radii_obs)
    core_mass_obs[core_mass_obs > max_core_mass] = max_core_mass
    sigma_obs = solid_surface_density_nHill(core_mass_obs, a_obs, Mstar=Mstar_obs, n=n)
    return sigma_obs, core_mass_obs, a_obs

def solid_surface_density_RC2014_given_observed_catalog(sss_per_sys, max_core_mass=10.):
    # Compute the solid surface density (g/cm^2) using the Raymond & Cossou (2014) prescription, for each planet in a given observed catalog, using a mass-radius relation on the observed radii
    # 'max_core_mass' is the maximum core mass (Earth masses)
    # Returns an array of solid surface densities and semi-major axes
    mult_obs = sss_per_sys['Mtot_obs']
    mult_obs_2p = []
    a_obs_2p = []
    core_mass_obs_2p = []
    sigma_obs_2p = []
    for i in np.arange(len(mult_obs))[mult_obs > 1]: # only consider multi-planet systems
        a_sys = gen.a_from_P(sss_per_sys['P_obs'][i], sss_per_sys['Mstar_obs'][i])
        core_mass_sys = generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec(sss_per_sys['radii_obs'][i][a_sys > 0])
        core_mass_sys[core_mass_sys > max_core_mass] = max_core_mass
        a_sys = a_sys[a_sys > 0]

        mult_obs_2p += [len(a_sys)]*len(a_sys)
        a_obs_2p += list(a_sys)
        core_mass_obs_2p += list(core_mass_sys)
        sigma_obs_2p += list(solid_surface_density_system_RC2014(core_mass_sys, a_sys))
    mult_obs_2p = np.array(mult_obs_2p)
    a_obs_2p = np.array(a_obs_2p)
    core_mass_obs_2p = np.array(core_mass_obs_2p)
    sigma_obs_2p = np.array(sigma_obs_2p)
    return sigma_obs_2p, core_mass_obs_2p, a_obs_2p, mult_obs_2p



##### Functions to fit a power-law to the MMEN (sigma = sigma0 * (a/a0)^beta <==> log(sigma) = log(sigma0) + beta*log(a/a0)):

def MMSN(a, F=1., Zrel=0.33):
    # Compute the minimum mass solar nebular (MMSN) surface density (g/cm^2) at a semi-major axis 'a' (AU), from equation 2 in Chiang & Youdin (2010) (https://arxiv.org/pdf/0909.2652.pdf)
    # Default values of F=1 and Zrel=0.33 give 1M_earth of solids in an annulus centered on Earth's orbit
    sigma_p = 33.*F*Zrel*(a**-1.5)
    return sigma_p

def MMEN_power_law(a, sigma0, beta, a0=1.):
    # Compute a power-law profile for the MMEN (solid surface density as a function of semi-major axis), given by:
    # sigma(a) = sigma0*(a/a0)^beta, where 'sigma(a)' (g/cm^2) is the solid surface density at semi-major axis 'a' (AU), 'sigma0' (g/cm^2) is the normalization at semi-major axis 'a0' (AU), and beta is the power-law index
    assert sigma0 > 0
    assert a0 > 0
    sigma_a = sigma0 * (a/a0)**beta
    return sigma_a

f_linear = lambda x, p0, p1: p0 + p1*x

def fit_power_law_MMEN(a_array, sigma_array, a0=1., p0=1., p1=-1.5):
    # Fit power-law parameters (normalization 'sigma0' (at separation 'a0') and slope 'beta') given an array of separations 'a_array' (AU) and solid surface densities 'sigma_array' (g/cm^2)
    # Optional parameters: separation for the normalization 'a0' (AU; i.e. sigma0 = sigma(a0)), initial guess for log(sigma0) 'p0', and initial guess for beta 'p1'
    mmen_fit = curve_fit(f_linear, np.log10(a_array/a0), np.log10(sigma_array), [p0, p1])[0]
    sigma0, beta = 10.**(mmen_fit[0]), mmen_fit[1]
    return sigma0, beta

def fit_power_law_MMEN_all_planets_observed(sss_per_sys, max_core_mass=10., prescription='CL2013', n=10., a0=1., p0=1., p1=-1.5):
    # Compute solid surface densities and fit a power-law to all planets in an observed catalog
    if prescription == 'CL2013':
        sigma_obs, core_mass_obs, a_obs = solid_surface_density_CL2013_given_observed_catalog(sss_per_sys, max_core_mass=max_core_mass)
    elif prescription == 'S2014':
        sigma_obs, core_mass_obs, a_obs = solid_surface_density_S2014_given_observed_catalog(sss_per_sys, max_core_mass=max_core_mass)
    elif prescription == 'nHill':
        sigma_obs, core_mass_obs, a_obs = solid_surface_density_nHill_given_observed_catalog(sss_per_sys, max_core_mass=max_core_mass, n=n)
    elif prescription == 'RC2014':
        sigma_obs, core_mass_obs, a_obs, mult_obs = solid_surface_density_RC2014_given_observed_catalog(sss_per_sys, max_core_mass=max_core_mass) # outputs are for systems with 2+ planets only
    else:
        print('No matching prescription!')
    sigma0, beta = fit_power_law_MMEN(a_obs, sigma_obs, a0=a0, p0=p0, p1=p1)
    outputs_dict = {'sigma_obs': sigma_obs, 'core_mass_obs': core_mass_obs, 'a_obs': a_obs, 'sigma0': sigma0, 'beta': beta}
    return outputs_dict

def fit_power_law_MMEN_all_planets_physical(sssp_per_sys, sssp, max_core_mass=10., prescription='CL2013', n=10., a0=1., p0=1., p1=-1.5):
    # Compute solid surface densities and fit a power-law to all planets in a physical catalog
    if prescription == 'CL2013':
        sigma_all, a_all = solid_surface_density_CL2013_given_physical_catalog(sssp_per_sys, max_core_mass=max_core_mass)
    elif prescription == 'S2014':
        sigma_all, a_all = solid_surface_density_S2014_given_physical_catalog(sssp_per_sys, sssp, max_core_mass=max_core_mass)
    elif prescription == 'nHill':
        sigma_all, a_all = solid_surface_density_nHill_given_physical_catalog(sssp_per_sys, sssp, max_core_mass=max_core_mass, n=n)
    elif prescription == 'RC2014':
        sigma_all, a_all, mult_all = solid_surface_density_RC2014_given_physical_catalog(sssp_per_sys, max_core_mass=max_core_mass) # outputs are for systems with 2+ planets only
    else:
        print('No matching prescription!')
    sigma0, beta = fit_power_law_MMEN(a_all, sigma_all, a0=a0, p0=p0, p1=p1)
    outputs_dict = {'sigma_all': sigma_all, 'a_all': a_all, 'sigma0': sigma0, 'beta': beta}
    return outputs_dict

def fit_power_law_MMEN_per_system_observed(sss_per_sys, max_core_mass=10., prescription='CL2013', n=10., a0=1., p0=1., p1=-1.5, scale_up=False):
    # Compute solid surface densities and fit power-law parameters to each multi-planet system in an observed catalog
    # If 'scale_up' is True, will scale up the power-law to be above the surface densities of all planets in the system (i.e. multiply 'sigma0' by a factor such that sigma0*(a_i/a0)^beta >= sigma_i for all planets)
    fit_per_sys_dict = {'m_obs':[], 'Mstar_obs':[], 'sigma0':[], 'scale_factor':[], 'beta':[]} # 'm_obs' is number of observed planets
    a_obs_per_sys = gen.a_from_P(sss_per_sys['P_obs'], sss_per_sys['Mstar_obs'][:,None])
    for i,a_sys in enumerate(a_obs_per_sys):
        if np.sum(a_sys > 0) > 1:
            #print(i)
            Mstar = sss_per_sys['Mstar_obs'][i]
            R_sys = sss_per_sys['radii_obs'][i][a_sys > 0]
            core_mass_sys = generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec(R_sys)
            core_mass_sys[core_mass_sys > max_core_mass] = max_core_mass
            a_sys = a_sys[a_sys > 0]
            sigma_obs_sys = solid_surface_density_prescription(core_mass_sys, R_sys, a_sys, Mstar=Mstar, n=n, prescription=prescription)
            sigma0, beta = fit_power_law_MMEN(a_sys, sigma_obs_sys, a0=a0, p0=p0, p1=p1)

            sigma_fit_sys = MMEN_power_law(a_sys, sigma0, beta, a0=a0) # the solid surface density at the semi-major axes of each planet in the system, as computed from the power-law fit
            sigma_fit_ratio_sys = sigma_obs_sys/sigma_fit_sys
            scale_factor_sigma0 = np.max(sigma_fit_ratio_sys)
            sigma0 = sigma0*scale_factor_sigma0 if scale_up else sigma0 # scale up sigma0 such that the power-law fit is above the solid surface density of all the planets

            fit_per_sys_dict['m_obs'].append(len(a_sys))
            fit_per_sys_dict['Mstar_obs'].append(sss_per_sys['Mstar_obs'][i])
            fit_per_sys_dict['sigma0'].append(sigma0)
            fit_per_sys_dict['scale_factor'].append(scale_factor_sigma0)
            fit_per_sys_dict['beta'].append(beta)

    fit_per_sys_dict['m_obs'] = np.array(fit_per_sys_dict['m_obs'])
    fit_per_sys_dict['Mstar_obs'] = np.array(fit_per_sys_dict['Mstar_obs'])
    fit_per_sys_dict['sigma0'] = np.array(fit_per_sys_dict['sigma0'])
    fit_per_sys_dict['scale_factor'] = np.array(fit_per_sys_dict['scale_factor'])
    fit_per_sys_dict['beta'] = np.array(fit_per_sys_dict['beta'])
    return fit_per_sys_dict

def fit_power_law_MMEN_per_system_physical(sssp_per_sys, sssp, max_core_mass=10., prescription='CL2013', n=10., a0=1., p0=1., p1=-1.5, scale_up=False, N_sys=10000):
    # Compute solid surface densities and fit power-law parameters to each multi-planet system in a physical catalog
    # If 'scale_up' is True, will scale up the power-law to be above the surface densities of all planets in the system (i.e. multiply 'sigma0' by a factor such that sigma0*(a_i/a0)^beta >= sigma_i for all planets)
    # 'N_sys' is the maximum number of systems to loop through (to save time)
    start = time.time()
    N_sys_tot = len(sssp_per_sys['a_all'])
    print('Fitting power-laws to the first %s systems (out of %s)...' % (min(N_sys,N_sys_tot), N_sys_tot))
    fit_per_sys_dict = {'n_pl':[], 'sigma0':[], 'scale_factor':[], 'beta':[]} # 'n_pl' is number of planets in each system
    for i,a_sys in enumerate(sssp_per_sys['a_all'][:N_sys]):
        if np.sum(a_sys > 0) > 1:
            Mstar = sssp['Mstar_all'][i]
            core_mass_sys = np.copy(sssp_per_sys['mass_all'][i])[a_sys > 0]
            core_mass_sys[core_mass_sys > max_core_mass] = max_core_mass
            R_sys = sssp_per_sys['radii_all'][i][a_sys > 0]
            a_sys = a_sys[a_sys > 0]
            sigma_sys = solid_surface_density_prescription(core_mass_sys, R_sys, a_sys, Mstar=Mstar, n=n, prescription=prescription)
            sigma0, beta = fit_power_law_MMEN(a_sys, sigma_sys, a0=a0, p0=p0, p1=p1)

            sigma_fit_sys = MMEN_power_law(a_sys, sigma0, beta, a0=a0) # the solid surface density at the semi-major axes of each planet in the system, as computed from the power-law fit
            sigma_fit_ratio_sys = sigma_sys/sigma_fit_sys
            scale_factor_sigma0 = np.max(sigma_fit_ratio_sys)
            sigma0 = sigma0*scale_factor_sigma0 if scale_up else sigma0 # scale up sigma0 such that the power-law fit is above the solid surface density of all the planets

            fit_per_sys_dict['n_pl'].append(len(a_sys))
            fit_per_sys_dict['sigma0'].append(sigma0)
            fit_per_sys_dict['scale_factor'].append(scale_factor_sigma0)
            fit_per_sys_dict['beta'].append(beta)

    fit_per_sys_dict['n_pl'] = np.array(fit_per_sys_dict['n_pl'])
    fit_per_sys_dict['sigma0'] = np.array(fit_per_sys_dict['sigma0'])
    fit_per_sys_dict['scale_factor'] = np.array(fit_per_sys_dict['scale_factor'])
    fit_per_sys_dict['beta'] = np.array(fit_per_sys_dict['beta'])
    stop = time.time()
    print('Time to compute: %s s' % (stop - start))
    return fit_per_sys_dict

def fit_power_law_MMEN_per_system_observed_and_physical(sssp_per_sys, sssp, max_core_mass=10., prescription='CL2013', n=10., a0=1., p0=1., p1=-1.5, scale_up=False):
    # Compute solid surface densities and fit power-law parameters to each multi-planet system in an observed catalog, for the observed planets only and then for all the planets in those systems (using the physical planet properties for both)
    # If 'scale_up' is True, will scale up the power-law to be above the surface densities of all planets in the system (i.e. multiply 'sigma0' by a factor such that sigma0*(a_i/a0)^beta >= sigma_i for all planets)
    fit_per_sys_dict = {'n_pl_true':[], 'n_pl_obs':[], 'sigma0_true':[], 'sigma0_obs':[], 'scale_factor_true':[], 'scale_factor_obs':[], 'beta_true':[], 'beta_obs':[]}
    for i,det_sys in enumerate(sssp_per_sys['det_all']):
        if np.sum(det_sys) > 1:
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

            fit_per_sys_dict['n_pl_true'].append(len(a_sys))
            fit_per_sys_dict['n_pl_obs'].append(len(a_sys_obs))
            fit_per_sys_dict['sigma0_true'].append(sigma0)
            fit_per_sys_dict['sigma0_obs'].append(sigma0_obs)
            fit_per_sys_dict['scale_factor_true'].append(scale_factor_sigma0)
            fit_per_sys_dict['scale_factor_obs'].append(scale_factor_sigma0_obs)
            fit_per_sys_dict['beta_true'].append(beta)
            fit_per_sys_dict['beta_obs'].append(beta_obs)

    fit_per_sys_dict['n_pl_true'] = np.array(fit_per_sys_dict['n_pl_true'])
    fit_per_sys_dict['n_pl_obs'] = np.array(fit_per_sys_dict['n_pl_obs'])
    fit_per_sys_dict['sigma0_true'] = np.array(fit_per_sys_dict['sigma0_true'])
    fit_per_sys_dict['sigma0_obs'] = np.array(fit_per_sys_dict['sigma0_obs'])
    fit_per_sys_dict['scale_factor_true'] = np.array(fit_per_sys_dict['scale_factor_true'])
    fit_per_sys_dict['scale_factor_obs'] = np.array(fit_per_sys_dict['scale_factor_obs'])
    fit_per_sys_dict['beta_true'] = np.array(fit_per_sys_dict['beta_true'])
    fit_per_sys_dict['beta_obs'] = np.array(fit_per_sys_dict['beta_obs'])
    return fit_per_sys_dict

# TODO: write unit tests
def plot_feeding_zones_and_power_law_fit_MMEN_per_system_observed_and_physical(sssp_per_sys, sssp, max_core_mass=10., prescription='CL2013', n=10., a0=1., p0=1., p1=-1.5, scale_up=False, N_sys=10):
    # If 'scale_up' is True, will scale up the power-law to be above the surface densities of all planets in the system (i.e. multiply 'sigma0' by a factor such that sigma0*(a_i/a0)^beta >= sigma_i for all planets)
    # 'N_sys' is the maximum number of systems to loop through (to save time)
    for i,det_sys in enumerate(sssp_per_sys['det_all'][:N_sys]):
        if np.sum(det_sys) > 1:
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

            delta_a_sys_S2014 = feeding_zone_S2014(core_mass_sys, R_sys, a_sys, Mstar=Mstar)
            delta_a_sys_nHill = feeding_zone_nHill(core_mass_sys, a_sys, Mstar=Mstar, n=n)
            delta_a_sys_RC2014, a_bounds_sys = feeding_zone_RC2014(a_sys)

            fig = plt.figure(figsize=(16,8))
            plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
            ax = plt.subplot(plot[0,0])
            plt.scatter(a_sys, np.log10(sigma_sys), marker='o', s=100.*R_sys**2., color='k', label='All planets')
            for j,a in enumerate(a_sys): # loop through each planet
                da_S2014 = delta_a_sys_S2014[j]
                da_nHill = delta_a_sys_nHill[j]
                #da_RC2014 = delta_a_sys_RC2014[j]
                plt.plot([0.5*a, 1.5*a], [np.log10(1.15*sigma_sys[j])]*2, lw=2, color='k')
                plt.plot([a - da_S2014/2., a + da_S2014/2.], [np.log10(1.05*sigma_sys[j])]*2, lw=2, color='r')
                plt.plot([a - da_nHill/2., a + da_nHill/2.], [np.log10(0.95*sigma_sys[j])]*2, lw=2, color='orange')
                plt.plot([a_bounds_sys[j], a_bounds_sys[j+1]], [np.log10(0.85*sigma_sys[j])]*2, lw=2, color='b')
            plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0, beta, a0=a0)), lw=3, ls='--', color='r', label=r'Fit to all planets ($\Sigma_0 = {:0.2f}$, $\beta = {:0.2f}$)'.format(sigma0, beta))
            plt.plot(a_array, np.log10(sigma_MMSN), lw=3, color='g', label=r'MMSN ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(MMSN(a0), -1.5)) #label=r'MMSN ($\sigma_{\rm solid} = 10.89(a/{\rm AU})^{-3/2}$ g/cm$^2$)'
            #plt.scatter(MeVeEa_a, np.log10(MeVeEa_sigmas), marker='o', s=100, color='g', label='') #label='Solar system planets (Mercury, Venus, Earth)'
            ax.tick_params(axis='both', labelsize=20)
            plt.gca().set_xscale("log")
            plt.xticks([0.05, 0.1, 0.2, 0.4, 0.8])
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            plt.xlim([0.04,0.9])
            plt.ylim([-0.5,5.5])
            plt.xlabel(r'Semimajor axis, $a$ (AU)', fontsize=20)
            plt.ylabel(r'Surface density, $\log_{10}(\Sigma/{\rm gcm}^{-2})$', fontsize=20)
            plt.legend(loc='lower left', bbox_to_anchor=(0.,0.), ncol=1, frameon=False, fontsize=lfs)

            plt.show()


##### Functions to compute the total integrated mass of solids within a given separation for a fitted power-law MMEN:

def solid_mass_integrated_r0_to_r_given_power_law_profile(r, r0, sigma0, beta, a0=1.):
    # Compute the total integrated mass in solids from separation 'r0' (AU) to 'r' (AU) given a power-law profile, sigma(a) = sigma0*(a/a0)^beta, where the fitted parameters are the normalization (at 'a0') 'sigma0' (g/cm^2) and slope 'beta'
    # Will convert the total integrated mass in solids to units of Earth masses
    assert 0 <= r0 < r # r0 must be nonzero to avoid infinite mass for beta <= -2
    assert 0 < sigma0
    assert 0 < a0
    if beta == -2:
        M_r = 2.*np.pi*sigma0*((a0*gen.AU)**2.)*np.log(r/r0) # total mass in grams
    else: # if beta != -2
        M_r = (2.*np.pi*sigma0/(2.+beta))*((r/a0)**(2.+beta) - (r0/a0)**(2.+beta)) *((a0*gen.AU)**2.) # total mass in grams
    M_r = M_r/(1e3*gen.Mearth) # convert total mass to Earth masses
    return M_r
