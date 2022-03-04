# To import required modules:
import numpy as np
import os
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
from scipy.stats import truncnorm
from scipy.optimize import curve_fit # for fitting functions

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),'SysSim_Plotting/src')) # TODO: update when those files get made into a package

import functions_general as gen





##### Mass-radius functions:

def mass_given_radius_density(R, ρ):
    # Compute mass M (Earth masses) given radius R (Earth radii) and mean density ρ (g/cm^3)
    return ρ * (4.*np.pi/3.) * (R*gen.Rearth)**3. * (1./(gen.Mearth*1e3))

def density_given_mass_radius(M, R):
    # Compute mean density ρ (g/cm^3) given mass M (Earth masses) and radius R (Earth radii)
    return (M*gen.Mearth*1e3) / ((4.*np.pi/3.)*(R*gen.Rearth)**3.)



#data_path = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/SysSimExClusters/src/mr_model/MRpredict_table_weights3025_R1001_Q1001.txt'
data_path = 'C:/Users/HeYMa/Documents/GradSchool/Research/SysSimExClusters/src/mr_model/MRpredict_table_weights3025_R1001_Q1001.txt'
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

#data_path = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/SysSimExClusters/src/mr_model/MR_earthlike_rocky.txt'
data_path = 'C:/Users/HeYMa/Documents/GradSchool/Research/SysSimExClusters/src/mr_model/MR_earthlike_rocky.txt'
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

def solid_surface_density_S2014(M, R, a, Mstar=1.):
    # Compute the solid surface density (g/cm^2) using the Schlichting (2014) prescription (delta_a = 2^(3/2)*a((a*M)/(R*Mstar))^(1/2), given a planet mass M (M_earth), planet radius R (R_earth), semi-major axis a (AU), and stellar mass Mstar (M_sun)
    delta_a = 2.**(3./2.)*a*np.sqrt(((a*gen.AU)/(R*gen.Rearth))*((M*gen.Mearth)/(Mstar*gen.Msun))) # AU
    return solid_surface_density(M, a, delta_a)

def solid_surface_density_nHill(M, a, Mstar=1., n=10.):
    # Compute the solid surface density (g/cm^2) using a number of Hill radii for the feeding zone width (delta_a = n*R_Hill), given a planet mass M (M_earth), semi-major axis a (AU), and stellar mass Mstar (M_sun)
    delta_a = n*a*((M*gen.Mearth)/(3.*Mstar*gen.Msun))**(1./3.) # AU
    return solid_surface_density(M, a, delta_a)

def solid_surface_density_system_RC2014(M_sys, a_sys):
    # Compute the solid surface density (g/cm^2) of planets in a multiplanet system using the Raymond & Cossou (2014) prescription (neighboring planets' feeding zones separated by the geometric means of their semi-major axes)
    n_pl = len(M_sys)
    assert n_pl == len(a_sys) > 1
    a_bounds_sys = np.zeros(n_pl+1) # bounds in semi-major axis for each planet
    a_bounds_sys[1:-1] = np.sqrt(a_sys[:-1]*a_sys[1:]) # geometric means between planets
    a_bounds_sys[0] = a_sys[0]*np.sqrt(a_sys[0]/a_sys[1]) # same ratio for upper bound to a_sys[1] as a_sys[1] to lower bound
    a_bounds_sys[-1] = a_sys[-1]*np.sqrt(a_sys[-1]/a_sys[-2]) # same ratio for upper bound to a_sys[-1] as a_sys[-1] to lower bound
    delta_a_sys = np.diff(a_bounds_sys)
    return solid_surface_density(M_sys, a_sys, delta_a_sys)

def solid_surface_density_CL2013_given_physical_catalog(sssp_per_sys):
    # Compute the solid surface density (g/cm^2) using the Chiang & Laughlin (2013) prescription, for each planet in a given physical catalog
    # Returns an array of solid surface densities and semi-major axes
    a_all = sssp_per_sys['a_all'][sssp_per_sys['a_all'] > 0]
    sigma_all = solid_surface_density_CL2013(sssp_per_sys['mass_all'], sssp_per_sys['a_all'])[sssp_per_sys['a_all'] > 0]
    return sigma_all, a_all

def solid_surface_density_S2014_given_physical_catalog(sssp_per_sys, sssp):
    # Compute the solid surface density (g/cm^2) using the Schlichting (2014) prescription, for each planet in a given physical catalog
    # Returns an array of solid surface densities and semi-major axes
    a_all = sssp_per_sys['a_all'][sssp_per_sys['a_all'] > 0]
    sigma_all = solid_surface_density_S2014(sssp_per_sys['mass_all'], sssp_per_sys['radii_all'], sssp_per_sys['a_all'], Mstar=sssp['Mstar_all'][:,None])[sssp_per_sys['a_all'] > 0]
    return sigma_all, a_all

def solid_surface_density_nHill_given_physical_catalog(sssp_per_sys, sssp, n=10.):
    # Compute the solid surface density (g/cm^2) using a number of Hill radii for the feeding zone width, for each planet in a given physical catalog
    # 'n' is the number of Hill radii for  the feeding zone width of each planet
    # Returns an array of solid surface densities and semi-major axes
    a_all = sssp_per_sys['a_all'][sssp_per_sys['a_all'] > 0]
    sigma_all = solid_surface_density_nHill(sssp_per_sys['mass_all'], sssp_per_sys['a_all'], Mstar=sssp['Mstar_all'][:,None], n=n)[sssp_per_sys['a_all'] > 0]
    return sigma_all, a_all

def solid_surface_density_RC2014_given_physical_catalog(sssp_per_sys):
    # Compute the solid surface density (g/cm^2) using the Raymond & Cossou (2014) prescription, for each planet in each multi-planet system in a given physical catalog
    # Returns an array of solid surface densities and semi-major axes
    mult_all = sssp_per_sys['Mtot_all']
    a_all_2p = []
    mult_all_2p = []
    sigma_all_2p = []
    for i in np.arange(len(mult_all))[mult_all > 1]: # only consider multi-planet systems
        a_sys = sssp_per_sys['a_all'][i]
        M_sys = sssp_per_sys['mass_all'][i][a_sys > 0]
        a_sys = a_sys[a_sys > 0]
        a_all_2p += list(a_sys)
        mult_all_2p += [len(a_sys)]*len(a_sys)
        sigma_all_2p += list(solid_surface_density_system_RC2014(M_sys, a_sys))
    a_all_2p = np.array(a_all_2p)
    mult_all_2p = np.array(mult_all_2p)
    sigma_all_2p = np.array(sigma_all_2p)
    return sigma_all_2p, a_all_2p, mult_all_2p

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



##### Functions to fit a power-law to the MMEN (sigma = sigma0 * (a/a0)^beta <==> log(sigma) = log(sigma0) + beta*log(a/a0)):

f_linear = lambda x, p0, p1: p0 + p1*x

def fit_power_law_MMEN(a_array, sigma_array, a0=1., p0=1., p1=-1.5):
    # Fit power-law parameters (normalization 'sigma0' (at separation 'a0') and slope 'beta') given an array of separations 'a_array' (AU) and solid surface densities 'sigma_array' (g/cm^2)
    # Optional parameters: separation for the normalization 'a0' (AU; i.e. sigma0 = sigma(a0)), initial guess for log(sigma0) 'p0', and initial guess for beta 'p1'
    mmen_fit = curve_fit(f_linear, np.log10(a_array/a0), np.log10(sigma_array), [p0, p1])[0]
    sigma0, beta = 10.**(mmen_fit[0]), mmen_fit[1]
    return sigma0, beta

def fit_power_law_MMEN_per_system_observed(sss_per_sys, solid_surface_density_prescription=solid_surface_density_system_RC2014, a0=1., p0=1., p1=-1.5):
    # Compute solid surface densities and fit power-law parameters to each multi-planet system in an observed catalog
    # WARNING: This function currently only works for 'solid_surface_density_prescription=solid_surface_density_system_RC2014'
    fit_per_sys_dict = {'m_obs':[], 'Mstar_obs':[], 'sigma0':[], 'beta':[]} # 'm_obs' is number of observed planets
    a_obs_per_sys = gen.a_from_P(sss_per_sys['P_obs'], sss_per_sys['Mstar_obs'][:,None])
    for i,a_sys in enumerate(a_obs_per_sys):
        if np.sum(a_sys > 0) > 1:
            #print(i)
            M_sys = generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec(sss_per_sys['radii_obs'][i][a_sys > 0])
            a_sys = a_sys[a_sys > 0]
            sigma_obs_sys = solid_surface_density_prescription(M_sys, a_sys)
            sigma0, beta = fit_power_law_MMEN(a_sys, sigma_obs_sys, a0=a0, p0=p0, p1=p1)
            fit_per_sys_dict['m_obs'].append(len(a_sys))
            fit_per_sys_dict['Mstar_obs'].append(sss_per_sys['Mstar_obs'][i])
            fit_per_sys_dict['sigma0'].append(sigma0)
            fit_per_sys_dict['beta'].append(beta)
    
    fit_per_sys_dict['m_obs'] = np.array(fit_per_sys_dict['m_obs'])
    fit_per_sys_dict['Mstar_obs'] = np.array(fit_per_sys_dict['Mstar_obs'])
    fit_per_sys_dict['sigma0'] = np.array(fit_per_sys_dict['sigma0'])
    fit_per_sys_dict['beta'] = np.array(fit_per_sys_dict['beta'])
    return fit_per_sys_dict

def fit_power_law_MMEN_per_system_physical(sssp_per_sys, solid_surface_density_prescription=solid_surface_density_system_RC2014, a0=1., p0=1., p1=-1.5):
    # Compute solid surface densities and fit power-law parameters to each multi-planet system in a physical catalog
    # WARNING: This function currently only works for 'solid_surface_density_prescription=solid_surface_density_system_RC2014'
    fit_per_sys_dict = {'n_pl':[], 'sigma0':[], 'beta':[]} # 'n_pl' is number of planets in each system
    for i,a_sys in enumerate(sssp_per_sys['a_all']):
        if np.sum(a_sys > 0) > 1:
            M_sys = sssp_per_sys['mass_all'][i][a_sys > 0]
            a_sys = a_sys[a_sys > 0]
            sigma_sys = solid_surface_density_prescription(M_sys, a_sys)
            sigma0, beta = fit_power_law_MMEN(a_sys, sigma_sys, a0=a0, p0=p0, p1=p1)
            fit_per_sys_dict['n_pl'].append(len(a_sys))
            fit_per_sys_dict['sigma0'].append(sigma0)
            fit_per_sys_dict['beta'].append(beta)
    
    fit_per_sys_dict['n_pl'] = np.array(fit_per_sys_dict['n_pl'])
    fit_per_sys_dict['sigma0'] = np.array(fit_per_sys_dict['sigma0'])
    fit_per_sys_dict['beta'] = np.array(fit_per_sys_dict['beta'])
    return fit_per_sys_dict



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
