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

import syssimpyplots
import syssimpyplots.general as gen
#####data_path = os.path.join(syssimpyplots.__path__[0], 'data') # some data files are part of the SysSimPyPlots package, but this breaks the ReadtheDocs builds since autodoc_mock_imports cannot read this path for some reason (IndexError implying 'syssimpyplots.__path__' is an empty list)
data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data') # including some data files as part of this package (some repeated/copied from SysSimPyPlots)




# Solar system planet properties:

MeVeEa_masses = np.array([0.0553, 0.815, 1.]) # masses of Mercury, Venus, and Earth, in Earth masses
MeVeEa_radii = np.array([0.383, 0.949, 1.]) # radii of Mercury, Venus, and Earth, in Earth radii
MeVeEa_a = np.array([0.387, 0.723, 1.]) # semi-major axes of Mercury, Venus, and Earth, in AU





# Mass-radius model and functions:

logMR_Ning2018_table = np.genfromtxt(os.path.join(data_path, 'MRpredict_table_weights3025_R1001_Q1001.txt'), delimiter=',', skip_header=2, names=True) # first column is array of log_R values

#table_array = logMR_Ning2018_table.view(np.float64).reshape(np.shape(logMR_Ning2018_table) + (-1,))[:,1:]
table_array = logMR_Ning2018_table.view(np.float64).reshape((len(logMR_Ning2018_table),-1))[:,1:]
log_R_table = logMR_Ning2018_table['log_R']
qtls_table = np.linspace(0,1,1001)
logMR_table_interp = RectBivariateSpline(log_R_table, qtls_table, table_array)

def generate_planet_mass_from_radius_Ning2018_table(R):
    """
    Draw a planet mass from the Ning et al. (2018) mass-radius model interpolated on a precomputed table.

    Parameters
    ----------
    R : float
        The planet radius (Earth radii).

    Returns
    -------
    The planet mass (Earth masses).


    Note
    ----
    Requires a globally defined interpolation object, ``logMR_table_interp`` (defined in the same module).
    """
    logR = np.log10(R)
    q = np.random.random() # drawn quantile
    logM = logMR_table_interp(logR, q)[0][0]
    return 10.**logM # planet mass in Earth masses

MR_earthlike_rocky = np.genfromtxt(os.path.join(data_path, 'MR_earthlike_rocky.txt'), delimiter='\t', skip_header=2, names=('mass','radius'))

M_earthlike_rocky_interp = interp1d(MR_earthlike_rocky['radius'], MR_earthlike_rocky['mass'])

radius_switch = 1.472
σ_logM_at_radius_switch = 0.3 # log10(M/M_earth); 0.3 corresponds to about a factor of 2, and appears to be close to the std of the distribution at R=1.472 R_earth for the NWG-2018 model
σ_logM_at_radius_min = 0.04 # log10(M/M_earth); 0.04 corresponds to about a factor of 10%

def σ_logM_linear_given_radius(R, σ_logM_r1=σ_logM_at_radius_min, σ_logM_r2=σ_logM_at_radius_switch, r1=0.5, r2=radius_switch):
    """
    Compute the standard deviation in log planet mass at a given planet radius based on a linear relation.

    Parameters
    ----------
    R : float
        The planet radius (Earth radii).
    σ_logM_r1 : float, default=σ_logM_at_radius_min
        The standard deviation in log(mass) (Earth masses) at radius `r1`.
    σ_logM_r2 : float, default=σ_logM_at_radius_switch
        The standard deviation in log(mass) (Earth masses) at radius `r2`.
    r1 : float, default=0.5
        The planet radius (Earth radii) corresponding to `σ_logM_r1`.
    r2 : float, default=radius_switch
        The planet radius (Earth radii) corresponding to `σ_logM_r2`.

    Returns
    -------
    The standard deviation in log(mass) (Earth masses) at the given radius.
    """
    return ((σ_logM_at_radius_switch - σ_logM_at_radius_min) / (r2 - r1))*(R - r1) + σ_logM_at_radius_min

def generate_planet_mass_from_radius_lognormal_mass_around_earthlike_rocky(R, ρ_min=1., ρ_max=100.):
    """
    Draw a planet mass from a lognormal distribution centered around the Earth-like rocky model from Li Zeng (https://www.cfa.harvard.edu/~lzeng/tables/massradiusEarthlikeRocky.txt).

    Parameters
    ----------
    R : float
        The planet radius (Earth radii).
    ρ_min : float, default=1.
        The minimum planet density (g/cm^3) allowed.
    ρ_max : float, default=100.
        The maximum planet density (g/cm^3) allowed.

    Returns
    -------
    The planet mass (Earth masses).


    Note
    ----
    Requires a globally defined interpolation object, ``M_earthlike_rocky_interp`` (defined in the same module).
    """
    M = M_earthlike_rocky_interp(R)
    assert M > 0
    μ_logM = np.log10(M)
    σ_logM = σ_logM_linear_given_radius(R)
    logM_min = np.log10(gen.M_from_R_rho(R, ρ_min))
    logM_max = np.log10(gen.M_from_R_rho(R, ρ_max))
    a, b = (logM_min - μ_logM)/σ_logM, (logM_max - μ_logM)/σ_logM
    logM = truncnorm.rvs(a, b, loc=μ_logM, scale=σ_logM, size=1)[0] # draw from truncated normal distribution for log10(M/M_earth)
    return 10.**logM # planet mass in Earth masses

def generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below(R, R_switch=radius_switch, ρ_min=1., ρ_max=100.):
    """
    Draw a planet mass from a combined model (the Ning et al. 2018 model for large radii, and the lognormal distribution centered around the Earth-like rocky model for small radii).

    Parameters
    ----------
    R : float
        The planet radius (Earth radii).
    R_switch : float, default=radius_switch
        The transition radius (Earth radii) defining whether to draw from the Ning et al. 2018 model (above this radius) or the lognormal-around Earth-like rocky model (below this radius).
    ρ_min : float, default=1.
        The minimum planet density (g/cm^3) allowed.
    ρ_max : float, default=100.
        The maximum planet density (g/cm^3) allowed.

    Returns
    -------
    The planet mass (Earth masses).
    """
    #assert R > 0
    if R > R_switch:
        M = generate_planet_mass_from_radius_Ning2018_table(R)
    elif R <= 0:
        M = 0.
    else:
        M = generate_planet_mass_from_radius_lognormal_mass_around_earthlike_rocky(R, ρ_min=ρ_min, ρ_max=ρ_max)
    return M

generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec = np.vectorize(generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below) # vectorized form of function





# Functions to compute various formulations of the solid surface density/minimum mass extrasolar nebular (MMEN):

def feeding_zone_S2014(M, R, a, Mstar=1.):
    """
    Compute the feeding zone width of a planet using the Schlichting (2014) prescription::

        delta_a = 2^(3/2)*a*((a*M)/(R*Mstar))^(1/2).


    Parameters
    ----------
    M : float or array[float]
        The planet mass (Earth masses).
    R : float or array[float]
        The planet radius (Earth radii).
    a : float or array[float]
        The semi-major axis (AU).
    Mstar : float or array[float], default=1.
        The stellar mass (solar masses).

    Returns
    -------
    delta_a : float or array[float]
        The feeding zone width (AU) of the planet.
    """
    delta_a = 2.**(3./2.)*a*np.sqrt(((a*gen.AU)/(R*gen.Rearth))*((M*gen.Mearth)/(Mstar*gen.Msun))) # AU
    return delta_a

def feeding_zone_nHill(M, a, Mstar=1., n=10.):
    """
    Compute the feeding zone width of a planet using a number of Hill radii::

        delta_a = n*R_Hill = n*a*(M/(3*Mstar))^(1/3).


    Parameters
    ----------
    M : float or array[float]
        The planet mass (Earth masses).
    a : float or array[float]
        The semi-major axis (AU).
    Mstar : float or array[float], default=1.
        The stellar mass (solar masses).
    n : float or array[float], default=10.
        The number of Hill radii to use as the feeding zone.

    Returns
    -------
    delta_a : float or array[float]
        The feeding zone width (AU) of the planet.
    """
    delta_a = n*a*((M*gen.Mearth)/(3.*Mstar*gen.Msun))**(1./3.) # AU
    return delta_a

def feeding_zone_RC2014(a_sys):
    """
    Compute the feeding zone widths of all the planets in a multi-planet system using the Raymond & Cossou (2014) prescription.

    Uses the geometric means of the neighboring planets' semi-major axes as the boundaries of their feeding zones.

    Note
    ----
    Assumes the same ratio for the inner edge of the innermost planet as its outer edge, and the same ratio for the outer edge of the outermost planet as its inner edge.

    Parameters
    ----------
    a_sys : array[float]
        The semi-major axes (AU) of all the planets.

    Returns
    -------
    delta_a_sys : array[float]
        The feeding zone widths (AU) of all the planets.
    a_bounds_sys : array[float]
        The boundaries (AU) of the feeding zones for the planets (length = n+1 where n is the number of planets). For example, `a_bounds_sys[i]` and `a_bounds_sys[i+1]` will be the inner and outer boundaries of the `i^th` planet.
    """
    a_bounds_sys = np.zeros(len(a_sys)+1) # bounds in semi-major axis for each planet
    a_bounds_sys[1:-1] = np.sqrt(a_sys[:-1]*a_sys[1:]) # geometric means between planets
    a_bounds_sys[0] = a_sys[0]*np.sqrt(a_sys[0]/a_sys[1]) # same ratio for upper bound to a_sys[1] as a_sys[1] to lower bound
    a_bounds_sys[-1] = a_sys[-1]*np.sqrt(a_sys[-1]/a_sys[-2]) # same ratio for upper bound to a_sys[-1] as a_sys[-1] to lower bound
    delta_a_sys = np.diff(a_bounds_sys)
    return delta_a_sys, a_bounds_sys



def solid_surface_density(M, a, delta_a):
    """
    Compute the solid surface density associated with a planet.

    This divides the mass of the planet by the surface area of an annulus of a given width, centered at its semi-major axis.

    Parameters
    ----------
    M : float or array[float]
        The planet mass (Earth masses).
    a : float or array[float]
        The semi-major axis (AU).
    delta_a : float or array[float]
        The feeding zone width (AU).

    Returns
    -------
    sigma_solid : float or array[float]
        The solid surface density (g/cm^2) local to the planet.
    """
    sigma_solid = (M*gen.Mearth*1e3)/(2.*np.pi*(a*gen.AU)*(delta_a*gen.AU))
    return sigma_solid

def solid_surface_density_CL2013(M, a):
    """
    Compute the solid surface density of a planet using the Chiang & Laughlin (2013) prescription for the feeding zone width (set equal to the semi-major axis; delta_a = a).

    Parameters
    ----------
    M : float or array[float]
        The planet mass (Earth masses).
    a : float or array[float]
        The semi-major axis (AU).

    Returns
    -------
    The solid surface density (g/cm^2) local to the planet.
    """
    return solid_surface_density(M, a, a)

def solid_surface_density_S2014(M, R, a, Mstar=1.):
    """
    Compute the solid surface density of a planet using the Schlichting (2014) prescription for the feeding zone width (see :py:func:`syssimpymmen.mmen.feeding_zone_S2014`).

    Parameters
    ----------
    M : float or array[float]
        The planet mass (Earth masses).
    R : float or array[float]
        The planet radius (Earth radii).
    a : float or array[float]
        The semi-major axis (AU).
    Mstar : float or array[float], default=1.
        The stellar mass (solar masses).

    Returns
    -------
    The solid surface density (g/cm^2) local to the planet.
    """
    delta_a = feeding_zone_S2014(M, R, a, Mstar=Mstar)
    return solid_surface_density(M, a, delta_a)

def solid_surface_density_nHill(M, a, Mstar=1., n=10.):
    """
    Compute the solid surface density of a planet using a number of Hill radii for the feeding zone width (see :py:func:`syssimpymmen.mmen.feeding_zone_nHill`).

    Parameters
    ----------
    M : float or array[float]
        The planet mass (Earth masses).
    a : float or array[float]
        The semi-major axis (AU).
    Mstar : float or array[float], default=1.
        The stellar mass (solar masses).
    n : float or array[float], default=10.
        The number of Hill radii to use as the feeding zone.

    Returns
    -------
    The solid surface density (g/cm^2) local to the planet.
    """
    delta_a = feeding_zone_nHill(M, a, Mstar=Mstar, n=n)
    return solid_surface_density(M, a, delta_a)

def solid_surface_density_system_RC2014(M_sys, a_sys):
    """
    Compute the solid surface densities of all planets in a multi-planet system using the Raymond & Cossou (2014) prescription for their feeding zone widths (see :py:func:`syssimpymmen.mmen.feeding_zone_RC2014`).

    Parameters
    ----------
    M_sys : array[float]
        The planet masses (Earth masses).
    a_sys: array[float]
        The semi-major axes (AU) of the planets.

    Returns
    -------
    The solid surface densities (g/cm^2) local to the planets.
    """
    n_pl = len(M_sys)
    assert n_pl == len(a_sys) > 1
    delta_a_sys, _ = feeding_zone_RC2014(a_sys)
    return solid_surface_density(M_sys, a_sys, delta_a_sys)

def solid_surface_density_prescription(M, R, a, Mstar=1., n=10., prescription='CL2013'):
    """
    Compute the solid surface density of a planet or planetary system using a given prescription for the feeding zone width.

    Wrapper function that calls the appropriate function based on the `prescription` string.

    Parameters
    ----------
    M : float or array[float]
        The planet masses (Earth masses).
    R : float or array[float]
        The planet radii (Earth radii). Only used for the 'S2014' prescription.
    a : float or array[float]
        The semi-major axes (AU) of the planets.
    Mstar : float or array[float], default=1.
        The stellar mass or masses (solar masses). Only used for the 'S2014' or 'nHill' prescriptions.
    n : float, default=10.
        The number of Hill radii to use as the feeding zone. Only used for the 'nHill' prescription.
    prescription : {'CL2013', 'S2014', 'nHill', 'RC2014'}, default='CL2013'
        The string indicating the prescription to use for computing the feeding zone widths.

    Returns
    -------
    The solid surface densities (g/cm^2) local to the planets.


    Note
    ----
    If the prescription is 'RC2014', the input arrays (i.e. `M` and `a`) must all correspond to planets in the same system!
    """
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
    """
    Compute the solid surface densities for all planets in a physical catalog, using the Chiang & Laughlin (2013) prescription for the feeding zone widths.

    Parameters
    ----------
    sssp_per_sys : dict
        The dictionary containing the planetary and stellar properties for each system in a physical catalog (2-d and 1-d arrays).
    max_core_mass : float, default=10.
        The maximum allowed (solid) core mass (Earth masses).

    Returns
    -------
    sigma_all : array[float]
        The solid surface densities (g/cm^2) of all the planets.
    a_all : array[float]
        The semi-major axes (AU) of all the planets.
    """
    a_all = sssp_per_sys['a_all'][sssp_per_sys['a_all'] > 0]
    core_mass_all = np.copy(sssp_per_sys['mass_all'][sssp_per_sys['a_all'] > 0])
    core_mass_all[core_mass_all > max_core_mass] = max_core_mass
    sigma_all = solid_surface_density_CL2013(core_mass_all, a_all)
    return sigma_all, a_all

def solid_surface_density_S2014_given_physical_catalog(sssp_per_sys, sssp, max_core_mass=10.):
    """
    Compute the solid surface densities for all planets in a physical catalog, using the Schlichting (2014) prescription for the feeding zone widths.

    Parameters
    ----------
    sssp_per_sys : dict
        The dictionary containing the planetary and stellar properties for each system in a physical catalog (2-d and 1-d arrays).
    sssp : dict
        The dictionary containing the planetary and stellar properties of all planets in a physical catalog (1-d arrays).
    max_core_mass : float, default=10.
        The maximum allowed (solid) core mass (Earth masses).

    Returns
    -------
    sigma_all : array[float]
        The solid surface densities (g/cm^2) of all the planets.
    a_all : array[float]
        The semi-major axes (AU) of all the planets.
    """
    a_all = sssp_per_sys['a_all'][sssp_per_sys['a_all'] > 0]
    core_mass_all = np.copy(sssp_per_sys['mass_all'])
    core_mass_all[core_mass_all > max_core_mass] = max_core_mass
    sigma_all = solid_surface_density_S2014(core_mass_all, sssp_per_sys['radii_all'], sssp_per_sys['a_all'], Mstar=sssp['Mstar_all'][:,None])[sssp_per_sys['a_all'] > 0]
    return sigma_all, a_all

def solid_surface_density_nHill_given_physical_catalog(sssp_per_sys, sssp, max_core_mass=10., n=10.):
    """
    Compute the solid surface densities for all planets in a physical catalog, using a number of Hill radii for the feeding zone widths.

    Parameters
    ----------
    sssp_per_sys : dict
        The dictionary containing the planetary and stellar properties for each system in a physical catalog (2-d and 1-d arrays).
    sssp : dict
        The dictionary containing the planetary and stellar properties of all planets in a physical catalog (1-d arrays).
    max_core_mass : float, default=10.
        The maximum allowed (solid) core mass (Earth masses).
    n : float, default=10.
        The number of Hill radii to use as the feeding zone.

    Returns
    -------
    sigma_all : array[float]
        The solid surface densities (g/cm^2) of all the planets.
    a_all : array[float]
        The semi-major axes (AU) of all the planets.
    """
    a_all = sssp_per_sys['a_all'][sssp_per_sys['a_all'] > 0]
    core_mass_all = np.copy(sssp_per_sys['mass_all'])
    core_mass_all[core_mass_all > max_core_mass] = max_core_mass
    sigma_all = solid_surface_density_nHill(core_mass_all, sssp_per_sys['a_all'], Mstar=sssp['Mstar_all'][:,None], n=n)[sssp_per_sys['a_all'] > 0]
    return sigma_all, a_all

def solid_surface_density_RC2014_given_physical_catalog(sssp_per_sys, max_core_mass=10.):
    """
    Compute the solid surface densities for all planets in multi-planet systems in a physical catalog, using the Raymond & Cossou (2014) prescription for the feeding zone widths.

    Parameters
    ----------
    sssp_per_sys : dict
        The dictionary containing the planetary and stellar properties for each system in a physical catalog (2-d and 1-d arrays).
    max_core_mass : float, default=10.
        The maximum allowed (solid) core mass (Earth masses).

    Returns
    -------
    sigma_all_2p : array[float]
        The solid surface densities (g/cm^2) of the planets in multi-planet systems.
    a_all_2p : array[float]
        The semi-major axes (AU) of the planets in multi-planet systems.
    mult_all_2p : array[float]
        The multiplicities of the multi-planet systems each planet belongs to.
    """
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
    """
    Compute the solid surface densities for all planets in an observed catalog, using the Chiang & Laughlin (2013) prescription for the feeding zone widths.

    Parameters
    ----------
    sss_per_sys : dict
        The dictionary containing the planetary and stellar properties for each system in a physical catalog (2-d and 1-d arrays).
    max_core_mass : float, default=10.
        The maximum allowed (solid) core mass (Earth masses).

    Returns
    -------
    sigma_obs : array[float]
        The solid surface densities (g/cm^2) of all the observed planets.
    core_mass_obs : array[float]
        The core masses (Earth masses) of all the observed planets. These are the total masses of the planets computed from their observed radii using a mass-radius relation, capped at `max_core_mass`.
    a_obs : array[float]
        The semi-major axes (AU) of all the observed planets.
    """
    a_obs_per_sys = gen.a_from_P(sss_per_sys['P_obs'], sss_per_sys['Mstar_obs'][:,None])
    a_obs = a_obs_per_sys[sss_per_sys['P_obs'] > 0]
    radii_obs = sss_per_sys['radii_obs'][sss_per_sys['P_obs'] > 0]
    core_mass_obs = generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec(radii_obs)
    core_mass_obs[core_mass_obs > max_core_mass] = max_core_mass
    sigma_obs = solid_surface_density_CL2013(core_mass_obs, a_obs)
    return sigma_obs, core_mass_obs, a_obs

def solid_surface_density_S2014_given_observed_catalog(sss_per_sys, max_core_mass=10.):
    """
    Compute the solid surface densities for all planets in an observed catalog, using the Schlichting (2014) prescription for the feeding zone widths.

    Parameters
    ----------
    sss_per_sys : dict
        The dictionary containing the planetary and stellar properties for each system in a physical catalog (2-d and 1-d arrays).
    max_core_mass : float, default=10.
        The maximum allowed (solid) core mass (Earth masses).

    Returns
    -------
    sigma_obs : array[float]
        The solid surface densities (g/cm^2) of all the observed planets.
    core_mass_obs : array[float]
        The core masses (Earth masses) of all the observed planets. These are the total masses of the planets computed from their observed radii using a mass-radius relation, capped at `max_core_mass`.
    a_obs : array[float]
        The semi-major axes (AU) of all the observed planets.
    """
    Mstar_obs = np.repeat(sss_per_sys['Mstar_obs'][:,None], np.shape(sss_per_sys['P_obs'])[1], axis=1)[sss_per_sys['P_obs'] > 0] # flattened array of stellar masses repeated for each planet
    a_obs_per_sys = gen.a_from_P(sss_per_sys['P_obs'], sss_per_sys['Mstar_obs'][:,None])
    a_obs = a_obs_per_sys[sss_per_sys['P_obs'] > 0]
    radii_obs = sss_per_sys['radii_obs'][sss_per_sys['P_obs'] > 0]
    core_mass_obs = generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec(radii_obs)
    core_mass_obs[core_mass_obs > max_core_mass] = max_core_mass
    sigma_obs = solid_surface_density_S2014(core_mass_obs, radii_obs, a_obs, Mstar=Mstar_obs)
    return sigma_obs, core_mass_obs, a_obs

def solid_surface_density_nHill_given_observed_catalog(sss_per_sys, max_core_mass=10., n=10.):
    """
    Compute the solid surface densities for all planets in an observed catalog, using a number of Hill radii for the feeding zone widths.

    Parameters
    ----------
    sss_per_sys : dict
        The dictionary containing the planetary and stellar properties for each system in a physical catalog (2-d and 1-d arrays).
    max_core_mass : float, default=10.
        The maximum allowed (solid) core mass (Earth masses).
    n : float, default=10.
        The number of Hill radii to use as the feeding zone.

    Returns
    -------
    sigma_obs : array[float]
        The solid surface densities (g/cm^2) of all the observed planets.
    core_mass_obs : array[float]
        The core masses (Earth masses) of all the observed planets. These are the total masses of the planets computed from their observed radii using a mass-radius relation, capped at `max_core_mass`.
    a_obs : array[float]
        The semi-major axes (AU) of all the observed planets.
    """
    Mstar_obs = np.repeat(sss_per_sys['Mstar_obs'][:,None], np.shape(sss_per_sys['P_obs'])[1], axis=1)[sss_per_sys['P_obs'] > 0] # flattened array of stellar masses repeated for each planet
    a_obs_per_sys = gen.a_from_P(sss_per_sys['P_obs'], sss_per_sys['Mstar_obs'][:,None])
    a_obs = a_obs_per_sys[sss_per_sys['P_obs'] > 0]
    radii_obs = sss_per_sys['radii_obs'][sss_per_sys['P_obs'] > 0]
    core_mass_obs = generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec(radii_obs)
    core_mass_obs[core_mass_obs > max_core_mass] = max_core_mass
    sigma_obs = solid_surface_density_nHill(core_mass_obs, a_obs, Mstar=Mstar_obs, n=n)
    return sigma_obs, core_mass_obs, a_obs

def solid_surface_density_RC2014_given_observed_catalog(sss_per_sys, max_core_mass=10.):
    """
    Compute the solid surface densities for all planets in observed multi-planet systems in an observed catalog, using the Raymond & Cossou (2014) prescription for the feeding zone widths.

    Parameters
    ----------
    sss_per_sys : dict
        The dictionary containing the planetary and stellar properties for each system in a physical catalog (2-d and 1-d arrays).
    max_core_mass : float, default=10.
        The maximum allowed (solid) core mass (Earth masses).

    Returns
    -------
    sigma_obs_2p : array[float]
        The solid surface densities (g/cm^2) of the observed planets in multi-planet systems.
    core_mass_obs_2p : array[float]
        The core masses (Earth masses) of the observed planets in multi-planet systems. These are the total masses of the planets computed from their observed radii using a mass-radius relation, capped at `max_core_mass`.
    a_obs_2p : array[float]
        The semi-major axes (AU) of the observed planets in multi-planet systems.
    mult_obs_2p : array[float]
        The multiplicities of the observed multi-planet systems each planet belongs to.
    """
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





# Functions to fit a power-law to the MMEN (sigma = sigma0 * (a/a0)^beta <==> log(sigma) = log(sigma0) + beta*log(a/a0)):

def MMSN(a, F=1., Zrel=0.33):
    """
    Compute the solid surface density of the minimum mass solar nebula (MMSN) at a given separation, as defined by Eq. 2 in Chiang & Youdin (2010) (https://arxiv.org/pdf/0909.2652.pdf).

    Note
    ----
    The normalization is such that the default values of `F=1` and `Zrel=0.33` give 1 Earth mass of solids in an annulus centered on Earth's semi-major axis.

    Parameters
    ----------
    a : float or array[float]
        The semi-major axis (AU).
    F : float or array[float], default=1.
        A fudge factor.
    Zrel : float or array[float], default=0.33
        The relative enrichment in metallicity.

    Returns
    -------
    sigma_p : float or array[float]
        The solid surface density (g/cm^2) of the MMSN at the given separation `a`.
    """
    sigma_p = 33.*F*Zrel*(a**-1.5)
    return sigma_p

def MMEN_power_law(a, sigma0, beta, a0=1.):
    """
    Evaluate a power-law profile for the solid surface density of the minimum mass extrasolar nebula (MMEN) at a given separation.

    Given by `sigma(a) = sigma0*(a/a0)^beta`, where `sigma0` (g/cm^2) is the normalization at semi-major axis `a0` (AU), and `beta` is the power-law index.

    Parameters
    ----------
    a : float or array[float]
        The semi-major axis (AU) at which to evaluate the solid surface density.
    sigma0 : float
        The solid surface density normalization (g/cm^2) at separation `a0`.
    beta : float
        The power-law index.
    a0 : float, default=1.
        The normalization point for the separation (AU).

    Returns
    -------
    sigma_a : float or array[float]
        The solid surface density (g/cm^2) at the given separation `a`.
    """
    assert sigma0 > 0
    assert a0 > 0
    sigma_a = sigma0 * (a/a0)**beta
    return sigma_a

f_linear = lambda x, p0, p1: p0 + p1*x

def fit_power_law_MMEN(a_array, sigma_array, a0=1., p0=1., p1=-1.5):
    """
    Fit a power-law representing the minimum mass extrasolar nebula (MMEN) of a planetary system given the planets' semi-major axes and solid surface densities.

    Parameters
    ----------
    a_array : array[float]
        The semi-major axes (AU) of the planets.
    sigma_array : array[float]
        The solid surface densities (g/cm^2) of the planets.
    a0 : float, default=1.
        The normalization point for the separation (AU).
    p0 : float, default=1.
        The initial guess for the normalization parameter, 'log(sigma0)'.
    p1 : float, default=-1.5
        The initial guess for the power-law index parameter, 'beta'.

    Returns
    -------
    sigma0 : float
        The best-fit value for the solid surface density normalization (g/cm^2). Unlike the initial guess (`p0`), this value is unlogged.
    beta : float
        The best-fit value for the power-law index.
    """
    assert len(a_array) == len(sigma_array) >= 2
    mmen_fit = curve_fit(f_linear, np.log10(a_array/a0), np.log10(sigma_array), [p0, p1])[0]
    sigma0, beta = 10.**(mmen_fit[0]), mmen_fit[1]
    return sigma0, beta

def fit_power_law_and_scale_up_MMEN(a_sys, sigma_sys, a0=1., p0=1., p1=-1.5):
    """
    Call :py:func:`syssimpymmen.mmen.fit_power_law_MMEN`, and then scale up the resulting power-law fit to be at/above all of the input solid surface densities given by `sigma_sys`.

    Note
    ----
    Should only be used for fitting MMEN to individual planetary systems; should NOT be used for fitting MMEN to an entire catalog or a collection of planets from different planetary systems.

    Parameters
    ----------
    a_sys : array[float]
        The semi-major axes (AU) of the planets in the system.
    sigma_sys : array[float]
        The solid surface densities (g/cm^2) of the planets in the system.
    a0 : float, default=1.
        The normalization point for the separation (AU).
    p0 : float, default=1.
        The initial guess for the normalization parameter, 'log(sigma0)'.
    p1 : float, default=-1.5
        The initial guess for the power-law index parameter, 'beta'.

    Returns
    -------
    sigma0 : float
        The best-fit value for the solid surface density normalization (g/cm^2), multiplied by a scale factor `scale_factor`. Unlike the initial guess (`p0`), this value is unlogged.
    beta : float
        The best-fit value for the power-law index.
    scale_factor : float
        The scale factor required to shift the power-law to be at/above all of the values in `sigma_sys`. This is always unity for systems with two planets, and greater than or equal to one in general.
    """
    sigma0, beta = fit_power_law_MMEN(a_sys, sigma_sys, a0=a0, p0=p0, p1=p1)

    if len(a_sys) > 2:
        sigma_fit_sys = MMEN_power_law(a_sys, sigma0, beta, a0=a0)
        sigma_fit_ratio_sys = sigma_sys/sigma_fit_sys
        scale_factor = np.max(sigma_fit_ratio_sys)
        sigma0 = sigma0 * scale_factor
    else: # for systems with 2 planets, the scale factor is not necessary
        scale_factor = 1.

    return sigma0, beta, scale_factor

def fit_power_law_MMEN_all_planets_observed(sss_per_sys, max_core_mass=10., prescription='CL2013', n=10., a0=1., p0=1., p1=-1.5):
    """
    Compute the solid surface densities and fit a power-law to them for all planets in an observed catalog.

    Parameters
    ----------
    sss_per_sys : dict
        The dictionary containing the planetary and stellar properties for each system in a physical catalog (2-d and 1-d arrays).
    max_core_mass : float, default=10.
        The maximum allowed (solid) core mass (Earth masses).
    prescription : {'CL2013', 'S2014', 'nHill', 'RC2014'}, default='CL2013'
        The string indicating the prescription to use for computing the feeding zone widths.
    n : float, default=10.
        The number of Hill radii to use for the feeding zones. Only used for the 'nHill' prescription.
    a0 : float, default=1.
        The normalization point for the separation (AU).
    p0 : float, default=1.
        The initial guess for the normalization parameter, 'log(sigma0)'.
    p1 : float, default=-1.5
        The initial guess for the power-law index parameter, 'beta'.

    Returns
    -------
    outputs_dict : dict
        A dictionary containing the solid surface densities, power-law fit parameters, and other properties computed from the observed catalog.


    The output dictionary contains the following fields:

    - `sigma_obs`: The solid surface densities (g/cm^2) of all the observed planets (1-d array).
    - `core_mass_obs`: The solid core masses (Earth masses) of all the observed planets (1-d array). These are the total masses of the planets computed from their observed radii using a mass-radius relation, capped at `max_core_mass`.
    - `a_obs`: The semi-major axes (AU) of all the observed planets (1-d array).
    - `sigma0`: The best-fit value for the solid surface density normalization (g/cm^2).
    - `beta`: The best-fit value for the power-law index.

    """
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
    """
    Compute the solid surface densities and fit a power-law to them for all planets in a physical catalog.

    Parameters
    ----------
    sssp_per_sys : dict
        The dictionary containing the planetary and stellar properties for each system in a physical catalog (2-d and 1-d arrays).
    sssp : dict
        The dictionary containing the planetary and stellar properties of all planets in a physical catalog (1-d arrays).
    max_core_mass : float, default=10.
        The maximum allowed (solid) core mass (Earth masses).
    prescription : {'CL2013', 'S2014', 'nHill', 'RC2014'}, default='CL2013'
        The string indicating the prescription to use for computing the feeding zone widths.
    n : float, default=10.
        The number of Hill radii to use for the feeding zones. Only used for the 'nHill' prescription.
    a0 : float, default=1.
        The normalization point for the separation (AU).
    p0 : float, default=1.
        The initial guess for the normalization parameter, 'log(sigma0)'.
    p1 : float, default=-1.5
        The initial guess for the power-law index parameter, 'beta'.

    Returns
    -------
    outputs_dict : dict
        A dictionary containing the solid surface densities, power-law fit parameters, and other properties computed from the observed catalog.


    The output dictionary contains the following fields:

    - `sigma_all`: The solid surface densities (g/cm^2) of all the planets (1-d array).
    - `a_all`: The semi-major axes (AU) of all the planets (1-d array).
    - `sigma0`: The best-fit value for the solid surface density normalization (g/cm^2).
    - `beta`: The best-fit value for the power-law index.

    """
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

def fit_power_law_MMEN_per_system_observed(sss_per_sys, n_mult_min=2, max_core_mass=10., prescription='CL2013', n=10., a0=1., p0=1., p1=-1.5, scale_up=False):
    """
    Fit power-laws to the solid surface densities of the planets *in each multi-planet system* in an observed catalog.

    Parameters
    ----------
    sss_per_sys : dict
        The dictionary containing the planetary and stellar properties for each system in a physical catalog (2-d and 1-d arrays).
    n_mult_min : int, default=2
        The minimum multiplicity to include.
    max_core_mass : float, default=10.
        The maximum allowed (solid) core mass (Earth masses).
    prescription : {'CL2013', 'S2014', 'nHill', 'RC2014'}, default='CL2013'
        The string indicating the prescription to use for computing the feeding zone widths.
    n : float, default=10.
        The number of Hill radii to use for the feeding zones. Only used for the 'nHill' prescription.
    a0 : float, default=1.
        The normalization point for the separation (AU).
    p0 : float, default=1.
        The initial guess for the normalization parameter, 'log(sigma0)'.
    p1 : float, default=-1.5
        The initial guess for the power-law index parameter, 'beta'.
    scale_up : bool, default=False
        Whether to scale up the normalization such that the power-law fit for each system is at/above the solid surface densities of every planet in the system.

    Returns
    -------
    fit_per_sys_dict : dict
        A dictionary containing the power-law fit parameters and other properties for each included system in the observed catalog.


    The output dictionary contains the following fields:

    - `m_obs`: The observed multiplicities of the included systems (greater than or equal to `n_mult_min`; 1-d array).
    - `Mstar_obs`: The stellar masses of the included system (1-d array).
    - `sigma0`: The best-fit values for the solid surface density normalizations (g/cm^2) of the included systems (1-d array). If `scale_factor=True`, these normalization values have already been multiplied by the scale factor of each system.
    - `scale_factor`: The scale factors required to increase the normalizations of each system such that the power-law fits are at/above the solid surface densities of every planet in the system (1-d array).
    - `beta`: The best-fit values for the power-law indices of the included systems (1-d array).

    """
    assert n_mult_min >= 2
    fit_per_sys_dict = {'m_obs':[], 'Mstar_obs':[], 'sigma0':[], 'scale_factor':[], 'beta':[]} # 'm_obs' is number of observed planets
    a_obs_per_sys = gen.a_from_P(sss_per_sys['P_obs'], sss_per_sys['Mstar_obs'][:,None])
    for i,a_sys in enumerate(a_obs_per_sys):
        if np.sum(a_sys > 0) >= n_mult_min:
            #print(i)
            Mstar = sss_per_sys['Mstar_obs'][i]
            R_sys = sss_per_sys['radii_obs'][i][a_sys > 0]
            core_mass_sys = generate_planet_mass_from_radius_Ning2018_table_above_lognormal_mass_earthlike_rocky_below_vec(R_sys)
            core_mass_sys[core_mass_sys > max_core_mass] = max_core_mass
            a_sys = a_sys[a_sys > 0]
            sigma_obs_sys = solid_surface_density_prescription(core_mass_sys, R_sys, a_sys, Mstar=Mstar, n=n, prescription=prescription)

            if scale_up:
                sigma0, beta, scale_factor_sigma0 = fit_power_law_and_scale_up_MMEN(a_sys, sigma_obs_sys, a0=a0, p0=p0, p1=p1)
            else:
                sigma0, beta = fit_power_law_MMEN(a_sys, sigma_obs_sys, a0=a0, p0=p0, p1=p1)
                scale_factor_sigma0 = 1.

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

def fit_power_law_MMEN_per_system_physical(sssp_per_sys, sssp, n_mult_min=2, max_core_mass=10., prescription='CL2013', n=10., a0=1., p0=1., p1=-1.5, scale_up=False, N_sys=10000):
    """
    Fit power-laws to the solid surface densities of the planets *in each multi-planet system* in a physical catalog.

    Parameters
    ----------
    sssp_per_sys : dict
        The dictionary containing the planetary and stellar properties for each system in a physical catalog (2-d and 1-d arrays).
    sssp : dict
        The dictionary containing the planetary and stellar properties of all planets in a physical catalog (1-d arrays).
    n_mult_min : int, default=2
        The minimum multiplicity to include.
    max_core_mass : float, default=10.
        The maximum allowed (solid) core mass (Earth masses).
    prescription : {'CL2013', 'S2014', 'nHill', 'RC2014'}, default='CL2013'
        The string indicating the prescription to use for computing the feeding zone widths.
    n : float, default=10.
        The number of Hill radii to use for the feeding zones. Only used for the 'nHill' prescription.
    a0 : float, default=1.
        The normalization point for the separation (AU).
    p0 : float, default=1.
        The initial guess for the normalization parameter, 'log(sigma0)'.
    p1 : float, default=-1.5
        The initial guess for the power-law index parameter, 'beta'.
    scale_up : bool, default=False
        Whether to scale up the normalization such that the power-law fit for each system is at/above the solid surface densities of every planet in the system.
    N_sys : int, default=10000
        The maximum number of systems to be included (to save time if there are too many systems in the physical catalog).

    Returns
    -------
    fit_per_sys_dict : dict
        A dictionary containing the power-law fit parameters and other properties for each included system in the physical catalog.


    The output dictionary contains the following fields:

    - `n_pl`: The multiplicities of the included systems (greater than or equal to `n_mult_min`; 1-d array).
    - `sigma0`: The best-fit values for the solid surface density normalizations (g/cm^2) of the included systems (1-d array). If `scale_factor=True`, these normalization values have already been multiplied by the scale factor of each system.
    - `scale_factor`: The scale factors required to increase the normalizations of each system such that the power-law fits are at/above the solid surface densities of every planet in the system (1-d array).
    - `beta`: The best-fit values for the power-law indices of the included systems (1-d array).

    """
    assert n_mult_min >= 2
    start = time.time()
    N_sys_tot = len(sssp_per_sys['a_all'])
    print('Fitting power-laws to the first %s systems (out of %s)...' % (min(N_sys,N_sys_tot), N_sys_tot))
    fit_per_sys_dict = {'n_pl':[], 'sigma0':[], 'scale_factor':[], 'beta':[]} # 'n_pl' is number of planets in each system
    for i,a_sys in enumerate(sssp_per_sys['a_all'][:N_sys]):
        if np.sum(a_sys > 0) >= n_mult_min:
            Mstar = sssp['Mstar_all'][i]
            core_mass_sys = np.copy(sssp_per_sys['mass_all'][i])[a_sys > 0]
            core_mass_sys[core_mass_sys > max_core_mass] = max_core_mass
            R_sys = sssp_per_sys['radii_all'][i][a_sys > 0]
            a_sys = a_sys[a_sys > 0]
            sigma_sys = solid_surface_density_prescription(core_mass_sys, R_sys, a_sys, Mstar=Mstar, n=n, prescription=prescription)

            if scale_up:
                sigma0, beta, scale_factor_sigma0 = fit_power_law_and_scale_up_MMEN(a_sys, sigma_sys, a0=a0, p0=p0, p1=p1)
            else:
                sigma0, beta = fit_power_law_MMEN(a_sys, sigma_sys, a0=a0, p0=p0, p1=p1)
                scale_factor_sigma0 = 1.

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

def fit_power_law_MMEN_per_system_observed_and_physical(sssp_per_sys, sssp, n_mult_min=2, max_core_mass=10., prescription='CL2013', n=10., a0=1., p0=1., p1=-1.5, scale_up=False):
    """
    Fit power-laws to the solid surface densities of the observed and physical planets *in each multi-planet system* in a physical catalog.

    Computes the solid surface densities and fit a power-law to each multi-planet system in the physical catalog, for the observed planets only and then for all the planets in those systems (using the physical planet properties for both).

    Parameters
    ----------
    sssp_per_sys : dict
        The dictionary containing the planetary and stellar properties for each system in a physical catalog (2-d and 1-d arrays).
    sssp : dict
        The dictionary containing the planetary and stellar properties of all planets in a physical catalog (1-d arrays).
    n_mult_min : int, default=2
        The minimum multiplicity to include.
    max_core_mass : float, default=10.
        The maximum allowed (solid) core mass (Earth masses).
    prescription : {'CL2013', 'S2014', 'nHill', 'RC2014'}, default='CL2013'
        The string indicating the prescription to use for computing the feeding zone widths.
    n : float, default=10.
        The number of Hill radii to use for the feeding zones. Only used for the 'nHill' prescription.
    a0 : float, default=1.
        The normalization point for the separation (AU).
    p0 : float, default=1.
        The initial guess for the normalization parameter, 'log(sigma0)'.
    p1 : float, default=-1.5
        The initial guess for the power-law index parameter, 'beta'.
    scale_up : bool, default=False
        Whether to scale up the normalization such that the power-law fit for each system is at/above the solid surface densities of every planet in the system.

    Returns
    -------
    fit_per_sys_dict : dict
        A dictionary containing the power-law fit parameters and other properties for each multi-planet system in the observed catalog.


    The output dictionary contains the following fields:

    - `n_pl_true`: The true multiplicities of the included systems (greater than or equal to `n_mult_min`) (1-d array).
    - `n_pl_obs`: The observed multiplicities of the included systems (1-d array).
    - `Mp_tot_true`: The total solid core mass (Earth masses) of all the planets in each included system (1-d array).
    - `Mp_tot_obs`: The total solid core mass (Earth masses) of only the observed planets in each included system (1-d array).
    - `sigma0_true`: The best-fit values for the solid surface density normalizations (g/cm^2) from fitting all the planets of the included systems (1-d array). If `scale_factor=True`, these normalization values have already been multiplied by the scale factor of each system.
    - `sigma0_obs`: The best-fit values for the solid surface density normalizations (g/cm^2) from fitting only the observed planets of the included systems (1-d array). If `scale_factor=True`, these normalization values have already been multiplied by the scale factor of each system.
    - `scale_factor_true`: The scale factors required to increase the normalizations of each system such that the power-law fits are at/above the solid surface densities of every planet in the system (1-d array).
    - `scale_factor_obs`: The scale factors required to increase the normalizations of each system such that the power-law fits are at/above the solid surface densities of every observed planet in the system (1-d array).
    - `beta_true`: The best-fit values for the power-law indices from fitting all the planets of the included systems (1-d array).
    - `beta_obs`: The best-fit values for the power-law indices from fitting only the observed planets of the included systems (1-d array).

    """
    fit_per_sys_dict = {'n_pl_true':[], 'n_pl_obs':[], 'Mp_tot_true':[], 'Mp_tot_obs':[], 'sigma0_true':[], 'sigma0_obs':[], 'scale_factor_true':[], 'scale_factor_obs':[], 'beta_true':[], 'beta_obs':[]}
    for i,det_sys in enumerate(sssp_per_sys['det_all']):
        if np.sum(sssp_per_sys['a_all'][i] > 0) >= n_mult_min and np.sum(det_sys) > 1:
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

            if scale_up:
                sigma0, beta, scale_factor_sigma0 = fit_power_law_and_scale_up_MMEN(a_sys, sigma_sys, a0=a0, p0=p0, p1=p1)
                sigma0_obs, beta_obs, scale_factor_sigma0_obs = fit_power_law_and_scale_up_MMEN(a_sys_obs, sigma_sys_obs, a0=a0, p0=p0, p1=p1)
            else:
                sigma0, beta = fit_power_law_MMEN(a_sys, sigma_sys, a0=a0, p0=p0, p1=p1)
                scale_factor_sigma0 = 1.
                sigma0_obs, beta_obs = fit_power_law_MMEN(a_sys_obs, sigma_sys_obs, a0=a0, p0=p0, p1=p1)
                scale_factor_sigma0_obs = 1.

            fit_per_sys_dict['n_pl_true'].append(len(a_sys))
            fit_per_sys_dict['n_pl_obs'].append(len(a_sys_obs))
            fit_per_sys_dict['Mp_tot_true'].append(np.sum(core_mass_sys))
            fit_per_sys_dict['Mp_tot_obs'].append(np.sum(core_mass_sys_obs))
            fit_per_sys_dict['sigma0_true'].append(sigma0)
            fit_per_sys_dict['sigma0_obs'].append(sigma0_obs)
            fit_per_sys_dict['scale_factor_true'].append(scale_factor_sigma0)
            fit_per_sys_dict['scale_factor_obs'].append(scale_factor_sigma0_obs)
            fit_per_sys_dict['beta_true'].append(beta)
            fit_per_sys_dict['beta_obs'].append(beta_obs)

    fit_per_sys_dict['n_pl_true'] = np.array(fit_per_sys_dict['n_pl_true'])
    fit_per_sys_dict['n_pl_obs'] = np.array(fit_per_sys_dict['n_pl_obs'])
    fit_per_sys_dict['Mp_tot_true'] = np.array(fit_per_sys_dict['Mp_tot_true'])
    fit_per_sys_dict['Mp_tot_obs'] = np.array(fit_per_sys_dict['Mp_tot_obs'])
    fit_per_sys_dict['sigma0_true'] = np.array(fit_per_sys_dict['sigma0_true'])
    fit_per_sys_dict['sigma0_obs'] = np.array(fit_per_sys_dict['sigma0_obs'])
    fit_per_sys_dict['scale_factor_true'] = np.array(fit_per_sys_dict['scale_factor_true'])
    fit_per_sys_dict['scale_factor_obs'] = np.array(fit_per_sys_dict['scale_factor_obs'])
    fit_per_sys_dict['beta_true'] = np.array(fit_per_sys_dict['beta_true'])
    fit_per_sys_dict['beta_obs'] = np.array(fit_per_sys_dict['beta_obs'])
    return fit_per_sys_dict

# TODO: write unit tests
def plot_feeding_zones_and_power_law_fit_MMEN_per_system_observed_and_physical(sssp_per_sys, sssp, n_mult_min=2, n_mult_max=10, max_core_mass=10., prescription='CL2013', n=10., a0=1., p0=1., p1=-1.5, scale_up=False, N_sys=10):
    """
    Make plots of solid surface density versus semi-major axis, including the feeding zones of each planet and a power-law fit to all the physical and observed planets, for each multi-planet system in a physical catalog.

    Parameters
    ----------
    sssp_per_sys : dict
        The dictionary containing the planetary and stellar properties for each system in a physical catalog (2-d and 1-d arrays).
    sssp : dict
        The dictionary containing the planetary and stellar properties of all planets in a physical catalog (1-d arrays).
    n_mult_min : int, default=2
        The minimum multiplicity to include.
    n_mult_max : int, default=10
        The maximum multiplicity to include.
    max_core_mass : float, default=10.
        The maximum allowed (solid) core mass (Earth masses).
    prescription : {'CL2013', 'S2014', 'nHill', 'RC2014'}, default='CL2013'
        The string indicating the prescription to use for computing the feeding zone widths.
    n : float, default=10.
        The number of Hill radii to use for the feeding zones. Only used for the 'nHill' prescription.
    a0 : float, default=1.
        The normalization point for the separation (AU).
    p0 : float, default=1.
        The initial guess for the normalization parameter, 'log(sigma0)'.
    p1 : float, default=-1.5
        The initial guess for the power-law index parameter, 'beta'.
    scale_up : bool, default=False
        Whether to scale up the normalization such that the power-law fit for each system is at/above the solid surface densities of every planet in the system.
    N_sys : int, default=10
        The maximum number of systems to plot. A separate figure will be plotted for each system.
    """
    y_sym_star = '*' if scale_up else ''
    assert 2 <= n_mult_min <= n_mult_max
    count = 0
    for i,det_sys in enumerate(sssp_per_sys['det_all']):
        if count >= N_sys:
            break
        if n_mult_min <= np.sum(det_sys) <= n_mult_max:
            count += 1
            print('##### System %s (i=%s):' % (count, i))

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

            if scale_up:
                sigma0, beta, scale_factor_sigma0 = fit_power_law_and_scale_up_MMEN(a_sys, sigma_sys, a0=a0, p0=p0, p1=p1)
                sigma0_obs, beta_obs, scale_factor_sigma0_obs = fit_power_law_and_scale_up_MMEN(a_sys_obs, sigma_sys_obs, a0=a0, p0=p0, p1=p1)
            else:
                sigma0, beta = fit_power_law_MMEN(a_sys, sigma_sys, a0=a0, p0=p0, p1=p1)
                scale_factor_sigma0 = 1.
                sigma0_obs, beta_obs = fit_power_law_MMEN(a_sys_obs, sigma_sys_obs, a0=a0, p0=p0, p1=p1)
                scale_factor_sigma0_obs = 1.

            # Plot the system:
            a_array = np.linspace(1e-3,2,1001)
            sigma_MMSN = MMSN(a_array)

            delta_a_sys_S2014 = feeding_zone_S2014(core_mass_sys, R_sys, a_sys, Mstar=Mstar)
            delta_a_sys_nHill = feeding_zone_nHill(core_mass_sys, a_sys, Mstar=Mstar, n=n)
            delta_a_sys_RC2014, a_bounds_sys = feeding_zone_RC2014(a_sys)

            fig = plt.figure(figsize=(12,8))
            plot = GridSpec(1,1,left=0.1,bottom=0.15,right=0.95,top=0.95,wspace=0,hspace=0)
            ax = plt.subplot(plot[0,0])
            plt.scatter(a_sys, np.log10(sigma_sys), marker='o', s=100.*R_sys**2., facecolors='none', edgecolors='k', label='All planets')
            plt.scatter(a_sys_obs, np.log10(sigma_sys_obs), marker='o', s=100.*R_sys_obs**2., color='k', label='Observed planets')
            for j,a in enumerate(a_sys):
                # Plot various feeding zones for each planet:
                da_S2014 = delta_a_sys_S2014[j]
                da_nHill = delta_a_sys_nHill[j]
                #da_RC2014 = delta_a_sys_RC2014[j]
                #plt.plot([0.5*a, 1.5*a], [np.log10(1.15*sigma_sys[j])]*2, lw=1, color='k')
                #plt.plot([a - da_S2014/2., a + da_S2014/2.], [np.log10(1.05*sigma_sys[j])]*2, lw=1, color='r')
                #plt.plot([a - da_nHill/2., a + da_nHill/2.], [np.log10(0.95*sigma_sys[j])]*2, lw=1, color='b')
                #plt.plot([a_bounds_sys[j], a_bounds_sys[j+1]], [np.log10(0.85*sigma_sys[j])]*2, lw=1, color='m')
                #plt.plot([a_bounds_sys[j], a_bounds_sys[j+1]], [np.log10(sigma_sys[j])]*2, lw=2, color='k') # lines for feeding zones
                plt.axvspan(a_bounds_sys[j], a_bounds_sys[j+1], alpha=0.2 if j%2==0 else 0.1, ec=None, fc='k')

                # To compare the planet masses to the integrated disk masses:
                Mp_core = core_mass_sys[j]
                plt.annotate(r'${:0.1f} M_\oplus$'.format(Mp_core), (a, np.log10(1.5*sigma_sys[j])), ha='center', fontsize=16)
                if prescription == 'CL2013':
                    Mp_intdisk = solid_mass_integrated_r0_to_r_given_power_law_profile(1.5*a, 0.5*a, sigma0, beta, a0=a0)
                    print('Planet (core) mass: {:0.2f} M_earth --- Integrated disk mass (CL2013): {:0.2f} M_earth'.format(Mp_core, Mp_intdisk))
                elif prescription == 'S2014':
                    Mp_intdisk = solid_mass_integrated_r0_to_r_given_power_law_profile(a + da_S2014/2., a - da_S2014/2., sigma0, beta, a0=a0)
                    print('Planet (core) mass: {:0.2f} M_earth --- Integrated disk mass (S2014): {:0.2f} M_earth'.format(Mp_core, Mp_intdisk))
                elif prescription == 'nHill':
                    Mp_intdisk = solid_mass_integrated_r0_to_r_given_power_law_profile(a + da_nHill/2., a - da_nHill/2., sigma0, beta, a0=a0)
                    print('Planet (core) mass: {:0.2f} M_earth --- Integrated disk mass (nHill): {:0.2f} M_earth'.format(Mp_core, Mp_intdisk))
                elif prescription == 'RC2014':
                    Mp_intdisk = solid_mass_integrated_r0_to_r_given_power_law_profile(a_bounds_sys[j+1], a_bounds_sys[j], sigma0, beta, a0=a0)
                    print('Planet (core) mass: {:0.2f} M_earth --- Integrated disk mass (RC2014): {:0.2f} M_earth'.format(Mp_core, Mp_intdisk))

            plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0, beta, a0=a0)), lw=3, ls='-', color='r', label=r'Fit to all planets ($\Sigma_0^%s = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0, beta) % y_sym_star)
            plt.plot(a_array, np.log10(MMEN_power_law(a_array, sigma0_obs, beta_obs, a0=a0)), lw=3, ls='--', color='r', label=r'Fit to observed planets ($\Sigma_0^%s = {:0.0f}$, $\beta = {:0.2f}$)'.format(sigma0_obs, beta_obs) % y_sym_star)
            plt.plot(a_array, np.log10(sigma_MMSN), lw=3, color='g', label=r'MMSN ($\Sigma_0 = {:0.0f}$, $\beta = {:0.2f}$)'.format(MMSN(a0), -1.5))
            ax.tick_params(axis='both', labelsize=20)
            plt.gca().set_xscale("log")
            plt.xticks([0.05, 0.1, 0.2, 0.4, 0.8])
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            plt.xlim([0.04,0.9])
            plt.ylim([-0.5,5.5])
            plt.xlabel(r'Semimajor axis, $a$ (AU)', fontsize=20)
            plt.ylabel(r'Surface density, $\log_{10}(\Sigma/{\rm g\,cm}^{-2})$', fontsize=20)
            plt.legend(loc='lower left', bbox_to_anchor=(0.,0.), ncol=1, frameon=False, fontsize=16)

            plt.show()



# Functions to compute the total integrated mass of solids within a given separation for a fitted power-law MMEN:

def solid_mass_integrated_r0_to_r_given_power_law_profile(r, r0, sigma0, beta, a0=1.):
    """
    Compute the total solid mass from a power-law profile for the solid surface density, within a given separation.

    Integrates the power-law over an annular area from radius `r0` to `r`.

    Parameters
    ----------
    r : float
        The separation (AU) of the outer boundary.
    r0 : float
        The separation (AU) of the inner boundary.
    sigma0 : float
        The normalization for the solid surface density (g/cm^2) at separation `a0`.
    beta : float
        The power-law index.
    a0 : float, default=1.
        The normalization point for the separation (AU).

    Returns
    -------
    M_r : float
        The total integrated solid mass (Earth masses).


    Warning
    -------
    The inner boundary `r0` must be non-zero to avoid infinite mass for `beta <= -2`.
    """
    assert 0 <= r0 < r # r0 must be nonzero to avoid infinite mass for beta <= -2
    assert 0 < sigma0
    assert 0 < a0
    if beta == -2:
        M_r = 2.*np.pi*sigma0*((a0*gen.AU)**2.)*np.log(r/r0) # total mass in grams
    else: # if beta != -2
        M_r = (2.*np.pi*sigma0/(2.+beta))*((r/a0)**(2.+beta) - (r0/a0)**(2.+beta)) *((a0*gen.AU)**2.) # total mass in grams
    M_r = M_r/(1e3*gen.Mearth) # convert total mass to Earth masses
    return M_r
