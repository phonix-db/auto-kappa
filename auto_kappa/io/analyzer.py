#
# analyzer.py
#
# This script contains a few calculator to analyze results obtained with
# Alamode.
#
# Copyright (c) 2022 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
# import sys
import numpy as np
import auto_kappa.units as units

import logging
logger = logging.getLogger(__name__)

def convert_gamma2tau(gammas):
    """Convert gamma[1/cm] to lifetime[ps]
    
    Args
    ----
    gammas : ndarray, float, shape=(nk,nb)
        GAMMS written in .result file
        unit : Kayser[1/cm]
    
    Return
    --------
    lifetime : ndarray, float, shape=(nk,nb)
        lifetime[ps]

    """
    eps = 1e-10
    g_tmp = np.where(gammas<eps, 1.0, gammas)
    lifetime = np.where(gammas<eps, 0.0,
            1e12*0.5/g_tmp/units.AlmCmToHz)
    return lifetime 

def get_average_at_degenerate_point(omega, tau, eps=1e-3):
    """Average value for degenerate point
    
    Args
    -----
    omega, tau : ndarray, float, shape=(nk,nb)
        frequencies and lifetime, but any value is available for the secont
        variable.

    """
    nk = len(omega)
    nb = len(omega[0])
    tau_ave = np.zeros_like(tau)
    for ik in range(nk):
        for ib1 in range(nb-1):
            # -- get degenerated modes
            idx_deg = []
            for ib2 in range(ib1,nb):
                if abs(omega[ik,ib2] - omega[ik,ib1]) > eps:
                    break
                else:
                    idx_deg.append(ib2)
            # -- cal average
            tau_ave[ik,idx_deg] = (
                    np.ones_like(idx_deg) * 
                    np.average(tau[ik,idx_deg]))
    return tau_ave

_skip_logged = set()

def get_kmode(volume, temp, frequencies, multiplicity, velocities, lifetime, verbose=True):
    """
    Calculate and return thermal conductivity of every modes

    Parameters
    -----------
    volume : float
        volume of a primitive unit cell [Bohr**3]
    frequencies : ndarray, float, shape=(nk,nbands)
        eigenvalues [1/cm]

    multiplicity : array, integer, shape=(nk)
        multiplicity
    velocities : ndarray, float, shape=(nk,nbands,MULTI_MAX,3)
        group velocity [m/s]
    lifetime : ndarray, float, shape=(nk,nbands)
        lifetime [ps]
    nk, nbands: integer
        # of kpoints and bands 
    temp : float
        temperature [K]
    
    mmax : integer
            maximum multiplicity

    Return
    -------
    
    kappa : ndarray, float, shape=(3,3)
        thermal conductivity
    
    kmode : ndarray, float, shape=(nkpoints,nbands,3,3)
        thermal conductivity of each mode
    """
    nk = len(frequencies)
    nb = len(frequencies[0])
    
    # mmax = int(np.max(multiplicity))
    kmode = np.zeros((((nk,nb,3,3))))
    
    count = 0
    for ik in range(nk):
        multi = multiplicity[ik]
        for ib in range(nb):

            v2tensor = np.zeros((3,3))
            for im in range(multi):
                v2tensor += ((
                        velocities[ik][ib][im] *
                        velocities[ik][ib][im].reshape(3,1)
                        ) / multi)
            # -- heat capacity
            if frequencies[ik,ib] < 0.:
                if verbose:
                    key = (ik, ib, round(frequencies[ik,ib], 2))
                    if key not in _skip_logged:
                        _skip_logged.add(key)
                        msg = " SKIP ik: %d  ib: %d  %.2f cm^-1"%(
                            ik, ib, frequencies[ik,ib])
                        logger.info(msg)
                        count += 1
                kmode[ik,ib] = np.zeros((3,3))
            else:
                Cph = get_heat_capacity(frequencies[ik,ib], temp)   # J/K
                kmode[ik,ib] = (
                        v2tensor[:,:] * Cph * 1e-12*lifetime[ik,ib] * multi)
                                                               # m^3 * W/(m*K) 
    
    if count != 0 and verbose:
        logger.info(" %d modes are skipped due to negative frequencies.\n" % count)
    
    ## --- dk
    ## np.sum(multiplicity) : number of k-points
    kmode *= (
        1./(volume * units.BohrToM**3)/
        float(np.sum(multiplicity)))  # W/(m*K)
    
    # --- thermal conductivity
    kappa = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            kappa[i,j] = np.sum(kmode[:,:,i,j])
    
    return kappa, kmode

def get_heat_capacity(fkay, temp, min_fkay=1e-8):
    """
    calculate heat capacity

    Parameters
    ----------
    fkay : float
        frequency [1/cm]
    temp : float
        temperature [K]
    
    Return
    ------
    cph : float
        heat capacity [J/K]
    """
    from auto_kappa.math.statistics import get_diffrential_statistics
    cph = (fkay*units.CmToJ * get_diffrential_statistics(
        fkay*units.CmToJ, temp, "be", "t"))
    return cph

#def matthiessen(gphph, isotope=None, 
#        size=None, velocities=None, multiplicity=None):
#    """Calculate phonon lifetime according to the Mattiessen's rule
#    gphph : ndarray, float, shape=(ntemps,nk,nbands)
#        temperature depending modal gamma due to ph-ph scattering
#    isotope : ndarray, float, shape=(nk,nbands)
#        modal gamma due to isotope
#    
#    size : float
#        grain size [nm]
#    multiplicity : array, int, shape=(nk)
#        multiplicity
#    velocities : ndarray, float, shape=(nk,nbands,MULTI_MAX,3)
#        velocities [m/s]
#    """
#    rscat = np.zeros_like(gphph)
#    for it in range(len(gphph)):
#        rscat[it,:,:] += 2.*gphph[it,:,:]
#        if isotope is not None:
#            rscat[it,:,:] += 2.*isotope[:,:]
#        
#        if (size is not None and
#                velocities is not None and
#                multiplicity is not None):
#            print("Error: boundary effect is not yet supported.")
#            sys.exit()
#            #rscat[it,:,:] += 2.*abs(velocities[it,:,)
#            #......
#    
#    return rscat


