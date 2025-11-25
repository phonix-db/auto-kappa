#
# scph.py
#
# This script helps to conduct SCPH calculations.
#
# Copyright (c) 2024 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
# import sys
import os
import os.path
import numpy as np
import subprocess
import shlex
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from auto_kappa.io.fcs import FCSxml
from auto_kappa.plot import make_figure, get_customized_cmap
from auto_kappa.io.band import get_scph_bands
from auto_kappa.structure import change_structure_format

import logging
logger = logging.getLogger(__name__)

def calculate_high_order_force_constants(
    almcalc, calculator, order=5, frac_nrandom=None, disp_temp=500,):
    """ Calculate high-order (up to 6th-order) force constants finally to
    calculate 4th order FCs for phonon renormalization.
    """
    ### calculate forces for high-order FCs
    almcalc.write_alamode_input(propt='suggest', order=order)
    almcalc.run_alamode(propt='suggest', order=order)
    almcalc.calc_forces(
            order=order, calculator=calculator,
            frac_nrandom=frac_nrandom,
            temperature=disp_temp,
            output_dfset=2,
            )
    
    ### calculate anharmonic force constants
    for propt in ['cv', 'lasso']:
        almcalc.write_alamode_input(propt=propt, order=order)
        almcalc.run_alamode(propt, order=order, ignore_log=False)
    
def conduct_scph_calculation(almcalc, order=6, temperatures=100*np.arange(1,11)):
    """ Conduct SCPH calculation and obtain effective harmonic FCs 
    """
    msg =  "\n Conduct SCPH calculation"
    msg += "\n ========================"
    logger.info(msg)
    
    ### parameters for SCPH
    tmin = np.min(temperatures)
    tmax = np.max(temperatures)
    dt = temperatures[1] - temperatures[0]
    
    ### SCPH calculation
    propt = "scph"
    almcalc.write_alamode_input(propt=propt, tmin=tmin, tmax=tmax, dt=dt)
    almcalc.run_alamode(propt, order=order, ignore_log=False)
    
    ### Create effectvie harmonic FCs
    for temp in temperatures:
        _create_effective_harmonic_fcs(almcalc, temperature=temp)
    
    if almcalc.calculate_forces:
        
        logger.info("")
        dir_scph = almcalc.out_dirs['higher']['scph']
        fig_width = 3.0
        aspect = 0.6
        lw = 0.3
        
        ## Plot band & DOS
        figname = f"{dir_scph}/fig_scph_bands.png"
        # logfile = f"{dir_scph}/scph.log"
        try:
            file_scph_bands = f"{dir_scph}/{almcalc.prefix}.scph_bands"
            _plot_scph_bands(file_scph_bands, figname=figname, lw=lw,
                             fig_width=fig_width, aspect=aspect, dpi=600)
        except Exception as e:
            logger.error(f"\n Failed to plot SCPH bands: {e}")
        
        ## Plot effective force constants
        try:
            plot_scph_force_constants(
                dir_scph, almcalc.prefix, temperatures, file_fc2_0K=almcalc.fc2xml, 
                fig_width=fig_width, aspect=aspect, dpi=600)
        except Exception as e:
            logger.error(f"\n Failed to plot SCPH force constants: {e}")

def plot_scph_force_constants(
    dir_work, prefix, temperatures, file_fc2_0K=None, 
    xlabel='Distance (${\\rm \\AA}$)', 
    ylabel='Eff. harm. FC (${\\rm eV/\\AA^2}$)',
    fig_width=2.8, aspect=0.5, dpi=600):
    """ Plot effective harmonic force constants obtained from SCPH calculation.
    """
    fig, axes = make_figure(1, 1, fontsize=7, fig_width=fig_width, aspect=aspect)
    ax = axes[0][0]
    
    try:
        availabilities = get_availabilities(f"{dir_work}/scph.log")
        idx_ua = [i for i, t in enumerate(temperatures) if availabilities.get(int(t), True) == False]
        temperatures = np.delete(temperatures, idx_ua)
    except:
        availabilities = None
    
    nt = len(temperatures)
    cmap = get_customized_cmap(nt)
    
    ## plot effective FC2
    for it in range(nt+1):
        
        if it == nt: # FC2 at 0K 
            temp = 0.
            file_xml = file_fc2_0K
            color = 'grey'
            lw = 0.2
        else: # effective FC2 at finite temperatures
            temp = temperatures[it]
            file_xml = f"{dir_work}/{prefix}_{int(temp)}K.xml"
            color = cmap(it)
            lw = 0.4
            
        try:
            fcs = FCSxml(file_xml)
        except Exception as e:
            logger.error(f" Failed to read {file_xml}: {e}")
            continue
        
        if it == 0:
            show_legend = True
        else:
            show_legend = False
        
        fcs.plot_fc2(ax, xlabel=None, ylabel=None, color=color, lw=lw, show_legend=show_legend)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    text  = "${\\rm T_{min} = %d K}$ (blue)" % temperatures[0]
    text += "\n${\\rm T_{max} = %d K}$ (red)" % temperatures[-1]
    text += "\n${\\rm \\Delta T = %dK}$" % (temperatures[1] - temperatures[0])
    if file_fc2_0K is not None:
        if os.path.exists(file_fc2_0K):
            text += "\n0K (grey)"
    ax.text(0.95, 0.95, text, transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='top', fontsize=6)
    
    ## Save the figure
    figname = f"{dir_work}/fig_scph_fc2.png"
    fig.savefig(figname, dpi=dpi, bbox_inches='tight')
    if os.path.isabs(figname):
        figname = "./" + os.path.relpath(figname, os.getcwd())
    logger.info(f"\n Force constants were plotted in {figname}")


def _create_effective_harmonic_fcs(almcalc, temperature=300, workdir=None):
    """ Create effective harmonic FCs """
    
    prefix = almcalc.prefix
    command = almcalc.commands['alamode']['dfc2']
    
    ### get file names
    xml_orig = os.path.relpath(
            almcalc.out_dirs["higher"]["lasso"] + "/%s.xml" % prefix,
            almcalc.out_dirs["higher"]["scph"])
    xml_new = "%s_%dK.xml" % (prefix, temperature)
    file_corr = "%s.scph_dfc2" % prefix
    
    ### change directory
    dir_cur = os.getcwd()
    if workdir is None:
        workdir = almcalc.out_dirs['higher']['scph']
    os.chdir(workdir)
    
    ### run the job
    cmd = "%s %s %s %s %d" % (command, shlex.quote(xml_orig), shlex.quote(xml_new), shlex.quote(file_corr), temperature)
    
    logfile = "dfc2.log"
    file_err = "std_err.txt"
    with open(logfile, 'w') as f, open(file_err, "w", buffering=1) as f_err:
        proc = subprocess.Popen(
                cmd, shell=True, env=os.environ, stdout=f, stderr=f_err)
    
    ### back to the original directory
    os.chdir(dir_cur)
    return 0

def get_sym_ops_from_pymatgen(primitive, symprec=1e-5):
    """ Get symmetry operations from a pymatgen Structure object.
    """
    sga = SpacegroupAnalyzer(primitive, symprec=symprec)
    symm_ops = sga.get_symmetry_operations()
    
    rotations = []
    for op in symm_ops:
        rotations.append(op.rotation_matrix)   # 3x3 numpy array
    
    return rotations

def is_diagonal_matrix(A, tol=1e-8):
    """ Check if a given 3x3 matrix A is diagonal within a specified tolerance.
    
    Args
    ------
    A : array-like
        Input 3x3 matrix to be checked.
    tol: float
        Tolerance for checking off-diagonal elements.
    """
    A = np.array(A)
    if A.shape != (3, 3):
        raise ValueError("Matrix must be 3x3")
    
    # Check if all off-diagonal elements are within the tolerance
    offdiag = A - np.diag(np.diag(A))
    return np.all(np.abs(offdiag) < tol)

def safe_kmesh_from_supercell(supercell_lattice, sym_ops, min_nk=2, atol=1e-5):
    """ Propose safe kmesh for ALAMODE SCPH calculation based on symmetry operations.
    
    Args
    -----
    supercell_lattice : (3,3) ndarray with row = a1, a2, a3 (direct lattice)
    sym_ops : list of symmetry rotation matrices in direct space (3x3 matrices)
    min_nk : minimum nk required (ALAMODE SCPH requires ≥2)
    
    Returns:
        tuple (nk1, nk2, nk3)
    """
    # 1. Calculate reciprocal lattice vectors
    A = np.array(supercell_lattice).T        # columns = a1,a2,a3
    B = 2 * np.pi * np.linalg.inv(A).T       # columns = b1,b2,b3
    bvecs = B.T                              # rows: b1,b2,b3
    
    # 2. symmetry operations check
    # direct space rotation R acts on reciprocal as (R^-1)^T
    safe = [True, True, True]   # safe[i] = True -> direction i can have nk >= 2
    for R in sym_ops:
        Rrec = np.linalg.inv(R).T
        for i in range(3):
            b_i = bvecs[i]
            Rb = Rrec @ b_i
            
            # Solve for coefficients in bvecs basis
            coeff = np.linalg.solve(bvecs.T, Rb)
            
            # Check if all coefficients are integers
            for j in range(3):
                if not np.isclose(coeff[j], round(coeff[j]), atol=atol):
                    # Non-integer -> direction i cannot have nk > 1
                    safe[i] = False
                    break
    
    # 3. Determine KMESH based on safe[i]
    km = []
    for i in range(3):
        if safe[i]:
            km.append(int(min_nk))     # safe direction
        else:
            km.append(1)          # fixed to 1 due to symmetry
    return list(km)

def get_kmesh_scph(primitive, kmesh_interpolate, kdensity_limit, dim=3):
    """ Propose kmesh_scph based on kmesh_interpolate and kdensity_limit.
    
    Args
    -----
    kmesh_interpolate : list of int
        Proposed kmesh_interpolate for SCPH calculation.
    kdensity_limit : float
        Minimum kdensity for SCPH calculation.
    dim : int
        Dimension of the system (default: 3)
    
    Returns
    -------
    list of int
        Proposed kmesh_scph for SCPH calculation.
    """
    from auto_kappa.structure.crystal import get_automatic_kmesh
    
    # Minimum kmesh_scph based on kdensity_limit
    kmesh_scph_limit = get_automatic_kmesh(
        primitive, reciprocal_density=kdensity_limit, dim=dim)
    
    # Find smallest kmesh_scph that is equal to or a multiple of kmesh_interpolate
    n = 1
    while True:
        kmesh_scph_n = [n * ki for ki in kmesh_interpolate]
        ratio = np.asarray(kmesh_scph_n) / np.asarray(kmesh_scph_limit)
        if all(ratio >= 1):
            break
        n += 1
    
    kmesh_scph = [n * ki for ki in kmesh_interpolate]
    return kmesh_scph

def set_parameters_scph(inp, primitive=None, scell=None, mat_p2s=None, deltak=0.01, kdensity_limit=20, **kwargs):
    """ Set ALAMODE parameters for SCPH calculation.
    
    Args
    =====
    inp : auto_kappa.io.AnphonInput
    primitive : primitive structure
    mat_p2s : 3x3 integer matrix (list or numpy array) 
        supercell matrix wrt primitive cell
    deltak : integer, 0.01
    kdensity_limit : array of float,
        minimum kdensity for SCPH calculation
        proposed kmesh_scph will be equal to or larger than this limit
    """
    scph_params = {
            ### general
            "tmin": 100,
            "tmax": 1000,
            "dt"  : 100,
            ### scph
            # "kmesh_scph": [1,1,1],
            # "kmesh_interpolate": [1,1,1],
            "self_offdiag": 1,    ## not default (0)
            "mixalpha": 0.1,      ## default
            "maxiter": 2000,      ## double of default
            "tol_scph": 1e-10,    ## default
            }
    
    inp.set_kpoint(deltak=deltak)
    
    if is_diagonal_matrix(mat_p2s):
        ## orthogonal case
        kmesh_interpolate = [int(mat_p2s[i][i]) for i in range(3)]
    else:
        ## skewed case
        prim_pmg = change_structure_format(primitive, format='pymatgen')
        sym_ops = get_sym_ops_from_pymatgen(prim_pmg)
        kmesh_interpolate = safe_kmesh_from_supercell(scell.cell.array, sym_ops, min_nk=2)
        
        msg = "\n Proposed KMESH_INTERPOLATE for SCPH calculation: %s" % str(kmesh_interpolate)
        msg += "\n Please note that the kmesh determination procedure is still experimental."
        msg += "\n If you will get unexpected results, please set KMESH_INTERPOLATE and KMESH_SCPH manually."
        logger.info(msg)
    
    ### get kmesh_scph
    kmesh_scph = get_kmesh_scph(primitive, kmesh_interpolate, kdensity_limit, dim=inp.dim)
    
    ### set parameters
    scph_params["kmesh_scph"] = kmesh_scph
    scph_params["kmesh_interpolate"] = kmesh_interpolate
    
    ###
    scph_params.update(kwargs)
    
    ### update!
    inp.update(scph_params) 

def get_availabilities(logfile):
    """ Check if SCPH calculation is available for each temperature.
    
    Args
    =====
    
    logfile : str
        Log file of SCPH calculation.

    Returns
    =======

    bool
        True if SCPH calculation is available, False otherwise.
    """
    if not os.path.exists(logfile):
        return None
    
    with open(logfile, 'r') as f:
        lines = f.readlines()
    
    availabilities = {}
    for line in lines:
        if "not converged" in line:
            temp = int(float(line.split()[2]))
            availabilities[temp] = False
        elif "convergence achieved" in line:
            temp = int(float(line.split()[2]))
            availabilities[temp] = True
    
    return availabilities

def _plot_scph_bands(filename, figname='fig_scph_bands.png',
                     lw=0.3, fig_width=3.0, aspect=0.6, dpi=600):
    
    from auto_kappa.io.band import SCPHBand
    
    if os.path.exists(filename) == False:
        return None    
    fig, axes = make_figure(1, 1, fig_width=fig_width, aspect=aspect)
    ax = axes[0][0]
    
    band = SCPHBand(filename)
    
    band.plot(ax, lw=lw)
    ax.axhline(0, color='grey', lw=0.2, ls='-')
    
    fig.savefig(figname, bbox_inches='tight', dpi=dpi)
    if os.path.isabs(figname):
        figname = "./" + os.path.relpath(figname, os.getcwd())
    msg = "\n SCPH bands were plotted: %s" % figname
    logger.info(msg)
    return fig, ax

# def plot_scph_bands(ax, filename, logfile=None):
#     """ Plot the phonon band structure from a given file.
    
#     Parameters:
#     ax : matplotlib.axes.Axes
#         The axes on which to plot the band structure.
#     filename : str
#         .scph_bands file generated by Alamode.
#     """
#     ## Read the phonon band structure data
#     temps, kpoints_list, frequencies_list = get_scph_bands(filename, logfile=logfile)
    
#     ## Get symmetry points and labels
#     sym_labels, sym_kpoints = get_symmetry_points_from_file(filename)
#     nt = len(temps)
#     cmap = get_customized_cmap(nt)
    
#     ## Plot the band structure
#     for it in range(nt):
#         kpoints = kpoints_list[it]
#         frequencies = frequencies_list[it]
        
#         if len(kpoints) == 0 or len(frequencies) == 0:
#             continue
        
#         if it == 0 or it == nt - 1:
#             lab = '%dK' % temps[it]
#         else:
#             lab = None
        
#         plot_bands(ax, kpoints, frequencies, col=cmap(it), label=lab)
    
#     ax.set_ylabel('Frequency (${\\rm cm^{-1}}$)')
#     set_xticks_labels(ax, kpoints_list[0][-1], sym_kpoints, sym_labels)
#     set_axis(ax)
#     set_legend(ax, loc='lower left', loc2=[0.0, 1.0], fs=6, ncol=2, alpha=0.5)

def get_fmin_scph(filename, logfile=None, temperature=None):
    """ Get the minimum frequency from SCPH bands.
    If temperature is specified, return the minimum frequency at that temperature.
    If temperature is not specified, return a dictionary with temperatures as keys
    and minimum frequencies as values.
    
    Args
    ------
    filename : str
        The filename of the SCPH bands file. (.scph_bands file)
    logfile : str, optional
        The logfile of the SCPH calculation. If None, it will look for 'scph
    """
    temps, _, fs_list = get_scph_bands(filename, logfile=logfile, verbose=False)
    fmins = {}
    for it, temp in enumerate(temps):
        fs = fs_list[it]
        fmin = np.amin(fs)
        fmins[int(temp)] = float(fmin)
        if temperature is not None:
            if abs(temp - temperature) < 0.1:
                return fmin
    
    if temperature is not None:
        msg = f"\n Temperature {temperature}K not found in the SCPH bands file."
        logger.warning(msg)
    
    return fmins
