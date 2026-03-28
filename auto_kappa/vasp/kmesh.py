# 
# kmesh.py
# 
# This script optimize kmesh density for VASP calculations.
# 
# Created on November 26, 2025
# Copyright (c) 2025 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
# 
import os
import numpy as np
import pandas as pd
import ase.io

import logging
logger = logging.getLogger(__name__)

from auto_kappa import output_directories
from auto_kappa.calculators.vasp import get_vasp_calculator, run_vasp

import matplotlib.pyplot as plt
from auto_kappa.plot.initialize import set_matplot, set_axis, sci2text

def klength2mesh(length, lattice, rotations=None):
    """ Convert length to mesh for q-point sampling.
    Reference: phonopy.structure.grid_points.klength2mesh
    
    This conversion for each reciprocal axis follows VASP convention by
    
    >>> N = max(1, int(l * |a|^* + 0.5))
    
    int means rounding down, not rounding to nearest integer.
    
    Parameters
    ----------
    length : float
        Length having the unit of direct space length.
    
    lattice : array_like
        Basis vectors of primitive cell in row vectors.
        dtype='double', shape=(3, 3)
    
    rotations : array of int, shape=(3,), optional
        Rotation matrices in real space. When given, mesh numbers that are
        symmetrically reasonable are returned. Default is None.

    Returns
    -------
    array_like : int, shape=(3,)
    
    Note
    -----
    This function is copied from Phonopy library.
    
    """
    rec_lattice = np.linalg.inv(lattice)
    rec_lat_lengths = np.sqrt(np.diagonal(np.dot(rec_lattice.T, rec_lattice)))
    mesh_numbers = np.rint(rec_lat_lengths * length).astype(int)

    if rotations is not None:
        from phonopy.structure.symmetry import get_lattice_vector_equivalence
        reclat_equiv = get_lattice_vector_equivalence(
            [r.T for r in np.array(rotations)])
        m = mesh_numbers
        mesh_equiv = [m[1] == m[2], m[2] == m[0], m[0] == m[1]]
        for i, pair in enumerate(([1, 2], [2, 0], [0, 1])):
            if reclat_equiv[i] and not mesh_equiv:
                m[pair] = max(m[pair])

    return np.maximum(mesh_numbers, [1, 1, 1])


def optimize_klength(structure,
                     vasp_params=None,
                     command={'mpirun': 'mpirun', 'nprocs': 1, 'vasp': 'vasp'},
                     klengths=[10, 5, 100],
                     min_klength=20,
                     tolerance=2e-3, # eV per atom
                     potcar_setups=None,
                     xc='pbesol',
                     base_dir='.'):
    """ Optimize kmesh density for a given structure.
    
    Parameters
    -----------
    structure : object
        The crystal structure for which to optimize the kmesh.
    klengths : list of int, optional
        List containing [min_klength, delta_klength, max_klength] for the optimization (default is [10, 10, 100]).
        kl = 10 (large gap insulators) and kl = 100 (d metals).
    tolerance : float, optional
        Tolerance for convergence (default is 1e-3 eV per atom).
    """
    _start_optimization(klengths, min_klength, tolerance)
    
    cell = structure.cell.array
    kl_min = klengths[0]
    delta_kl = klengths[1]
    kl_max = klengths[2]
    
    iteration = 0
    ene_prev = None
    delta_ene = None
    flag = False
    dump = []
    while True:
        
        if iteration == 0:
            kl_curr = kl_min
            kpts_curr = klength2mesh(kl_curr, cell)
        else:
            kpts_prev = kpts_curr.copy()
            while True:
                kl_curr += delta_kl
                kpts_curr = klength2mesh(kl_curr, cell)
                if any(kpts_curr > kpts_prev):
                    break
                kpts_prev = kpts_curr.copy()
        
        if kl_curr > kl_max:
            # logger.info(f"\n Reached maximum k-length: {kl_curr} > {kl_max}")
            break
        
        ### prepare output directory
        dir_kl = output_directories['preparation']['klength']
        outdir = f"{base_dir}/{dir_kl}/kl_{kl_curr}"
        
        ###
        vasprun = f"{outdir}/vasprun.xml"
        if not _job_finished(vasprun):
                
            ### get VASP calculator
            calc = get_vasp_calculator(
                vasp_params, 
                directory=outdir,
                atoms=structure, 
                kpts=kpts_curr,
                setups=potcar_setups,
                xc=xc)
            
            ### set VASP command
            mpirun = command.get('mpirun', 'mpirun')
            nprocs = command.get('nprocs', 1)
            if list(kpts_curr) == [1, 1, 1]:
                vasp = command.get('vasp_gam')
            else:
                vasp = command.get('vasp', 'vasp')
            calc.command = f"{mpirun} -n {nprocs} {vasp}"
            
            ### run VASP
            run_vasp(calc, structure, method='custodian')
            
        ### get energy per atoms
        ene_curr = get_energy(vasprun, len(structure))
        if ene_curr is None:
            logger.info(f" ❌ VASP calculation failed for k-length = {kl_curr}, k-mesh = {kpts_curr}.")
            iteration += 1
            continue
        
        ### print results
        if iteration == 0 or ene_prev is None:
            delta_ene = None
        else:
            delta_ene = abs(ene_curr - ene_prev)
        print_result(iteration+1, kl_curr, kpts_curr, ene_curr, delta_ene)
        dump.append([kl_curr, ene_curr])
        
        ### check convergence
        if delta_ene is not None and delta_ene <= tolerance:
            msg = "Converged"
            flag = True
        if kl_curr >= kl_max:
            msg = "Reached maximum k-length"
            flag = True
        if flag and kl_prev >= min_klength:
            kl_opt = kl_prev
            kpts_opt = kpts_prev.copy()
            ener_opt = ene_prev
            logger.info(f" {msg}: k-length = {kl_opt}, k-mesh = {kpts_opt}, E = {ener_opt:.3e} eV")
            break
        
        iteration += 1
        kl_prev = kl_curr
        kpts_prev = kpts_curr.copy()
        ene_prev = ene_curr
    
    logger.info(f"\n Optimized k-length: {kl_opt}")
    
    ## summarize results
    df = pd.DataFrame(dump, columns=['k-length', 'energy'])
    outfile = f"{base_dir}/{dir_kl}/kl_energy.csv"
    df.to_csv(outfile, index=False)
    logger.info(f"\n Output {_relpath(outfile)}")
    
    ## Plot results
    figname = f"{base_dir}/{dir_kl}/fig_kl_ene.png"
    plot_kl_energy(df['k-length'], df['energy'], tolerance=tolerance, figname=_relpath(figname))
    
    return kl_opt


def _relpath(path):
    """ Get relative path """
    if not path.startswith("./"):
        path = "./" + os.path.relpath(path)
    return path


def _start_optimization(klengths, min_klength, tolerance):
    """ Print start message """
    line = "k-length optimization"
    msg = f"\n {line}"
    msg += "\n " + "=" * (len(line))
    logger.info(msg)
    
    msg = f"\n k-length: (initial, delta, last) = ({klengths[0]}, {klengths[1]}, {klengths[2]})"
    msg += f"\n min. k-length : {min_klength}"
    msg += f"\n tolerance     : {tolerance} eV per atom"
    msg += "\n"
    logger.info(msg)


def get_energy(filename, natoms):
    """ Summarize each calculation """
    atoms = ase.io.read(filename, format='vasp-xml', index=-1)
    try:
        ene_curr = atoms.get_potential_energy() / natoms # energy per atom
    except Exception:
        ene_curr = None
    return ene_curr


def print_result(index, kl, kpts, energy, delta_energy=None):
    """ Print optimization results """
    msg = f" {index}: k-length = {kl}, k-mesh = {kpts}, E = {energy:.3e} eV"
    if delta_energy is not None:
        msg += f", ΔE = {delta_energy:.3e} eV"
    logger.info(msg)


def _job_finished(filename):
    """ Check whether a VASP job is finished successfully """
    if not os.path.exists(filename):
        return False
    atoms = ase.io.read(filename, format='vasp-xml', index=-1)
    
    try:
        ene = atoms.get_potential_energy() / len(atoms)
        forces = atoms.get_forces()
    except Exception:
        ene = None
        forces = None
    
    if ene is None or forces is None:
        return False
    return True


def plot_kl_energy(xdat, ydat, tolerance=None, figname='fig_kl_ene.png', color='blue',
                   dpi=600, fontsize=7, fig_width=2.5, aspect=0.9, lw=0.8, ms=3.0):
    """ Plot k-length vs. energy """
    set_matplot(fontsize=fontsize)
    fig = plt.figure(figsize=(fig_width, aspect*fig_width))
    
    ax = plt.subplot()
    ax.set_xlabel('k-length')
    ax.set_ylabel('Energy (eV/atom)')
    
    ax.plot(xdat, ydat, linestyle='-', lw=lw, c=color,
            marker='o', markersize=ms, mec=color, mfc='none', mew=lw)
    
    if tolerance is not None:
        text = f"Tolerance: {sci2text(tolerance)} eV/atom"
        ax.text(0.95, 0.95, text, fontsize=6, transform=ax.transAxes, ha="right", va="top")
    
    set_axis(ax)
    fig.savefig(figname, dpi=dpi, bbox_inches='tight')
    logger.info(f" Output {figname}")
