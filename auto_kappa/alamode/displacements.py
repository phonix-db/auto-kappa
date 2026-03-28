# 
# displacements.py
# 
# This script manages the displacement patterns.
# 
# Created on February 02, 2026
# Copyright (c) 2026 Masato Ohnishi
# Created by Masato Ohnishi and Michimasa Morita
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
# 
# import sys
import numpy as np
import pandas as pd
from ase import Atoms

from auto_kappa.structure.crystal import change_structure_format

def generate_displaced_structures(structure, all_disps, structures):
    """ Generate displaced structures from the given displacements.
    """
    ## set calculator for each pattern
    disps_all = {}
    for i, displacements in enumerate(all_disps):
        scell = structure.copy()
        scell.translate(displacements)
        structures[i+1] = scell
        disps_all[i+1] = displacements
    return disps_all
        
def adjust_random_displacements(
    structure, displacements, max_abs_disp=10.0, max_rel_disp=0.1):
    """ Adjust random displacements to be within the specified limits.
    
    Parameters
    ----------
    structure : ase.Atoms
        The pristine supercell structure
    displacements : np.ndarray, shape=(N, 3)
        The array of displacements to be adjusted.
    max_abs_disp : float, optional
        The maximum absolute displacement allowed. Default is 1.0.
        If None, only relative displacement is considered.
    max_rel_disp : float, optional
        The maximum relative displacement allowed, as a fraction of the
        nearest neighbor distance. Default is 0.1 (10%).
    
    Returns
    -------
    adjusted_displacements : np.ndarray, shape=(N, 3)
        The adjusted displacements within the specified limits.
    
    Note
    ----
    disp_max = min(max_abs_disp, max_rel_disp * nearest_neighbor_distance)
    
    """
    natoms_sc = len(structure)
    assert len(displacements) == len(structure), \
        " Displacement array length must match number of atoms in structure."
    
    if type(structure) is not Atoms:
        structure = change_structure_format(structure, to_format='ase')
    
    ## Calculate nearest neighbor distance
    distances = structure.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    nearest_neighbor_distances = [float(np.min(ds)) for ds in distances]
    assert len(nearest_neighbor_distances) == natoms_sc, \
        " Nearest neighbor distances calculation error."
    
    ## Get maximum allowed displacement based on absolute and relative limit
    disp_max_values = []
    for ia in range(natoms_sc):
        rel_limit = max_rel_disp * nearest_neighbor_distances[ia]
        if max_abs_disp is not None:
            disp_max = min(max_abs_disp, rel_limit)
        else:
            disp_max = rel_limit
        
        disp_max_values.append(disp_max)
    
    ## Adjust displacements for each atom and make a new displacement array
    new_displacements = displacements.copy()
    mod_info = []
    for ia in range(natoms_sc):
        disp_magnitude = np.linalg.norm(displacements[ia])
        disp_max = disp_max_values[ia]
        if disp_magnitude > disp_max:
            # print(f" {ia}th : {disp_magnitude:.4f} -> {disp_max:.4f}")
            scaling_factor = disp_max / disp_magnitude
            new_displacements[ia] *= scaling_factor
            mod_info.append({
                "atom_index": ia, 
                "original_magnitude": disp_magnitude, 
                "adjusted_magnitude": disp_max,
                "displacement_vector": new_displacements[ia],
                })
    
    mod_info = pd.DataFrame(mod_info)
    
    ## Make a new structure with adjusted displacements
    new_structure = structure.copy()
    new_structure.translate(new_displacements)    
    return new_structure, new_displacements, mod_info

