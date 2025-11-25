#
# utils.py
#
# Copyright (c) 2022 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
# import numpy as np
# from scipy.spatial.distance import cdist
# from scipy.optimize import linear_sum_assignment

import numpy as np

import ase
from ase.data import atomic_numbers, atomic_masses
import pymatgen.core.structure as str_pmg
from phonopy.structure.atoms import PhonopyAtoms

import logging
logger = logging.getLogger(__name__)

def get_transformation_matrix(from_structure, to_structure):
    """ Return the transformation matrix from `from_structure` to `to_structure`
    Transformation matrix is defined as follows, which is consistent with Phonopy.
    See also https://phonopy.github.io/phonopy/setting-tags.html
    
    Args
    -----
    from_structure : Structure object
        initial structure
    to_structure : Structure object
        final structure
    
    Note
    ----
    T = (from_cell.T)^-1 @ (to_cell.T)
    (to_cell.T) = (from_cell.T) @ T
    (to_cell) = T.T @ (from_cell)
    """
    from_structure = change_structure_format(from_structure, format='ase')
    to_structure = change_structure_format(to_structure, format='ase')
    from_cell = from_structure.cell.array
    to_cell = to_structure.cell.array
    trans_mat = np.linalg.inv(from_cell.T) @ to_cell.T
    return trans_mat

def change_structure_format(structure, format='pymatgen-IStructure'):
    """ Convert from arbitrary crystal format to an arbitrary crystal format

    Args
    -----
    structure (Structure object):

    Return
    ------
    pymatgen's IStructure object
    """
    if (isinstance(structure, str_pmg.Structure) or
        isinstance(structure, str_pmg.IStructure)):
        
        ## from pymatgen's (I)Structure object
        lattice = structure.lattice.matrix
        all_symbols = []
        for specie in structure.species:
            all_symbols.append(specie.name)
        coords = structure.frac_coords
    
    elif isinstance(structure, ase.Atoms):
        lattice = structure.cell
        all_symbols = structure.get_chemical_symbols()
        coords = structure.get_scaled_positions()
    
    elif isinstance(structure, PhonopyAtoms):
        
        try:
            lattice = structure.cell
            all_symbols = structure.get_chemical_symbols()
            coords = structure.get_scaled_positions()
        except AttributeError:
            lattice = structure.cell
            all_symbols = structure.chemical_symbols
            coords = structure.scaled_positions
        
    else:
        logger.error(" Structure type {} is not supported".format(
            type(structure)))
        exit()
    
    ## set atomic numbers
    numbers = []
    for name in all_symbols:
        numbers.append(atomic_numbers[name])
    
    ## return the structure
    form = format.lower()
    if 'pymatgen' in form or 'pmg' in form:
        
        if format == 'pymatgen-structure' or format == 'pmg-structure':
            return str_pmg.Structure(lattice, all_symbols, coords)
        else:
            return str_pmg.IStructure(lattice, all_symbols, coords)
        
    elif form == 'ase' or form == 'atoms':
        
        return ase.Atoms(
            cell=lattice,
            scaled_positions=coords,
            numbers=numbers,
            pbc=True
            )
    
    elif form == 'phonopyatoms' or form == 'phonopy':
        
        masses = []
        for name in all_symbols:
            masses.append(atomic_masses[atomic_numbers[name]])
        
        return PhonopyAtoms(
            cell=lattice,
            symbols=all_symbols,
            masses=masses,
            scaled_positions=coords,
            pbc=True
            )
    else:
        
        logger.warning(" Structure type '{}' is not supported. "\
                "The structure type did not changed.".format(format))
        return structure

