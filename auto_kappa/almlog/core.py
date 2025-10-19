# 
# core.py
# 
# Created on September 13, 2025
# Copyright (c) 2025 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
# 
import os
import numpy as np
from datetime import datetime

from auto_kappa.almlog.utils import parse_data_line, replace_symbols_to_blank
from auto_kappa.almlog.variables import read_variables

import logging
logger = logging.getLogger(__name__)

class ALMLOG():
    def __init__(self, filename=None):
        
        self.filename = filename
        
        if os.path.exists(self.filename):
            self._info = read_alamode_log(self.filename)
    
    @property
    def sections(self):
        try:
            return list(self._info.keys())
        except:
            return None
    
    def __getitem__(self, key):
        if key in self._info:
            return self._info[key]
        else:
            return None
    
    #######################
    ## Variables section ##
    #######################
    @property
    def variables(self):
        if 'variables' in self._info:
            return self._info['variables']
        else:
            return None
    
    #########################
    ## Frequencies section ##
    #########################
    @property
    def kpoints(self):
        if 'frequencies' in self._info:
            if 'kpoints' in self._info['frequencies']:
                return self._info['frequencies']['kpoints']
        return None
    @property
    def frequencies(self):
        if 'frequencies' in self._info:
            if 'frequencies' in self._info['frequencies']:
                return self._info['frequencies']['frequencies']
        return None
    
    ######################
    ## SYMMETRY section ##
    ######################
    def get_primitive_info(self):
        """ Set primitive structure from symmetry information
        """
        if 'symmetry' not in self._info or 'system' not in self._info:
            return
        structure = self._info['system'].get('structure', None)
        cell_atom_mapping = self._info['symmetry'].get('cell_atom_mapping', None)
        if structure is None or cell_atom_mapping is None:
            return
        prim_positions, prim_symbols = _generate_primitives(structure, cell_atom_mapping)
        return prim_positions, prim_symbols
    
    #######################
    ## Structure section ##
    #######################
    @property
    def primitive(self):
        if 'structure' in self._info:
            if 'primitive' in self._info['structure']:
                return self._info['structure']['primitive']
        return None
    
    # def get_structure(self):
        
    
    # @property
    # def variables(self):
    #     if self._vars is None:
    #         self.set_variables()
    #     return self._vars
    
    # def set_variables(self):
    #     if self._filename is not None and os.path.exists(self._filename):
    #         self._vars = read_alamode_log(self._filename)
    #     else:
    #         self._vars = None

def read_alamode_log(filename):
    """ Read ALAMODE log file and extract information 
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    log_info = {}
    
    ## Read basic information
    for line in lines:
        
        line = line.strip().lower()
        data = line.split()
        if len(data) == 0:
            continue
        
        if "program" in line:
            log_info['program'] = parse_data_line(line, key="program", idx_dist=1, dtype=str)
        if "ver." in line:
            log_info['version'] = parse_data_line(line, key="ver.", idx_dist=1, dtype=str)
        if "version" in line:
            log_info['version'] = parse_data_line(line, key="version", idx_dist=1, dtype=str)
        if "job started at" in line:
            time_str = line.split(" at ")[-1].strip() # Note blanks before and after 'at' are important!!
            log_info['start_time'] = datetime.strptime(time_str, "%a %b %d %H:%M:%S %Y")
        if "job finished at" in line:
            time_str = line.split(" at ")[-1].strip()
            log_info['finish_time'] = datetime.strptime(time_str, "%a %b %d %H:%M:%S %Y")
        if "number of mpi processes" in line:
            line = replace_symbols_to_blank(line)
            log_info['nproc'] = parse_data_line(line, index=-1, dtype=int)
        if "number of openmp threads" in line:
            line = replace_symbols_to_blank(line)
            log_info['nthreads'] = parse_data_line(line, index=-1, dtype=int)
        
        ###
        title = "Phonon frequencies below"
        if line.lower().startswith(title.lower()):
            from auto_kappa.almlog.frequencies import read_frequencies
            try:
                log_info['frequencies'] = read_frequencies(lines)
            except Exception as e:
                logger.info(f"Warning: Failed to read frequencies section in {filename}: {e}")
                log_info['frequencies'] = None
    
    ## Duration [seconds]
    if 'start_time' in log_info and 'finish_time' in log_info:
        log_info['duration'] = (log_info['finish_time'] - 
                                log_info['start_time']).total_seconds()
    
    ## Read variables
    log_info['variables'] = read_variables(lines)
    
    ## Prepare sections, which do not have blank lines
    section_lines = _divide_sections(lines)
    
    ## Read sections, which underlined by "==========="
    if 'crystal structure' in section_lines:
        from auto_kappa.almlog.structure import read_structure
        log_info['structure'] = read_structure(section_lines['crystal structure'])
    if 'system' in section_lines:
        from auto_kappa.almlog.system import read_system
        log_info['system'] = read_system(section_lines['system'])
    if 'symmetry' in section_lines:
        from auto_kappa.almlog.symmetry import read_symmetry
        log_info['symmetry'] = read_symmetry(section_lines['symmetry'])
    if 'k points' in section_lines:
        from auto_kappa.almlog.kpoint import read_kpoints
        log_info['k points'] = read_kpoints(section_lines['k points'])
    if 'force constant' in section_lines:
        from auto_kappa.almlog.fcs import read_fcs
        log_info['force constant'] = read_fcs(section_lines['force constant'])
    if 'dynamical matrix' in section_lines:
        from auto_kappa.almlog.dmatrix import read_dmatrix
        log_info['dynamical matrix'] = read_dmatrix(section_lines['dynamical matrix'])
    if 'interaction' in section_lines:
        from auto_kappa.almlog.interaction import read_interaction
        log_info['interaction'] = read_interaction(section_lines['interaction'])
    
    return log_info

def _divide_sections(lines):
    """ Divide lines into sections based on headers which are 
    underlined by '=' characters
    """
    sections = {}
    header = None
    nl = len(lines)
    for il, line in enumerate(lines):
        line = lines[il].strip()
        if not line:
            continue
        
        if line == "-" * len(line):
            header = None
        
        if line == "=" * len(line) and il > 0:
            header = lines[il-1].strip().lower()
            sections[header] = []
            continue
        
        if header is None:
            continue
        
        if il < nl - 1:
            line1 = lines[il+1].strip()
            if line1 and line1 == "=" * len(line1):
                continue
        
        sections[header].append(line)    
    
    return sections

def _generate_primitives(structure, cell_atom_mapping):
    """ Generate primitive structures from supercell structure and cell-atom mapping
    """
    key0 = list(cell_atom_mapping.keys())[0]
    natoms_prim = len(cell_atom_mapping[key0])
    ncells = len(structure) // natoms_prim
    
    ### Get atomic positions and symbols
    positions = np.zeros((ncells, natoms_prim, 3))
    symbols = []
    for idx_cell, indices in cell_atom_mapping.items():
        symbols.append([])
        for iat, idx_atom in enumerate(indices):
            positions[idx_cell-1, iat, :] = structure.get_positions()[idx_atom-1, :]
            symbols[-1].append(structure.get_chemical_symbols()[idx_atom-1])
    
    # ### Find primitive cell basis vectors
    # scell = structure.cell.array
    # idx_atom = 0
    # site_positions = []
    # pos0 = positions[0][idx_atom]
    # for icell in range(ncells):
    #     site_positions.append(positions[icell][idx_atom])
    #     print(positions[icell][idx_atom])
    # for vec in scell:
    #     site_positions.append(pos0 + vec)
    
    # # pcell = find_primitive_cell_basis(np.array(site_positions))
    
    return positions, symbols
    
# def find_primitive_cell_basis(site_positions: np.ndarray, tol=1e-3) -> np.ndarray:
#     """ Find primitive cell basis vectors from translations and atomic positions
    
#     Parameters:
#     - site_positions (np.ndarray): List of site positions in different primitive cells in the supercell (M, 3)
#     - positions (np.ndarray): Atomic positions of a primitive cell in the supercell (M, 3)
#     - tol (float): Tolerance for checking if coordinates are within
    
#     Returns:
#     - np.ndarray: Basis vectors of the primitive cell (3, 3)
#     """
#     origin = site_positions[0]
#     diffs = site_positions - origin
        
#     for vecs in combinations(diffs[1:], 3):
#         mat = np.array(vecs).T
#         if abs(np.linalg.det(mat)) > tol:
#             return mat.T
    
#     raise ValueError("No valid primitive cell basis found.")
