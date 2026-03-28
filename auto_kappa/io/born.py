#
# born.py
#
# This file helps to generate BORNINFO file for ALM calculation
# from vasprun.xml and force constants XML file.
#
# Copyright (c) 2025 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
import os
import sys
import numpy as np
import xmltodict

from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

from auto_kappa.io.fcs import FCSxml
# from auto_kappa.structure import change_structure_format

import matplotlib.pyplot as plt
from auto_kappa.plot import set_matplot, set_axis

import logging
logger = logging.getLogger(__name__)

class CHARGE:
    def __init__(self):
        self.structure = None
        self.dielectric_tensor = None
        self.born_charges = None

class BORNINFO:
    """ Class for Born effective charge and dielectric tensor
    
    Args
    -----
    file_vasp (str) : vasprun.xml containing Born effective charge
    file_fcs (str) : force constants XML file
    prim_fcs (ase.Atoms or pymatgen.Structure) : primitive structure set in force constants file
    tol (float) : tolerance for atom mapping from vasprun.xml
    """
    def __init__(self, file_vasp, file_fcs=None, prim_fcs=None):
        self.file_vasp = file_vasp
        self._prim_vasp = None
        self.file_fcs = file_fcs
        self._prim_fcs = prim_fcs
        
        self.vasp = CHARGE()
        self.alm = CHARGE()
        
        if os.path.exists(self.file_vasp):
            self._parse_vasprun()
        else:
            raise ValueError(f"Cannot find {self.file_vasp}.")
        
        if os.path.exists(self.file_fcs):
            self.set_prim_fcs()        
            self.set_born_info()
        else:
            raise ValueError(f"Cannot find {self.file_fcs}.")
        
    def _parse_vasprun(self):
        vasprun = Vasprun(self.file_vasp, parse_potcar_file=False)
        self.vasp.structure = vasprun.structures[0]
        self.vasp.dielectric_tensor = vasprun.epsilon_static
        self.vasp.born_charges = get_born_charges_from_vasprun(self.file_vasp)
    
    def set_prim_fcs(self):
        if self.file_fcs is None:
            raise ValueError("file_fcs must be provided.")
        fcs = FCSxml(self.file_fcs)
        self.alm.structure = Structure(
            coords=fcs.primitive_positions,
            species=fcs.primitive_symbols,
            lattice=self.prim_vasp.lattice,
            coords_are_cartesian=True)
    
    @property
    def prim_vasp(self):
        try:
            return self.vasp.structure
        except:
            return None
    @property
    def prim_fcs(self):
        if self.alm.structure is None and self.file_fcs is not None:
            self.set_prim_fcs()
        return self.alm.structure
    
    def get_transformation(self, params={"ltol": 0.05, "stol": 0.1, "angle_tol": 0.5}):
        ## Get transformation from VASP primitive to FC primitive
        matcher = StructureMatcher(primitive_cell=False, **params)
        if not matcher.fit(self.vasp.structure, self.alm.structure):
            msg = f"\n Structures in vasprun.xml and force constants file do not match."
            msg += f"\n - VASP file       : {self.file_vasp}"
            msg += f"\n - Force constants : {self.file_fcs}"
            logger.error(msg)
            sys.exit()
        
        tmat_v2a, ftrans_v2a, map_v2a = \
            matcher.get_transformation(self.vasp.structure, self.alm.structure)
        
        ## Get rotation matrix from VASP to FC primitive
        L_vasp = self.vasp.structure.lattice.matrix
        L_fc = self.alm.structure.lattice.matrix
        R_v2a = np.linalg.inv(L_fc) @ L_vasp
        return tmat_v2a, ftrans_v2a, map_v2a, R_v2a
        
    def set_born_info(self):
        """ Get Born effective charge and dielectric tensor for ALM calculation
        1. Read vasprun.xml to get Born effective charge and dielectric tensor
        2. Get transformed dielectric tensor and Born effective charge. 
        """
        # tmat : transformation (supercell) matrix
        # fracT : fractional translation vector
        # map   : mapping of atom indices from VASP to ALM
        tmat_v2a, fracT_v2a, map_v2a, R_v2a = self.get_transformation()
        self.alm.dielectric_tensor = _transform_dielectric_tensor(self.vasp.dielectric_tensor, R_v2a)
        self.alm.born_charges = _transform_born_charges(self.vasp.born_charges, R_v2a, map_v2a)
        # return self.alm.dielectric_tensor, self.alm.born_charges
        
    def write(self, outfile='BORNINFO'):
        """ Write BORNINFO file for ALM calculation
        """
        if self.alm.dielectric_tensor is None or self.alm.born_charges is None:
            self.set_born_info()
        write_born_info(self.alm.dielectric_tensor, self.alm.born_charges, outfile=outfile)
        # if outfile.startswith('/'):
        #     outfile = "./" + os.path.relpath(outfile, os.getcwd())
        # logger.info(f" Output {outfile}")


def plot_tensor_ellipsoid(ax, Z, center=(0,0,0), color='orange', alpha=0.6):
    """ Plot ellipsoid representing a symmetric 3x3 tensor Z on a given ax
    """
    ## Eigen values and vectors
    vals, vecs = np.linalg.eigh(Z)
    ## Semi-axis lengths
    rx, ry, rz = np.sqrt(np.abs(vals))
    
    ## Create data for "unrotated" ellipsoid
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))
    
    coords = np.stack([x, y, z], axis=-1)  # shape: (N, M, 3)
    rotated = coords @ vecs.T + np.array(center)
    x, y, z = rotated[..., 0], rotated[..., 1], rotated[..., 2]
    
    ## Plot
    ax.plot_surface(x, y, z, rstride=2, cstride=2, color=color, alpha=alpha)

def make_figure_bec(charges, centers, figname='fig.png',
                    dpi=300, fontsize=7, fig_width=2.8, aspect=0.9, lw=0.5, ms=2.0):

    set_matplot(fontsize=fontsize)
    fig = plt.figure(figsize=(fig_width, aspect*fig_width))
    
    # ax = plt.subplot()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('')
    ax.set_ylabel('')

    for (charge, center) in zip(charges, centers):
        plot_tensor_ellipsoid(ax, charge, center=center, color='orange', alpha=0.3)
        
    set_axis(ax)
    
    fig.savefig(figname, dpi=dpi, bbox_inches='tight')
    # print(" Output", figname)
    return fig

def _transform_structure(struct2, R, t, mapping):
    # Function to transform structure
    new_lattice = Lattice(np.dot(struct2.lattice.matrix, R))
    new_frac_coords = np.array([struct2[i].frac_coords for i in mapping])
    new_species = [struct2[i].specie for i in mapping]
    new_frac_coords = (new_frac_coords @ R + t) % 1.0
    new_struct = Structure(
        lattice=new_lattice,
        species=new_species,
        coords=new_frac_coords,
        to_unit_cell=True,
        coords_are_cartesian=False
    )
    return new_struct

def _transform_dielectric_tensor(tensor_a, R_a2b):
    """ Transform dielectric tensor from structure a to b
    Args
    -----
    tensor_a (array) : dielectric tensor for structure a
    R_a2b (array) : rotation matrix from structure a to b
    
    Returns
    -------
    tensor_b (array) : dielectric tensor for structure b
    """
    R = np.array(R_a2b)
    return R.T @ np.array(tensor_a) @ R

def _transform_born_charges(born_a, R_a2b, map_a2b):
    """ Transform Born effective charges from structure a to b
    Args
    -----
    born_a (array) : Born effective charges for structure a
    R_a2b (array) : rotation matrix from structure a to b
    map_a2b (list) : mapping of atom indices from structure a to b,
                     where map_a2b[idx_a] = idx_b

    Returns
    -------
    born_b (array) : Born effective charges for structure b
    """
    R = np.array(R_a2b)
    # map_a2b[idx_a] = idx_b : invert to get inv_map[idx_b] = idx_a
    inv_map = np.argsort(map_a2b)
    born_a_arr = np.array(born_a)[inv_map]  # (N, 3, 3) reordered
    return R.T @ born_a_arr @ R             # batched matmul: (3,3)@(N,3,3)@(3,3)

def get_born_charges_from_vasprun(filename):
    """
    Args
    ---------
    filename (str) : vasprun.xml
    """
    out = None
    with open(filename, 'r') as f:
        out = xmltodict.parse(f.read())
    array = out['modeling']['calculation']['array']
    
    ## Check
    if array['@name'] != 'born_charges':
        logger.warning(f" Cannot find born_charges in {filename}.")
        return None
    
    ## number of atoms
    natoms = len(array['set'])
    
    ## Read contents in "born_charges"
    borns = []
    for i in range(natoms):
        
        ### modified on April 17, 2023
        if natoms == 1:
            lines = array['set']['v']
        else:
            lines = array['set'][i]['v']
        
        born = np.zeros((3,3))
        for i1, line in enumerate(lines):
            data = line.split()
            for i2 in range(3):
                born[i1,i2] = float(data[i2])
        borns.append(born)
    return borns

def write_born_info(dielectric_tensor, born_charges, outfile='BORNINFO'):
    
    lines = []
    
    ## dielectric tensor
    for j in range(3):
        lines.append("%18.13f " * 3 % tuple(dielectric_tensor[j]))
    
    ## Born effective charge
    for ia in range(len(born_charges)):
        for j in range(3):
            lines.append("%18.13f " * 3 % tuple(born_charges[ia][j]))
    
    lines.append("")
    f = open(outfile, 'w')
    f.write('\n'.join(lines))

def read_born_info(filename):
    """ Read BORNINFO file and return dielectric tensor and Born effective charges
    Args
    -----
    filename (str) : BORNINFO file name
    
    Returns
    -------
    dielectric_tensor (array) : dielectric tensor
    born_charges (array) : Born effective charges
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    ## Dielectric tensor
    dielectric_tensor = []
    for j in range(3):
        dielectric_tensor.append([float(x) for x in lines[j].split()])
    dielectric_tensor = np.array(dielectric_tensor)
    
    ## Born effective charges
    born_charges = []
    n_atoms = (len(lines) - 3) // 3
    for ia in range(n_atoms):
        Z = []
        for j in range(3):
            Z.append([float(x) for x in lines[3 + ia*3 + j].split()])
        born_charges.append(np.array(Z))
    
    return dielectric_tensor, np.array(born_charges)
