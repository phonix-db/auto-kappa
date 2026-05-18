#!/usr/bin/env python
import os
import numpy as np
import argparse

import ase.io

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from auto_kappa.apdb import ApdbVasp
from auto_kappa.structure.crystal import get_primitive_structure_spglib

def main(options):
    
    ### prepare structures
    unit = ase.io.read(options.filename, format='vasp')
    
    prim = get_primitive_structure_spglib(unit)
    
    ### get the primitive matrix
    pmat = np.dot(np.linalg.inv(unit.cell), prim.cell)
    
    ### set ApdbVasp
    apdb = ApdbVasp(unit, primitive_matrix=pmat, scell_matrix=np.identity(3))
    
    ### make calculator
    os.makedirs(options.outdir, exist_ok=True)
    kpts = [4, 4, 4]
    calc = apdb.get_calculator('relax', options.outdir, kpts)
    
    ### Make VASP input files
    calc.write_input(prim)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input parameters')
    parser.add_argument('-f', '--filename', dest='filename', type=str,
                        default="POSCAR-unitcell", help="input file name")
    parser.add_argument('-o', '--outdir', dest='outdir', type=str,
                        default='./out', help="output directory name")
    args = parser.parse_args()
    main(args)
