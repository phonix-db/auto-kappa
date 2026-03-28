#!/usr/bin/env python
"""
make_vasp_files.py

Generate VASP input files (INCAR, POSCAR, POTCAR, KPOINTS) for structure
relaxation using auto-kappa. The output matches the conditions used for
"./relax/full-1" in the standard auto-kappa workflow.

Usage:
    python make_vasp_files.py [--poscar POSCAR_FILE] [--outdir OUTPUT_DIR]
                              [--k_length K_LENGTH] [--config CONFIG_FILE]
"""
import os
import argparse

from pymatgen.core.structure import Structure

import ase.io
from auto_kappa.structure import change_structure_format
from auto_kappa.cui.suggest import get_unitcell_and_primitive_matrix
from auto_kappa.vasp.kmesh import klength2mesh
from auto_kappa.utils.config import (
    get_vasp_parameters,
    get_vasp_parameters_by_mode,
    get_potcar_setups,
    get_xc,
    load_user_config,
)
from auto_kappa.calculators.vasp import get_vasp_calculator


def main():
    parser = argparse.ArgumentParser(
        description="Generate VASP input files for structure relaxation (relax/full-1)."
    )
    parser.add_argument(
        "--poscar", default="./Structure/POSCAR.Si",
        help="Path to the input POSCAR file (default: ./Structure/POSCAR.Si)",
    )
    parser.add_argument(
        "--outdir", default="./out",
        help="Output directory for VASP files (default: ./out)",
    )
    parser.add_argument(
        "--k_length", type=float, default=20,
        help="K-point length parameter (default: 20)",
    )
    parser.add_argument(
        "--config", default=None,
        help="User config YAML file to override default VASP parameters",
    )
    parser.add_argument(
        "--encut_scale_factor", type=float, default=1.3,
        help="ENCUT scale factor applied to POTCAR ENMAX (default: 1.3)",
    )
    parser.add_argument(
        "--nsw", type=int, default=200,
        help="Number of ionic steps NSW (default: 200)",
    )
    args = parser.parse_args()
    
    # --- Load structure ---
    struct_ase = ase.io.read(args.poscar)
    
    # Get standardized unitcell and primitive matrix
    unitcell, prim_mat = get_unitcell_and_primitive_matrix(struct_ase)
    
    # Get primitive cell
    from auto_kappa.structure import transform_unit2prim
    primitive = transform_unit2prim(unitcell, prim_mat, format='ase')
    
    # --- Load configuration ---
    user_config = load_user_config(args.config)
    vasp_params = get_vasp_parameters(user_config)
    potcar_setups = get_potcar_setups(user_config)
    xc = get_xc(user_config)
    
    # --- VASP parameters for relax-full mode ---
    # Merges: shared -> relax -> relax-full (same as auto-kappa workflow)
    params = get_vasp_parameters_by_mode(vasp_params, mode='relax-full')
    params['nsw'] = args.nsw
    
    # --- K-points ---
    kpts = klength2mesh(args.k_length, primitive.cell.array)

    # --- Create VASP calculator and write input files ---
    calc = get_vasp_calculator(
        params,
        directory=args.outdir,
        atoms=primitive,
        kpts=kpts,
        encut_scale_factor=args.encut_scale_factor,
        setups=potcar_setups,
        xc=xc,
    )

    os.makedirs(args.outdir, exist_ok=True)
    calc.write_input(primitive)

    # --- Summary ---
    print(f"Structure file : {args.poscar}")
    print(f"Output directory: {args.outdir}")
    print(f"XC functional  : {xc}")
    print(f"K-length       : {args.k_length}")
    print(f"K-mesh         : {list(kpts)}")
    print(f"ENCUT factor   : {args.encut_scale_factor}")
    print(f"NSW            : {args.nsw}")
    print(f"Atoms (prim)   : {len(primitive)}")
    print()
    print("Generated files:")
    for f in ["INCAR", "POSCAR", "POTCAR", "KPOINTS"]:
        path = os.path.join(args.outdir, f)
        if os.path.exists(path):
            print(f"  {path}")


if __name__ == "__main__":
    main()
