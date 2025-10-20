#
# initialization.py
#
# This script helps to initialize the automation calculation based on the given
# parameters as well as parameters used for the previous calculation.
#
# Copyright (c) 2022 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
import sys
import os
import os.path
import numpy as np
import glob
import pickle

import ase.io
from pymatgen.io.vasp import Kpoints

from auto_kappa.io.phonondb import Phonondb
from auto_kappa import output_directories
from auto_kappa.cui.suggest import klength2mesh
from auto_kappa.structure import get_transformation_matrix, get_supercell, transform_unit2prim

import logging
logger = logging.getLogger(__name__)

def _get_celltype4relaxation(ctype_given, base_dir, natoms_prim=None):
    """ Return the cell type (primitive or unit (conventional) cell) used for 
    the relaxation.
    
    Algorithm
    ---------
    1. If the relaxation calculation has already been started, read the 
    previously used cell type.
    
    2. If it is the initial calculation,
    
    2-1. If the --relaxed_cell (``ctype_suggested``) is not given, the cell type
    for the relaxation is "unitcell".

    2-2. If the --relaxed_cell is given, the cell type is deteremined by this
    option.

    Parameters
    -----------
    ctype_given : string
        cell type given with the options (--relaxed_cell)

    base_dir : string
        base output directory for the automation calculation

    natoms_prim : int
        number of atoms in the primitive cell
    """
    cell_type = None
    
    ### check previous calculations
    dir_relax = base_dir + "/" + output_directories["relax"]
    filenames = [dir_relax + "/vapsun.xml", dir_relax + "/freeze-1/vasprun.xml"]
    
    for fn in filenames:
        if os.path.exists(fn):
            atoms = ase.io.read(fn, format='vasp-xml')
            if len(atoms) == natoms_prim:
                cell_type = "primitive"
            elif len(atoms) > natoms_prim:
                cell_type = "unitcell"
            else:
                msg = " Error: cell type cannot be deteremined."
                msg += " Stop the calculation."
                logger.error(msg)
                sys.exit()
    
    ### determine ``cell_type`` ("unitcell" or "primitive")
    if cell_type is None:
        cell_type = 'unitcell' if ctype_given is None else ctype_given
    
    return cell_type

def read_phonondb(directory):
    """ Read data in Phonondb. Phonondb is used to obtain structures 
    (primitive, unit, and super cells) and k-points.
    
    Note that the definition of transformation matrix in Phononpy (Phonondb)
    is different from that in ASE and Pymatgen.
    With Phonopy definition    : P = mat_u2p_pp   @ U
    With ASE or pmg definition : P = mat_u2p_pp.T @ U
    _pp in ``mat_u2p_pp`` and ``mat_u2s_pp`` denotes Phonopy.
    
    To do
    --------
    If the material is not contained in Phonondb, the trasnformation
    matrices need to be obtained with the same manner as spglib. Pymatgen
    may not be able to be used because of different definitions of the
    transformation matrix in Phonopy and Pymatgen.
    
    """
    phdb = Phonondb(directory)
    
    unitcell = phdb.get_unitcell(format='ase')
    
    if phdb.nac == 1:
        kpts_for_nac = phdb.get_kpoints(mode='nac').kpts[0]
    else:
        kpts_for_nac = None
    
    ##
    matrices = {"primitive": phdb.primitive_matrix,
                "supercell": phdb.scell_matrix}
    
    kpts_all = {"relax": phdb.get_kpoints(mode='relax').kpts[0],
                "harm": phdb.get_kpoints(mode='force').kpts[0],
                "nac": kpts_for_nac}
    
    return unitcell, matrices, kpts_all, phdb.nac

def get_base_directory_name(label, restart=True):
    """ Return the base directory name with an absolute path
    """
    outdir = os.getcwd() + "/"
    if restart:
        """ Case 1: when the job can be restarted """
        outdir += label
    else:
        """ Case 2: when the job cannot be restarted """
        if os.path.exists(label) == False:
            outdir += label
        else:
            for i in range(2, 100):
                outdir = label + "-%d" % i
                if os.path.exists(outdir) == False:
                    break
    return outdir

def _get_previously_used_parameters(outdir, cell_types=None):
    """ Read previously used parameters: transformation matrices and k-meshes.
    
    Parameters
    -----------
    outdir : string
        base output directory
    
    lattice_unit : ndarray, shape=(3,3)
        lattice vectors of unitcell
        Note that this parameter is not that of the relaxed structure.

    Returns
    -------
    params_prev : dict
        keys1 = "trans_matrix" and "kpts"
        keys2 = "relax", "nac", "harm", and "cube"
        "trans_matrix" : transformation matrix
        "kpts" : k-mesh for different calculations
    
    """
    ### prepare dictionaries
    ## calc_types = ["relax", "nac", "harm", "cube"]
    params_prev = {calc_type: {} for calc_type in cell_types.keys()}
    
    ### check k-meshes and transformation matrices
    structures_cell = {}
    # kpts_cell = {}
    calc_types = ["relax", "nac", "harm", "cube"]
    for i, calc_type in enumerate(calc_types):
        if i == 0: 
            # relaxation
            dirs = [outdir + "/relax", 
                    outdir + "/relax/full-1", 
                    outdir + "/relax/full-2",
                    outdir + "/relax/freeze-1",
                    outdir + "/relax/volume"]
        elif i == 1: 
            # NAC
            dirs = [outdir + f"/{output_directories['nac']}"]
        elif i == 2: 
            # harmonic FCs
            dirs = [outdir + f"/{output_directories['harm']['force']}/prist"]
        elif i == 3: 
            # cubic FCs
            dirs = [outdir + "/cube/force_fd/prist",
                    outdir + "/cube/force_lasso/prist"]
        
        cell_type = cell_types[calc_type]
        for dd in dirs:
            
            if dd.endswith("volume"):
                fn_poscar = dd + "/POSCAR.opt"
            else:
                fn_poscar = dd + "/POSCAR"
            
            fn_kpoints = dd + "/KPOINTS"
            
            if os.path.exists(fn_poscar):
                atoms = ase.io.read(fn_poscar, format='vasp')
                params_prev[calc_type]['structure'] = atoms.copy()
                structures_cell[cell_type] = atoms.copy()
                
            if os.path.exists(fn_kpoints):
                ### k-mesh
                kpts = Kpoints.from_file(fn_kpoints).kpts[0]
                if params_prev[calc_type].get('kpts', None) is not None:
                    if not np.allclose(params_prev[calc_type]['kpts'], kpts):
                        if fn_kpoints.startswith("/"):
                            fn_kpoints = "./" + os.path.relpath(fn_kpoints, os.getcwd())
                        msg = f"\n Error: different k-meshes were used for the \"{cell_type}\" structure"
                        msg += f" and {calc_type} calculation."
                        msg += f"\n Check {fn_kpoints}."
                        msg += f"\n {params_prev[calc_type]['kpts']} : previously used"
                        msg += f"\n {kpts} : currently suggested"
                        logger.warning(msg)
                        sys.exit()
                
                params_prev[calc_type]['kpts'] = kpts
                # kpts_cell[cell_type] = params_prev[calc_type]['kpts']
    
    ## Get transformation matrices
    trans_matrices = {}
    if 'primitive' in structures_cell and 'unitcell' in structures_cell:
        pmat = get_transformation_matrix(structures_cell['unitcell'], 
                                         structures_cell['primitive'])
        pmat_inv = np.linalg.inv(pmat)
        pmat_inv_round = np.rint(pmat_inv).astype(int)
        pmat_mod = np.linalg.inv(pmat_inv_round)
        
        diff = np.abs(pmat - pmat_mod)
        if np.max(diff) > 0.1:
            msg = "\n Error: the primitive matrix may not be correct."
            msg += "\n From " + ' '.join(f"{x:.3f}" for x in pmat.flatten())
            msg += "\n To   " + ' '.join(f"{x:.3f}" for x in pmat_mod.flatten())
            logger.error(msg)
            sys.exit()
        trans_matrices['primitive'] = pmat
    
    if 'supercell' in structures_cell and 'unitcell' in structures_cell:
        super_mat = get_transformation_matrix(structures_cell['unitcell'], 
                                              structures_cell['supercell'])
        super_mat_round = np.rint(super_mat).astype(int)
        diff = np.abs(super_mat - super_mat_round)
        if np.max(diff) > 0.1:
            msg = "\n Error: the supercell matrix may not be correct."
            msg += "\n From " + ' '.join(f"{x:.2f}" for x in super_mat.flatten())
            msg += "\n To   " + ' '.join(f"{x:.2f}" for x in super_mat_round.flatten())
            logger.error(msg)
            sys.exit()
        trans_matrices['supercell'] = super_mat
                
    return params_prev, trans_matrices

def _make_structures(unitcell, primitive_matrix=None, supercell_matrix=None):
    """
    Parameters
    ------------
    unitcell : Structure obj

    Return
    -------
    structures : dict of Structure obj

    """
    structures = {}
    structures["unitcell"] = unitcell
    
    if primitive_matrix is not None:
        structures['primitive'] = transform_unit2prim(unitcell, primitive_matrix, format='ase')
        
    if supercell_matrix is not None:
        structures['supercell'] = get_supercell(unitcell, supercell_matrix, format='ase')
        
    #if supercell_matrix3 is not None:
    #    structures["supercell3"] = change_structure_format(
    #            get_supercell(unit_pp, supercell_matrix3), format='ase')
    
    return structures

def get_required_parameters(
        base_directory=None,
        dir_phdb=None, file_structure=None,
        max_natoms=None, 
        k_length=None,
        celltype_relax_given=None,
        dim=3,
        ):
    """ Return the required parameters for the automation calculation: 
    structures, transformation matrices, and k-meshes.
    
    Parameters
    -----------
    base_directory : string
        base output directory
    
    dir_phdb : string
        Directory of Phonondb

    file_structure : string
        structure file name. ``dir_phdb`` or ``file_structure`` must be given
        while, if both of them are given, ``dir_phdb`` is used.
    
    max_natoms(3) : int
        maximum limit of the number of atoms in the supercell for FC2(3)
    
    k_length : float
    celltype_relax_given : string

    Return
    --------
    structures : dict of Structure obj.
        keys=["primitive", "unitcell", "supercell", "supercell3"]
    
    trans_matrices : dict of arrays
        keys=["primitive", "supercell", "supercell3"]
        transformation matrices for the corresponding structures

    kpts_suggested : dict of arrays
        keys=["primitive", "unitcell", "supercell", supercell3]
        kpoints suggested based on the given structures

    kpts_prev : dict of arrays
        keys=["relax", "nac", "harm", "cube"]
        kpoints which have been used for previous calculations
    
    """
    structures = None
    trans_matrices = None
    cell_types = None
    if dir_phdb is not None:
        ### Case 1: Phonondb directory is given.
        structures, trans_matrices, kpts_suggested, nac = (
            read_parameters_from_phonondb(dir_phdb, k_length)
        )
    elif file_structure is not None:
        ### Case 2: A structure is given.
        ### Every required parameters are suggested.
        from auto_kappa.cui.suggest import suggest_structures_and_kmeshes
        
        structures, trans_matrices, kpts_suggested = (
                suggest_structures_and_kmeshes(filename=file_structure,
                                               max_natoms=max_natoms,
                                               k_length=k_length,
                                               dim=dim
                                               ))
        
        ### This part can be modified. So far, NAC is considered for materials
        ### which is not included in Phonondb.
        nac = 2
        
    else:
        """ Case 3: error
        """
        msg = "\n Error: --directory or --file_structure must be given."
        logger.error(msg)
        sys.exit()
    
    ## Check the primitive matrix
    pmat = get_transformation_matrix(structures['unitcell'], structures['primitive'])
    diff = np.abs(pmat - trans_matrices['primitive'])
    if np.max(diff) > 1e-5:
        msg = "\n Error: the primitive matrix is not correct."
        msg += "\n Check the primitive matrix in Phonondb."
        logger.error(msg)
        sys.exit()
    
    ### Cell type used for the relaxation
    ### primitive or unitcell (conventional)
    ### All the elements of ``cell_types`` must be given.
    if len(structures['primitive']) == len(structures['unitcell']):
        cell_type_relax = 'unitcell'
    else:
        cell_type_relax = _get_celltype4relaxation(
                celltype_relax_given, base_directory,
                natoms_prim=len(structures['primitive']))
    
    cell_types = {"relax": cell_type_relax,
                  "nac": "primitive",
                  "harm": "supercell",
                  "cube": "supercell"}
    
    kpts_calc_type = {calc_type: kpts_suggested[cell_type] 
                      for calc_type, cell_type in cell_types.items()}
    
    ### Get previously used transformation matrices, structures, and kpoints
    ### Keys are ['relax', 'nac', 'harm', 'cube']
    params_prev, tmat_prev = _get_previously_used_parameters(base_directory, cell_types=cell_types)
    
    ## Check k-points
    for calc_type, each_param in params_prev.items():
        kpts_prev = each_param.get('kpts', None)
        if kpts_prev is not None:
            if not np.allclose(kpts_calc_type[calc_type], kpts_prev):
                kpts_calc_type[calc_type] = kpts_prev
                # msg  = f"\n Warning: previously used k-mesh for \"{calc_type}\" calculation "
                # msg += "is different from the suggested one."
                # msg += f"\n Suggested : {kpts_calc_type[calc_type]}"
                # msg += f"\n Previous  : {kpts_prev}"
                # logger.info(msg)
    
    ## Check transformation matrices
    for cell_type, mat1 in tmat_prev.items():
        if cell_type in trans_matrices:
            if not np.allclose(trans_matrices[cell_type], mat1, atol=0.01):
                msg  = f"\n Caution: transformation matrix for \"{cell_type}\" "
                msg += "is different from the previously used one."
                array1 = np.array(trans_matrices[cell_type]).flatten()
                array2 = mat1.flatten()
                names = ['Suggested', 'Previous']
                for i, array in enumerate([array1, array2]):
                    msg += f"\n {names[i]:<9s} : "
                    for val in array:
                        msg += f"{val:8.5f} "
                logger.info(msg)
    
    ### Sort atoms in each structure according to the order of the supercell
    sym_list = list(dict.fromkeys(structures['supercell'].get_chemical_symbols()))
    for key in structures:
        if key != 'supercell':
            structures[key] = _sort_atoms_according_to_elements(structures[key].copy(), sym_list)
    
    _save_initial_setting(base_directory, cell_types, structures, trans_matrices, kpts_calc_type)
    
    return cell_types, structures, trans_matrices, kpts_calc_type, nac

def read_parameters_from_phonondb(dir_phdb, k_length):
    """ Read data in the given Phonondb directory and suggest parameters for FC3
    
    Note
    -----
    For Phonondb, the conventional cell is used for both of the
    relaxation and NAC calculations. Auto-kappa, however, can accept
    different cell types while the default types are the conventional and 
    primitive cells for the relaxation and NAC, respectively. Therefore,
    the k-meshes used for Phonondb basically will not be used for the
    automation calculation.
    
    Example
    -------
    
    The Phonondb data directory typically includes::
    
        BORN           FORCE_SETS       INCAR-nac       KPOINTS-force
        KPOINTS-relax  phonon.yaml      POSCAR-unitcell disp.yaml
        INCAR-force    INCAR-relax      KPOINTS-nac     PAW_dataset.txt
        phonopy.conf   POSCAR-unitcell.yaml
    """
    ### Read Phonondb directory
    unitcell, trans_matrices, _, nac = read_phonondb(dir_phdb)
    trans_matrices["unitcell"] = np.identity(3).astype(int)
    
    ### Set structures
    structures = _make_structures(
            unitcell,
            primitive_matrix=trans_matrices['primitive'],
            supercell_matrix=trans_matrices['supercell'],
            )
    
    ### get suggested k-mesh
    kpts_suggested = {
        "primitive": klength2mesh(k_length, structures["primitive"].cell.array),
        "unitcell": klength2mesh(k_length, structures["unitcell"].cell.array),
        "supercell": klength2mesh(k_length, structures["supercell"].cell.array)
        }
    
    return structures, trans_matrices, kpts_suggested, nac

def _save_initial_setting(base_directory, cell_types, structures, trans_matrices, kpts_calc_type):
    """ Save the initial setting to a file.
    """
    dir_out = base_directory + "/init"
    os.makedirs(dir_out, exist_ok=True)
    if dir_out.startswith("/"):
        dir_out = "./" + os.path.relpath(dir_out, os.getcwd())
    
    ## Cell types for each calculation
    file_init = dir_out + "/initial_setting.pkl"
    init_setting = {'cell_types': cell_types,
                    'transformation_matrices': trans_matrices,
                    'kpoints': kpts_calc_type}
    
    with open(file_init, 'wb') as f:
        pickle.dump(init_setting, f)
    # msg = f" Output {file_init}"
    # logger.info(msg)
    
    ## Structures
    for key in structures:
        outfile = dir_out + f"/POSCAR.{key}"
        ase.io.write(outfile, structures[key], format='vasp', vasp5=True, direct=True, sort=False)
        # msg = f" Output {outfile}"
        # logger.info(msg)

def _sort_atoms_according_to_elements(atoms, sym_list):
    """ Sort atoms according to the order of elements in sym_list.
    
    Parameters
    -----------
    atoms : Atoms obj
        structure to be sorted

    sym_list : list of string
        order of elements

    Return
    -------
    atoms_sorted : Atoms obj
        sorted structure

    """
    indices_sorted = []
    for sym in sym_list:
        indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == sym]
        indices_sorted.extend(indices)
    atoms_sorted = atoms[indices_sorted]
    return atoms_sorted

def get_previous_nac(base_dir):
    """ Get previously-used NAC parameter. If it cannot be found, return None.
    """
    ### Check bug during the VASP job for NAC
    dir_nac = base_dir + "/" + output_directories["nac"]
    file_err = dir_nac + "/std_err.txt"
    if os.path.exists(file_err):
        lines = open(file_err, 'r').readlines()
        for line in lines:
            ### If the previous VASP job was stopped due to a bug,
            if "Please submit a bug report." in line:
                msg = ("\n The previous VASP calculation in %s was aborted "
                        "due to a bug." % (dir_nac))
                msg += f'\n See {file_err} for details.'
                logger.info(msg)
                return 0
    
    ### Check the optimal NAC in previous calculations.
    dir_bandos = base_dir + "/" + output_directories["harm"]["bandos"]
    for mode in ["band", "dos"]:
        logfile = dir_bandos + "/%s.log" % mode
        if os.path.exists(logfile) == False:
            continue
        try:
            lines = open(logfile, 'r').readlines()
            for line in lines:
                if "NONANALYTIC =" in line:
                    data = line.split()
                    prev_nac = int(data[2])
                    return prev_nac
        except Exception:
            pass
    ###
    return None

def use_omp_for_anphon(base_dir):
    """ Read log files for previous calculations and check the memory.
    """
    from auto_kappa.alamode.log_parser import exceed_memory
    line1 = base_dir + "/harm/bandos/dos.log"
    line2 = base_dir + "/*/harm/bandos/dos.log"
    line3 = base_dir + "/cube/kappa*/kappa.log"
    for line in [line1, line2, line3]:
        fns = glob.glob(line)
        if len(fns) > 0:
            for fn in fns:
                if exceed_memory(fn):
                    return True
    return False

