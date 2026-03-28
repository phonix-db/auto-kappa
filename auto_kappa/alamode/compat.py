# 
# compat.py
# 
# Ensure compatibility across updated versions
# 
# Author      : M. Ohnishi
# Created on  : April 23, 2025
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
import glob
import subprocess
import shutil
import multiprocessing as mp
import ase.io
from ase.geometry import get_distances

from auto_kappa.io import AlmInput
from auto_kappa.io.born import BORNINFO, read_born_info
from auto_kappa.structure.crystal import (
    change_structure_format, inverse_transformation, get_primitive_structure_spglib)
from auto_kappa.structure.comparison import generate_mapping_s2p, match_structures

import logging
logger = logging.getLogger(__name__)

def _custom_sort_key(item):
    key = item[0]
    if key == 'prist':
        return (0, 0)
    try:
        numeric_key = int(key)
        return (1, numeric_key)
    except (ValueError, TypeError):
        return (2, str(key))

def _get_previously_suggested_structures(outdir):
    """ Read the previously suggested structures and their displacement patterns 
    to align with those from the previous version.
    """
    line = f"{outdir}/*/POSCAR"
    structures = {}
    for fn in glob.glob(line):
        try:
            key = fn.split("/")[-2]
            structures[key] = ase.io.read(fn)
        except:
            continue    
    return dict(sorted(structures.items(), key=_custom_sort_key))

def _match_wrapper(args):
    new_key, new_structure, prev_structures = args
    for prev_key, prev_structure in prev_structures.items():
        match = match_structures(new_structure, prev_structure,
                                 ignore_order=False, verbose=False)
        if match:
            return (new_key, prev_key)
    return None

def _parallel_structure_match(new_structures, prev_structures, nprocs=1):
    map_new2prev = {}
    assigned_keys = set()
    
    ## Each task will check if the new structure matches any of the previous structures
    tasks = [
        (new_key, new_structure, {
            k: v for k, v in prev_structures.items() if k not in assigned_keys
        })
        for new_key, new_structure in new_structures.items()
    ]

    with mp.Pool(processes=nprocs) as pool:
        results = pool.map(_match_wrapper, tasks)

    for res in results:
        if res:
            new_key, prev_key = res
            if prev_key not in assigned_keys:
                map_new2prev[new_key] = prev_key
                assigned_keys.add(prev_key)

    return map_new2prev

def adjust_keys_of_suggested_structures(new_structures, outdir, dim=3, nprocs=None):
    """ Sort the suggested structures and their displacement patterns 
    to align with those from the previous version.
    
    Args
    ------
    new_structures : dict
        The new structures
    
    outdir : str
        The output directory where the previous structures are stored.
    
    tolerance : float
        The tolerance for the distance between the new and previous structures.
    
    mag : float
        The magnitude of the atom displacement.
    """
    prev_structures = _get_previously_suggested_structures(outdir)
    
    if dim == 2:
        msg = "\n Not supported for 2D structures yet."
        logger.error(msg)
        sys.exit()
    
    logger.info("\n Check previous structures (Previous: %d, New: %d)...", 
                len(prev_structures), len(new_structures))
    
    ## ver.2: Parallel
    nprocs = nprocs if nprocs is not None else mp.cpu_count()
    map_new2prev = _parallel_structure_match(new_structures, prev_structures, nprocs=nprocs)
    
    ### Make a dict of structures with adjusted keys
    ## structures contained in prev_structures
    # avail_keys = [str(key) for key in list(new_structures.keys())]
    adjusted_key_structures = {}
    for new_key, prev_key in map_new2prev.items():
        adjusted_key_structures[prev_key] = new_structures[new_key]
        
    ## maximum key in prev_structures
    prev_key_max = 0
    for prev_key in prev_structures.keys():
        try:
            prev_key_max = max(prev_key_max, int(prev_key))
        except:
            pass
    
    ## new structures
    key_cur = prev_key_max + 1
    for new_key, new_structure in new_structures.items():
        if new_key not in map_new2prev:
            if new_key == 'prist':
                key = 'prist'
            else:
                key = str(key_cur)
                key_cur += 1
            adjusted_key_structures[key] = new_structure
    return adjusted_key_structures

def get_previously_calculated_structure(dir_forces, include_pristine=True):
    """ Get already calculated structures with displacement.
    """
    from auto_kappa.io.vasp import wasfinished as wasfinished_vasp
    line = f"{dir_forces}/*/vasprun.xml"
    fns = glob.glob(line)
    structures = {}
    for fn in fns:
        dirname = os.path.dirname(fn)
        key = dirname.split("/")[-1]
        if key == 'prist' and include_pristine == False:
            continue
        if wasfinished_vasp(dirname):
            structures[key] = ase.io.read(fn, format='vasp-xml')
    
    return dict(sorted(structures.items(), key=_custom_sort_key))

def get_number_of_same_structures(structures1, structures2):
    """ Get the number of same structures.
    
    Args
    ------
    structures1 : dict
        The first set of structures.

    structures2 : dict
        The second set of structures.
    """
    count = 0
    for key1, struct1 in structures1.items():
        if key1 in structures2:
            struct2 = structures2[key1]
            if match_structures(struct1, struct2, ignore_order=False, verbose=False):
                count += 1
        else:
            for key2, struct2 in structures2.items():
                if key2 == key1:
                    continue
                if match_structures(struct1, struct2, ignore_order=False, verbose=False):
                    count += 1
                    break
    return count

def check_directory_name_for_pristine(path_force, pristine):
    """ Check the directory name for the pristine structure.
    
    Args
    ------
    dir_force : str
        The directory name for the force calculation.
    
    pristine : ase.Atoms
        The pristine structure.
    """
    ## If "prist" directory already exists, return
    dir_prist = os.path.join(path_force, 'prist')
    if os.path.exists(dir_prist):
        return
    
    if os.path.exists(path_force) == False:
        return
    
    ## Get the directory names
    dirs_tmp = [entry.name for entry in os.scandir(path_force) if entry.is_dir()]
    
    labels = []
    for li in dirs_tmp:
        dir_i = os.path.join(path_force, li)
        fn = dir_i + "/POSCAR"
        if os.path.exists(fn):
            labels.append(li)
    
    ## Check the directory name for the pristine structure
    for lab in labels:
        fn = os.path.join(path_force, lab, "POSCAR")
        structure = ase.io.read(fn)
        # if same_structures(structure, pristine):
        if match_structures(structure, pristine):
            if lab != 'prist':
                dir1 = os.path.join(path_force, lab)
                msg = (
                    f"\n The directory name for the pristine structure "
                    f"was changed from \"{lab}\" to \"prist\".")
                logger.info(msg)
                os.rename(dir1, dir_prist)
                return 1
    
    return 0
    
def was_primitive_changed(struct_tmp, tol_prev, tol_new):
    """ Check whether the primitive cell was changed.
    
    Args
    ------
    structure : ase.Atoms
        The structure to be checked.
    
    tol_prev : float
        The tolerance for the previous version.
    
    tol_new : float
        The tolerance for the new version.
    """
    structure = change_structure_format(struct_tmp, format='pmg')
    
    prim_prev = structure.get_primitive_structure(tolerance=tol_prev)
    prim_new = structure.get_primitive_structure(tolerance=tol_new)
    
    if len(prim_prev) != len(prim_new):
        return True
    else:
        return False

def was_tolerance_changed(file_prev, new_params):
    """ Check whether the new parameters are different from the previous ones.
    
    Args
    ------
    file_prev : str
        The name of the previous ALAMODE input file.
    
    new_params : dict
        The new parameters to be compared with the previous ones.
    """
    if os.path.exists(file_prev) == False:
        return False
    
    ## read previous parameters
    prev_params = AlmInput().from_file(file_prev).as_dict()
    prev_tol = prev_params.get('tolerance')
    new_tol = new_params.get('tolerance')
    if prev_tol != new_tol:
        msg = f"\n Tolerance was changed from {prev_tol} to {new_tol}."
        logger.info(msg)
        return True
    else:
        return False

def backup_previous_results(directory, propt, prefix=None):
    """ Backup the previous results for ALAMODE in directory.
    """
    if os.path.exists(directory) == False:
        return
    
    ##
    if propt == 'suggest':
        cmd = f"rm {directory}/*"
        subprocess.run(cmd, shell=True)
    elif propt in ['fc2', 'fc3']:
        cmd = f"rm {directory}/{prefix}.* {directory}/{propt}.* {directory}/std_err.txt"
        subprocess.run(cmd, shell=True)
    elif propt in ['band', 'evec_commensurate', 'cv', 'kappa']:
        ##
        ## Make a backup directory and 
        ## all existing files are moved to the backup directory
        ##
        count = 1
        while True:
            out_backup = f"{directory}/backup{count}.tar.gz"
            if os.path.exists(out_backup) == False:
                break
            count += 1
        
        dir_backup = f"{directory}/backup{count}"
        
        ## Make a directory for backup
        os.makedirs(dir_backup, exist_ok=True)
        
        ## Move all files to the backup directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))] 
        for f in files:
            if 'BORNINFO' in f:
                continue
            cmd = f"mv {directory}/{f} {directory}/backup{count}/"
            subprocess.run(cmd, shell=True)
        
        ## Compress the backup directory
        shutil.make_archive(
            base_name=dir_backup, 
            format='gztar', 
            root_dir=directory,
            base_dir=f"backup{count}")
        shutil.rmtree(dir_backup)
        
    else:
        print(directory, propt, prefix)
        cmd = f"\n Backup process for {propt} was not implemented yet."
        logger.info(cmd)
        sys.exit()

    # backup_dir = directory + "_backup"
    # if os.path.exists(backup_dir):
    #     os.system(f"rm -rf {backup_dir}")
    
    # os.system(f"mv {directory} {backup_dir}")

def check_previous_borninfo(dir_work, born_xml, fc2xml=None, fc3xml=None, fcsxml=None):
    """ Check the previous BORNINFO file and update it if necessary.
    """
    ## Get existing force constants files
    files = [fn for fn in [fc2xml, fc3xml, fcsxml] if fn is not None]
    if len(files) == 0:
        msg = "\n Error: fc2xml, fc3xml, or fcsxml must be given."
        logger.error(msg)
        return None
    
    ## Read dielectric tensor and Born effective charge from vasprun.xml
    ## and transform for ALAMODE calculation
    if os.path.isabs(files[0]):
        fn_fcs = files[0]
    else:
        fn_fcs = os.path.join(dir_work, files[0])
    
    if os.path.exists(fn_fcs) == False:
        # msg = f"\n Error: Cannot find {fn_fcs}."
        # logger.error(msg)
        return None
    
    born = BORNINFO(born_xml, file_fcs=fn_fcs)
    eps = born.alm.dielectric_tensor
    charges = born.alm.born_charges
    
    ## Check the previous BORNINFO file
    outfile = dir_work + "/BORNINFO"
    if os.path.exists(outfile):
        eps_prev, charges_prev = read_born_info(outfile)
        diff_eps = np.abs(eps - eps_prev)
        diff_charges = np.abs(charges - charges_prev)
        
        if np.max(diff_eps) > 1e-4 or np.max(diff_charges) > 1e-4:
            
            # n1, n2 = diff_eps.shape
            # for i in range(n1):
            #     print(" ".join(["%10.6f" % diff_eps[i,j] for j in range(n2)]))
            # n1, n2, n3 = diff_charges.shape
            # for i in range(n1):
            #     for j in range(n2):
            #         print(" ".join(["%10.6f" % diff_charges[i,j,k] for k in range(n3)]))
            
            logger.info(f"\n Error in {dir_work}/BORNINFO")
            
            ## Backup the previous BORNINFO file
            count = 1
            while os.path.exists(out_backup := f"{outfile}.{count}"):
                count += 1    
            shutil.move(outfile, out_backup)
            
            ## Make "./box" directory if it does not exist
            count = 1
            while os.path.exists(dir_box := f"{dir_work}/box{count}"):
                count += 1    
            if dir_box.startswith("/"):
                dir_box = os.path.relpath(dir_box, os.getcwd())
            
            ## Move log files to the box directory
            dir1 = os.path.relpath(dir_work, os.getcwd())
            names = glob.glob(f"{dir1}/*")
            if len(names) > 0:
                os.makedirs(dir_box, exist_ok=True)
            for name in names:
                if name.startswith(f"{dir1}/box"):
                    continue
                shutil.move(name, os.path.join(dir_box, os.path.basename(name)))
                cmd = f" >>> move {name} to {dir_box}/"
                logger.info(cmd)

def _relative_path(filename):
    if filename.startswith("/"):
        return os.path.relpath(filename, os.getcwd())
    else:
        return filename

def check_previous_structures(outdirs, primitive, unitcell, prim_mat=None, sc_mat=None):
    """ Check the previous structures used for force and FCs calculations.
    If multiple structures are found, the first one is used.
    
    Args
    ------
    outdirs : dict
        The output directories for force and FCs calculations.
    primitive : ase.Atoms
        The primitive cell.
    unitcell : ase.Atoms
        The unit cell.
    prim_mat : array-like
        The transformation matrix from the unit cell to the primitive cell with the Phonopy representation.
    sc_mat : array-like
        The transformation matrix from the primitive cell to the supercell with the Phonopy representation.
    """
    from auto_kappa.io.fcs import FCSxml
    
    ref_sc = None
    
    ## Supercell used for force and FCs calculations
    lines = [
        outdirs.get('harm', {}).get('force', None) + "/prist/POSCAR",
        outdirs.get('harm', {}).get('force', None) + "/*.xml",
        outdirs.get('cube', {}).get('force_fd', None) + "/prist/POSCAR",
        outdirs.get('cube', {}).get('force_lasso', None) + "/prist/POSCAR",
        outdirs.get('cube', {}).get('force_fd', None) + "/*.xml",
        outdirs.get('cube', {}).get('lasso', None) + "/*.xml"
    ]
    ref_sc = None
    for line in lines:
        
        fns = glob.glob(line)
        if len(fns) == 0:
            continue
        fn = fns[0]
        
        if fn.endswith('.xml'):
            fcs = FCSxml(fn)
            sc = fcs.supercell
        else:
            sc = ase.io.read(fn)
        
        if sc is None:
            continue
        
        if ref_sc is None:
            ref_file = fn
            ref_sc = sc.copy()
        else:
            D, D_len = get_distances(ref_sc.get_positions(), sc.get_positions(), ref_sc.cell, pbc=True)
            diff = np.diag(D_len)
            if np.max(np.abs(diff)) > 1e-4:
                ref_fn = _relative_path(ref_file)
                fn = _relative_path(fn)
                msg = "\n Error: max. diff. = %.6f between" % (np.max(np.abs(diff)))
                msg += f"\n {ref_fn} and {fn}"
                logger.error(msg)
                sys.exit()
    
    if ref_sc is None:
        return
    
    try:
        generate_mapping_s2p(ref_sc, primitive, verbose=False)
    except:
        unit_new = inverse_transformation(ref_sc, sc_mat)
        prim_new = get_primitive_structure_spglib(unit_new)
        unitcell = unit_new.copy()
        primitive = prim_new.copy()
        
        try:
            generate_mapping_s2p(ref_sc, unitcell, verbose=False)
        except:
            msg = "\n Error(1): The unitcell does not match the supercell."
            msg += "\n Please report the bug to the developer."
            logger.error(msg)
            sys.exit()
            
        try:
            generate_mapping_s2p(ref_sc, primitive, verbose=False)
        except:
            msg = "\n Error(2): The primitive cell does not match the supercell."
            msg += "\n Please report the bug to the developer."
            logger.error(msg)
            sys.exit()
    
    # Because the generate_mapping_s2p rounds the transformation matrix, 
    # it can pass even if the lattice constants are different.
    # If the lattice constants of ref_sc and primitive are not consistent,
    # re-derive from ref_sc.
    _cell_ratio = np.linalg.inv(primitive.cell.array) @ ref_sc.cell.array
    if not np.allclose(_cell_ratio, np.rint(_cell_ratio), atol=1e-4):
        unit_new = inverse_transformation(ref_sc, sc_mat)
        prim_new = get_primitive_structure_spglib(unit_new)
        unitcell = unit_new.copy()
        primitive = prim_new.copy()
    
    try:
        generate_mapping_s2p(ref_sc, unitcell, verbose=False)
    except:
        #
        # TODO: find the compatible unitcell
        #
        msg = "\n Error(3): The unitcell cell does not match the supercell."
        msg += "\n Please report the bug to the developer."
        logger.error(msg)
        sys.exit()
    
    return {
        'primitive': primitive,
        'unitcell': unitcell,
        'supercell': ref_sc
    }
