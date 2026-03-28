# 
# compat.py
# 
# This script provides functions to handle compatibility with 
# different versions of auto-kappa.
# 
# Created on August 03, 2025
# Copyright (c) 2025 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
# 
import os
import sys
import numpy as np
# import pandas as pd
import shutil
import json
import pickle
import glob
import re

import ase.io
from ase.geometry import get_distances
from pymatgen.io.vasp.inputs import Incar

from auto_kappa import output_directories
from auto_kappa.structure import match_structures

import logging
logger = logging.getLogger(__name__)

def check_ak_options(params, file_params):
    """ Check the auto-kappa options. """
    ### Check previous parameters
    if os.path.exists(file_params):
        with open(file_params, 'r', encoding='utf-8') as f:
            if file_params.endswith('.json'):
                prev_params = json.load(f)
            elif file_params.endswith('.pkl') or file_params.endswith('.pickle'):
                prev_params = pickle.load(f)
            else:
                logger.error(f"\n Unsupported file format: {file_params}")
                return None
        _check_previous_parameters(params, prev_params, file_prev=file_params)

    _check_deprecated_options(params)
    _check_command_options(params)
    _check_alamode_version(params)
    _adjust_displacement_magnitude(params)
    _check_previous_structures(params['outdir'])
    
    if params['mater_dim'] != 3:
        msg = f"\n Error: Auto-kappa does not support {params['mater_dim']}D systems yet."
        logger.error(msg)
        sys.exit()

def _check_deprecated_options(params):
    """ Check deprecated option names and set new option names."""
    ## --ncores
    if params.get('ncores') is not None:
        params['nprocs'] = params['ncores']    
        logger.warning("\n The option --ncores is not used any more. "
                       "Please use --nprocs instead.")
    ## --material_name
    if params.get('material_name') is not None:
        params['outdir'] = params['material_name']
        logger.warning("\n The option --material_name is not used any more. "
                       "Please use --outdir instead.")

def _check_command_options(params):
    """ Check the command options. """
    count = 0
    for name, cmd in params.items():
        if name.startswith("command_") == False:
            continue
        exists = shutil.which(cmd) is not None
        if not exists:
            if count == 0:
                logger.info("")
            
            #msg = f" Warning : '{cmd}' does not exist in the PATH!"
            #logger.error(msg)
            
            count += 1

def _check_alamode_version(params):
    
    from auto_kappa.utils.version import get_version
    from packaging.version import Version
    
    ## 'alm' command is not supported.
    ##if options.command_alm is not None:
    ##    ver = get_version(options.command_alm)
    if params['command_anphon'] is not None:
        ver = get_version(params['command_anphon'])
        if ver is not None:
            if Version(ver) >= Version("1.8.0"):
                msg = f"\n Warning: Anphon ver.{ver} for command_anphon may not be too old."
                msg += "\n Please check the anphon version for --command_anphon."
                logger.error(msg)
    if params['command_anphon_ver2'] is not None:
        ver = get_version(params['command_anphon_ver2'])
        if ver is not None:
            if Version(ver) < Version("1.9.0"):
                msg = f"\n Warning: Anphon ver.{ver} for command_anphon_ver2 may be too old."
                msg += "\n Please check the anphon version for --command_anphon_ver2."
                logger.error(msg)

# def parse_vasp_params(params_string):
#     """
#     Parameters
#     ------------
#     params_string : string
#         VASP parameters which differ from the default values are given by 
#         ``--vasp_parameters`` option.
#         The parmeters can be given like following.

#     How to set the option
#     ---------------------
#     >>> --vasp_parameters=\"lsorbit=1,ediff=1e-9,ediffg=-1e7\"
    
#     Return
#     ---------
#     dictionary of the parameters which will be modified
#     """
#     if params_string is None:
#         return None
    
#     ### parse each parameter
#     params_dict = {}
#     list_params_mod = re.split(r'[,:]', params_string)
#     for each in list_params_mod:
        
#         data = each.split("=")
        
#         ### check format
#         if len(data) != 2:
#             msg = f"\n Error: vasp_parameters is not given properly ({each})"
#             logger.error(msg)
        
#         name = data[0].replace(" ", "").upper()
        
#         ### parse a given value
#         try:
#             val = Incar.proc_val(name, data[1])
#         except Exception:
#             msg = f"\n Error: vasp_parameters may not given properly."
#             msg += f"\n {name} may not exist for VASP parameter."
#             logger.error(msg)
#             sys.exit()
        
#         params_dict[name] = val
    
#     return params_dict

def _adjust_displacement_magnitude(params, tolerance=1e-5):
    """ Adjust the displacement magnitude for harmonic and cubic force calculations.
    """
    ## Order is important: harmonic, cubic_fd -> cubic_lasso
    ## The disp. magnitude for 'cubic_lasso' is prior to 'cubic_fd' in the order.
    dirs_tobe_checked = [
        params['outdir'] + "/" + output_directories['harm']['force'],
        params['outdir'] + "/" + output_directories['cube']['force_fd'],
        params['outdir'] + "/" + output_directories['cube']['force_lasso']
    ]
    orders = ['harm', 'cube_fd', 'cube_lasso']
    
    for i, dir_force in enumerate(dirs_tobe_checked):
        
        file0 = dir_force + "/prist/POSCAR"
        file1 = dir_force + "/1/POSCAR"
        if not os.path.exists(file0) or not os.path.exists(file1):
            continue
        
        try:
            atoms0 = ase.io.read(file0)
            atoms1 = ase.io.read(file1)
            mag_comp, mag_disp = _get_displacement_magnitude(atoms0, atoms1)
        except Exception as e:
            msg = f"\n Error reading files in {dir_force}: {e}"
            logger.error(msg)
            sys.exit()
        
        ## The displacement magnitude is evaluated from
        ## the displacement along each direction (mag_comp) for FD method,
        ## the maximum displacement magnitude (mag_disp) for Lasso method,
        adjusted = False
        if orders[i] == 'harm':
            if abs(mag_comp - params['mag_harm']) > tolerance:
                mag_orig = params['mag_harm']
                mag_mod = mag_comp
                params['mag_harm'] = mag_mod
                adjusted = True
        elif orders[i] == 'cube_fd':
            if abs(mag_comp - params['mag_cubic']) > tolerance:
                mag_orig = params['mag_cubic']
                mag_mod = mag_comp
                params['mag_cubic'] = mag_comp
                adjusted = True
        elif orders[i] == 'cube_lasso':
            if abs(mag_disp - params['mag_cubic']) > tolerance:
                mag_orig = params['mag_cubic']
                mag_mod = mag_disp
                params['mag_cubic'] = mag_disp
                adjusted = True
        else:
            msg = f"\n Error: Unknown order '{orders[i]}' in adjust_displacement_magnitude."
            logger.error(msg)
        
        if adjusted:
            opt_name = 'mag_harm' if orders[i].startswith('harm') else 'mag_cubic'
            msg = f"\n *** Warning ***"
            msg += f"\n Adjust '{opt_name}' option from {mag_orig:.3f} to {mag_mod:.3f}."
            msg += f"\n If you want to use '{opt_name}={mag_orig:.3f}', "
            msg += f"\n please run the job with another output directory name (--outdir)."
            logger.info(msg)
    
def _get_displacement_magnitude(atoms0, atoms1):
    """ Calculate the displacement magnitude between two ASE atoms objects.
    
    Return
    ------
    disp_comp : float
        The maximum displacement component.
    
    mag : float
        The maximum magnitude of the displacement vector.
    
    """
    if len(atoms0) != len(atoms1):
        msg = "The number of atoms in both structures must be the same."
        logger.error(msg)
        sys.exit()
    
    D, D_len = get_distances(
        atoms0.positions, atoms1.positions, 
        cell=atoms0.cell, pbc=atoms0.pbc)
    
    natoms = len(D)
    disp_comps = np.array([D[i,i,:] for i in range(natoms)])
    disp_comp = np.max(disp_comps.flatten())
    
    mag = np.max(np.diag(D_len))
    return disp_comp, mag

def _check_previous_structures(base_dir):
    """ Adjust the displacement magnitude for harmonic and cubic force calculations.
    """
    ## Order is important: harmonic, cubic_fd -> cubic_lasso
    ## The disp. magnitude for 'cubic_lasso' is prior to 'cubic_fd' in the order.
    pos_volume = "./" + base_dir + "/" + output_directories['relax'] + "/volume/POSCAR.opt"
    pos_freeze = "./" + base_dir + "/" + output_directories['relax'] + "/freeze-1/CONTCAR"
    if os.path.exists(pos_volume):
        file_relax = pos_volume
    elif os.path.exists(pos_freeze):
        file_relax = pos_freeze
    else:
        file_relax = None
    
    structure_files = {
        'relax'     : file_relax,
        'opt_unit'  : "./" + base_dir + "/" + output_directories['relax'] + "/structures/POSCAR.unit",
        'opt_super' : "./" + base_dir + "/" + output_directories['relax'] + "/structures/POSCAR.super",
        'nac'       : "./" + base_dir + "/" + output_directories['nac'] + "/POSCAR",
        'harm'      : "./" + base_dir + "/" + output_directories['harm']['force'] + "/prist/POSCAR",
        'cube_fd'   : "./" + base_dir + "/" + output_directories['cube']['force_fd'] + "/prist/POSCAR",
        'cube_lasso': "./" + base_dir + "/" + output_directories['cube']['force_lasso'] + "/prist/POSCAR"
    }
    
    structures = {}
    for key, file in structure_files.items():
        try:
            structures[key] = ase.io.read(file) if os.path.exists(file) else None
        except Exception as e:
            pass
    nstruct = len(structures)
    first_output = True
    for i1 in range(nstruct):
        for i2 in range(i1 + 1, nstruct):
            key1, key2 = list(structures.keys())[i1], list(structures.keys())[i2]
            struct1, struct2 = structures[key1], structures[key2]
            if struct1 is None or struct2 is None:
                continue
            
            ## Use the strict conditions
            ## StructureMatcher may sometimes be instable.
            ignore_order = True
            if len(struct1) != len(struct2):
                primitive_cell = True
                # ignore_order = True
            else:
                primitive_cell = False
                # ignore_order = False
            
            match = match_structures(struct1, struct2, primitive_cell=primitive_cell, 
                                     ltol=1e-7, stol=1e-7, ignore_order=ignore_order, verbose=False)
            
            if not match:
                if first_output:
                    logger.info("\n The following structures are not matched:")
                    first_output = False
                msg = f" - %s and %s" % (structure_files[key1], structure_files[key2])
                logger.info(msg)

def _check_previous_parameters(given_params, prev_params, file_prev=None, rtol=1e-4):
    """ Check if the current parameters match the previous ones.
    """
    columns_must_be_same = [
        'cutoff_cubic', 'min_nearest', 'nmax_suggest', 'frac_nrandom',
        'mag_harm', 'mag_cubic', 'negative_freq', 'relaxed_cell', 
        'k_length', 'optimize_klength', 'energy_tolerance',
        'max_natoms', 'random_disp_temperature',
        'volume_relaxation', 'nonanalytic',
        'config_file']
    
    base_dir = os.path.dirname(file_prev) if file_prev is not None else None
    
    count = 0
    for key in given_params:
        if key not in columns_must_be_same:
            continue
        if prev_params.get(key) is None:
            continue
        
        if key.startswith("mag_") and base_dir is not None:
            if key == "mag_harm":
                line = f"{base_dir}/harm/force/1/vasprun.xml"
            elif key == 'mag_cubic':
                line = f"{base_dir}/cube/force_*/1/vasprun.xml"
            dirs = glob.glob(line)
            if len(dirs) == 0:
                continue
        
        match = True
        if isinstance(given_params[key], float):
            if not np.isclose(given_params[key], prev_params[key], rtol=rtol):
                match = False
        else:
            if given_params[key] != prev_params[key]:
                match = False
        
        if not match:
            if count == 0:
                if file_prev.startswith("/"):
                    file_prev = "./" + os.path.relpath(file_prev)
                logger.info("\n The following parameters are modified to match the previous ones:")
                logger.info(" If you want to keep the current parameters, please modify")
                logger.info(" the parameters in the file \"%s\"." % file_prev)
                count += 1
            msg = " - %s: %s -> %s" % (key, given_params[key], prev_params[key])
            logger.info(msg)
            
            ### Modify the parameter
            given_params[key] = prev_params[key]
    
    if count != 0:
        logger.info("")
