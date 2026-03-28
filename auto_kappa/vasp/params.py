# -*- coding: utf-8 -*-
#
# params.py
#
# Copyright (c) 2024 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
#
import os.path
import sys
import numpy as np
import glob

import logging
logger = logging.getLogger(__name__)

def get_previous_parameters(directory):
    """ Get VASP parameters written in INCAR file in the given directory """
    filename = directory + "/INCAR"
    if os.path.exists(filename) == False:
        return None
    
    ###
    from pymatgen.io.vasp.inputs import Incar
    incar = Incar.from_file(filename)
    return incar

def reflect_previous_jobs(calc, structure, method=None, amin_params=None):
    """
    Parameters
    -----------
    calc : ase.calculators.vasp.vasp.Vasp object
    structure : ase.atoms.Atoms object
    """
    msges = {}

    ### update AMIN
    amin = get_amin_parameter(calc.directory, structure.cell.array, **amin_params)
    if amin is not None:
        msges['amin'] = "\n AMIN = %.3f" % amin
        calc.set(amin=amin)
    
    ### update SYMPREC
    symprec = get_symprec(calc.directory, scale=0.1)
    if symprec is not None:
        msges['symprec'] = "\n SYMPREC = %.3e" % symprec
        calc.set(symprec=symprec)
    
    ### Add functions below if other parameters need to be modified.
    #
    # Example : get_other_vasp_param(calc.directory, structure)
    #
    
    ### print message
    if len(msges) > 0:
        msg = "\n"
        msg += "\n Newly set VASP parameters :"
        for key in msges:
            msg += msges[key]
        logger.info(msg)
    

def get_symprec(directory, scale=None, default=1e-5):
    """ Determine SYMPREC parameter """

    line_error = "PRICELV: current lattice and primitive lattice are incommensurate"
    outcar = f"{directory}/OUTCAR"
    
    ### check the presence of OUTCAR file
    if os.path.exists(outcar) == False:
        return None
    
    ### check OUTCAR file
    with open(outcar, 'r') as f:
        lines = f.readlines()
    
    found_error = False
    for line in lines:
        if line_error in line:
            found_error = True
            break
    
    ### if the error is not found,
    if found_error == False:
        return None
    
    ### check previously used parameters
    incar = get_previous_parameters(directory)
    if incar is None:
        return None
    
    params = incar.as_dict()
    if "SYMPREC" in params:
        base_symprec = params["SYMPREC"]
    else:
        base_symprec = default
    
    ### Stop the calculation if the parameter is too small.
    if base_symprec < 1e-10:
        msg = "\n Error : SYMPREC is too small (%.3e)." % base_symprec
        msg += "\n Stop the calculation."
        logger.error(msg)
        sys.exit()
    
    return base_symprec * scale

def get_amin_parameter(directory, lattice, **args):
    """ Get and return AMIN """
    ### get AMIN parameters
    from auto_kappa import default_amin_parameters
    amin_params = default_amin_parameters.copy()
    amin_params.update(args)
    
    ### get number of errors
    num_errors = get_number_of_errors(directory)
    if num_errors < amin_params['num_of_errors']:
        return None
    
    ### if error exists,
    # amin = None
    for j in range(3):
        length = np.linalg.norm(lattice[j])
        if length > amin_params['tol_length']:
            return amin_params['value']
    return None

def get_number_of_errors(directory):
    """ Get and return the number of errors in the given directory. """
    num_errors = 0
    ### number of errors
    for suffix in ["tar", "tar.gz"]:
        line = directory + "/error.*." + suffix
        fns = glob.glob(line)
        num_errors += len(fns)

    ####
    #line = directory + "/INCAR"
    #fns = glob.glob(line)
    #num_errors += len(fns)
    return num_errors

