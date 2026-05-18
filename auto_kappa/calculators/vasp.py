#
# calculator.py
#
# This script helps to run VASP calculation with Custodian. While ASE is also
# available, it is recommended to use Custodian to run VASP jobs.
#
# Copyright (c) 2022 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
import sys
import os.path
import numpy as np
import re
import tarfile
import glob
import signal

import ase.io
from ase.calculators.vasp import Vasp
from ase.calculators.vasp.create_input import GenerateVaspInput

from custodian.custodian import Custodian
from custodian.vasp.handlers import (VaspErrorHandler, 
                                     UnconvergedErrorHandler)
from custodian.vasp.jobs import VaspJob

import logging
logger = logging.getLogger(__name__)

def run_vasp_with_custodian(calc, atoms, max_errors=10):
    """ Run a VASP job with Custodian
    
    Args
    ------
    calc : ASE calculator
    
    atoms : ASE Atoms obj
    
    """
    ### prepare a directory and write files
    os.makedirs(calc.directory, exist_ok=True)
    calc.write_input(atoms)
    
    ### change directory
    cwd = os.getcwd()
    outdir = calc.directory
    os.chdir(outdir)
    
    ### set handlers and run VASP
    handlers = [VaspErrorHandler()]  ##, UnconvergedErrorHandler()]
    vjob = VaspJob(calc.command.split(), auto_npar=True)
    c = Custodian(handlers, [vjob], max_errors=max_errors)
    
    def _on_term(signum, frame):
        logger.info(f"\n Caught signal {signum} — terminating VASP job cleanly...")
        try:
            vjob.terminate(directory=".")
        except Exception as e:
            print(f"\n 📛 Failed to terminate VASP properly: {e}")
        sys.exit(1)
    
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    
    try:
        c.run()
    except Exception as e:
        logger.error(f"\n 📛 Error occurred while running VASP: {e}")
    finally:
        ### return to the initial directory
        os.chdir(cwd)
    
    ### check the final structure
    file_vasprun = f"{outdir}/vasprun.xml"
    if os.path.exists(file_vasprun):
        final_atoms = ase.io.read(file_vasprun, index=-1)
        try:
            isif = calc.parameters['isif']
            if not _same_structures(atoms, final_atoms, isif=isif):
                msg = "\n ❌ Final structure differs from initial structure."
                msg += "\n This error could be avoided by resubmitting the job."
                msg += "\n Please resubmit the job to check if the error persists."
                logger.info(msg)
                sys.exit(1)
        except KeyError:
            pass
    
    return 1

def _same_structures(atoms1, atoms2, isif=None, rtol=0.001, atol=0.01):
    if len(atoms1) != len(atoms2):
        return False
    if list(atoms1.symbols) != list(atoms2.symbols):
        return False
    if isif in [0, 1, 2, 4, 5]:
        if abs(atoms1.volume - atoms2.volume) > 1e-3:
            return False
    if isif < 3:
        cell1 = atoms1.cell.array.copy()
        cell2 = atoms2.cell.array.copy()
        return np.allclose(cell1, cell2, rtol=rtol, atol=atol)
    return True

def run_vasp(calc, atoms, method='custodian', max_errors=10):
    """ Run a VASP job 
    
    Args
    -----
    
    calc : ASE VASP calculator
    
    atoms : ASE Atoms obj
    
    method : string
        "ase" or "custodian"
    
    max_errors : int
        parameter for "custodian"
    
    """
    if calc.directory is None:
        msg = "\n Error: output directory is not set in the calculator."
        logger.warning(msg)
        sys.exit()
    
    if 'ase' in method.lower():
        atoms.calc = calc
        atoms.get_potential_energy()
        return 1
    elif 'custodian' in method.lower():
        value = run_vasp_with_custodian(calc, atoms, max_errors=max_errors)
        return value
    else:
        msg = "\n Error: method %s is not supported." % (method)
        logger.error(msg)
        sys.exit()

def get_vasp_calculator(
    vasp_params,
    atoms=None, directory=None, kpts=None,
    encut_scale_factor=1.3, 
    setups={'base': 'recommended', 'W': '_sv'}, 
    xc='pbesol', 
    **args
    ):
    """ Get VASP parameters for the given mode. Parameters are similar to those
    used for phonondb.
    
    Args
    -------
    vasp_params : dict
        VASP parameters given as a dictionary
        e.g., {'ediffg': -1e-6, 'ibrion': 2, ...}
    
    atoms : ASE Atoms object
    
    directory : string
        output directory
    
    kpts : list of float, shape=(3)
    
    args : dict
        input parameters for VASP
    
    How to Use
    -----------
    >>> from auto_kappa.calculator import get_vasp_calculator
    >>> atoms = ase.io.read('POSCAR.primitive', format='vasp')
    >>> mode = 'relax'
    >>> vasp_params = {'ediffg': -1e-6, 'ibrion': 2, 'nsw': 50}
    >>> calc = get_vasp_calculator(directory='./out', kpts=[10,10,10], vasp_params=vasp_params)
    >>> calc.command = "mpirun -n 2 vasp"
    >>> calc.write_input(structure)
    
    """
    calc = Vasp(setups=setups, xc=xc)
    
    ### initialization
    calc.initialize(atoms)
    
    params = {}
    
    ### encut
    enmax = get_enmax(calc.ppp_list)
    params['encut'] = enmax * encut_scale_factor
    
    ### kpoints
    if kpts is not None:
        calc.set(kpts=[int(k) for k in kpts])
        params['gamma'] = True
    
    ### update
    params.update(vasp_params)
    params.update(args)
    
    ### make every keys lowercase letters
    tmp = {key.lower(): value for key, value in params.items()}
    params = tmp.copy()
    
    ### output directory
    if directory is None:
        # outdir = 'out_%s' % mode
        outdir = 'out'
    else:
        outdir = directory
    calc.directory = outdir
    
    ### Generate
    vasp = GenerateVaspInput.set(calc, **params)
    calc.results.clear()
    return calc

def get_enmax(ppp_list):
    """
    Args
    ----------
    ppp_list : list of string
        ppp_list[*] is a POTCAR file

    Return
    ---------
    Maximum value of ENMAX
    
    """
    enmaxes = []
    for filename in ppp_list:
        lines = open(filename, 'r').readlines()
        line = [l for l in lines if 'ENMAX' in l]
        if len(line) == 0:
            logger.error(" Error")
            return 500.
        data = re.split(r'\s+|;|=', line[0])
        data2 = [d for d in data if d != '']
        enmaxes.append(float(data2[1]))
    ##
    enmaxes = np.asarray(enmaxes)
    return np.max(enmaxes)

def backup_vasp(
    directory,
    filenames = {
        "INCAR", "KPOINTS", "POSCAR", "OUTCAR", "CONTCAR", "OSZICAR",
        "vasprun.xml", "vasp.out", "std_err.txt"},
    prefix="error", delete_files=False):
    """
    Backup files to a tar.gz file. Used, for example, in backing up the
    files of an errored run before performing corrections.
    
    This module is originally in custodian.utils.
    
    Args:
        filenames ([str]): List of files to backup. Supports wildcards, e.g.,
            *.*.
        prefix (str): prefix to the files. Defaults to error, which means a
            series of error.1.tar.gz, error.2.tar.gz, ... will be generated.
    """
    ### move to the new directory
    cwd = os.getcwd()
    if os.path.exists(directory):
        os.chdir(directory)
    else:
        msg = "\n Error: cannot find %s" % directory
        logger.error(msg)
        return None
    
    num = max([0] + [int(f.split(".")[1]) for f in glob.glob(f"{prefix}.*.tar.gz")])
    filename = f"{prefix}.{num + 1}.tar.gz"
    
    msg = " Back up the job to %s" % filename
    logging.info(msg)
    with tarfile.open(filename, "w:gz") as tar:
        for fname in filenames:
            for f in glob.glob(fname):
                tar.add(f)
    
    ### delete files
    if delete_files:
        for filename in filenames:
            os.remove(filename)
    
    ### go back to the original directory
    os.chdir(cwd)
    return 0

