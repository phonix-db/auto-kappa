#
# log.py
#
# Copyright (c) 2023 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
import os
import numpy as np
import datetime
import yaml

from auto_kappa.utils.system import get_cpu_info, get_os_info

import logging
logger = logging.getLogger(__name__)

def set_logging(filename='log.txt', level=logging.DEBUG,
                format=" %(levelname)8s : %(message)s"):
    
    ### file handler
    fh = logging.FileHandler(filename=filename, mode='w')
    fh.setFormatter(logging.Formatter(format))

    ### stream handler
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(format))

    logging.basicConfig(level=level, handlers=[fh, sh])

def start_autokappa():
    """ Print the logo.
    Font: stick letters
    """
    from auto_kappa.version import __version__
    msg = "\n"
    msg += "              ___  __                __   __       \n"
    msg += "     /\  |  |  |  /  \ __ |__/  /\  |__) |__)  /\  \n"
    msg += "    /~~\ \__/  |  \__/    |  \ /~~\ |    |    /~~\ \n"
    msg += "    ver. %s\n" % __version__
    msg += "\n"
    time = datetime.datetime.now()
    msg += " Start at " + time.strftime("%m/%d/%Y %H:%M:%S") + "\n"
    logger.info(msg)

def end_autokappa():

    time = datetime.datetime.now()
    msg = "\n"
    msg += "     ___       __  \n"
    msg += "    |__  |\ | |  \ \n"
    msg += "    |___ | \| |__/ \n"
    msg += "    \n"
    msg += " at " + time.strftime("%m/%d/%Y %H:%M:%S") + " \n"
    logger.info(msg)

def print_system_info():
    """ Print system information """
    # ### Host name
    # try:
    #     print_hostname()
    # except Exception:
    #     pass
    
    ### CPU info
    try:
        print_cpu_info()
    except Exception:
        pass
    
    ### OS info
    try:
        print_os_info()
    except Exception:
        pass

def print_hostname():
    """ Print host name
    """
    ### host name
    msg = "\n"
    try:
        import socket
        msg += " Hostname: %s" % socket.gethostname()
    except Exception:
        try:
            msg += " Hostname: %s" % os.uname()[1]
        except Exception:
            pass
    logger.info(msg)

def print_cpu_info():
    """ Print CPU information
    """
    cpu_info = get_cpu_info()
    msg = "\n CPU information:"
    msg += "\n ----------------"
    msg += "\n OS              : %s" % cpu_info["os"]
    msg += "\n CPU model       : %s" % cpu_info["cpu_model"]
    msg += "\n Physical cores  : %d" % cpu_info["physical_cores"]
    msg += "\n Logical cores   : %d" % cpu_info["logical_cores"]
    # msg += "\n Current freq.   : %.2f MHz" % cpu_info["freq_mhz_current"]
    msg += "\n Max freq.       : %.2f GHz" % (cpu_info["freq_mhz_max"] * 1e-3)
    logger.info(msg)

def print_os_info():
    """ Print OS information
    """
    os_info = get_os_info()
    msg = "\n OS information:"
    msg += "\n ---------------"
    for key in os_info:
        msg += "\n %15s : %s" % (key, str(os_info[key]))
    logger.info(msg)

def start_larger_supercell(almcalc):
    ### print
    line = "Analyze with a larger supercell (%dx%dx%d)" % (
            almcalc.scell_matrix[0][0],
            almcalc.scell_matrix[1][1],
            almcalc.scell_matrix[2][2])
    msg = "\n"
    msg += " ###" + "#" * (len(line)) + "###\n"
    msg += " ## " + " " * (len(line)) + " ##\n"
    msg += " ## " + line              + " ##\n"
    msg += " ## " + " " * (len(line)) + " ##\n"
    msg += " ###" + "#" * (len(line)) + "###"
    logger.info(msg)

    msg = "\n"
    msg += " Number of atoms : %d\n" % len(almcalc.supercell)
    msg += " Cell size : %.2f, %.2f, %.2f\n" % (
            np.linalg.norm(almcalc.supercell.cell[0]),
            np.linalg.norm(almcalc.supercell.cell[1]),
            np.linalg.norm(almcalc.supercell.cell[2]),
            )
    logger.info(msg)

def print_options(params):
    msg = ""
    msg += "\n Input parameters:"
    msg += "\n ================="
    for key in params.keys():
        if params[key] is not None:
            msg += "\n " + key.ljust(25) + " : " + str(params[key])
    logger.info(msg)

def print_conditions(cell_types=None, trans_matrices=None, kpts_all=None):

    msg = ""
    if cell_types is not None:
        msg += "\n Cell type"
        msg += "\n ---------"
        for cal_type in cell_types:
            msg += "\n %6s : %10s" % (cal_type, cell_types[cal_type])
    
    ###
    if trans_matrices is not None:
        msg += "\n\n Transformation matrix"
        msg += "\n ---------------------"
        for cell_type in trans_matrices:
            msg += "\n %10s : " % cell_type
            for i in range(3):
                vec = trans_matrices[cell_type][i]
                if cell_type == "primitive":
                    msg += "%.3f " * 3 % tuple(vec)
                else:
                    msg += "%d " * 3 % tuple(vec)
    
    ###
    if kpts_all is not None:
        msg += "\n\n k-mesh"
        msg += "\n ------"
        for cal_type in kpts_all:
            msg += f"\n {cal_type:6s} :" + " %d" * 3 % tuple(kpts_all[cal_type])
    logger.info(msg)

def print_space_group(structure):
    from auto_kappa.structure.crystal import get_symmetry_dataset
    dataset = get_symmetry_dataset(structure)
    try:
        spg_number = dataset['number']
        spg_name = dataset['international']
    except Exception:
        spg_number = dataset.number
        spg_name = dataset.international
    
    msg  = f"\n Space group : {spg_name} ({spg_number})"
    logger.info(msg)

def print_times(times, labels):

    msg = "\n\n"
    msg += " Calculation times:\n"
    msg += " ==================\n"
    logger.info(msg)
    
    nchar = 0
    ttot = np.sum(np.asarray(times))
    for i, lab in enumerate(labels):
        tt = float(times[i])         ### sec
        percentage = 100.*tt/ttot
        line = "%25s (sec): %13.2f (%.1f%%)" % (lab, tt, percentage)
        nchar = max(nchar, len(line))
        msg = " " + line
        logger.info(msg)
        
    msg = " " + "-" * (nchar + 5) + "\n"
    msg += " %25s (sec): %13.2f \n" % ("total", ttot)
    logger.info(msg)

def negative_frequency(fmin):
    
    logger = logging.getLogger(__name__)
    msg = "\n Negative eigenvalues were found."
    msg += "\n Minimum frequency : %.2f" % (fmin)
    logger.warning(msg)

def symmetry_error(spg_init, spg_fin):

    logger = logging.getLogger(__name__)
    msg = "\n Warning: The spacegroup number was changed from %d to %d "\
            "due to a relaxation calculation.\n" % (spg_init, spg_fin)
    logger.warning(msg)

def rerun_with_omp():
    
    logger = logging.getLogger(__name__)
    msg = "\n Change the parallel method from MPI to OpenMP and rerun the "\
            "calculation."
    logger.info(msg)

def write_parameters(
        outfile, unitcell, cell_types, trans_matrices, kpts_used, nac):
    """ Output parameters.yaml
    """
    params = {}

    ### unitcell
    params["lattice"] = unitcell.cell.array.tolist()
    params["symbols"] = unitcell.get_chemical_symbols()
    params["positions"] = unitcell.get_scaled_positions().tolist()

    ### cell types
    params["cell_types"] = cell_types

    ### transformation matrix
    params["trans_matrices"] = {}
    for key in trans_matrices:
        params["trans_matrices"][key] = np.asarray(trans_matrices[key]).tolist()

    ### k-mesh
    params["kpts"] = {}
    for key in kpts_used:
        params["kpts"][key] = np.asarray(kpts_used[key]).tolist()

    ### prepare the directory
    with open(outfile, "w") as f:
        yaml.dump(params, f)

