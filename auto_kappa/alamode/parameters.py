#
# parameters.py
#
# Copyright (c) 2022 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
# import numpy as np
from auto_kappa.structure.crystal import (
    get_transformation_matrix_prim2scell, 
    get_commensurate_points
)

import logging
logger = logging.getLogger(__name__)

def set_parameters_evec(inp, primitive_matrix, scell_matrix, dim=3):
    """ """
    ## supercell matrix wrt primitive cell
    # ver. old
    # mat_p2s_tmp = np.dot(np.linalg.inv(primitive_matrix), scell_matrix)
    # mat_p2s = np.rint(mat_p2s_tmp).astype(int)
    # diff_max = np.amax(abs(mat_p2s - mat_p2s_tmp))
    # if diff_max > 1e-3:
    #     msg = "\n Warning: please check the cell size of primitive and supercell\n"
    #     msg += str(diff_max)
    #     logger.warning(msg)
    #
    # ver. new
    mat_p2s = get_transformation_matrix_prim2scell(
        primitive_matrix, scell_matrix)
    
    ## commensurate points    
    comm_pts = get_commensurate_points(mat_p2s)
    inp.update({'printevec': 1})
    inp.set_kpoint(kpoints=comm_pts, dim=dim)

def set_parameters_kappa(
    inp, kpts=None, nac=None, isotope=2, 
    kappa_coherent=1, kappa_spec=1,
    tmin=50, tmax=1000, dt=50, **kwargs):
    """ Set parameters for kappa calculation. """
    params = {
            "kpts": kpts,
            "nac": nac,
            "isotope": isotope,
            "kappa_coherent": kappa_coherent,
            "kappa_spec": kappa_spec,
            "tmin": tmin,
            "tmax": tmax,
            "dt": dt,
            }
    params.update(kwargs)
    inp.update(params)
