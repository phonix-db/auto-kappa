# 
# kmesh_int.py
# 
# This script generates KMESH_INTERPOLATE from an input file.
# 
# Created on February 03, 2026
# Copyright (c) 2026 Masato Ohnishi
# Created by Masato Ohnishi and Michimasa Morita
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
# 
import numpy as np
import math
from functools import reduce

from auto_kappa.structure.crystal import get_transformation_matrix_prim2scell

def _gcd_nonzero(*nums: int) -> int:
    nz = [abs(int(x)) for x in nums if int(x) != 0]
    return reduce(math.gcd, nz) if nz else 1

def get_kmesh_interpolate(mat_p2s: np.ndarray, *, tol: float = 1e-8) -> list[int]:
    # primitive_cell: np.ndarray,
    # supercell_cell: np.ndarray,
    """ Decide KMESH_INTERPOLATE from primitive_matrix (unit->prim) and
    supercell_matrix (unit->scell).
    
    Strategy:
        1) M_p2s = inv(P) @ S  (primitive->supercell) must be (almost) integer
        2) KMESH_INTERPOLATE[i] = gcd of column i of M_p2s  (0 ignored)
        3) (optional) ensure kmesh_scph is a multiple of kmesh_interpolate
    """
    # M = get_transformation_matrix_prim2scell(
    #     primitive_cell, supercell_cell, tol=tol)   # prim -> scell
    
    # if not np.allclose(M, np.rint(M), atol=tol):
    #     maxdev = float(np.max(np.abs(M - np.rint(M))))
    #     raise ValueError(f"M_p2s is not integer. max deviation={maxdev:g}")
    
    # M = np.rint(M).astype(int)
    
    # Column-wise gcd gives the "largest" commensurate axis mesh in this representation
    kmesh_int = [_gcd_nonzero(*mat_p2s[:, j]) for j in range(3)]
    kmesh_int = [x if x > 0 else 1 for x in kmesh_int]    
    return kmesh_int
