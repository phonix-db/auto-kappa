# ak_relax

`make_vasp_files.py` generates VASP input files (INCAR, POSCAR, POTCAR, KPOINTS) for structure relaxation using auto-kappa's internal routines. The generated files use the same VASP parameters as the `relax/full-1` step in the standard auto-kappa workflow.

## Note

In the default auto-kappa workflow, relaxation is performed using the conventional (unit) cell. This script, however, performs relaxation using the input structure as-is, without converting it to the conventional cell.
