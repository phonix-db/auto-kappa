import numpy as np
import ase
import pprint

from auto_kappa.almlog.utils import (
    parse_data_line, get_position_list, extract_afterward_lines, replace_symbols_to_blank,
    replace_symbols_to_blank
    )
from auto_kappa.units import BohrToA

def _get_cell_vectors(lines, title, dtype=float, BohrToAngstrom=BohrToA):
    extracted_lines = extract_afterward_lines(lines, title=title, num_lines=6)
    vec_list = {'a': [], 'b': []}
    for line in extracted_lines:
        line = replace_symbols_to_blank(line)
        parts = line.strip().split()
        if not parts:
            continue
        key = parts[-1][0]
        if key not in vec_list:
            continue
        try:
            vec_list[key].append([dtype(v) for v in parts[:3]])
        except ValueError:
            print(f" Error reading {title}.")
            pass
    for key in vec_list:
        vec_list[key] = np.array(vec_list[key])
        if key == 'a':
            # Convert from Bohr to Angstrom
            vec_list[key] *= BohrToAngstrom
        elif key == 'b':
            # Convert from 1/Bohr to 1/Angstrom
            # b = a
            vec_list[key] /= BohrToAngstrom
            # vec_list[key] /= (2 * np.pi)  # Convert to ASE format (2*pi/a -> 1/a)
    return vec_list

def _get_atomic_positions_and_species(lines, cell_type='primitive', coord_type='fractional'):
    if cell_type == 'primitive':
        title = f"Atomic positions in the {cell_type} cell ({coord_type})"
    elif cell_type == 'supercell':
        title = f"Atomic positions in the {cell_type} ({coord_type})"
    indices, positions, tag_list = get_position_list(lines, title)
    return (indices, positions, tag_list)

def _get_atomic_masses(lines, num_species):
    
    title = "Mass of atomic species (u)"
    extracted_lines = extract_afterward_lines(lines, title=title, num_lines=num_species)
    
    masses = {}
    for line in extracted_lines:
        if not line.strip():
            continue
        line = replace_symbols_to_blank(line)
        parts = line.strip().split()
        try:
            element = parts[0].strip()
            mass = float(parts[1].strip())
            masses[element] = mass
        except (IndexError, ValueError):
            print(f" Error reading atomic masses: {line}")
            pass
    
    return masses

def read_structure(lines):
    """ Read crystal structure from log file lines (Crystal structure section)
    """
    info = {
        "supercell": {"a": None, "b": None, "structure": None},
        "primitive": {"a": None, "b": None, "structure": None},
        "volume_primitive": None, ## in Angstrom^3
        "num_atoms_supercell": None,
        "num_atoms_primitive": None,
        "masses": {}
    }
    
    ## Cell vectors (Angstrom)
    for cell_type in ['supercell', 'primitive']:
        try:
            out = _get_cell_vectors(lines, title=f"* {cell_type.capitalize()}")
            info[cell_type]['a'], info[cell_type]['b'] = out['a'], out['b']
        except ValueError:
            info[cell_type]['a'], info[cell_type]['b'] = None, None
    
    ## Get atomic positions and species
    try:
        out = _get_atomic_positions_and_species(lines, cell_type='primitive', coord_type='fractional')
        info['primitive']['structure'] = ase.Atoms(
            cell=info['primitive']['a'],
            scaled_positions=out[1],
            symbols=out[2],
            pbc=True)
    except TypeError:
        info['primitive']['structure'] = None
    
    ### Check reciprocal lattice vectors
    ## ASE uses 1/a for reciprocal lattice vectors,
    ## while ALAMODE log uses 2*pi/a.
    try:
        prim = info['primitive']['structure']
        if prim is None:
            raise TypeError(" Primitive structure is not available.")
        assert np.allclose(prim.cell.reciprocal() * 2*np.pi, info['primitive']['b'])
    except (AssertionError, TypeError, AttributeError):
        # print(" Warning: Reciprocal lattice vectors do not match.")
        # pprint.pprint(info['primitive']['b'])
        # pprint.pprint(prim.cell.reciprocal() * 2*np.pi)
        pass
    
    ### Get atomic masses (u) used in ALAMODE
    try:
        num_species = len(set(info['primitive']['structure'].get_chemical_symbols()))
        info['masses'] = _get_atomic_masses(lines, num_species)
    except TypeError:
        pass
    
    ### Get other information (Angstrom)
    for il, line in enumerate(lines):
        
        line = line.strip()
        if not line:
            continue
        
        ### Volume
        if line.startswith("Volume of the primitive cell"):
            val = parse_data_line(line, index=-2, dtype=float)
            info['volume_primitive'] = val  ## a.u.
            continue
        elif line.startswith("Number of atoms in the supercell"):
            info['num_atoms_supercell'] = parse_data_line(line, index=-1, dtype=int)
            continue
        elif line.startswith("Number of atoms in the primitive"):
            info['num_atoms_primitive'] = parse_data_line(line, index=-1, dtype=int)
            continue
    
    return info
