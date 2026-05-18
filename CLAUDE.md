# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**auto-kappa** is a Python automation framework for first-principles calculation of anharmonic phonon properties (lattice thermal conductivity, phonon lifetimes, 3- and 4-phonon scattering) using **VASP** and **ALAMODE**. The entry point is the `akrun` CLI script.

## Installation

```bash
# Build and install (from repo root)
sh install.sh

# Or manually:
python setup.py sdist
pip install dist/auto-kappa-1.1.1.tar.gz
```

The `install.sh` script runs `setup.py sdist` then installs from the tarball. After install, `akrun` should be available on `$PATH`.

## Running a calculation

```bash
# Basic usage
akrun --file_structure POSCAR.Si --outdir Si

# With common options
akrun --file_structure POSCAR.Si --outdir Si --nprocs 16 --mpirun "mpirun -np" \
      --command_vasp vasp_std --command_alm alm --command_anphon anphon

# Show all options
akrun -h
```

Required environment variable: `VASP_PP_PATH` must point to the directory containing `potpaw_PBE/{element}` pseudopotential files (used by ASE).

Optional config file override (YAML or JSON):
```bash
akrun --file_structure POSCAR.Si --outdir Si --config my_config.yaml
```

## Branch Strategy

- `develop` — active development branch; all PRs target here
- `main` — public release branch; merged from `develop`
- `gh-pages` — auto-deployed docs (do not edit manually)

Standard flow: develop on `develop` → merge to `main` → GitHub Actions deploys docs.

## Architecture

### Entry point flow

`scripts/akrun` → `auto_kappa.cui.ak_scripts.main()` which:
1. Parses CLI args via `auto_kappa.cui.ak_parser.get_parser()`
2. Loads VASP params (defaults from `ak_default_config.yaml`, optionally overridden by `--config`)
3. Constructs an `ApdbVasp` object (handles VASP relaxation + NAC)
4. Calls `analyze_phonon_properties()` which drives ALAMODE calculations

### Key classes and modules

| Module | Role |
|--------|------|
| `auto_kappa/apdb.py` — `ApdbVasp` | Manages crystal structure handling, VASP relaxation, Born charge (NAC) calculations |
| `auto_kappa/alamode/almcalc.py` — `AlamodeCalc` | Main interface to ALAMODE; inherits from `AlamodeForceCalculator`, `AlamodeInputWriter`, `AlamodePlotter`, `NameHandler`, `GruneisenCalculator` |
| `auto_kappa/calculators/alamode.py` — `analyze_phonon_properties()` | Orchestrates the full phonon pipeline: harmonic FCs → cubic/higher FCs → SCPH → kappa |
| `auto_kappa/calculators/vasp.py` — `run_vasp_with_custodian()` | Runs VASP via Custodian with error handlers; handles signal cleanup |
| `auto_kappa/calculators/scph.py` | SCPH (self-consistent phonon) renormalization workflow |
| `auto_kappa/vasp/` | k-mesh optimization, VASP parameter helpers, relaxation logic |
| `auto_kappa/io/` | Input/output: ALAMODE XML/input files, BORN info, band/DOS/kappa results |
| `auto_kappa/structure/` | Structure manipulation (spglib/pymatgen), supercell generation, 2D material handling |
| `auto_kappa/alamode/` | ALAMODE job runner, displacement generation, log parsing, LASSO fitting, Grüneisen |

### Output directory layout (under `--outdir`)

```
outdir/
  relax/              # VASP relaxation
  nac/                # Born effective charges
  harm/suggest/       # ALAMODE displacement patterns for FC2
  harm/force/         # VASP force calculations
  harm/bandos/        # Phonon band + DOS
  cube/suggest/       # Displacement patterns for FC3
  cube/force_fd/      # FC3 via finite displacement
  cube/force_lasso/   # FC3 via LASSO (random displacements)
  higher/             # Higher-order FCs, SCPH, 4-phonon kappa
  result/             # Final kappa and parameters
```

### VASP parameters

Defaults are in `auto_kappa/ak_default_config.yaml`. Users can override with `--config path/to/config.yaml`. The config supports sections: `vasp_parameters` (per mode: `relax`, `force`, `nac`, `shared`, etc.) and `potcar_setups`.

### FC method selection

When the number of displacement patterns exceeds `--nmax_suggest` (default 1), the code automatically switches from finite-displacement to LASSO regression for cubic FC calculation. This threshold controls the tradeoff between accuracy and computational cost.

## Dependencies

External codes required (not Python packages):
- **VASP** 6.3.0+ (`vasp`, `vasp_gam` commands)
- **ALAMODE** 1.5 recommended (`alm`, `anphon` commands)
- **anphon** 1.9.9 (optional, for 4-phonon: `anphon.2.0` command)

Python packages installed automatically via `setup.py`: numpy, phonopy, spglib, seekpath, ase, pymatgen, custodian, xmltodict, mkl, f90nml, PyYAML, psutil, scikit-learn, lxml.
