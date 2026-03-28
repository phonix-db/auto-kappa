#!/bin/bash
#
# Description: 
# ------------
# Example job script to run auto-kappa for Silicon
# Please modify it according to your environment.
#
#
# Description of auto-kappa options:
# ----------------------------------
# file_structure    : Input structure file (formats such as POSCAR, CIF, etc.)
# outdir            : Output directory for results
# nprocs            : Number of processors to use
# mpirun            : Command to execute MPI jobs
# command_vasp      : Command to execute VASP
# command_alm       : Command to execute ALM
# command_anphon    : Command to execute ANPHON
# volume_relaxation : Whether to perform structural relaxation using an equation of state
# analyze_with_larger_supercell : Analyze harmonic FCs with a larger supercell if imaginary (negative) frequencies are detected
# max_natoms        : Maximum allowed number of atoms in a supercell
# nmax_suggest      : Maximum number of displacement patterns in the FD method
# calculate_forces  : Whether to calculate interatomic forces
#
# Abbreviation:
# -------------
# FC : force constant
# FD : finite displacement
#

### Input for each material
file_structure=./POSCAR.Si
mpid=mp-149   ## material ID for Silicon from Materials Project
nprocs=2

if [ ! -e $file_structure ]; then
    echo "Error: $file_structure does not exist."
    exit
fi

### Commands
c_mpirun=mpirun
c_vasp=vasp
c_alm=alm
c_anphon=anphon

### Parameters for phonon calculation
vol_relax=1
larger_sc=0
max_natoms=150
nmax_suggest=1

### Check command existence
for command in $c_mpirun $c_vasp $c_alm $c_anphon; do
    if ! command -v $command &> /dev/null; then
        echo "Error: $command could not be found"
        exit
    fi
done

### Run auto-kappa
akrun \
    --file_structure $file_structure \
    --outdir $mpid \
    --nprocs $nprocs \
    --mpirun         $c_mpirun \
    --command_vasp   $c_vasp \
    --command_alm    $c_alm \
    --command_anphon $c_anphon \
    --volume_relaxation $vol_relax \
    --max_natoms   $max_natoms \
    --nmax_suggest $nmax_suggest \
    --calculate_forces 1 \
    --config ak_config.yaml

