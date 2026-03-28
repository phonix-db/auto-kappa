#
# This script helps to run akrun command of auto_kappa.
# If this script works property, anharmonic phonon properties of Silicon 
# can be calculated automatically. Then, other materials can be easily 
# calculated if a few variables are modified in this script.
#
# Notes:
# Please see ``examples/5_single/from_file/run.sh`` for more detailed comments.
#

nprocs=2

#
# Parameters for each material
#
mpid=mp-149    ## Si
dir_db=./phonondb-20180417/${mpid}

if [ ! -e $dir_db ]; then
    continue
fi

akrun \
    --directory $dir_db \
    --outdir $mpid \
    --nprocs $nprocs \
    --mpirun mpirun \
    --command_vasp vasp \
    --command_alm alm \
    --command_anphon anphon \
    --volume_relaxation 1
