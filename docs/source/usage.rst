Examples
=========

Examples for required packages
-------------------------------

Example jobs are available in the ``auto-kappa/examples`` directory 
for ``alm`` (``examples/1_alm``), ``anphon`` (``examples/2_anphon``),
ASE (``examples/3_ase``), and Custodian (``examples/4_custodian``).
Please go to each directory and run the jobs.

auto-kappa example
--------------------

The ``examples/5_single`` directory contains an example to compute thermal conductivity of Silicon using auto-kappa.
An example job script is shown below. 
Since all processes are included in this job, it takes about half an hour to complete.
It is recommended to use a job scheduler to submit this job.
Please modify it according to your environment.


.. literalinclude:: ./files/run_ex5.sh
   :language: bash


.. .. code-block:: shell
    
..     #!/bin/sh
..     #PBS -q default
..     #PBS -l nodes=1:ppn=24  ## Only single node calculation is available
..     #PBS -j oe
..     #PBS -N test            ## job name
    
..     export LANG=C
..     export OMP_NUM_THREADS=1  ## Please set OMP_NUM_THREADS=1
..     cd $PBS_O_WORKDIR

..     nprocs=24               ## Number of processes

..     mpid=mp-149
..     dir_db=${directory_of_downloaded_phoonondb}/${mpid}  ## This line must be modified.
    
..     if [ ! -e $dir_db ]; then
..         echo " Cannot find $dir_db"    
..         exit
..     fi
    
..     akrun \
..         --directory $dir_db \
..         --outdir $mpid \
..         --nprocs $nprocs

