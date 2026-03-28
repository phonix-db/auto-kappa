# -*- coding: utf-8 -*-
#
# runjob.py
#
# This script helps to run ALAMODE job
#
# Copyright (c) 2024 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
import os
import sys
import subprocess
import time
import signal

try:
    import psutil
except ImportError:
    psutil = None

from auto_kappa.alamode.io import wasfinished_alamode
from auto_kappa.alamode.errors import check_unexpected_errors

import logging
logger = logging.getLogger(__name__)

def run_alamode(
        filename, logfile, workdir='.', ignore_log=False, file_err="std_err.txt",
        mpirun='mpirun', nprocs=1, nthreads=1, command='anphon',
        max_num_corrections=None):
    """ Run alamode with a command (alm or anphon)

    Args
    ======
    filename : string
        input script of Alamode in workdir

    logfile : string
        log file name in workdir

    workdir : string
        work directory
    
    Return
    =======
    int :
        ``-1`` when the job had been finished.
        ``0`` when the job was conducted.
        ``1`` when the job was not finished.
    
    """
    omp_keys = ["OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK"]
    
    ### change directory
    dir_init = os.getcwd()
    os.chdir(workdir)
    
    ## If the job has been finished, the same calculation is not conducted.
    ## The job status is determined from *.log file.
    status = None
    count = 0
    while True:
        
        if wasfinished_alamode(logfile) and ignore_log == False:
            status = -1
            break
        
        if count == 0:
            ppn_i = nprocs
            nth_i = nthreads
        else:
            ppn_prev = ppn_i
            nth_prev = nth_i
            
            ppn_i = max(1, int(ppn_prev / 2))
            if ppn_i > 1:
                ppn_i += int(ppn_i % 2)
            
            if ppn_prev == ppn_i:
                status = 1
                break
            
            ###
            if count == 1:
                logger.info("")
            msg = " Processes per node : %d => %d" % (ppn_prev, ppn_i)
            logger.info(msg)
        
        ### set number of parallelization
        cmd = "%s -n %d %s %s" %(mpirun, ppn_i, command, filename)
        
        ### set OpenMP
        for key in omp_keys:
            os.environ[key] = str(nth_i)
        
        ### run the job
        _run_job(cmd, logfile=logfile, file_err=file_err)
        
        count += 1
        
        if wasfinished_alamode(logfile):
            status = 0
            break
        
        if max_num_corrections is not None:
            if count >= max_num_corrections:
                status = 1
                break
    
    ### set back OpenMP
    for key in omp_keys:
        os.environ[key] = "1"
    
    ### check logfile
    dir_base = dir_init + "/" + workdir.replace(dir_init, ".").split("/")[1]
    check_unexpected_errors(logfile, dir_base=dir_base)
    
    #### Return to the original directory
    os.chdir(dir_init)
    
    return status

def _run_job(cmd, logfile="log.txt", file_err="std_err.txt"):
    """ Run a job with subprocss
    
    Args
    -----
    cmd : string
        command to run a job
    """    
    ## run the job!!
    status = None
    proc = None
    
    def terminate(signum, frame):
        logger.info(f" Received signum {signum}, terminating the subprocess group...")
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as e:
                logger.error(f" Error while terminating the subprocess group: {e}")
        sys.exit(0)
    
    ## Register signal handlers
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    
    with open(logfile, 'w') as f_out, open(file_err, 'w', buffering=1) as f_err:
        proc = subprocess.Popen(
            cmd, shell=True, env=os.environ,
            stdout=f_out, stderr=f_err,
            preexec_fn=os.setsid  # Start new process group (available on Linux only)
            )
        
        count = 0
        mem_max = 0
        if psutil is None:
            proc.wait()
        else:
            process = psutil.Process(proc.pid)
        
        try:
            while True:
                if proc.poll() is not None:
                    break
                
                try:
                    ### Modified on 2025/09/06 (This may not be working properly.)
                    mem_info = process.memory_info()
                    mem_used_mb = mem_info.rss / (1024.**2)
                    mem_max = max(mem_max, mem_used_mb)
                    
                    total_mem = psutil.virtual_memory().total / (1024.**2)
                    percentage = mem_used_mb / total_mem * 100.
                    
                    if percentage > 95.:
                        logger.info(f"\n ⚠️ Error: high memory usage: {percentage:.2f}%")
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        break
                
                except Exception:
                    break
                
                time.sleep(min(10, count + 1))
                count += 1
            
            status = proc.wait()
            
        finally:
            # Ensure that the process is terminated
            if proc.poll() is None:
                logger.info(" Clearning up: killing leftover subprocess group...")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception as e:
                    logger.error(f" Error while killing the subprocess group: {e}")
    
    return status

