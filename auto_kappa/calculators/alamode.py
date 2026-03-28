#
# alamode.py
#
# This script helps to preform phonon analyses using ALAMODE and VASP.
#
# Copyright (c) 2022 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
import sys
import os
import os.path
from typing import Sequence
import numpy as np
import glob
import shutil

from auto_kappa.alamode.plotter import plot_kappa_vs_grain_size
from auto_kappa.apdb import ApdbVasp
from auto_kappa.alamode.almcalc import AlamodeCalc
from auto_kappa.alamode.helpers import should_rerun_alamode
from auto_kappa.alamode.log_parser import AkLog, read_log_fc
from auto_kappa.structure.crystal import get_automatic_kmesh, get_supercell
from auto_kappa.cui.suggest import klength2mesh
from auto_kappa.cui import ak_log
from auto_kappa.calculators.compat import remove_old_kappa_data
from auto_kappa.calculators.scph import (
    calculate_high_order_force_constants, conduct_scph_calculation)
from auto_kappa.alamode.plotter import plot_kappa_vs_grain_size
from auto_kappa.almlog import ALMLOG

import logging
logger = logging.getLogger(__name__)

def analyze_phonon_properties(
        almcalc, calc_force=None, negative_freq=-1e-3, 
        base_dir=None, ignore_log=False, 
        harmonic_only=False, calc_kappa=True,
        nmax_suggest=None, frac_nrandom=1.0,
        params_nac={'apdb': None, 'kpts': None},
        kdensities_for_kappa=None,
        ## SCPH
        scph=0, disp_temp=500., frac_nrandom_higher=0.5,
        scph_temperatures={'scph': 100*np.arange(1, 11), 'kappa': 300},
        ## 4-phonon
        four=0, frac_kdensity_4ph=0.2,
        ):
    """ Analyze phonon properties
    
    Parameters
    -----------
    almcalc : auto_kappa.alamode.almcalc.AlamodeCalc
    
    calc_force : ase.calculators.vasp.vasp.Vasp

    negative_freq : float   
        threshold for negative frequency
    
    harmonic_only : int, 0
        0 : stop the calculation after harmonic properties are calculated.
    
    calc_kappa : bool
        Calculate thermal conductivity (True) or not (False)

    nmax_suggest : int
        Maximum number of suggested partterns for finite displacement method. If
        the number exceeds this value, LASSO is used for cubic FCs.

    frac_nrandom : int
        Coefficient to determine the number of random patterns for LASSO.

    params_nac : dict
        parameters for NAC calculation.
        ``params_nac[apdb]``: auto_kappa.apdb.ApdbVasp
    
    scph : int
        Self-consistent phonon approach is used (1) or not (0).
    
    disp_temp : float, 500
        temperature for random displacement
    
    frac_nrandom_higher : float, 0.34
        fractional number of random displacement patterns for high-order FCs
    
    scph_temperatures : float, [100, 200, ..., 1000]
        temperatures for SCPH
    
    Returns
    --------
    integer :
        ``0`` when the calculation was finished properly,
        ``-1`` when negative frequencies were found,
    
    """
    ###
    ### Calculate harmonic FCs and harmonic phonon properties
    ###
    analyze_harmonic_properties(almcalc, calc_force, negative_freq=negative_freq, params_nac=params_nac)
    
    ### Check negative frequency
    if almcalc.minimum_frequency < negative_freq and almcalc.calculate_forces:
        
        ### If negative frequencies were found,
        log = AkLog(base_dir)
        log.write_yaml()
        ak_log.negative_frequency(almcalc.minimum_frequency)
        
        ## If the negative frequency is found, old kappa data are removed.
        remove_old_kappa_data(almcalc.out_dirs)
        
        ### If SCPH is not used and negative frequencies were found, return -1.
        if scph == 0:
            return -1
    
    if harmonic_only:
        msg = "\n Harmonic properties have been calculated.\n"
        logger.info(msg)
        return 0
    
    ## Calculate cubic FCs
    calculate_cubic_force_constants(
            almcalc, calc_force,
            nmax_suggest=nmax_suggest, 
            frac_nrandom=frac_nrandom, 
            ignore_log=ignore_log
            )
    
    ## Calculate Grüneisen parameters
    almcalc.calculate_gruneisen_parameters()
    
    ## Calculate high-order FCs using LASSO
    if scph > 0 or four > 0:
        
        ### calculate forces for SCPH
        calculate_high_order_force_constants(
                almcalc, calc_force,
                frac_nrandom=frac_nrandom_higher,
                disp_temp=disp_temp)
        
        ### Perform SCPH calculation
        if scph > 0:
            
            conduct_scph_calculation(almcalc, temperatures=scph_temperatures['scph'])
            
            fmin_scph = almcalc.get_fmin_scph(temperature=scph_temperatures['kappa'])
            try:
                if fmin_scph < negative_freq:
                    msg = f"\n Imaginary frequency was found at {scph_temperatures['kappa']}K: {fmin_scph:.3f} cm^-1"
                    logger.warning(msg)
                    calc_kappa = False
            except Exception as e:
                msg = f"\n Warning: cannot get SCPH minimum frequency: {e}"
                logger.warning(msg)
                calc_kappa = False
    
    ## Calculate kappa with different k-mesh densities
    if calc_kappa:
        if scph == 0 and four == 0:
            calc_type = "cubic"
        elif scph == 0 and four > 0:
            calc_type = "4ph"
        elif scph > 0 and four == 0:
            calc_type = "scph"
        elif scph > 0 and four > 0:
            calc_type = "scph_4ph"
        else:
            msg = f"\n Unknown combination of scph and four: {scph}, {four}"
            logger.error(msg)
            sys.exit()
        
        ## k-mesh densities for thermal conductivity
        if kdensities_for_kappa is None:
            if four > 0 or scph > 0:
                kdensities_for_kappa = [1500]
            else:
                kdensities_for_kappa = [500, 1000, 1500]
        
        ## temperatures for spectral analysis
        temp_kappa_scph = scph_temperatures['kappa']
        if scph == 0:
            temperatures_for_spectral = "300:500"
        else:
            temperatures_for_spectral = str(temp_kappa_scph)
        
        params_kappa = {}
        
        ## FCs XML files
        if scph > 0 or four > 0:
            
            ### Anharmonic FCs XML file (obtained using LASSO)
            fcsxml_abs = (
                almcalc.out_dirs["higher"]["lasso"] + 
                "/%s.xml" % (almcalc.prefix))
            params_kappa['fcsxml'] = os.path.relpath(
                fcsxml_abs, almcalc.out_dirs["higher"][f"kappa_{calc_type}"])
            
            ### Harmonic FCs XML file (obtained using SCPH)
            if scph > 0:
                fc2xml_abs = (
                    almcalc.out_dirs["higher"]["scph"] + 
                    "/%s_%dK.xml" % (almcalc.prefix, temp_kappa_scph))
                params_kappa['fc2xml'] = os.path.relpath(
                    fc2xml_abs, almcalc.out_dirs["higher"][f"kappa_{calc_type}"])
        
        ## Temperature
        if scph > 0:
            params_kappa['tmin'] = 300
            params_kappa['tmax'] = 301
            params_kappa['dt'] = 100
        
        if four > 0:
            frac_kdensity_4ph = frac_kdensity_4ph if frac_kdensity_4ph is not None else 0.2
        
        calculate_thermal_conductivities(
            almcalc,
            kdensities=kdensities_for_kappa,
            ignore_log=ignore_log,
            temperatures_for_spectral=temperatures_for_spectral,
            calc_type=calc_type,
            frac_kdensity_4ph=frac_kdensity_4ph,
            **params_kappa)
    
    ### output log.yaml
    if almcalc.calculate_forces:
        log = AkLog(base_dir)
        log.write_yaml()
    
    return 0

def _use_the_initial_sc(sc_matrix: Sequence[Sequence[int]]) -> None:
    """Log a framed message about using the initial supercell size.
    
    Parameters
    ----------
    sc_matrix : array_like of int, shape (3, 3)
        Supercell transformation matrix.
        
    """
    dims = tuple(int(sc_matrix[i][i]) for i in range(3))
    main_msg = "Analysis using the supercell of initial size"
    line_size = f"Supercell size : {dims[0]} x {dims[1]} x {dims[2]}"
    
    inner = max(len(main_msg), len(line_size)) + 2  # includes a space at both sides
    n_tot = inner + 4  # '##' + '##' の分

    top_bottom = " " + "#" * n_tot
    blank_line  = " ##" + " " * (n_tot - 4) + "##"

    def pad_line(text: str) -> str:
        extra = inner - (len(text) + 2)   # include both sides
        return f" ## {text}" + " " * extra + " ##"
    
    msg = "\n".join([
        top_bottom,
        blank_line,
        pad_line(main_msg),
        pad_line(line_size),
        blank_line,
        top_bottom,
    ])
    logger.info("\n" + msg)

def analyze_phonon_properties_with_larger_supercells(
    almcalc, calc_force,
    negative_freq=-1e-3, 
    ignore_log=False,
    harmonic_only=False, 
    nmax_suggest=100, frac_nrandom=1.0, 
    #
    vasp_config=None,
    max_natoms=300, 
    delta_max_natoms=50, 
    max_loop_for_largesc=2,
    k_length=20, 
    restart=1, 
    ## Higher-order FCs
    frac_nrandom_higher=0.5,
    disp_temp=500.,
    ## SCPH
    scph=0,
    scph_temperatures={'scph': 100*np.arange(1, 11), 'kappa': 300},
    ## 4-phonon
    four=0, frac_kdensity_4ph=0.13,
    pes=0
    ):
    """ Analyze phonon properties with larger supercells """
    almcalc_large = analyze_harmonic_with_larger_supercells(
            almcalc, vasp_config,
            max_natoms_init=max_natoms,
            delta_max_natoms=delta_max_natoms,
            max_loop=max_loop_for_largesc,
            k_length=k_length,
            negative_freq=negative_freq,
            restart=restart,
            )
    
    if not almcalc.calculate_forces:
        return
    
    ### plot band and DOS
    from auto_kappa.plot.bandos import plot_bandos_for_different_sizes
    figname = almcalc.out_dirs["result"] + "/fig_bandos.png"
    # figname = almcalc_large.out_dirs['harm']['bandos'] + '/fig_bandos.png'
    plot_bandos_for_different_sizes(almcalc_large, almcalc, figname=figname)
    
    ### If negative frequencies could be removed, calculate cubic FCs with
    ### the supercell of initial size
    if (almcalc_large.minimum_frequency > negative_freq and harmonic_only == 0):
        
        ### calculate cubic force constants
        # If scph is not 0, cubic FCs were already calculated.
        if scph == 0:
            
            _use_the_initial_sc(almcalc.scell_matrix)
            
            calculate_cubic_force_constants(
                    almcalc, calc_force,
                    nmax_suggest=nmax_suggest, 
                    frac_nrandom=frac_nrandom, 
                    ignore_log=ignore_log,
                    )
            
            almcalc_large._fc3_type = almcalc.fc3_type
            
            ## Calculate Grüneisen parameters
            almcalc.calculate_gruneisen_parameters()
            
            ### Calculate higher-order force constants
            if four > 0:
                calculate_high_order_force_constants(
                        almcalc, calc_force,
                        frac_nrandom=frac_nrandom_higher,
                        disp_temp=disp_temp,
                        )
            
            fc2xml = almcalc_large.fc2xml
            
        else:
            almcalc_large._fc3_type = almcalc.fc3_type
            
            ## Use the initial supercell for the kmesh in the SCPH calculation.
            from auto_kappa.calculators.kmesh_int import get_transformation_matrix_prim2scell
            mat_p2s_fcs = get_transformation_matrix_prim2scell(
                almcalc.primitive_matrix, almcalc.scell_matrix)
            
            conduct_scph_calculation(
                almcalc_large,
                temperatures=scph_temperatures['scph'],
                fcsxml=almcalc.higher_fcsxml,
                fc2xml=almcalc_large.fc2xml,
                mat_p2s_fcs=mat_p2s_fcs,
            )
            
            _tmp = scph_temperatures['kappa']
            fc2xml = almcalc_large.out_dirs['higher']['scph'] + "/%s_%dK.xml" % (
                almcalc_large.prefix, _tmp)
        
        ### calculate kappa
        if scph == 0 and four == 0:
            kdensities = [500, 1000, 1500]
            calc_type = 'cubic'
            xml_files = {'fc2xml': fc2xml, 'fcsxml': almcalc.fc3xml}
        else:
            kdensities = [1500]
            xml_files = {'fc2xml': fc2xml, 'fcsxml': almcalc.higher_fcsxml}
            if scph > 0 and four == 0:
                calc_type = 'scph'
            elif four > 0 and scph == 0:
                calc_type = '4ph'
            elif four > 0 and scph > 0:
                calc_type = 'scph_4ph'
        
        calculate_thermal_conductivities(
                almcalc_large,
                kdensities=kdensities,
                calc_type=calc_type,
                ignore_log=ignore_log,
                temperatures_for_spectral="300:500",
                frac_kdensity_4ph=frac_kdensity_4ph,
                **xml_files
                )
        
    else:
        ak_log.negative_frequency(almcalc_large.minimum_frequency)
        ### calculate PES
        if pes > 0:
            almcalc_large.calculate_pes(negative_freq=negative_freq)
            sys.exit()
    
    return almcalc_large
    
def analyze_harmonic_with_larger_supercells(
        almcalc_orig, vasp_config,
        base_dir=None,
        #
        max_natoms_init=None,
        delta_max_natoms=50,
        max_loop=1,
        #
        k_length=20,
        negative_freq=-1e-3,
        # ignore_log=False,
        restart=1,
        ):
    """ Analyze harmonic properties with larger supercell(s) """
    from auto_kappa.structure.supercell import estimate_supercell_matrix
    from auto_kappa.cui.ak_log import start_larger_supercell
    
    if base_dir is not None:
        msg = " \n Warning: 'base_dir' option is deprecated and will be removed in future versions."
        logger.warning(msg)
    
    count = 0
    isc = 0
    sc_mat_prev = almcalc_orig.scell_matrix
    almcalc_new = None
    while isc <= max_loop:
        
        count += 1
        max_natoms = max_natoms_init + delta_max_natoms * count
        
        ### estimate a supercell from the max number of atoms
        sc_mat = estimate_supercell_matrix(
                almcalc_orig.unitcell, max_num_atoms=max_natoms
                )
        
        ### check if the supercell is changed
        diff = np.amax(np.asarray(sc_mat) - np.asarray(sc_mat_prev))
        
        if diff < 0.1:
            continue
        
        ### label
        sc_label = "%dx%dx%d" % (sc_mat[0,0], sc_mat[1,1], sc_mat[2,2])
        added_dir = "sc-" + sc_label
        
        ### Set AlamodeCalc obj
        almcalc_new = AlamodeCalc(
                almcalc_orig.primitive,
                base_directory=almcalc_orig.base_directory,
                additional_directory=added_dir,
                restart=restart,
                primitive_matrix=almcalc_orig.primitive_matrix,
                scell_matrix=sc_mat,
                cutoff2=almcalc_orig.cutoff2,
                cutoff3=almcalc_orig.cutoff3,
                magnitude=almcalc_orig.magnitude,
                magnitude2=almcalc_orig.magnitude2,
                nac=almcalc_orig.nac,
                commands=almcalc_orig.commands,
                verbose=almcalc_orig.verbose,
                yamlfile_for_outdir=almcalc_orig.yamlfile_for_outdir,
                dim=almcalc_orig.dim,
                calculate_forces=almcalc_orig.calculate_forces,
                )
        
        ### Check structures >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        from auto_kappa.structure.comparison import match_structures
        match = match_structures(almcalc_orig.primitive, almcalc_new.primitive,
                                 ignore_order=False)
        if not match:
            almcalc_new._unitcell = almcalc_orig.primitive
            almcalc_new._primitive = almcalc_orig.primitive
            almcalc_new._supercell = get_supercell(almcalc_orig.unitcell,
                                                   almcalc_new.scell_matrix,
                                                   format='ase')
        
        from auto_kappa.structure.comparison import generate_mapping_s2p
        generate_mapping_s2p(almcalc_new.supercell, almcalc_new.primitive)
        ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        start_larger_supercell(almcalc_new)
        
        ### Set kmesh for VASP calculation
        kpts = klength2mesh(k_length, almcalc_new.supercell.cell.array)
        
        ### ASE calculator for forces
        apdb = ApdbVasp(
                almcalc_orig.unitcell,
                primitive_matrix=almcalc_orig.primitive_matrix,
                scell_matrix=sc_mat,
                command=almcalc_orig.commands["vasp"],
                mater_dim=almcalc_orig.dim,
                vasp_config=vasp_config,
                )
        calc_force = apdb.get_calculator('force', kpts=kpts)
        
        ### analyze phonon properties with a supercell
        analyze_harmonic_properties(
            almcalc_new, calc_force, negative_freq=negative_freq)
        
        ### If negative frequencies were eliminated, escape the loop
        if almcalc_new.minimum_frequency > negative_freq:
            msg = "\n Negative frequencies were eliminated!"
            logger.info(msg)
            return almcalc_new
        
        isc += 1
        if isc == max_loop:
            break
        
        ### prepare for the next step
        sc_mat_prev = almcalc_new.scell_matrix
        
    return almcalc_new

def analyze_harmonic_properties(
        almcalc, calculator, 
        ignore_log=False, max_num_corrections=5,
        deltak=0.01, reciprocal_density=1500,
        negative_freq=-1e-3,
        params_nac={'apdb': None, 'kpts': None},
        ):
    """ Analyze harmonic FCs and phonon properties.
    
    Args
    =====
    almcalc : AlamodeCalc obj

    calculator : ASE calculator for VASP
    
    deltak : float
        resolution of phonon dispersion

    reciprocal_density : float
        resolution of DOS
    
    """
    ### suggest and creat structures for the harmonic FCs
    almcalc.write_alamode_input(propt='suggest', order=1)
    almcalc.run_alamode(propt='suggest', order=1, ignore_log=True)
    
    # logfile = almcalc.out_dirs['harm']['suggest'] + '/suggest.log'
    # almlog = ALMLOG(logfile)
    
    ### calculate forces for the harmonic FCs
    almcalc.calc_forces(order=1, calculator=calculator)
    
    _nprocs_orig = almcalc.commands['alamode']['nprocs']
    _para_orig = almcalc.commands['alamode']['anphon_para']
    
    nac_orig = almcalc.nac
    for propt in ["fc2", "band", "dos"]:    
        ### analyze harmonic properties 
        almcalc.analyze_harmonic_property(
                propt, 
                max_num_corrections=max_num_corrections,
                deltak=deltak, 
                reciprocal_density=reciprocal_density,
                )
    
    ### optimize NAC option
    fmin = almcalc.get_minimum_frequency(which="both")
    
    ### calculate Born effective charge
    if fmin < negative_freq and almcalc.nac == 0:
        try:
            ## set NONANALYTICAL option for Alamode
            almcalc.nac = 2
            mode = 'nac'
            params_nac["apdb"].run_vasp(
                    mode,
                    almcalc.out_dirs[mode],
                    params_nac["kpts"],
                    print_params=True
                    )
        except Exception as e:
            msg = f"\n Warning: cannot get parameters for NAC: {e}"
            logger.warning(msg)
    
    ### get optimal NONANALYTICAL option for Alamode
    if fmin < negative_freq and almcalc.nac != 0:
        
        nac_new = almcalc.get_optimal_nac(
                tol_neg_frac=0.1,
                max_num_corrections=max_num_corrections,
                deltak=deltak,
                reciprocal_density=reciprocal_density,
                negative_freq=negative_freq,
                )
        
        ### If a different NAC can remove negative frequencies,
        ### harmonic properties are calculated again.
        if nac_new is not None:
            
            ### message
            nac_orig = almcalc.nac
            msg = "\n Modify NONANALYTIC option from %d to %d, " % (
                    nac_orig, nac_new)
            msg += "which can remove negative frequencies."
            logger.info(msg)
            
            ### update NAC
            almcalc.nac = nac_new
            
            ### copy the calculated results with the optimal NAC option
            try:
                dir_bandos = almcalc.out_dirs["harm"]["bandos"]
                dir_nac = dir_bandos + "/nac_%d" % almcalc.nac
                msg = "\n >>> Move files in %s to %s." % (
                        almcalc.get_relative_path(dir_nac),
                        almcalc.get_relative_path(dir_bandos))
                logger.info(msg)
                if os.path.exists(dir_nac):
                    fns = glob.glob(dir_nac + "/*")
                    for ff in fns:
                        shutil.copy(ff, dir_bandos)
                ig_log = 0
            except Exception:
                ig_log = 1
            
            ### overwrite harmonic properties
            for propt in ["band", "dos"]:
                almcalc.analyze_harmonic_property(
                    propt,
                    max_num_corrections=max_num_corrections,
                    deltak=deltak, 
                    reciprocal_density=reciprocal_density,
                    ignore_log=ig_log)
    
    ###
    almcalc.commands['alamode']['nprocs'] = _nprocs_orig
    almcalc.commands['alamode']['anphon_para'] = _para_orig
    
    ### plot band and DOS
    if almcalc.calculate_forces:
        try:
            almcalc.plot_bandos()
        except Exception as e:
            logger.warning(f"\n Failed to plot band and DOS: {e}")
        try:
            almcalc.plot_thermodynamic_properties()
        except Exception as e:
            logger.warning(f"\n Failed to plot thermodynamic properties: {e}")
    
    ### eigenvalues at commensurate points
    almcalc.write_alamode_input(propt='evec_commensurate')
    almcalc.run_alamode(propt='evec_commensurate', ignore_log=ignore_log)

def calculate_cubic_force_constants(
        almcalc, calculator,
        nmax_suggest=None, frac_nrandom=None, ignore_log=False,
        ):
    """ Calculate cubic force constants
    """
    ## Adjust the cutoff length for cubic FCs if the cutoff is too short.
    ## `almcalc.min_nearest`-th shortest atomic distance
    flag = almcalc.adjust_cutoff3(index=almcalc.min_nearest)
    file_log = almcalc.out_dirs['cube']['suggest'] + '/suggest.log'
    if flag and os.path.exists(file_log):
        ignore_log = True
    else:
        ignore_log = False
    
    ### calculate forces for cubic FCs
    almcalc.write_alamode_input(propt='suggest', order=2)
    almcalc.run_alamode(propt='suggest', order=2, ignore_log=ignore_log)
    almcalc.calc_forces(
            order=2, calculator=calculator,
            nmax_suggest=nmax_suggest,
            frac_nrandom=frac_nrandom,
            output_dfset=2,
            )
    
    ### calculate anharmonic force constants
    if almcalc.fc3_type == 'lasso':
        for propt in ['cv', 'lasso']:
            almcalc.write_alamode_input(propt=propt, order=2)
            almcalc.run_alamode(propt, order=2, ignore_log=ignore_log)
        almcalc.print_fc_error('lasso')
    else:
        ## ver.1 : with ALM library
        ##almcalc.calc_anharm_force_constants()
        ## ver.2: with alm command
        almcalc.write_alamode_input(propt='fc3')
        almcalc.run_alamode(propt='fc3', ignore_log=ignore_log)
        almcalc.print_fc_error('fc3')

def calculate_thermal_conductivities(
        almcalc, 
        kdensities=[500, 1000, 1500], 
        ignore_log=False,
        temperatures_for_spectral="300:500",
        calc_type="cubic",
        frac_kdensity_4ph=None, ## fractional k-densities for 4-phonon scattering
        **kwargs,
        ):
    """ Calculate thermal conductivities and plot spectral info
    
    Args
    ======
    almcalc : AlamodeCalc obj

    kdensities : array of float

    ignore_log : bool

    temperatures_for_spectral : string of floats seperated by ":"
    
    calc_type : string
        "cubic", "scph", "4phonon", or "scph+4phonon"

    kwargs : dict   
        Parameters for ALAMODE
    
    """
    for kdensity in kdensities:
        
        kpts = get_automatic_kmesh(
                almcalc.primitive, reciprocal_density=kdensity, dim=almcalc.dim)
        kpts_suffix = "_%dx%dx%d" % (int(kpts[0]), int(kpts[1]), int(kpts[2]))
        
        if calc_type == "cubic":
            outdir = (almcalc.out_dirs['cube']['kappa_%s' % almcalc.fc3_type] + kpts_suffix)
        elif calc_type in ["scph", "4ph", "scph_4ph"]:
            outdir = almcalc.out_dirs['higher'][f"kappa_{calc_type}"] + kpts_suffix
        else:
            msg = f"\n Unknown calc_type: %s" % calc_type
            logger.error(msg)
            sys.exit()
        
        ##
        propt = 'kappa'
        if 'scp' in calc_type:
            propt += '_scph'
        
        if '4ph' in calc_type:
            propt += '_4ph'  ## anphon >= ver.1.9    
            
            if frac_kdensity_4ph is None:
                raise ValueError(" Error: 'frac_kdensity_4ph' option must be given for 4-phonon calculation.")
            
            kpts_4ph = get_automatic_kmesh(
                almcalc.primitive, 
                reciprocal_density=int(kdensity * frac_kdensity_4ph + 0.5),
                dim=almcalc.dim)
            kwargs['kmesh_coarse'] = kpts_4ph
        
        almcalc.write_alamode_input(
                propt=propt, order=None, kpts=kpts, outdir=outdir, **kwargs)
        
        if almcalc.calculate_forces == False:
            continue
        
        almcalc.run_alamode(
                propt=propt, ignore_log=ignore_log, outdir=outdir,
                logfile=f"{propt}.log")
        
        ### check output file for kappa
        kappa_log = f"{outdir}/{propt}.log"
        flag = should_rerun_alamode(kappa_log)
        
        if flag and almcalc.commands['alamode']['anphon_para'] == "mpi":
            ##
            ## if thermal conductivity could not be calculated with MPI, run the 
            ## calculation again with OpenMP.
            ##
            ak_log.rerun_with_omp()
            almcalc.commands['alamode']['anphon_para'] = "omp"
            almcalc.run_alamode(
                    propt=propt, ignore_log=ignore_log, 
                    outdir=outdir, **kwargs)
        
        ### Print calculation time
        almlog = ALMLOG(kappa_log)
        if almlog.duration is not None:
            duration = almlog.duration / 60.  # in minutes
            logger.info(f"\n Computation time: {duration:.2f} mins")
        
        # === Plot anharmonic properties === #
        # ================================== #
        ## Kind of scattering process
        if '4ph' in calc_type:
            process = '4ph'
        else:
            process = '3ph'
        
        ## Print thermal conductivity at room temperature
        try:
            _print_rt_kappa(outdir, almcalc.prefix)
            logger.info("")
        except Exception as e:
            msg = f"\n Warning: cannot print thermal conductivity: {e}"
            logger.warning(msg)
        
        ## Plot T-dependent thermal conductivity
        try:
            if 'scp' not in calc_type:
                almcalc.plot_kappa(outdir)
        except Exception as e:
            msg = f"\n Warning: cannot plot thermal conductivity: {e}"
            logger.warning(msg)
        
        ## Make csv files containing lifetime etc.
        try:
            for T in temperatures_for_spectral.split(':'):
                almcalc.write_lifetime_at_given_temperature(outdir, temperature=float(T), process='3ph')
                if process == '4ph':
                    almcalc.write_lifetime_at_given_temperature(outdir, temperature=float(T), process=process)
        except Exception as e:
            msg = f"\n Warning: lifetime was not written properly. {e}"
            logger.warning(msg)
        
        ## Plot lifetime
        try:
            almcalc.plot_lifetime(outdir,
                                  temperatures=temperatures_for_spectral, 
                                  process=process)
        except Exception as e:
            msg = f"\n Warning: the figure of lifetime was not created properly. {e}"
            logger.warning(msg)
        
        ## Plot scattering rates
        try:
            almcalc.plot_scattering_rates(outdir, temperature=300., grain_size=1000., process='3ph')
            if process == '4ph':
                almcalc.plot_scattering_rates(outdir, temperature=300., grain_size=1000., process='4ph')
        except Exception as e:
            msg = f"\n Warning: the figure of scattering rate was not created properly. {e}"
            logger.warning(msg)
        
        ## Plot spectral and cumulative kappa wrt frequency and MFP
        for wrt in ['frequency', 'mfp']:
            if wrt == 'frequency':
                xscale = 'linear'
            else:
                xscale = 'log'
            try:
                almcalc.plot_cumulative_kappa(outdir, process='3ph',
                                              temperatures=temperatures_for_spectral, 
                                              wrt=wrt, xscale=xscale)
                if process == '4ph':
                    almcalc.plot_cumulative_kappa(outdir, process='4ph',
                                                  temperatures=temperatures_for_spectral, 
                                                  wrt=wrt, xscale=xscale)
            except Exception as e:
                msg = f"\n Warning: the figure of cumulative thermal conductivity wrt {wrt} "
                msg += f"was not created properly. {e}"
                logger.warning(msg)

        ## Write and plot thermal conductivity as a function of grain size
        try:
            nprocs = almcalc.commands['alamode']['nprocs']
            filename = outdir + f"/kappa_vs_grain_size.csv"
            figname = outdir + f"/fig_kappa_vs_grain_size.png"
            almcalc.write_kappa_vs_grain_size(outdir, process='3ph', outfile=filename, nprocs=nprocs)
            plot_kappa_vs_grain_size(filename, figname)
            if process == '4ph':
                filename = outdir + f"/kappa_vs_grain_size_4ph.csv"
                figname = outdir + f"/fig_kappa_vs_grain_size_4ph.png"
                almcalc.write_kappa_vs_grain_size(outdir, process=process, outfile=filename, nprocs=nprocs)
                plot_kappa_vs_grain_size(filename, figname)
        
        except Exception as e:
            msg = f"\n Warning: the figure of thermal conductivity vs grain size was not created properly. {e}"
            logger.warning(msg)
        
    if almcalc.calculate_forces == False:
        return None
    else:
        msg  = "\n - .... . .-. -- .- .-..   -.-. --- -. -.. ..- -.-. - - .. ...- .. - .. . ...   "
        msg += "\n .-- . .-. .   -.-. .- .-.. -.-. ..- .-.. .- - . -."
        logger.info(msg)
    
    if almcalc.calculate_forces and calc_type == "cubic":
        logger.info("\n Plot thermal conductivity for different k-mesh densities:")
        almcalc.plot_all_kappa(figname=None, calc_type=calc_type)


def _print_rt_kappa(outdir, prefix, rt=300):
    
    from auto_kappa.io.kl import Kboth, KL
    
    files = {}
    suffixes = ['kl', 'kl3', 'kl4', 'kl_coherent']
    for ik in range(4):
        suffix = suffixes[ik]
        file = f"{outdir}/{prefix}.{suffix}"
        if not os.path.exists(file):
            continue
        if file.startswith("/"):
            file = "./" + os.path.relpath(file)
        files[suffix] = file

    def _print_each(kappa_temp):
        msg = " "
        for ii, dtype in enumerate(['kp', 'kc', 'klat']):
            if ii  != 0:
                msg += "\n "
            for direct in ['xx', 'yy', 'zz', 'ave']:
                key = f"{dtype}_{direct}"
                if key in kappa_temp:
                    msg += f"{key:8s}: {kappa_temp[key]:8.3f}   "
                    # if key != 'ave':
                    #     msg += "  |  "
        logger.info(msg)
    
    for key in ['kl', 'kl3', 'kl4']:
        if key not in files:
            continue
        if 'kl_coherent' not in files:
            kobj = KL(files[key])
        else:
            kobj = Kboth(files[key], files['kl_coherent'])
        
        ks = kobj.get_kappa(temperature=rt)
        if ks is None:
            logger.error(f"\n Error: could not read data at {rt}K from {files[key]}")
            continue
        
        process = '4ph' if key == 'kl4' else '3ph'
        logger.info(f"\n Thermal conductivity at {rt}K ({process}):")
        _print_each(ks)
    