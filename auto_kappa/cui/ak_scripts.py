#
# ak_script.py
#
# This script is akrun command user interface.
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
import datetime
import json
import numpy as np

from auto_kappa.apdb import ApdbVasp
from auto_kappa import output_directories
from auto_kappa.alamode.almcalc import AlamodeCalc
from auto_kappa.io.files import write_output_yaml
from auto_kappa.calculators.alamode import analyze_phonon_properties
from auto_kappa.cui import ak_log
from auto_kappa.utils.config import load_user_config, get_potcar_setups, get_xc
from auto_kappa.cui.initialization import (
        use_omp_for_anphon,
        get_previous_nac,
        get_required_parameters,
        get_base_directory_name,
        )
from auto_kappa.utils.config import (
    get_vasp_parameters, 
    get_vasp_parameters_by_mode)
from auto_kappa.vasp.kmesh import optimize_klength, klength2mesh

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('custodian').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')

def _set_outdirs(base_dir):
    """ Prepare and return names of output directories """
    out_dirs = {}
    for k1 in output_directories.keys():
        values1 = output_directories[k1]
        if type(values1) == str:
            out_dirs[k1] = base_dir + '/' + values1
        else:
            out_dirs[k1] = {}
            for k2 in values1.keys():
                values2 = values1[k2]
                out_dirs[k1][k2] = base_dir + '/' + values2
    return out_dirs

def _stop_symmetry_error(out):
    """ Stop the calculation because of the symmetry change """
    if out == -1:
        msg = "\n Error: crystal symmetry was changed during the "\
                "relaxation calculation."
    elif out == -2:
        msg = "\n Error: too many errors for the relaxation calculation."
    else:
        msg = ""
    
    msg += "\n"
    msg += "\n STOP THE CALCULATION"
    time = datetime.datetime.now()
    msg += "\n at " + time.strftime("%m/%d/%Y %H:%M:%S")
    msg += "\n"
    logger.error(msg)
    sys.exit()

def _store_parameters(outfile, params):
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=4, ensure_ascii=False)

def main():
    
    ### Parse given parameters
    from auto_kappa.cui.ak_parser import get_parser
    parser = get_parser()
    options = parser.parse_args()
    
    ### Prepare the base directory
    base_dir = get_base_directory_name(options.outdir, restart=options.restart)
    os.makedirs(base_dir, exist_ok=True)
    
    ### Set logger
    logfile = os.path.join(base_dir, 'ak.log')
    ak_log.set_logging(filename=logfile, level=logging.DEBUG, format="%(message)s")
    
    ### Start auto-kappa
    ak_log.start_autokappa()
    ak_log.print_system_info()
    
    ### Auto-kappa parameters
    ak_params = vars(options)
    
    ## Check auto-kappa parameters
    from auto_kappa.cui.compat import check_ak_options
    param_file = ak_params['outdir'] + "/parameters.json"
    check_ak_options(ak_params, param_file)
    
    ### Load user configuration (if exists)
    config_file = ak_params.get('config_file', None)
    user_config = load_user_config(config_file)
    
    ### Get VASP parameters (merge defaults with user config)
    ### e.g. {'relax': {...}, 'nac': {...}, ..., 'shared': {...}}
    vasp_params = get_vasp_parameters(user_config)
    
    ## Note for the 2D system
    if ak_params['mater_dim'] == 2:
        from auto_kappa.structure.two import print_2d_system_notation
        print_2d_system_notation()
    
    ### Set output directories
    out_dirs = _set_outdirs(base_dir)
    
    ### memory check
    if use_omp_for_anphon(base_dir):
        if ak_params["anphon_para"] != "omp":
            msg = "\n Change anphon_para option to \"omp\".\n"
            logger.info(msg)
            ak_params["anphon_para"] = "omp"
    
    ### relaxed_cell
    if ak_params['relaxed_cell'] is not None:
        if (ak_params['relaxed_cell'].lower()[0] == "u" or 
                ak_params['relaxed_cell'].lower()[0] == "c"):
            ak_params['relaxed_cell'] = "unitcell"
        elif ak_params['relaxed_cell'].lower()[0] == "p":
            ak_params['relaxed_cell'] = "primitive"
    
    ### Get required parameters for the calculation!
    cell_types, structures, trans_matrices, kpts_used, nac = (
        get_required_parameters(
            base_directory=base_dir,
            dir_phdb=ak_params['directory'], 
            file_structure=ak_params['file_structure'],
            max_natoms=ak_params['max_natoms'], 
            #max_natoms3=ak_params['max_natoms3'],
            k_length=ak_params['k_length'],
            celltype_relax_given=ak_params['relaxed_cell'],
            dim=ak_params['mater_dim'],
        ))
    
    ### If the previous structure was not the same as the given structure,
    if structures.get('supercell', None) is not None:
        num_new = len(structures['supercell'])
        if ak_params['max_natoms'] < num_new:
            ak_params['max_natoms'] = num_new + 1
            msg = "\n Modify \"max_natoms\" parameter to %d." % (ak_params['max_natoms'])
            logger.error(msg)
    
    ### NONANALYTIC (primitive)
    if nac != 0:
        if ak_params['nonanalytic'] is not None:
            nac = ak_params['nonanalytic']
        
        ### check previously-used NONANALYTIC parameter
        try:
            dir0 = base_dir.replace(os.getcwd(), ".")
        except Exception:
            dir0 = base_dir
        
        prev_nac = get_previous_nac(dir0)
        if prev_nac is not None and nac != prev_nac:
            msg = "\n NONANALYTIC was modified to %s" % prev_nac
            logger.info(msg)
            nac = ak_params["nonanalytic"] = prev_nac
    
    ### command to run VASP jobs
    command_vasp = {
            'mpirun': ak_params['mpirun'], 
            'nprocs': ak_params['nprocs'], 
            'nthreads': 1, 
            'vasp': ak_params['command_vasp'],
            'vasp_gam': ak_params['command_vasp_gam'],
            }
    
    ### Pseudopotential setups (load from user config or use defaults)
    potcar_setups = get_potcar_setups(user_config)
    xc = get_xc(user_config)
    
    ### Optimize k-length for VASP calculations
    if ak_params['optimize_klength'] == 1:
        
        # Optimize k-length using the unitcell
        structure = structures[
            ak_params['relaxed_cell'] 
            if ak_params['relaxed_cell'] is not None 
            else 'unitcell']
        
        params_force = get_vasp_parameters_by_mode(vasp_params, mode='force')
        
        klength_opt = optimize_klength(
            structure,
            vasp_params=params_force,
            command=command_vasp,
            klengths=[10, 5, 100],                    # min, delta, max
            min_klength=20,
            tolerance=ak_params['energy_tolerance'],  # in eV/atom
            potcar_setups=potcar_setups,
            xc=xc,
            base_dir=base_dir
            )
        
        ak_params['k_length'] = klength_opt
        
        # Update k-points for all cell types
        for calc_mode in kpts_used.keys():
            cell_type = cell_types[calc_mode]
            structure = structures[cell_type]
            kpts = klength2mesh(ak_params['k_length'], structure.cell.array)
            kpts_used[calc_mode] = kpts
    
    ### Store Auto-kappa parameters
    _store_parameters(param_file, ak_params)
    
    ### print parameters
    ak_log.print_options(ak_params)
    ak_log.print_conditions(cell_types=cell_types, 
                            trans_matrices=trans_matrices,
                            kpts_all=kpts_used)
    ak_log.print_space_group(structures['primitive'])
    
    ### write file
    os.makedirs(out_dirs["result"], exist_ok=True)
    filename = out_dirs["result"] + "/parameters.yaml"
    ak_log.write_parameters(filename, structures["unitcell"], cell_types, trans_matrices, kpts_used, nac)
    
    try:
        fn_print = filename.replace(os.getcwd(), ".")
    except Exception:
        fn_print = filename
    msg = "\n Output %s" % fn_print
    logger.info(msg)
    
    ### output yaml file
    yaml_outdir = base_dir + "/output_directories.yaml"
    info = {"directory": out_dirs["result"].replace(base_dir, "."),
            "kind": "others",
            "note": "results"}
    write_output_yaml(yaml_outdir, "result", info, overwrite=False)
    
    ### Set ApdbVasp object
    vasp_config={
        'params': vasp_params,
        'setups': potcar_setups,
        'xc': xc,
        }
    apdb = ApdbVasp(
            structures["unitcell"],
            primitive_matrix=trans_matrices["primitive"],
            scell_matrix=trans_matrices["supercell"],
            command=command_vasp,
            mater_dim=ak_params['mater_dim'],
            vasp_config=vasp_config,
            base_directory=base_dir,
            )
    
    ### Relaxation calculation
    out = apdb.run_relaxation(
            out_dirs["relax"],
            kpts_used["relax"],
            volume_relaxation=ak_params['volume_relaxation'],
            cell_type=cell_types["relax"],
            # max_error=ak_params["max_relax_error"],
            nsw_params=ak_params["nsw_params"],
            )
    
    ### Get relaxed structures
    structures_relax = apdb.structures.copy()
    apdb.output_structures()
    
    ### Stop the calculation because of the symmetry error
    if out < 0:
        _stop_symmetry_error(out)
    
    ## Modify the cutoff for harmonic force constants
    if ak_params['mater_dim'] == 2:
        from auto_kappa.structure.two import suggest_fc2_cutoff, print_length_info
        cutoff_harm = suggest_fc2_cutoff(structures_relax['super'])
        print_length_info(structures_relax['super'])
    else:
        cutoff_harm = -1
    
    ## output yaml file
    info = {
        "directory": out_dirs["relax"].replace(base_dir, "."), 
        "kind": "others",
        "note": "structure optimization",
        }
    write_output_yaml(yaml_outdir, "relax", info)
    
    ### Calculate Born effective charge
    if nac:
        mode = 'nac'
        vaccum_thickness = 20. if ak_params['mater_dim'] == 2 else None
        apdb.run_vasp(mode, out_dirs[mode], kpts_used["nac"], 
                      print_params=True, vaccum_thickness=vaccum_thickness)
        
        ### output yaml file
        info = {
                "directory": out_dirs[mode].replace(base_dir, "."), 
                "kind": "VASP",
                "note": "Born effective charge",
                }
        write_output_yaml(yaml_outdir, mode, info)
    
    ### command for ALAMODE
    command_alamode = {
        'mpirun': ak_params['mpirun'], 
        'anphon_para': ak_params['anphon_para'], 
        'nprocs': ak_params['nprocs'],
        'anphon': ak_params['command_anphon'],
        'alm': ak_params['command_alm'],
        'dfc2': ak_params['command_dfc2'],
        'anphon_ver2': ak_params['command_anphon_ver2'],
        }
    
    ### Set AlmCalc
    almcalc = AlamodeCalc(
            structures_relax['prim'],  # primitive structure
            base_directory=base_dir,
            restart=ak_params['restart'],
            primitive_matrix=trans_matrices['primitive'],
            scell_matrix=trans_matrices['supercell'],
            cutoff2=cutoff_harm,
            cutoff3=ak_params['cutoff_cubic'],
            min_nearest=ak_params['min_nearest'],
            magnitude=ak_params['mag_harm'],
            magnitude2=ak_params['mag_cubic'],
            ##mag_high=ak_params['mag_high'],
            nac=nac,
            commands={'alamode': command_alamode, 'vasp': command_vasp},
            verbose=ak_params['verbose'],
            yamlfile_for_outdir=yaml_outdir,
            dim=ak_params['mater_dim'],
            calculate_forces=ak_params['calculate_forces'],
            )
    
    ### Prepare an ase.calculators.vasp.vasp.Vasp obj for force calculation
    calc_force = apdb.get_calculator('force', kpts=kpts_used['harm'])
    
    ### Analyze phonon properties
    ignore_log = ak_params['ignore_log']
    out = analyze_phonon_properties(
            almcalc,
            calc_force=calc_force,
            negative_freq=ak_params['negative_freq'],
            base_dir=base_dir,
            ignore_log=ignore_log,
            harmonic_only=ak_params['harmonic_only'],
            #
            nmax_suggest=ak_params['nmax_suggest'],
            frac_nrandom=ak_params['frac_nrandom'],
            #
            params_nac={'apdb': apdb, 'kpts': kpts_used['nac']},
            #
            scph=ak_params['scph'],
            disp_temp=ak_params['random_disp_temperature'],
            frac_nrandom_higher=ak_params['frac_nrandom_higher'],
            #
            four=ak_params['four'],
            frac_kdensity_4ph=ak_params['frac_kdensity_4ph'],
            )
    
    ### Calculate PES
    if (almcalc.minimum_frequency < ak_params['negative_freq'] and ak_params['pes'] == 2):
        almcalc.calculate_pes(negative_freq=ak_params['negative_freq'])
    
    ########################
    ##  Larger supercell  ##
    ########################
    fmin_scph = 0.
    if ak_params['scph']:
        fmins_scph = almcalc.get_fmin_scph(temperature=None)
        fmin_scph = np.inf
        # temp_fmin = None
        for key in fmins_scph.keys():
            if fmins_scph[key] < fmin_scph:
                fmin_scph = fmins_scph[key]
                # temp_fmin = int(key)
        # print(f"Minimum frequency in SCPH: {fmin_scph} at {temp_fmin} K")
    
    ### Calculate phonon properties with larger supercells
    _has_neg_freq = (almcalc.minimum_frequency < ak_params['negative_freq']
                     or not ak_params['calculate_forces'])
    _scph_also_neg = ak_params['scph'] != 1 or fmin_scph < ak_params['negative_freq']
    
    if ak_params["analyze_with_largersc"] and _has_neg_freq and _scph_also_neg:
        
        from auto_kappa.calculators.alamode import analyze_phonon_properties_with_larger_supercells
        
        analyze_phonon_properties_with_larger_supercells(
                almcalc, calc_force,
                negative_freq=ak_params['negative_freq'],
                ignore_log=ignore_log,
                harmonic_only=ak_params['harmonic_only'],
                nmax_suggest=ak_params['nmax_suggest'],
                frac_nrandom=ak_params['frac_nrandom'],
                #
                vasp_config=vasp_config,
                max_natoms=ak_params['max_natoms'],
                delta_max_natoms=ak_params['delta_max_natoms'],
                max_loop_for_largesc=ak_params['max_loop_for_largesc'],
                k_length=ak_params['k_length'],
                restart=ak_params['restart'],
                ## Higher-order force constants
                frac_nrandom_higher=ak_params['frac_nrandom_higher'],
                disp_temp=ak_params['random_disp_temperature'],
                ## SCPH
                scph=ak_params['scph'],
                ## 4-phonon
                four=ak_params['four'],
                frac_kdensity_4ph=ak_params['frac_kdensity_4ph'],
                #
                pes=ak_params['pes'],
                )
    
    ### plot and print calculation durations
    from auto_kappa.io.times import get_times
    times, labels = get_times(base_dir)
    
    from auto_kappa.plot.pltalm import plot_times_with_pie
    figname = base_dir + "/result/fig_times.png"
    plot_times_with_pie(
        times, labels, figname=almcalc.get_relative_path(figname))
    
    ak_log.print_times(times, labels)
        
    ### END of calculations
    ak_log.end_autokappa()

