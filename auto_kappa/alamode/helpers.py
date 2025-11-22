#
# utils.py
#
# Interface for ALAMODE
#
# Copyright (c) 2025 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
import os
import sys
import numpy as np
import pandas as pd
import ase.io

from auto_kappa.structure import change_structure_format, match_structures
from auto_kappa.io.alm import AlmInput, AnphonInput
from auto_kappa.io.vasp import wasfinished, get_dfset, read_outcar, print_vasp_params
from auto_kappa.io.files import extract_data_from_file
from auto_kappa.vasp.params import get_previous_parameters, get_amin_parameter
from auto_kappa.alamode.io import wasfinished_alamode, write_displacement_info
from auto_kappa.alamode.errors import found_rank_deficient
from auto_kappa.calculators.vasp import run_vasp, backup_vasp
from auto_kappa.structure.crystal import get_transformation_matrix_prim2scell

import logging
logger = logging.getLogger(__name__)

class AlamodeForceCalculator():
    
    def _start_force_calculation(self, order, calculate_forces):
        
        ### Message to start force calculation
        if calculate_forces:
            line = f"Force calculation (order: {order})"
        else:
            line = f"Generate displacement structures (order: {order})"
        msg = "\n " + line
        msg += "\n " + "=" * (len(line))
        logger.info(msg)
        
    def _get_base_directory_for_forces(self, order, nsuggest, nmax_suggest):
        
        if order == 1:
            outdir = self.out_dirs['harm']['force']
        elif order == 2:
            if nsuggest <= nmax_suggest:
                
                ## Avoid the unexpected error due to the rank deficiency
                use_lasso = False
                logfile = self.out_dirs['cube']['force_fd'] + '/fc3.log'
                if os.path.exists(logfile):
                    if found_rank_deficient(logfile):
                        msg = "\n *** Warning: rank deficient error is found. ***"
                        msg += "\n Previous calculation for the cubic FCs contained rank deficient error."
                        msg += "\n Use lasso regression to avoid the error."
                        logger.warning(msg)
                        use_lasso = True
                
                if use_lasso:
                    self._fc3_type = 'lasso'
                    outdir = self.out_dirs['cube']['force_lasso']
                else:
                    self._fc3_type = 'fd'
                    outdir = self.out_dirs['cube']['force_fd']
                
            else:
                self._fc3_type = 'lasso'
                outdir = self.out_dirs['cube']['force_lasso']
        elif order > 2:
            outdir = self.out_dirs['higher']['force']
        else:
            msg = "\n WARNING: given order (%d) is not supported yet." % order
            logger.error(msg)
            sys.exit()
        
        return outdir
    
    def _get_suggested_structures_for_lasso(
        self, order, num_fcs, num_atoms, frac_nrandom, nmax_suggest,
        temperature=300, classical=False, nmin_generated=10):
        
        ### number of random displacement patters
        nrandom = int(frac_nrandom * num_fcs / num_atoms)
        ngenerated = max(nmin_generated, nrandom)
        
        ### name of order
        order_names = {1: "harmonic", 2: "cubic"}
        if order <= 2:
            name = order_names[order]
        else:
            name = "4th-order"
        
        msg = ""
        msg += f"\n Maximum limit of the number of suggested patterns : {nmax_suggest}"
        msg += f"\n The number of suggested patterns exceeds the maximum limit."
        msg += f"\n"
        msg += f"\n Number of generated random patterns (Ngen) : {ngenerated}"
        msg += f"\n - Ngen = max({nmin_generated}, int(frac * Nfcs / Natoms))"
        msg += f"\n - Nfcs   : Number of free {name} force constants, {num_fcs}"
        msg += f"\n - Natoms : Number of atoms in a supercell, {num_atoms}"
        msg += f"\n - frac   : Fractional number of random patterns, {frac_nrandom:.3f}"
        logger.info(msg)
        
        if order == 2:
            ## FC3 is obtained with random-displacement method
            ## with a fixed displacement magnitude
            structures, displacements = self.get_suggested_structures(
                    order, 
                    disp_mode='random',
                    number_of_displacements=ngenerated,
                    )
        
        elif order > 2:
            ## High order FCs are obtained with
            ## a random-displacment based on normal coordinate
            structures, displacements = self.get_suggested_structures(
                    order, 
                    disp_mode='random_normalcoordinate',
                    number_of_displacements=ngenerated,
                    temperature=temperature,
                    classical=classical
                    )
        return structures, displacements
    
    def _job_for_each_structure(
        self, job_idx, structures, base_dir, order, calculator, **amin_params_set):
        
        # print(structures.keys())
        # for key, struct in structures.items():
        #     ase.io.write("./check/POSCAR." + key, struct)
        
        struct_keys = list(structures.keys())
        key = struct_keys[job_idx]
        
        ### name of output directory
        outdir = base_dir + '/' + str(key)
        
        ### get previous parameters
        if job_idx == 0:
            prev_params = None
        else:
            dir_prev = (base_dir + "/" + str(struct_keys[job_idx-1]))
            prev_params = get_previous_parameters(dir_prev)
        
        ### Check whether structures are same for the finite-displacement method
        if order == 2 and self.fc3_type == 'fd':
            file_poscar = outdir + "/POSCAR"
            if os.path.exists(file_poscar):
                structure_prev = ase.io.read(file_poscar)
                if not match_structures(structures[key], structure_prev, ignore_order=False):
                    msg = (
                        f"\n Error: The structure is not the same as "
                        f"the previous one. ({key})")
                    logger.error(msg)
                    sys.exit()
        
        ### Wheter the calculation has been finished
        filename = outdir + "/vasprun.xml"
        file_outcar = outdir + "/OUTCAR"
        if wasfinished(outdir):
            if are_forces_available(filename):
                
                try:
                    value = extract_data_from_file(file_outcar, "Total CPU time used")
                    run_time = float(value[0]) / 60.  # mins
                except Exception:
                    run_time = -1.0
                
                path = self.get_relative_path(outdir)
                msg = " ( %d / %d ) %s : skip | %.1f mins" % (job_idx + 1, len(struct_keys), path, run_time)
                logger.info(msg)
                self._counter_done += 1
                return
            else:
                backup_vasp(outdir)
        
        ## set output directory
        calculator.directory = outdir
        
        if job_idx == 0:
            print_vasp_params(calculator.asdict()['inputs'])
            logger.info("")
        
        structure = structures[key].copy()
        
        ## Prepare the structure for VASP. This part is for 2D systems.
        if self.dim == 3:
            struct4vasp = structure.copy()
        elif self.dim == 2:
            logger.info("\n 2D structure is not supported.")
            sys.exit()
            # from auto_kappa.structure.two import set_vacuum_to_2d_structure
            # struct_mod = set_vacuum_to_2d_structure(structure, vacuum_thickness=25.0)
            # struct4vasp = change_structure_format(struct_mod, format='ase')
            # if job_idx == 0:
            #     logger.info("\n Modify vacuum space for 2D structure.")
        else:
            msg = " Error: dim must be 2 or 3."
            logger.error(msg)
            sys.exit()
        
        ### set AMIN
        try:
            amin = prev_params["AMIN"]
        except Exception:
            amin = get_amin_parameter(
                calculator.directory, struct4vasp.cell.array, **amin_params_set)
        
        if amin is not None:
            calculator.set(amin=amin)
            
            ### once AMIN is used, AMIN is set for calculations afterward.
            amin_params_set["num_of_errors"] = 0

        ### Write VASP input files only when the directory does not exist
        ### to make sure that the structure file is not overwritten.
        self._prepare_vasp_files(calculator, struct4vasp, 
                                 pristine_structure=structures['prist'])
        
        ### Calculate forces with Custodian
        if self.calculate_forces:
            self._calculate_forces_for_each(
                calculator, struct4vasp, method='custodian', max_num_try=3)
            self._counter_done += 1
            
        ### print log
        time_min = self._print_each_end(
            job_idx + 1, len(struct_keys),
            self.get_relative_path(calculator.directory))
        
        ### estimate remaining time
        if self._counter_calc == 0:
            try:
                num_remain = len(struct_keys) - job_idx - 1
                rtime_est = time_min * num_remain / 60.  # hour
                msg = "\n Estimated remaining time for this part : %.2f hours\n" % (rtime_est)
                logger.info(msg)
            except Exception:
                pass
        
        self._counter_calc += 1
    
    def _prepare_vasp_files(self, calculator, structure, pristine_structure=None):
        """ Prepare VASP input files and write displacement information.
        """
        ## Make VASP input files
        if os.path.exists(calculator.directory) == False:
            os.makedirs(calculator.directory, exist_ok=True)
            calculator.write_input(structure)
            
        ### Write displacement information
        try:
            write_displacement_info(
                structure=structure,
                pristine_structure=pristine_structure,
                outdir=calculator.directory
            )
        except Exception as e:
            logger.error(f"\n Failed to write displacement information: {e}")
    
    def _calculate_forces_for_each(
        self, calculator, structure, method='custodian', max_num_try=3):
        count = 0
        while True:
            run_vasp(calculator, structure, method=method)
            
            ### check forces
            filename = calculator.directory + "/vasprun.xml"
            if are_forces_available(filename):
                break
            else:
                ### backup the previous result
                backup_vasp(calculator.directory)
            
            count += 1
            if count == max_num_try:
                msg = " Error: atomic forces could not be calculated properly."
                logger.error(msg)
                sys.exit()

    def _print_each_end(self, count, num_all, dir_each):
        msg = f" ( {count} / {num_all} ) {dir_each}"
        try:
            outinfo = read_outcar(dir_each + "/OUTCAR")
            msg += " : %7.2f min" % outinfo["cpu_time(min)"]
            time = outinfo["cpu_time(min)"]
        except Exception:
            time = 0.0
        logger.info(msg)
        return time
        
    def _make_dfset_file(self, order, nsuggest, base_dir, fd2d=False):
        
        if order == 1:
            fn0 = self.outfiles['harm_dfset']
        elif order == 2:
            fn0 = self.outfiles['cube_%s_dfset' % self.fc3_type]
        else:
            fn0 = self.outfiles['higher_dfset']
        
        ###
        os.makedirs(self.out_dirs['result'], exist_ok=True)
        offset_xml = base_dir + '/prist/vasprun.xml'
        outfile = self.out_dirs['result'] + "/" + fn0
        
        out = get_dfset(
            base_dir, offset_xml=offset_xml,
            outfile=self.get_relative_path(outfile),
            nset=nsuggest-1, fd2d=fd2d)

        return out

class AlamodeInputWriter():
    
    def get_working_directory(self, propt, order=None):
        """ Get the working directory for the given property. """
        if propt in ['band', 'dos']:
            workdir = self.out_dirs['harm']['bandos']
        elif propt == 'evec_commensurate':
            workdir = self.out_dirs['harm']['evec']
        elif propt == 'kappa':
            workdir = self.out_dirs['cube'][f'kappa_{self.fc3_type}']
        elif propt.startswith('gruneisen'):
            workdir = self.out_dirs['cube']['gruneisen']
        elif propt in ['kappa_4ph', 'kappa_scph', 'kappa_scph_4ph']:
            workdir = self.out_dirs['higher'][propt]
        elif propt in ['lasso', 'cv']:
            if order == 2:
                workdir = self.out_dirs['cube'][propt]
            else:
                workdir = self.out_dirs['higher'][propt]
        elif propt == 'fc2':
            workdir = self.out_dirs['harm']['force']
        elif propt == 'fc3':
            workdir = self.out_dirs['cube']['force_%s' % self.fc3_type]
        elif propt == 'scph':
            workdir = self.out_dirs['higher']['scph']
        elif propt == 'suggest':
            if order == 1:
                workdir = self.out_dirs['harm']['suggest']
            elif order == 2:
                workdir = self.out_dirs['cube']['suggest']
            elif order > 2:
                workdir = self.out_dirs['higher']['suggest']
            else:
                logger.error(f"\n Error: order= {str(order)} is not supported.")
                sys.exit()
        else:
            logger.error(f"\n Error: {propt} is not supported yet.")
            sys.exit()
        return workdir    
    
    def _get_filenames(self, propt, order, **kwargs):
        
        fc2xml = None
        fc3xml = None
        fcsxml = None
        dfset = None
        workdir = self.get_working_directory(propt, order)
        
        if propt in ['band', 'dos', 'evec_commensurate']:
            # fcsxml = os.path.relpath(self.fc2xml, self.workdir)
            fcsxml = self.fc2xml
        elif propt.startswith('kappa') or propt.startswith('gruneisen'):
            if "fcsxml" in kwargs.keys():
                fcsxml = kwargs["fcsxml"]
            else:
                fcsxml = self.fc3xml
        elif propt in ['lasso', 'cv']:
            if order == 2:
                dfset = self.cube_dfset
            else:
                dfset = self.higher_dfset
                fc3xml = self.fc3xml
                if os.path.exists(fc3xml) == False and self.calculate_forces:
                    logger.error("\n Error: FC3 has not been calculated.")
                    sys.exit()
            fc2xml = self.fc2xml
            if os.path.exists(fc2xml) == False and self.calculate_forces:
                logger.error("\n Error: FC2 has not been calculated.")
                # sys.exit()
        elif propt == 'fc2':
            dfset = self.harm_dfset
            fc2xml = None
        elif propt == 'fc3':
            dfset = self.cube_dfset
            fc2xml = self.fc2xml
        elif propt == 'scph':
            fc2xml = self.fc2xml
            fcsxml = self.higher_fcsxml
        elif propt == 'suggest':
            if order is None:
                logger.info("\n Order is not given. Set order = 1.")
                order = 1
            dfset = None
            fc2xml = None
        else:
            logger.error(f"\n Error: {propt} is not supported yet.")
            sys.exit()
        
        ### Get relative paths from the working directory for filenames
        files_abs = {'fc2xml': fc2xml, 'fc3xml': fc3xml, 'fcsxml': fcsxml, 'dfset': dfset}
        files = {}
        for name in files_abs:
            if files_abs[name] is not None:
                if os.path.isabs(files_abs[name]):
                    files[name] = os.path.relpath(files_abs[name], workdir)
                else:
                    files[name] = files_abs[name]
            else:
                files[name] = None
        
        ## Add other paths
        files['dir_work'] = workdir
        files['born_xml'] = self.out_dirs['nac'] + '/vasprun.xml'
        
        return files

    def _get_input_object(
        self, alamode_type=None, mode=None, kpmode=None, fc2xml=None, fc3xml=None, fcsxml=None,
        dfset=None, borninfo=None):
        
        if alamode_type.startswith('anphon'):
            inp = AnphonInput.from_structure(
                    self.primitive,
                    mode=mode,
                    kpmode=kpmode,
                    fcsxml=fcsxml,
                    nonanalytic=self.nac, 
                    borninfo=borninfo,
                )
            ### set primitive cell with Pymatgen-structure
            inp.set_primitive(
                change_structure_format(
                    self.primitive, format="pymatgen-structure"))
        
        elif alamode_type == 'alm':
            inp = AlmInput.from_structure(
                    self.supercell,
                    mode=mode,
                    dfset=dfset,
                    fc2xml=fc2xml,
                    fc3xml=fc3xml,
                    nonanalytic=self.nac, 
                    borninfo=borninfo
                )
        inp.dim = self.dim
        return inp
    
    def _set_emin_emax_delta_e(self, inp, npoints=301, frac_buffer=0.05):
        """ Set emin, emax, and delta_e for DOS calculation if possible. """
        if self.frequency_range is not None:
            diff = self.frequency_range[1] - self.frequency_range[0]
            fmin = self.frequency_range[0] - diff * frac_buffer
            fmax = self.frequency_range[1] + diff * frac_buffer
            inp.update({'emin': fmin, 'emax': fmax})
            inp.update({'delta_e': (fmax-fmin)/(npoints-1)})
    
    def _set_parameters_for_property(
        self, inp, propt=None, deltak=None, kpts=None, order=None):
        
        if propt == 'band' or propt.startswith('gruneisen_band'):
            inp.set_kpoint(deltak=deltak, dim=self.dim, norm_idx=self.norm_idx_xyz)
            if propt == 'band':
                inp.update({'printpr': 1})
        
        elif propt == 'dos' or propt.startswith('gruneisen_dos'):
            self._set_emin_emax_delta_e(inp)
            inp.update({'kpts': kpts})
            inp.update({'pdos': 1})
        
        elif propt == 'evec_commensurate':
            from auto_kappa.alamode.parameters import set_parameters_evec
            set_parameters_evec(
                inp, self.primitive_matrix, self.scell_matrix, dim=self.dim)
            
        elif propt.startswith('kappa'):
            from auto_kappa.alamode.parameters import set_parameters_kappa
            set_parameters_kappa(inp, kpts=kpts, nac=self.nac)
            self._set_emin_emax_delta_e(inp)
            
        elif propt == "scph":
            
            from auto_kappa.calculators.scph import set_parameters_scph
            
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            mat_p2s = get_transformation_matrix_prim2scell(
                self.primitive_matrix, self.scell_matrix)
            
            # from auto_kappa.units import AToBohr
            # print("# supercell")
            # print(np.asarray(self.supercell.cell) * AToBohr)
            # print("# conventional cell")
            # print(np.asarray(self.unitcell.cell) * AToBohr)
            # print("# primitive cell")
            # print(np.asarray(self.primitive.cell) * AToBohr)
            # print()
            # print(self.primitive_matrix)
            # print(self.scell_matrix)
            # print(mat_p2s)
            # sys.exit()
            
            set_parameters_scph(
                inp, 
                primitive=self.primitive, 
                scell=self.supercell,
                mat_p2s=mat_p2s, 
                deltak=deltak, 
                kdensity_limit=10,
                )
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            
        elif propt in ['cv', 'lasso', 'fc2', 'fc3', 'suggest']:
            """
            Comments
            ---------
            * maxalpha is automatically set by the code.
            * minalpha = maxalpha * 1e-6 

            * set nbody and cutoffs
            
            """
            if propt == 'fc2':
                order = 1
            elif propt == 'fc3':
                order = 2
            elif propt == 'cv' or propt == 'lasso':
                if order is None:
                    logger.error("\n Error: order must be given.")
                    sys.exit()
            elif propt == 'suggest':
                if order is None:
                    logger.error("\n Error: order must be given.")
                    sys.exit()
            else:
                logger.error(" Error")
                sys.exit()
            
            if len(self.nbody) < order:
                self.set_nbody_automatically()
            
            inp.update({'norder': order})
            inp.update({'nbody': [self.nbody[i] for i in range(order)]})
            
            ### set cutoffs for alamode
            cutoffs_alamode = {}
            for i1 in range(self.num_elems):
                for i2 in range(self.num_elems):
                    lab = "%s-%s" % (
                            inp.as_dict()['kd'][i1],
                            inp.as_dict()['kd'][i2]
                            )
                    cutoffs_alamode[lab] = np.where(
                            self.cutoffs[:order,i1,i2]<0., 
                            None, self.cutoffs[:order,i1,i2]
                            )
            
            inp.update({'cutoff': cutoffs_alamode})
            
            if propt in ['cv', 'lasso']:
                inp.update({'lmodel': 'enet'})
                inp.update({'l1_ratio': 1.0})
                inp.update({'maxiter': 10000})   ## use the original default, 10000
                inp.update({'conv_tol': 1e-10})  ## strincter than the default, 1e-8
                if propt == 'cv':
                    inp.update({'cv': 5})
                    inp.update({'cv_nalpha': 50})
                elif propt == 'lasso':
                    alpha = self.get_suggested_l1alpha(order=order)
                    inp.update({'cv': 0})
                    inp.update({'l1_alpha': alpha})      ### read l1_alpha
        
        else:
            logger.error(" Error: %s is not supported." % propt)
            sys.exit()
        
        ## Add Grüneisen parameters
        if propt.startswith('gruneisen'):
            inp.update({'gruneisen': 1})


class NameHandler():
    
    def _get_file_pattern(self, order):
        if order == 1:
            filename = (self.out_dirs['harm']['suggest'] + 
                    '/%s.pattern_HARMONIC' % (self.prefix))
        elif order == 2:
            filename = (self.out_dirs['cube']['suggest'] + 
                    '/%s.pattern_ANHARM%d' % (self.prefix, order+1))
        else:
            logger.error(f"\n Error: order {order} is not supported.")
            sys.exit()
        return filename
    
    def _get_logfile_suggest(self, order):
        if order == 1:
            filename = self.out_dirs['harm']['suggest'] + '/suggest.log'
        elif order == 2:
            filename = self.out_dirs['cube']['suggest'] + '/suggest.log'
        else:
            filename = self.out_dirs['higher']['suggest'] + "/suggest.log"
        return filename
    
    def _get_work_directory(self, propt, order=None):
        
        if propt in ['band', 'dos']:
            workdir = self.out_dirs['harm']['bandos']
            
        elif propt.startswith('kappa'):
            workdir = self.out_dirs['cube']['kappa_%s' % self.fc3_type]
        
        elif propt.startswith('gruneisen'):
            workdir = self.out_dirs['cube']['gruneisen']
        
        elif propt == 'evec_commensurate':
            workdir = self.out_dirs['harm']['evec']
        
        elif propt in ['cv', 'lasso']:
            if order == 2:
                workdir = self.out_dirs['cube'][propt]
            else:
                workdir = self.out_dirs['higher'][propt]
            
        elif propt == 'fc2':
            workdir = self.out_dirs['harm']['force']
        
        elif propt == 'fc3':
            workdir = self.out_dirs['cube']['force_%s' % self.fc3_type]
        
        elif propt == 'suggest':
            if order is None:
                order = 1
            
            if order == 1:
                workdir = self.out_dirs['harm']['suggest']
            elif order == 2:
                workdir = self.out_dirs['cube']['suggest']
            elif order > 2:
                workdir = self.out_dirs['higher']['suggest']
            else:
                logger.error("\n Error: order must be given properly.")
                sys.exit()
        
        elif propt == 'scph':
            workdir = self.out_dirs['higher']['scph']
        
        else:
            msg = "\n Error: %s property is not supported.\n" % (propt)
            logger.error(msg)
            sys.exit()
        
        return workdir

def get_cutoffs_automatically(cutoff2=-1, cutoff3=4.3, num_elems=None, order=5):
    """ 
    Parameters
    -----------
    cutoff2 : float, unit=Ang
    cutoff3 : float, unit=Ang
    """
    cutoffs = []
    n = num_elems
    for i in range(order):
        if i == 0:
            cc = cutoff2
        else:
            cc = cutoff3
        cutoffs.append(np.asarray(np.ones((n,n)) * cc))
    return np.asarray(cutoffs)

def read_frequency_range(filename, format='anphon'):
    """ read minimum and maximum frequencies from .bands file created by anphon 
    """
    if format == 'anphon':
        
        data = np.genfromtxt(filename)
        
        fmax = np.nan
        fmin = np.nan
        for idx in [-1, 1]:
            branch = data[:,idx]
            branch = branch[~np.isnan(branch)]
            if idx == -1:
                fmax = np.max(branch)
            elif idx == 1:
                fmin = np.min(branch)
        
        return fmin, fmax
    else:
        msg = " Error: %s is not supported yet." % format
        logger.error(msg)
        sys.exit()

def should_rerun_alamode(logfile):
    """
    Args
    ======
    logfile : string
        alamode log file
    """
    if not os.path.exists(logfile):
        if logfile.startswith('/'):
            logfile = os.path.join(".", os.path.relpath(logfile, os.getcwd()))
        msg = "\n Warning: %s does not exist." % logfile
        logger.warning(msg)
        sys.exit()
        # return True
    else:
        if not wasfinished_alamode(logfile):
            return True
    return False

def should_rerun_band(filename):
    """ Check phonon dispersion calculated with ALAMODE
    
    Parameters
    -----------
    filename : string
        alamode band file (.bands file)
        
    """
    data = np.genfromtxt(filename)
    data = data[:,1:]
    n1 = len(data)
    n2 = len(data[0])
    eigenvalues = data.reshape(n1*n2)
    nan_data = eigenvalues[np.isnan(eigenvalues)]
    if len(nan_data) == 0:
        return False
    else:
        return True

def read_kappa(dir_kappa, prefix, dim=3, norm_idx=None, kappa_scale=1.0):
    """ Read .kl and .kl_coherent files and return pandas.DataFrame object
    
    Parameters
    -----------
    dir_kappa : string
        directory in which .kl and .kl_coherent should exist.
    
    prefix : string
    
    dim : int
        dimension of the system, 2 or 3.
    
    norm_idx : int or None
        The out-of-plane direction index for 2D systems.
        If dim == 2, norm_idx must be given.
    
    """
    fn_kp = dir_kappa + '/' + prefix + '.kl'
    fn_kc = dir_kappa + '/' + prefix + '.kl_coherent'
    
    ##
    if os.path.exists(fn_kp) == False:
        return None
    
    df = pd.DataFrame()
    data = np.genfromtxt(fn_kp)
    
    if os.path.exists(fn_kc) == False:
        fn_kc = None
        data2 = None
    else:
        data2 = np.genfromtxt(fn_kc)

        if np.max(abs(data[:,0] - data2[:,0])) > 0.5:
            msg = " Warning: temperatures are incompatible."
            logger.warning(msg)
    
    ## Prepare direction keys
    dirs = ['x', 'y', 'z']
    if dim == 2:
        if norm_idx is None:
            msg = "\n Warning: norm_idx must be given for 2D systems."
            msg += "\n Thermal conductivity may is not calculated properly."
            logger.error(msg)
            norm_idx = 2
        dirs = [dirs[i] for i in range(3) if i != norm_idx]
    
    nt = len(data)
    df['temperature'] = data[:,0]
    for i1 in range(dim):
        d1 = dirs[i1]
        
        if data2 is not None:
            dd = dirs[i1]
            lab2 = 'kc_%s%s' % (dd, dd)
            df[lab2] = data2[:,i1+1] * kappa_scale
        
        for i2 in range(dim):
            d2 = dirs[i2]
            num = i1*3 + i2 + 1
            lab = 'kp_%s%s' % (d1, d2)
            df[lab] = data[:,num] * kappa_scale
    
    kave = np.zeros(nt)
    for i in range(dim):
        key = 'kp_%s%s' % (dirs[i], dirs[i])
        kave += df[key].values / dim
    df['kp_ave'] = kave
    
    ### If coherent contribution is available
    if data2 is not None:    
        ## Coherent contribution
        if dim >= 2:
            kave = np.zeros(nt)
            for idim in range(dim):
                key = 'kc_%s%s' % (dirs[idim], dirs[idim])
                kave += df[key].values / dim
        else:
            msg = "\n Error: dim=%d is not supported." % dim
            logger.error(msg)
            sys.exit()
        df['kc_ave'] = kave
        
        ## Total thermal conductivity (kp + kc)
        for idir, pre in enumerate(['xx', 'yy', 'zz', 'ave']):
            if dim == 2 and idir == norm_idx:
                continue
            key = 'ksum_%s' % pre
            df[key] = df[f'kp_{pre}'].values + df[f'kc_{pre}'].values
    
    return df

def are_forces_available(filename):
    """ Check vasprun.xml file. If forces are available, return True, while if
    not, return False. """
    
    import ase.io
    try:
        atoms = ase.io.read(filename, format='vasp-xml')
        forces = atoms.get_forces()
        n1 = len(forces)
        for i1 in range(n1):
            for j in range(3):
                if isinstance(forces[i1,j], float) == False:
                    return False
        ##
        return True
    except Exception:
        return False

