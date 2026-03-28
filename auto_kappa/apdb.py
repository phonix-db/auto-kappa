#
# apdb.py
#
# ApdbVasp class treats structures and run a few calculations such as relaxation
# calculation and Born effective charge calculation for anharmonic phonon
# database (APDB).
#
# Copyright (c) 2022 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
import os.path
import os
import sys
import glob

import ase.io
from pymatgen.core.structure import Structure

from auto_kappa.structure.crystal import (
    get_primitive_structure_spglib,
    get_standardized_structure_spglib,
    change_structure_format,
    get_spg_number,
    transform_prim2unit,
    get_supercell
    )
from auto_kappa.structure.two import adjust_vacuum_size
from auto_kappa.calculators.vasp import run_vasp, backup_vasp
from auto_kappa.io.vasp import print_vasp_params, wasfinished
from auto_kappa.cui import ak_log
from auto_kappa.vasp.params import reflect_previous_jobs
from auto_kappa.compat import get_previously_used_structure

import logging
logger = logging.getLogger(__name__)

class ApdbVasp():
    
    def __init__(
            self, unitcell, 
            primitive_matrix=None,
            scell_matrix=None,
            encut_scale_factor=1.3,
            command={'mpirun': 'mpirun', 'nprocs': 2, 'vasp': 'vasp'},
            amin_params = {},
            vasp_config={
                'params': None,
                'setups': {'base': 'recommended', 'W': '_sv'},
                'xc': 'pbesol',
            },
            mater_dim=3,
            base_directory=None
            ):
        """
        Args
        -------
        unitcell : structure object
            original unitcell (conventional) structure
        
        primitive_matrix : float, shape=(3,3)
            transformation matrix from the unitcell to the primitive cell with 
            the definition in Phonopy. Note that the definition in Phonopy is 
            not same as that in Pymatgen and ASE
            
        scell_matrix : float, shape=(3,3)
            transformation matrix from the unitcell to the supercell with 
            the definition in Phonopy, which is not same as Pymatgen and ASE
        
        vasp_config : dict
            vasp_params : dict
                INCAR parameters for different modes of VASP calculations.
                VASP calculation will be performed using the default parameters for
                different calculations, including relax, nac, force calculations, 
                if this parameter is not given (None). If this parameter is given,
                the given VASP paremeters with this function parameter will be set 
                for every calculations.
        
        mater_dim : int
            Material dimension. Default is 3.
        
        Note
        ------
        Translational vectors of the primitive cell can be calculated as
        $$
        pcell = primitive_matrix.T @ unitcell.cell
        pcell = np.dot(primitive_matrix.T, unitcell.cell)
        $$
        
        """
        ### Transformation matrices
        ### The definition is the same as that for Phonopy.
        ### Please see the tutorial of Phonopy in detail.
        ### https://phonopy.github.io/phonopy/setting-tags.html#basic-tags
        self._mat_u2p = primitive_matrix
        self._mat_u2s = scell_matrix
        self._mater_dim = mater_dim
        self._base_dir = base_directory
        
        if primitive_matrix is None:
            msg = " Error: primitive_matrix must be given."
            logger(msg)
        
        if scell_matrix is None:
            msg = " Error: scell_matrix must be given."
            logger(msg)
        
        ### set structure variables
        ### Every structures will be stored in ``self._trajectory``.
        ### For example, self.trajectory[0] is the initial structures and 
        ### self.trajectory[1] is the latest structures.
        self._structures = None
        self._trajectory = []
        self.update_structures(unitcell)
        
        ### VASP command
        self._command = command
        
        ### parameters
        self.encut_factor = encut_scale_factor

        #self._yamlfile_for_outdir = yamlfile_for_outdir
        
        ### AMIN parameters
        from auto_kappa import default_amin_parameters
        self.amin_params = {}
        for key in default_amin_parameters:
            if key in amin_params.keys():
                if amin_params[key] is not None:
                    self.amin_params[key] = amin_params[key]
            if key not in self.amin_params.keys():
                self.amin_params[key] = default_amin_parameters[key]
        
        ### parameters that differ from the default values
        self._vasp_config = vasp_config
    
    @property
    def mater_dim(self):
        """ Material dimension """
        return self._mater_dim
    @property
    def base_directory(self):
        """ Base directory for the auto-kappa calculation """
        return self._base_dir
    
    @property
    def primitive_matrix(self):
        return self._mat_u2p
    
    @property
    def scell_matrix(self):
        return self._mat_u2s
    
    @property
    def command(self):
        return self._command
    
    @property
    def vasp_config(self):
        return self._vasp_config
    
    @property
    def vasp_params(self):
        return self._vasp_config.get('params', {})
    
    @property
    def potcar_setups(self):
        return self._vasp_config.get('setups', {})
    
    @property
    def xc(self):
        return self._vasp_config.get('xc', 'pbesol')
    
    def update_command(self, val):
        self._command.update(val)
    
    def update_structures(self, unitcell, format='ase', standardization=True):
        """ Update unit, primitive, supercells with the given new unit cell.
        Args
        -----
        unitcell : structures obj
            unit cell structure
        """
        if standardization:
            unitcell = get_standardized_structure_spglib(unitcell, to_primitive=False, format=format)
        ##
        structures = self.get_structures(unitcell, format=format)
        self._structures = structures
        self._trajectory.append(structures)
    
    def get_structures(self, unitcell, format='ase'):
        """ Get primitive and supercells with the stored unitcell and 
        transformation matrices.
        """
        try:
            from phonopy import Phonopy
            phonon = Phonopy(
                    change_structure_format(unitcell, format='phonopy'),
                    self._mat_u2s,
                    primitive_matrix=self._mat_u2p
                    )
            unit = change_structure_format(phonon.unitcell , format=format) 
            prim = change_structure_format(phonon.primitive , format=format) 
            sc   = change_structure_format(phonon.supercell , format=format)
        
        except Exception:
            
            unit = change_structure_format(unitcell , format=format)
            prim = get_primitive_structure_spglib(unitcell)
            prim = change_structure_format(prim, format=format)
            sc = get_supercell(unitcell, self.scell_matrix, format=format)
            
        structures = {"unit": unit, "prim": prim, "super": sc}
        return structures
    
    @property
    def structures(self):
        if self._structures is not None:
            return self._structures
        else:
            return None
    
    @property
    def trajectory(self):
        return self._trajectory
    
    @property
    def primitive(self):
        return self._structures['prim']
    
    @property
    def unitcell(self):
        return self._structures['unit']
    
    @property
    def supercell(self):
        return self._structures['super']
    
    def get_calculator(self, mode, directory=None, kpts=None, **args):
        """ Return VASP calculator created by ASE
        
        Args
        ------
        mode : string
            'relax', 'force', 'nac', or 'md'
        
        directory : string
            output directory
        
        kpts : list of float, shape=(3,)
            k-mesh for VASP calculation
        
        **args : dict
            VASP parameters that will be modified which are prior to 
            ``self.vasp_params``
        
        Return
        ------
        ase.calculators.vasp.Vasp
        
        """
        from auto_kappa.calculators.vasp import get_vasp_calculator
        
        ### get structure (Atoms obj)
        if 'relax' in mode.lower() or mode.lower() == 'nac':
            structure = self.primitive
        elif 'force' in mode.lower() or mode.lower() == 'md':
            structure = self.supercell
        
        ## get VASP parameters for the mode: e.g. {'ediff': 1e-8, 'ibrion': 2, ...}
        from auto_kappa.utils.config import get_vasp_parameters_by_mode
        params_ = get_vasp_parameters_by_mode(self.vasp_params, mode=mode)
        
        calc = get_vasp_calculator(params_,
                                   directory=directory, 
                                   atoms=structure,
                                   kpts=kpts,
                                   encut_scale_factor=self.encut_factor,
                                   setups=self.potcar_setups,
                                   xc=self.xc
                                   )
        
        calc.command = f"{self.command['mpirun']} -n {self.command['nprocs']} "
        if list(kpts) == [1, 1, 1]:
            calc.command += f"{self.command['vasp_gam']}"
        else:
            calc.command += f"{self.command['vasp']}"
        
        return calc
    
    def run_relaxation(
            self, directory: str, kpts: None,
            standardize_each_time=True,
            volume_relaxation=0,
            cell_type='p',
            force=False, num_full=2, verbose=1,
            max_error=None, nsw_params=None, 
            **args
            ):
        """ Perform relaxation calculation, including full relaxation 
        calculations (ISIF=3 for 3D while ISIF=4 for 2D?) with "num_full" times 
        and a relaxation of atomic positions (ISIF=2). See descriptions for 
        self.run_vasp for details.
        
        Args
        =======

        directory : string
            working directory for VASP

        kpts : array, shape=(3)
            k-mesh for VASP

        standardize_each_time : bool
        
        volume_relaxation : int

        cell_type : string

        force : bool

        num_full : int,
            Number of relxation calculation w/o any restriction [default: 2]

        verbose : int
        
        max_error : int
            Max number of retry the calculation. If error.{max_error}.tar(.gz) 
            exists, stop the calculation.
        
        args : dictionary
            input parameters for VASP
        
        Return
        ========
        
        integer :
            If negative value, stop the job.
            -1 : symmetry error
            -2 : too many errors
        
        """
        ### relaxation cell type
        if cell_type[0].lower() == 'p':
            cell_type = 'primitive'
            to_primitive = True
        elif cell_type[0].lower() == 'c' or cell_type[0].lower() == 'u':
            cell_type = 'conventional'
            to_primitive = False
        else:
            msg = " Error"
            logger.info(msg)
            sys.exit()
        
        ### message
        if verbose != 0:
            line = "Structure optimization"
            msg = "\n\n " + line
            msg += "\n " + "=" * (len(line))
            msg += "\n\n Cell type : %s" % cell_type
            logger.info(msg)
        
        ### Get the relaxed structure obtained with the old version
        ### For the old version, the xml file is located under ``directory``.
        if volume_relaxation == 0 and wasfinished(directory, filename='vasprun.xml'):
            filename = directory + "/vasprun.xml"
            if os.path.isabs(filename):
                filename = "./" + os.path.relpath(filename, os.getcwd())
            prim = ase.io.read(filename, format='vasp-xml')
            unitcell = transform_prim2unit(prim, self.primitive_matrix)
            self.update_structures(unitcell)
            msg = "\n Already finised with the old version (single full relaxation)"
            msg += "\n Read the structure from %s" % filename
            logger.info(msg)
            return 0
        
        ### Read previously used structure
        test_job = False
        
        ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        unitcell = get_previously_used_structure(self.base_directory, self.primitive_matrix,
                                                 orig_structures=self.structures.copy())
        if unitcell is not None and not test_job:
            self.update_structures(unitcell)
            return 0
        ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        ### NSW parameters
        out = _parse_nsw_params(nsw_params)
        nsw_init = out[0]
        nsw_diff = out[1]
        nsw_min = out[2]
        
        ### symmetry
        spg_before = get_spg_number(self.unitcell)
        
        ### perform relaxation calculations
        count = 0
        count_err = 0
        max_sym_err = 2
        while True:
            
            if self.mater_dim < 3:
                break
            
            ### set working directory and mode
            if count < num_full:
                ## full relxation
                num = count + 1
                dir_cur = directory + "/full-%d" % num
                mode = 'relax-full'
            else:
                ## relaxation of atomic positions
                num = count - num_full + 1
                dir_cur = directory + "/freeze-%d" % num
                mode = 'relax-freeze'
            
            #### check the number of errors
            #if max_error is not None:
            #    if too_many_errors(dir_cur, max_error=max_error):
            #        return -2
            
            #### determine NSW parameter based on the number of errors
            args['nsw'] = _get_nsw_parameter(
                    dir_cur, nsw_init=nsw_init, 
                    nsw_diff=nsw_diff, nsw_min=nsw_min)
            
            ### print message
            if verbose != 0:
                line = "%s (%d)" % (mode, num)
                msg = "\n " + line
                msg += "\n " + "-" * len(line)
                logger.info(msg)
            
            ##
            if count == 0:
                if count_err == 0:
                    print_params = True
            else:
                fn = dir_pre + "/CONTCAR"
                if os.path.exists(fn) == False:
                    msg = "\n Error: %s does not exist." % fn
                    logger.error(msg)
                    sys.exit()
                
                print_params = False
            
            ### get the structure used for the analysis
            if to_primitive:
                structure = self.primitive
            else:
                structure = self.unitcell
            
            ### run a relaxation calculation
            ### out == -1 : symmetry was changed
            out = self.run_vasp(
                    mode, dir_cur, kpts, 
                    structure=structure, force=force, 
                    print_params=print_params,
                    cell_type=cell_type,
                    verbose=0,
                    standardization=standardize_each_time,
                    **args
                    )
            
            if out == -1:
                
                ### backup failed result
                backup_vasp(dir_cur, delete_files=True)
                
                ### set ISYM = 2 explicitly
                #args["isym"] = 2
                
                count_err += 1
                if max_sym_err == count_err:
                    msg =  "\n The calculation was failed %d times." % (count_err)
                    msg += "\n Abort the relaxation calculation."
                    logger.info(msg)
                    return -1
                else:
                    logger.info("\n Retry the relaxation calculation.")
                    continue
            
            ### update
            dir_pre = dir_cur
            
            count += 1
            count_err = 0
            if count == num_full + 1:
                break
        
        ### update structures
        self.update_structures(self.unitcell, standardization=True)
        
        ### strict relaxation with Birch-Murnaghan EOS
        if volume_relaxation or self.mater_dim < 3:
            
            from auto_kappa.vasp.relax import StrictRelaxation
            outdir = directory + "/volume"
            
            if to_primitive:
                structure = self.primitive
            else:
                structure = self.unitcell
            
            init_struct = change_structure_format(structure, format='pmg')
            
            ### check the previous optimal structure
            ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            struct_opt = None
            struct_opt = _get_previous_optimal_structure(
                directory, prim_matrix=self.primitive_matrix, to_primitive=to_primitive)
            ###
            ## struct_opt = None; logger.info(" TEST JOB!!!!!!!!")
            ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            
            if struct_opt is None or test_job:
                relax = StrictRelaxation(init_struct, outdir=outdir, dim=self.mater_dim)
                Vs, Es = relax.with_different_volumes(
                        kpts=kpts, command=self.command, 
                        vasp_params=self.vasp_params,
                        potcar_setups=self.potcar_setups,
                        xc=self.xc,
                        initial_strain_range=[-0.03, 0.05], nstrains=15
                        )
                
                ### output optimized structure file
                struct_opt = relax.get_optimal_structure()
            
            try:
                # ### output figure
                # figname = outdir + '/fig_bm.png'
                # relax.plot_bm(figname=figname.replace(os.getcwd(), "."))
                
                figname = outdir + '/fig_bm.png'
                relax.plot_physical_properties(figname=figname)
                
                ### print results
                relax.print_results()
            
            except Exception:
                pass
            
            ### output optimized structure file (POSCAR.opt)
            outfile = outdir + "/POSCAR.opt"
            if not isinstance(struct_opt, ase.Atoms):
                struct_ase = change_structure_format(struct_opt, format='ase')
            else:
                struct_ase = struct_opt.copy()
            ase.io.write(outfile, struct_ase, format='vasp', direct=True, vasp5=True, sort=False)
            
            ### update structures
            if to_primitive:
                unitcell = transform_prim2unit(struct_ase, self.primitive_matrix, format='ase')
            else:
                unitcell = struct_ase.copy()
            
            ## Adjust the vaccum space size for VASP calculation
            if self.mater_dim < 3:
                unitcell = adjust_vacuum_size(unitcell)
            
            self.update_structures(unitcell)
        
        ## Adjust the vacuum space size based the supercell
        if self.mater_dim < 3:    
            unit_mod = adjust_vacuum_size(self.unitcell, self.scell_matrix)
            self.update_structures(unit_mod)
        
        ### Check the crystal symmetry before and after the relaxation
        spg_after = get_spg_number(self.primitive)
        
        self._write_relax_yaml({
            'directory': directory,
            'cell_type': cell_type,
            'structure': self.unitcell,
            'spg': [spg_before, spg_after],
            'volume_relaxation': volume_relaxation,
            })
        
        self.output_structures(verbose=False)
        
        if spg_before != spg_after:
            ak_log.symmetry_error(spg_before, spg_after)
            return -1
        
        return 0
    
    def output_structures(self, verbose=True):
        """ Output structures (>= ver.0.4.0)
        """
        outdir = self.base_directory + "/relax/structures"
        os.makedirs(outdir, exist_ok=True)
        if verbose:
            logger.info("")
        for key in self.structures.keys():
            fn = outdir.replace(os.getcwd(), ".") + "/POSCAR.%s" % key
            ase.io.write(fn, self.structures[key], format='vasp', direct=True, vasp5=True, sort=False)
            if verbose:
                logger.info(" Output %s" % fn)
        
        
    def _write_relax_yaml(self, params):
        import yaml
        outfile = params['directory'] + '/relax.yaml'
        structure = change_structure_format(params['structure'], format='pymatgen') 
        
        ### lattice vectors
        lattice = []
        for v1 in structure.lattice.matrix:
            lattice.append([])
            for val in v1:
                lattice[-1].append(float(val))
        
        ### fractional coords
        frac_coord = []
        for pos in structure.frac_coords:
            frac_coord.append([])
            for j in range(3):
                frac_coord[-1].append(float(pos[j]))
        
        ### species
        species = [el.name for el in structure.species]
        
        dict_data = {
                'directory': params['directory'],
                'cell_type_for_relaxation': params['cell_type'],
                'spg_before': params['spg'][0],
                'spg_after': params['spg'][1],
                'lattice': lattice,
                'positions': frac_coord,
                'species': species,
                'volume_relaxation': params['volume_relaxation'],
                }

        with open(outfile, 'w') as f:
            yaml.dump(dict_data, f)
            
    def run_vasp(self, mode: None, directory: str, kpts: None, 
                 structure=None, cell_type=None,
                 method='custodian', force=False, print_params=False, 
                 standardization=True, verbose=1, vaccum_thickness=None,
                 **args):
        """ Run relaxation and born effective charge calculation
        
        Args
        -------
        mode : string
            "relax-full", "relax-freeze", "force", "nac", or "md"
        
        directory : string
            output directory
        
        kpts : array of float, shape=(3,)

        structure : structure obj

        cell_tyep : string
            cell type of ``structure``: primitive or conventional
            This is used only for ``mode = relax-***``
        
        method : string
            "custodian" or "ase"

        force : bool, default=False
            If it's True, the calculation will be done forcelly even if it had
            already finished.
        
        vaccum_thickness : float, default=None
            If the material dimension is 2 and this parameter is given,
            the vacuum thickness will be set to the given value.
        
        args : dict
            input parameters for VASP
        
        Return
        --------
        integer :
            0. w/o error
            1. symmetry was changed during the relaxation calculation
        
        """
        if verbose != 0:
            line = "VASP calculation (%s)" % (mode)
            msg = "\n\n " + line
            msg += "\n " + "=" * (len(line))
            logger.info(msg)
        
        ### set OpenMP
        omp_keys = ["OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK"]
        for key in omp_keys:
            os.environ[key] = str(self.command['nthreads'])
        
        ### perform the calculation
        if wasfinished(directory, filename='vasprun.xml') and force == False:
            msg = "\n The calculation has already been done."
            logger.info(msg)
        
        else:
            ### ver.1 relax with one shot
            calc = self.get_calculator(
                    mode.lower(), directory=directory, kpts=kpts, **args)
            
            ### set structure
            if structure is None:
                structure = self.primitive
            
            ### update VASP parameters based on the previous jobs
            reflect_previous_jobs(
                calc, structure, method=method, 
                amin_params=self.amin_params)
            
            ### print VASP parameters
            if print_params:
                print_vasp_params(calc.asdict()['inputs'])
            
            ### Adjust the vacuum size for VASP calculation
            if self.mater_dim == 2 and vaccum_thickness is not None:
                #
                # Note: This adjustment of the vacuum size leads to a difference 
                # in the cell size along the out-of-plane direction and 
                # alters the displacement-force dataset. However, this change 
                # does not affect the final result, i.e., the force constants.
                # 
                from auto_kappa.structure.two import set_vacuum_to_2d_structure
                struct_2d = set_vacuum_to_2d_structure(structure, vaccum_thickness)
                struct4vasp = change_structure_format(struct_2d, format='ase')
            else:
                struct4vasp = structure
            
            ### run a VASP job
            run_vasp(calc, struct4vasp, method=method)
            
        ### set back OpenMP 
        for key in omp_keys:
            os.environ[key] = "1"
         
        ### Read the relaxed structure
        if 'relax' in mode.lower():
            
            vasprun = directory + "/vasprun.xml"
            cell_type = cell_type.lower()
            
            if cell_type.startswith('conv') or cell_type.startswith('unit'):    
                try:
                    new_unitcell = ase.io.read(vasprun, format='vasp-xml')
                except Exception:
                    _error_in_vasprun(vasprun)
            elif cell_type.startswith('prim'):
                ### read primitive and transform it to the unit cell
                new_prim = ase.io.read(vasprun, format='vasp-xml')
                new_unitcell = transform_prim2unit(new_prim, self.primitive_matrix)
            else:
                msg = "\n Error: cell_type must be primitive or conventional/unitcell."
                logger.info(msg)
                sys.exit()
            
            num_init = get_spg_number(structure)
            num_mod = get_spg_number(new_unitcell)
            if num_init != num_mod:
                ak_log.symmetry_error(num_init, num_mod)
                return -1
            
            self.update_structures(new_unitcell, standardization=standardization)
        
        return 0

def _error_in_vasprun(filename):
    dir_file = os.path.dirname(filename)
    msg = "\n Error in %s" % filename 
    msg += "\n Abort the calculation"
    logger.info(msg)
    sys.exit()

def too_many_errors(directory, max_error=100):
    """ check the number of errors in ``directory`` """    
    for file_err in glob.glob(directory+"/error.*"):
        try:
            num = int(file_err.split("/")[-1].split(".")[1])
        except Exception:
            continue
        if num >= max_error:
            return True
    return False

#def _get_number_of_errors(directory):
#    """ Get and return the number of errors in the given directory. """
#    num_errors = 0
#    ### number of errors
#    for suffix in ["tar", "tar.gz"]:
#        line = directory + "/error.*." + suffix
#        fns = glob.glob(line)
#        num_errors += len(fns)
#    ####
#    #line = directory + "/INCAR"
#    #fns = glob.glob(line)
#    #num_errors += len(fns)
#    return num_errors

def _parse_nsw_params(line, params_default=[200, 10, 20]):
    """ Return NSW params with an array 
    Args
    ======
    line : string, "**:**:**"

    Return
    =======
    array, shape=(3)
        initial, interval, and minimum NSW
    """
    data = line.split(":")
    params = []
    for j in range(3):
        try:
            params.append(int(data[j]))
        except Exception:
            params.append(int(params_default[j]))
    return params

def _get_nsw_parameter(directory, nsw_init=200, nsw_diff=10, nsw_min=20):
    """ Determine the number of NSW based on the number of errors """
    from auto_kappa.vasp.params import get_number_of_errors
    num_errors = get_number_of_errors(directory)
    nsw = max(nsw_min, nsw_init - nsw_diff * num_errors)
    return nsw

def _get_previous_optimal_structure(outdir, prim_matrix=None, to_primitive=True):
    
    format = 'pmg'
    
    filenames = [
        f"{outdir}/volume/POSCAR.opt",
        # f"{outdir}/../harm/force/prist/POSCAR",
    ]
    
    for i, fn in enumerate(filenames):
        if os.path.exists(fn):
            struct = Structure.from_file(fn)
            prim = get_primitive_structure_spglib(struct)
            
            relpath = os.path.relpath(fn, os.getcwd())
            msg = f"\n Read the previous optimal structure: ./{relpath}"
            logger.info(msg)
            
            if to_primitive:
                opt_struct = change_structure_format(prim, format=format)
            else:
                unitcell = transform_prim2unit(prim, prim_matrix, format=format)
                opt_struct = unitcell
            
            return opt_struct
    return None
