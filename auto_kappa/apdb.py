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
from phonopy.structure.cells import get_supercell

from auto_kappa.structure.crystal import (
    get_primitive_structure_spglib,
    get_standardized_structure_spglib,
    change_structure_format,
    get_spg_number,
    convert_primitive_to_unitcell
    )
from auto_kappa.structure.two import adjust_vacuum_size
from auto_kappa.calculators.vasp import run_vasp, backup_vasp
from auto_kappa.calculators.mlips import MLIPSCalculatorFactory, run_calculation
from auto_kappa.io.vasp import print_vasp_params, wasfinished, wasfinished_mlips
from auto_kappa.cui import ak_log
from auto_kappa.vasp.params import reflect_previous_jobs
from auto_kappa.compat import get_previously_used_structure
from auto_kappa import output_directories

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
            params_modified = None,
            mater_dim=3,
            base_directory=None,
            use_mlips=False,
            model_name="esen",
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
        
        params_modified : dict
            INCAR parameters to be modified.
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
        self.use_mlips = use_mlips
        self.model_name = model_name

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
        self.set_modified_params(params_modified)
    
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
    def params_mod(self):
        if self._params_mod is None:
            return {}
        else:
            return self._params_mod
    
    def set_modified_params(self, params_mod):
        self._params_mod = params_mod
    
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
            sc = get_supercell(
                    change_structure_format(unitcell, format='phonopy'),
                    self.scell_matrix)
            sc = change_structure_format(sc, format=format)
        
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
    
    def get_calculator(self, mode, directory=None, kpts=None, use_mlips=False, model_name='esen', **args):
        """ Return calculator (VASP or MLIPS) based on configuration
        
        Args
        ------
        mode : string
            'relax', 'force', 'nac', or 'md'
        
        directory : string
            output directory
        
        kpts : list of float, shape=(3,)
            k-point mesh
        
        use_mlips : bool, optional
            Whether to use MLIPS. If None, uses self.use_mlips
            
        model_name : string, optional
            MLIPS model name ("esen", "mace"). If None, uses self.model_name
        
        **args : dict
            Additional parameters
        
        """
        from auto_kappa.calculators.vasp import get_vasp_calculator
        
        ### get structure (Atoms obj)
        if 'relax' in mode.lower() or mode.lower() == 'nac':
            structure = self.primitive
        elif 'force' in mode.lower() or mode.lower() == 'md':
            structure = self.supercell
        
        if use_mlips:
            calc = MLIPSCalculatorFactory.create_calculator(model_name=model_name)
            if directory is not None:
                calc.directory = directory
            return calc
        ### merge ``args`` and ``self.params_mod``
        ### `args`` is prior to ``self.params_mod`
        merged_params_mod = self.params_mod.copy()
        merged_params_mod.update(args)
        
        calc = get_vasp_calculator(mode, 
                                   directory=directory, 
                                   atoms=structure,
                                   kpts=kpts,
                                   encut_scale_factor=self.encut_factor,
                                   **merged_params_mod)
        
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
            force=False, num_full=2, verbosity=1,
            max_error=None, nsw_params=None, 
            **args
            ):
        """ Perform relaxation calculation, including full relaxation 
        calculations (ISIF=3) with "num_full" times and a relaxation of atomic
        positions (ISIF=2). See descriptions for self.run_vasp for details.
        
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

        verbosity : int
        
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
        if verbosity != 0:
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
            unitcell = convert_primitive_to_unitcell(prim, self.primitive_matrix)
            self.update_structures(unitcell)
            msg = "\n Already finised with the old version (single full relaxation)"
            msg += "\n Read the structure from %s" % filename
            logger.info(msg)
            return 0
        
        ### Read previously used structure
        unitcell = get_previously_used_structure(
            self.base_directory, self.primitive_matrix, self.scell_matrix)
        
        if unitcell is not None:
            self.update_structures(unitcell)
            return 0
        
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
            if verbosity != 0:
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
                    verbosity=0,
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
            struct_opt = _get_previous_optimal_structure(
                directory, prim_matrix=self.primitive_matrix, to_primitive=to_primitive)
            ###
            # struct_opt = None; logger.info(" TEST JOB!!!!!!!!")
            
            if struct_opt is None:
                relax = StrictRelaxation(init_struct, outdir=outdir, dim=self.mater_dim)
                Vs, Es = relax.with_different_volumes(
                        kpts=kpts, command=self.command, params_mod=self.params_mod,
                        initial_strain_range=[-0.03, 0.05], nstrains=15
                        )
                
                ### output figure
                figname = outdir + '/fig_bm.png'
                relax.plot_bm(figname=figname.replace(os.getcwd(), "."))
                
                ### print results
                relax.print_results()
                
                ### output optimized structure file
                struct_opt = relax.get_optimal_structure()
                
            outfile = outdir + "/POSCAR.opt"
            struct_opt.to(filename=outfile.replace(os.getcwd(), "."))
            struct_ase = change_structure_format(struct_opt, format='ase')
            
            ### update structures
            if to_primitive:
                # _mat_p2u = np.linalg.inv(self.primitive_matrix)
                # _mat_p2u = np.array(np.sign(_mat_p2u) * 0.5 + _mat_p2u, dtype="intc")
                # unitcell = get_supercell(
                #         change_structure_format(struct_ase, format='phonopy'),
                #         _mat_p2u)
                
                unitcell = convert_primitive_to_unitcell(struct_ase, self.primitive_matrix, format='ase')
                
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
        
        ### output structures (>= ver.0.4.0)
        outdir = directory + "/structures"
        os.makedirs(outdir, exist_ok=True)
        logger.info("")
        for key in self.structures.keys():
            fn = outdir.replace(os.getcwd(), ".") + "/POSCAR.%s" % key
            ase.io.write(fn, self.structures[key], format='vasp', direct=True, vasp5=True, sort=True)
            logger.info(" Output %s" % fn)
        
        if spg_before != spg_after:
            ak_log.symmetry_error(spg_before, spg_after)
            return -1
        
        return 0

    def run_mlips_relaxation(
        self, directory=None, kpts=None, 
        standardize_each_time=True,
        volume_relaxation=0,
        cell_type='p',
        force=False, num_full=2, verbose=1,
        fmax=0.01, max_steps=500, 
        **args):
        """
        Run MLIPS calculation for strict relaxation.

        For MLIPS: Simplified single-step comprehensive relaxation that optimizes
        both atomic positions and cell parameters simultaneously, taking advantage
        of MLIPS's fast evaluation and stability.
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
            line = "MLIPS Structure optimization"
            msg = "\n\n " + line
            msg += "\n " + "=" * (len(line))
            msg += "\n\n Cell type : %s" % cell_type
            logger.info(msg)

        ### Get the relaxed structure obtained with the old version
        ### For the old version, check appropriate files based on the calculator method
        calculation_finished = False
        if volume_relaxation == 0:
            calculation_finished = wasfinished_mlips(directory)
        if calculation_finished:
            contcar_file = directory + "/CONTCAR"
            if os.path.isabs(contcar_file):
                prim = ase.io.read(contcar_file, format='vasp')
            else:
                # Try to read from force.xyz or other MLIPS output files
                msg = "\n Warning: CONTCAR not found for MLIPS calculation"
                logger.warning(msg)
                calculation_finished = False
        
        if calculation_finished:
            unitcell = transform_prim2unit(prim, self.primitive_matrix)
            self.update_structures(unitcell)
            msg = "\n Already finised with the old version (single full relaxation)"
            msg += "\n Read the structure from CONTCAR"
            logger.info(msg)
            return 0

        ### Read previously used structure
        test_job = False
        
        unitcell = get_previously_used_structure(self.base_directory, self.primitive_matrix, self.scell_matrix)
        
        if unitcell is not None and not test_job:
            self.update_structures(unitcell)
            return 0

        ### symmetry
        spg_before = get_spg_number(self.unitcell)

        ### perform relaxation calculations
        count = 0
        count_err = 0
        max_sym_err = 2

        ### run a single MLIPS relaxation calculation
        max_relax_steps = 1
        logger.info("\n MLIPS relaxation calculation (single step)")

        while True:
            if self.mater_dim < 3:
                break
            
            ### set working directory and mode
            if count == 0:
                dir_cur = directory + "/mlips_relax"
                mode = 'relax-full'
                num = 1
            else:
                break # only one step for MLIPS relaxation

            ### print message
            if verbose != 0:
                line = "MLIPS conprehensive relaxation (%d)" % num
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
            out = self.run_mlips(
                mode, dir_cur, kpts,
                structure=structure, force=force,
                print_params=print_params,
                cell_type=cell_type,
                verbose=0,
                standardization=standardize_each_time,
                model_name=self.model_name,
                **args
            )

            if out == -1:

                ### backup failed result (MLIPS specific files)
                mlips_files = {"POSCAR", "CONTCAR", "forces.xyz", "energy.txt", "stress.txt"}
                backup_vasp(dir_cur, filenames=mlips_files, delete_files=True)

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

            # check termination condition
            if count == max_relax_steps:
                break
        
        ### update structures
        self.update_structures(self.unitcell, standardization=True)

        ### strict relaxation with Birch-Murnaghan EOS
        if volume_relaxation or self.mater_dim < 3:
            
            from auto_kappa.vasp.relax import MLIPSStrictRelaxation
            outdir = directory + "/volume_mlips"

            if to_primitive:
                structure = self.primitive
            else:
                structure = self.unitcell

            init_struct = change_structure_format(structure, format='pmg')

            ### check the previous optimal structure
            struct_opt = None
            struct_opt = _get_previous_optimal_structure(
                directory, prim_matrix=self.primitive_matrix, to_primitive=to_primitive)

            if struct_opt is None or test_job:
                # Create MLIPS calculator
                mlips_calc = MLIPSCalculatorFactory.create_calculator(
                    model_name=self.model_name)
                # Create StrictRelaxation object with MLIPS calculator
                relax = MLIPSStrictRelaxation(
                    initial_structure=init_struct, 
                    mlips_calculator=mlips_calc,
                    calc_type=self.model_name,
                    outdir=outdir, 
                    dim=self.mater_dim
                )

                Vs, Es = relax.with_different_volumes(
                        initial_strain_range=[-0.03, 0.05], 
                        nstrains=15,
                        fmax=0.01,      # Force convergence for MLIPS
                        maxstep=0.2,    # Maximum step size
                        max_steps=500   # Maximum optimization steps
                        )

                ### output optimized structure file
                struct_opt = relax.get_optimal_structure()
            else:
                logger.info(" Found previous optimal structure. Skip MLIPS strict relaxation.")

            try:
                # ### output figure
                # figname = outdir + '/fig_bm.png'
                # relax.plot_bm(figname=figname.replace(os.getcwd(), "."))

                figname = outdir + f'/fig_bm_{self.model_name}.png'
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

        ### output structures
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
            standardization=True, verbosity=1, vaccum_thickness=None,
            **args
            ):
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
        if verbosity != 0:
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
                new_unitcell = convert_primitive_to_unitcell(new_prim, self.primitive_matrix)
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

    def run_mlips(self, mode: None, directory: str, kpts: None, 
                 structure=None, cell_type=None,
                 method='ase', force=False, print_params=False, 
                 standardization=True, verbose=1, vaccum_thickness=None,
                 model_name='esen', **args):
        """ Run MLIPS calculation for relaxation and born effective charge calculation
        
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
            "ase"

        force : bool, default=False
            If it's True, the calculation will be done forcelly even if it had
            already finished.
        
        vaccum_thickness : float, default=None
            If the material dimension is 2 and this parameter is given,
            the vacuum thickness will be set to the given value.
        
        args : dict
            input parameters for MLIPS
        
        Return
        --------
        integer :
            0. w/o error
            1. symmetry was changed during the relaxation calculation
        
        """
        if verbose != 0:
            line = "MLIPS calculation (%s)" % (mode)
            msg = "\n\n " + line
            msg += "\n " + "=" * (len(line))
            logger.info(msg)

        ### perform the calculation
        if wasfinished_mlips(directory) and force == False:
            msg = "\n The calculation has already been done."
            logger.info(msg)
        
        else:
            ### ver.1 relax with one shot
            calc = self.get_calculator(
                    mode.lower(), directory=directory, kpts=kpts, 
                    use_mlips=True, model_name=model_name, **args)
            
            ### set structure
            if structure is None:
                structure = self.primitive
            
            ### update MLIPS parameters based on the previous jobs
            reflect_previous_jobs(
                calc, structure, method=method, 
                amin_params=self.amin_params)
            
            ### print MLIPS parameters
            if print_params:
                logger.info(f"\n MLIPS model: {model_name}"
                            f"\n Calculation mode: {mode}")
            
            ### Adjust the vacuum size for calculation
            if self.mater_dim == 2 and vaccum_thickness is not None:
                #
                # Note: This adjustment of the vacuum size leads to a difference 
                # in the cell size along the out-of-plane direction and 
                # alters the displacement-force dataset. However, this change 
                # does not affect the final result, i.e., the force constants.
                # 
                from auto_kappa.structure.two import set_vacuum_to_2d_structure
                struct_2d = set_vacuum_to_2d_structure(structure, vaccum_thickness)
                struct4calc = change_structure_format(struct_2d, format='ase')
            else:
                struct4calc = structure
            
            ### run a MLIPS job
            if 'relax' in mode.lower():
                # For relaxation calculation, use ASE's optimizer
                result = run_calculation(
                    calc, 
                    struct4calc, 
                    calc_type=self.model_name,
                    fmax=0.01,      # Force convergence for MLIPS
                    max_steps=500   # Maximum optimization steps
                )
            else:
                # For single-point calculation, use standard function
                struct4calc.calc = calc
                struct4calc.get_potential_energy()
                result = 0

        ### Read the relaxed structure
        if 'relax' in mode.lower():
            contcar_file = directory + "/CONTCAR"
            if os.path.exists(contcar_file):
                try:
                    new_unitcell = ase.io.read(contcar_file, format='vasp')
                except Exception:
                    # Fallback: use the current structure from calculator
                    new_unitcell = struct4calc.copy()
            else:
                new_unitcell = struct4calc.copy()

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
        f"{outdir}/volume/POSCAR.opt",        # VASP strict relaxation
        f"{outdir}/volume_mlips/POSCAR.opt",  # MLIPS strict relaxation
        # f"{outdir}/../harm/force/prist/POSCAR",
    ]
    
    # mat_p2u = np.linalg.inv(prim_matrix)
    # mat_p2u = np.array(np.sign(mat_p2u) * 0.5 + mat_p2u, dtype="intc")
    
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
                # unitcell = get_supercell(
                #     change_structure_format(prim, format='phonopy'),
                #     mat_p2u)
                # unitcell = change_structure_format(unitcell, format=format)
                
                unitcell = convert_primitive_to_unitcell(prim, prim_matrix, format=format)
                opt_struct = unitcell
            
            return opt_struct
    return None
