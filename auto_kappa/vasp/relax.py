# -*- coding: utf-8 -*-
#
# relax.py
#
# StrincRelaxation is a class for a strict structure relaxation with
# Birch-Murnaghan equation of state
#
# Copyright (c) 2022 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
#
import sys
import os.path
import math
import numpy as np
import pandas as pd
import ase
import yaml

from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Vasprun

## Birch-Murnaghan equation of state
from pymatgen.analysis.eos import BirchMurnaghan

import matplotlib.pyplot as plt
from auto_kappa.plot import set_matplot, set_axis

from auto_kappa.calculators.vasp import get_vasp_calculator, run_vasp
from auto_kappa.structure import change_structure_format
from auto_kappa.structure.two import get_thickness, get_normal_index
from auto_kappa.units import EvToJ
from auto_kappa.plot.fitting import plot_bm_result

import logging
logger = logging.getLogger(__name__)

class StrictRelaxation():

    def __init__(self, initial_structure, dim=3, outdir="./volume"):
        """
        Args
        ----
        initial_structure : pymatgen Structure or ASE Atoms object
            Initial structure to be relaxed
        """
        if isinstance(initial_structure, Structure) == False:
            initial_structure = change_structure_format(
                initial_structure, format='pmg-Structure')
        
        self.initial_structure = initial_structure
        self.outdir = outdir
        self.outfile_yaml = outdir + "/result.yaml"
        self._optimal_volume = None
        self._dim = dim
        
        self._volumes = None
        self._energies = None
    
    @property
    def optimal_volume(self):
        if self._optimal_volume is not None:
            return self._optimal_volume
        else:
            msg = "\n Caution: Optimal volume is not yet obtained.\n"
            logger.warning(msg)
    
    @property
    def volumes(self):
        if self._volumes is not None:
            return self._volumes
        else:
            msg = "\n Caution: volumes were not yet set.\n"
            logger.warning(msg)

    @property
    def energies(self):
        if self._energies is not None:
            return self._energies
        else:
            msg = "\n Caution: energies were not yet set.\n"
            logger.warning(msg)
    
    @property
    def dim(self):
        return self._dim
    
    def with_different_volumes(
        self, 
        initial_strain_range=[-0.03, 0.05], nstrains=15, tol_strain=None, 
        kpts=[2,2,2],
        command={'mpirun': 'mpirun', 'nprocs': 1, 'vasp': 'vasp'},
        encut_factor=1.3, 
        verbose=1,
        vasp_params=None,
        potcar_setups=None,
        xc='pbesol',
        ):
        """ Run VASP jobs for different volumes and calculate the energy.
        
        Args
        ----
        strain_range : list of float
            The range of strain to explore (min, max)
        nstrains : int
            The number of strain points to calculate
        max_strain : float
            The maximum strain from the minimum energy point
        kpts : list of int
            The k-point grid for VASP calculations
        command : dict
            The command to run VASP
        encut_factor : float
            The factor to scale the plane-wave cutoff energy
        verbose : int
            The verbose level for logging
        vasp_params : dict
            VASP parameters given as a dictionary
        potcar_setups : dict or None
            POTCAR setups to be used in VASP calculations
            e.g., {'base': 'recommended', 'W': '_sv'}
        xc : string
            Exchange-correlation functional to be used in VASP calculations
            e.g., 'pbesol'
        """
        if tol_strain is None:
            tol_strain = (initial_strain_range[1] - initial_strain_range[0]) / nstrains / 2.
        
        ### Run VASP jobs for different volumes
        line = " Strict structure optimization"
        msg = "\n" + line
        msg += "\n " + "=" * len(line)
        if verbose > 0:
            logger.info(msg)
        
        ## volume of the initial structure
        init_vol = get_volume(self.initial_structure, dim=self.dim)
        
        ##
        max_iterations = 3
        flag = False
        count = 0
        while flag == False:
            
            ## Check the number of iterations
            if count == max_iterations:
                msg = "\n Number of iterations reached the limit (%d)." % count
                logger.warning(msg)
                break
            
            if count == 0:
                verbose = 2
                strains = np.linspace(initial_strain_range[0], initial_strain_range[1], nstrains)
            else:
                verbose = 1
            
            df_results = relaxation_with_different_volumes(
                self.initial_structure, strains,
                base_directory=self.outdir,
                tol_strain=tol_strain,
                #
                kpts=kpts, encut_factor=encut_factor,
                command=command,
                vasp_params=vasp_params,
                potcar_setups=potcar_setups,
                xc=xc,
                verbose=verbose,
                dim=self.dim
            )
            df_results = df_results.sort_values(by='strain')
            
            if len(df_results) < nstrains:
                msg = "\n Caution: Not all the strains were calculated."
                logger.warning(msg)
                count += 1
                continue
            
            ## Get the volume minimizing the energy
            if len(df_results) != 0:
                imin = np.argmin(df_results['energy[eV]'].values)
                opt_vol = df_results['volume[A^3]'].values[imin]
            else:
                opt_vol = init_vol
            
            ## Set the next strain list
            min_vol = opt_vol * (1. + initial_strain_range[0])
            max_vol = opt_vol * (1. + initial_strain_range[1])
            s0 = (min_vol - init_vol) / init_vol
            s1 = (max_vol - init_vol) / init_vol
            strains = []
            for new_str in np.linspace(s0, s1, nstrains):
                if np.min(abs(df_results['strain'].values - new_str)) > tol_strain:
                    strains.append(float(new_str))
            
            if len(strains) == 0:
                flag = True
                logger.info("\n Optimal volume was found properly.")
                break
                
            count += 1
        
        ##
        Vs, Es = self.get_calculated_volumes_and_energies()
        self._volumes = Vs
        self._energies = Es
        
        self._fit()
        vol_strains = (Vs - self.optimal_volume) / self.optimal_volume
        min_strain = np.min(vol_strains)
        max_strain = np.max(vol_strains)
        if abs(min_strain) < 0.005 or abs(max_strain) < 0.005:
            msg = "\n Error: The applied strains are too small."
            msg += "\n Please check the result of the volume relaxation."
            logger.error(msg)
            sys.exit()
        
        self.output_structures()
        return Vs, Es
    
    def output_structures(self):
        
        logger.info("")

        outfile = self.outdir + "/POSCAR.init"
        self.initial_structure.to(filename=outfile)
        msg = " Output %s" % outfile.replace(os.getcwd(), ".")
        logger.info(msg)

        outfile = self.outdir + "/POSCAR.opt"
        self.get_optimal_structure().to(filename=outfile)
        msg = " Output %s" % outfile.replace(os.getcwd(), ".")
        logger.info(msg)
    
    def get_calculated_volumes_and_energies(self):
        init_cell = self.initial_structure.lattice.matrix
        df = _get_calculated_results(self.outdir, cell_pristine=init_cell, dim=self.dim)
        df_sort = df.sort_values(by='volume[A^3]')
        self._volumes = df_sort['volume[A^3]'].values
        self._energies = df_sort['energy[eV]'].values
        return self._volumes, self._energies
     
    def _fit(self):
        Vs, Es = self.get_calculated_volumes_and_energies()
        bm = BirchMurnaghan(Vs, Es)
        bm.fit()
        self._optimal_volume = bm.v0
        
    def get_fitting_error(self, type='mae'):
        bm = BirchMurnaghan(self.volumes, self.energies)
        bm.fit()
        Es_fit = bm.func(self.volumes)
        if type == 'mae':
            mae = np.sum(abs(self.energies - Es_fit)) / len(self.energies)
        else:
            logger.error(" Not yet supported.")
            sys.exit()
        return mae
    
    def plot_bm(self, figname='fig_bm.png'):
        """ Plot a result of fitting with Birch-Murnaghan EOS
        """
        Vs, Es = self.get_calculated_volumes_and_energies()
        bm = BirchMurnaghan(Vs, Es)
        bm.fit()
        
        if self.dim == 3:
            thickness = None
            modulus = bm.b0_GPa # GPa
            modulus_unit = "GPa"
        elif self.dim == 2:
            opt_struct = self.get_optimal_structure(format='pmg')
            thickness = get_thickness(opt_struct) # Angstrom
            modulus = bm.b0_GPa * thickness * 0.1 # N/m
            modulus_unit = "N/m"
        modulus_info = {'value': float(modulus), 'unit': modulus_unit, 'thickness': thickness}
        
        plot_bm_result(bm, figname=figname, dim=self.dim, modulus_info=modulus_info)
    
    def plot_physical_properties(self, figname='fig_fitting.png',
                                 fontsize=7, fig_width=2.3, aspect=1.5,
                                 mew=0.5, ms=2.3, color='black'):
        """ Plot physical properties vs volume
        """
        cell0 = self.initial_structure.lattice.matrix
        df = _get_calculated_results(self.outdir, cell_pristine=cell0, dim=self.dim)
        Vs = df['volume[A^3]'].values
        Ps = df['pressure[GPa]'].values
        Fs = df['force_max[eV/A]'].values
        Es = df['energy[eV]'].values
        ys = [Es, Ps, Fs]
        
        ylabels = ['E (eV)', 'P (GPa)', '${\\rm F_{max}}$ (eV/${\\rm \\AA}$)']
        
        vol0 = self.initial_structure.volume
        
        set_matplot(fontsize=fontsize)
        fig = plt.figure(figsize=(fig_width, aspect*fig_width))
        plt.subplots_adjust(hspace=0.03)
        nfig = len(ys)
        xdat = Vs
        for ifig, ydat in enumerate(ys):
            
            ax = plt.subplot(len(ys), 1, ifig+1)
            
            if ifig == nfig - 1:
                xlabel = 'Volume (${\\rm \\AA^3}$)'
            else:
                xlabel = None
                ax.set_xticklabels([])
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabels[ifig])
            
            ax.plot(xdat, ydat, linestyle='None', lw=0.5,
                    marker='o', markersize=ms, mec=color, mfc='none', mew=mew)
            
            if ifig == 1 or ifig == 2:
                ysort = np.sort(ydat)
                ax.axhline(y=0.0, linestyle='-', c='gray', lw=0.3)
                xs_zero = _find_zero_crossings(xdat, ydat)
                # text = "${\\rm V_{opt}}$ [${\\rm \\AA^3}$] = "
                for i, x0 in enumerate(xs_zero):
                    ax.axvline(x=x0, linestyle='-', c='gray', lw=0.3)
                    text = "%.2f" % (x0)
                    ax.text(x0, ysort[1], text, fontsize=5,
                            horizontalalignment="center", verticalalignment="bottom",
                            bbox=dict(facecolor='white', edgecolor='none', alpha=1.0, pad=0.1))
                
                if len(xs_zero) != 1:
                    msg = "\n Warning: The number of zero crossings is not one."
                    msg += "\n Please check the relaxation process."
                    logger.warning(msg)
                
            elif ifig == 0:
                bm = BirchMurnaghan(Vs, Es)
                bm.fit()
                
                ax.axvline(x=bm.v0, linestyle='-', c='gray', lw=0.3)
                
                if self.dim == 3:
                    thickness = None
                    modulus = bm.b0_GPa # GPa
                    modulus_unit = "GPa"
                elif self.dim == 2:
                    # opt_struct = self.get_optimal_structure(format='pmg')
                    # thickness = get_thickness(opt_struct) # Angstrom
                    # modulus = bm.b0_GPa * thickness * 0.1 # N/m
                    # modulus_unit = "N/m"
                    print(" Not yet supported for 2D.")
                    sys.exit()
                
                modulus_info = {'value': float(modulus), 'unit': modulus_unit, 'thickness': thickness}
                plot_bm_result(bm, ax=ax, dim=self.dim, modulus_info=modulus_info, ylabel=ylabels[ifig])
                
                
            set_axis(ax)
        
        fig.savefig(figname, dpi=600, bbox_inches='tight')
        if figname.startswith("/"):
            figname = "./" + os.path.relpath(figname, os.getcwd())
        msg = " Output %s" % figname
        logger.info(msg)
        
    
    def _strain_volume2length(self, s_vol):
        """ Convert volume strain to length strain
        """
        return math.pow((s_vol + 1.), 1 / self.dim) - 1
    
    def get_optimal_structure(self, format='pmg'):
        vinit = get_volume(self.initial_structure, dim=self.dim)
        s_vol = (self.optimal_volume - vinit) / vinit
        s_len = self._strain_volume2length(s_vol)
        struct_opt = get_strained_structure(
                self.initial_structure, s_len, format=format, dim=self.dim)
        
        vol_opt = get_volume(struct_opt, dim=self.dim)
        if abs(vol_opt - self.optimal_volume) > 1e-4:
            msg = "\n Error: There is a discrepancy in the optimal volumes."
            msg += "\n Volume 1: %.4f, volume 2: %.4f" % (self.optimal_volume, vol_opt)
            logger.error(msg)
            sys.exit()
        
        return struct_opt
        
    def print_results(self):
        
        Vs, Es = self.get_calculated_volumes_and_energies()
        bm = BirchMurnaghan(Vs, Es)
        bm.fit()
        
        # vinit= self.initial_structure.volume
        vinit = get_volume(self.initial_structure, dim=self.dim)
        s_vol = (vinit - self.optimal_volume) / self.optimal_volume
        s_len = self._strain_volume2length(s_vol)
        mae = self.get_fitting_error()
        
        msg = "\n"
        msg += " Minimum energy : %8.3f eV\n" % bm.e0
        if self.dim == 3:
            modulus = bm.b0_GPa # GPa
            modulus_unit = "GPa"
        elif self.dim == 2:
            opt_struct = self.get_optimal_structure(format='pmg')
            thickness = get_thickness(opt_struct)
            modulus = bm.b0_GPa * thickness * 0.1 # N/m
            modulus_unit = "N/m"
        msg += " Bulk modulus   : %8.3f %s\n" % (modulus, modulus_unit)
        msg += " Optimal volume : %8.3f A^3\n" % self.optimal_volume
        msg += " Initial volume : %8.3f A^3\n" % vinit
        msg += " Initial strain : %8.3f\n" % (s_len)
        msg += " Error (MAE)    : %8.5f eV" % (mae)
        if self.dim == 2:
            thickness = get_thickness(self.initial_structure)
            msg += "\n"
            msg += " 2D thickness   : %8.3f A" % (thickness)
        logger.info(msg)
        
        out = {}
        out['minimum_energy'] = [float(bm.e0), 'eV']
        out['bulk_modulus'] = [float(modulus), modulus_unit]
        out['optimal_volume'] = [float(self.optimal_volume), 'A^3']
        out['initial_volume'] = [float(vinit), 'A^3']
        out['initial_strain'] = [float(s_len), '-']
        out['mae'] = [float(mae), 'eV']
        if self.dim == 2:
            out['thickness'] = [float(thickness), 'A']
        
        with open(self.outfile_yaml, 'w') as f:
            yaml.dump(out, f)
            msg = "\n Output %s" % self.outfile_yaml.replace(os.getcwd(), ".")
            logger.info(msg)
        
def _get_calculated_results(base_dir, cell_pristine=None, num_max=200, dim=3):
    """ Get all the results which have been already obtained.
    
    Args
    ----
    base_dir : string
        VASP calculations were performed in base_dir+"/*".
    
    cell_pristine : ndarray, shape=(3,3)
        cell size w/o strain
    
    Return
    -------
    all_results : list of dictionary
        each element contains the info in strain.yaml file
    """
    ### get possible directories 
    dirs = []
    for ii in range(num_max):
        dir_each = base_dir + "/%d" % ii
        file_xml = dir_each + "/vasprun.xml"
        if os.path.exists(file_xml):
            dirs.append(dir_each)
    
    ### check each directory
    all_results = []
    for diri in dirs:
        
        ### Make strain.yaml if it doesn't exist and possible.
        file_yaml = diri + "/strain.yaml"
        data = None
        try:
            data = _make_strain_yaml(diri, cell_pristine=cell_pristine, dim=dim, verbose=0)
        except Exception:
            continue
        
        if data is None and os.path.exists(file_yaml):
            with open(file_yaml, 'r') as yml:
                data = yaml.safe_load(yml)
        
        file_xml = diri + "/vasprun.xml"
        data['filename'] = "./" + os.path.relpath(file_xml, base_dir)
        all_results.append(data)
    
    if len(all_results) == 0:
        return None
    else:
        return pd.DataFrame(all_results)

# def _get_external_pressure_from_outcar(file_outcar):
#     """ Get external pressure from OUTCAR file
#     """
#     pressure = None
#     unit = None
#     if os.path.exists(file_outcar):
#         with open(file_outcar, 'r') as f:
#             lines = f.readlines()
#         for line in reversed(lines):
#             if "external pressure" in line:
#                 try:
#                     pressure = float(line.split()[3])
#                     unit = line.split()[4]
#                 except ValueError:
#                     continue
#                 break
#     return pressure, unit

def _make_strain_yaml(directory, cell_pristine=None, dim=3, verbose=1):
    """ Make strain.yaml file """
    
    ### read file 
    file_xml = directory + "/vasprun.xml"
    vasprun = Vasprun(file_xml, parse_potcar_file=False)
    structure = vasprun.final_structure
    atoms = ase.io.read(file_xml, index=-1, format='vasp-xml')
    
    ### get strain
    l1 = np.linalg.norm(structure.lattice.matrix[0])
    l0 = np.linalg.norm(cell_pristine[0])
    strain = (l1 - l0) / l0
    
    ### get parameters
    out_data = {}
    out_data['strain'] = float(strain)
    ene_unit = str(vasprun.final_energy.unit)
    out_data[f'energy[{ene_unit}]'] = float(vasprun.final_energy.real)  # should be in eV
    ## out_data['energy_unit'] = str(vasprun.final_energy.unit)
    ##
    out_data['stress[eV/A^3]'] = (atoms.get_stress()).tolist()  # in eV/A^3
    out_data['pressure[eV/A^3]'] = float(np.mean(atoms.get_stress()[:3]))
    out_data['pressure[GPa]'] = out_data['pressure[eV/A^3]'] * EvToJ * 1e21  # in GPa
    ##
    forces = atoms.get_forces().flatten()
    imax = np.argmax(np.abs(forces))
    out_data['force_max[eV/A]'] = float(forces[imax])
    ##
    out_data['volume[A^3]'] = float(get_volume(structure, dim=dim))
    # out_data['volume_unit'] = 'A^3'
    out_data['dim'] = int(dim)
    if dim == 2:
        thickness = get_thickness(structure)
        out_data['thickness[A]'] = float(thickness)
    
    ### output file
    outfile = f"{directory}/strain.yaml"
    with open(outfile, "w") as f:
        yaml.dump(out_data, f)
        if verbose > 0:
            msg = "\n Output ./%s" % os.path.relpath(outfile, os.getcwd())
            logger.info(msg)
    return out_data

def get_strained_structure(structure0, strain, format='pmg', dim=3):
    """
    Args
    ----
    structure0 : pymatgen structure obj.
        reference structure
    
    strain : float, unit=[-]
        applied strain
    
    Return
    --------
    structure : 
        strained structure with the assigened format
    """
    if isinstance(structure0, Structure) == False:
        structure0 = change_structure_format(structure0, format='pmg-Structure')
    
    cell0 = structure0.lattice.matrix.copy()
    positions = structure0.cart_coords
    symbols = []
    for el in structure0.species:
        symbols.append(el.name)
    
    ### prepare a strained structure
    if dim == 3:
        s = (1. + np.array(strain)) * np.eye(3)
    elif dim == 2:
        normal_idx = get_normal_index(structure0, base='xyz')
        s = (1. + np.array(strain)) * np.eye(3)
        s[normal_idx, normal_idx] = 1.0
    else:
        logger.error(f" dim == {dim} is not supported.")
        sys.exit()
    
    cell = np.dot(cell0.T, s).T
    positions = np.dot(positions, s.T)
    
    if format.lower().startswith('pmg') or format.lower() == 'pymatgen':
        ## pymatgen
        structure = Structure(
                lattice=cell,
                species=symbols,
                coords=positions,
                coords_are_cartesian=True
                )
    elif format.lower() == 'ase':
        ## ASE
        structure = ase.Atoms(
                cell=cell, pbc=True,
                positions=positions,
                symbols=symbols,
                )
    else:
        logger.error("\n Error: Structure format {} is not supported".format(format))
        sys.exit()
    
    return structure

def get_optimal_volume(volumes, energies, min_num_data=5):
    if len(volumes) >= min_num_data:
        bm = BirchMurnaghan(volumes, energies)
        bm.fit()
        return bm.v0
    else:
        return None

def relaxation_with_different_volumes(
        struct_init, strains,
        base_directory='./volume',
        tol_strain=1e-5,
        kpts=[2,2,2], encut_factor=1.3, 
        command={'mpirun': 'mpirun', 'nprocs': 1, 'vasp': 'vasp'},
        vasp_params=None,
        potcar_setups=None,
        xc='pbesol',
        verbose=1, dim=3,
        ):
    """ Structure relaxation with the Birch-Murnaghan equation of state
    Args
    ----
    strains : array of float, shape=(nstrains), unit=[-]
        strain values to be applied to the initial structure
    
    vasp_params : dict
        VASP parameters given as a dictionary for different modes
        e.g., {'relax': {...}, 'relax-freeze': {...}}
    
    potcar_setups : dict or None
        POTCAR setups to be used in VASP calculations
        e.g., {'base': 'recommended', 'W': '_sv'}
    
    nstrains : integer
        number of strains to be applied
    """
    ### get results in previous calculations
    df_results = _get_calculated_results(
        base_directory, cell_pristine=struct_init.lattice.matrix, dim=dim)
    if df_results is None:
        calculated_strains = []
    else:
        calculated_strains = np.sort(df_results['strain'].values)
    
    ### print already-analyzed-strains
    if len(calculated_strains) > 0:
        line = "Strains already calcuclated (%):"
        msg = "\n %s" % line
        msg += "\n " + "-" * len(line)
        msg += "\n "
        for each in calculated_strains:
            msg += "%.3f, " % (each*100.)
        logger.info(msg)
    
    ### calculate the potential energy for each strain value
    for ist, strain in enumerate(strains):
        
        if len(calculated_strains) > 0:
            if np.min(abs(calculated_strains - strain)) < tol_strain:
                continue
        
        ### print strain value
        line = "Strain: %f" % strain
        msg = "\n " + line
        msg += "\n " + "-" * len(line)
        if verbose > 0:
            logger.info(msg)
        
        ### set the output directory
        flag = False
        outdir = None
        count = 1
        while flag == False:
            outdir = base_directory + "/%d" % int(count)
            if os.path.exists(outdir) == False:
                flag = True
            count += 1
        
        if verbose > 0:
            logger.info("\n Output directory: %s" % (
                outdir.replace(os.getcwd(), ".")))
        
        #### prepare a strained structure
        atoms = get_strained_structure(struct_init, strain, format='ase', dim=dim)
        
        ### set calculator object
        from auto_kappa.utils.config import get_vasp_parameters_by_mode
        _params = get_vasp_parameters_by_mode(vasp_params, mode='relax-freeze')
        
        ## set VASP calculator
        calc = get_vasp_calculator(
                _params,
                directory=outdir,
                atoms=atoms,
                kpts=kpts,
                encut_scale_factor=encut_factor,
                setups=potcar_setups,
                xc=xc,
                )
        
        ## set VASP command
        mpirun = command.get('mpirun', 'mpirun')
        nprocs = command.get('nprocs', 1)
        if list(kpts) == [1, 1, 1]:
            vasp = command.get('vasp_gam')
        else:
            vasp = command.get('vasp', 'vasp')
        calc.command = f"{mpirun} -n {nprocs} {vasp}"
        
        ### run VASP
        run_vasp(calc, atoms, method='custodian')
        
        ### Get each result
        out_data = _make_strain_yaml(outdir, cell_pristine=struct_init.lattice.matrix, dim=dim)
        vol_unit = 'A^3'
        ene_unit = "eV"
        volume = out_data['volume[A^3]']
        energy = out_data[f'energy[{ene_unit}]']
        # ene_unit = out_data['energy_unit']
        
        if verbose > 0:
            msg = (
                f" strain(%): {strain*100:.3f}   "
                f"volume({vol_unit}): {volume:.3f}   "
                f"energy({ene_unit}): {energy:.3f}")
            logger.info(msg)
    
    ###
    init_cell = struct_init.lattice.matrix
    df_results = _get_calculated_results(base_directory, cell_pristine=init_cell, dim=dim)
    
    outfile = base_directory + "/volume_energy.csv"
    df_results.to_csv(outfile, index=False)
    msg = "\n Output %s" % outfile.replace(os.getcwd(), ".")
    logger.info(msg)
    
    return df_results

def get_volume(struct, dim=3):
    
    struct_pmg = change_structure_format(struct, format='pmg-Structure')
    vol = struct_pmg.volume
    cell = struct_pmg.lattice.matrix
    
    if dim == 2:
        norm_idx_abc = get_normal_index(struct_pmg, base='abc')
        # norm_idx_xyz = get_normal_index(struct_pmg, base='xyz')
        thickness = get_thickness(struct_pmg)
        cell_height = np.linalg.norm(cell[norm_idx_abc])
        
        ## Check cell angles
        angles = struct_pmg.lattice.angles
        angles = np.delete(angles, norm_idx_abc)
        if all(abs(angle - 90.0) < 1e-5 for angle in angles) == False:
            msg = "\n Error: The angles of the cell are not 90 degrees."
            msg += "\n Please check the structure."
            logger.error(msg)
            sys.exit()
        
        vol = vol * thickness / cell_height
    
    return vol

def _find_zero_crossings(x, y):
    idx_order = np.argsort(x)
    x = np.array(x)[idx_order]
    y = np.array(y)[idx_order]
    zero_crossings = []
    for i in range(len(y) - 1):
        if y[i] * y[i+1] < 0:
            x0, x1 = x[i], x[i+1]
            y0, y1 = y[i], y[i+1]
            x_zero = x0 - y0 * (x1 - x0) / (y1 - y0)
            zero_crossings.append(x_zero)
    return zero_crossings

#from pymatgen.io.vasp import Poscar
#
#struct_init = Poscar.from_file("POSCAR.init").structure
#
#relax = StrictRelaxation(struct_init)
#Vs, Es = relax.with_different_volumes()
#relax.plot_bm(figname='fig_bm.png')
#relax.print_results()
#
#struct_opt = relax.get_optimal_structure()
#struct_opt.to(filename="POSCAR.opt")
#
