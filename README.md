
<p align='left'>
  <a href="https://masato1122.github.io/auto-kappa/" target="_blank">
    <img src="./docs/source/img/ak_logo.png" alt="logo" width="250"/>
  </a>
</p>


<<<<<<< HEAD
# Auto-kappa: v1.0.1

**Auto-kappa** is an automated workflow tool for performing **first-principles calculations of anharmonic phonon properties**, including  
=======
# auto-kappa v1.1.1

**auto-kappa** is an automated framework for performing **first-principles calculations of anharmonic phonon properties**, including  
>>>>>>> develop
- **lattice thermal conductivity**,  
- **mode-dependent phonon lifetimes**,  
- **three-phonon and four-phonon scattering**,  

using **VASP** and **ALAMODE**.  
It provides a streamlined pipeline that generates input files, submits calculations, checks convergence, and post-processes results automatically.
<<<<<<< HEAD
=======

>>>>>>> develop

Requirements
-------------

Users of auto-kappa need to install VASP and ALAMODE in advance, 
while the required Python libraries are installed automatically along with auto-kappa.

* VASP : 6.3.0 or later
* ALAMODE : 1.4 or 1.5 (1.5 recommended) *
* [Optional] anphon : 1.9.9 (required for four-phonon scattering)
* Python : 3.9 or later
<<<<<<< HEAD
* Phonopy : 2.43.2 or former
* ASE
* Pymatgen
* Spglib
* Custodian
=======
* Phonopy : 2.45.1
* ASE : 3.26.0
* Pymatgen : 2025.10.7
* Spglib : 2.6.0
* Custodian : 2025.5.12
>>>>>>> develop

---

\* Note: The force constant file format in ALAMODE 1.4 is incompatible with that of version 1.5. 
Therfore, version 1.5 is recommended.

Installation
-------------

Follow these steps to install the package:

1. git clone https://github.com/masato1122/auto-kappa.git
2. cd ./auto-kappa
3. sh install.sh

After installation, ensure that the ``akrun`` command is available.
You can view a description of the input parameters by running ``akrun -h``.

Preparation
--------------

You can perform a simple calculation following the steps below. 
Please refer to example jobs in ``auto_kappa/examples`` and the manual for details.

1. Set the ``VASP_PP_PATH`` environment variable so that ASE can locate VASP pseudopotential files:
([Pseudopotential with ASE](https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html#pseudopotentials))

ASE expects the pseudopotential files to be in ``${VASP_PP_PATH}/potpaw_PBE/{element name}``.

2. Prepare a structure file, e.g., ``POSCAR.Si``
3. Run the following command: ``akrun --file_structure POSCAR.Si --outdir Si``.

Several Important Options
---------------------------

You can view the available options by running ``akrun -h`` 
as well as in the [manual](https://masato1122.github.io/auto-kappa/params_ak.html).
Frequently used commands are listed below.

- **file_structure**: Structure file name. Different formats, including POSCAR and CIF, are accepted.

- **outdir**: Name of the output directory

- **mpirun**: MPI command [Default: mpirun]

- **nprocs**: Number of processes for the calculation [Default: 2]

- **command\_{vasp/vasp\_gam/alm/anphon/anphon_ver2}**: Command to run ``VASP``, ``alm``, and ``anphon`` [Default: vasp, vasp_gam, alm, anphon, anphon.2.0]

- **volume\_relaxation**: Perform relaxation calculations using the Birch-Murnaghan equation of state [Default: 1]

- **analyze\_with\_larger\_supercell**: Use a larger supercell when imaginary frequencies appear [Default: 0]

- **max\_natoms**: Maximum number of atoms in the supercell used for the force constant calculation [Default: 150]

- **nmax\_suggest**: Maximum number of displacement patterns for the finite-displacement method. If the number of generated patterns exceeds this value, the LASSO regression approach will be applied [Default: 100].

- **scph**: Flag for considering phonon renormalization using the self-consistent phonon (SCPH) approach [Default: 0]

- **four**: Flag for considering four phonon scattering. The "command_anphon_ver2" option must be set properly. [Default: 0]

<!-- - **material_dimension**: Dimension of the material (2 or 3) [Default: 3] -->

Documentation
-------------

For more details on auto-kappa, please visit the following webpage: [**HERE**](https://masato1122.github.io/auto-kappa).

Workflow
---------

| scph | four | No imag. freq.<br>with initial SC | No imag. freq.<br>after SCPH | Use larger SC | FC2 [kappa(init SC)] | FC2 [kappa(larger SC)] | Anharmonic FCs | kappa(init SC) | kappa(larger SC) |
|------|------|------------------------------------|------------------------------|---------------|----------------------|------------------------|----------------|----------------|------------------|
| 0 | 0 | TRUE | - | × | init SC | - | FC3(init SC) | 3ph | - |
| 0 | 1 | TRUE | - | × | init SC | - | Higher(init SC) | 4ph | - |
| 1, 2 | 0 | TRUE | - | × | renorm(init SC) | - | Higher(init SC) | SCPH+3ph | - |
| 1, 2 | 1 | TRUE | - | × | renorm(init SC) | - | Higher(init SC) | SCPH+4ph | - |
| 0 | 0 | FALSE | - | ○ | - | larger SC | FC3(init SC) | - | 3ph |
| 0 | 1 | FALSE | - | ○ | - | larger SC | Higher(init SC) | - | 4ph |
| 1 | 0 | FALSE | TRUE | × | renorm(init SC) | - | Higher(init SC) | SCPH | - |
| 1 | 1 | FALSE | TRUE | × | renorm(init SC) | - | Higher(init SC) | SCPH+4ph | - |
| 1, 2 | 0 | FALSE | FALSE | ○ | - | renorm(larger SC) | Higher(init SC) | - | SCPH |
| 1, 2 | 1 | FALSE | FALSE | ○ | - | renorm(larger SC) | Higher(init SC) | - | SCPH+4ph |
| 2 | 0 | FALSE | TRUE | ○ | renorm(init SC) | renorm(larger SC) | Higher(init SC) | SCPH | SCPH |
| 2 | 1 | FALSE | TRUE | ○ | renorm(init SC) | renorm(larger SC) | Higher(init SC) | SCPH+4ph | SCPH+4ph |

- init SC: Initial supercell determined by --max_natoms.
- larger SC: Larger supercell determined by parameters such as --delta_max_natoms.
- kappa(init/larger SC): Thermal conductivity calculated using the initial/larger supercell.
- FC2 [kappa(init/larger SC)]: Harmonic force constants (FC2) used to compute kappa(init/larger SC).
- renorm(init/larger SC): Renormalized harmonic force constants (FC2) derived from the FC2 calculated with the initial/larger supercell.
- FC3/Higher(init SC): Cubic and higher-order force constants, always computed using the initial supercell.
- SCPH: Self-consistent phonon (SCPH) calculation.

Citation
---------

If you use auto-kappa, please cite the following paper, along with any related papers listed in the references:

- M. Ohnishi et al., "Database and deep-learning scalability of anharmonic phonon properties by automated brute-force first-principles calculations," 
npj Computational Materials (2025), [arXiv:2504.21245](https://arxiv.org/abs/2504.21245) (2025).

References
-----------

- [**ALAMODE**](https://alamode.readthedocs.io/en/latest): 
T. Tadano, Y. Gohda, and S. Tsuneyuki, J. Phys.: Condens. Matter 26, 225402 (2014).

- [**ALAMODE (SCP)**](https://alamode.readthedocs.io/en/latest/anphondir/formalism_anphon.html#self-consistent-phonon-scph-calculation): T. Tadano and S. Tsuneyuki, Phys. Rev. B 92, 054301 (2015).

- [**VASP**](https://www.vasp.at/wiki/The_VASP_Manual): 
G. Kresse, and J. Furthmuller, Phys. Rev. B 54, 11169-11186 (1996).

- [**Spglib**](https://spglib.readthedocs.io/en/stable/): A. Togo, K. Shinohara, and I. Tanaka, Sci. technol. adv. material, Meth. 4, 1 (2025).

- [**SeeK-path**](https://seekpath.readthedocs.io/en/latest/index.html): Y. Hinuma, G. Pizzi, Y. Kumagai, F. Oba, and I. Tanaka, Comp. Mat. Sci. 128, 140 (2017).

- [**Phonopy**](https://phonopy.github.io/phonopy/): A. Togo and I. Tanaka, Scr. Mater., 108, 1-5 (2015).

- [**Pymatgen** and **Custodian**](https://pymatgen.org/): S. P. Ong et al., Comp. Mater. Sci. 68, 314-319 (2013).

- [**ASE**](https://ase-lib.org/): A. H. Larsen et al., J. Phys.: Cond. Matter 29, 273002 (2017).


Developpers
-------------

- Tianqie Deng, 
Michimasa Morita, 
Wei Nong, 
Masato Ohnishi, 
Terumasa Tadano, 
Pol Torres, 
Zeyu Wang

(alphabetical order)

