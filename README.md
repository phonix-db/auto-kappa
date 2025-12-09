<!-- Auto-kappa
============ -->

<p align='left'>
  <a href="https://masato1122.github.io/auto-kappa/" target="_blank">
    <img src="./docs/source/img/ak_logo.png" alt="logo" width="250"/>
  </a>
</p>

Auto-kappa: ver1.0.0
---------------------

Auto-kappa is an automation framework for first-principles calculations of anharmonic phonon properties
—including thermal conductivity and mode-dependent phonon lifetimes—based on VASP and ALAMODE.

Requirements
-------------

Users of auto-kappa need to install VASP and ALAMODE in advance, 
while the required Python libraries are installed automatically along with auto-kappa.

* VASP : 6.3.0 or later
* ALAMODE : 1.4 or 1.5 (1.5 recommended) *
* [Optional] anphon : 1.9.9 (required for four-phonon scattering)
* Python : 3.9 or later
* Phonopy
* ASE
* Pymatgen
* Spglib
* Custodian

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


Citation
---------

If you use auto-kappa, please cite the following paper, along with any related papers listed in the references:

- M. Ohnishi et al., "Database and deep-learning scalability of anharmonic phonon properties by automated brute-force first-principles calculations", 
[arXiv:2504.21245](https://arxiv.org/abs/2504.21245) (2025).

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


<!-- To Do
------

- Iterative calculation

- Cell size for 2D systems: fix cell size for VASP calculations -->

