#
# ak_parser.py
#
# Parser of akrun command
#
# Copyright (c) 2022 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
import argparse

def get_parser():
    
    parser = argparse.ArgumentParser(
        description="Parser for akrun command",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    #### parameters that need to be modified for each calculation
    
    ## Input and output directories
    parser.add_argument("-d", "--directory", dest="directory", type=str, default=None,
                      help=
                      "[This option will be deprecated in future versions.]\n"
                      "Name of the directory for the PhononDB dataset, containing files \n"
                      "such as POSCAR-unitcell, phonopy.conf, KPOINTS-**, etc. are located.\n"
                      "``--directory`` or ``--file_structure`` must be given. \n"
                      "When both of them are given, ``--directory`` takes priority \n"
                      "over ``--file_structure``.\n")
    
    parser.add_argument("--file_structure", dest="file_structure", type=str, default=None, 
                      help=
                      "Name of the structure file:\n"
                      "Different kinds of format including POSCAR and CIF file can be read \n"
                      "using ``Structure.from_file`` module in ``Pymatgen.core.structure``.")
    
    parser.add_argument("--outdir", dest="outdir", type=str, default="./out", 
                        help="Output directory name [./out]")
    
    parser.add_argument("--config", dest="config_file", type=str, default=None,
                        help=
                        "Path to configuration file (YAML or JSON) for custom parameters.\n"
                        "Configuration file can override default VASP parameters and POTCAR setups. \n\n\n")
    
    ### Parameters that need to be modified depending on the environment
    parser.add_argument("-n", "--nprocs", dest="nprocs", type=int, default=2, 
                        help="Number of processes for the calculation [2]")
    
    parser.add_argument("--anphon_para", dest="anphon_para", type=str, default="mpi", 
                        help=argparse.SUPPRESS)
                        # "[This option may not be supported in future]\n"
                        # "Parallel mode for \"anphon\": \n"
                        # "Input value should be \"mpi\" for MPI and \"omp\" for OpenMP [mpi]. \n"
                        # "While \"mpi\" (default) is faster in most cases, \"omp\" is recommended \n"
                        # "for complex systems requiring large memory.")

    parser.add_argument("--mpirun", dest="mpirun", type=str,
                        default="mpirun", help="MPI command [mpirun]\n\n\n")
    
    ### Commands for ALAMODE and VASP
    parser.add_argument("--command_vasp", dest="command_vasp", type=str, default="vasp", 
                        help="Command to run vasp [vasp]")
    parser.add_argument("--command_vasp_gam", dest="command_vasp_gam", type=str, default="vasp_gam", 
                        help="Command to run vasp_gam [vasp_gam]")
    parser.add_argument("--command_alm", dest="command_alm", type=str, default="alm", 
                        help="Command to run alm [alm]")
    parser.add_argument("--command_anphon", dest="command_anphon", type=str, default="anphon", 
                        help="Command to run anphon [anphon]")
    parser.add_argument("--command_anphon_ver2", dest="command_anphon_ver2", type=str, default="anphon.2.0",
                        help="Command to run anphon for 4-phonon scattering [anphon.2.0]\n\n\n")
    
    ### Parameters for the calculation condictions
    parser.add_argument("--nonanalytic", dest="nonanalytic", type=int, default=2, 
                        help=
                        "NONANALYTIC tag for Anphon calculation [2]:\n"
                        "While the default value is 2, the value is adjusted \n"
                        "when imaginary frequencies are present.")
    
    parser.add_argument("--cutoff_cubic", dest="cutoff_cubic", type=float, default=4.3, 
                        help=
                        "Cutoff distance for cubic force constants (Å) [4.3]:\n"
                        "If the provided value is too small, the cutoff distance will be adjusted \n"
                        "using ``min_nearest`` option.")
    parser.add_argument("--min_nearest", dest="min_nearest", type=int, default=3,
                        help="Minimum nearest neighbor atoms considered for cubic FCs [3]")
    
    parser.add_argument("--nmax_suggest", dest="nmax_suggest", type=int, default=1, 
                        help="Threshold of suggested patterns (``N_{suggest}``) for the finite-displacement method [1]:\n"
                        "If ``N_{suggest}`` exceeds ``nmax_suggest``, LASSO regression is applied \n"
                        "for computing cubic force constants.\n"
                        "The default value is set to 1. \n"
                        "To use the finite-displacement method, set this option to a large value such as 10000.\n\n\n")
    
    parser.add_argument("--frac_nrandom", dest="frac_nrandom", type=float, default=1.0,
                        help=
                        "``{frac_nrandom} = Npattern * Natom / Nfc3``:\n"
                        "- Npattern: number of generated random displacement patterns\n"
                        "- Natom   : number of atoms in a supercell\n"
                        "- Nfc3    : number of FC3 [1.0]. \n"
                        "{frac_nrandom} should be larger than 1/3.")
    
    parser.add_argument("--frac_nrandom_higher", 
                        dest="frac_nrandom_higher", type=float, default=0.5, help=
                        "``Npattern * Natom / Nfc4`` [0.5],\n"
                        "- ``Npattern``: number of generated random displacement patterns, \n"
                        "- ``Natom``   : number of atoms in the supercell,\n"
                        "- ``Nfc4``    : number of FC4\n"
                        "See the comment for ``frac_nrandom`` option.\n\n\n")
    
    parser.add_argument("--mag_harm", dest="mag_harm", type=float, default=0.01, 
                        help="Displacement magnitude for harmonic FCs (Å) [0.01]")
    parser.add_argument("--mag_cubic", dest="mag_cubic", type=float, default=0.03, 
                        help="Displacement magnitude for cubic FCs (Å) [0.03]\n\n\n")
    
    parser.add_argument("--negative_freq", dest="negative_freq", type=float, default=-0.001, 
                        help="Threshold for negative frequency (cm^-1) [-0.001]\n\n\n")
            
    parser.add_argument("--volume_relaxation", dest="volume_relaxation", type=int, default=1,
                        help=
                        "Flag for the strict structure relaxation \n"
                        "using the Birch-Murnaghan equation of state (0.off or 1.on) [1]")
    
    parser.add_argument("--relaxed_cell", dest="relaxed_cell", type=str, default=None,
                        help=argparse.SUPPRESS)
                        # "Cell type used for relaxation calculation [None]: \n"
                        # "Use 'primitive'/'p' for a primitive cell, or\n"
                        # "'conventional'/'c'/'unitcell'/'u' for a conventional cell.\n"
                        # "For a restarted calculation, the same type as the previous run is used, \n"
                        # "while a new calculation defaults to the conventional cell.\n\n\n")
    
    ### Parameters to determine k-mesh densities and size of supercells
    parser.add_argument("--k_length", dest="k_length", type=float, default=20, help=
                        "Length to determine k-mesh for VASP calculations [20]. \n"
                        "The following equation is used to determine the k-mesh; \n"
                        "``N = max(1, int(k_length * |a|^* + 0.5))``.\n"
                        "10 for large gap insulators and 100 for d metals are recommended.\n"
                        "See the official documentation at https://www.vasp.at/wiki/index.php/KPOINTS."
                        )
    
    parser.add_argument("--optimize_klength", dest="optimize_klength", type=int, default=0, 
                        help="Flag to optimize k-length for VASP calculations (0.off or 1.on) [0]\n"
                        "This options prioritizes ``k_length`` option if both are given."
                        )
    parser.add_argument("--energy_tolerance", dest="energy_tolerance", type=float, default=2e-3, 
                        help="Energy tolerance for optimizing k-length (eV/atom) [0.002]\n"
                        "This option is used only when ``--optimize_klength = 1``."
                        )
    
    ### options for supercell
    parser.add_argument("--max_natoms", dest="max_natoms", type=int, default=150, help=
                        "Initial maximum number of atoms in supercells [150]: \n"
                        "If imaginary frequencies are found, the maximum limit for harmonic FCs \n"
                        "is increased in steps of ``delta_max_natoms``. \n"
                        "The supercell size for cubic FCs is not changed during the simulation.")
    
    # parser.add_argument("--supercell_matrix", dest="supercell_matrix", 
    #                     nargs=3, type=int, default=None, help=
    #                     "[This option is currently in testing.] \n"
    #                     "Supercell matrix size with respect to the unit cell [None]:\n"
    #                     "Three integers should be given, e.g., \"2 2 2\".\n"
    #                     "If this option is provided, 'max_natoms' option is ignored.\n\n\n")
    
    ### three options for larger supercell
    parser.add_argument("--analyze_with_larger_supercell", dest="analyze_with_largersc", type=int, default=0, help=
                        "Flag for analyzing harmonic properties with larger supercells \n"
                        "if negative frequencies are present. (0.no, 1.yes) [0]")
    
    parser.add_argument("--delta_max_natoms", dest="delta_max_natoms", type=int, default=50, help=
                        "Increasing interval of the maximum number of atoms in the supercell \n"
                        "for calculating FC2 [50]. \n"
                        "This option is used only when ``--analyze_with_largersc = 1``.")
    
    parser.add_argument("--max_loop_for_largesc", dest="max_loop_for_largesc", type=int, default=1, help=
                        "Maximum number of loops for increasing the supercell size [1].\n"
                        "This option is used only when ``--analyze_with_largersc = 1``.\n\n\n")
    
    ### parameters for NSW
    parser.add_argument("--nsw_params", dest="nsw_params", type=str, default="200:20:20", help=
                        "Parameters which determine NSW for relaxation calculations [200:20:20]. \n"
                        "\"{nsw_init}:{nsw_diff}:{nsw_min}\": NSW = min(``nsw_min``, \n"
                        "``nsw_init`` - ``nsw_diff`` * ``num_errors``), where \n"
                        "``num_errors`` is the number of errors.\n\n\n")
    
    ##### parameters for amin
    ##parser.add_argument("--amin", dest="amin", 
    ##        type=float, default=None, 
    ##        help=("AMIN parameter for VASP [None]: If the length of a lattice "
    ##            "vector exceeds 5 nm, AMIN of the given value is set for the "
    ##            "VASP job.")
    ##        )
    
    parser.add_argument("--calculate_forces", dest="calculate_forces", type=int, default=1, help=
                        "For calculating forces (0: off, 1: on) [1].\n"
                        "If set to 1, forces are calculated.\n"
                        "If set to 0, forces are not calculated but displacement structures are generated.")
    
    parser.add_argument("--harmonic_only", dest="harmonic_only", type=int, default=0, 
                        help="Calculate harmonic properties only (0.No, 1.Yes) [0]\n\n\n")
    
    ##############################################
    ### parameters for high-order (>= 4th) FCs ###
    ##############################################
    ### on/off SCPH
    parser.add_argument("--command_dfc2", dest="command_dfc2", type=str, default="dfc2", 
                        help="Command to run 'dfc2' implemented in ALAMODE package. [dfc2]")
    
    parser.add_argument("--scph", dest="scph", type=int, default=0, help=
                        "Flag for self-consistent phonon (SCPH) calculation. (0: off, 1-2: on) [0]\n"
                        "  1: Run SCPH. If negative frequencies are resolved by SCPH, stop\n"
                        "     without proceeding to larger-supercell analysis.\n"
                        "  2: Run SCPH. If --analyze_with_largersc is also 1, always proceed\n"
                        "     to SCPH+larger-supercell analysis regardless of SCPH result.\n")
    
    ###### 4-phonon scattering
    parser.add_argument("--four", dest="four", type=int, default=0, help=
                        "Flag for considering four-phonon scattering (0.off, 1.on) [0]. \n"
                        "If 'scph' option is also set to 1, SCPH+4ph is performed. \n"
                        "--command_anphon_ver2 must be set properly.")
    parser.add_argument("--frac_kdensity_4ph", dest="frac_kdensity_4ph", type=float, default=0.13, 
                        help=
                        "Fractional k-point density for four-phonon scattering relative to the \n"
                        "k-point density used in three-phonon scattering calculations [0.13].\n\n\n")
    
    ### temperature for random displacements
    parser.add_argument("--random_disp_temperature", dest="random_disp_temperature", type=float, default=500., 
                        help="Temperature for the random displacement method in high-order FC calculations [500]")
    
    ##### displacement magnitude for high-order FCs
    ##parser.add_argument("--mag_high", 
    ##        dest="mag_high", type=float, default=0.03, 
    ##        help="magnitude of displacements for cubic FCs [0.03]")
    
    ##parser.add_argument("--max_natoms3", dest="max_natoms3", type=int, 
    ##        default=None, 
    ##        help="This options is invalid! PLEASE DO NOT USE this option."\
    ##                "Maximum limit of the number of atoms in the supercell for "\
    ##                " FC3 [None].")
    
    #########################
    ## parameters for VASP ##
    #########################
    # VASP parameters are now configured via --config file
    # See ak_default_config.yaml for available options
    parser.add_argument("--vasp_parameters", dest="vasp_parameters", type=str, default=None, 
                        help=argparse.SUPPRESS)
                        # "VASP parameters for the INCAR file, separated by commas: \n"
                        # "e.g., \"ISORBIT=False,DIFFG=1e-7\"")
    
    #########################################
    ### Parameters for test calculations  ###
    #########################################
    parser.add_argument("--restart", dest="restart", type=int, default=1, 
                        help=argparse.SUPPRESS)
                        # "The calculation will restart (1) or will NOT restart (0) \n"
                        # "when the directory exsits. [1]")
            
    parser.add_argument("--verbose", dest="verbose", type=int, default=1, 
                        help=argparse.SUPPRESS)
                        # help="Verbose [0]")
    
    parser.add_argument("--ignore_log", dest="ignore_log", type=int, default=0, 
                        help=argparse.SUPPRESS)
                        # help="Ignore the job log such as \"band.log\"... (0: No, 1: Yes) [0]")
    
    parser.add_argument("--max_relax_error", dest="max_relax_error", type=int, default=500, 
                        help=argparse.SUPPRESS)
                        # help=
                        # "Maximum number of errors for relaxation calculations with VASP.\n"
                        # "Set this option if the number of error is too many. [500]")
    
    ######################
    ### Future options ###
    
    ## for 2D, ISIF=4 for "full-relax"??
    ## Carefully determine the cell size along c-axis
    parser.add_argument("--material_dimension", dest="mater_dim", type=int, default=3, 
                        help=argparse.SUPPRESS)
    # help="[Not available yet] Material dimension [3]")
    
    #### calculate potential energy surface (This option will be available in future versions)
    parser.add_argument("--pes", dest="pes", type=int, default=0, 
                        help=argparse.SUPPRESS)
    # help="[Not available yet] "
    # "Flag for calculating potential energy surface (PES) "
    # "for phonon mode with negative frequency. [0] "
    # "0: not calculated, "
    # "1: calculated only for larger supercells, or "
    # "2: calculated for both of small and larger supercells. "
    # "A representative k-point is chosen for the PES calculation: "
    # "a Gamma point for DOS.")
    
    return parser

