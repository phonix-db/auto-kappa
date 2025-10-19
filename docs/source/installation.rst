==============
Installation
==============

Requirements
=============

Auto-kappa requires the following packages.
Python packages are installed automatically during setup, but VASP and ALAMODE must be installed separately.

* VASP (≥ ver.6.3)
* ALAMODE (alm and anphon commands: ver.1.4-1.5)
* Phonopy
* Pymatgen
* ASE
* Custodian
* Spglib
* SeeK-path

Preparation
============

For phonon calculations using auto-kappa, the ``alm``, ``anphon``, and ``vasp`` commands, 
as well as the ``VASP_PP_PATH`` environment variable, are required. 

VASP and ALAMODE
-----------------

Auto-kappa requires ``vasp``, ``alm``, and ``anphon`` commands.
Please install VASP and `ALAMODE <https://alamode.readthedocs.io/en/latest/index.html>`_ in advance.

ALAMODE requires 
`Eigen <https://eigen.tuxfamily.org/index.php?title=Main_Page>`_,
`BOOST <https://www.boost.org/releases/latest/>`_, and
`Spglib <https://spglib.readthedocs.io/en/stable/>`_.
These packages must be installed before installing ALAMODE.
Note that ALAMODE may not be compatible with the latest versions of these packages.
You may need to install older versions to ensure compatibility.


VASP_PP_PATH variable
-----------------------

To enable ASE to access the pseudopotential (POTCAR) files of VASP,
set the ``VASP_PP_PATH`` variable in your shell configuration file, such as ``~/.bash_profile``.
See `the ASE documentation <https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html>`_ for details.

.. code-block:: bash
    
    export VASP_PP_PATH=$HOME/(directory containing ``potpaw_PBE``)
    
``potpaw_PBE.54.tar.gz`` or ``potpaw_PBE.64.tar.gz`` supposed to be expanded in ``potpaw_PBE``.


PhononDB (optional)
--------------------

While auto-kappa can perform phonon calculations solely from a structure file (e.g., POSCAR, CIF),
it can also utilize parameters from PhononDB.
To use the parameters from PhononDB, download the data from `HERE <https://github.com/WMD-group/phononDB>`_.


Installation
=============

**1. Create a virtual Python environment**

The following example creates a virtual environment named ``kappa`` in the ``.venv`` directory using ``python -m venv``.
Alternatively, you may use ``conda`` or other tools to create a virtual environment, depending on your preference.

.. code-block:: bash

    (mkdir ~/.venv)
    cd ~/.venv
    python -m venv kappa
    source ./kappa/bin/activate
    
To set ``kappa`` as the default, add the following line in ``.bash_profile``.

.. code-block:: bash

    source ~/.venv/kappa/bin/activate

**2. Download the code**

.. code-block:: bash
    
    git clone git@github.com:masato1122/auto-kappa.git

Update and install the code:

.. code-block:: bash

    cd ./auto-kappa
    git pull
    sh ./install.sh

After completing the process, verify the installation:

.. code-block:: bash

    akrun -h



.. Automation Calculation
.. =======================

.. Scripts in ``examples/phonondb`` and ``examples/massive`` may be useful to run the automation calculation.
.. First, data of Phonondb need to be downloaded

.. 1. Download data from Phonondb

.. .. code-block:: shell
    
..     $ cd (arbitrary directory in which Phonondb will be downloaded.)
..     $ cp .../examples/phonondb/* ./
    
..     ## modify "imin" and "imax" in get_phonondb.sh
..     $ vi get_phonondb.sh
..     $ sh get_phonondb.sh


.. 2. Start the calculation

.. .. code-block:: shell
    
..     $ dir="APDB_0-10000"
..     $ mkdir $dir
..     $ cd $dir
..     $ cp .../auto-kappa/examples/massive/run_massive.sh ./
..     ## modify the script and submit jobs


.. Installation of python libraries
.. ---------------------------------
.. 
.. .. code-block:: bash
.. 
..     $ conda create -n alm python=3.8
..     $ conda activate alm
..     $ pip install pymatgen 
..     $ conda install -c conda-forge phonopy
..     $ pip install ase
..     $ pip install seekpath
..     $ pip install custodian
..     $ conda install -c conda-forge eigen
..     $ conda install -c conda-forge gcc
..     $ pip install xmltodict
..     $ pip install f90nml
..     $
..     $ conda install -c conda-forge mkl
..     $
..     $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CONDA_PREFIX}/lib
.. 
.. 
.. Installation of Eigen
.. ^^^^^^^^^^^^^^^^^^^^^^^
.. 
.. .. code-block:: bash
..     
..     $ cd .../eigen-3.4.0
..     $ mkdir build
..     $ cd ./build
..     $ cmake3 ..
..     $ cmake3 . -DCMAKE_INSTALL_PREFIX=/home/*****/usr/local
..     $ make install
.. 
.. * Check /home/*****/usr/local/include/eigen3


.. Setting for POTCAR with ASE
.. -----------------------------
.. 
.. Add the following line. In the directory, potpaw_PBE exists.
.. See the following pages for details:
.. `1 (ASE) <https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html>`_ and
.. `2 (pymatgen <https://pymatgen.org/installation.html#potcar-setup>`_.
.. 
.. .. code-block:: bash
..     
..     $ cat ~/.bash_profile
..     
..     ...
..     export VASP_PP_PATH=(directory in which potpaw_PBE is located.)
..     ...
.. 
.. .. code-block:: bash
..     
..     $ cat .pmgrc.yaml
..     
..     ...
..     PMG_VASP_PSP_DIR: (directory in which potpaw_PBE is located.)
..     PMG_MAPI_KEY: **********
..     ...

.. Installation of ALM
.. ----------------------
.. 
.. .. code-block:: bash
..     
..     $ source activate alm
..     $ git clone https://github.com/ttadano/ALM.git
..     $ cd ./ALM
..     $ git pull
..     $ cd ./python
..     $ python setup.py install
.. 
.. .. For Grand-Chariot, the following line may need to be added in setup.py.
.. .. 
.. .. .. code-block:: bash
.. .. 
.. ..     os.environ["CC"] = /usr/bin/gcc
.. 
