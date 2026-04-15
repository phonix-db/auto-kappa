import pathlib
from setuptools import setup, find_packages

def _get_version():
    
    filename = "./auto_kappa/version.py"
    
    try:
        lines = open(filename, 'r').readlines()
        for line in lines:
            if '__version__' in line:
                line = line.translate(str.maketrans(
                    {"\"": " ", "\'": " ", "=": " "}
                    ))
                break
        data = line.split()
        return data[-1]
    
    except Exception:
        return "x.x"

def main(build_dir):
    
    scripts_autokappa = [
            'scripts/akrun',
            #'scripts/ak-logger',
            #'scripts/ak-plotter',
            ]
    
    version = _get_version()
    
    setup(
            name='auto_kappa',
            version=version,
            description='automation software for anharmonic phonon properties',
            author='Masato Ohnishi',
            author_email='masato.ohnishi.ac@gmail.com',
            packages=find_packages(),
            include_package_data=True,
            install_requires=[
              'setuptools', 'numpy', 'phonopy', 'spglib', 'seekpath', 'ase', 'pymatgen', 
              'custodian', 'xmltodict', 'mkl', 'f90nml', 'PyYAML',
              'psutil', 'scikit-learn', 'lxml'
              ],
            scripts=scripts_autokappa,
            url='https://github.com/phonix-db/auto_kappa.git',
            license='MIT',
            provides=['auto_kappa'],
            )

if __name__ == "__main__":
    
    build_dir = pathlib.Path.cwd() / "_build"
    
    main(build_dir)

