from auto_kappa.version import __version__

# Load default configuration from YAML file
from auto_kappa.cui.ak_log import set_logging
from auto_kappa.utils.config import load_default_config

_default_config = load_default_config()

default_vasp_parameters = _default_config.get('vasp_parameters', {})
default_potcar_setups = _default_config.get('potcar_setups', {})


output_directories = {
    'preparation': {
        'klength': 'preparation/klength',
        },
    'relax': 'relax',
    'nac': 'nac',
    ### Harmonic FCs with finite displacement method
    'harm':{
        'suggest': 'harm/suggest',
        'force':   'harm/force',     ## FC2 with finite displacement
        'bandos':  'harm/bandos',
        'evec' :   'harm/evec',
        'pes'  :   'harm/pes',
        },
    ### cubic FCs
    'cube':{
        'suggest':  'cube/suggest',
        'force_fd': 'cube/force_fd',         ## FC3 with finite displacement
        'kappa_fd': 'cube/kappa_fd',
        'force_lasso': 'cube/force_lasso',   ## FC3 with random displacement
        'cv'   :       'cube/lasso',
        'lasso':       'cube/lasso',
        'kappa_lasso': 'cube/kappa_lasso',
        'gruneisen': 'cube/gruneisen',
        },
    ### high-order FCs using LASSO and fixed harmonic and cubic FCs
    'higher':{
        'suggest': 'higher/suggest',
        'force'  : 'higher/force',
        'cv'     : 'higher/lasso',
        'lasso'  : 'higher/lasso',
        'scph'   : 'higher/scph',
        'kappa_scph'    : 'higher/kappa_scph',
        'kappa_4ph'     : 'higher/kappa_4ph',
        'kappa_scph_4ph': 'higher/kappa_scph_4ph',
        },
    'result': 'result'
    }

output_files = {
        'harm_dfset' : 'DFSET.harm',
        'harm_xml'   : "FC2.xml",
        #
        'cube_fd_dfset'   : 'DFSET.cube_fd',
        'cube_fd_xml'     : "FC3_fd.xml",
        'cube_lasso_dfset': 'DFSET.cube_lasso',
        'cube_lasso_xml'  : "FC3_lasso.xml",
        #
        'higher_dfset': 'DFSET.high_lasso',
        'higher_xml'  : "FCs_high.xml",
        ##'lasso_dfset': 'DFSET.lasso',
        ##'lasso_xml'  : "FCs_lasso.xml",
        }

default_amin_parameters = {"value": 0.01, "tol_length": 50, "num_of_errors": 1}
