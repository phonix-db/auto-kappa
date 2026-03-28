#
# config.py
#
# This module handles loading and merging user configuration files
# with default parameters.
#
# Copyright (c) 2025 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
import os
import yaml
import json
from copy import deepcopy
from pathlib import Path

import logging
logger = logging.getLogger(__name__)


def load_default_config():
    """Load default configuration from ak_default_config.yaml.
    
    Returns:
        dict: Default configuration dictionary
    """
    # Get path to the default config file in the auto_kappa package
    package_dir = Path(__file__).parent.parent
    default_config_path = package_dir / 'ak_default_config.yaml'
    
    try:
        with open(default_config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config if config is not None else {}
    except Exception as e:
        msg = f"\n Warning: Failed to load default configuration: {e}"
        logger.warning(msg)
        return {}


def deep_update(base_dict, update_dict):
    """Recursively update a dictionary with another dictionary.
    
    Args:
        base_dict: Base dictionary to update
        update_dict: Dictionary with values to update
        
    Returns:
        Updated dictionary (modifies base_dict in place)
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def load_user_config(config_file=None):
    """Load user configuration from YAML or JSON file.
    
    Args:
        config_file: Path to configuration file. If None, searches for
                    'auto_kappa_config.yaml' or 'auto_kappa_config.json'
                    in current directory.
                    
    Returns:
        dict: Configuration dictionary with keys:
              - 'vasp_parameters': VASP parameters to override defaults
              - 'potcar_setups': POTCAR setups to override defaults
              - 'amin_parameters': AMIN parameters to override defaults
    """
    # Return empty dict if no config file found
    if config_file is None or not os.path.exists(config_file):
        return {}
    
    # Load the configuration file
    try:
        with open(config_file, 'r') as f:
            if config_file.endswith('.json'):
                config = json.load(f)
            else:  # Assume YAML
                config = yaml.safe_load(f)
        
        msg = f"\n Loaded user configuration from: {config_file}"
        logger.info(msg)
        
        return config if config is not None else {}
        
    except Exception as e:
        msg = f"\n Warning: Failed to load configuration file {config_file}: {e}"
        logger.warning(msg)
        return {}


def get_vasp_parameters(user_config=None, config_file=None):
    """Get VASP parameters, merging defaults with user configuration.
    
    Args:
        user_config: User configuration dictionary (optional)
        config_file: Path to configuration file (optional)
        
    Returns:
        dict: Merged VASP parameters
    """
    # Load default configuration
    default_config = load_default_config()
    params = deepcopy(default_config.get('vasp_parameters', {}))
    
    # Load user config if provided via file
    if user_config is None and config_file is not None:
        user_config = load_user_config(config_file)
    
    # Merge user configuration
    if user_config and 'vasp_parameters' in user_config:
        deep_update(params, user_config['vasp_parameters'])
        msg = "\n VASP parameters updated from user configuration"
        logger.info(msg)
    
    return params


def get_vasp_parameters_by_mode(all_params, mode=None):
    """Get VASP parameters for a specific mode, merging defaults with user configuration.
    
    Args:
        mode: Calculation mode (e.g., 'relax', 'nac', 'harm', etc.)
        all_params: Dictionary of all VASP parameters
        
    Returns:
        dict: Merged VASP parameters for the specified mode
    """
    # Copy shared params to avoid mutating the original dict
    params = all_params.get('shared', {}).copy()
    if 'relax' in mode.lower():
        params.update(all_params.get('relax', {}).copy())
    params.update(all_params.get(mode.lower(), {}).copy())
    return params


def get_potcar_setups(user_config=None, config_file=None):
    """Get POTCAR setups, merging defaults with user configuration.
    
    Args:
        user_config: User configuration dictionary (optional)
        config_file: Path to configuration file (optional)
        
    Returns:
        dict: Merged POTCAR setups
    """
    # Load default configuration
    default_config = load_default_config()
    setups = deepcopy(default_config.get('potcar_setups', {}))
    
    # Load user config if provided via file
    if user_config is None and config_file is not None:
        user_config = load_user_config(config_file)
    
    # Merge user configuration
    if user_config and 'potcar_setups' in user_config:
        # setups.update(user_config['potcar_setups']) # update
        setups = user_config['potcar_setups'] # replace
        msg = "\n POTCAR setups replaced by user configuration"
        logger.info(msg)
    
    return setups

def get_xc(user_config=None, config_file=None):
    """Get exchange-correlation functional, merging defaults with user configuration.
    
    Args:
        user_config: User configuration dictionary (optional)
        config_file: Path to configuration file (optional)
        
    Returns:
        str: Exchange-correlation functional
    """
    # Load default configuration
    default_config = load_default_config()
    xc = default_config.get('xc', None)
    
    # Load user config if provided via file
    if user_config is None and config_file is not None:
        user_config = load_user_config(config_file)
    
    # Override with user configuration
    if user_config and 'xc' in user_config:
        xc = user_config['xc']
        msg = "\n Exchange-correlation functional replaced by user configuration"
        logger.info(msg)
    
    return xc


# def get_amin_parameters(user_config=None, config_file=None):
#     """Get AMIN parameters, merging defaults with user configuration.
    
#     Args:
#         user_config: User configuration dictionary (optional)
#         config_file: Path to configuration file (optional)
        
#     Returns:
#         dict: Merged AMIN parameters
#     """
#     # Load default configuration
#     default_config = load_default_config()
#     params = deepcopy(default_config.get('amin_parameters', {}))
    
#     # Load user config if provided via file
#     if user_config is None and config_file is not None:
#         user_config = load_user_config(config_file)
    
#     # Merge user configuration
#     if user_config and 'amin_parameters' in user_config:
#         params.update(user_config['amin_parameters'])
#         msg = "\n AMIN parameters updated from user configuration"
#         logger.info(msg)
    
#     return params


# def get_all_parameters(config_file=None):
#     """Get all parameters (VASP, POTCAR, AMIN) from configuration.
    
#     Args:
#         config_file: Path to configuration file (optional)
        
#     Returns:
#         dict: Dictionary with keys 'vasp_parameters', 'potcar_setups', 'amin_parameters'
#     """
#     user_config = load_user_config(config_file)
    
#     return {
#         'vasp_parameters': get_vasp_parameters(user_config),
#         'potcar_setups': get_potcar_setups(user_config),
#         'amin_parameters': get_amin_parameters(user_config),
#     }
