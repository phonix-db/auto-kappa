# 
# utils.py
# 
# This script contains utility functions
# 
# Created on June 12, 2025
# Copyright (c) 2025 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
# 
import subprocess

import logging
logger = logging.getLogger(__name__)

def get_output(command='anphon'):
    """ Get the outptu of a command.
    
    Args:
    command (str): The command to run to get the version information.
    
    Returns:
    list: A list of lines from the command output.
    """
    try:
        process = subprocess.Popen(
            command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, bufsize=1)
        
        lines = []
        for line in iter(process.stdout.readline, ''):
            lines.append(line.strip())
        return lines
    except Exception:
        lines = None
    return lines

def get_version(command):
    """ Get the version of a command.
    Args:
    command (str): The command to check the version of ('anphon' or 'vasp').
    
    Returns:
    str: The version of the command, or None if not found.
    """
    if 'alm' in command.lower():
        logger.info("\n Version of 'alm' cannot be obtained in this function.")
        return None
    elif 'anphon' in command.lower():
        lines = get_output(command)
        ver = None
        try:
            for line in lines:
                if 'ver.' in line.lower():
                    data = line.split()
                    for i, each in enumerate(data):
                        if 'ver.' in each.lower():
                            ver = data[i+1]
                            break
            return ver
        except Exception:
            logger.error("\n Failed to get the version of 'anphon'.")
            return None
    elif 'vasp' in command.lower():
        lines = get_output(f"{command} --version")
        if lines is None:
            return None
        ver = None
        for line in lines:
            if 'vasp.' in line.lower():
                ver = line.split()[0].lower().replace("vasp.", "")
                break
        return ver
    else:
        logger.info("\n Unsupported command. Use 'alm*', 'anphon*', or 'vasp*'.")
        return None

# ver = get_version('alm')
# ver = get_version('anphon')
# print(ver)
# ver = get_version('vasp')
# print(ver)
