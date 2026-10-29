# 
# kl.py
# 
# This script handles kl, kl3, kl4, and kl_coherent files.
# 
# Created on August 29, 2025
# Copyright (c) 2025 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
# 
import os
import numpy as np
import pandas as pd
# from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

from auto_kappa.plot import set_axis, set_legend

import logging
logger = logging.getLogger(__name__)

class KappaShare:
    
    @property
    def temperatures(self):
        try:
            return self.data['temperature'].values
        except:
            return None
    
    def get_kappa(self, temperature, tol=1):
        """ Get the kappa values at a specific temperature. 
        """
        diff = np.abs(self.temperatures - temperature)
        imin = np.argmin(diff)
        if diff[imin] < tol:
            dump = {}
            for col in self.data:
                dump[col] = self.data[col].values[imin]
            return dump
        else:
            return None

class Kboth(KappaShare):
    def __init__(self, file_kp, file_kc):
        """ Class to handle both kp and kc data obtained using ALAMODE.
        
        How to use
        -----------
        >>> file_kp = "./mp-149/cube/kappa_14x14x14/Si.kl"
        >>> file_kc = "./mp-149/cube/kappa_14x14x14/Si.kl_coherent"
        >>> kboth = Kboth(file_kp, file_kc)
        >>> kboth.get_kappa(300)
        >>> kboth.kp.data
        >>> kboth.kc.data
        >>> kboth.plot(ax)
        
        """
        self.file_kp = file_kp
        self.file_kc = file_kc
        self.kp = KL(file_kp)
        self.kc = KL(file_kc)
        self._set_data()
    
    def _set_data(self):
        
        kp = self.kp
        kc = self.kc
        ts_kp = kp.data['temperature'].values
        ts_kc = kc.data['temperature'].values
        
        ## Concate kp and kc
        df_new = pd.DataFrame()
        dirs = ['xx', 'yy', 'zz', 'ave']
        if len(ts_kp) == len(ts_kc) and np.allclose(ts_kp, ts_kc):
            df_new['temperature'] = kp.data['temperature']
            for j in range(4):
                df_new['kp_' + dirs[j]] = kp.data['k' + dirs[j]]
            for j in range(4):
                df_new['kc_' + dirs[j]] = kc.data['k' + dirs[j]]
        else:
            ts_kp = kp.data['temperature'].values
            ts_kc = kc.data['temperature'].values
            ts = [tp for tp in ts_kp if np.min(abs(tp - ts_kc)) < 1.0]
            df_new = pd.DataFrame()
            df_new['temperature'] = ts
            for j in range(4):
                df_new['kp_' + dirs[j]] = np.interp(ts, kp.data['temperature'], 
                                                    kp.data['k' + dirs[j]])
            for j in range(4):
                df_new['kc_' + dirs[j]] = np.interp(ts, kc.data['temperature'], 
                                                    kc.data['k' + dirs[j]])
        ## klat = kp + kc
        for j in range(4):
            df_new['klat_' + dirs[j]] = df_new['kp_' + dirs[j]] + df_new['kc_' + dirs[j]]
        
        self.data = df_new
    
    def plot(self, ax, process=None, lw=0.4, ms=2.3, color='blue',
             xlabel="Temperature (K)",
             ylabel="Thermal conductivity (${\\rm W m^{-1} K^{-1}}$)"):
        
        dirs = ['xx', 'yy', 'zz', 'ave']
        cmap = plt.get_cmap("tab10")
        
        markers = ['+', 'x', '_', 'o']
        for j in range(3):
            xdat = self.data['temperature'].values
            ydat = self.data['kp_' + dirs[j]].values
            label = "$\\kappa_{p}^{%s}$" % (dirs[j])
            if process is not None:
                label += f"({process})"
            ax.plot(xdat, ydat, linestyle='None', lw=lw, color=cmap(j), 
                    marker=markers[j], mew=lw, mec=color,
                    ms=ms, mfc='none', 
                    label=label)
        
        lw_ave = lw * 1.3
        ms_ave = ms * 1.3
        
        ## kp_ave
        ydat = self.data['kp_ave'].values
        label = "$\\kappa_{p}^{ave}$"
        if process is not None:
            label += f"({process})"
        ax.plot(xdat, ydat, linestyle='none', 
                marker='o', ms=ms_ave, mew=lw_ave, mec=color, mfc='none',
                label=label)
        
        ## klat_ave
        ydat = self.data['klat_ave'].values
        label = "$\\kappa_{p+c}^{ave}$"
        if process is not None:
            label += f"({process})"
        ax.plot(xdat, ydat, linestyle='-', lw=lw_ave, color=color, label=label)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        set_axis(ax, xscale='log', yscale='log')
        set_legend(ax)
        
class KL(KappaShare):
    def __init__(self, filename):
        """ Class to handle both kp or kc data obtained using ALAMODE.
        
        How to use
        -----------
        >>> file_kp = "./mp-149/cube/kappa_14x14x14/Si.kl"
        >>> kl = KL(file_kp)
        >>> kl.get_kappa(300)
        >>> kl.data
        >>> kl.data
        >>> kl.plot(ax)
        
        """
        self.filename = filename
        self.suffix = filename.split("/")[-1].split(".")[-1]
        self.data = read_kl_file(filename)
        
        if filename.endswith('.kl_coherent'):
            self.kappa_type = 'kc'
        elif filename.endswith(('.kl', '.kl3', '.kl4')):
            self.kappa_type = 'kp'
        else:
            self.kappa_type = 'unknown'

    def include_isotope_effect(self):
        words = ['isotope', 'included']
        with open(self.filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = lines.split()
                if len(data) == 0:
                    continue
                if line[0] != '#':
                    continue
                if all(word in line.lower() for word in words):
                    return True
        return False
    
    def plot(self, ax, process=None, linestyle='none', color='blue',
             lw=0.4, ms=2.3, marker=None, xlabel=None, ylabel=None, xscale='log', yscale='log'):
        """ Plot kappa vs temperature.
        """
        if self.kappa_type == 'kp':
            base_label = "\\kappa_{p}"
        elif self.kappa_type == 'kc':
            base_label = "\\kappa_{c}"
        else:
            base_label = "\\kappa"
        
        dirs = ['xx', 'yy', 'zz', 'ave']
        markers = ['+', 'x', '_', 'o']
        xdat = self.data['temperature'].values
        for j in range(4):
            
            if dirs[j] == "ave":
                alpha = 1.0
                msi = ms * 1.3
                lwi = lw * 1.3
                markeri = marker if marker is not None else markers[j]
            else:
                alpha = 0.5
                msi = ms
                lwi = lw
                markeri = marker if marker is not None else markers[j]
            
            ## label
            label = "${\\rm %s^{%s}}$" % (base_label, dirs[j])
            if process is not None:
                label += f"({process})"
            
            col = f"k{dirs[j]}"
            ydat = self.data[col].values
            ax.plot(xdat, ydat, 
                    linestyle=linestyle, lw=lwi, color=color,
                    marker=markeri, mew=lwi, mec=color, 
                    mfc='none', ms=msi, alpha=alpha,
                    label=label)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        set_axis(ax, xscale=xscale, yscale=yscale)

def read_kl_file(filename):
    """ Read a .kl or .kl_coherent file and return a DataFrame.
    
    Return
    ------
    DataFrame
        Columns: ['temperature', 'kxx', 'kyy', 'kzz']
    
    """
    ends = ['kl', 'kl3', 'kl4']
    if filename.endswith('.kl_coherent'):
        ndata = 4
        nsteps = 1
    elif filename.endswith(ends[0]) or filename.endswith(ends[1]) or filename.endswith(ends[2]):
        ndata = 10
        nsteps = 4
    else:
        return
    
    dirs = ['xx', 'yy', 'zz']
    dump = {'temperature': [], 'kxx': [], 'kyy': [], 'kzz': []}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) == 0:
                continue
            if parts[0].startswith('#'):
                continue
            if len(parts) != ndata:
                ## Columns merge when a component overflows the fixed-width
                ## format, which means the calculation failed. Do not read it.
                logger.warning(f"\n Warning: Skip an unreadable line in {filename}: {line.rstrip()}")
                continue
            
            dump['temperature'].append(float(parts[0]))
            for j in range(3):
                dump['k'+dirs[j]].append(float(parts[1 + j * nsteps]))
        df = pd.DataFrame(dump)
        df['kave'] = df[['kxx', 'kyy', 'kzz']].mean(axis=1)
    
    if df['kave'].max() > 1e6:
        if filename.startswith('/'):
            filename = os.path.relpath(filename)
        kmax = df['kave'].max()
        logger.warning(f"\n Warning: Unusually high thermal conductivity values found in {filename} ({kmax:.3e}).")
    
    return df
