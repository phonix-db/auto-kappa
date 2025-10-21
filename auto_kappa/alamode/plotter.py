#
# plotter.py
#
# This module contains functions for plotting data from ALAMODE calculations.
#
# Copyright (c) 2025 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
import os
import sys
import numpy as np
import pandas as pd
import glob
import itertools
import copy
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

import auto_kappa.alamode.helpers as helper
from auto_kappa.alamode.io import wasfinished_alamode
from auto_kappa.plot import make_figure, set_matplot, set_axis, set_legend, get_customized_cmap
from auto_kappa.io.scattering import Scattering

import logging
logger = logging.getLogger(__name__)

class AlamodePlotter:
    
    def plot_bandos(self, fontsize=7, fig_width=4.0, aspect=0.5, lw=0.5, plot_with_pr=True, plot_G2G=False):
        """ Plot band structure and DOS. 
        
        Args
        -----    
        fontsize : int
            Font size for the plot.
        
        fig_width : float
            Width of the figure in inches.
        
        aspect : float
            Aspect ratio of the figure.
        
        lw : float
            Line width for the band and DOS plots.
        
        plot_with_pr : bool
            If True, plot the band structure with participation ratio.
        
        plot_G2G : bool
            If True, plot the band structure only between the first and second Gamma points.
        
        """
        from auto_kappa.io.band import Band
        from auto_kappa.io.participation import BandPR
        from auto_kappa.io.dos import Dos
        from auto_kappa.plot.initialize import prepare_bandos_axes
        
        fig, ax1, ax2 = prepare_bandos_axes(
            fontsize=fontsize, fig_width=fig_width, aspect=aspect)
        
        ### File names
        workdir = self.out_dirs['harm']['bandos']
        if os.path.isabs(workdir):
            workdir = "./" + os.path.relpath(workdir, os.getcwd())
        
        file_band = workdir + "/" + self.prefix + ".bands"
        file_band_pr = workdir + "/" + self.prefix + ".band.pr"
        file_dos = workdir + "/" + self.prefix + ".dos"
        figname = workdir + "/fig_bandos_pr.png"
        
        ### Plot band structure
        if plot_with_pr: # plot band with participation ratio
            try:
                pr = BandPR(file_band_pr)
                pr.plot(ax1, lw=lw, cbar_location='upper left', plot_G2G=plot_G2G)
            except Exception as e:
                msg = "\n Warning: cannot plot band participation ratio. %s" % str(e)
                logger.warning(msg)
                pr = None
        else: # plot band without participation ratio
            try:
                band = Band(file_band)
                band.plot(ax1, lw=lw, plot_G2G=plot_G2G)
            except Exception as e:
                msg = "\n Warning: cannot plot band structure. %s" % str(e)
                logger.warning(msg)
                band = None
        
        ### Plot DOS
        try:
            if lw is not None:
                lw_dos = lw * 1.3
            else:
                lw_dos = 0.8
            
            dos = Dos(file_dos)
            dos.plot(ax2, xlabel=None, lw=lw_dos, frac_lw=0.7)
            
        except Exception as e:
            msg = "\n Warning: cannot plot DOS. %s" % str(e)
            logger.warning(msg)
            dos = None
        
        ## Set ylim
        ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
        ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)
        
        fig.savefig(figname, dpi=600, bbox_inches='tight')
        msg = "\n Plot band structure and DOS: %s" % figname
        logger.info(msg)
    
    
    def plot_thermodynamic_properties(self, fontsize=7, fig_width=2.5, aspect=1.3, dpi=600):
        """ plot thermodynamic properties
        """
        from auto_kappa.io.thermo import Thermo
        filename = self.out_dirs['harm']['bandos'] + "/" + self.prefix + ".thermo"
        figname = self.out_dirs['harm']['bandos'] + "/fig_thermo.png"
        if figname.startswith("/"):
            figname = "./" + os.path.relpath(figname, os.getcwd())
        
        if not os.path.exists(filename):
            logger.warning(f"\n Thermodynamic properties file not found: {filename}")
            return
        
        set_matplot(fontsize=fontsize)
        fig = plt.figure(figsize=(fig_width, fig_width*aspect))
        plt.subplots_adjust(hspace=0.05)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        thermo = Thermo(filename)
        thermo.plot(ax1, ax2)
        
        fig.savefig(figname, dpi=dpi, bbox_inches='tight')
        msg = " Plot thermodynamic properties: %s" % figname
        logger.info(msg)
    
    
    def plot_force_constants(self, outdir=None, order=None, dpi=600, force_plot=False):
        """ Plot force constants from the ALAMODE output file. """
        from auto_kappa.io.fcs import FCSxml
        
        if self.calculate_forces == False:
            return None
        
        file_fcs = f"{outdir}/{self.prefix}.xml"
        if os.path.exists(file_fcs) == False:
            msg = "\n Error: %s does not exist." % file_fcs
            logger.error(msg)
            sys.exit()
            return None

        ## Already plotted
        figname = os.path.dirname(file_fcs) + f"/fig_fc{order+1}.png"
        if force_plot == False and os.path.exists(figname):
            if figname.startswith("/"):
                figname = "./" + os.path.relpath(figname, os.getcwd())
            logger.info("\n Figure %s already exists." % figname)
            return None
        
        try:
            fcs = FCSxml(file_fcs)
            fig, axes = make_figure(order, 1, aspect=0.5*order, fig_width=2.5, hspace=0.2)
            
            if order == 1:
                xlabel = 'Distance (${\\rm \\AA}$)'
            else:
                xlabel = None
            fcs.plot_fc2(axes[0][0], xlabel=xlabel)
            
            if order > 1:
                if order == 2:
                    xlabel = 'Distance (${\\rm \\AA}$)'
                else:
                    xlabel = None
                fcs.plot_fc3(axes[1][0], xlabel=xlabel)
            
            if order > 2:
                if order == 3:
                    xlabel = 'Max. atomic distance (${\\rm \\AA}$)'
                else:
                    xlabel = None
                fcs.plot_fc4(axes[2][0], xlabel=xlabel)
            
            fig.savefig(figname, dpi=dpi, bbox_inches='tight')
            if os.path.isabs(figname):
                figname = "./" + os.path.relpath(figname, os.getcwd())
            logger.info(f" Force constants were plotted in {figname}")
            
        except Exception:
            logger.warning("\n Warning: the figure of force constants was not created properly.")
    
    def get_scattering_info(self, dir_kappa, grain_size=None, temperature=300, process='3ph', verbose=True):
        
        ### file check
        if process == '3ph':
            file_result = dir_kappa + '/%s.result' % (self.prefix)
            file_isotope = dir_kappa + '/%s.self_isotope' % (self.prefix)
        elif process == '4ph':
            file_result = dir_kappa + '/%s.4ph.result' % (self.prefix)
            file_isotope = None # Usually, different k-mesh was used for 4ph scattering.
        else:
            logger.warning("\n Error: process should be '3ph' or '4ph'.")
            return
        
        msg1 = "\n Scattering info is obtained from the following files:"
        if os.path.exists(file_result) == False:
            msg_error = " Error: %s does not exist." % (
                    self.get_relative_path(file_result))
            logger.error(msg_error)
        else:
            msg1 += "\n - %s." % (self.get_relative_path(file_result))
        
        if file_isotope is not None:
            if os.path.exists(file_isotope) == False:
                msg_error = " Note : %s does not exist." % (
                        self.get_relative_path(file_isotope))
                logger.error(msg_error)
                file_isotope = None
            else:
                msg1 += "\n - %s." % (self.get_relative_path(file_isotope))
        
        if verbose:
            logger.info(msg1)
        
        ###
        return Scattering(file_result, 
                          file_isotope=file_isotope,
                          grain_size=grain_size, 
                          temperature=temperature)
        
    def write_lifetime_at_given_temperature(self, dir_kappa, temperature=300, grain_size=None,
                                            outfile=None, process='3ph'):
        """ Write lifetime at a given temperature in a csv file in a format similar to ALAMODE.
        
        Args
        ------
        temperature : float
            Temperature in K.
        
        outfile : string
            Output file name.
        
        grain_size : float
            Grain size in nm.
        
        """
        scat = self.get_scattering_info(dir_kappa, temperature=temperature, grain_size=grain_size, 
                                        process=process, verbose=False)
        if scat is None:
            return
        
        dump = {"ik": [], "is": [], 
                "frequency[cm^-1]": [], "lifetime[ps]": [], 
                "|velocity|[m/s]": [], "MFP[nm]": [], "multiplicity": [],
                "kxx": [], "kxy": [], "kxz": [],
                "kyx": [], "kyy": [], "kyz": [],
                "kzx": [], "kzy": [], "kzz": []}
        
        nk = len(scat.result['frequencies'])
        nbands = len(scat.result['frequencies'][0])
        for ik in range(nk):
            
            multiplicity = scat.result['multiplicity'][ik]
            
            for ib in range(nbands):
                
                frequency = scat.result['frequencies'][ik][ib] ## cm^-1
                lifetime = scat.lifetime[ik][ib]  ## ps
                
                ## velocity
                velo_all = scat.result['velocities'][ik][ib]  ## shape(multiplicity, 3)
                velo_all = np.asarray(velo_all)
                assert len(velo_all) == multiplicity
                
                vave = 0.
                for im in range(multiplicity):
                    vave += np.linalg.norm(velo_all[im]) / multiplicity
                
                ## kappa at each mode
                kappa_each = np.zeros((3, 3))
                for i1 in range(3):
                    for i2 in range(3):
                        kappa_each[i1, i2] += (
                            scat.kmode[ik,ib,i1,i2] / multiplicity
                        )
                
                dump["ik"].append(ik + 1)
                dump["is"].append(ib + 1)
                dump["frequency[cm^-1]"].append(frequency)
                dump["lifetime[ps]"].append(lifetime)
                dump["|velocity|[m/s]"].append(vave)
                dump["MFP[nm]"].append(lifetime * vave)
                dump["multiplicity"].append(multiplicity)
                for i, j in itertools.product(range(3), repeat=2):
                    dump["k" + "xyz"[i] + "xyz"[j]].append(kappa_each[i, j])
            
        ### write a csv file
        if outfile is None:
            if process == '3ph':
                outfile = dir_kappa + "/tau_%dK.csv" % (int(temperature))
            else:
                outfile = dir_kappa + "/tau_%dK_%s.csv" % (int(temperature), process)
        
        df = pd.DataFrame(dump)
        with open(outfile, 'w') as f:
            f.write("# Created by auto-kappa in a format similar to ALAMODE\n")
            f.write("# Input file   : %s\n" % self.get_relative_path(scat.result.filename))
            if scat.file_isotope is not None:
               if os.path.exists(scat.file_isotope):
                    f.write("# Isotope      : %s\n" % self.get_relative_path(scat.file_isotope))
            if grain_size is not None:
                f.write("# Grain size   : %f nm\n" % grain_size)
            f.write("# Temperature  : %d K\n" % temperature)
            f.write("# kpoint range : 1 %d\n" % nk)
            f.write("# mode   range : 1 %d\n" % nbands)
            df.to_csv(f, index=False, float_format='%.7e')
            msg = " Output %s" % self.get_relative_path(outfile)
            logger.info(msg)
    
    
    def plot_lifetime(self, dir_kappa=None, temperatures="300:500", process='3ph', 
                      fontsize=7, fig_width=2.3, aspect=0.9, dpi=600):
        """ Plot the lifetime as a function of frequency for different temperatures or
        diffrent phonon scattering process. Isotope effects are not considered.
        """
        cmap = plt.get_cmap("tab10")
        set_matplot(fontsize=fontsize)
        fig = plt.figure(figsize=(fig_width, aspect*fig_width))
        ax = plt.subplot()
        
        def _update_ax_range(ax, xlim, ylim):
            vlim_curr = ax.get_xlim()
            xlim[0] = min(xlim[0], vlim_curr[0])
            xlim[1] = max(xlim[1], vlim_curr[1])
            vlim_curr = ax.get_ylim()
            ylim[0] = min(ylim[0], vlim_curr[0])
            ylim[1] = max(ylim[1], vlim_curr[1])

        xlim = [1e3, 1e-3]
        ylim = [1e3, 1e-3]
        
        ## Plot 3-phonon scattering info when process is 4ph
        if process == '4ph':
            t = float(temperatures.split(":")[0])
            scat3 = self.get_scattering_info(dir_kappa=dir_kappa, temperature=t, 
                                             process='3ph',
                                             grain_size=None, verbose=False)
            label = f"{t:.0f}K (3ph)"
            scat3.plot_lifetime(ax, temperature=t, label=label, marker="x", color='grey')
            _update_ax_range(ax, xlim, ylim)
        
        ## Main scattering process
        scat = self.get_scattering_info(dir_kappa=dir_kappa, temperature=300,
                                        grain_size=None, process=process, verbose=False)
        
        markers = ['o', '^', 's', 'v', 'D']
        ts = [float(t) for t in temperatures.split(":")]
        for i, t in enumerate(ts):
            if process == '3ph':
                label = f"{t:.0f}K"
            else:
                label = f"{t:.0f}K ({process})"
            
            scat.plot_lifetime(ax, temperature=t, label=label,
                               marker=markers[i % len(markers)],
                               color=cmap(i % 10))    
            _update_ax_range(ax, xlim, ylim)
        
        ax.set_xlim(_get_log_range(xlim[0], xlim[1], space=0.05))
        ax.set_ylim(_get_log_range(ylim[0], ylim[1], space=0.05))
        
        set_axis(ax, xscale='log', yscale='log')
        set_legend(ax, fs=6, alpha=0.5, loc='best')
        
        ax.set_xlabel("Frequency (${\\rm cm^{-1}}$)")
        ax.set_ylabel("Lifetime (ps)")
        
        figname = f'{dir_kappa}/fig_lifetime.png'
        fig.savefig(figname, dpi=dpi, bbox_inches='tight')
        logger.info(f" Output {self.get_relative_path(figname)}")
        
        
    def plot_scattering_rates(self, dir_kappa, temperature=300., grain_size=1000., process='3ph',
                              dpi=600, fontsize=7, fig_width=2.3, aspect=0.9, lw=0.3, ms=1.3):
        """ Plot scattering rates at a given temperature and grain size.
        """
        scat = self.get_scattering_info(dir_kappa, temperature=temperature, grain_size=grain_size, 
                                        process=process, verbose=False)
        if scat is None:
            return None
        
        ## set temperature 
        if abs(scat.temperature - temperature) > 0.1:
            scat.change_temperature(temperature)
        
        ## set grain size
        if scat.size is None:
            scat.change_grain_size(grain_size)
        else:
            if abs(scat.size - grain_size) > 1.:
                scat.change_grain_size(grain_size)
        
        ## get frequencies
        frequencies = scat.result['frequencies']
        n1, n2 = frequencies.shape
        frequencies = frequencies.reshape(n1*n2)
        
        ## get scattering rates and set labels
        labels = {}
        scat_rates = {}
        for key in scat.scattering_rates:
            
            if scat.scattering_rates[key] is None:
                continue
            
            scat_rates[key] = scat.scattering_rates[key].reshape(n1*n2)
            
            if key == 'phph':
                labels[key] = f'{process} ({int(temperature)}K)'
            elif key == 'isotope':
                labels[key] = 'isotope'
            elif key == 'boundary':
                labels[key] = f'L={grain_size}nm'
            else:
                labels[key] = key
        
        ## plot a figure
        if process != '4ph':
            figname = dir_kappa + '/fig_scat_rates.png'
        else:
            figname = dir_kappa + '/fig_scat_rates_4ph.png'
        
        ## Prepare figure
        set_matplot(fontsize=fontsize)
        fig = plt.figure(figsize=(fig_width, aspect*fig_width))
        ax = plt.subplot()
        
        scat.plot_scattering_rates(ax, process=process)
        
        ax.set_xlabel('Frequency (${\\rm cm^{-1}}$)')
        ax.set_ylabel('Scattering Rate (${\\rm ps^{-1}}$)')
        
        set_axis(ax, xscale='log', yscale='log')
        set_legend(ax, fs=6, alpha=0.5, loc='best')
        fig.savefig(figname, dpi=dpi, bbox_inches='tight')
        logger.info(f" Output {self.get_relative_path(figname)}")
        
        
    def _plot_cvsets(self, order=None):
        """ Plot CV results """
        from auto_kappa.plot.pltalm import plot_cvsets
        msg = "\n ### Plot CV results ###"
        logger.info(msg)
        
        if order == 2:
            figname = self.out_dirs['result'] + '/fig_cvsets_cube.png'
            plot_cvsets(
                    directory=self.out_dirs['cube']['cv'],
                    figname=figname)
        else:
            figname = self.out_dirs['result'] + '/fig_cvsets_high.png'
            plot_cvsets(
                    directory=self.out_dirs['higher']['cv'], 
                    figname=figname)
        
        logger.info("")
    
    
    def plot_cumulative_kappa(self, dir_kappa, 
                              temperatures="100:300:500", wrt='frequency',
                              figname=None, xscale='linear', nbins=150, process='3ph'):
        
        ## set scattering info
        scat = self.get_scattering_info(dir_kappa, process=process, 
                                        temperature=300, grain_size=None, verbose=False)
        
        ## set temperatures
        data = temperatures.split(":")
        ts = [float(t) for t in data]
        dfs = {}
        for t in ts:
            
            ## Calculate cumulative and spectral kappa at a given temperature
            scat.change_temperature(t)
            dfs[int(t)] = scat.get_cumulative_kappa(temperature=t, wrt=wrt, xscale=xscale, nbins=nbins)
            
            ## output file for the cumulative and spectral kappa
            try:
                df_each = dfs[int(t)].rename(columns={'xdat': wrt})
                if process != '3ph':
                    outfile = dir_kappa + '/kspec_%s_%dK_%s.csv' % (wrt, int(t), process)
                else:
                    outfile = dir_kappa + '/kspec_%s_%dK.csv' % (wrt, int(t))
                
                ### save spectral kappa
                with open(outfile, 'w') as f:
                    
                    ## comment
                    f.write("# Temperature : %d\n" % t)
                    if scat.size is not None:
                        f.write("# Grain size  : %f\n" % scat.size)
                    
                    df_each.to_csv(f, index=False, float_format='%.6e')
                    msg = " Output %s" % self.get_relative_path(outfile)
                    logger.info(msg)
            except:
                pass
        
        ##
        lab_kappa = "${\\rm \\kappa_p}$"
        if 'freq' in wrt:
            xlabel = "Frequency (${\\rm cm^{-1}}$)"
            unit1 = "${\\rm Wm^{-1}K^{-1}/cm^{-1}}$"
            ylabel1 = "Spectral %s (%s)" % (lab_kappa, unit1)
        else:
            xlabel = "Mean free path (nm)"
            unit1 = "${\\rm Wm^{-1}K^{-1}/nm}$"
            ylabel1 = "Spectral %s (%s)" % (lab_kappa, unit1)
        
        if figname is None:
            if process == '3ph':
                figname = dir_kappa + f'/fig_cumu_{wrt}.png'
            else:
                figname = dir_kappa + f'/fig_cumu_{wrt}_{process}.png'
        
        from auto_kappa.plot.pltalm import plot_cumulative_kappa
        plot_cumulative_kappa(
                dfs, xlabel=xlabel, 
                figname=self.get_relative_path(figname), 
                xscale=xscale, ylabel=ylabel1)
    
    
    def write_kappa_vs_grain_size(self, dir_kappa, process='3ph', 
                                  outfile=None, force_compute=False,
                                  temperatures=None,
                                  grain_sizes=np.logspace(0.5, 5, 50),
                                  nprocs=1):
        """ Write thermal conductivity as a function of grain size and
        return the obtained result in the format of DataFrame
        """
        ## Output file name
        if outfile is None:
            outfile = dir_kappa + '/kappa_vs_grain_size.csv'
        outfile = self.get_relative_path(outfile)
        if not force_compute and os.path.exists(outfile):
            logger.info(f" Read {outfile}")
            df = pd.read_csv(outfile, comment="#")
            return df
        
        ## Read files such as .result, .4ph.result, .self_isotope files
        scat = self.get_scattering_info(dir_kappa, temperature=300, grain_size=None,
                                        process=process, verbose=False)
        if scat is None:
            return None
        
        ## Prepare dictionary
        directs = ['x', 'y', 'z']
        dump = {'temperature[K]': [], 'grain_size[nm]': []}
        for i, j in itertools.product(range(3), repeat=2):
            dump[f'k{directs[i]}{directs[j]}[W/mK]'] = []
        dump['kave[W/mK]'] = []
        
        ## Set temperatures
        if temperatures is None:
            temperatures = scat.temperatures
        
        ## Calculate T- and grain-size-dependent kappa
        ### ver.1: serial
        # for temp in temperatures:
        #     scat.change_temperature(temp)
        #     for size in grain_sizes:
        #         scat.change_grain_size(size)
        #         dump['temperature[K]'].append(temp)
        #         dump['grain_size[nm]'].append(size)
        #         for i, j in itertools.product(range(3), repeat=2):
        #             dump[f'k{directs[i]}{directs[j]}[W/mK]'].append(scat.kappa[i,j])
        #         dump['kave[W/mK]'].append(np.mean(np.diag(scat.kappa)))
        # df = pd.DataFrame(dump)
        
        ### ver.2: parallel
        df = _parallel_kappa_vs_grain(scat, temperatures, grain_sizes, nprocs=nprocs)
        
        with open(outfile, 'w') as f:
            f.write("# Created by auto-kappa\n")
            f.write("# %s scattering : %s\n" % (process, self.get_relative_path(scat.result.filename)))
            if scat.file_isotope is not None:
               if os.path.exists(scat.file_isotope):
                   f.write("# Isotope        : %s\n" % self.get_relative_path(scat.file_isotope))
            f.write("# Temperature [K] : %.2f to %.2f\n" % (df['temperature[K]'].min(), df['temperature[K]'].max()))
            f.write("# Grain size [nm] : %.2f to %.2f\n" % (df['grain_size[nm]'].min(), df['grain_size[nm]'].max()))
            df.to_csv(f, index=False, float_format="%.7e")
            msg = " Output %s" % self.get_relative_path(outfile)
            logger.info(msg)
        return df
    
    def plot_kappa(self, kappa_dir, dpi=600, fig_width=2.3, fontsize=7, aspect=0.9, lw=0.4, ms=2.3):
        """ Plot lattice thermal conductivity from Peierls and coherence contribution
        
        Parameters
        ----------
        kappa_dir : str
            Directory for kappa results
        """
        from auto_kappa.io.kl import Kboth
        files = {}
        for suffix in ['kl', 'kl3', 'kl4', 'kl_coherent']:
            fn = kappa_dir + '/' + self.prefix + '.' + suffix
            if os.path.exists(fn):
                files[suffix] = fn
        
        figname = kappa_dir + '/fig_kappa.png'
        
        ## Prepare figure
        set_matplot(fontsize=fontsize)
        fig = plt.figure(figsize=(fig_width, aspect*fig_width))
        ax = plt.subplot()
        
        ## plot kappa
        cmap = plt.get_cmap("tab10")
        if 'kl_coherent' in files:
            if 'kl' in files:
                try:
                    both_obj = Kboth(files['kl'], files['kl_coherent'])
                    both_obj.plot(ax, color=cmap(0), lw=lw, ms=ms)
                except Exception as e:
                    logger.error(f" Error plotting kl and kl_coherent: {e}")
            elif 'kl4' in files:
                ## data with 4ph scattering
                try:
                    both_obj = Kboth(files['kl4'], files['kl_coherent'])
                    both_obj.plot(ax, color=cmap(0), lw=lw, ms=ms, process='4ph')
                except Exception as e:
                    logger.error(f" Error plotting kl4 and kl_coherent: {e}")
                
                if 'kl3' in files:
                    ## data with 3ph scattering
                    try:
                        both_obj = Kboth(files['kl3'], files['kl_coherent'])
                        xdat = both_obj.data['temperature']
                        ydat = both_obj.data['klat_ave']
                        label = '${\\rm \\kappa_{p+c}^{ave}(3ph)}$'
                        if len(xdat) > 0:
                            ls = None if len(xdat) == 1 else '--'
                            marker = 'o' if len(xdat) == 1 else None
                            ax.plot(xdat, ydat, linestyle=ls, marker=marker, 
                                    ms=ms, mec=cmap(1), mfc='none', mew=lw,
                                    color=cmap(1), lw=lw, label=label)
                    except Exception as e:
                        logger.error(f" Error plotting kl3 and kl_coherent: {e}")
        
        set_axis(ax, xscale='log', yscale='log')
        set_legend(ax, fs=6, alpha=0.5, loc='best')
        
        fig.savefig(figname, dpi=dpi, bbox_inches='tight')
        if figname.startswith('/'):
            figname = "./" + os.path.relpath(figname, os.getcwd())
        logger.info(f" Output {figname}")
    
    
    def get_kappa_directories(self, calc_type="cubic"):
        
        if calc_type == "cubic":
            ll_tmp = self.out_dirs['cube']["kappa_%s" % self.fc3_type] + "_"
        elif calc_type == "scph":
            ll_tmp = self.out_dirs["higher"]["kappa"] + "_"
        
        line = ll_tmp + "*"
        
        dirs = glob.glob(line)
        dirs_done = {}
        for dd in dirs:
            fn_log = dd + "/kappa.log"
            if os.path.exists(fn_log) == False:
                continue
            if wasfinished_alamode(fn_log):
                label = dd.split(ll_tmp)[1]
                dirs_done[label] = dd
        return dirs_done
    
    
    def plot_all_kappa(self, figname=None, calc_type="cubic"):
        """ Plot all kappa results
        
        Parameters
        ----------
        calc_type : string
            "cubic", "scph"
        """
        dirs_kappa = self.get_kappa_directories(calc_type=calc_type)
        keys = dirs_kappa.keys()
        try:
            keys = sorted(keys, key=lambda x: eval(x.replace('x', '*')))
        except Exception:
            pass
        
        ## Scale of kappa for 2D systems
        if self.dim == 2:
            c_len = np.linalg.norm(self.unitcell.cell[self.norm_idx_abc])
            kappa_scale = c_len / self.thickness
        else:
            kappa_scale = 1.0
        
        dfs = {}
        logger.info("")
        for i, key in enumerate(keys):
            try:
                dir1 = self.get_relative_path(dirs_kappa[key])
                msg = " Read %s" % (dir1)
                logger.info(msg)
                dfs[key] = helper.read_kappa(
                    dirs_kappa[key], self.prefix, dim=self.dim, 
                    norm_idx=self.norm_idx_xyz, kappa_scale=kappa_scale)
            except Exception:
                continue
        
        from auto_kappa.plot.pltalm import plot_all_kappa
        if figname is None:
            figname = self.out_dirs['result'] + '/fig_kappa.png'
        logger.info("")
        plot_all_kappa(dfs, figname=self.get_relative_path(figname), dim=self.dim)
    

def _simulate_each_temp(temp, grain_sizes, scat_orig):
    """ Simulate thermal conductivity vs grainsize at a given temperature. 
    """
    scat = copy.deepcopy(scat_orig)
    scat.change_temperature(temp)
    
    directs = ['x', 'y', 'z']
    local_result = []
    for size in grain_sizes:
        scat.change_grain_size(size)
        result = {
            'temperature[K]': temp,
            'grain_size[nm]': size,
            'kave[W/mK]': float(np.mean(np.diag(scat.kappa)))
        }
        for i, j in itertools.product(range(3), repeat=2):
            result[f'k{directs[i]}{directs[j]}[W/mK]'] = float(scat.kappa[i, j])
        local_result.append(result)
        
    return local_result

def _parallel_kappa_vs_grain(scat, temperatures, grain_sizes, nprocs=1):
    """ Compute kappa vs grain size at different temperatures 
    """
    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        futures = [
            executor.submit(_simulate_each_temp, temp, grain_sizes, scat)
            for temp in temperatures
        ]
        results = []
        for f in futures:
            results.extend(f.result())
    
    ## Convert to a dict
    dump = {key: [] for key in results[0].keys()}
    for row in results:
        for key, val in row.items():
            dump[key].append(val)
    df = pd.DataFrame(dump)
    return df


def plot_kappa_vs_grain_size(filename: str, figname: str, fontsize=7, marker=None,
                             fig_width=2.3, aspect=0.9, dpi=600, lw=0.4, ms=2.3):
    """ Plot kappa vs grain size
    """
    df = pd.read_csv(filename, comment="#")
    
    set_matplot(fontsize=fontsize)
    fig = plt.figure(figsize=(fig_width, aspect*fig_width))
    
    ax = plt.subplot()
    ax.set_xlabel('Grain size (nm)')
    ax.set_ylabel('Thermal conductivity (${\\rm W m^{-1} K^{-1}}$)')
    
    temp_list = np.sort(df['temperature[K]'].unique())
    nt = len(temp_list)
    mycmap = get_customized_cmap(nt, color1='blue', color2='red')
    for it, temp in enumerate(temp_list):
        
        df_t = df[df['temperature[K]'] == temp]
        xdat = df_t['grain_size[nm]'].values
        ydat = df_t['kave[W/mK]'].values
        
        if it == 0 or it == nt - 1:
            label = f"{temp:.0f}K"
        else:
            label = None
        
        ax.plot(xdat, ydat, linestyle='-', lw=lw, color=mycmap(it),
                marker=marker, ms=ms, mec=mycmap(it), mfc='none', mew=lw,
                label=label)
    
    ## 300K
    if np.min(abs(temp_list - 300)) < 1.0:
        df_300K = df[abs(df['temperature[K]'] - 300) < 1.0]
        xdat_300K = df_300K['grain_size[nm]'].values
        ydat_300K = df_300K['kave[W/mK]'].values
        ax.plot(xdat_300K, ydat_300K, linestyle='-', lw=lw*2, color='black',
                marker=marker, ms=ms, mec='black', mfc='none', mew=lw*2,
                label="300K")
    
    set_axis(ax, xscale='log', yscale='log')
    set_legend(ax, fs=6, alpha=0.5, loc='best')
    
    fig.savefig(figname, dpi=dpi, bbox_inches='tight')
    if figname.startswith('/'):
        figname = "./" + os.path.relpath(figname, os.getcwd())
    logger.info(f" Output: {figname}")
    return fig

def _get_log_range(vmin, vmax, space=0.05):
    if vmin <= 0 or vmax <= 0:
        return [vmin, vmax]
    vmin_log = np.log10(vmin)
    vmax_log = np.log10(vmax)
    dv_log = vmax_log - vmin_log
    v0_log = vmin_log - dv_log * space
    v1_log = vmax_log + dv_log * space
    v0 = np.power(10, v0_log)
    v1 = np.power(10, v1_log)
    return [v0, v1]
