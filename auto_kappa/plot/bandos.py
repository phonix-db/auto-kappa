#
# bandos.py
#
# Usuful functions to plot phonon band structure (.bands) and DOS (.dos)
# calculated with Alamode.
#
# Copyright (c) 2022 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
# import sys
import os.path

from auto_kappa.plot.initialize import set_legend

import logging
logger = logging.getLogger(__name__)

def plot_bandos_for_different_sizes(
    almcalc1, almcalc2, 
    fontsize=7, fig_width=4.0, aspect=0.5, dpi=600, lw=0.5,
    color1='blue', color2='grey',
    figname="fig_bandos.png"):
    """ Plot phonon dispersions and DOS for different supercell sizes
    
    Args
    =====
    
    almcalc : auto_kappa.alamode.AlmCalc
    
    """
    ### Plot band and DOS
    lab1 = ""
    lab2 = ""
    for j in range(3):
        lab1 += "%d" % (almcalc1.scell_matrix[j][j])
        lab2 += "%d" % (almcalc2.scell_matrix[j][j])
        if j != 2:
            lab1 += "x"
            lab2 += "x"
    
    ## Prepare Band and Dos objects
    from auto_kappa.io.band import Band
    from auto_kappa.io.dos import Dos
    file_band1 = almcalc1.out_dirs["harm"]["bandos"] + '/' + almcalc1.prefix + '.bands'
    file_band2 = almcalc2.out_dirs["harm"]["bandos"] + '/' + almcalc2.prefix + '.bands'
    file_dos1 = almcalc1.out_dirs["harm"]["bandos"] + '/' + almcalc1.prefix + '.dos'
    file_dos2 = almcalc2.out_dirs["harm"]["bandos"] + '/' + almcalc2.prefix + '.dos'
    band1 = Band(filename=file_band1)
    band2 = Band(filename=file_band2)
    dos1 = Dos(filename=file_dos1)
    dos2 = Dos(filename=file_dos2)
    
    ## Prepare figure
    from auto_kappa.plot.initialize import prepare_bandos_axes
    fig, ax1, ax2 = prepare_bandos_axes(fontsize=fontsize, fig_width=fig_width, aspect=aspect)
    
    ## Plot band
    band2.plot(ax1, color=color2, linestyle='--', lw=lw, label=lab2)
    band1.plot(ax1, color=color1, lw=lw, label=lab1)
    set_legend(ax1, fontsize=6, loc='upper right', loc2=(1.0, 1.0))
    
    ## Plot DOS
    dos2.plot(ax2, color=color2, lw=lw, xlabel=None, plot_pdos=False, show_legend=False)
    dos1.plot(ax2, color=color1, lw=lw, xlabel=None, plot_pdos=False, show_legend=False)
    
    ## Set axis limits
    ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)
    
    ## Save figure
    if os.path.isabs(figname):
        figname = "./" + os.path.relpath(figname, os.getcwd())
    fig.savefig(figname, dpi=dpi, bbox_inches='tight')
    msg = " Output %s" % figname
    logger.info(msg)
    
    
def set_xticks_labels(ax, kmax, ksym, labels):
    """ """    
    dk_all = kmax
    
    label_mod = []
    for i, label in enumerate(labels):
        
        lab = label.replace("GAMMA", " \\Gamma ")
        lab = lab.replace("DELTA", " \\Delta ")
        lab = lab.replace("SIGMA", " \\Sigma ")
        lab = lab.replace("LAMBDA", " \\Lambda ")
        
        if i < len(labels)-1:
            for j in range(2):
                if j == 0:
                    num = i - 1
                else:
                    num = i + 1
                dk_each = ksym[num] - ksym[i]
                fw_each = dk_each / dk_all
                
                if "|" in label and fw_each < 0.1:
                    names = lab.split('|')
                    lab = "^{%s}/_{%s}" % (names[0], names[1])
                    break
        ###
        label_mod.append("${\\rm %s}$" % lab)
    
    ###
    ax.set_xticks(ksym)
    ax.set_xticklabels(label_mod)


# def get_label(FILE):
#     line = FILE.split("/")
#     line = line[ len(line)-1 ]
#     line = line.split(".")
#     label = line[ len(line)-2 ]
#     return label

# def conv_unit(unit, band, dos):
    
#     from auto_kappa.units import CmToHz, CmToEv

#     if unit.lower() == "thz":
#         unit_conv = CmToHz * 1e-12
#         ylabel = "THz"
#     elif unit.lower() == "mev":
#         unit_conv = CmToEv * 1e3
#         ylabel = "meV"
#     elif unit.lower() == "cm" or unit.lower() == "kayser":
#         unit_conv = 1.
#         ylabel = "$\\rm{cm^{-1}}$"
#     else:
#         msg = " Error: %s is not defined." % unit
#         logger.error(msg)
#         sys.exit()
#     band.frequencies *= unit_conv
#     if dos is not None:
#         dos.frequencies *= unit_conv
#     return unit_conv, ylabel



# def file_check(file):
#     if os.path.exists(file) == False:
#         return False
#     return True

# def _get_pdos(dos):
#     """ 
#     dos : Dos object
#     """
#     nene = len(dos.dos_atom)
#     nel = len(dos.nat_el)
#     pdos = np.zeros((nel, nene))
#     i1 = 0
#     for iel in range(nel):
#         i0 = i1
#         i1 = i0 + dos.nat_el[iel]

#         ratio = 1.
#         pdos[iel, :] = (
#                 np.sum(dos.dos_atom[:,i0:i1], axis=1) / ratio)
#     return pdos

# def _plot_pdos(ax, frequencies, pdos, lw=0.8, labels=None):
    
#     cmap = plt.get_cmap("tab10")
#     nelements = len(pdos)
#     for ie in range(nelements):
#         ax.plot(pdos[ie,:], frequencies,
#                 linestyle='-', c=cmap(ie),
#                 lw=lw, label=labels[ie])

# def set_colorbar(sc, ax):
    
#     from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#     axins = inset_axes(ax,
#             width="50%", height="3%", 
#             loc='upper center',
#             bbox_to_anchor=[0, -0.03, 1, 1],
#             bbox_transform=ax.transAxes,
#             borderpad=0
#             )
#     set_axis(axins)
#     cb = plt.colorbar(sc, cax=axins, orientation='horizontal')
#     sc.set_clim([0.0, 1.0])

# def _get_band_files(directory, prefix):
#     # dir1 = directory + '/' + prefix
#     extensions = ['bands', 'dos', 'pr', 'band.pr']
#     filenames = {}
#     for ext in extensions:
#         filename = directory+'/'+prefix+'.'+ext
#         if os.path.exists(filename):
#             filenames [ext] = filename
#     return filenames
# 
# def plot_bandos(
#     directory='.', prefix=None, figname=None,
#     directory2=None, prefix2=None,
#     fig_width=4.0, fig_aspect=0.5,
#     ymin=None, ymax=None, xmax2=None, yticks=None, myticks=None,
#     fig_labels=[None, None],
#     lw=0.3, lw2=0.6, wspace=0.05,
#     linestyle="-", linestyle2=":",
#     dpi=600, col='blue', col2='grey',
#     unit='cm', legend_loc='best',
#     plot_dos=True, plot_pdos=True, plot_pr=True,
#     plot_dos2=False, 
#     show_legend=True, show_colorbar=True,
#     ):
#     """ Plot phonon dispersion and DOS. If .band.pr file (participation ratio) 
#     is located in the given directory and ``plot_pr`` is True, it is shown with
#     colors on phonon dispersion.
    
#     Args
#     -----
#     directory : string, default=.
#     prefix : string, default None
#         {directory}/{prefix}.*** are searched.
    
#     figname : string, default=None
#         outputfigure name
    
#     directory2 : string, default None
#     prefix2 : string, default None
#         Prefix for the second band dispersion
    
#     fig_width : double, default 3.5
#         figure width
#     fig_aspect : double, default 0.5
#         figure aspect

#     ymin : double, default None
#     ymax : double, default None
#     xmax2 : double, default None
    
#     yticks : double, default None
#     myticks : int, default None

#     lw : double, default 0.3
#     lw2 : double, default 0.5
#         line width for the second phonon dispersion
#     wspace : double, default 0.05

#     dpi : double, default 600
#     col : default blue
#     col2 : default grey
#     unit : default cm
#     legend_loc : default best

#     plot_dos, plot_pdos, plot_pr : default True
#     show_legend, show_colorbar : default True
    
#     How to Use
#     ------------
#     >>> plot_bandos(
#             directory='.', prefix='Si',
#             figname='fig_bandos.png',
#             plot_pr=True)
    
#     """
#     ### anphon file names
#     filenames = _get_band_files(directory, prefix)
    
#     ### get band
#     from auto_kappa.io.band import Band
#     band = None
#     if 'bands' not in filenames:
#         msg = "\n"
#         msg += " Error: fn_band must be given.\n"
#         msg += "\n"
#         logger.error(msg)
#         return None
#     else:
#         if os.path.exists(filenames['bands']) == False:
#             msg = " %s does not exist." % filenames['bands']
#             logger.info(msg)
#         else:
#             band = Band(filename=filenames['bands'])
     
#     ### get DOS
#     from auto_kappa.io.dos import Dos
#     dos = None
#     if 'dos' in filenames and plot_dos:
#         if os.path.exists(filenames['dos']) == False:
#             msg = " %s does not exist." % filenames['dos']
#             logger.info(msg)
#         else:
#             dos = Dos(filename=filenames['dos'])
    
#     ### min and max frequencies
#     f0 = np.amin(band.frequencies)
#     f1 = np.amax(band.frequencies)
#     df = f1 - f0
    
#     if ymin is None:
#         fmin = f0 - df * 0.05
#     else:
#         fmin = ymin
    
#     if ymax is None:
#         if ('pr' in filenames or 'band.pr' in filenames) and plot_pr:
#             fmax = f1 + df * 0.2
#         else:
#             fmax = f1 + df * 0.05
#     else:
#         fmax = ymax
    
#     ### get participation ratio
#     pr_ratio = None
#     for ext in ['band.pr', 'pr']:
#         if ext in filenames and plot_pr:
#             prfile = filenames[ext]
#             pr_ratio = Participation(file=prfile)
    
#     freq_scale, ylabel = conv_unit(unit, band, dos)
    
#     ##
#     if fig_labels[0] is None:
#         fig_labels[0] = prefix
    
#     if fig_labels[1] is None:
#         fig_labels[1] = prefix2
    
#     global plt
    
#     ### set figure
#     set_matplot(fontsize=7)
#     fig = plt.figure(figsize=(fig_width, fig_aspect*fig_width))
    
#     plt.subplots_adjust(wspace=wspace)
    
#     ax1, ax2 = prepare_two_axes(ratio='3:1')
    
#     x2label = "DOS (a.u.)"
#     ylabel = "Frequency (%s)" % ylabel
#     ax1.set_ylabel(ylabel)
#     ax2.set_xlabel(x2label)
    
#     ### set ticks and labels
#     if pr_ratio is None:
        
#         if prefix2 is not None:
#             lab = fig_labels[0]
#         else:
#             lab = None

#         plot_bands(
#             ax1, band.kpoints, band.frequencies, 
#             col=col, lw=lw, linestyle=linestyle, zorder=2, label=lab)

#         set_xticks_labels(ax1, band.kpoints[-1], band.symmetry_kpoints, band.symmetry_labels)
#         ax1 = set_axis(ax1, yticks=yticks, myticks=myticks)
#         ax2 = set_axis(ax2, yticks=yticks, myticks=myticks)
        
#     else:
#         # --- coloring fllowing the participation ratio
#         cdict = {
#                 'red':   ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
#                 'green': ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
#                 'blue':  ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
#                 }
#         cmap = matplotlib.colors.LinearSegmentedColormap(
#                 'my_colormap',cdict,256)
#         ## get all data
#         nk = band.nk
#         nb = band.nbands
#         xdat = np.zeros((nk,nb))
#         for ib in range(band.nbands):
#             xdat[:,ib] = band.kpoints[:]
#         xdat = xdat.reshape(nk*nb)
#         ydat = band.frequencies.reshape(nk*nb)
#         cdat = pr_ratio.ratio.reshape(nk*nb)
#         isort = np.argsort(cdat)[::-1]
#         sc = ax1.scatter(
#                 xdat[isort], 
#                 ydat[isort],
#                 c=cdat[isort],
#                 cmap=cmap,
#                 marker=".", s=0.3, lw=lw, zorder=2)
#         ##
#         if show_colorbar:
#             set_colorbar(sc, ax=ax1)

#         ### Set axes
#         set_xticks_labels(ax1, band.kpoints[-1], band.symmetry_kpoints, band.symmetry_labels)
#         ax1 = set_axis(ax1, yticks=yticks, myticks=myticks)
#         ax2 = set_axis(ax2, yticks=yticks, myticks=myticks)
        
#     ## zero-line
#     #ax1.axhline(0, ls='-', lw=0.2, c='grey')
    
#     ### plot DOS
#     if dos is not None:
#         ax2.plot(dos.dos, dos.frequencies, 
#                 marker="None", c=col, mew=lw, ms=1, mfc='none', lw=lw,
#                 zorder=2, label='total')
        
#         xmin = 0.
#         if xmax2 is None:
#             xmax = np.max(dos.dos)
#         else:
#             xmax = xmax2
#         dx = xmax - xmin
#         ax2.set_xlim(xmin - 0.05*dx, xmax + 0.05*dx)
        
#         ### plot PDOS
#         if plot_pdos:
            
#             pdos = _get_pdos(dos)
            
#             if len(pdos) > 1:
#                 _plot_pdos(
#                         ax2, dos.frequencies, pdos,
#                         lw=lw*0.5, labels=dos.elements
#                         )
#                 if show_legend:
#                     set_legend(ax2, fs=6, alpha=0.5, loc=legend_loc)
    
#     else:
#         ax2.set_xlim(-0.1, 1.1)
        
#     ### Second band and DOS
#     if directory2 is not None and prefix2 is not None:
        
#         filenames2 = _get_band_files(directory2, prefix2)
        
#         if 'bands' in filenames2:
#             band2 = Band(filename=filenames2['bands'])
#             plot_bands(ax1, band2.kpoints, band2.frequencies,
#                     linestyle=linestyle2,
#                     col=col2, lw=lw2, zorder=1, label=fig_labels[1])
        
#         if 'dos' in filenames2 and plot_dos2:
#             dos2 = Dos(filename=filenames2['dos'])
#             ax2.plot(dos2.dos, dos2.frequencies, 
#                     marker="None", c=col2,
#                     mew=lw, ms=1, mfc='none', lw=lw,
#                     zorder=1)
        
#     ###        
#     ax1.set_ylim(fmin*freq_scale, fmax*freq_scale)
#     ax2.set_ylim(fmin*freq_scale, fmax*freq_scale)
    
#     if prefix2 is not None:
#         set_legend(ax1, fs=6, alpha=0.5, loc='best')
        
#     if figname is not None:
#         plt.savefig(figname, dpi=dpi, bbox_inches='tight')
#         plt.close()
#         msg = " Output %s" % figname
#         logger.info(msg)
    
#     return fig

# def plot_bands(ax, ks_tmp, frequencies,
#         col='blue', lw=0.5, zorder=10, label=None, 
#         linestyle="-", marker="None", ms=1, mfc='none',
#         **args):
    
#     kpoints = ks_tmp.copy()
    
#     nbands = len(frequencies[0])
#     for ib in range(nbands):

#         idx_zero = np.where(np.diff(kpoints) < 1e-10)[0]
        
#         # koffset = 0.
#         for isec in range(len(idx_zero)+1):
             
#             if isec == 0:
#                 i0 = 0
#                 if len(idx_zero) == 0:
#                     i1 = len(kpoints)
#                 else:
#                     i1 = idx_zero[0] + 1
#             elif isec < len(idx_zero):
#                 i0 = idx_zero[isec-1] + 1
#                 i1 = idx_zero[isec] + 1
#             else:
#                 i0 = idx_zero[isec-1] + 1
#                 i1 = len(kpoints)
            
#             if ib == 0 and isec == 0:
#                 lab = label
#             else:
#                 lab = None
            
#             ax.plot(kpoints[i0:i1], frequencies[i0:i1,ib], 
#                     linestyle=linestyle,
#                     marker=marker, c=col, 
#                     mew=lw, ms=ms, mfc=mfc, lw=lw,
#                     zorder=zorder, label=lab,
#                     **args
#                     )

