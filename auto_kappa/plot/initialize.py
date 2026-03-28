#
# initialize.py
#
# Useful functions to generate figures using matplotlib
#
# Copyright (c) 2022 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.ticker import *
import matplotlib.gridspec as gridspec

import logging
logger = logging.getLogger(__name__)

def make_figure(nrows, ncols, fontsize=7, fig_width=2.5, aspect=0.9, hspace=0.15, wspace=0.1):
    """ Make a figure with a grid of subplots.
    """
    set_matplot(fontsize=fontsize)
    fig = plt.figure(figsize=(fig_width, aspect*fig_width))
    plt.subplots_adjust()
    
    gs_main = gridspec.GridSpec(1, 1, figure=fig)    
    gs_sub = gridspec.GridSpecFromSubplotSpec(
        nrows, ncols, subplot_spec=gs_main[0], wspace=wspace, hspace=hspace)    
    
    axes = []
    for ir in range(nrows):
        axes.append([])
        for ic in range(ncols):        
            ax = plt.Subplot(fig, gs_sub[ir, ic])
            fig.add_subplot(ax)
            axes[ir].append(ax)
    return fig, axes

def set_matplot(fontsize=9, lw=0.5):
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["mathtext.fontset"] = 'dejavusans'
    plt.rcParams['axes.linewidth'] = lw
    plt.rcParams['xtick.major.width'] = lw
    plt.rcParams['xtick.minor.width'] = lw
    plt.rcParams['ytick.major.width'] = lw
    plt.rcParams['ytick.minor.width'] = lw 

def set_spaces(plt, left=0.14, bottom=0.14, right=0.98, top=0.98, ratio=1.0,
               wspace=0., hspace=0.):
    plt.subplots_adjust(
            left=left, bottom=bottom,
            right=right, top=top, wspace=wspace, hspace=hspace)

def set_axis(ax, xscale="linear", yscale="linear", 
             xticks=None, mxticks=None, yticks=None, myticks=None,
             labelbottom=None, length=2.4, width=0.5):
    
    ax.tick_params(axis='both', which='major', direction='in', length=length, width=width)
    ax.tick_params(axis='both', which='minor', direction='in', length=length*0.6, width=width)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    
    #--- for linear scale
    if xticks is not None:
        ax.xaxis.set_major_locator(tick.MultipleLocator(xticks))
    if mxticks is not None:
        interval = float(xticks) / float(mxticks)
        ax.xaxis.set_minor_locator(tick.MultipleLocator(interval))
    if yticks is not None:
        ax.yaxis.set_major_locator(tick.MultipleLocator(yticks))
    if myticks is not None:
        interval = float(yticks) / float(myticks)
        ax.yaxis.set_minor_locator(tick.MultipleLocator(interval))
    #--- for logscale
    if xscale.lower() == "log":
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(tick.LogLocator(base=10.0, numticks=15))
    if yscale.lower() == "log":
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(tick.LogLocator(base=10.0, numticks=15))
    return ax

def get_both_axis(ratio="2:1"):
    logger.info("\n 'get_both_axis' will be deprecated in the future version.")
    return prepare_two_axes(ratio=ratio)

def prepare_bandos_axes(ratio='3:1', fontsize=7, fig_width=4.0, aspect=0.5):
    set_matplot(fontsize=fontsize)
    fig = plt.figure(figsize=(fig_width, aspect*fig_width))
    plt.subplots_adjust(wspace=0.05)
    ax1, ax2 = prepare_two_axes(ratio='3:1')
    return fig, ax1, ax2

def prepare_two_axes(ratio="2:1"):
    w1 = int(ratio.split(':')[0])
    w2 = int(ratio.split(':')[1])
    gs = gridspec.GridSpec(1,w1+w2)
    ax1 = plt.subplot(gs[0,:w1])
    ax2 = plt.subplot(gs[0,w1:])
    set_axis(ax1)
    set_axis(ax2)
    ax2.yaxis.set_major_formatter(NullFormatter())
    return ax1, ax2 

def set_legend(plt, ncol=1, fontsize=6, fs=None, loc="best", loc2=None,
               alpha=1.0, lw=0.2, length=1.0, labelspacing=0.3, borderpad=None,
               title=None, edgecolor='black', facecolor='white'):
    
    fontsize = fs if fs is not None else fontsize
    
    leg = plt.legend(
            loc=loc, ncol=ncol, fontsize=fontsize, fancybox=False,
            facecolor=facecolor, edgecolor=edgecolor, handletextpad=0.4,
            handlelength=length, labelspacing=labelspacing,
            borderpad=borderpad, title=title)
    
    if loc2 is not None:
        leg.set_bbox_to_anchor([loc2[0], loc2[1]])
    
    leg.get_frame().set_alpha(alpha)
    leg.get_frame().set_linewidth(lw)	
    return leg

def set4bandos():
    FIG_WIDTH = 3.3
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH*0.9))
    plt.subplots_adjust(
            left=0.14, bottom=0.14,
            right=0.98, top=0.98, wspace=0, hspace=0)
    return fig, plt

def set_axis_lim(ax, data, axis='x', alpha=0.05, scale='linear'):
    if scale == 'linear':
        vmin = np.min(data)
        vmax = np.max(data)
        x0 = vmin - alpha*(vmax - vmin)
        x1 = vmax + alpha*(vmax - vmin)
    elif scale == 'log':
        cmin = np.log10(np.min(data))
        cmax = np.log10(np.max(data))
        c0 = cmin - alpha*(cmax - cmin)
        c1 = cmax + alpha*(cmax - cmin)
        x0 = np.power(10, c0)
        x1 = np.power(10, c1)
    else:
        return None
    if axis == 'x':
        ax.set_xlim([x0, x1])
    else:
        ax.set_ylim([x0, x1])

def set_second_axis(ax):
    ax2 = ax.twinx()
    set_axis(ax)
    set_axis(ax2)
    ax.tick_params(labelright=False, right=False, which='both')
    ax2.tick_params(labelleft=False, left=False, which='both')
    return ax2

def get_customized_cmap(nbins, color1='blue', color2='red'):
    from matplotlib.colors import LinearSegmentedColormap
    colors = [color1, color2]
    cm = LinearSegmentedColormap.from_list('mylist', colors, N=nbins)
    return cm

def sci2text(value, ndigits=2):
    """ Convert a scientific notation number to text format.
    """
    fmt = "{:." + str(ndigits) + "e}"
    sval = fmt.format(value)
    if 'e' in sval:
        base, exponent = sval.split('e')
        exponent = int(exponent)
        if exponent == 0:
            return f"{base}"
        else:
            return f"{base} $\\times$ 10$^{{{exponent}}}$"
    else:
        return sval

