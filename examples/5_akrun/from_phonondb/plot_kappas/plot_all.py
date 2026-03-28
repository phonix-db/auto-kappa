# -*- coding: utf-8 -*-
import os
import numpy as np
from optparse import OptionParser
import glob
from pymatgen.core.composition import Composition
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from auto_kappa.plot import set_matplot, set_axis, set_legend

all_dir = './all_data'

def copy_all_files():
    
    os.makedirs(all_dir, exist_ok=True)
    
    ##
    all_data = {}
    for label in ['cube', 'lasso']:
        for ext in ['kl', 'kl_coherent']:
            #print("=== %s ===" % label)
            line = '../mp-*/%s/kappa/*.%s' % (label, ext)
            fns = glob.glob(line)
            for fn1 in fns:
                if 'boundary' in fn1:
                    continue

                words = fn1.split('/')
                mpid = words[1]
                comp = words[-1].split('.')[0]
                
                fn2 = all_dir + '/' + mpid + '_' + comp + '.' + ext
                shutil.copy(fn1, fn2)
                
def get_all_data():
    
    ##
    all_data = {}
    for label in ['cube', 'lasso']:
        for ext in ['kl', 'kl_coherent']:
            print("=== %s ===" % label)
            line = '%s/*.%s' % (all_dir, ext)
            fns = glob.glob(line)
            for fn in fns:
                if 'boundary' in fn:
                    continue

                words = fn.split('all_data/')[-1].split('_')
                mpid = words[0]
                comp = words[-1].split('.')[0]
                
                print(mpid, comp)
                if mpid not in all_data:
                    all_data[mpid] = {
                            'mpid': mpid,
                            'composition': comp
                            }
                all_data[mpid][ext] = np.genfromtxt(fn)
    ##
    return all_data

def plot_all_data(all_data, kappa_type='both', figname='fig_all_kappas.png', 
        dpi=300, fontsize=7, fig_width=2.3, aspect=0.9, lw=0.5, ms=0.5):
    
    cmap = plt.get_cmap("tab10")
    set_matplot(fontsize=fontsize)
    fig = plt.figure(figsize=(fig_width, aspect*fig_width))
        
    ax = plt.subplot()
    ax.set_xlabel('T (K)')
    ax.set_ylabel('${\\rm \\kappa_{lat} (Wm^{-1}K^{-1})}$')
    
    for ii, each in enumerate(all_data):
        
        comp = Composition(all_data[each]['composition']).as_dict()
        lab_comp = ""
        for cc in comp:
            if int(comp[cc]) == 1:
                lab_comp += cc
            else:
                lab_comp += "%s_%d" % (cc, int(comp[cc]))
        label = "${\\rm %s}$ (%s)" % (lab_comp, all_data[each]['mpid'])
        
        xdat = all_data[each]['kl'][:,0]
        kps = np.zeros(len(xdat))
        kcs = np.zeros(len(xdat))
        for j in range(3):
            kps += all_data[each]['kl'][:,1+4*j] / 3.
            kcs += all_data[each]['kl_coherent'][:,1+j] / 3.
        
        ##
        if kappa_type == 'both':
            ks = kps + kcs
        elif kappa_type == 'kp':
            ks = kps.copy()
        elif kappa_type == 'kc':
            ks = kcs.copy()
        else:
            print("Error")

        ##
        icol = ii % 10
        ax.plot(xdat, ks, linestyle='-', c=cmap(icol),
                lw=lw, marker='o', markersize=ms,
                mfc='none', mew=lw, label=label)
    
    ncol = max(int(len(all_data) / 15 + 0.5), 1)
    ##
    set_axis(ax, xformat='log', yformat='log')
    set_legend(ax, fs=4, loc='upper left', loc2=[1.0,1.0], ncol=ncol)
    
    print(len(all_data), 'data')

    fig.savefig(figname, dpi=dpi, bbox_inches='tight')
    print(" Output", figname)
    return fig


def main(options):

    copy_all_files()
    all_data = get_all_data()
    
    for kappa in ['both', 'kp']:
        figname = 'fig_%s.png' % kappa
        plot_all_data(all_data, kappa_type=kappa, figname=figname)

    
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-f", "--filename", dest="filename", type="string",
        help="input file name")
    (options, args) = parser.parse_args()
    #file_check(options.filename)
    main(options)

