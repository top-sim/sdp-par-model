#!/usr/bin/env python3
#
# Generate a sequence of observations for each telescope from the
# corresponding list of projects (HPSOs).

import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

iform = 'cb_max_{t}.npz'
oform = 'cb_hist_{t}.png'

tlist = [('low', 'SKA1-Low'),
         ('mid', 'SKA1-Mid')]

for tele, tnam in tlist:

    ifile = iform.format(t=tele)
    ofile = oform.format(t=tele)

    with np.load(ifile) as data:
        batch_size = data['batch_size']
        cb_max = data['cb_max']

    cb_max /= 1000

    nbs = len(batch_size)
    nbin = 101
    cb_size = np.linspace(0.0, 100.0, nbin)
    cb_frac = np.zeros(nbin)

    plt.figure()
    plt.title(tnam)
    for ibs in range(nbs):
        for ibin in range(nbin):
            s = cb_size[ibin]
            cb_frac[ibin] = np.mean(cb_max[:, ibs] < s)
        plt.plot(cb_size, cb_frac, label=batch_size[ibs])
    plt.legend()
    plt.xlabel('Cold buffer size / PB')
    plt.ylabel('Fraction of sequences accommodated')
    plt.savefig(ofile)
    plt.close()
