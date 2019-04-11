#!/usr/bin/env python3
#
# Plot fraction of observation sequences that can be accommodated
# against buffer size.

import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

iform = 'buffer_mc_{t}.npz'
oform = 'buffer_mc_{t}.png'

tlist = [('low', 'SKA1-Low'),
         ('mid', 'SKA1-Mid')]

for tel, tname in tlist:

    ifile = iform.format(t=tel)
    ofile = oform.format(t=tel)

    with np.load(ifile) as data:
        batch_factor = data['batch_factor']
        buff_max = data['buff_max']
        buff_size = data['buff_size']
        buff_frac = data['buff_frac']

    nbf = len(batch_factor)
    nbin = len(buff_size)

    plt.figure()
    plt.title('{} buffer usage'.format(tname))

    # Pessimistic version (computes fractions based on maximum buffer
    # size in each realisation).

    #buff_frac_max = np.zeros(nbin)
    #for ibf in range(nbf):
    #    for ibin in range(nbin):
    #        s = buff_size[ibin]
    #        buff_frac_max[ibin] = np.mean(buff_max[:, ibf] < s)
    #    plt.plot(buff_size, buff_frac_max, label=batch_factor[ibf])

    # More realistic version (used pre-computed values based on
    # fraction of time).

    for ibf in range(nbf):
        plt.plot(buff_size, buff_frac[ibf, :], label=batch_factor[ibf])

    plt.legend(title='batch_factor', loc='lower right')
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Total buffer size / PB')
    plt.ylabel('Fraction of sequences accommodated')
    plt.savefig(ofile)
    plt.close()
