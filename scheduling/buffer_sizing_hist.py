#!/usr/bin/env python3
#
# Generate a sequence of observations for each telescope from the
# corresponding list of projects (HPSOs).

import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

iform = 'buffer_mc_{t}.npz'
oform = 'buffer_mc_{t}_{b}.png'

tlist = [('low', 'SKA1-Low'),
         ('mid', 'SKA1-Mid')]

nbin = 101
buff_size = np.linspace(0.0, 100.0, nbin)

for tele, tname in tlist:

    ifile = iform.format(t=tele)

    with np.load(ifile) as data:
        batch_factor = data['batch_factor']
        cb_max = data['cb_max']
        hb_max = data['hb_max']
        tb_max = data['tb_max']

    cb_max /= 1000
    hb_max /= 1000
    tb_max /= 1000

    nbf = len(batch_factor)
    blist = [('cb', cb_max, 'Cold buffer' ),
             ('tb', tb_max, 'Total buffer')]

    for buff, bmax, btitle in blist:

        ofile = oform.format(t=tele, b=buff)

        buff_frac = np.zeros(nbin)

        plt.figure()
        plt.title(tname)
        for ibf in range(nbf):
            for ibin in range(nbin):
                s = buff_size[ibin]
                buff_frac[ibin] = np.mean(bmax[:, ibf] < s)
            plt.plot(buff_size, buff_frac, label=batch_factor[ibf])
        plt.legend(title='batch_factor', loc='upper left')
        plt.xlabel('{} size / PB'.format(btitle))
        plt.ylabel('Fraction of sequences accommodated')
        plt.savefig(ofile)
        plt.close()
