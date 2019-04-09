#!/usr/bin/env python3
#
# Generate a sequence of observations for each telescope from the
# corresponding list of projects (HPSOs).

import numpy as np
import scheduling as sched

iform = 'hpsos_{t}.csv'
oform = 'buffer_mc_{t}.npz'

tlist = [('low', 5.0),
         ('mid', 8.0)]

tseq = 100.0 * 24.0 * 3600.0
allow_short_tobs = False
nreal = 1000

batch_factor = np.array([1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40,
                       1.50, 2.00])
nbf = len(batch_factor)

print('batch factors:', batch_factor)

for tele, tsched_hours in tlist:

    # Set the length of a scheduling block and the length of the sequence
    # to generate.

    tsched = tsched_hours * 3600.0

    ifile = iform.format(t=tele)
    ofile = oform.format(t=tele)

    # Read list of projects.

    proj = sched.read_projects(ifile)

    print(tele)
    print('projects:')
    print(proj)

    # Compute flop values.

    r_rflop_max = np.max(proj.r_rflop)
    b_rflop_avg = np.sum(proj.b_rflop * proj.texp) / np.sum(proj.texp)

    print('real-time max:', r_rflop_max, 'PFLOPS')
    print('batch average:', b_rflop_avg, 'PFLOPS')

    # Generate sequence of observations.

    cb_max = np.zeros((nreal, nbf))
    hb_max = np.zeros((nreal, nbf))
    tb_max = np.zeros((nreal, nbf))

    for ireal in range(nreal):

        print('Realisation', ireal)

        seq = sched.generate_sequence(proj, tsched, tseq,
                                      allow_short_tobs=allow_short_tobs)

        for ibf in range(nbf):

            b_rflop = batch_factor[ibf] * b_rflop_avg
            sc, cb, hb, tb = sched.schedule_simple(r_rflop_max, b_rflop, seq)
            cb_max[ireal, ibf] = np.max(cb.s)
            hb_max[ireal, ibf] = np.max(hb.s)
            tb_max[ireal, ibf] = np.max(tb.s)

    np.savez(ofile, batch_factor=batch_factor, cb_max=cb_max,
             hb_max=hb_max, tb_max=tb_max)
