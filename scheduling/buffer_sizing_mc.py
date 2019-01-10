#!/usr/bin/env python3
#
# Generate a sequence of observations for each telescope from the
# corresponding list of projects (HPSOs).

import numpy as np
import scheduling as sched

iform = 'hpsos_{t}.csv'
oform = 'cb_max_{t}.npz'

tlist = [('low', 5.0),
         ('mid', 8.0)]

tseq = 100.0 * 24.0 * 3600.0
allow_short_tobs = False
nreal = 1000

batch_size = np.array([0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30,
                       1.40, 1.50, 2.00])
nbs = len(batch_size)

print('batch sizes:', batch_size)

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

    cb_max = np.zeros((nreal, nbs))

    for ireal in range(nreal):

        print('Realisation', ireal)

        seq = sched.generate_sequence(proj, tsched, tseq,
                                      allow_short_tobs=allow_short_tobs)

        for ibs in range(nbs):

            b_rflop_max = batch_size[ibs] * b_rflop_avg
            sc, cb, hb = sched.schedule_simple(r_rflop_max, b_rflop_max, seq)
            cb_max[ireal, ibs] = np.max(cb.s)

    np.savez(ofile, batch_size=batch_size, cb_max=cb_max)
