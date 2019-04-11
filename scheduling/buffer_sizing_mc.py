#!/usr/bin/env python3
#
# Generate MC realisations of obervation sequences from the list of
# projects (HPSOs), schedule them with various batch computing
# capacities, and store statistics of the total buffer size required.

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

nbin = 101
buff_size = np.linspace(0.0, 100.0, nbin)

print('batch factors:', batch_factor)

for tel, tsched_hours in tlist:

    # Set the length of a scheduling block and the length of the sequence
    # to generate.

    tsched = tsched_hours * 3600.0

    ifile = iform.format(t=tel)
    ofile = oform.format(t=tel)

    # Read list of projects.

    proj = sched.read_projects(ifile)

    print(tel)
    print('projects:')
    print(proj)

    # Compute flop values.

    r_rflop_max = np.max(proj.r_rflop)
    b_rflop_avg = np.sum(proj.b_rflop * proj.texp) / np.sum(proj.texp)

    print('real-time max:', r_rflop_max, 'PFLOPS')
    print('batch average:', b_rflop_avg, 'PFLOPS')

    # Loop over MC realisations.

    buff_max = np.zeros((nreal, nbf))
    buff_frac = np.zeros((nbf, nbin))

    for ireal in range(nreal):

        print('Realisation', ireal)

        # Generate sequence of observations.

        seq = sched.generate_sequence(proj, tsched, tseq,
                                      allow_short_tobs=allow_short_tobs)

        # Loop over batch_factor values.

        for ibf in range(nbf):

            b_rflop = batch_factor[ibf] * b_rflop_avg

            # Generate schedule.

            sc, cb, hb, tb = sched.schedule_simple(r_rflop_max, b_rflop, seq)

            # Convert buffer sizes to PB.

            cb.s /= 1000.0
            hb.s /= 1000.0
            tb.s /= 1000.0

            # Store maximum buffer usage and accumulate fractions of
            # time below buff_size values.

            buff_max[ireal, ibf] = sched.buffer_max(tb)
            buff_frac[ibf, :] += sched.buffer_cumfrac(tb, buff_size)

    # Normalise fractions of time.

    buff_frac /= nreal

    np.savez(ofile, batch_factor=batch_factor, buff_max=buff_max,
             buff_size=buff_size, buff_frac=buff_frac)
