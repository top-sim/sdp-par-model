#!/usr/bin/env python3
#
# Generate a sequence of observations for each telescope from the
# corresponding list of projects (HPSOs).

import numpy as np
import scheduling as sched

iform = 'hpsos_{t}.txt'
oform = 'sequence_{t}.txt'

tele = ['low', 'mid']

# Set the length of a scheduling block and the length of the sequence
# to generate.

tsched = 6.0 * 3600.0
tseq = 10.0 * 24.0 * 3600.0
allow_short_tobs = False

# Loop over telescopes.

for t in tele:

    ifile = iform.format(t=t)
    ofile = oform.format(t=t)

    # Read list of projects.

    proj = sched.read_projects(ifile)

    # Generate sequence of observations.

    seq = sched.generate_sequence(proj, tsched, tseq,
                                  allow_short_tobs=allow_short_tobs)

    # Write sequence of observations.

    sched.write_sequence(ofile, seq)
