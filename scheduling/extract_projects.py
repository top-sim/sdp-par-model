#!/usr/bin/env python3

import scheduling as sched

ifile = 'hpsos.csv'
oform = 'hpsos_{t}.txt'

tele = [('low', 'SKA1_Low'),
        ('mid', 'SKA1_Mid')]

# Loop over telescopes.

for t, tinp in tele:

    ofile = oform.format(t=t)

    # Extract project information.

    proj = sched.extract_projects(ifile, tinp)

    # Write list of projects.

    sched.write_projects(ofile, proj)

