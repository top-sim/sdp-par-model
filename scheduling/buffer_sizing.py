#!/usr/bin/env python3
#
# Generate a random sequence of observations from the corresponding
# list of projects (HPSOs) and calculate how the buffer usage changes
# over time.

import sys
import numpy as np
import matplotlib.pyplot as plt
import scheduling as sched

if len(sys.argv) < 3:
    print('Usage: {} telescope batch_factor'.format(sys.argv[0]))
    sys.exit()

tele = sys.argv[1]
batch_factor = map(float, sys.argv[2:])

iform = 'hpsos_{t}.csv'

if tele == 'low':
    tnam = 'SKA1-Low'
    tsched_hours = 5.0
elif tele == 'mid':
    tnam = 'SKA1-Mid'
    tsched_hours = 8.0

# Set the length of a scheduling block and the length of the sequence
# to generate.

tsched = tsched_hours * 3600.0
tseq = 100.0 * 24.0 * 3600.0
allow_short_tobs = False

ifile = iform.format(t=tele)

# Read list of projects.

proj = sched.read_projects(ifile)

print('projects:')
print(proj)

# Compute flop values.

r_rflop_max = np.max(proj.r_rflop)
b_rflop_avg = np.sum(proj.b_rflop * proj.texp) / np.sum(proj.texp)

print('real-time max:', r_rflop_max, 'PFLOPS')
print('batch average:', b_rflop_avg, 'PFLOPS')

# Generate sequence of observations.

seq = sched.generate_sequence(proj, tsched, tseq,
                              allow_short_tobs=allow_short_tobs)

# Loop over batch_factor values.

for bf in batch_factor:

    b_rflop = bf * b_rflop_avg
    print('batch compute set to:', b_rflop, 'PFLOPS')

    # Generate schedule.

    sc, cb, hb, tb = sched.schedule_simple(r_rflop_max, b_rflop, seq)

    # Convert time to hours and sizes to PB for plotting.

    sc.r_beg /= 3600
    sc.r_end /= 3600
    sc.b_beg /= 3600
    sc.b_end /= 3600

    r_beg = 0.0
    r_end = sc[-1].r_end
    b_end = np.nanmax(sc.b_end)

    tmin = -1.0
    tmax = max(r_end, b_end) + 1.0

    cb.t /= 3600
    cb.s /= 1000

    hb.t /= 3600
    hb.s /= 1000

    tb.t /= 3600
    tb.s /= 1000

    # Plot.

    fig, ax = plt.subplots(3, sharex=True)
    fig.suptitle('{} buffer usage (batch_factor = {})'.format(tnam, bf))
    ax[0].set_ylabel('Cold buffer / PB')
    ax[0].axvline(x=r_beg, color='grey', linestyle=':', linewidth=1.0)
    ax[0].axvline(x=r_end, color='grey', linestyle=':', linewidth=1.0)
    t, s = sched.steps_for_plot(cb, tmin=tmin, tmax=tmax)
    ax[0].plot(t, s)
    ax[1].set_ylabel('Hot buffer / PB')
    ax[1].axvline(x=r_beg, color='grey', linestyle=':', linewidth=1.0)
    ax[1].axvline(x=r_end, color='grey', linestyle=':', linewidth=1.0)
    t, s = sched.steps_for_plot(hb, tmin=tmin, tmax=tmax)
    ax[1].plot(t, s)
    ax[2].set_xlabel('Time / hours')
    ax[2].set_ylabel('Total buffer / PB')
    ax[2].axvline(x=r_beg, color='grey', linestyle=':', linewidth=1.0)
    ax[2].axvline(x=r_end, color='grey', linestyle=':', linewidth=1.0)
    t, s = sched.steps_for_plot(tb, tmin=tmin, tmax=tmax)
    ax[2].plot(t, s)

plt.show()
