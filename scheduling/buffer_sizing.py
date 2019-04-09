#!/usr/bin/env python3
#
# Generate a random observation sequence from the list of projects
# (HPSOs) and calculate how the buffer usage changes over time with
# different batch computing capacities.

import argparse
import numpy as np
import matplotlib.pyplot as plt
import scheduling as sched

parser = argparse.ArgumentParser()
parser.add_argument('telescope', type=str, help='name of telescope (low/mid)')
parser.add_argument('batch_factor', type=float, nargs='+', help='multiplier for batch capacity')
args = parser.parse_args()

iform = 'hpsos_{t}.csv'
nbin = 101
buff_max = 100.0

if args.telescope == 'low':
    tname = 'SKA1-Low'
    tsched_hours = 5.0
elif args.telescope == 'mid':
    tname = 'SKA1-Mid'
    tsched_hours = 8.0
else:
    raise ValueError('Unknown telescope {}'.format(args.telescope))

# Set the length of a scheduling block and the length of the sequence
# to generate.

tsched = tsched_hours * 3600.0
tseq = 100.0 * 24.0 * 3600.0
allow_short_tobs = False

ifile = iform.format(t=args.telescope)

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

buff_size = np.linspace(0.0, buff_max, nbin)
buff_frac = np.zeros((len(args.batch_factor), nbin))

for i, bf in enumerate(args.batch_factor):

    b_rflop = bf * b_rflop_avg
    print('batch capacity set to:', b_rflop, 'PFLOPS')

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

    # Get fraction of time buffer usage is below each size value.

    buff_frac[i, :] = sched.buffer_cumfrac(tb, buff_size)

    # Plot.

    fig, ax = plt.subplots(3, sharex=True)
    fig.suptitle('{} buffer usage (batch_factor = {})'.format(tname, bf))
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

plt.figure()
plt.title('{} buffer usage'.format(tname))
for i, bf in enumerate(args.batch_factor):
    plt.plot(buff_size, buff_frac[i, :], label=bf)
plt.legend(title='batch_factor', loc='lower right')
plt.ylim(-0.05, 1.05)
plt.xlabel('Total buffer size / PB')
plt.ylabel('Fraction of sequence accommodated')

plt.show()
