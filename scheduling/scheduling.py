import csv
import numpy as np

# Data type for parametric model output (extracted from CSV file).
#
# - name: name
# - tele: telescope
# - pipe: pipeline
# - tobs: observation time [s]
# - tpoint: total observation time for a single pointing [s]
# - texp: total observation time for all pointings [s]
# - rflop: compute rate [PFLOP/s]
# - mvis: input visibility size [TB]
# - mout: output data size [TB]

dtype_pmout = np.dtype([('name',   'U30'),
                        ('tele',   'U10'),
                        ('pipe',   'U10'),
                        ('tobs',   'f8' ),
                        ('tpoint', 'f8' ),
                        ('texp',   'f8' ),
                        ('rflop',  'f8' ),
                        ('mvis',   'f8' ),
                        ('mout',   'f8' )])

# Data type for project list.
#
# - name: name of project ('HPSO-x')
# - tpoint: total observation time for a single pointing [s]
# - texp: total observation time for all pointings [s]
# - r_rflop: real-time compute rate [PFLOP/s]
# - b_rflop: batch compute rate [PFLOP/s]
# - rinp: input data rate [TB/s] (for visibilities)
# - rout: output data rate [TB/s] (for data produced continuously,
#         like averaged visibilities)
# - mout: output data size per pointing [TB] (for fixed-size data
#         objects, like images)

dtype_proj = np.dtype([('name',    'U12'),
                       ('tpoint',  'f8' ),
                       ('texp',    'f8' ),
                       ('r_rflop', 'f8' ),
                       ('b_rflop', 'f8' ),
                       ('rinp',    'f8' ),
                       ('rout',    'f8' ),
                       ('mout',    'f8' )])

# Data type for observation sequence.
#
# - uid: unique ID of scheduling block
# - name: name of project ('HPSO-x')
# - tobs: observation time [s]
# - r_rflop: real-time compute rate [PFLOP/s]
# - b_rflop: batch compute rate [PFLOP/s]
# - minp: size of input data [TB]
# - mout: size of output data [TB]

dtype_seq = np.dtype([('uid',     'i8' ),
                      ('name',    'U12'),
                      ('tobs',    'f8' ),
                      ('r_rflop', 'f8' ),
                      ('b_rflop', 'f8' ),
                      ('minp',    'f8' ),
                      ('mout',    'f8' )])

# Data type for scheduled sequence.
#
# - uid: unique ID of scheduling block
# - name: name of project ('HPSO-x')
# - r_beg: beginning of real-time processing [s]
# - r_end: end of real-time processing [s]
# - b_beg: beginning of batch processing [s]
# - b_end: end of batch processing [s]

dtype_sched = np.dtype([('uid',   'i8' ),
                        ('name',  'U12'),
                        ('r_beg', 'f8' ),
                        ('r_end', 'f8' ),
                        ('b_beg', 'f8' ),
                        ('b_end', 'f8' )])

# Data type for buffer sizes and deltas.
#
# - t: time [s]
# - s: size [TB]

dtype_buff = np.dtype([('t', 'f8'),
                       ('s', 'f8')])

def write_projects(ofile, projects):
    '''Write csv list of projects to file'''
    header = 'name,Tpoint,Texp,r_Rflop,b_Rflop,Rinp,Rout,Mout'
    fmt = '%s' + 7 * ',%e'
    np.savetxt(ofile, projects, header=header, fmt=fmt)

def read_projects(ifile):
    '''Read list of projects from file'''
    projects = np.loadtxt(ifile, dtype=dtype_proj, delimiter=',').view(np.recarray)
    return projects

def write_sequence(ofile, sequence):
    '''Write csv sequence of observations to file'''
    header = 'uid,name,Tobs,r_Rflop,b_Rflop,Minp,Mout'
    fmt = '%d,%s' + 5 * ',%f'
    np.savetxt(ofile, sequence, header=header, fmt=fmt)

def extract_projects(ifile, tele):
    '''Extract list of projects from parametric model output'''

    # Read data from parametric model output CSV file.

    with open(ifile, 'r') as f:
        r = csv.reader(f)
        l = next(r)
        pmout = np.recarray(len(l)-1, dtype=dtype_pmout)
        pmout.name = [x.split()[0] for x in l[1:]]
        for l in r:
            if len(l) == 0:
                pass  # empty line; skip
            elif l[0] == 'Telescope':
                pmout.tele = l[1:]
            elif l[0] == 'Pipeline':
                pmout.pipe = l[1:]
            elif l[0] == 'Observation Time [s]':
                pmout.tobs = l[1:]
            elif l[0] == 'Pointing Time [s]':
                pmout.tpoint = l[1:]
            elif l[0] == 'Total Time [s]':
                pmout.texp = l[1:]
            elif l[0] == 'Total Compute Requirement [PetaFLOP/s]':
                pmout.rflop = l[1:]
            elif l[0] == 'Visibility Buffer [PetaBytes]':
                pmout.mvis = l[1:]
            elif l[0] == 'Output size [TB]':
                pmout.mout = l[1:]

    # Convert visibility data size to TB.

    pmout.mvis *= 1000.0

    # Get list of HPSOs for this telescope by finding the Ingest
    # pipelines. Extract the properties which do not change between
    # pipelines (times and visibility data size).

    i = (pmout.tele == tele) & (pmout.pipe == 'Ingest')
    ninp = [x.split()[0] for x in pmout.name[i]]
    name = [x.replace('hpso', 'HPSO-') for x in ninp]
    tobs = pmout[i].tobs
    tpoint = pmout[i].tpoint
    texp = pmout[i].texp
    mvis = pmout[i].mvis
    npoint = np.ceil(tobs/tpoint).astype(int)

    proj = np.recarray(len(name), dtype=dtype_proj)

    proj.name = name
    proj.tpoint = tpoint
    proj.texp = texp
    proj.rinp = mvis / tobs

    # For each HPSO, get sum of compute rate values for all real-time
    # and all batch pipelines, the output data sizes for DPrep{A,B,C},
    # and the output visibility rate for DPrepD.

    pipe_r = {'Ingest', 'RCAL', 'FastImg'}
    pipe_b = {'ICAL', 'DPrepA', 'DPrepA_Image', 'DPrepB', 'DPrepC', 'DPrepD'}

    for k, n in enumerate(ninp):
        i = (pmout.tele == tele) & (np.char.find(pmout.name, n) == 0)
        r_rflop = 0.0
        b_rflop = 0.0
        rout = 0.0
        mout = 0.0
        for j in pmout[i]:
            if j.pipe in pipe_r:
                r_rflop += j.rflop
            elif j.pipe in pipe_b:
                b_rflop += j.rflop
                if j.pipe == 'DPrepD':
                    rout += j.mout / tobs[k]
                else:
                    mout += j.mout / npoint[k]
            else:
                print('Unknown pipeline: {}'.format(j.pipe))
        proj[k].r_rflop = r_rflop
        proj[k].b_rflop = b_rflop
        if b_rflop == 0.0:
            # This is a non-imaging pipeline, so the visibilities do
            # not need to be stored.
            proj[k].rinp = 0.0
        proj[k].rout = rout
        proj[k].mout = mout

    return proj

def generate_sequence(projects, tsched, tseq, allow_short_tobs=False):
    '''Generate sequence of observations from list of projects'''

    if allow_short_tobs:

        # This case allows short observations for suitable projects:
        # if a project has tpoint less than tsched, then its
        # observations will be of length tpoint, otherwise they will
        # be of length tsched.

        # Probability for each project.

        texp = projects.texp
        tpoint = projects.tpoint
        tobs = np.where(tpoint < tsched, tpoint, tsched)
        prob = texp / tobs
        prob /= np.sum(prob)

        # Calculate number of scheduling blocks to draw. This number
        # is twice the number needed to get a long enough sequence,
        # given the average observation length. This should give a
        # long enough sequence in most draws.

        tobs_mean = np.sum(prob * tobs)
        nsched = np.ceil(2.0 * tseq / tobs_mean).astype(int)

        # Draw sequence until it is long enough, then truncate it to
        # the right length.

        ttot = 0.0
        while ttot < tseq:
            p = np.random.choice(projects, size=nsched, p=prob).view(np.recarray)
            tobs = np.where(p.tpoint < tsched, p.tpoint, tsched)
            ttot = np.sum(tobs)

        tcum = np.cumsum(tobs)
        nsched = np.where(tcum >= tseq)[0][0] + 1
        p = p[0:nsched]
        tobs = tobs[0:nsched]

        # Calculate quantities for each entry.

        seq = np.recarray(nsched, dtype=dtype_seq)
        seq.uid = np.arange(nsched)
        seq.name = p.name
        seq.tobs = tobs
        seq.r_rflop = p.r_rflop
        seq.b_rflop = p.b_rflop
        seq.minp = p.rinp * tobs
        seq.mout = p.rout * tobs + p.mout

    else:

        # This case generates scheduling blocks which are all of
        # length tsched.

        # Probability for each project.

        texp = projects.texp
        prob = texp / np.sum(texp)

        # Calculate number of scheduling blocks.

        nsched = np.ceil(tseq/tsched).astype(int)

        # Draw sequence and calculate quantities for each entry.

        p = np.random.choice(projects, size=nsched, p=prob).view(np.recarray)
        seq = np.recarray(nsched, dtype=dtype_seq)
        seq.uid = np.arange(nsched)
        seq.name = p.name
        seq.tobs = tsched
        seq.r_rflop = p.r_rflop
        seq.b_rflop = p.b_rflop
        seq.minp = p.rinp * tsched
        seq.mout = p.rout * tsched + p.mout * tsched/p.tpoint

    return seq

def schedule_simple(r_rflop_max, b_rflop_max, seq):
    '''Do scheduling in simplest possible fashion.'''

    if np.any(seq.r_rflop > r_rflop_max):
        print('Error: maximum flops for real-time processing exceeded')

    n = len(seq)

    sched = np.recarray(n, dtype=dtype_sched)

    sched.uid = seq.uid
    sched.name = seq.name

    cb_tmp = []
    hb_tmp = []

    r_end = 0.0
    b_end = 0.0

    for i in range(n):

        tobs = seq[i].tobs
        r_rflop = seq[i].r_rflop
        b_rflop = seq[i].b_rflop
        minp = seq[i].minp

        # Real-time processing.

        r_beg = r_end
        r_end = r_beg + tobs

        sched[i].r_beg = r_beg
        sched[i].r_end = r_end

        if b_rflop == 0.0:

            # This is a non-imaging pipeline. Visibilities are not
            # stored (so no change to buffers) and there is no batch
            # processing to be done.

            sched[i].b_beg = np.nan
            sched[i].b_end = np.nan

        else:

            # This is an imaging pipeline.

            b_beg = max(r_end, b_end)
            b_end = b_beg + tobs * b_rflop / b_rflop_max

            sched[i].b_beg = b_beg
            sched[i].b_end = b_end

            cb_tmp.append((r_beg, minp))
            cb_tmp.append((b_beg, -minp))
            hb_tmp.append((b_beg, minp))
            hb_tmp.append((b_end, -minp))

    cb_deltas = np.array(cb_tmp, dtype=dtype_buff).view(np.recarray)
    hb_deltas = np.array(hb_tmp, dtype=dtype_buff).view(np.recarray)

    cb_size = sum_deltas(cb_deltas)
    hb_size = sum_deltas(hb_deltas)

    return sched, cb_size, hb_size

def sum_deltas(deltas):
    '''Sum deltas to get size as a function of time.'''

    # First, combine deltas that happen at the same time.

    u = np.unique(deltas.t)
    tmp = np.recarray(len(u), dtype=dtype_buff)

    for i, t in enumerate(u):
        j = deltas.t == t
        tmp[i].t = t
        tmp[i].s = np.sum(deltas[j].s)

    # Second, keep only the deltas that are not zero after combining.

    tmp = tmp[tmp.s != 0.0].copy()

    # Last, calculate the cumulative sum.

    tmp.s = np.cumsum(tmp.s)

    return tmp

def steps_for_plot(buff, tmin=None, tmax=None):
    '''Turn size as a function of time into lines for plotting.'''

    t = []
    s = []

    sold = 0.0
    if tmin is not None and tmin < buff[0].t:
        t.append(tmin)
        s.append(sold)
    for ti, si in buff:
        t.append(ti)
        s.append(sold)
        t.append(ti)
        s.append(si)
        sold = si
    if tmax is not None and tmax > buff[-1].t:
        t.append(tmax)
        s.append(sold)

    return t, s
